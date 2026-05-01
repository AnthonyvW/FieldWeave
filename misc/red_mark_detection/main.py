"""
Standalone red mark detection overlay.

Processes a folder of images, detects red registration marks, and saves
annotated copies to an output directory.

Usage:
    python red_detection_overlay.py <input_folder> [output_folder]
    python red_detection_overlay.py <input_folder> --profile
    python red_detection_overlay.py <input_folder> --debug

If output_folder is omitted, results are saved to <input_folder>/output/.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

COLOR_VALID = (0, 255, 0)
COLOR_FILTERED = (0, 0, 255)
COLOR_LINE = (0, 255, 255)
COLOR_LINE_DIM = (0, 180, 180)
COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_BG = (0, 0, 0)


class DetectionResult:
    def __init__(
        self,
        mean_x: float | None,
        mean_y: float | None,
        valid_centroids: np.ndarray,    # shape (N, 2) float, col 0 = x, col 1 = y
        filtered_centroids: np.ndarray, # shape (M, 2) float
        valid_mask: np.ndarray,
        filtered_mask: np.ndarray,
        red_isolated: np.ndarray,
        opened: np.ndarray,
        cleaned: np.ndarray,
        threshold_value: float,
    ) -> None:
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.valid_centroids = valid_centroids
        self.filtered_centroids = filtered_centroids
        self.valid_mask = valid_mask
        self.filtered_mask = filtered_mask
        self.red_isolated = red_isolated
        self.opened = opened
        self.cleaned = cleaned
        self.threshold_value = threshold_value


def detect_red_marks(
    img_hsv: np.ndarray,
    open_kernel_size: int = 5,
    min_blob_area: int = 500,
    max_aspect_ratio: float = 8.0,
    min_area_fraction: float = 0.1,
    hue_low: int = 160,
    hue_high: int = 10,
    sat_min: int = 100,
    val_min: int = 50,
) -> DetectionResult:
    # Red wraps around 0/180 in OpenCV HSV so we OR two ranges.
    lower1 = np.array([hue_low,  sat_min, val_min], dtype=np.uint8)
    upper1 = np.array([180,      255,     255],     dtype=np.uint8)
    lower2 = np.array([0,        sat_min, val_min], dtype=np.uint8)
    upper2 = np.array([hue_high, 255,     255],     dtype=np.uint8)

    red_isolated = cv2.bitwise_or(
        cv2.inRange(img_hsv, lower1, upper1),
        cv2.inRange(img_hsv, lower2, upper2),
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size)
    )
    opened = cv2.morphologyEx(red_isolated, cv2.MORPH_OPEN, open_kernel)

    threshold_value = 127.0

    # centroids shape: (num_labels, 2) — col 0 = cx, col 1 = cy
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        opened, connectivity=8
    )

    empty_mask = np.zeros_like(opened)

    if num_labels <= 1:
        empty = np.empty((0, 2), dtype=float)
        return DetectionResult(
            mean_x=None, mean_y=None,
            valid_centroids=empty, filtered_centroids=empty,
            valid_mask=empty_mask, filtered_mask=empty_mask,
            red_isolated=red_isolated, opened=opened, cleaned=empty_mask,
            threshold_value=threshold_value,
        )

    # Vectorised area and aspect-ratio filter over all labels at once.
    areas = stats[1:, cv2.CC_STAT_AREA]
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    aspect_ratios = np.maximum(widths, heights) / np.maximum(np.minimum(widths, heights), 1)

    # Primary filters: minimum absolute area and aspect ratio.
    keep = (areas >= min_blob_area) & (aspect_ratios <= max_aspect_ratio)

    # Relative area filter: discard blobs smaller than min_area_fraction of the
    # largest surviving blob. This prevents small strays from being mistaken for
    # the mark when a large mark is also present.
    if keep.any():
        max_area = float(areas[keep].max())
        keep &= areas >= min_area_fraction * max_area

    kept_indices = np.where(keep)[0] + 1  # shift back to label space (label 0 = background)

    if kept_indices.size == 0:
        empty = np.empty((0, 2), dtype=float)
        return DetectionResult(
            mean_x=None, mean_y=None,
            valid_centroids=empty, filtered_centroids=empty,
            valid_mask=empty_mask, filtered_mask=empty_mask,
            red_isolated=red_isolated, opened=opened, cleaned=empty_mask,
            threshold_value=threshold_value,
        )

    kept_centroids = centroids[kept_indices]  # (N, 2), cx in col 0

    # IQR outlier rejection on centroid X coordinates.
    cx = kept_centroids[:, 0]
    if kept_indices.size > 3:
        q1, q3 = np.percentile(cx, 25), np.percentile(cx, 75)
        iqr = q3 - q1
        inlier_mask = (cx >= q1 - 1.5 * iqr) & (cx <= q3 + 1.5 * iqr)
    else:
        inlier_mask = np.ones(kept_indices.size, dtype=bool)

    valid_label_ids = kept_indices[inlier_mask]
    filtered_label_ids = kept_indices[~inlier_mask]
    valid_centroids = kept_centroids[inlier_mask]
    filtered_centroids = kept_centroids[~inlier_mask]

    # Build masks via a lookup table indexed by label id — O(pixels) with no
    # per-label scan, unlike np.isin which scans the full array per element.
    lut = np.zeros(num_labels, dtype=np.uint8)
    lut[valid_label_ids] = 255
    valid_mask = lut[labels]

    lut[:] = 0
    lut[filtered_label_ids] = 255
    filtered_mask = lut[labels]

    cleaned = valid_mask | filtered_mask

    # mean_x / mean_y = mean of per-blob centroids, weighting each blob equally.
    if valid_centroids.shape[0] > 0:
        mean_x: float | None = float(np.mean(valid_centroids[:, 0]))
        mean_y: float | None = float(np.mean(valid_centroids[:, 1]))
    else:
        mean_x = None
        mean_y = None

    return DetectionResult(
        mean_x=mean_x,
        mean_y=mean_y,
        valid_centroids=valid_centroids,
        filtered_centroids=filtered_centroids,
        valid_mask=valid_mask,
        filtered_mask=filtered_mask,
        red_isolated=red_isolated,
        opened=opened,
        cleaned=cleaned,
        threshold_value=threshold_value,
    )


def clustered_on_one_side(
    valid_centroids: np.ndarray,
    img_w: int,
    side_cluster_fraction: float = 0.85,
    side_cluster_margin: float = 0.18,
) -> bool:
    if valid_centroids.shape[0] == 0 or img_w <= 0:
        return False

    margin = float(np.clip(side_cluster_margin, 0.0, 0.45))
    left_edge = img_w * margin
    right_edge = img_w * (1.0 - margin)
    cx = valid_centroids[:, 0]

    left_frac = float(np.mean(cx <= left_edge))
    right_frac = float(np.mean(cx >= right_edge))
    return max(left_frac, right_frac) >= side_cluster_fraction


def draw_text_with_bg(
    img: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    font_scale: float = 0.5,
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = origin
    line_height = int(cv2.getTextSize("A", font, font_scale, thickness)[0][1] * 2.2)

    for line in lines:
        (tw, th), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        pad = 4
        cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), COLOR_TEXT_BG, -1)
        cv2.putText(img, line, (x, y), font, font_scale, COLOR_TEXT, thickness, cv2.LINE_AA)
        y += line_height


def draw_overlay(
    img_bgr: np.ndarray,
    result: DetectionResult,
    line_orientation: str,
    distance_from_center: float | None,
) -> np.ndarray:
    out = img_bgr.copy()
    img_h, img_w = out.shape[:2]

    if result.filtered_mask.any():
        out[result.filtered_mask > 0] = COLOR_FILTERED
    if result.valid_mask.any():
        out[result.valid_mask > 0] = COLOR_VALID

    if line_orientation == "horizontal":
        if result.mean_y is not None:
            cv2.line(out, (0, int(result.mean_y)), (img_w, int(result.mean_y)), COLOR_LINE, 2)
            cv2.line(out, (0, img_h // 2), (img_w, img_h // 2), COLOR_LINE_DIM, 1)
    else:
        if result.mean_x is not None:
            cv2.line(out, (int(result.mean_x), 0), (int(result.mean_x), img_h), COLOR_LINE, 2)
            cv2.line(out, (img_w // 2, 0), (img_w // 2, img_h), COLOR_LINE_DIM, 1)

    n_valid = result.valid_centroids.shape[0]
    n_filtered = result.filtered_centroids.shape[0]
    info_lines: list[str] = [f"Valid blobs: {n_valid}"]
    if n_filtered > 0:
        info_lines.append(f"Filtered blobs: {n_filtered}")
    info_lines.append(f"Line: {line_orientation}")

    if line_orientation == "horizontal" and result.mean_y is not None and distance_from_center is not None:
        center_y = img_h / 2.0
        percent = (distance_from_center / center_y) * 100 if center_y > 0 else 0.0
        direction = "down" if result.mean_y > center_y else "up"
        info_lines.append(f"Center Y: {result.mean_y:.1f} px")
        info_lines.append(f"Distance: {distance_from_center:.1f} px ({percent:.1f}% {direction})")
    elif line_orientation == "vertical" and result.mean_x is not None and distance_from_center is not None:
        center_x = img_w / 2.0
        percent = (distance_from_center / center_x) * 100 if center_x > 0 else 0.0
        direction = "right" if result.mean_x > center_x else "left"
        info_lines.append(f"Center X: {result.mean_x:.1f} px")
        info_lines.append(f"Distance: {distance_from_center:.1f} px ({percent:.1f}% {direction})")

    draw_text_with_bg(out, info_lines, (10, 20))
    return out


def detect_and_measure(
    img_bgr: np.ndarray,
    open_kernel_size: int = 5,
    min_blob_area: int = 500,
    max_aspect_ratio: float = 8.0,
    min_area_fraction: float = 0.1,
    side_cluster_fraction: float = 0.85,
    side_cluster_margin: float = 0.18,
    hue_low: int = 160,
    hue_high: int = 10,
    sat_min: int = 100,
    val_min: int = 50,
    orientation: str | None = None,
    scale: float = 1.0,
) -> tuple[DetectionResult, str, float | None]:
    """
    Pure machine-vision step — no drawing.
    Returns (result, line_orientation, distance_from_center).

    orientation: 'horizontal' or 'vertical' to override auto-detection,
                 or None to infer from the mark positions.
    scale: downsample factor applied before processing (e.g. 0.5 = half resolution).
           Centroids and mean_x/y are scaled back to full-resolution coordinates.
           min_blob_area is adjusted proportionally.
    """
    img_h, img_w = img_bgr.shape[:2]

    if scale != 1.0:
        proc = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        scaled_min_blob_area = max(1, int(min_blob_area * scale * scale))
    else:
        proc = img_bgr
        scaled_min_blob_area = min_blob_area

    img_hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    proc_h, proc_w = proc.shape[:2]

    result = detect_red_marks(
        img_hsv,
        open_kernel_size=open_kernel_size,
        min_blob_area=scaled_min_blob_area,
        max_aspect_ratio=max_aspect_ratio,
        min_area_fraction=min_area_fraction,
        hue_low=hue_low,
        hue_high=hue_high,
        sat_min=sat_min,
        val_min=val_min,
    )

    # Scale centroids and mean back to full-resolution coordinates.
    if scale != 1.0:
        if result.valid_centroids.shape[0] > 0:
            result.valid_centroids = result.valid_centroids / scale
        if result.filtered_centroids.shape[0] > 0:
            result.filtered_centroids = result.filtered_centroids / scale
        if result.mean_x is not None:
            result.mean_x = result.mean_x / scale
        if result.mean_y is not None:
            result.mean_y = result.mean_y / scale
        # Upscale masks to full resolution so draw_overlay can apply them to
        # the original image.
        result.valid_mask = cv2.resize(
            result.valid_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST
        )
        result.filtered_mask = cv2.resize(
            result.filtered_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST
        )

    if orientation is not None:
        line_orientation = orientation
    else:
        line_orientation = "horizontal" if clustered_on_one_side(
            result.valid_centroids, img_w, side_cluster_fraction, side_cluster_margin
        ) else "vertical"

    distance_from_center: float | None = None
    if line_orientation == "horizontal" and result.mean_y is not None:
        distance_from_center = abs(result.mean_y - img_h / 2.0)
    elif line_orientation == "vertical" and result.mean_x is not None:
        distance_from_center = abs(result.mean_x - img_w / 2.0)

    return result, line_orientation, distance_from_center


def process_image(
    img_bgr: np.ndarray,
    open_kernel_size: int = 5,
    min_blob_area: int = 500,
    max_aspect_ratio: float = 8.0,
    min_area_fraction: float = 0.1,
    side_cluster_fraction: float = 0.85,
    side_cluster_margin: float = 0.18,
    hue_low: int = 160,
    hue_high: int = 10,
    sat_min: int = 100,
    val_min: int = 50,
    orientation: str | None = None,
    scale: float = 1.0,
) -> tuple[np.ndarray, DetectionResult]:
    result, line_orientation, distance_from_center = detect_and_measure(
        img_bgr,
        open_kernel_size=open_kernel_size,
        min_blob_area=min_blob_area,
        max_aspect_ratio=max_aspect_ratio,
        min_area_fraction=min_area_fraction,
        side_cluster_fraction=side_cluster_fraction,
        side_cluster_margin=side_cluster_margin,
        hue_low=hue_low,
        hue_high=hue_high,
        sat_min=sat_min,
        val_min=val_min,
        orientation=orientation,
        scale=scale,
    )
    overlay = draw_overlay(img_bgr, result, line_orientation, distance_from_center)
    return overlay, result


def save_intermediates(result: DetectionResult, out_dir: Path, stem: str) -> None:
    cv2.imwrite(str(out_dir / f"{stem}_1_red_isolated.png"), result.red_isolated)
    cv2.imwrite(str(out_dir / f"{stem}_2_opened.png"), result.opened)
    cv2.imwrite(str(out_dir / f"{stem}_3_cleaned.png"), result.cleaned)

    stats_path = out_dir / f"{stem}_stats.txt"
    lines = [
        f"threshold_value: {result.threshold_value:.2f}",
        f"mean_x: {result.mean_x}",
        f"mean_y: {result.mean_y}",
        f"valid_blobs: {result.valid_centroids.shape[0]}",
        f"filtered_blobs: {result.filtered_centroids.shape[0]}",
        f"valid_centroids: {result.valid_centroids.tolist()}",
        f"filtered_centroids: {result.filtered_centroids.tolist()}",
    ]
    stats_path.write_text("\n".join(lines))


def _print_profile(profiler: cProfile.Profile, top_n: int) -> None:
    buf = io.StringIO()
    ps = pstats.Stats(profiler, stream=buf)
    ps.strip_dirs()
    ps.sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(top_n)
    print(buf.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description="Red mark detection overlay for a folder of images.")
    parser.add_argument("input_folder", type=Path, help="Folder containing input images.")
    parser.add_argument(
        "output_folder",
        type=Path,
        nargs="?",
        default=None,
        help="Folder to save annotated images (default: <input_folder>/output).",
    )
    parser.add_argument("--open-kernel-size", type=int, default=5,
                        help="Morphological opening kernel size; larger kills more noise (default: 5).")
    parser.add_argument("--min-blob-area", type=int, default=500,
                        help="Minimum pixel area for a blob to survive noise removal (default: 500).")
    parser.add_argument("--max-aspect-ratio", type=float, default=8.0,
                        help="Maximum bounding-box aspect ratio for a valid mark blob (default: 8.0).")
    parser.add_argument("--min-area-fraction", type=float, default=0.1,
                        help="Minimum blob area as a fraction of the largest blob; filters small strays (default: 0.1).")
    parser.add_argument("--side-cluster-fraction", type=float, default=0.85)
    parser.add_argument("--side-cluster-margin", type=float, default=0.18)
    parser.add_argument("--hue-low", type=int, default=160,
                        help="Lower hue boundary for red (160-180 range, default: 160).")
    parser.add_argument("--hue-high", type=int, default=10,
                        help="Upper hue boundary for red (0-hue_high range, default: 10).")
    parser.add_argument("--sat-min", type=int, default=100,
                        help="Minimum HSV saturation for a pixel to count as red (default: 100).")
    parser.add_argument("--val-min", type=int, default=50,
                        help="Minimum HSV value for a pixel to count as red (default: 50).")
    parser.add_argument("--orientation", choices=["horizontal", "vertical"], default=None,
                        help="Force line orientation instead of auto-detecting from mark positions.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Downsample factor before processing, e.g. 0.5 = half resolution (default: 1.0).")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate pipeline images to <output>/debug/.")
    parser.add_argument("--profile", action="store_true",
                        help="Profile execution and print a per-function timing summary.")
    parser.add_argument("--profile-lines", type=int, default=20, metavar="N",
                        help="Number of functions to show in the profile output (default: 20).")
    args = parser.parse_args()

    input_folder: Path = args.input_folder
    output_folder: Path = args.output_folder or input_folder / "output"

    if not input_folder.is_dir():
        print(f"Error: '{input_folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in sorted(input_folder.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]

    if not image_paths:
        print(f"No images found in '{input_folder}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(image_paths)} image(s) -> '{output_folder}'")

    profiler: cProfile.Profile | None = cProfile.Profile() if args.profile else None

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  Skipping (unreadable): {path.name}")
            continue

        if profiler is not None:
            profiler.enable()

        t0 = time.perf_counter()
        detection, line_orientation, distance_from_center = detect_and_measure(
            img,
            open_kernel_size=args.open_kernel_size,
            min_blob_area=args.min_blob_area,
            max_aspect_ratio=args.max_aspect_ratio,
            min_area_fraction=args.min_area_fraction,
            side_cluster_fraction=args.side_cluster_fraction,
            side_cluster_margin=args.side_cluster_margin,
            hue_low=args.hue_low,
            hue_high=args.hue_high,
            sat_min=args.sat_min,
            val_min=args.val_min,
            orientation=args.orientation,
            scale=args.scale,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if profiler is not None:
            profiler.disable()

        overlay = draw_overlay(img, detection, line_orientation, distance_from_center)
        cv2.imwrite(str(output_folder / path.name), overlay)

        if args.debug:
            debug_dir = output_folder / "debug"
            debug_dir.mkdir(exist_ok=True)
            save_intermediates(detection, debug_dir, path.stem)

        n_valid = detection.valid_centroids.shape[0]
        n_filtered = detection.filtered_centroids.shape[0]
        print(
            f"  {path.name}: {n_valid} valid blobs, {n_filtered} filtered  "
            f"(threshold={detection.threshold_value:.1f})  "
            f"{elapsed_ms:.1f}ms"
        )

    if profiler is not None:
        print(f"\n--- Profile: cumulative time, top {args.profile_lines} functions ---")
        _print_profile(profiler, args.profile_lines)

    print("Done.")


if __name__ == "__main__":
    main()