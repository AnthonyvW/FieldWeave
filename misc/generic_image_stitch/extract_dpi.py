from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


MM_PER_INCH = 25.4


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        print(f"Error: could not read image at {path}", file=sys.stderr)
        sys.exit(1)
    return img


def binarize(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def longest_run(row: np.ndarray) -> int:
    max_run = cur_run = 0
    for px in row:
        if px > 0:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


def find_bar_axis(binary: np.ndarray) -> tuple[str, int]:
    """
    Determine whether the scalebar runs horizontally or vertically.
    Returns ('horizontal', best_row) or ('vertical', best_col).
    """
    row_best = max(longest_run(binary[r]) for r in range(binary.shape[0]))
    col_best = max(longest_run(binary[:, c]) for c in range(binary.shape[1]))

    if row_best >= col_best:
        best = max(range(binary.shape[0]), key=lambda r: longest_run(binary[r]))
        return "horizontal", best
    else:
        best = max(range(binary.shape[1]), key=lambda c: longest_run(binary[:, c]))
        return "vertical", best


def find_tick_intersections(
    binary: np.ndarray,
    axis: str,
    baseline_index: int,
    baseline_band: int = 3,
    tick_min_length: int = 200,
) -> tuple[int, int]:
    """
    Find the positions of the two outermost tick marks that cross the baseline.

    Strategy: scan along the baseline. At each position, check whether there
    is a perpendicular run of foreground pixels (a tick) crossing through the
    baseline. Collect all such positions and return the outermost two.

    tick_min_length: minimum perpendicular run length to count as a tick
                     (filters out noise / antialiasing on the baseline itself).
    """
    h, w = binary.shape

    if axis == "horizontal":
        # baseline runs left-right; ticks run vertically
        along_size = w
        perp_size = h

        def perp_run_at(pos: int) -> int:
            # count foreground pixels in the column at x=pos
            col = binary[:, pos]
            return int(np.count_nonzero(col))

        def baseline_hit(pos: int) -> bool:
            # is the baseline row foreground at this position?
            y0 = max(0, baseline_index - baseline_band)
            y1 = min(h, baseline_index + baseline_band + 1)
            return bool(binary[y0:y1, pos].max() > 0)

    else:
        # baseline runs top-bottom; ticks run horizontally
        along_size = h
        perp_size = w

        def perp_run_at(pos: int) -> int:
            row = binary[pos, :]
            return int(np.count_nonzero(row))

        def baseline_hit(pos: int) -> bool:
            x0 = max(0, baseline_index - baseline_band)
            x1 = min(w, baseline_index + baseline_band + 1)
            return bool(binary[pos, x0:x1].max() > 0)

    tick_positions = []
    for pos in range(along_size):
        if baseline_hit(pos) and perp_run_at(pos) >= tick_min_length:
            tick_positions.append(pos)

    if len(tick_positions) < 2:
        # fallback: outermost foreground pixels along the baseline band
        if axis == "horizontal":
            y0 = max(0, baseline_index - baseline_band)
            y1 = min(h, baseline_index + baseline_band + 1)
            presence = binary[y0:y1].max(axis=0)
        else:
            x0 = max(0, baseline_index - baseline_band)
            x1 = min(w, baseline_index + baseline_band + 1)
            presence = binary[:, x0:x1].max(axis=1)
        nz = np.where(presence > 0)[0]
        return int(nz[0]), int(nz[-1])

    return tick_positions[0], tick_positions[-1]


def compute_dpi(pixel_length: int, physical_mm: float) -> float:
    return (pixel_length / physical_mm) * MM_PER_INCH


def save_debug(
    image_path: Path,
    original: np.ndarray,
    axis: str,
    index: int,
    start: int,
    end: int,
    dpi: float,
) -> None:
    vis = original.copy() if len(original.shape) == 3 else cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    if axis == "horizontal":
        pt1, pt2 = (start, index), (end, index)
    else:
        pt1, pt2 = (index, start), (index, end)

    cv2.line(vis, pt1, pt2, (0, 0, 255), 2)
    cv2.circle(vis, pt1, 6, (0, 255, 0), -1)
    cv2.circle(vis, pt2, 6, (0, 255, 0), -1)
    label_pt = (max(pt1[0] - 5, 5), max(pt1[1] - 14, 20))
    cv2.putText(vis, f"{dpi:.1f} DPI", label_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    out = image_path.with_name(image_path.stem + "_debug.png")
    cv2.imwrite(str(out), vis)
    print(f"Debug image saved to: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract DPI from a scalebar image (default 0-10 mm)."
    )
    parser.add_argument("image", type=Path, help="Path to the scalebar image.")
    parser.add_argument(
        "--scale-mm", type=float, default=10.0,
        help="Physical length the scalebar represents in mm (default: 10).",
    )
    parser.add_argument(
        "--tick-min-length", type=int, default=200,
        help="Minimum perpendicular run to count as a tick crossing (default: 200).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save an annotated overlay image alongside the input.",
    )
    args = parser.parse_args()

    original = load_image(args.image)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    binary = binarize(gray)

    axis, baseline_index = find_bar_axis(binary)
    start, end = find_tick_intersections(
        binary, axis, baseline_index, tick_min_length=args.tick_min_length
    )
    pixel_length = end - start

    if pixel_length <= 0:
        print("Error: could not measure a valid span.", file=sys.stderr)
        sys.exit(1)

    dpi = compute_dpi(pixel_length, args.scale_mm)

    print(f"Bar orientation     : {axis}")
    print(f"Baseline index      : {baseline_index} px")
    print(f"Scalebar pixel span : {pixel_length} px  ({start} -> {end})")
    print(f"Physical length     : {args.scale_mm} mm")
    print(f"Pixels per mm       : {pixel_length / args.scale_mm:.4f}")
    print(f"Estimated DPI       : {dpi:.2f}")

    if args.debug:
        save_debug(args.image, original, axis, baseline_index, start, end, dpi)


if __name__ == "__main__":
    main()