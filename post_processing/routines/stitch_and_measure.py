"""
post_processing_routines.py

All concrete post-processing routines live here.  Adding a new routine means
adding a new subclass of :class:`PostProcessingRoutine` in this file — no
other files need to change.

Current routines
----------------
- :class:`StitchAndMeasureRoutine` — stitches a folder of Y-ordered images into
  a panorama, corrects orientation, crops black borders, saves the result, then
  measures DPI from the embedded scale bar.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Generator

# When run directly, the project root (two levels up from this file) must be
# on sys.path so that `common` and `post_processing` are importable.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from common.logger import debug, warning
from common.setting_types import FileFormat
from common.fieldweaveConfig import FieldWeaveSettings
from post_processing.routines.post_processing_routine import PostProcessingRoutine


MM_PER_INCH = 25.4
DEBUG = False


# ---------------------------------------------------------------------------
# StitchAndMeasureRoutine — internal helpers
# ---------------------------------------------------------------------------

def _parse_y_position(filename: str) -> int:
    match = re.search(r'_y(\d+)', filename)
    return int(match.group(1)) if match else 0


def _collect_images(image_folder: str) -> list[tuple[cv2.Mat, str, int]]:
    valid_extensions = {f".{fmt.value}" for fmt in FileFormat}
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ])
    if not image_files:
        raise ValueError(f"No supported images found in {image_folder}")

    entries: list[tuple[int, str]] = [
        (_parse_y_position(f), f) for f in image_files
    ]
    entries.sort(key=lambda e: e[0], reverse=True)

    images: list[tuple[cv2.Mat, str, int]] = []
    for y_pos, filename in entries:
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            warning(f"StitchAndMeasureRoutine: could not load {img_path}, skipping")
            continue
        if DEBUG:
            debug(f"  + y={y_pos:>12}  {filename}")
        images.append((img, filename, y_pos))
    return images


def _mask_large_blobs(
    binary: np.ndarray,
    steps_folder: str | None = None,
    steps_prefix: str = "",
) -> np.ndarray:
    """Return a copy of *binary* with unwanted blobs zeroed out.

    Operates directly on a binary image so masking and binarization share the
    same Otsu threshold and the output remains a clean 0/255 mask.

    Three passes are applied in order:
    1. Any blob whose bounding rect exceeds 20% of the image width or height
       is removed by drawing its filled contour so nearby blobs that happen to
       fall inside the same bounding rect are not collaterally erased.
    2. Among the survivors, overlapping bounding rects are collapsed: only the
       largest blob by area in each overlapping group is kept.
    3. Any blobs beyond the top 10 by area are removed to discard noise.

    If *steps_folder* is set, the binary at each stage is saved as a PNG with
    filenames prefixed by *steps_prefix*.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary.copy()

    img_h, img_w = binary.shape

    if steps_folder is not None:
        cv2.imwrite(os.path.join(steps_folder, f"{steps_prefix}1_raw_binary.png"), binary)

    # Pass 1: remove blobs that span more than 20% of width or height.
    masked = binary.copy()
    small: list[tuple[float, tuple[int, int, int, int]]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w / img_w > 0.2 or h / img_h > 0.2:
            cv2.drawContours(masked, [c], -1, 0, cv2.FILLED)
        else:
            small.append((cv2.contourArea(c), (x, y, w, h)))

    if steps_folder is not None:
        cv2.imwrite(os.path.join(steps_folder, f"{steps_prefix}2_after_large_removal.png"), masked)

    # Pass 2: re-detect contours from the masked image so the NMS works on
    # exactly the pixels that survived pass 1, then keep only the largest blob
    # in each group of overlapping bounding rects.
    survivors, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small: list[tuple[float, tuple[int, int, int, int]]] = [
        (cv2.contourArea(c), cv2.boundingRect(c)) for c in survivors
    ]
    small.sort(key=lambda t: t[0], reverse=True)

    def overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by

    kept: list[tuple[int, int, int, int]] = []
    for area, rect in small:
        if any(overlaps(rect, k) for k in kept):
            x, y, w, h = rect
            masked[y:y + h, x:x + w] = 0
        else:
            kept.append(rect)

    if steps_folder is not None:
        cv2.imwrite(os.path.join(steps_folder, f"{steps_prefix}3_after_nms.png"), masked)

    # Pass 3: keep only the top 10 survivors by area.
    for rect in kept[10:]:
        masked[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 0

    if steps_folder is not None:
        cv2.imwrite(os.path.join(steps_folder, f"{steps_prefix}4_after_top10.png"), masked)

    return masked


def _refine_offset_with_template(
    img_a: cv2.Mat,
    img_b: cv2.Mat,
    nominal_offset: int,
    overlap_px: int,
    mask_debug_folder: str | None = None,
    pair_index: int = 0,
    steps_folder: str | None = None,
) -> int:
    """Find the Y step offset between two images by matching blobs from A into B.

    Blobs are found across the full processed binary of A (after removing the
    scale bar). Each blob's tight bounding patch is matched against B using
    normalised cross-correlation. The median Y displacement across all
    successful matches is used as the refined offset.

    Falls back to *nominal_offset* when fewer than two blobs match successfully.

    If *mask_debug_folder* is set, a composite debug image is saved as
    ``pair_<pair_index>_masks.png`` showing both source images side by side with the
    binary masks overlaid in green and matched blob bounding boxes in red.
    """
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    h_a = gray_a.shape[0]

    # Binarize and mask the full images so Otsu thresholding uses the full pixel
    # distribution, then slice the overlap band for matching: bottom of A overlaps
    # with top of B.
    full_binary_a = _mask_large_blobs(_binarize(gray_a), steps_folder=steps_folder, steps_prefix=f"pair{pair_index:02d}_a_")
    full_binary_b = _mask_large_blobs(_binarize(gray_b), steps_folder=steps_folder, steps_prefix=f"pair{pair_index:02d}_b_")

    cy_a = max(0, h_a - overlap_px)
    binary_a = full_binary_a[cy_a:, :]
    binary_b = full_binary_b[:overlap_px, :]

    contours, _ = cv2.findContours(binary_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        debug(f"  template match: no blobs found in A, using nominal={nominal_offset}px")
        return nominal_offset

    padding = 4
    # score, displacement, blob_a rect, blob_b rect (position + template size in B)
    all_candidates: list[tuple[float, int, tuple[int, int, int, int], tuple[int, int, int, int]]] = []

    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw < 5 or bh < 5:
            continue

        x0 = max(0, bx - padding)
        y0 = max(0, by - padding)
        x1 = min(binary_a.shape[1], bx + bw + padding)
        y1 = min(binary_a.shape[0], by + bh + padding)
        template = binary_a[y0:y1, x0:x1]

        if template.shape[0] > binary_b.shape[0] or template.shape[1] > binary_b.shape[1]:
            continue

        result = cv2.matchTemplate(binary_b, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, max_loc = cv2.minMaxLoc(result)
        if DEBUG:
            debug(f"    blob y0={y0} size=({y1-y0}h,{x1-x0}w) score={score:.3f} max_loc={max_loc}")
        if score < 0.5:
            continue

        match_y_in_b = max_loc[1]
        displacement = (y0 + cy_a) - match_y_in_b
        th, tw = template.shape[:2]
        all_candidates.append((score, displacement, (x0, y0, x1 - x0, y1 - y0), (max_loc[0], match_y_in_b, tw, th)))

    # NMS on match regions in B: sort by score descending, suppress any candidate
    # whose match rect overlaps one already accepted.
    all_candidates.sort(key=lambda t: t[0], reverse=True)

    def match_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by

    accepted: list[tuple[float, int, tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    accepted_b_rects: list[tuple[int, int, int, int]] = []
    for candidate in all_candidates:
        if not any(match_overlaps(candidate[3], r) for r in accepted_b_rects):
            accepted.append(candidate)
            accepted_b_rects.append(candidate[3])

    # Order-consistency filter: sort by Y in A and keep only the longest
    # subsequence whose Y positions in B are also increasing. This rejects
    # matches where blobs appear in a different vertical order in B than in A,
    # which indicates a wrong match location.
    accepted.sort(key=lambda t: t[2][1])
    lis: list[tuple[float, int, tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for candidate in accepted:
        b_y = candidate[3][1]
        if not lis or b_y > lis[-1][3][1]:
            lis.append(candidate)
        else:
            for i, kept in enumerate(lis):
                if kept[3][1] >= b_y:
                    lis[i] = candidate
                    break

    y_displacements: list[int] = [c[1] for c in lis]
    matched_blobs_a: list[tuple[int, int, int, int]] = [c[2] for c in lis]
    matched_blobs_b: list[tuple[int, int, int, int]] = [c[3] for c in lis]
    if DEBUG:
        debug(f"  template match: {len(y_displacements)} blob(s) matched, displacements={y_displacements}")

    if mask_debug_folder is not None:
        def _make_overlay(
            img_bgr: cv2.Mat,
            binary: np.ndarray,
            boxes: list[tuple[int, int, int, int]],
            row_start: int,
            row_end: int,
        ) -> cv2.Mat:
            vis = img_bgr.copy()
            green_layer = np.zeros_like(vis)
            green_layer[row_start:row_end][binary > 0] = (0, 255, 0)
            vis = cv2.addWeighted(vis, 0.7, green_layer, 0.3, 0)
            for bx, by, bw, bh in boxes:
                ry = by + row_start
                cv2.rectangle(vis, (bx, ry), (bx + bw, ry + bh), (0, 0, 255), 2)
            return vis

        vis_a = _make_overlay(img_a, binary_a, matched_blobs_a, cy_a, img_a.shape[0])
        vis_b = _make_overlay(img_b, binary_b, matched_blobs_b, 0, overlap_px)

        h_a_vis, w_a_vis = vis_a.shape[:2]
        h_b_vis, w_b_vis = vis_b.shape[:2]
        combined_h = max(h_a_vis, h_b_vis)
        combined_w = w_a_vis + w_b_vis + 4
        combined = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
        combined[:h_a_vis, :w_a_vis] = vis_a
        combined[:h_b_vis, w_a_vis + 4:] = vis_b
        cv2.line(combined, (w_a_vis + 2, 0), (w_a_vis + 2, combined_h), (128, 128, 128), 2)
        cv2.imwrite(os.path.join(mask_debug_folder, f"pair_{pair_index:02d}_masks.png"), combined)

    if len(y_displacements) < 1:
        debug(f"  template match: too few matches, using nominal={nominal_offset}px")
        return nominal_offset

    refined = int(np.median(y_displacements))
    if DEBUG:
        debug(f"  template match: nominal={nominal_offset}px  refined={refined}px")
    if refined <= 0:
        debug(f"  template match: refined offset non-positive, falling back to nominal={nominal_offset}px")
        return nominal_offset
    return refined


def _save_overlap_debug(
    img_a: cv2.Mat,
    img_b: cv2.Mat,
    offset: int,
    pair_index: int,
    mask_debug_folder: str,
) -> None:
    """Save a composite showing the overlap between img_a and img_b at the given offset.

    The overlap region is rendered as a 50/50 alpha blend of A (bottom rows) and
    B (top rows) so misalignment appears as ghosting or fringing. A horizontal red
    line marks the seam midpoint.
    """
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    overlap_rows = h_a - offset
    if overlap_rows <= 0:
        return

    seam = overlap_rows // 2
    canvas_w = max(w_a, w_b)
    canvas = np.zeros((overlap_rows, canvas_w, 3), dtype=np.uint8)

    strip_a = img_a[offset:, :]
    strip_b = img_b[:overlap_rows, :]

    rows = min(strip_a.shape[0], strip_b.shape[0], overlap_rows)
    canvas[:rows, :w_a] = strip_a[:rows, :]
    blend = canvas.copy()
    blend[:rows, :w_b] = strip_b[:rows, :]
    canvas = cv2.addWeighted(canvas, 0.5, blend, 0.5, 0)
    cv2.line(canvas, (0, seam), (canvas_w, seam), (0, 0, 255), 1)

    cv2.imwrite(os.path.join(mask_debug_folder, f"pair_{pair_index:02d}_overlap.png"), canvas)


def _composite_images(
    images: list[tuple[cv2.Mat, str, int]],
    refined_offsets: list[int],
) -> cv2.Mat:
    h_a, w_a = images[0][0].shape[:2]
    total_h = h_a + sum(refined_offsets)
    canvas_w = max(img.shape[1] for img, _, _ in images)
    debug(f"_composite_images: canvas=({total_h}h, {canvas_w}w)")
    canvas = np.zeros((total_h, canvas_w, 3), dtype=np.uint8)

    img_a, _, _ = images[0]
    canvas[:h_a, :w_a] = img_a

    cum_y = 0
    for i, (img_b, _, _) in enumerate(images[1:]):
        offset = refined_offsets[i]
        prev_img, _, _ = images[i]
        h_prev = prev_img.shape[0]
        h_b, w_b = img_b.shape[:2]

        overlap_start_canvas = cum_y + offset
        overlap_end_canvas = cum_y + h_prev
        overlap_rows = overlap_end_canvas - overlap_start_canvas

        if overlap_rows > 0:
            seam_canvas = overlap_start_canvas + overlap_rows // 2
            seam_in_b = seam_canvas - (cum_y + offset)
            b_start_canvas = cum_y + offset
            b_end_canvas = min(cum_y + offset + h_b, total_h)
            b_src_rows = b_end_canvas - b_start_canvas

            canvas[b_start_canvas + seam_in_b:b_end_canvas, :w_b] = img_b[seam_in_b:seam_in_b + (b_src_rows - seam_in_b), :]
            if DEBUG:
                debug(f"  pair {i}: offset={offset}px overlap={overlap_rows}px seam at canvas row {seam_canvas} (b row {seam_in_b})")
        else:
            b_start_canvas = cum_y + offset
            b_end_canvas = min(b_start_canvas + h_b, total_h)
            src_rows = b_end_canvas - b_start_canvas
            canvas[b_start_canvas:b_end_canvas, :w_b] = img_b[:src_rows, :]
            if DEBUG:
                debug(f"  pair {i}: offset={offset}px no overlap, placing directly")

        cum_y += offset

    return canvas


def _stitch_images(
    images: list[tuple[cv2.Mat, str, int]],
    overlap_frac: float,
    mask_debug_folder: str | None = None,
    steps_folder: str | None = None,
) -> tuple[cv2.Mat, list[int]] | None:
    debug(f"_stitch_images: stitching {len(images)} images with {overlap_frac:.1%} overlap")

    h_a, w_a = images[0][0].shape[:2]
    overlap_px = round(h_a * overlap_frac)
    nominal_offset = h_a - overlap_px
    debug(f"_stitch_images: image shape=({h_a}h, {w_a}w)  overlap={overlap_px}px  nominal offset={nominal_offset}px")

    refined_offsets: list[int] = []
    for i in range(len(images) - 1):
        img_a, _, _ = images[i]
        img_b, _, _ = images[i + 1]
        refined = _refine_offset_with_template(
            img_a, img_b, nominal_offset,
            overlap_px=overlap_px,
            mask_debug_folder=mask_debug_folder,
            pair_index=i,
            steps_folder=steps_folder,
        )
        refined_offsets.append(refined)
        if mask_debug_folder is not None:
            _save_overlap_debug(img_a, img_b, refined, i, mask_debug_folder)

    debug(f"_stitch_images: refined offsets={refined_offsets}")
    canvas = _composite_images(images, refined_offsets)
    debug("_stitch_images: stitching complete")
    return canvas, refined_offsets


def _crop_black_borders(image: cv2.Mat) -> cv2.Mat:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y:y + h, x:x + w]
    return image


def _rotate_image(image: cv2.Mat, degrees: int) -> cv2.Mat:
    if degrees == 0:
        return image
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"degrees must be 0, 90, 180, or 270; got {degrees}")


def _binarize(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _longest_runs(matrix: np.ndarray) -> np.ndarray:
    """Return the longest contiguous non-zero run for each row of *matrix*.

    Uses vectorised diff operations so no Python loop runs over individual
    pixels, keeping the GIL free for the duration.
    """
    h, w = matrix.shape
    mask = matrix > 0
    # Pad each row with a False sentinel on each side so diff captures runs
    # that start at column 0 or end at column w-1.
    padded = np.concatenate(
        [np.zeros((h, 1), dtype=bool), mask, np.zeros((h, 1), dtype=bool)],
        axis=1,
    )
    diff = np.diff(padded.astype(np.int8), axis=1)
    runs = np.zeros(h, dtype=np.int32)
    for r in range(h):
        starts = np.where(diff[r] == 1)[0]
        ends = np.where(diff[r] == -1)[0]
        if starts.size:
            runs[r] = int((ends - starts).max())
    return runs


def _find_bar_axis(binary: np.ndarray) -> tuple[str, int]:
    h, w = binary.shape
    row_runs = _longest_runs(binary)
    col_runs = _longest_runs(binary.T)
    long_side_is_horizontal = w >= h
    if long_side_is_horizontal:
        return "horizontal", int(row_runs.argmax())
    return "vertical", int(col_runs.argmax())


def _detect_orientation(image: cv2.Mat) -> int:
    """Return the clockwise rotation (0, 90, 180, or 270) needed to correct orientation.

    The scale bar runs along the long axis of the image.  _find_bar_axis gives
    the axis and baseline index of the bar using longest-run detection, which is
    robust to tilted or fragmented bars.  All pixels within a band around the
    baseline are zeroed out, leaving only number glyphs and dust.  The centroid
    of those remaining pixels relative to the image centre determines which side
    the numbers are on, which directly maps to the required rotation.

    For a landscape image (horizontal bar):
      - numbers above centre  -> already upright (0°)
      - numbers below centre  -> upside-down (180°)

    For a portrait image (vertical bar):
      - numbers left of centre  -> 270°
      - numbers right of centre -> 90°
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)
    h, w = image.shape[:2]
    axis, baseline_index = _find_bar_axis(binary)

    bar_half_width = (h if axis == "horizontal" else w) // 10
    numbers_only = binary.copy()
    if axis == "horizontal":
        y0 = max(0, baseline_index - bar_half_width)
        y1 = min(h, baseline_index + bar_half_width + 1)
        numbers_only[y0:y1, :] = 0
        ys, _ = np.where(numbers_only > 0)
        offset = float(ys.mean()) - h / 2.0 if ys.size > 0 else 0.0
        rotation = 0 if offset < 0 else 180
    else:
        x0 = max(0, baseline_index - bar_half_width)
        x1 = min(w, baseline_index + bar_half_width + 1)
        numbers_only[:, x0:x1] = 0
        _, xs = np.where(numbers_only > 0)
        offset = float(xs.mean()) - w / 2.0 if xs.size > 0 else 0.0
        rotation = 90 if offset < 0 else 270

    debug(f"_detect_orientation: image=({h}h,{w}w) axis={axis} baseline={baseline_index} offset={offset:.1f}")
    debug(f"_detect_orientation: selected {rotation}°")
    return rotation


def _find_tick_intersections(
    binary: np.ndarray,
    axis: str,
    baseline_index: int,
    baseline_band: int = 3,
    tick_min_length: int = 200,
) -> tuple[int, int]:
    h, w = binary.shape
    if axis == "horizontal":
        along_size = w

        def perp_run_at(pos: int) -> int:
            return int(np.count_nonzero(binary[:, pos]))

        def baseline_hit(pos: int) -> bool:
            y0 = max(0, baseline_index - baseline_band)
            y1 = min(h, baseline_index + baseline_band + 1)
            return bool(binary[y0:y1, pos].max() > 0)
    else:
        along_size = h

        def perp_run_at(pos: int) -> int:
            return int(np.count_nonzero(binary[pos, :]))

        def baseline_hit(pos: int) -> bool:
            x0 = max(0, baseline_index - baseline_band)
            x1 = min(w, baseline_index + baseline_band + 1)
            return bool(binary[pos, x0:x1].max() > 0)

    tick_positions = [
        pos for pos in range(along_size)
        if baseline_hit(pos) and perp_run_at(pos) >= tick_min_length
    ]
    if len(tick_positions) < 2:
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


EXPECTED_TICK_COUNT = 101


def _find_ticks(
    binary: np.ndarray,
    axis: str,
    baseline_index: int,
    baseline_band: int = 3,
    tick_min_length: int = 200,
) -> list[int]:
    """Return the center position of each distinct tick mark along the scale bar.

    Positions are in the along-axis coordinate (x for horizontal, y for vertical).
    Each contiguous run of qualifying pixel positions is collapsed to its midpoint.
    """
    h, w = binary.shape
    if axis == "horizontal":
        along_size = w

        def perp_run_at(pos: int) -> int:
            return int(np.count_nonzero(binary[:, pos]))

        def baseline_hit(pos: int) -> bool:
            y0 = max(0, baseline_index - baseline_band)
            y1 = min(h, baseline_index + baseline_band + 1)
            return bool(binary[y0:y1, pos].max() > 0)
    else:
        along_size = h

        def perp_run_at(pos: int) -> int:
            return int(np.count_nonzero(binary[pos, :]))

        def baseline_hit(pos: int) -> bool:
            x0 = max(0, baseline_index - baseline_band)
            x1 = min(w, baseline_index + baseline_band + 1)
            return bool(binary[pos, x0:x1].max() > 0)

    tick_positions = [
        pos for pos in range(along_size)
        if baseline_hit(pos) and perp_run_at(pos) >= tick_min_length
    ]
    if not tick_positions:
        return []

    centers: list[int] = []
    run_start = tick_positions[0]
    run_end = tick_positions[0]
    for i in range(1, len(tick_positions)):
        if tick_positions[i] - tick_positions[i - 1] > 1:
            centers.append((run_start + run_end) // 2)
            run_start = tick_positions[i]
        run_end = tick_positions[i]
    centers.append((run_start + run_end) // 2)
    return centers


TICK_SPACING_TOLERANCE = 0.20


def _find_outlier_gaps(tick_centers: list[int]) -> list[tuple[int, int, float]]:
    """Return gaps between adjacent ticks that deviate from the median by more than
    TICK_SPACING_TOLERANCE.

    Each entry is (left_tick_index, right_tick_index, gap_px) for the outlying gap.
    """
    if len(tick_centers) < 2:
        return []
    gaps = [tick_centers[i + 1] - tick_centers[i] for i in range(len(tick_centers) - 1)]
    median_gap = float(np.median(gaps))
    return [
        (i, i + 1, float(gap))
        for i, gap in enumerate(gaps)
        if abs(gap - median_gap) / median_gap > TICK_SPACING_TOLERANCE
    ]


def _restitch_with_corrected_offsets(
    images: list[tuple[cv2.Mat, str, int]],
    refined_offsets: list[int],
    ticks: list[int],
    outlier_gaps: list[tuple[int, int, float]],
    tick_count: int,
) -> cv2.Mat:
    """Restitch *images* with corrected offsets to fix outlier tick spacing.

    The scale bar is vertical so tick positions are Y coordinates in the panorama.
    The visible seam between image i and image i+1 sits at the midpoint of their
    overlap region.  For each outlier gap, the seam that falls between its two tick
    Y positions is identified and its offset is adjusted.

    When tick_count > EXPECTED_TICK_COUNT, a spuriously small gap means two ticks
    were produced by a misaligned seam — the gap is closed entirely by absorbing the
    full actual_gap into the offset.  Otherwise the offset is adjusted by the
    difference between the actual gap and the median so the spacing matches.
    """
    gaps = [ticks[i + 1] - ticks[i] for i in range(len(ticks) - 1)]
    median_gap = int(round(float(np.median(gaps))))
    too_many_ticks = tick_count > EXPECTED_TICK_COUNT

    seam_y_positions: list[int] = []
    cum_y = 0
    for i, offset in enumerate(refined_offsets):
        h_prev = images[i][0].shape[0]
        overlap_rows = h_prev - offset
        seam_y = cum_y + offset + max(0, overlap_rows) // 2
        seam_y_positions.append(seam_y)
        cum_y += offset

    corrected_offsets = list(refined_offsets)
    for left_i, right_i, _ in outlier_gaps:
        left_tick = ticks[left_i]
        right_tick = ticks[right_i]
        actual_gap = right_tick - left_tick
        delta = actual_gap if too_many_ticks else actual_gap - median_gap

        seam_idx = next(
            (j for j, seam_y in enumerate(seam_y_positions) if left_tick <= seam_y <= right_tick),
            None,
        )
        if seam_idx is None:
            debug(f"_restitch_with_corrected_offsets: no seam found between ticks {left_i} (y={left_tick}) and {right_i} (y={right_tick}), skipping")
            continue

        corrected_offsets[seam_idx] -= delta
        debug(
            f"_restitch_with_corrected_offsets: gap #{left_i}-#{right_i} "
            f"actual={actual_gap}px median={median_gap}px delta={delta}px "
            f"({'close gap entirely' if too_many_ticks else 'adjust to median'}) "
            f"-> offset[{seam_idx}] {refined_offsets[seam_idx]} -> {corrected_offsets[seam_idx]}"
        )

    return _composite_images(images, corrected_offsets)


def _extract_dpi(image: cv2.Mat, scale_mm: float, tick_min_length: int) -> tuple[float, int, list[tuple[int, int, float]]] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)
    axis, baseline_index = _find_bar_axis(binary)
    start, end = _find_tick_intersections(binary, axis, baseline_index, tick_min_length=tick_min_length)
    pixel_length = end - start
    if pixel_length <= 0:
        warning("_extract_dpi: could not measure a valid scalebar span")
        return None

    ticks = _find_ticks(binary, axis, baseline_index, tick_min_length=tick_min_length)
    tick_count = len(ticks)
    debug(f"_extract_dpi: tick count={tick_count} (expected {EXPECTED_TICK_COUNT})")
    if tick_count != EXPECTED_TICK_COUNT:
        warning(f"_extract_dpi: expected {EXPECTED_TICK_COUNT} ticks but found {tick_count}; DPI measurement may be unreliable")

    outlier_gaps = _find_outlier_gaps(ticks)
    if outlier_gaps:
        warning(f"_extract_dpi: {len(outlier_gaps)} gap(s) with unusual spacing detected")

    dpi = (pixel_length / scale_mm) * MM_PER_INCH
    debug(f"_extract_dpi: axis={axis} baseline={baseline_index}px span={pixel_length}px ({start}->{end})")
    debug(f"_extract_dpi: physical={scale_mm}mm px/mm={pixel_length / scale_mm:.4f} DPI={dpi:.2f}")
    return dpi, tick_count, outlier_gaps


def _build_dpi_debug_overlay(
    image: cv2.Mat, scale_mm: float, tick_min_length: int, *, debug: bool = False
) -> cv2.Mat | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)
    axis, baseline_index = _find_bar_axis(binary)
    start, end = _find_tick_intersections(binary, axis, baseline_index, tick_min_length=tick_min_length)
    pixel_length = end - start
    if pixel_length <= 0:
        return None

    ticks = _find_ticks(binary, axis, baseline_index, tick_min_length=tick_min_length)
    tick_count = len(ticks)
    dpi = (pixel_length / scale_mm) * MM_PER_INCH
    vis = image.copy()

    if debug:
        detected_mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        detected_mask[:, :, 0] = 0
        detected_mask[:, :, 2] = 0
        vis = cv2.addWeighted(vis, 0.7, detected_mask, 0.3, 0)

        h, w = vis.shape[:2]
        if axis == "horizontal":
            cv2.line(vis, (0, baseline_index), (w, baseline_index), (255, 165, 0), 1)
        else:
            cv2.line(vis, (baseline_index, 0), (baseline_index, h), (255, 165, 0), 1)

    h, w = vis.shape[:2]
    tick_color = (0, 255, 255)
    tick_half_len = 12
    for center in ticks:
        if axis == "horizontal":
            pt_top = (center, max(0, baseline_index - tick_half_len))
            pt_bot = (center, min(h - 1, baseline_index + tick_half_len))
            cv2.line(vis, pt_top, pt_bot, tick_color, 1)
        else:
            pt_left = (max(0, baseline_index - tick_half_len), center)
            pt_right = (min(w - 1, baseline_index + tick_half_len), center)
            cv2.line(vis, pt_left, pt_right, tick_color, 1)

    if axis == "horizontal":
        pt1, pt2 = (start, baseline_index), (end, baseline_index)
    else:
        pt1, pt2 = (baseline_index, start), (baseline_index, end)
    cv2.line(vis, pt1, pt2, (0, 0, 255), 2)
    cv2.circle(vis, pt1, 6, (0, 255, 0), -1)
    cv2.circle(vis, pt2, 6, (0, 255, 0), -1)
    label_pt = (max(pt1[0] - 5, 5), max(pt1[1] - 14, 20))
    cv2.putText(vis, f"{dpi:.1f} DPI  ticks={tick_count}/{EXPECTED_TICK_COUNT}", label_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return vis


# ---------------------------------------------------------------------------
# StitchAndMeasureRoutine
# ---------------------------------------------------------------------------

class StitchAndMeasureRoutine(PostProcessingRoutine):
    """
    Stitch a folder of Y-ordered images into a panorama, then measure DPI.

    *input_folder* must contain exactly one subfolder of images whose
    filenames include ``_y<position>`` tags.  The output image is written to
    *input_folder* using the folder's own name as the filename stem, with the
    extension determined by the camera's current format setting.

    On success, :attr:`~PostProcessingRoutine.result` is populated with
    ``success=True`` and the following keys in ``result.data``:

    - ``output_path`` (:class:`str`): absolute path to the saved panorama.
    - ``debug_path`` (:class:`str` or ``None``): path to the DPI debug overlay, or None.
    - ``dpi`` (:class:`float` or ``None``): measured DPI, or None if detection failed.
    - ``tick_count`` (:class:`int` or ``None``): number of ticks detected on the scale bar.
    - ``tick_count_valid`` (:class:`bool` or ``None``): whether the detected tick count matches the expected value.
    - ``tick_spacing_valid`` (:class:`bool`): whether all inter-tick gaps were within tolerance.
    - ``tick_outlier_gaps`` (:class:`list`): gaps flagged as outliers, each a ``(left_i, right_i, gap_px)`` tuple.
    - ``qa_pass`` (:class:`bool`): True if tick count and spacing both passed; False otherwise or if DPI detection failed.
    - ``qa_warnings`` (:class:`list` of :class:`str`): human-readable warning messages for any QA failures (empty on a clean pass).
    - ``image_width`` (:class:`int`): panorama width in pixels.
    - ``image_height`` (:class:`int`): panorama height in pixels.
    - ``stitched_rgb`` (:class:`numpy.ndarray`): final panorama in RGB888 order,
      shape ``(H, W, 3)``, dtype ``uint8``.  Safe to wrap in QImage directly on
      the GUI thread after :meth:`~PostProcessingRoutine.wait` returns.

    Parameters
    ----------
    settings:
        Application-wide settings.  Stitching parameters are read from
        ``settings.post_processing.stitch_and_measure``.
    input_folder:
        Absolute path to the root folder containing exactly one image subfolder.
    """

    job_name = "Stitch and Measure"

    def __init__(
        self,
        settings: FieldWeaveSettings,
        input_folder: str,
        *,
        save_dpi: bool = True,
        standalone: bool = False,
        overlap_frac: float | None = None,
    ) -> None:
        super().__init__(settings)
        self.input_folder = input_folder
        self._save_dpi = save_dpi
        self._standalone = standalone
        self._overlap_frac = overlap_frac

    def steps(self) -> Generator[None, None, None]:
        sm = self.settings.post_processing.stitch_and_measure

        if self._standalone:
            ctx = None
            overlap_frac = self._overlap_frac
            if overlap_frac is None:
                raise ValueError("overlap_frac must be provided when running standalone")
        else:
            from common.app_context import get_app_context
            ctx = get_app_context()
            overlap_frac = ctx.motion.settings.automation.overlap_y_pct / 100.0

        # ----------------------------------------------------------------
        # Step 1: discover and load images
        # ----------------------------------------------------------------
        self._set_status("Collecting images", 0, 4)

        subfolders = [
            entry.path
            for entry in os.scandir(self.input_folder)
            if entry.is_dir() and os.path.basename(entry.path) not in {"masks", "steps"}
        ]
        if len(subfolders) == 0:
            raise ValueError("No subfolders found in the input folder")
        if len(subfolders) > 1:
            names = ", ".join(os.path.basename(s) for s in subfolders)
            raise ValueError(
                f"Expected exactly one subfolder, found {len(subfolders)}: {names}"
            )

        image_folder = subfolders[0]
        debug(f"StitchAndMeasureRoutine: image folder: {image_folder}")

        images = _collect_images(image_folder)
        if len(images) < 2:
            raise ValueError(
                f"Need at least 2 images to stitch, found {len(images)}"
            )

        yield

        # ----------------------------------------------------------------
        # Step 2: stitch
        # ----------------------------------------------------------------
        self._set_status("Stitching images", 1, 4)

        stitch_result = _stitch_images(images, overlap_frac, mask_debug_folder=None)
        if stitch_result is None:
            raise RuntimeError(
                "Stitching failed — ensure the overlap value is correct and "
                "that Y positions in filenames reflect the correct scan order"
            )
        stitched, refined_offsets = stitch_result

        yield

        # ----------------------------------------------------------------
        # Step 3: gap correction, crop, and rotate
        # ----------------------------------------------------------------
        self._set_status("Correcting orientation", 2, 4)

        # Gap correction must run on the raw stitched panorama before any
        # crop or rotation, because _restitch_with_corrected_offsets locates
        # seams using the same Y coordinate space as the tick positions
        # returned by _extract_dpi. Rotating first would misalign the two.
        dpi_result = _extract_dpi(stitched, sm.scale_mm, sm.tick_min_length)
        dpi: float | None = None
        tick_count: int | None = None
        outlier_gaps: list[tuple[int, int, float]] = []
        if dpi_result is not None:
            dpi, tick_count, outlier_gaps = dpi_result
            if outlier_gaps:
                gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
                binary = _binarize(gray)
                axis, baseline_index = _find_bar_axis(binary)
                ticks = _find_ticks(binary, axis, baseline_index, tick_min_length=sm.tick_min_length)
                corrected = _restitch_with_corrected_offsets(images, refined_offsets, ticks, outlier_gaps, tick_count)
                corrected_result = _extract_dpi(corrected, sm.scale_mm, sm.tick_min_length)
                if corrected_result is not None:
                    stitched = corrected
                    dpi, tick_count, outlier_gaps = corrected_result

        if sm.crop_borders:
            debug("StitchAndMeasureRoutine: cropping black borders")
            stitched = _crop_black_borders(stitched)

        if sm.auto_rotate:
            rotation = _detect_orientation(stitched)
            if rotation != 0:
                debug(f"StitchAndMeasureRoutine: rotating {rotation}° clockwise")
                stitched = _rotate_image(stitched, rotation)
            else:
                debug("StitchAndMeasureRoutine: orientation correct, no rotation needed")

        yield

        # ----------------------------------------------------------------
        # Step 4: save and write debug overlay
        # ----------------------------------------------------------------
        self._set_status("Measuring DPI", 3, 4)

        root_name = os.path.basename(self.input_folder)
        if self._save_dpi and not self._standalone and ctx is not None:
            extension = ctx.camera.underlying_camera.settings.fformat.value
            output_path = os.path.join(self.input_folder, f"{root_name}.{extension}")
        else:
            output_path = os.path.join(self.input_folder, "stitched.jpg")

        cv2.imwrite(output_path, stitched)
        debug(f"StitchAndMeasureRoutine: panorama saved to {output_path}")

        qa_warnings: list[str] = []
        if tick_count is not None and tick_count != EXPECTED_TICK_COUNT:
            qa_warnings.append(f"expected {EXPECTED_TICK_COUNT} ticks but found {tick_count}")
        for left_i, right_i, gap in outlier_gaps:
            qa_warnings.append(f"unusual gap between tick #{left_i} and #{right_i} ({gap:.0f}px)")
        qa_pass = tick_count == EXPECTED_TICK_COUNT and len(outlier_gaps) == 0 if dpi is not None else False

        if dpi is not None and self._save_dpi:
            lines: list[str] = [
                f"{dpi:.2f}",
                "PASS" if qa_pass else "FAIL",
                f"{tick_count if tick_count is not None else 0}/{EXPECTED_TICK_COUNT} ticks",
            ]
            for msg in qa_warnings:
                lines.append(f"WARNING: {msg}")
            dpi_txt_path = os.path.join(self.input_folder, "DPI.txt")
            with open(dpi_txt_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            debug(f"StitchAndMeasureRoutine: DPI written to {dpi_txt_path}")

        if dpi is not None and self._save_dpi and not self._standalone:
            mv = ctx.machine_vision
            mv.settings.dpi = dpi
            mv.save_settings()
            mv.notify_settings_changed()
            debug(f"StitchAndMeasureRoutine: DPI {dpi:.2f} updated and saved")

        debug_path: str | None = None
        if sm.save_debug_overlay and dpi is not None:
            overlay = _build_dpi_debug_overlay(stitched, sm.scale_mm, sm.tick_min_length, debug=DEBUG)
            if overlay is not None:
                stem, ext = os.path.splitext(output_path)
                debug_path = f"{stem}_debug{ext}"
                cv2.imwrite(debug_path, overlay)
                debug(f"StitchAndMeasureRoutine: debug overlay saved to {debug_path}")

        h, w = stitched.shape[:2]
        self._set_result(
            success=True,
            output_path=output_path,
            debug_path=debug_path,
            dpi=dpi,
            tick_count=tick_count,
            tick_count_valid=tick_count == EXPECTED_TICK_COUNT if tick_count is not None else None,
            tick_spacing_valid=len(outlier_gaps) == 0,
            tick_outlier_gaps=outlier_gaps,
            qa_pass=qa_pass,
            qa_warnings=qa_warnings,
            image_width=w,
            image_height=h,
            stitched_rgb=cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB),
        )

        self._set_status("Done", 4, 4)
        yield

# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import subprocess
    import sys

    parser = argparse.ArgumentParser(
        description="Stitch a folder of Y-ordered images into a panorama."
    )
    parser.add_argument("folder", help="Path to the folder containing the image subfolder")
    parser.add_argument(
        "--overlap",
        type=float,
        required=True,
        metavar="FRAC",
        help="Fraction of image height that consecutive images overlap (e.g. 0.7 for 70%%)",
    )
    parser.add_argument("--save-masks", action="store_true", help="Save per-pair debug images to a masks/ subfolder showing binary masks overlaid on source images with matched blob bounding boxes")
    parser.add_argument("--save-steps", action="store_true", help="Save intermediate binary mask images at each filtering stage to a steps/ subfolder for debugging blob detection")
    parser.add_argument("--save-dpi", action="store_true", help="Write DPI.txt containing the measured DPI to the output folder")
    parser.add_argument("--debug", action="store_true", help="Overlay detected points and lines on the debug image")
    args = parser.parse_args()

    DEBUG = args.debug

    if args.overlap > 1.0:
        args.overlap /= 100.0

    settings = FieldWeaveSettings()

    mask_debug_folder: str | None = None
    if args.save_masks:
        mask_debug_folder = os.path.join(args.folder, "masks")
        os.makedirs(mask_debug_folder, exist_ok=True)

    steps_folder: str | None = None
    if args.save_steps:
        steps_folder = os.path.join(args.folder, "steps")
        os.makedirs(steps_folder, exist_ok=True)

    def _apply_gap_correction(
        images: list[tuple[cv2.Mat, str, int]],
        refined_offsets: list[int],
        dpi: float,
        tick_count: int,
        ticks: list[int],
        outlier_gaps: list[tuple[int, int, float]],
        base_output_path: str,
        scale_mm: float,
        tick_min_length: int,
    ) -> None:
        corrected = _restitch_with_corrected_offsets(images, refined_offsets, ticks, outlier_gaps, tick_count)
        corrected_result = _extract_dpi(corrected, scale_mm, tick_min_length)
        stem, ext = os.path.splitext(base_output_path)
        corrected_path = f"{stem}_corrected{ext}"
        cv2.imwrite(corrected_path, corrected)
        print(f"DPI (original):  {dpi:.2f}")
        if corrected_result is not None:
            corrected_dpi, corrected_tick_count, corrected_outlier_gaps = corrected_result
            print(f"DPI (corrected): {corrected_dpi:.2f}")
            status = "PASS" if corrected_tick_count == EXPECTED_TICK_COUNT else "FAIL"
            print(f"Ticks (corrected): {corrected_tick_count}/{EXPECTED_TICK_COUNT} [{status}]")
            if corrected_outlier_gaps:
                gaps_str = ", ".join(f"#{i}-#{j} ({gap:.0f}px)" for i, j, gap in corrected_outlier_gaps)
                print(f"Spacing FAIL (corrected): unusual gaps at {gaps_str}")
        print(f"Corrected: {corrected_path}")

    if mask_debug_folder is not None:
        subfolders = [e.path for e in os.scandir(args.folder) if e.is_dir() and os.path.basename(e.path) not in {"masks", "steps"}]
        if len(subfolders) != 1:
            print("Expected exactly one image subfolder.", file=sys.stderr)
            sys.exit(1)
        images = _collect_images(subfolders[0])
        sm = settings.post_processing.stitch_and_measure
        stitch_result = _stitch_images(images, args.overlap, mask_debug_folder=mask_debug_folder, steps_folder=steps_folder)
        if stitch_result is None:
            print("Stitching failed.", file=sys.stderr)
            sys.exit(1)
        stitched, refined_offsets = stitch_result
        output_path = os.path.join(args.folder, "stitched.jpg")
        cv2.imwrite(output_path, stitched)
        h, w = stitched.shape[:2]
        print(f"Saved:  {output_path}")
        print(f"Size:   {w}w x {h}h px")
        print(f"Masks:  {mask_debug_folder}")
        dpi_result = _extract_dpi(stitched, sm.scale_mm, sm.tick_min_length)
        if dpi_result is not None:
            dpi, tick_count, outlier_gaps = dpi_result
            status = "PASS" if tick_count == EXPECTED_TICK_COUNT else "FAIL"
            print(f"Ticks:  {tick_count}/{EXPECTED_TICK_COUNT} [{status}]")
            if outlier_gaps:
                gaps_str = ", ".join(f"#{i}-#{j} ({gap:.0f}px)" for i, j, gap in outlier_gaps)
                print(f"Spacing FAIL: unusual gaps at {gaps_str}")
                gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
                binary = _binarize(gray)
                axis, baseline_index = _find_bar_axis(binary)
                ticks = _find_ticks(binary, axis, baseline_index, tick_min_length=sm.tick_min_length)
                _apply_gap_correction(images, refined_offsets, dpi, tick_count, ticks, outlier_gaps, output_path, sm.scale_mm, sm.tick_min_length)
            else:
                print(f"DPI:    {dpi:.2f}")
    else:
        routine = StitchAndMeasureRoutine(
            settings=settings,
            input_folder=args.folder,
            save_dpi=args.save_dpi,
            standalone=True,
            overlap_frac=args.overlap,
        )
        routine.start()
        routine.wait()

        result = routine.result
        if result is None or not result.success:
            print("Stitching failed.", file=sys.stderr)
            sys.exit(1)

        output_path = result.get("output_path")
        print(f"Saved:  {output_path}")
        print(f"Size:   {result.get('image_width')}w x {result.get('image_height')}h px")
        dpi = result.get("dpi")
        tick_count = result.get("tick_count")
        if tick_count is not None:
            status = "PASS" if result.get("tick_count_valid") else "FAIL"
            print(f"Ticks:  {tick_count}/{EXPECTED_TICK_COUNT} [{status}]")
        outlier_gaps = result.get("tick_outlier_gaps") or []
        if outlier_gaps:
            gaps_str = ", ".join(f"#{i}-#{j} ({gap:.0f}px)" for i, j, gap in outlier_gaps)
            print(f"Spacing FAIL: unusual gaps at {gaps_str}")
        if dpi is not None:
            print(f"DPI:    {dpi:.2f}")

    if DEBUG:
        sm = FieldWeaveSettings().post_processing.stitch_and_measure
        if args.save_masks:
            stitched_bgr = cv2.imread(output_path)
        else:
            stitched_bgr = cv2.cvtColor(result.get("stitched_rgb"), cv2.COLOR_RGB2BGR)
        overlay = _build_dpi_debug_overlay(stitched_bgr, sm.scale_mm, sm.tick_min_length, debug=True)
        if overlay is not None:
            stem, ext = os.path.splitext(output_path)
            debug_path = f"{stem}_debug{ext}"
            cv2.imwrite(debug_path, overlay)
            print(f"Debug: {debug_path}")
        output_path = debug_path

    if sys.platform == "win32":
        os.startfile(output_path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", output_path])
    else:
        subprocess.Popen(["xdg-open", output_path])