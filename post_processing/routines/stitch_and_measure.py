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
        debug(f"  + y={y_pos:>12}  {filename}")
        images.append((img, filename, y_pos))
    return images


def _mask_longest_blob(gray: np.ndarray) -> np.ndarray:
    """Return a copy of *gray* with the blob whose bounding rect spans the most
    image width zeroed out.

    Using width fraction rather than absolute width ensures the tick bar —
    which always runs nearly edge to edge — is selected even when the first
    image has fewer ticks and a number label happens to be wider in pixels.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray.copy()
    img_w = gray.shape[1]
    x, y, w, h = cv2.boundingRect(max(contours, key=lambda c: cv2.boundingRect(c)[2] / img_w))
    masked = gray.copy()
    masked[y:y + h, x:x + w] = 0
    return masked


def _refine_offset_with_template(
    img_a: cv2.Mat,
    img_b: cv2.Mat,
    nominal_offset: int,
    mask_debug_folder: str | None = None,
    pair_index: int = 0,
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

    h_a, w_a = gray_a.shape
    h_b, w_b = gray_b.shape
    cy_a, cx_a = int(h_a * 0.2), int(w_a * 0.2)
    cy_b, cx_b = int(h_b * 0.2), int(w_b * 0.2)
    cropped_gray_a = gray_a[cy_a:h_a - cy_a, cx_a:w_a - cx_a]
    cropped_gray_b = gray_b[cy_b:h_b - cy_b, cx_b:w_b - cx_b]

    binary_a = cv2.bitwise_and(_mask_longest_blob(cropped_gray_a), _binarize(cropped_gray_a))
    binary_b = cv2.bitwise_and(_mask_longest_blob(cropped_gray_b), _binarize(cropped_gray_b))

    contours, _ = cv2.findContours(binary_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        debug(f"  template match: no blobs found in A, using nominal={nominal_offset}px")
        return nominal_offset

    padding = 4
    y_displacements: list[int] = []
    matched_blobs_a: list[tuple[int, int, int, int]] = []
    matched_blobs_b: list[tuple[int, int, int, int]] = []

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
        debug(f"    blob y0={y0} size=({y1-y0}h,{x1-x0}w) score={score:.3f} max_loc={max_loc}")
        if score < 0.5:
            continue

        # y0 is in binary_a coordinates; adding cy_a gives original image A coordinates.
        # max_loc[1] is in binary_b coordinates; adding cy_b gives original image B coordinates.
        # The blob appears higher up in B than in A because B starts lower, so the feature
        # row in B is less than in A. The offset (distance from top of A to top of B) is
        # therefore the negation of this difference.
        match_y_in_b = max_loc[1]
        displacement = (y0 + cy_a) - (match_y_in_b + cy_b)
        y_displacements.append(displacement)
        matched_blobs_a.append((x0, y0, x1 - x0, y1 - y0))
        matched_blobs_b.append((max_loc[0], match_y_in_b, x1 - x0, y1 - y0))

    debug(f"  template match: {len(y_displacements)} blob(s) matched, displacements={y_displacements}")

    if mask_debug_folder is not None:
        def _make_overlay(img_bgr: cv2.Mat, binary: np.ndarray, boxes: list[tuple[int, int, int, int]], cy: int, cx: int) -> cv2.Mat:
            vis = img_bgr.copy()
            h, w = vis.shape[:2]
            green_layer = np.zeros_like(vis)
            green_layer[cy:h - cy, cx:w - cx][binary > 0] = (0, 255, 0)
            vis = cv2.addWeighted(vis, 0.7, green_layer, 0.3, 0)
            for bx, by, bw, bh in boxes:
                rx, ry = bx + cx, by + cy
                cv2.rectangle(vis, (rx, ry), (rx + bw, ry + bh), (0, 0, 255), 2)
            return vis

        vis_a = _make_overlay(img_a, binary_a, matched_blobs_a, cy_a, cx_a)
        vis_b = _make_overlay(img_b, binary_b, matched_blobs_b, cy_b, cx_b)

        h_a_vis, w_a_vis = vis_a.shape[:2]
        h_b_vis, w_b_vis = vis_b.shape[:2]
        combined_h = max(h_a_vis, h_b_vis)
        combined_w = w_a_vis + w_b_vis + 4
        combined = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
        combined[:h_a_vis, :w_a_vis] = vis_a
        combined[:h_b_vis, w_a_vis + 4:] = vis_b
        cv2.line(combined, (w_a_vis + 2, 0), (w_a_vis + 2, combined_h), (128, 128, 128), 2)
        cv2.imwrite(os.path.join(mask_debug_folder, f"pair_{pair_index:02d}_masks.png"), combined)

    if len(y_displacements) < 2:
        debug(f"  template match: too few matches, using nominal={nominal_offset}px")
        return nominal_offset

    refined = int(np.median(y_displacements))
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


def _stitch_images(
    images: list[tuple[cv2.Mat, str, int]],
    overlap_frac: float,
    mask_debug_folder: str | None = None,
) -> cv2.Mat | None:
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
            mask_debug_folder=mask_debug_folder,
            pair_index=i,
        )
        refined_offsets.append(refined)
        if mask_debug_folder is not None:
            _save_overlap_debug(img_a, img_b, refined, i, mask_debug_folder)

    debug(f"_stitch_images: refined offsets={refined_offsets}")

    total_h = h_a + sum(refined_offsets)
    canvas_w = max(img.shape[1] for img, _, _ in images)
    debug(f"_stitch_images: canvas=({total_h}h, {canvas_w}w)")
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
            debug(f"  pair {i}: offset={offset}px overlap={overlap_rows}px seam at canvas row {seam_canvas} (b row {seam_in_b})")
        else:
            b_start_canvas = cum_y + offset
            b_end_canvas = min(b_start_canvas + h_b, total_h)
            src_rows = b_end_canvas - b_start_canvas
            canvas[b_start_canvas:b_end_canvas, :w_b] = img_b[:src_rows, :]
            debug(f"  pair {i}: offset={offset}px no overlap, placing directly")

        cum_y += offset

    debug("_stitch_images: stitching complete")
    return canvas


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


def _text_mass_above_vs_below(
    binary: np.ndarray, baseline_index: int, axis: str
) -> tuple[float, float]:
    band = 5
    if axis == "horizontal":
        mask = np.ones_like(binary)
        mask[max(0, baseline_index - band):baseline_index + band + 1, :] = 0
        masked = binary & mask
        above = float(np.count_nonzero(masked[:baseline_index, :]))
        below = float(np.count_nonzero(masked[baseline_index:, :]))
    else:
        mask = np.ones_like(binary)
        mask[:, max(0, baseline_index - band):baseline_index + band + 1] = 0
        masked = binary & mask
        above = float(np.count_nonzero(masked[:, :baseline_index]))
        below = float(np.count_nonzero(masked[:, baseline_index:]))
    return above, below


def _detect_orientation(image: cv2.Mat) -> int:
    h, w = image.shape[:2]
    candidates = [0, 180] if w >= h else [90, 270]
    best_rotation = candidates[0]
    best_ratio = -1.0
    for rot in candidates:
        rotated = _rotate_image(image, rot)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        binary = _binarize(gray)
        axis, baseline_index = _find_bar_axis(binary)
        above, below = _text_mass_above_vs_below(binary, baseline_index, axis)
        total = above + below
        ratio = above / total if total > 0 else 0.0
        debug(f"    {rot}°: axis={axis} baseline={baseline_index}px | above={above:.0f} below={below:.0f} ratio={ratio:.3f}")
        if ratio > best_ratio:
            best_ratio = ratio
            best_rotation = rot
    debug(f"_detect_orientation: selected {best_rotation}° (ratio={best_ratio:.3f})")
    return best_rotation


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


def _extract_dpi(image: cv2.Mat, scale_mm: float, tick_min_length: int) -> float | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)
    axis, baseline_index = _find_bar_axis(binary)
    start, end = _find_tick_intersections(binary, axis, baseline_index, tick_min_length=tick_min_length)
    pixel_length = end - start
    if pixel_length <= 0:
        warning("_extract_dpi: could not measure a valid scalebar span")
        return None
    dpi = (pixel_length / scale_mm) * MM_PER_INCH
    debug(f"_extract_dpi: axis={axis} baseline={baseline_index}px span={pixel_length}px ({start}->{end})")
    debug(f"_extract_dpi: physical={scale_mm}mm px/mm={pixel_length / scale_mm:.4f} DPI={dpi:.2f}")
    return dpi


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

    if axis == "horizontal":
        pt1, pt2 = (start, baseline_index), (end, baseline_index)
    else:
        pt1, pt2 = (baseline_index, start), (baseline_index, end)
    cv2.line(vis, pt1, pt2, (0, 0, 255), 2)
    cv2.circle(vis, pt1, 6, (0, 255, 0), -1)
    cv2.circle(vis, pt2, 6, (0, 255, 0), -1)
    label_pt = (max(pt1[0] - 5, 5), max(pt1[1] - 14, 20))
    cv2.putText(vis, f"{dpi:.1f} DPI", label_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
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
    ) -> None:
        super().__init__(settings)
        self.input_folder = input_folder
        self._save_dpi = save_dpi

    def steps(self) -> Generator[None, None, None]:
        sm = self.settings.post_processing.stitch_and_measure

        # ----------------------------------------------------------------
        # Step 1: discover and load images
        # ----------------------------------------------------------------
        self._set_status("Collecting images", 0, 4)

        subfolders = [
            entry.path
            for entry in os.scandir(self.input_folder)
            if entry.is_dir()
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

        stitched = _stitch_images(images, sm.overlap_frac, mask_debug_folder=None)
        if stitched is None:
            raise RuntimeError(
                "Stitching failed — ensure the overlap value is correct and "
                "that Y positions in filenames reflect the correct scan order"
            )

        yield

        # ----------------------------------------------------------------
        # Step 3: crop and rotate
        # ----------------------------------------------------------------
        self._set_status("Correcting orientation", 2, 4)

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
        # Step 4: save, measure DPI, optionally write debug overlay
        # ----------------------------------------------------------------
        self._set_status("Measuring DPI", 3, 4)

        root_name = os.path.basename(self.input_folder)
        if self._save_dpi:
            from common.app_context import get_app_context
            ctx = get_app_context()
            extension = ctx.camera.underlying_camera.settings.fformat.value
            output_path = os.path.join(self.input_folder, f"{root_name}.{extension}")
        else:
            output_path = os.path.join(self.input_folder, "stitched.jpg")

        cv2.imwrite(output_path, stitched)
        debug(f"StitchAndMeasureRoutine: panorama saved to {output_path}")

        dpi = _extract_dpi(stitched, sm.scale_mm, sm.tick_min_length)
        if dpi is not None and self._save_dpi:
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
        metavar="PCT",
        help="Percentage of image height that consecutive images overlap (e.g. 70 for 70%%)",
    )
    parser.add_argument("--save-masks", action="store_true", help="Save per-pair debug images to a masks/ subfolder showing binary masks overlaid on source images with matched blob bounding boxes")
    parser.add_argument("--debug", action="store_true", help="Overlay detected points and lines on the debug image")
    args = parser.parse_args()

    DEBUG = args.debug

    settings = FieldWeaveSettings()
    settings.post_processing.stitch_and_measure.overlap_frac = args.overlap / 100.0

    mask_debug_folder: str | None = None
    if args.save_masks:
        mask_debug_folder = os.path.join(args.folder, "masks")
        os.makedirs(mask_debug_folder, exist_ok=True)

    if mask_debug_folder is not None:
        subfolders = [e.path for e in os.scandir(args.folder) if e.is_dir() and os.path.basename(e.path) != "masks"]
        if len(subfolders) != 1:
            print("Expected exactly one image subfolder.", file=sys.stderr)
            sys.exit(1)
        images = _collect_images(subfolders[0])
        sm = settings.post_processing.stitch_and_measure
        stitched = _stitch_images(images, sm.overlap_frac, mask_debug_folder=mask_debug_folder)
        if stitched is None:
            print("Stitching failed.", file=sys.stderr)
            sys.exit(1)
        output_path = os.path.join(args.folder, "stitched.jpg")
        cv2.imwrite(output_path, stitched)
        h, w = stitched.shape[:2]
        print(f"Saved:  {output_path}")
        print(f"Size:   {w}w x {h}h px")
        print(f"Masks:  {mask_debug_folder}")
    else:
        routine = StitchAndMeasureRoutine(
            settings=settings,
            input_folder=args.folder,
            save_dpi=False,
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