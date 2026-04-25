"""
post_processing_routines.py

All concrete post-processing routines live here.  Adding a new routine means
adding a new subclass of :class:`PostProcessingRoutine` in this file — no
other files need to change.

Current routines
----------------
- :class:`StitchAndMeasureRoutine` — stitches a folder of Y-ordered JPEGs into
  a panorama, corrects orientation, crops black borders, saves the result, then
  measures DPI from the embedded scale bar.
"""

from __future__ import annotations

import os
import re
from typing import Generator

import cv2
import numpy as np

from common.app_context import get_app_context
from common.logger import debug, warning
from common.fieldweaveConfig import FieldWeaveSettings
from post_processing.routines.post_processing_routine import PostProcessingRoutine


MM_PER_INCH = 25.4


# ---------------------------------------------------------------------------
# StitchAndMeasureRoutine — internal helpers
# ---------------------------------------------------------------------------

def _parse_y_position(filename: str) -> int:
    match = re.search(r'_y(\d+)', filename)
    return int(match.group(1)) if match else 0


def _collect_images(image_folder: str) -> list[tuple[cv2.Mat, str, int]]:
    valid_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
    jpeg_files = sorted([
        f for f in os.listdir(image_folder)
        if os.path.splitext(f)[1] in valid_extensions
    ])
    if not jpeg_files:
        raise ValueError(f"No JPEG images found in {image_folder}")

    entries: list[tuple[int, str]] = [
        (_parse_y_position(f), f) for f in jpeg_files
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


def _correlation_profile(a: cv2.Mat, b: cv2.Mat, min_overlap_frac: float = 0.05) -> np.ndarray:
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    h_a, h_b = gray_a.shape[0], gray_b.shape[0]
    template = gray_b[:h_b // 2, :]
    result = cv2.matchTemplate(gray_a, template, cv2.TM_CCOEFF_NORMED)
    profile = result.max(axis=1)
    min_y = int(h_a * min_overlap_frac)
    profile[:min_y] = -1.0
    return profile


def _first_peak(profile: np.ndarray, threshold_frac: float = 0.85) -> int | None:
    global_max = float(profile.max())
    if global_max <= 0:
        return None
    threshold = threshold_frac * global_max
    kernel = np.ones(5) / 5
    smoothed = np.convolve(profile, kernel, mode='same')
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] >= threshold and smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            return i
    return None


def _find_offset_first_peak(
    images: list[tuple[cv2.Mat, str, int]],
    threshold_frac: float = 0.85,
    min_overlap_frac: float = 0.05,
) -> int | None:
    offsets: list[int] = []
    for i in range(len(images) - 1):
        img_a, _, _ = images[i]
        img_b, _, _ = images[i + 1]
        profile = _correlation_profile(img_a, img_b, min_overlap_frac)
        peak = _first_peak(profile, threshold_frac)
        if peak is not None:
            global_max_val = float(profile.max())
            peak_val = float(profile[peak])
            debug(f"  Pair {i:2d}: first_peak={peak}px  peak_conf={peak_val:.3f}  global_max_conf={global_max_val:.3f}")
            offsets.append(peak)
        else:
            debug(f"  Pair {i:2d}: no peak found")
    if not offsets:
        return None
    return int(round(float(np.median(offsets))))


def _calibrate_offset(images: list[tuple[cv2.Mat, str, int]]) -> int | None:
    offset = _find_offset_first_peak(images)
    if offset is None:
        warning("_calibrate_offset: no peaks found in any pair")
        return None
    debug(f"_calibrate_offset: step offset={offset}px")
    return offset


def _place_image(
    canvas: cv2.Mat, image: cv2.Mat, y_offset: int, x_offset: int, prev_end_y: int
) -> None:
    h_i, w_i = image.shape[:2]
    seam_y = (y_offset + prev_end_y) // 2
    src_seam = seam_y - y_offset
    x_end = x_offset + w_i
    canvas_w = canvas.shape[1]
    src_x_end = min(w_i, canvas_w - x_offset)
    if x_offset >= 0:
        canvas[seam_y:y_offset + h_i, x_offset:x_offset + src_x_end] = image[src_seam:, :src_x_end]
    else:
        src_x_start = -x_offset
        canvas[seam_y:y_offset + h_i, :x_end] = image[src_seam:, src_x_start:src_x_start + x_end]


def _stitch_images(images: list[tuple[cv2.Mat, str, int]]) -> cv2.Mat | None:
    debug(f"_stitch_images: stitching {len(images)} images sequentially by Y position")

    offset_px = _calibrate_offset(images)
    if offset_px is None:
        warning("_stitch_images: failed to calibrate step offset")
        return None

    h_a, w_a = images[0][0].shape[:2]
    pair_offsets: list[tuple[int, int]] = [(offset_px, 0)] * (len(images) - 1)

    y_off, x_off = pair_offsets[0]
    h_b, w_b = images[1][0].shape[:2]
    cum_x = x_off
    canvas_h = max(y_off + h_b, h_a)
    canvas_w = max(w_a, cum_x + w_b) if cum_x >= 0 else max(w_a, w_b - cum_x)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h_a, :w_a] = images[0][0]
    _place_image(canvas, images[1][0], y_off, cum_x, h_a)

    cum_y = y_off
    prev_end_y = y_off + h_b

    for i in range(2, len(images)):
        img, label, _ = images[i]
        y_off, x_off = pair_offsets[i - 1]
        cum_y += y_off
        cum_x += x_off
        h_i, w_i = img.shape[:2]
        new_h = max(cum_y + h_i, canvas.shape[0])
        new_w = max(cum_x + w_i, canvas.shape[1]) if cum_x >= 0 else max(canvas.shape[1], w_i - cum_x)
        if new_h > canvas.shape[0] or new_w > canvas.shape[1]:
            expanded = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            expanded[:canvas.shape[0], :canvas.shape[1]] = canvas
            canvas = expanded
        _place_image(canvas, img, cum_y, cum_x, prev_end_y)
        prev_end_y = cum_y + h_i

    debug("_stitch_images: stitching successful")
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


def _longest_run(row: np.ndarray) -> int:
    max_run = cur_run = 0
    for px in row:
        if px > 0:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


def _find_bar_axis(binary: np.ndarray) -> tuple[str, int]:
    row_best = max(_longest_run(binary[r]) for r in range(binary.shape[0]))
    col_best = max(_longest_run(binary[:, c]) for c in range(binary.shape[1]))
    if row_best >= col_best:
        best = max(range(binary.shape[0]), key=lambda r: _longest_run(binary[r]))
        return "horizontal", best
    best = max(range(binary.shape[1]), key=lambda c: _longest_run(binary[:, c]))
    return "vertical", best


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
    image: cv2.Mat, scale_mm: float, tick_min_length: int
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
    Stitch a folder of Y-ordered JPEG images into a panorama, then measure DPI.

    *input_folder* must contain exactly one subfolder of JPEG images whose
    filenames include ``_y<position>`` tags.  The output JPEG is written to
    *input_folder* using the folder's own name as the filename stem.

    On success, :attr:`~PostProcessingRoutine.result` is populated with
    ``success=True`` and the following keys in ``result.data``:

    - ``output_path`` (:class:`str`): absolute path to the saved panorama JPEG.
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

    def __init__(self, settings: FieldWeaveSettings, input_folder: str) -> None:
        super().__init__(settings)
        self.input_folder = input_folder

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

        stitched = _stitch_images(images)
        if stitched is None:
            raise RuntimeError(
                "Stitching failed — ensure images have sufficient overlap (30-50%) "
                "and that Y positions in filenames reflect the correct scan order"
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
        output_path = os.path.join(self.input_folder, f"{root_name}.jpg")
        cv2.imwrite(output_path, stitched)
        debug(f"StitchAndMeasureRoutine: panorama saved to {output_path}")

        dpi = _extract_dpi(stitched, sm.scale_mm, sm.tick_min_length)

        if dpi is not None:
            ctx = get_app_context()
            if ctx is not None and ctx.machine_vision is not None:
                s = ctx.machine_vision._copy_settings()
                s.dpi = dpi
                ctx.machine_vision.apply_settings(s)
                debug(f"StitchAndMeasureRoutine: DPI {dpi:.2f} written to machine vision settings")

        debug_path: str | None = None
        if sm.save_debug_overlay and dpi is not None:
            overlay = _build_dpi_debug_overlay(stitched, sm.scale_mm, sm.tick_min_length)
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