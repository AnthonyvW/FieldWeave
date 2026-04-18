"""
machine_vision_worker.py

Low-level worker object that performs computer-vision tasks on a dedicated
QThread.  It must never touch any GUI object directly; all results are
returned to the GUI thread through Qt signals.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from common.logger import debug, error
from machine_vision.camera_calibration import (
    CameraCalibration,
    build_calibration,
    compute_edge_map,
    rgb_to_gray,
)
from machine_vision.machine_vision_config import (
    FOCUS_METHOD_TENENGRAD,
    FOCUS_METHOD_LAPLACIAN,
    FocusMethod,
    TenengradSettings,
    LaplacianSettings,
)
from machine_vision.focus_detection import (
    FocusRegion,
    FocusScores,
    generate_focus_map,
    generate_focus_map_laplacian,
    normalize_score_map,
    apply_focus_overlay,
    compute_focus_scores,
)
from machine_vision.calibration_bar_detection import (
    AxisState,
    BarDetectionResult,
    process_frame,
)
from machine_vision.red_mark_detection import (
    detect_red_marks,
    smooth_value,
    line_orientation,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class FocusResult:
    """
    Result of a single focus analysis pass.

    All arrays are freshly allocated (not views into any shared buffer) and
    are safe to read from the GUI thread after the signal is delivered.
    """
    scores: FocusScores
    """Whole-image, center, and peak focus scores in [0, 1]."""

    heatmap_rgb: np.ndarray
    """
    Composited heatmap blended over the original frame, RGB888 order,
    shape (H, W, 3), dtype uint8.  Ready to wrap in QImage directly.
    """

    source_width: int
    source_height: int

    raw_score_max: float
    """
    Maximum value of the raw (un-normalised) score map for this frame.
    Use this to calibrate score_ceiling: focus sharply, note this value,
    then enter it as the ceiling for stable cross-frame normalisation.
    """

    method: FocusMethod
    """Which focus measure produced this result."""


@dataclass
class InspectCalibrationResult:
    """
    Result of a single calibration-bar inspection pass.

    All arrays are freshly allocated (not views into any shared buffer) and
    are safe to read from the GUI thread after the signal is delivered.
    """

    detection: BarDetectionResult
    """Raw detection result from the vision pipeline."""

    source_width: int
    source_height: int


@dataclass
class RedMarkDetectionResult:
    """
    Result of a single red-mark detection pass.

    All coordinate values are in full source-image pixel space (not
    downsampled), matching the frame dimensions in ``source_width`` /
    ``source_height``.  The overlay reprojects them into display space at
    paint time.
    """

    valid_centers: list[tuple[float, float]]
    """Centroids of accepted red-mark blobs, (x, y) in source pixels."""

    filtered_centers: list[tuple[float, float]]
    """Centroids of blobs that were rejected by area, aspect-ratio, or
    position filters, kept for diagnostic display."""

    valid_mask: np.ndarray
    """
    uint8 pixel mask of accepted blob footprints (255 = blob, 0 = background),
    shape (source_height, source_width).  Used by the overlay to paint actual
    blob pixels rather than just centroid markers.
    """

    filtered_mask: np.ndarray
    """uint8 pixel mask of rejected blob footprints, same shape as valid_mask."""

    mean_x: float | None
    """Mean X of valid centroid coordinates, or None when no valid blobs."""

    mean_y: float | None
    """Mean Y of valid centroid coordinates, or None when no valid blobs."""

    stabilized_x: float | None
    """
    EMA-smoothed mean X of valid mark centroids; used for the vertical
    reference line.  ``None`` when no valid marks are present.
    """

    stabilized_y: float | None
    """
    EMA-smoothed mean Y of valid mark centroids; used for the horizontal
    reference line.  ``None`` when no valid marks are present.
    """

    image_center_x: float | None
    """Horizontal midpoint of the source frame (source_width / 2)."""

    image_center_y: float | None
    """Vertical midpoint of the source frame (source_height / 2)."""

    line_orientation: str
    """``'vertical'`` or ``'horizontal'`` — which axis the marks indicate."""

    elapsed_ms: float
    """Wall-clock time taken for this detection pass in milliseconds."""

    source_width: int
    source_height: int


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class MachineVisionWorker(QObject):
    """
    Vision processing worker — lives on a dedicated QThread.

    Parameter attributes are written by the manager from the GUI thread
    between frames.  Plain Python scalar writes are atomic on CPython so
    no lock is required.
    """

    focus_result_ready = Signal(object)   # FocusResult
    analysis_error = Signal(str)
    calibration_ready = Signal(object)    # CameraCalibration
    calibration_error = Signal(str)
    inspect_calibration_result_ready = Signal(object)   # InspectCalibrationResult
    red_mark_detection_result_ready = Signal(object)    # RedMarkDetectionResult

    # Active method — controls which parameter block is used.
    focus_method: FocusMethod = FOCUS_METHOD_LAPLACIAN

    # Tenengrad parameters
    tenengrad_kernel_size: int = 3
    tenengrad_radius: float = 8.0
    tenengrad_threshold: float = 0.0
    tenengrad_half_resolution: bool = True
    tenengrad_overlay_alpha: float = 0.55
    tenengrad_score_ceiling: float = 15.0
    tenengrad_auto_ceiling: bool = False

    # Laplacian parameters
    laplacian_window_size: int = 15
    laplacian_radius: float = 8.0
    laplacian_threshold: float = 0.0
    laplacian_half_resolution: bool = True
    laplacian_overlay_alpha: float = 0.55
    laplacian_score_ceiling: float = 15.0
    laplacian_auto_ceiling: bool = False

    # Focus region (shared across methods; ignored when focus_region_enabled is False)
    focus_region_enabled: bool = False
    focus_region_left: float = 0.0
    focus_region_right: float = 0.0
    focus_region_top: float = 0.0
    focus_region_bottom: float = 0.0

    # Inspect-calibration parameters — preview mode (live overlay)
    inspect_calibration_preview_downsample: int | None = None
    inspect_calibration_preview_tick_min_length: int = 150

    # Inspect-calibration parameters — snap mode (full-image capture)
    inspect_calibration_snap_downsample: int | None = 2
    inspect_calibration_snap_tick_min_length: int = 200

    # Shared
    overlay_colormap: int = cv2.COLORMAP_JET

    # Red-mark detection parameters
    red_mark_scale: int = 4
    red_mark_open_kernel_size: int = 5
    red_mark_min_area: int = 500
    red_mark_max_aspect_ratio: float = 8.0
    red_mark_min_area_fraction: float = 0.1
    red_mark_hue_low: int = 160
    red_mark_hue_high: int = 10
    red_mark_sat_min: int = 100
    red_mark_val_min: int = 50
    red_mark_smoothing_alpha: float = 0.25
    red_mark_deadband_px: float = 2.0
    red_mark_max_step_px: float = 30.0
    red_mark_jump_threshold_px: float = 50.0
    red_mark_side_cluster_fraction: float = 0.85
    red_mark_side_cluster_margin: float = 0.18

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._axis_state = AxisState()
        self._red_smoothed_x: float | None = None
        self._red_smoothed_y: float | None = None
        self._red_cluster_frames: int = 0

    @Slot(bytes, int, int, bool)
    def run_inspect_calibration(self, frame_bytes: bytes, width: int, height: int, snap: bool = False) -> None:
        """
        Run a calibration-bar inspection pass on the given RGB888 frame.

        Detects the bar's orientation and tick marks, then emits
        ``inspect_calibration_result_ready`` with an ``InspectCalibrationResult``.
        The ``AxisState`` is preserved across calls so the confirmed-axis
        hysteresis works correctly during live preview.

        Pass ``snap=True`` to use the snap-mode parameters (2x downsampling,
        tick_min_length 200) instead of the preview-mode parameters (no
        downsampling, tick_min_length 150).

        frame_bytes must be a *copy* of the raw RGB888 data
        (stride == width * 3).
        """
        try:
            arr = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3)).copy()
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            if snap:
                downsample = self.inspect_calibration_snap_downsample
                tick_min_length = self.inspect_calibration_snap_tick_min_length
            else:
                downsample = self.inspect_calibration_preview_downsample
                tick_min_length = self.inspect_calibration_preview_tick_min_length

            detection = process_frame(
                bgr,
                self._axis_state,
                downsample=downsample,
                tick_min_length=tick_min_length,
            )

            result = InspectCalibrationResult(
                detection=detection,
                source_width=width,
                source_height=height,
            )
            self.inspect_calibration_result_ready.emit(result)

        except Exception:
            msg = traceback.format_exc()
            error(f"MachineVisionWorker: inspect calibration failed:\n{msg}")
            self.analysis_error.emit(msg)

    def reset_inspect_calibration_state(self) -> None:
        """Reset the axis hysteresis state. Call when starting a new inspection session."""
        self._axis_state = AxisState()

    def reset_red_mark_state(self) -> None:
        """Reset the smoothing and hysteresis state for red-mark detection."""
        self._red_smoothed_x = None
        self._red_smoothed_y = None
        self._red_cluster_frames = 0

    @Slot(bytes, int, int)
    def run_red_mark_detection(self, frame_bytes: bytes, width: int, height: int) -> None:
        """
        Detect red registration marks in the given RGB888 frame.

        Emits ``red_mark_detection_result_ready`` with a
        ``RedMarkDetectionResult`` containing accepted and rejected centroids,
        the stabilized reference position, and the active line orientation.

        Smoothing state (EMA, hysteresis) is preserved across calls so the
        overlay remains stable during live preview.  Call
        ``reset_red_mark_state`` when starting a new detection session.

        frame_bytes must be a *copy* of the raw RGB888 data
        (stride == width * 3).
        """
        try:
            t0 = time.perf_counter()

            arr = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3)).copy()

            valid, filtered, valid_mask, filtered_mask, mean_x, mean_y = detect_red_marks(
                arr,
                open_kernel_size=self.red_mark_open_kernel_size,
                min_area=self.red_mark_min_area,
                scale=self.red_mark_scale,
                max_aspect_ratio=self.red_mark_max_aspect_ratio,
                min_area_fraction=self.red_mark_min_area_fraction,
                hue_low=self.red_mark_hue_low,
                hue_high=self.red_mark_hue_high,
                sat_min=self.red_mark_sat_min,
                val_min=self.red_mark_val_min,
            )

            self._red_smoothed_x = smooth_value(
                self._red_smoothed_x, mean_x,
                self.red_mark_smoothing_alpha,
                self.red_mark_deadband_px,
                self.red_mark_max_step_px,
                self.red_mark_jump_threshold_px,
            )
            self._red_smoothed_y = smooth_value(
                self._red_smoothed_y, mean_y,
                self.red_mark_smoothing_alpha,
                self.red_mark_deadband_px,
                self.red_mark_max_step_px,
                self.red_mark_jump_threshold_px,
            )

            orientation, self._red_cluster_frames = line_orientation(
                valid,
                width,
                self.red_mark_side_cluster_fraction,
                self.red_mark_side_cluster_margin,
                self._red_cluster_frames,
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            result = RedMarkDetectionResult(
                valid_centers=valid,
                filtered_centers=filtered,
                valid_mask=valid_mask,
                filtered_mask=filtered_mask,
                mean_x=mean_x,
                mean_y=mean_y,
                stabilized_x=self._red_smoothed_x,
                stabilized_y=self._red_smoothed_y,
                image_center_x=float(width) / 2.0,
                image_center_y=float(height) / 2.0,
                line_orientation=orientation,
                elapsed_ms=elapsed_ms,
                source_width=width,
                source_height=height,
            )
            self.red_mark_detection_result_ready.emit(result)

        except Exception:
            msg = traceback.format_exc()
            error(f"MachineVisionWorker: red mark detection failed:\n{msg}")
            self.analysis_error.emit(msg)

    @Slot(bytes, int, int)
    def run_focus_analysis(self, frame_bytes: bytes, width: int, height: int) -> None:
        """
        Run a focus analysis pass using the currently configured method.

        frame_bytes must be a *copy* of the raw RGB888 data
        (stride == width * 3).
        """
        try:
            arr = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3)).copy()
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            method = self.focus_method
            colormap = self.overlay_colormap

            focus_region: FocusRegion | None = (
                FocusRegion(
                    left=self.focus_region_left / 100.0,
                    right=self.focus_region_right / 100.0,
                    top=self.focus_region_top / 100.0,
                    bottom=self.focus_region_bottom / 100.0,
                )
                if self.focus_region_enabled
                else None
            )

            if method == FOCUS_METHOD_TENENGRAD:
                raw_map, ceiling, alpha = self._run_tenengrad(bgr, focus_region)
            else:
                raw_map, ceiling, alpha = self._run_laplacian(bgr, focus_region)

            raw_score_max = float(raw_map.max())

            score_map = normalize_score_map(
                raw_map,
                ceiling=ceiling if ceiling is not None else None,
            )

            scores = compute_focus_scores(score_map, focus_region=focus_region)

            overlay_bgr = apply_focus_overlay(bgr, score_map, alpha=alpha, colormap=colormap)
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

            result = FocusResult(
                scores=scores,
                heatmap_rgb=overlay_rgb,
                source_width=width,
                source_height=height,
                raw_score_max=raw_score_max,
                method=method,
            )
            self.focus_result_ready.emit(result)

        except Exception:
            msg = traceback.format_exc()
            error(f"MachineVisionWorker: focus analysis failed:\n{msg}")
            self.analysis_error.emit(msg)

    @Slot(bytes, int, int, bytes, int, int, bytes, int, int, int, int, int, int, int)
    def run_calibration_build(
        self,
        base_bytes: bytes,
        base_width: int,
        base_height: int,
        x_bytes: bytes,
        x_width: int,
        x_height: int,
        y_bytes: bytes,
        y_width: int,
        y_height: int,
        move_x_ticks: int,
        move_y_ticks: int,
        ref_x: int,
        ref_y: int,
        ref_z: int,
    ) -> None:
        """
        Build a ``CameraCalibration`` from three RGB888 frame buffers.

        Each frame is an RGB888 byte string with stride == width * 3, matching
        the format passed to ``run_focus_analysis``.  The three frames must
        have been captured at the base position, after a +X move, and after a
        +Y move respectively (with the stage returned to base between the X
        and Y captures).

        Emits ``calibration_ready(CameraCalibration)`` on success or
        ``calibration_error(str)`` on failure.  All frames are converted to
        greyscale edge maps here on the worker thread so the GUI thread is
        never blocked.
        """
        try:
            def _to_edge(frame_bytes: bytes, w: int, h: int) -> np.ndarray:
                arr = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w, 3))
                gray = rgb_to_gray(arr)
                return compute_edge_map(gray)

            edges_base = _to_edge(base_bytes, base_width, base_height)
            edges_x    = _to_edge(x_bytes,    x_width,    x_height)
            edges_y    = _to_edge(y_bytes,    y_width,    y_height)

            calibration = build_calibration(
                edges_base=edges_base,
                edges_x=edges_x,
                edges_y=edges_y,
                move_x_ticks=move_x_ticks,
                move_y_ticks=move_y_ticks,
                ref_x=ref_x,
                ref_y=ref_y,
                ref_z=ref_z,
                image_width=base_width,
                image_height=base_height,
            )
            self.calibration_ready.emit(calibration)

        except Exception:
            msg = traceback.format_exc()
            error(f"MachineVisionWorker: calibration build failed:\n{msg}")
            self.calibration_error.emit(msg)

    # ------------------------------------------------------------------
    # Private per-method helpers
    # ------------------------------------------------------------------

    def _run_tenengrad(
        self,
        bgr: np.ndarray,
        focus_region: FocusRegion | None,
    ) -> tuple[np.ndarray, float | None, float]:
        """Return (raw_map, ceiling_or_None, alpha)."""
        raw_map = generate_focus_map(
            bgr,
            kernel_size=self.tenengrad_kernel_size,
            radius=self.tenengrad_radius,
            threshold=self.tenengrad_threshold,
            half_resolution=self.tenengrad_half_resolution,
            box_blur=True,
            verbose=False,
            normalize=False,
            focus_region=focus_region,
        )
        ceiling = None if self.tenengrad_auto_ceiling else self.tenengrad_score_ceiling
        return raw_map, ceiling, self.tenengrad_overlay_alpha

    def _run_laplacian(
        self,
        bgr: np.ndarray,
        focus_region: FocusRegion | None,
    ) -> tuple[np.ndarray, float | None, float]:
        """Return (raw_map, ceiling_or_None, alpha)."""
        raw_map = generate_focus_map_laplacian(
            bgr,
            window_size=self.laplacian_window_size,
            radius=self.laplacian_radius,
            threshold=self.laplacian_threshold,
            half_resolution=self.laplacian_half_resolution,
            box_blur=True,
            verbose=False,
            normalize=False,
            focus_region=focus_region,
        )
        ceiling = None if self.laplacian_auto_ceiling else self.laplacian_score_ceiling
        return raw_map, ceiling, self.laplacian_overlay_alpha