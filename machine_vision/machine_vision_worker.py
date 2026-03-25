"""
machine_vision_worker.py

Low-level worker object that performs computer-vision tasks on a dedicated
QThread.  It must never touch any GUI object directly; all results are
returned to the GUI thread through Qt signals.

Architecture
------------
  MachineVisionWorker  -- QObject that lives on a worker QThread.
                          Receives analysis requests via queued signals,
                          copies the pixel data before processing, then
                          emits typed result signals back to the GUI thread.

Consumers should not instantiate this class directly; use
MachineVisionManager (machine_vision_manager.py) instead.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from common.logger import debug, error, warning
from machine_vision.focus_detection import (
    FocusScores,
    generate_focus_map_laplacian,
    normalize_score_map,
    apply_focus_overlay,
    compute_focus_scores,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class FocusResult:
    """
    Result of a single Laplacian focus analysis pass.

    All arrays are freshly allocated (not views into any shared buffer) and
    are safe to read from the GUI thread after the signal is delivered.
    """
    scores: FocusScores
    """Whole-image, center, and peak focus scores in [0, 1]."""

    heatmap_rgb: np.ndarray
    """
    Composited heatmap blended over the original frame, in RGB888 order,
    shape (H, W, 3), dtype uint8.  Ready to wrap in QImage directly.
    """

    source_width: int
    """Width of the original frame that was analysed."""

    source_height: int
    """Height of the original frame that was analysed."""

    raw_score_max: float
    """
    Maximum value of the raw (un-normalised) score map for this frame.

    Use this to calibrate score_ceiling: observe the value while the sample
    is in sharp focus, then set score_ceiling to that value (or slightly
    above it) to get a stable, non-per-frame-normalised heatmap.
    """


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class MachineVisionWorker(QObject):
    """
    Vision processing worker — lives on a dedicated QThread.

    Signals (all emitted on the worker thread, delivered queued to GUI):
        focus_result_ready(FocusResult):
            Emitted after a successful focus analysis.
        analysis_error(str):
            Emitted if an unhandled exception occurs during analysis.
    """

    focus_result_ready = Signal(object)   # FocusResult
    analysis_error = Signal(str)

    # Parameters for the Laplacian focus pipeline.
    # Exposed as instance attributes so the manager can update them from the
    # GUI thread between frames (plain Python attribute writes are atomic on
    # CPython, so no lock is needed for these scalar values).
    laplacian_window_size: int = 15
    laplacian_radius: float = 8.0
    laplacian_threshold: float = 0.0
    half_resolution: bool = True
    overlay_alpha: float = 0.55
    overlay_colormap: int = cv2.COLORMAP_JET
    score_ceiling: float = 0.0
    """
    Fixed divisor used to normalise the raw score map to [0, 1].

    When > 0, the same ceiling is applied to every frame so the heatmap
    brightness is stable across frames — a fully in-focus frame at this
    ceiling value will saturate the colourmap.

    When 0 (default), each frame is normalised to its own maximum
    (per-frame normalisation).  This always fills the full colourmap range
    regardless of how sharp or soft the image is, which makes the heatmap
    look vivid but prevents meaningful cross-frame comparison.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    # ------------------------------------------------------------------
    # Slots (called via queued connection from the GUI thread)
    # ------------------------------------------------------------------

    @Slot(bytes, int, int)
    def run_focus_analysis(self, frame_bytes: bytes, width: int, height: int) -> None:
        """
        Perform a Laplacian focus analysis on *frame_bytes*.

        Parameters
        ----------
        frame_bytes:
            A *copy* of the raw RGB888 pixel data (stride == width * 3).
            The caller is responsible for ensuring this is a copy, not a
            view into a live camera buffer.
        width, height:
            Frame dimensions in pixels.
        """
        try:
            # Reconstruct a numpy array from the copied bytes.
            # np.frombuffer gives a read-only view of the bytes object; we
            # reshape without copying because the array never escapes this
            # method — the *result* arrays we emit are separate allocations.
            arr = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))

            # Convert RGB → BGR for OpenCV (camera_preview stores RGB888).
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            # Capture parameters atomically (reads of Python scalars are safe).
            window_size = self.laplacian_window_size
            radius = self.laplacian_radius
            threshold = self.laplacian_threshold
            half_res = self.half_resolution
            alpha = self.overlay_alpha
            colormap = self.overlay_colormap
            ceiling = self.score_ceiling

            # --- Focus map -------------------------------------------------
            # Always request the raw (un-normalised) map so we can apply a
            # fixed ceiling ourselves.  Per-frame normalisation is used only
            # when the user has not set a ceiling (ceiling == 0).
            raw_map = generate_focus_map_laplacian(
                bgr,
                window_size=window_size,
                radius=radius,
                threshold=threshold,
                half_resolution=half_res,
                box_blur=True,      # faster; visually equivalent for a heatmap
                verbose=False,
                normalize=False,
            )

            raw_score_max = float(raw_map.max())

            # Normalise: use the user-supplied ceiling when set, otherwise
            # fall back to per-frame normalisation.
            score_map = normalize_score_map(
                raw_map,
                ceiling=ceiling if ceiling > 0.0 else None,
            )

            scores = compute_focus_scores(score_map)

            # --- Composite overlay -----------------------------------------
            # apply_focus_overlay returns a BGR uint8 image.
            overlay_bgr = apply_focus_overlay(bgr, score_map, alpha=alpha, colormap=colormap)

            # Convert back to RGB so the GUI can wrap it directly in
            # QImage::Format_RGB888 without another conversion.
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

            result = FocusResult(
                scores=scores,
                heatmap_rgb=overlay_rgb,
                source_width=width,
                source_height=height,
                raw_score_max=raw_score_max,
            )

            debug(
                f"Focus analysis complete — whole={scores.whole:.3f} "
                f"center={scores.center:.3f} peak={scores.peak:.3f} "
                f"raw_max={raw_score_max:.1f}"
            )
            self.focus_result_ready.emit(result)

        except Exception:
            msg = traceback.format_exc()
            error(f"MachineVisionWorker: focus analysis failed:\n{msg}")
            self.analysis_error.emit(msg)