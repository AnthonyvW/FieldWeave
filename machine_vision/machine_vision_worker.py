"""
machine_vision_worker.py

Low-level worker object that performs computer-vision tasks on a dedicated
QThread.  It must never touch any GUI object directly; all results are
returned to the GUI thread through Qt signals.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from common.logger import debug, error
from machine_vision.machine_vision_config import (
    FOCUS_METHOD_TENENGRAD,
    FOCUS_METHOD_LAPLACIAN,
    FocusMethod,
    TenengradSettings,
    LaplacianSettings,
)
from machine_vision.focus_detection import (
    FocusScores,
    generate_focus_map,
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

    # Shared
    overlay_colormap: int = cv2.COLORMAP_JET

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

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

            if method == FOCUS_METHOD_TENENGRAD:
                raw_map, ceiling, alpha = self._run_tenengrad(bgr)
            else:
                raw_map, ceiling, alpha = self._run_laplacian(bgr)

            raw_score_max = float(raw_map.max())

            score_map = normalize_score_map(
                raw_map,
                ceiling=ceiling if ceiling is not None else None,
            )

            scores = compute_focus_scores(score_map)

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

    # ------------------------------------------------------------------
    # Private per-method helpers
    # ------------------------------------------------------------------

    def _run_tenengrad(self, bgr: np.ndarray) -> tuple[np.ndarray, float | None, float]:
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
        )
        ceiling = None if self.tenengrad_auto_ceiling else self.tenengrad_score_ceiling
        return raw_map, ceiling, self.tenengrad_overlay_alpha

    def _run_laplacian(self, bgr: np.ndarray) -> tuple[np.ndarray, float | None, float]:
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
        )
        ceiling = None if self.laplacian_auto_ceiling else self.laplacian_score_ceiling
        return raw_map, ceiling, self.laplacian_overlay_alpha