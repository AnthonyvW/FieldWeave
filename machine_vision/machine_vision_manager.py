"""
machine_vision_manager.py

GUI-thread owner of the machine-vision worker thread.
Owns persistent settings and exposes a clean API for requesting analysis.
"""

from __future__ import annotations
import threading

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from common.logger import debug, info, error, warning
from .machine_vision_worker import MachineVisionWorker, FocusResult
from .machine_vision_config import (
    FocusDetectionSettings,
    FocusMethod,
    LaplacianSettings,
    MachineVisionSettings,
    MachineVisionSettingsManager,
    TenengradSettings,
    FOCUS_METHOD_TENENGRAD,
    FOCUS_METHOD_LAPLACIAN,
)


class MachineVisionManager(QObject):
    """
    GUI-thread owner of the machine-vision pipeline.

    Signals
    -------
    focus_result_ready(FocusResult):
        Delivered on the GUI thread after a successful focus pass.
    analysis_error(str):
        Delivered on the GUI thread when a worker exception occurs.
    settings_changed():
        Emitted after settings are applied so UI pages can refresh.
    """

    focus_result_ready = Signal(object)   # FocusResult
    analysis_error = Signal(str)
    settings_changed = Signal()

    _request_focus = Signal(bytes, int, int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        self._busy: bool = False
        self._busy_lock = threading.Lock()

        self._settings_manager = MachineVisionSettingsManager()
        self._settings: MachineVisionSettings = self._load_settings()

        self._thread = QThread(self)
        self._thread.setObjectName("MachineVisionThread")

        self._worker = MachineVisionWorker()
        self._worker.moveToThread(self._thread)

        self._request_focus.connect(self._worker.run_focus_analysis)
        self._worker.focus_result_ready.connect(self._on_focus_result)
        self._worker.analysis_error.connect(self._on_analysis_error)

        self._thread.start()
        self._apply_settings(self._settings)
        info("MachineVisionManager: worker thread started")

    # ------------------------------------------------------------------
    # Settings API
    # ------------------------------------------------------------------

    @property
    def settings(self) -> MachineVisionSettings:
        return self._settings

    def apply_settings(self, settings: MachineVisionSettings) -> None:
        """Apply *settings* to the worker immediately without saving to disk."""
        try:
            settings.validate()
        except ValueError as exc:
            error(f"MachineVisionManager: invalid settings — {exc}")
            return
        self._settings = settings
        self._apply_settings(settings)
        self.settings_changed.emit()

    def save_settings(self) -> None:
        """Persist the current settings to disk."""
        try:
            self._settings_manager.save(self._settings)
            info("MachineVisionManager: settings saved")
        except Exception as exc:
            error(f"MachineVisionManager: failed to save settings — {exc}")

    # ------------------------------------------------------------------
    # Analysis API
    # ------------------------------------------------------------------

    def request_focus_analysis(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
    ) -> bool:
        """
        Submit a focus analysis request.  Returns False if the worker is busy.

        The frame is copied immediately so the camera buffer may be reused.
        """
        with self._busy_lock:
            if self._busy:
                return False
            self._busy = True
        frame_bytes = bytes(frame)  # simpler and equivalent
        self._request_focus.emit(frame_bytes, width, height)
        return True


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_settings(self) -> MachineVisionSettings:
        try:
            s = self._settings_manager.load()
            info("MachineVisionManager: settings loaded")
            return s
        except Exception as exc:
            error(f"MachineVisionManager: failed to load settings — {exc}; using defaults")
            return MachineVisionSettings()

    def _apply_settings(self, settings: MachineVisionSettings) -> None:
        """Push all settings values onto the worker."""
        f = settings.focus
        w = self._worker

        w.focus_method = f.method

        t = f.tenengrad
        w.tenengrad_kernel_size = t.kernel_size
        w.tenengrad_radius = t.radius
        w.tenengrad_threshold = t.threshold
        w.tenengrad_half_resolution = t.half_resolution
        w.tenengrad_overlay_alpha = t.overlay_alpha
        w.tenengrad_score_ceiling = t.score_ceiling
        w.tenengrad_auto_ceiling = t.auto_ceiling

        lap = f.laplacian
        w.laplacian_window_size = lap.window_size
        w.laplacian_radius = lap.radius
        w.laplacian_threshold = lap.threshold
        w.laplacian_half_resolution = lap.half_resolution
        w.laplacian_overlay_alpha = lap.overlay_alpha
        w.laplacian_score_ceiling = lap.score_ceiling
        w.laplacian_auto_ceiling = lap.auto_ceiling

    def _copy_settings(self) -> MachineVisionSettings:
        """Return a deep copy of the current settings for mutation."""
        f = self._settings.focus
        t, lap = f.tenengrad, f.laplacian
        return MachineVisionSettings(
            focus=FocusDetectionSettings(
                method=f.method,
                tenengrad=TenengradSettings(
                    kernel_size=t.kernel_size,
                    radius=t.radius,
                    threshold=t.threshold,
                    half_resolution=t.half_resolution,
                    overlay_alpha=t.overlay_alpha,
                    score_ceiling=t.score_ceiling,
                    auto_ceiling=t.auto_ceiling,
                ),
                laplacian=LaplacianSettings(
                    window_size=lap.window_size,
                    radius=lap.radius,
                    threshold=lap.threshold,
                    half_resolution=lap.half_resolution,
                    overlay_alpha=lap.overlay_alpha,
                    score_ceiling=lap.score_ceiling,
                    auto_ceiling=lap.auto_ceiling,
                ),
            )
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_focus_result(self, result: FocusResult) -> None:
        with self._busy_lock:
            self._busy = False
        self.focus_result_ready.emit(result)


    @Slot(str)
    def _on_analysis_error(self, msg: str) -> None:
        with self._busy_lock:
            self._busy = False
        error(f"MachineVisionManager: worker error: {msg}")
        self.analysis_error.emit(msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if not self._thread.isRunning():
            return
        info("MachineVisionManager: shutting down worker thread...")
        self._thread.quit()
        if not self._thread.wait(3000):
            warning("MachineVisionManager: worker thread did not exit in time; terminating")
            self._thread.terminate()
            self._thread.wait()
        info("MachineVisionManager: worker thread stopped")