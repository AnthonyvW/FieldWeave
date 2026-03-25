"""
machine_vision_manager.py

Manages the machine-vision worker thread and exposes a clean, GUI-safe API
for requesting analysis passes.

Architecture
------------
  MachineVisionManager  -- QObject (lives on the GUI thread).
                           Owns a QThread + MachineVisionWorker.
                           Owns a MachineVisionSettingsManager and the
                           current MachineVisionSettings.
                           Provides request_*() methods that fire-and-forget;
                           results are delivered back as typed signals.

Thread model
------------
  * MachineVisionManager itself lives on the GUI thread.
  * MachineVisionWorker is moved to _thread via moveToThread().
  * request_focus_analysis() emits an internal *queued* signal which
    cross-thread delivers the call to the worker.
  * The worker emits focus_result_ready back; because the worker lives on
    the worker thread the delivery is again queued → arrives on GUI thread.
  * A _busy flag prevents request pile-up: if the worker is still processing
    the previous frame, new requests are dropped silently.  Callers should
    connect to focus_result_ready and re-request if they need continuous
    updates (e.g. from a timer or on every N-th preview frame).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from common.logger import debug, info, error, warning
from .machine_vision_worker import MachineVisionWorker, FocusResult
from .machine_vision_config import (
    MachineVisionSettings,
    MachineVisionSettingsManager,
    FocusDetectionSettings,
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
        Emitted after settings are applied (live or saved), so UI widgets
        can refresh their displayed values.
    """

    focus_result_ready = Signal(object)   # FocusResult
    analysis_error = Signal(str)
    settings_changed = Signal()

    # Internal signal used to push work across the thread boundary.
    _request_focus = Signal(bytes, int, int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        self._busy: bool = False

        # Load persistent settings before the worker is configured.
        self._settings_manager = MachineVisionSettingsManager()
        self._settings: MachineVisionSettings = self._load_settings()

        # Create worker and thread.
        self._thread = QThread(self)
        self._thread.setObjectName("MachineVisionThread")

        self._worker = MachineVisionWorker()
        self._worker.moveToThread(self._thread)

        # Wire internal dispatch signal → worker slot (queued across thread).
        self._request_focus.connect(self._worker.run_focus_analysis)

        # Wire worker result signals → our public signals (queued back to GUI).
        self._worker.focus_result_ready.connect(self._on_focus_result)
        self._worker.analysis_error.connect(self._on_analysis_error)

        self._thread.start()

        # Apply loaded settings to the worker now that the thread is running.
        self._apply_settings(self._settings)
        info("MachineVisionManager: worker thread started")

    # ------------------------------------------------------------------
    # Settings API
    # ------------------------------------------------------------------

    @property
    def settings(self) -> MachineVisionSettings:
        """The current machine-vision settings (live, not necessarily saved)."""
        return self._settings

    def apply_settings(self, settings: MachineVisionSettings) -> None:
        """
        Apply *settings* to the worker immediately without saving to disk.

        The UI calls this to give live preview of parameter changes.
        Call save_settings() afterwards to persist them.
        """
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
    # Convenience pass-through properties
    # ------------------------------------------------------------------

    @property
    def laplacian_window_size(self) -> int:
        return self._settings.focus.window_size

    @laplacian_window_size.setter
    def laplacian_window_size(self, value: int) -> None:
        s = self._copy_settings()
        s.focus.window_size = value
        self.apply_settings(s)

    @property
    def laplacian_radius(self) -> float:
        return self._settings.focus.radius

    @laplacian_radius.setter
    def laplacian_radius(self, value: float) -> None:
        s = self._copy_settings()
        s.focus.radius = value
        self.apply_settings(s)

    @property
    def laplacian_threshold(self) -> float:
        return self._settings.focus.threshold

    @laplacian_threshold.setter
    def laplacian_threshold(self, value: float) -> None:
        s = self._copy_settings()
        s.focus.threshold = value
        self.apply_settings(s)

    @property
    def half_resolution(self) -> bool:
        return self._settings.focus.half_resolution

    @half_resolution.setter
    def half_resolution(self, value: bool) -> None:
        s = self._copy_settings()
        s.focus.half_resolution = value
        self.apply_settings(s)

    @property
    def overlay_alpha(self) -> float:
        return self._settings.focus.overlay_alpha

    @overlay_alpha.setter
    def overlay_alpha(self, value: float) -> None:
        s = self._copy_settings()
        s.focus.overlay_alpha = value
        self.apply_settings(s)

    @property
    def score_ceiling(self) -> float:
        """
        Fixed normalisation ceiling for the focus score map.

        Set to a value > 0 to lock the heatmap scale across frames.  Set to
        0 to revert to per-frame normalisation.
        """
        return self._settings.focus.score_ceiling

    @score_ceiling.setter
    def score_ceiling(self, value: float) -> None:
        s = self._copy_settings()
        s.focus.score_ceiling = value
        self.apply_settings(s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_focus_analysis(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
    ) -> bool:
        """
        Submit a focus analysis request.

        The frame data is copied immediately before this method returns, so
        the caller does not need to keep the array alive.

        Parameters
        ----------
        frame:
            RGB888 numpy array of shape (H, W, 3) or a flat uint8 array.
            Must correspond to *width* × *height* pixels.
        width, height:
            Frame dimensions in pixels.

        Returns
        -------
        bool
            True  — request accepted and queued to the worker.
            False — worker is still busy with the previous frame; request
                    dropped.  The caller can retry on the next frame.
        """
        if self._busy:
            debug("MachineVisionManager: worker busy, dropping focus request")
            return False

        # Copy the raw pixel bytes now, on the GUI thread, before the camera
        # manager may overwrite the live buffer with the next frame.
        frame_bytes = bytes(frame.data if isinstance(frame, np.ndarray) else frame)

        self._busy = True
        self._request_focus.emit(frame_bytes, width, height)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_settings(self) -> MachineVisionSettings:
        try:
            settings = self._settings_manager.load()
            info("MachineVisionManager: settings loaded")
            return settings
        except Exception as exc:
            error(f"MachineVisionManager: failed to load settings — {exc}; using defaults")
            return MachineVisionSettings()

    def _apply_settings(self, settings: MachineVisionSettings) -> None:
        """Push settings values onto the worker attributes."""
        f = settings.focus
        self._worker.laplacian_window_size = f.window_size
        self._worker.laplacian_radius = f.radius
        self._worker.laplacian_threshold = f.threshold
        self._worker.half_resolution = f.half_resolution
        self._worker.overlay_alpha = f.overlay_alpha
        self._worker.score_ceiling = f.score_ceiling

    def _copy_settings(self) -> MachineVisionSettings:
        """Return a deep copy of the current settings for mutation."""
        f = self._settings.focus
        return MachineVisionSettings(
            focus=FocusDetectionSettings(
                window_size=f.window_size,
                radius=f.radius,
                threshold=f.threshold,
                half_resolution=f.half_resolution,
                overlay_alpha=f.overlay_alpha,
                score_ceiling=f.score_ceiling,
            )
        )

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_focus_result(self, result: FocusResult) -> None:
        """Receive result from worker; clear busy flag; re-emit."""
        self._busy = False
        self.focus_result_ready.emit(result)

    @Slot(str)
    def _on_analysis_error(self, msg: str) -> None:
        self._busy = False
        error(f"MachineVisionManager: worker error: {msg}")
        self.analysis_error.emit(msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """
        Stop the worker thread gracefully.  Safe to call more than once.
        Blocks until the thread exits (with a 3-second timeout).
        """
        if not self._thread.isRunning():
            return

        info("MachineVisionManager: shutting down worker thread...")
        self._thread.quit()
        if not self._thread.wait(3000):
            warning("MachineVisionManager: worker thread did not exit in time; terminating")
            self._thread.terminate()
            self._thread.wait()
        info("MachineVisionManager: worker thread stopped")