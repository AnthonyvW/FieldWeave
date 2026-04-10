"""
machine_vision_manager.py

GUI-thread owner of the machine-vision worker thread.
Owns persistent settings and exposes a clean API for requesting analysis.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from common.logger import info, error, warning
from machine_vision.machine_vision_worker import MachineVisionWorker, FocusResult
from machine_vision.machine_vision_config import (
    FocusDetectionSettings,
    FocusRegionSettings,
    LaplacianSettings,
    MachineVisionSettings,
    MachineVisionSettingsManager,
    TenengradSettings,
)


@dataclass
class _PendingRequest:
    """A queued focus analysis request paired with its future."""
    frame_bytes: bytes
    width: int
    height: int
    future: Future[FocusResult]


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

    Guaranteed-result API
    ---------------------
    Use ``request_focus_analysis_async`` to get a ``Future[FocusResult]``
    that is always resolved — either with the result or with an exception.

    Latest-frame-wins policy
    ------------------------
    At most one request sits pending behind the in-flight job.  When a new
    request arrives while the worker is busy, any previously waiting request
    is cancelled and replaced with the newest frame.  This keeps the overlay
    current for live preview: stale queued frames are never processed.

    Fire-and-forget API
    -------------------
    ``request_focus_analysis`` is a convenience wrapper that wires the future
    to the ``focus_result_ready`` / ``analysis_error`` signals as before.
    """

    focus_result_ready = Signal(object)   # FocusResult
    analysis_error = Signal(str)
    settings_changed = Signal()

    _request_focus = Signal(bytes, int, int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        # At most one request waits behind the in-flight job (latest-frame-wins).
        # Both fields are only ever touched from the GUI thread.
        self._pending: _PendingRequest | None = None
        self._current_pending: _PendingRequest | None = None
        self._busy: bool = False

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

    def request_focus_analysis_async(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
    ) -> Future[FocusResult]:
        """
        Submit a focus analysis request and return a ``Future[FocusResult]``.

        The future is always resolved: either set to the ``FocusResult`` on
        success, or set to an exception if the worker raises.

        If the worker is currently busy, any previously queued (but not yet
        dispatched) request is cancelled and replaced by this one.  Only the
        most recent frame waits behind the in-flight job, so the overlay
        never falls behind during live preview.

        The frame is copied immediately so the camera buffer may be reused
        before the future resolves.

        This method is safe to call from the GUI thread only.
        """
        future: Future[FocusResult] = Future()
        pending = _PendingRequest(
            frame_bytes=bytes(frame),
            width=width,
            height=height,
            future=future,
        )

        # Discard the previous waiting request (if any) before replacing it.
        if self._pending is not None:
            self._pending.future.cancel()

        self._pending = pending
        self._try_dispatch()
        return future

    def request_focus_analysis(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
    ) -> bool:
        """
        Submit a focus analysis request.

        Always returns ``True``; the request is queued if the worker is busy.
        Results and errors arrive via the ``focus_result_ready`` and
        ``analysis_error`` signals as before.

        The frame is copied immediately so the camera buffer may be reused.
        """
        future = self.request_focus_analysis_async(frame, width, height)
        future.add_done_callback(self._signal_from_future)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_dispatch(self) -> None:
        """Dispatch the pending request if the worker is free."""
        if self._busy or self._pending is None:
            return
        pending = self._pending
        self._pending = None
        self._busy = True
        self._current_pending = pending
        self._request_focus.emit(pending.frame_bytes, pending.width, pending.height)

    def _signal_from_future(self, future: Future[FocusResult]) -> None:
        """Done-callback that fans out a resolved future to the public signals."""
        if future.cancelled():
            return
        exc = future.exception()
        if exc is not None:
            self.analysis_error.emit(str(exc))
        else:
            self.focus_result_ready.emit(future.result())

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

        fr = f.focus_region
        w.focus_region_enabled = fr.enabled
        w.focus_region_left = fr.left
        w.focus_region_right = fr.right
        w.focus_region_top = fr.top
        w.focus_region_bottom = fr.bottom

    def _copy_settings(self) -> MachineVisionSettings:
        """Return a deep copy of the current settings for mutation."""
        f = self._settings.focus
        t, lap = f.tenengrad, f.laplacian
        fr = f.focus_region
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
                focus_region=FocusRegionSettings(
                    enabled=fr.enabled,
                    left=fr.left,
                    right=fr.right,
                    top=fr.top,
                    bottom=fr.bottom,
                ),
            )
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_focus_result(self, result: FocusResult) -> None:
        pending = self._current_pending
        self._busy = False
        pending.future.set_result(result)
        self._try_dispatch()

    @Slot(str)
    def _on_analysis_error(self, msg: str) -> None:
        pending = self._current_pending
        self._busy = False
        error(f"MachineVisionManager: worker error: {msg}")
        pending.future.set_exception(RuntimeError(msg))
        self._try_dispatch()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if not self._thread.isRunning():
            return
        info("MachineVisionManager: shutting down worker thread...")

        # Cancel any waiting request that will never be dispatched.
        if self._pending is not None:
            self._pending.future.cancel()
            self._pending = None

        self._thread.quit()
        if not self._thread.wait(3000):
            warning("MachineVisionManager: worker thread did not exit in time; terminating")
            self._thread.terminate()
            self._thread.wait()
        info("MachineVisionManager: worker thread stopped")