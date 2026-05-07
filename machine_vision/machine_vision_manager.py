"""
machine_vision_manager.py

GUI-thread owner of the machine-vision pipeline.
"""

from __future__ import annotations

import traceback
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from typing import Generic, TypeVar

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from common.logger import info, error, warning
from machine_vision.algorithms.camera_calibration import CameraCalibration, CalibrationBuild
from machine_vision.algorithms.focus_detection import FocusAnalysis, FocusResult
from machine_vision.algorithms.calibration_bar_detection import InspectCalibration, InspectCalibrationResult
from machine_vision.algorithms.red_mark_detection import RedMarkDetection, RedMarkDetectionResult
from machine_vision.algorithms.background_detection import BackgroundDetection, BackgroundDetectionResult
from machine_vision.machine_vision_config import (
    MachineVisionSettings,
    MachineVisionSettingsManager,
)

_T = TypeVar("_T")


# ---------------------------------------------------------------------------
# JobQueue — generic latest-frame-wins dispatch abstraction
# ---------------------------------------------------------------------------

class JobQueue(Generic[_T]):
    """
    Manages pending/current/busy state for a single worker job type.

    ``dispatch`` is called with the pending args tuple when the queue is free.
    ``result_signal`` and ``error_signal`` are connected at construction time
    so all wiring for a job type is colocated at the registration call site.

    Two submission modes are provided:

    ``submit`` — latest-frame-wins.  If a request is already waiting it is
    cancelled and replaced.  Use this for streaming preview frames where only
    the most recent result matters.

    ``submit_guaranteed`` — FIFO, never dropped.  Requests are appended to a
    separate deque and always executed in order.  Guaranteed jobs are drained
    before any pending droppable job is dispatched.  Use this when a result
    for a specific frame must be obtained (e.g. autofocus scoring).

    All methods must be called from the GUI thread only.
    """

    def __init__(
        self,
        dispatch: Callable[[tuple], None],
        result_signal: Signal,
        error_signal: Signal,
    ) -> None:
        self._dispatch = dispatch
        self._pending: tuple | None = None
        self._pending_future: Future[_T] | None = None
        self._guaranteed: deque[tuple[tuple, Future[_T]]] = deque()
        self._current_future: Future[_T] | None = None
        self._busy: bool = False
        result_signal.connect(self._handle_result)
        error_signal.connect(self._handle_error)

    def submit(self, args: tuple) -> Future[_T]:
        future: Future[_T] = Future()
        if self._pending_future is not None:
            self._pending_future.cancel()
        self._pending = args
        self._pending_future = future
        self._try_dispatch()
        return future

    def submit_guaranteed(self, args: tuple) -> Future[_T]:
        future: Future[_T] = Future()
        self._guaranteed.append((args, future))
        self._try_dispatch()
        return future

    def cancel_pending(self) -> None:
        if self._pending_future is not None:
            self._pending_future.cancel()
            self._pending_future = None
            self._pending = None

    def _handle_result(self, result: _T) -> None:
        if not self._busy:
            return
        future = self._current_future
        self._busy = False
        self._current_future = None
        future.set_result(result)
        self._try_dispatch()

    def _handle_error(self, msg: str) -> None:
        if not self._busy:
            return
        future = self._current_future
        self._busy = False
        self._current_future = None
        future.set_exception(RuntimeError(msg))
        self._try_dispatch()

    def _try_dispatch(self) -> None:
        if self._busy:
            return
        if self._guaranteed:
            args, future = self._guaranteed.popleft()
            self._current_future = future
            self._busy = True
            self._dispatch(args)
            return
        if self._pending is not None:
            args = self._pending
            self._current_future = self._pending_future
            self._pending = None
            self._pending_future = None
            self._busy = True
            self._dispatch(args)


# ---------------------------------------------------------------------------
# Worker — private QObject that lives on the vision thread
# ---------------------------------------------------------------------------

class _VisionWorker(QObject):
    """
    Thin QObject dispatcher that lives on the vision thread.

    Holds the signals and slots that must execute off the GUI thread.
    All algorithm logic is delegated to ``VisionAlgorithm`` instances, which
    read their parameters directly from the shared ``MachineVisionSettings``.
    """

    focus_result_ready = Signal(object)                 # FocusResult
    focus_error = Signal(str)
    calibration_ready = Signal(object)                  # CameraCalibration
    calibration_error = Signal(str)
    inspect_calibration_result_ready = Signal(object)   # InspectCalibrationResult
    inspect_calibration_error = Signal(str)
    red_mark_detection_result_ready = Signal(object)    # RedMarkDetectionResult
    red_mark_detection_error = Signal(str)
    background_detection_result_ready = Signal(object)  # BackgroundDetectionResult
    background_detection_error = Signal(str)

    def __init__(self, settings: MachineVisionSettings, save_settings: Callable[[], None], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._focus = FocusAnalysis(settings)
        self._calibration_build = CalibrationBuild(settings, save_settings)
        self._inspect_calibration = InspectCalibration(settings)
        self._red_mark_detection = RedMarkDetection(settings)
        self._background_detection = BackgroundDetection(settings)

    @Slot(bytes, int, int, bool)
    def run_focus_analysis(self, frame_bytes: bytes, width: int, height: int, use_region: bool = True) -> None:
        try:
            self.focus_result_ready.emit(self._focus.process(frame_bytes, width, height, use_region))
        except Exception:
            msg = traceback.format_exc()
            error(f"_VisionWorker: focus analysis failed:\n{msg}")
            self.focus_error.emit(msg)

    @Slot(bytes, int, int, bytes, int, int, bytes, int, int, int, int, int, int, int)
    def run_calibration_build(
        self,
        base_bytes: bytes, base_width: int, base_height: int,
        x_bytes: bytes,    x_width: int,    x_height: int,
        y_bytes: bytes,    y_width: int,    y_height: int,
        move_x_ticks: int, move_y_ticks: int,
        ref_x: int, ref_y: int, ref_z: int,
    ) -> None:
        try:
            self.calibration_ready.emit(self._calibration_build.process(
                base_bytes, base_width, base_height,
                x_bytes,    x_width,    x_height,
                y_bytes,    y_width,    y_height,
                move_x_ticks, move_y_ticks,
                ref_x, ref_y, ref_z,
            ))
        except Exception:
            msg = traceback.format_exc()
            error(f"_VisionWorker: calibration build failed:\n{msg}")
            self.calibration_error.emit(msg)

    @Slot(bytes, int, int, bool)
    def run_inspect_calibration(self, frame_bytes: bytes, width: int, height: int, snap: bool = False) -> None:
        try:
            self.inspect_calibration_result_ready.emit(
                self._inspect_calibration.process(frame_bytes, width, height, snap)
            )
        except Exception:
            msg = traceback.format_exc()
            error(f"_VisionWorker: inspect calibration failed:\n{msg}")
            self.inspect_calibration_error.emit(msg)

    @Slot(bytes, int, int)
    def run_red_mark_detection(self, frame_bytes: bytes, width: int, height: int) -> None:
        try:
            self.red_mark_detection_result_ready.emit(
                self._red_mark_detection.process(frame_bytes, width, height)
            )
        except Exception:
            msg = traceback.format_exc()
            error(f"_VisionWorker: red mark detection failed:\n{msg}")
            self.red_mark_detection_error.emit(msg)

    def reset_inspect_calibration_state(self) -> None:
        self._inspect_calibration.reset()

    def reset_red_mark_state(self) -> None:
        self._red_mark_detection.reset()

    @Slot(bytes, int, int)
    def run_background_detection(self, frame_bytes: bytes, width: int, height: int) -> None:
        try:
            self.background_detection_result_ready.emit(
                self._background_detection.process(frame_bytes, width, height)
            )
        except Exception:
            msg = traceback.format_exc()
            error(f"_VisionWorker: background detection failed:\n{msg}")
            self.background_detection_error.emit(msg)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class MachineVisionManager(QObject):
    """
    GUI-thread owner of the machine-vision pipeline.

    Signals
    -------
    settings_changed():
        Emitted after settings are applied so UI pages can refresh.

    Request API
    -----------
    Every ``request_*`` / ``submit_*`` method returns a ``Future`` that is
    always resolved — either with the result or with the worker exception
    reraised.  Callers that want signal-based delivery attach a done-callback
    to the returned ``Future``; no PySide6 import is required in the callback.

    Latest-frame-wins policy
    ------------------------
    At most one request waits behind the in-flight job per job type.  When a
    new request arrives while the worker is busy the previously waiting
    request is cancelled and replaced.
    """

    settings_changed = Signal()

    _request_focus = Signal(bytes, int, int, bool)
    _request_calibration_build = Signal(
        bytes, int, int,
        bytes, int, int,
        bytes, int, int,
        int, int,
        int, int, int,
    )
    _request_inspect_calibration = Signal(bytes, int, int, bool)
    _request_red_mark_detection = Signal(bytes, int, int)
    _request_background_detection = Signal(bytes, int, int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        self._settings_manager = MachineVisionSettingsManager()
        self._settings: MachineVisionSettings = self._load_settings()

        self._thread = QThread(self)
        self._thread.setObjectName("MachineVisionThread")

        self._worker = _VisionWorker(self._settings, self.save_settings)
        self._worker.moveToThread(self._thread)

        self._focus_queue = self._register_queue(
            request_signal=self._request_focus,
            worker_slot=self._worker.run_focus_analysis,
            result_signal=self._worker.focus_result_ready,
            error_signal=self._worker.focus_error,
        )
        self._calibration_queue = self._register_queue(
            request_signal=self._request_calibration_build,
            worker_slot=self._worker.run_calibration_build,
            result_signal=self._worker.calibration_ready,
            error_signal=self._worker.calibration_error,
        )
        self._inspect_queue = self._register_queue(
            request_signal=self._request_inspect_calibration,
            worker_slot=self._worker.run_inspect_calibration,
            result_signal=self._worker.inspect_calibration_result_ready,
            error_signal=self._worker.inspect_calibration_error,
        )
        self._red_mark_queue = self._register_queue(
            request_signal=self._request_red_mark_detection,
            worker_slot=self._worker.run_red_mark_detection,
            result_signal=self._worker.red_mark_detection_result_ready,
            error_signal=self._worker.red_mark_detection_error,
        )
        self._background_queue = self._register_queue(
            request_signal=self._request_background_detection,
            worker_slot=self._worker.run_background_detection,
            result_signal=self._worker.background_detection_result_ready,
            error_signal=self._worker.background_detection_error,
        )

        self._thread.start()
        info("MachineVisionManager: worker thread started")

    def _register_queue(
        self,
        request_signal: Signal,
        worker_slot,
        result_signal: Signal,
        error_signal: Signal,
    ) -> JobQueue:
        request_signal.connect(worker_slot)
        return JobQueue(
            dispatch=lambda args: request_signal.emit(*args),
            result_signal=result_signal,
            error_signal=error_signal,
        )

    # ------------------------------------------------------------------
    # Settings API
    # ------------------------------------------------------------------

    @property
    def settings(self) -> MachineVisionSettings:
        """
        The live settings object shared with all worker algorithms.

        Mutate fields directly, then call ``notify_settings_changed()`` if the
        UI needs to refresh, and ``save_settings()`` to persist to disk.
        """
        return self._settings

    def notify_settings_changed(self) -> None:
        """Emit ``settings_changed`` so UI pages can refresh after a direct mutation."""
        self.settings_changed.emit()

    def save_settings(self) -> None:
        """Persist the current settings to disk."""
        try:
            self._settings_manager.save(self._settings)
            info("MachineVisionManager: settings saved")
        except Exception as exc:
            error(f"MachineVisionManager: failed to save settings — {exc}")

    # ------------------------------------------------------------------
    # Calibration API
    # ------------------------------------------------------------------

    @property
    def calibration(self) -> CameraCalibration | None:
        """The most recently completed calibration, or ``None`` if uncalibrated."""
        return self._settings.camera_calibration.calibration

    @property
    def is_calibrated(self) -> bool:
        """``True`` when a valid calibration is available."""
        return self._settings.camera_calibration.calibration is not None

    def pixel_to_world_delta(
        self,
        pixel_x: float,
        pixel_y: float,
        image_center_x: float | None = None,
        image_center_y: float | None = None,
    ) -> tuple[float, float] | None:
        """
        Convert a pixel coordinate to a stage delta in tick units.

        Delegates to ``CameraCalibration.pixel_to_world_delta``.  Returns
        ``None`` when no calibration is available.
        """
        cal = self._settings.camera_calibration.calibration
        if cal is None:
            return None
        return cal.pixel_to_world_delta(pixel_x, pixel_y, image_center_x, image_center_y)

    def submit_calibration_frames(
        self,
        base_frame: np.ndarray,
        base_width: int,
        base_height: int,
        x_frame: np.ndarray,
        x_width: int,
        x_height: int,
        y_frame: np.ndarray,
        y_width: int,
        y_height: int,
        ref_x: int,
        ref_y: int,
        ref_z: int,
        move_x_ticks: int | None = None,
        move_y_ticks: int | None = None,
    ) -> Future[CameraCalibration]:
        """
        Submit three RGB888 frames for calibration and return a
        ``Future[CameraCalibration]``.

        The three frames must have been captured at the base (reference)
        position, after a +X move, and after a +Y move respectively.  The
        stage must have returned to the base position between the X and Y
        captures.

        Each frame is copied immediately so camera buffers may be reused.
        ``move_x_ticks`` and ``move_y_ticks`` default to the values stored in
        ``settings.camera_calibration`` if not supplied.

        On success the calibration is written into the settings object and
        persisted automatically.
        """
        cc = self._settings.camera_calibration
        mx = move_x_ticks if move_x_ticks is not None else cc.move_x_ticks
        my = move_y_ticks if move_y_ticks is not None else cc.move_y_ticks

        args = (
            bytes(base_frame), base_width, base_height,
            bytes(x_frame),    x_width,    x_height,
            bytes(y_frame),    y_width,    y_height,
            mx, my,
            ref_x, ref_y, ref_z,
        )
        return self._calibration_queue.submit(args)

    def submit_calibration_frames_guaranteed(
        self,
        base_frame: np.ndarray,
        base_width: int,
        base_height: int,
        x_frame: np.ndarray,
        x_width: int,
        x_height: int,
        y_frame: np.ndarray,
        y_width: int,
        y_height: int,
        ref_x: int,
        ref_y: int,
        ref_z: int,
        move_x_ticks: int | None = None,
        move_y_ticks: int | None = None,
    ) -> Future[CameraCalibration]:
        """
        Submit three RGB888 frames for calibration, guaranteed to be executed.

        Identical to ``submit_calibration_frames`` but the request is appended
        to a FIFO queue that is never dropped or cancelled.  Guaranteed requests
        are always drained before any waiting droppable request is dispatched.

        Each frame is copied immediately so camera buffers may be reused.
        ``move_x_ticks`` and ``move_y_ticks`` default to the values stored in
        ``settings.camera_calibration`` if not supplied.

        On success the calibration is written into the settings object and
        persisted automatically.
        """
        cc = self._settings.camera_calibration
        mx = move_x_ticks if move_x_ticks is not None else cc.move_x_ticks
        my = move_y_ticks if move_y_ticks is not None else cc.move_y_ticks

        args = (
            bytes(base_frame), base_width, base_height,
            bytes(x_frame),    x_width,    x_height,
            bytes(y_frame),    y_width,    y_height,
            mx, my,
            ref_x, ref_y, ref_z,
        )
        return self._calibration_queue.submit_guaranteed(args)

    def clear_calibration(self) -> None:
        """Discard the current calibration from memory and from persisted settings."""
        self._settings.camera_calibration.calibration = None
        self.save_settings()
        info("MachineVisionManager: calibration cleared")

    # ------------------------------------------------------------------
    # Analysis API
    # ------------------------------------------------------------------

    def request_focus_analysis(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        use_region: bool = True,
    ) -> Future[FocusResult]:
        """
        Submit a focus analysis request using latest-frame-wins policy.

        If a request is already waiting it is cancelled and replaced with this
        one.  Use this for continuous preview feeds where only the most recent
        frame matters.

        Pass ``use_region=False`` to ignore the focus region even when it is
        enabled in settings.

        The frame is copied immediately so the camera buffer may be reused
        before the future resolves.
        """
        return self._focus_queue.submit((bytes(frame), width, height, use_region))

    def request_focus_analysis_guaranteed(
        self,
        frame: np.ndarray,
        use_region: bool = True,
    ) -> Future[FocusResult]:
        """
        Submit a focus analysis request that is guaranteed to be executed.

        The request is appended to a FIFO queue that is never dropped or
        cancelled.  Guaranteed requests are always drained before any waiting
        droppable preview request is dispatched.  Use this when a result for a
        specific captured frame must be obtained (e.g. autofocus scoring).

        Pass ``use_region=False`` to ignore the focus region even when it is
        enabled in settings.

        The frame is copied immediately so the camera buffer may be reused
        before the future resolves.
        """
        return self._focus_queue.submit_guaranteed((bytes(frame), frame.shape[1], frame.shape[0], use_region))

    def request_inspect_calibration(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        snap: bool = False,
    ) -> Future[InspectCalibrationResult]:
        """
        Submit a calibration-bar inspection request and return a
        ``Future[InspectCalibrationResult]``.

        Pass ``snap=True`` to use snap-mode parameters (2x downsampling,
        tick_min_length 200).

        The frame is copied immediately so the camera buffer may be reused
        before the future resolves.
        """
        return self._inspect_queue.submit((bytes(frame), width, height, snap))

    def request_inspect_calibration_guaranteed(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        snap: bool = False,
    ) -> Future[InspectCalibrationResult]:
        """
        Submit a calibration-bar inspection request that is guaranteed to be executed.

        The request is appended to a FIFO queue that is never dropped or
        cancelled.  Guaranteed requests are always drained before any waiting
        droppable preview request is dispatched.  Use this when a result for a
        specific captured frame must be obtained.

        Pass ``snap=True`` to use snap-mode parameters (2x downsampling,
        tick_min_length 200).

        The frame is copied immediately so the camera buffer may be reused
        before the future resolves.
        """
        return self._inspect_queue.submit_guaranteed((bytes(frame), width, height, snap))

    def reset_inspect_calibration_state(self) -> None:
        """
        Reset the axis hysteresis state on the worker.

        Call this when starting a new inspection session so the confirmed-axis
        streak does not carry over from a previous run.
        """
        self._worker.reset_inspect_calibration_state()

    def request_red_mark_detection(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
    ) -> Future[RedMarkDetectionResult]:
        """
        Submit a red-mark detection request and return a
        ``Future[RedMarkDetectionResult]``.

        The frame is copied immediately so the camera buffer may be reused
        before the future resolves.
        """
        return self._red_mark_queue.submit((bytes(frame), width, height))

    def request_red_mark_detection_guaranteed(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
    ) -> Future[RedMarkDetectionResult]:
        """
        Submit a red-mark detection request that is guaranteed to be executed.

        The request is appended to a FIFO queue that is never dropped or
        cancelled.  Guaranteed requests are always drained before any waiting
        droppable preview request is dispatched.  Use this when a result for a
        specific captured frame must be obtained.

        The frame is copied immediately so the camera buffer may be reused
        before the future resolves.
        """
        return self._red_mark_queue.submit_guaranteed((bytes(frame), width, height))

    def reset_red_mark_state(self) -> None:
        """
        Reset the smoothing and hysteresis state for red-mark detection.

        Call this when starting a new detection session so stale EMA values
        from a previous run do not bias the first result.
        """
        self._worker.reset_red_mark_state()

    def request_background_detection(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
    ) -> Future[BackgroundDetectionResult]:
        """
        Submit a background detection request and return a
        ``Future[BackgroundDetectionResult]``.

        The frame is copied immediately so the camera buffer may be reused
        before the future resolves.
        """
        return self._background_queue.submit((bytes(frame), width, height))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if not self._thread.isRunning():
            return
        info("MachineVisionManager: shutting down worker thread...")

        self._focus_queue.cancel_pending()
        self._calibration_queue.cancel_pending()
        self._inspect_queue.cancel_pending()
        self._red_mark_queue.cancel_pending()
        self._background_queue.cancel_pending()

        self._thread.quit()
        if not self._thread.wait(3000):
            warning("MachineVisionManager: worker thread did not exit in time; terminating")
            self._thread.terminate()
            self._thread.wait()
        info("MachineVisionManager: worker thread stopped")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_settings(self) -> MachineVisionSettings:
        try:
            s = self._settings_manager.load()
            info("MachineVisionManager: settings loaded")
            return s
        except Exception as exc:
            error(f"MachineVisionManager: failed to load settings — {exc}; using defaults")
            return MachineVisionSettings()