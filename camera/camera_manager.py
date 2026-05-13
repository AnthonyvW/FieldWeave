"""
Camera manager for handling camera enumeration, selection, and lifecycle.
Provides plugin architecture for multiple camera types and manages frame acquisition.
"""

from __future__ import annotations

from typing import Callable, Any

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, Qt, QTimer

from camera.cameras.base_camera import BaseCamera
from camera.cameras.usb_camera import UsbCamera
from camera.threaded_camera import ThreadedCamera
from camera.camera_enumerator import (
    CameraEnumerator,
    CameraInfo,
    CameraType,
    AmscopeEnumerator,
    GenericUSBEnumerator,
)
from common.logger import info, error, warning, exception, debug


class CameraManager(QObject):
    """
    Manages camera enumeration, selection, lifecycle, and frame acquisition.

    Two frame-delivery models are supported transparently:

    - **Event-driven** (e.g. Amscope): the SDK fires an integer event on a
      background thread; ``_camera_callback`` forwards it via
      ``_camera_event`` signal to the main thread, which pulls the frame
      from the SDK buffer in ``_on_camera_event``.

    - **Push-callback** (e.g. USB/OpenCV): the capture loop calls
      ``_usb_frame_callback`` with a numpy array on a background thread;
      the frame is stored in plain data and the poll timer on the main thread
      wakes the main thread to update state and emit public signals.

    Signals:
        camera_list_changed:    Emitted when available cameras change.
        active_camera_changed:  Emitted when active camera changes (CameraInfo or None).
        enumeration_complete:   Emitted when camera enumeration completes (camera_count).
        preview_frame_ready:    Emitted when a new preview frame is available (width, height).
        still_frame_ready:      Emitted when a new still frame is available (width, height).
        streaming_started:      Emitted when camera streaming starts (width, height).
        streaming_stopped:      Emitted when camera streaming stops.
        camera_error:           Emitted when a camera error occurs.
        camera_disconnected:    Emitted when camera is disconnected.
    """

    camera_list_changed = Signal()
    active_camera_changed = Signal(object)
    enumeration_complete = Signal(int)
    preview_frame_ready = Signal(int, int)
    still_frame_ready = Signal(int, int)
    streaming_started = Signal(int, int)
    streaming_stopped = Signal()
    camera_error = Signal(str)
    camera_disconnected = Signal()

    _camera_event = Signal(int)

    _USB_POLL_INTERVAL_MS = 1000/60

    def __init__(self):
        super().__init__()

        self._enumerators: list[CameraEnumerator] = [
            AmscopeEnumerator(),
            GenericUSBEnumerator(),
        ]

        self._available_cameras: list[CameraInfo] = []

        self._active_camera: ThreadedCamera | None = None
        self._active_camera_info: CameraInfo | None = None
        self._camera_thread_started = False

        self._current_frame_buffer: bytes | None = None
        self._frame_width = 0
        self._frame_height = 0
        self._frame_stride = 0
        self._preview_frame_seq: int = 0

        self._current_still_buffer: bytes | None = None
        self._still_frame_width: int = 0
        self._still_frame_height: int = 0
        self._still_frame_stride: int = 0
        self._still_frame_seq: int = 0

        # Latest numpy frame written by the USB capture thread; read by poll timer on main thread
        self._pending_usb_frame: np.ndarray | None = None

        self._is_streaming = False
        self._last_enumerated_count: int | None = None

        self._frame_poll_timer = QTimer(self)
        self._frame_poll_timer.setInterval(self._USB_POLL_INTERVAL_MS)
        self._frame_poll_timer.timeout.connect(self._poll_usb_frame)

        self._camera_event.connect(self._on_camera_event, Qt.ConnectionType.QueuedConnection)

        info("Camera manager initialized")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available_cameras(self) -> list[CameraInfo]:
        return self._available_cameras.copy()

    @property
    def active_camera(self) -> ThreadedCamera | None:
        return self._active_camera

    @property
    def active_camera_info(self) -> CameraInfo | None:
        return self._active_camera_info

    @property
    def has_active_camera(self) -> bool:
        return self._active_camera is not None

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def frame_dimensions(self) -> tuple[int, int]:
        return (self._frame_width, self._frame_height)

    @property
    def preview_frame_stride(self) -> int:
        return self._frame_stride

    @property
    def still_frame_dimensions(self) -> tuple[int, int]:
        return (self._still_frame_width, self._still_frame_height)

    @property
    def still_frame_stride(self) -> int:
        return self._still_frame_stride

    @property
    def preview_frame_seq(self) -> int:
        return self._preview_frame_seq

    @property
    def still_frame_seq(self) -> int:
        return self._still_frame_seq

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def enumerate_cameras(self) -> list[CameraInfo]:
        cameras = []

        for enumerator in self._enumerators:
            enumerator_type = enumerator.get_camera_type().value
            if enumerator.is_available():
                enum_cameras = enumerator.enumerate()
                cameras.extend(enum_cameras)
            else:
                debug(f"{enumerator_type} enumerator not available")

        self._available_cameras = cameras

        count = len(cameras)
        if count != self._last_enumerated_count:
            if cameras:
                info(f"Found {count} camera(s):")
                for idx, cam in enumerate(cameras):
                    info(f"  [{idx}] {cam.display_name} ({cam.model})")
            else:
                info("No cameras found")
            self._last_enumerated_count = count

        self.camera_list_changed.emit()
        self.enumeration_complete.emit(count)

        return cameras

    def get_camera_by_id(self, device_id: str) -> CameraInfo | None:
        for camera_info in self._available_cameras:
            if camera_info.device_id == device_id:
                return camera_info
        return None

    def get_cameras_by_type(self, camera_type: CameraType) -> list[CameraInfo]:
        return [cam for cam in self._available_cameras if cam.camera_type == camera_type]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def switch_camera(self, camera_info: CameraInfo, start_streaming: bool = True) -> bool:
        info(f"Switching to camera: {camera_info}")

        if self._active_camera is not None:
            info("Closing current camera before switching")
            self.close_camera()

        camera = self._create_camera_instance(camera_info)
        if camera is None:
            error(f"Failed to create camera instance for {camera_info}")
            return False

        threaded_camera = ThreadedCamera(camera)
        threaded_camera.start_thread()
        self._camera_thread_started = True

        self._active_camera = threaded_camera
        self._active_camera_info = camera_info

        info(f"Opening camera: {camera_info.display_name}")

        def _on_open_complete(success: bool, _result: object) -> None:
            if not success:
                error(f"Failed to open camera: {camera_info}")
                threaded_camera.stop_thread(wait=False)
                self._active_camera = None
                self._active_camera_info = None
                self._camera_thread_started = False
                return

            debug(f"Successfully switched to camera: {camera_info}")
            self.active_camera_changed.emit(camera_info)

            if start_streaming:
                self.start_streaming()

        open_kwargs: dict = {"on_complete": _on_open_complete}
        if camera_info.camera_type == CameraType.GENERIC_USB and camera_info.metadata:
            backend = camera_info.metadata.get("backend")
            if backend is not None:
                open_kwargs["backend"] = backend
            threaded_camera.open(int(camera_info.device_id), **open_kwargs)
        else:
            threaded_camera.open(camera_info.device_id, **open_kwargs)

        return True

    def open_first_available(self, start_streaming: bool = True) -> bool:
        cameras = self.enumerate_cameras()

        if not cameras:
            warning("No cameras available to open")
            return False

        return self.switch_camera(cameras[0], start_streaming=start_streaming)

    def close_camera(self) -> bool:
        if self._active_camera is None:
            info("No active camera to close")
            return True

        info(f"Closing camera: {self._active_camera_info}")

        self.stop_streaming()

        result = self._active_camera.close(wait=True)

        if result is not None:
            success, _ = result
            if not success:
                warning("Camera close returned failure")

        if self._camera_thread_started:
            self._active_camera.stop_thread(wait=True)
            self._camera_thread_started = False

        prev_info = self._active_camera_info
        self._active_camera = None
        self._active_camera_info = None

        info(f"Camera closed: {prev_info}")
        self.active_camera_changed.emit(None)
        return True

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def start_streaming(self) -> bool:
        if not self._active_camera:
            error("Cannot start streaming - no active camera")
            return False

        if self._is_streaming:
            debug("Streaming already active")
            return True

        base_camera = self._active_camera.underlying_camera

        if isinstance(base_camera, UsbCamera):
            return self._start_usb_streaming()

        return self._start_event_driven_streaming(base_camera)

    def _start_usb_streaming(self) -> bool:
        def _on_capture_started(success: bool, _result: object) -> None:
            if not success:
                error("USB start_capture returned False")
                return

            self._is_streaming = True
            self._frame_poll_timer.start()
            info("USB streaming started — dimensions will be set on first frame")
            self.streaming_started.emit(0, 0)

        self._active_camera.start_capture(self._usb_frame_callback, self, on_complete=_on_capture_started)
        return True

    def _start_event_driven_streaming(self, base_camera: BaseCamera) -> bool:
        _, width, height = base_camera.get_current_preview_resolution()

        if width == 0 or height == 0:
            info("Setting default resolution...")
            resolutions = base_camera.get_resolutions()
            if not resolutions:
                error("No resolutions available")
                return False
            _, width, height = base_camera.get_current_preview_resolution()

        width, height = self._active_camera.settings.get_output_dimensions()
        self._frame_width = width
        self._frame_height = height

        success = base_camera.start_capture(self._camera_callback, self)

        if not success:
            error("start_capture returned False")
            return False

        self._is_streaming = True
        info(f"Streaming started ({width}x{height})")
        self.streaming_started.emit(width, height)
        return True

    def stop_streaming(self) -> bool:
        if not self._is_streaming:
            debug("Streaming not active")
            return True

        self._frame_poll_timer.stop()
        self._pending_usb_frame = None

        if self._active_camera and self._active_camera.underlying_camera.is_open:
            info("Stopping camera streaming...")
            self._active_camera.stop_capture()

        self._is_streaming = False
        self._current_frame_buffer = None
        self._frame_width = 0
        self._frame_height = 0
        self._frame_stride = 0
        self._preview_frame_seq = 0
        self._current_still_buffer = None
        self._still_frame_width = 0
        self._still_frame_height = 0
        self._still_frame_stride = 0
        self._still_frame_seq = 0
        info("Streaming stopped")
        self.streaming_stopped.emit()
        return True

    # ------------------------------------------------------------------
    # Still capture
    # ------------------------------------------------------------------

    def capture_still(
        self,
        resolution_index: int | None = None,
        timeout_ms: int = 5000,
        on_captured: Callable[[], None] | None = None,
        on_complete: Callable[[bool, np.ndarray | None], None] | None = None,
    ) -> bool:
        if not self._active_camera:
            error("Cannot capture still - no active camera")
            return False

        base_camera = self._active_camera.underlying_camera

        if isinstance(base_camera, UsbCamera):
            self._active_camera.capture_still(
                resolution_index=resolution_index,
                timeout_ms=timeout_ms,
                on_captured=on_captured,
                on_complete=on_complete,
            )
            return True

        return base_camera.capture_still(
            resolution_index=resolution_index,
            timeout_ms=timeout_ms,
            on_captured=on_captured,
            on_complete=on_complete,
        )

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_current_frame(self) -> bytes | None:
        return self._current_frame_buffer

    def get_current_still_frame(self) -> bytes | None:
        return self._current_still_buffer

    def copy_current_frame_to_numpy(self) -> np.ndarray | None:
        if not self._current_frame_buffer or self._frame_width == 0 or self._frame_height == 0:
            return None

        stride = self._frame_stride
        if stride == 0:
            return None

        required = stride * self._frame_height
        if len(self._current_frame_buffer) < required:
            return None

        arr = np.frombuffer(self._current_frame_buffer, dtype=np.uint8)
        bytes_per_pixel = 3

        if stride == self._frame_width * bytes_per_pixel:
            return arr[:required].reshape((self._frame_height, self._frame_width, bytes_per_pixel)).copy()

        arr_2d = arr[:required].reshape((self._frame_height, stride))
        return arr_2d[:, :self._frame_width * bytes_per_pixel].reshape(
            (self._frame_height, self._frame_width, bytes_per_pixel)
        ).copy()

    def copy_current_still_frame_to_numpy(self) -> np.ndarray | None:
        if (
            not self._current_still_buffer
            or self._still_frame_width == 0
            or self._still_frame_height == 0
        ):
            return None

        stride = self._still_frame_stride
        if stride == 0:
            return None

        arr = np.frombuffer(self._current_still_buffer, dtype=np.uint8)
        bytes_per_pixel = 3

        if stride == self._still_frame_width * bytes_per_pixel:
            return arr.reshape(
                (self._still_frame_height, self._still_frame_width, bytes_per_pixel)
            ).copy()

        arr_2d = arr.reshape((self._still_frame_height, stride))
        return arr_2d[:, :self._still_frame_width * bytes_per_pixel].reshape(
            (self._still_frame_height, self._still_frame_width, bytes_per_pixel)
        ).copy()

    # ------------------------------------------------------------------
    # USB push-callback path (background thread → main thread via signal)
    # ------------------------------------------------------------------

    @staticmethod
    def _usb_frame_callback(frame: np.ndarray, context: Any) -> None:
        """Called by the USB capture loop on a background thread.

        Only writes plain data — no signals, no GUI calls.
        The poll timer on the main thread drains _pending_usb_frame.
        """
        import cv2  # noqa: PLC0415
        if not isinstance(context, CameraManager):
            return
        context._pending_usb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @Slot()
    def _poll_usb_frame(self) -> None:
        """Drain the latest USB frame on the main thread (called by poll timer)."""
        frame = self._pending_usb_frame
        self._pending_usb_frame = None

        if frame is None or not self._is_streaming:
            return

        height, width = frame.shape[:2]

        if width != self._frame_width or height != self._frame_height:
            info(f"USB resolution changed to {width}x{height}")
            self._frame_width = width
            self._frame_height = height
            if self._frame_width > 0 and self._frame_height > 0:
                self.streaming_started.emit(self._frame_width, self._frame_height)

        self._current_frame_buffer = frame.tobytes()
        self._frame_stride = width * 3
        self._preview_frame_seq += 1
        self.preview_frame_ready.emit(self._frame_width, self._frame_height)

    # ------------------------------------------------------------------
    # Event-driven path (Amscope SDK integer events)
    # ------------------------------------------------------------------

    @staticmethod
    def _camera_callback(event: int, context: Any) -> None:
        """SDK event callback — runs on a background thread.

        Forwards the raw event integer to the main thread via signal without
        touching any GUI state.
        """
        if isinstance(context, CameraManager):
            context._camera_event.emit(event)

    @Slot(int)
    def _on_camera_event(self, event: int) -> None:
        if not self._active_camera:
            return

        base_camera = self._active_camera.underlying_camera

        if not base_camera.is_open:
            return

        events = base_camera.get_event_constants()

        if event == events.IMAGE:
            self._handle_image_event()
        elif event == events.STILLIMAGE:
            self._handle_still_image_event()
        elif event == events.ERROR:
            self._handle_error()
        elif event == events.DISCONNECTED:
            self._handle_disconnected()

    def _handle_image_event(self) -> None:
        if not self._active_camera:
            return

        base_camera = self._active_camera.underlying_camera
        result = base_camera.get_frame_buffer()
        if result is None:
            debug("Preview image event received but frame buffer not yet allocated")
            return

        frame_buf, current_width, current_height = result

        if current_width != self._frame_width or current_height != self._frame_height:
            info(f"Preview resolution changed to {current_width}x{current_height}")
            self._frame_width = current_width
            self._frame_height = current_height

        self._current_frame_buffer = frame_buf
        self._frame_stride = type(base_camera).calculate_stride(current_width, 24)
        self._preview_frame_seq += 1
        self.preview_frame_ready.emit(self._frame_width, self._frame_height)

    def _handle_still_image_event(self) -> None:
        if not self._active_camera:
            return

        base_camera = self._active_camera.underlying_camera
        result = base_camera.get_still_buffer()
        if result is None:
            debug("Still image event received but still buffer not yet allocated")
            return

        still_buf, still_w, still_h = result

        if still_w != self._still_frame_width or still_h != self._still_frame_height:
            info(f"Still resolution changed to {still_w}x{still_h}")
            self._still_frame_width = still_w
            self._still_frame_height = still_h

        self._current_still_buffer = bytes(still_buf)
        self._still_frame_stride = type(base_camera).calculate_stride(still_w, 24)
        self._still_frame_seq += 1
        self.still_frame_ready.emit(self._still_frame_width, self._still_frame_height)

    def _handle_error(self) -> None:
        description = "unknown"
        if self._active_camera is not None:
            base_camera = self._active_camera.underlying_camera
            if hasattr(base_camera, "get_last_error_description"):
                description = base_camera.get_last_error_description()

        error(f"Camera error occurred: {description}")
        self.camera_error.emit(description)
        self.stop_streaming()

    def _handle_disconnected(self) -> None:
        warning("Camera disconnected")
        self.camera_disconnected.emit()
        self.stop_streaming()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def _create_camera_instance(self, camera_info: CameraInfo) -> BaseCamera | None:
        if camera_info.camera_type == CameraType.AMSCOPE:
            from camera.cameras.amscope_camera import AmscopeCamera  # noqa: PLC0415
            return AmscopeCamera(camera_info.model)

        if camera_info.camera_type == CameraType.GENERIC_USB:
            return UsbCamera(camera_info.model)

        error(f"Unsupported camera type: {camera_info.camera_type}")
        return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        info("Cleaning up camera manager")
        self.stop_streaming()
        self.close_camera()
        self._available_cameras.clear()
        self.camera_list_changed.emit()