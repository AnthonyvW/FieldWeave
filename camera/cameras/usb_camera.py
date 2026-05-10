from __future__ import annotations

from pathlib import Path
from typing import Callable, Any
import threading

import cv2
import numpy as np

from camera.cameras.base_camera import BaseCamera
from camera.settings.usb_settings import UsbCameraSettings
from common.logger import info, debug, error


class UsbCamera(BaseCamera):
    """
    USB/webcam implementation using OpenCV.

    USB cameras do not support a dedicated still-capture mode. When a still is
    requested the current frame is read directly from the live stream and saved.
    The first time a camera is opened, all supported resolutions are probed and
    stored in settings so the slow enumeration is never repeated.
    """

    def __init__(self, model: str):
        super().__init__(model=model)
        self._cap: cv2.VideoCapture | None = None
        self._device_index: int = 0
        self._frame_buffer: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._capture_thread: threading.Thread | None = None
        self._running = threading.Event()
        self._paused = threading.Event()

    @property
    def settings(self) -> UsbCameraSettings:
        if self._settings is None:
            raise RuntimeError("Settings not initialized. Call initialize_settings() first.")
        return self._settings

    def _get_settings_class(self) -> type[UsbCameraSettings]:
        return UsbCameraSettings

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------

    def open(self, device_index: int = 0, backend: int | None = None) -> bool:
        self._device_index = device_index

        if backend is not None:
            cap = cv2.VideoCapture(device_index, backend)
        else:
            cap = cv2.VideoCapture(device_index, cv2.CAP_MSMF)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(device_index)

        if not cap.isOpened():
            error(f"Failed to open USB camera at index {device_index}")
            return False

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self.initialize_settings()
        self._is_open = True
        info(f"USB camera opened at index {device_index}: {self.model}")
        return True

    def close(self) -> None:
        self._running.clear()

        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self._capture_thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._frame_buffer = None
        self._is_open = False
        self._callback = None
        self._callback_context = None
        info(f"USB camera {self.model} closed")

    # ------------------------------------------------------------------
    # Capture loop
    # ------------------------------------------------------------------

    def start_capture(self, callback: Callable, context: Any) -> bool:
        if not self._cap or not self._cap.isOpened():
            error("Cannot start capture: camera not open")
            return False

        self._callback = callback
        self._callback_context = context
        self._running.set()

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="UsbCameraCapture",
        )
        self._capture_thread.start()
        info(f"USB camera capture started for {self.model}")
        return True

    def stop_capture(self) -> None:
        self._running.clear()
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self._capture_thread = None
        info(f"USB camera capture stopped for {self.model}")

    def _capture_loop(self) -> None:
        while self._running.is_set():
            if self._cap is None or not self._cap.isOpened():
                break

            if self._paused.is_set():
                self._paused.wait(timeout=0.005)
                continue

            if not self._cap.grab():
                continue

            ret, frame = self._cap.retrieve()
            if not ret:
                continue

            with self._frame_lock:
                self._frame_buffer = frame

            if self._callback is not None:
                self._callback(frame, self._callback_context)

    # ------------------------------------------------------------------
    # Still capture — reads a frame from the live stream
    # ------------------------------------------------------------------

    def capture_still(
        self,
        resolution_index: int | None = None,
        timeout_ms: int = 5000,
        on_captured: Callable[[], None] | None = None,
        on_complete: Callable[[bool, np.ndarray | None], None] | None = None,
    ) -> bool:
        """Capture a frame from the live stream and return it as a numpy array.

        USB cameras have no dedicated still mode so this reads the most recent
        buffered frame (or grabs a fresh one when the stream is not running).
        resolution_index and timeout_ms are accepted for interface compatibility
        but unused. Runs synchronously on the calling thread — this is always
        the CameraThread background thread, so no secondary thread is needed.
        on_captured and on_complete fire before returning.
        """
        frame = self._grab_frame()
        if frame is None:
            error("USB still capture: no frame available")
            if on_complete is not None:
                on_complete(False, None)
            return False

        if on_captured is not None:
            on_captured()

        if on_complete is not None:
            on_complete(True, frame.copy())

        return True

    def capture_and_save_still(
        self,
        filepath: Path,
        additional_metadata: dict[str, Any] | None = None,
        resolution_index: int | None = None,
        timeout_ms: int = 5000,
        on_captured: Callable[[], None] | None = None,
        on_complete: Callable[[bool], None] | None = None,
    ) -> bool:
        """Capture a frame from the live stream and save it as a still image.

        Delegates to capture_still. Runs synchronously on the calling thread —
        save completes and on_complete fires before this method returns.
        """
        def _on_complete(success: bool, image: np.ndarray | None) -> None:
            if image is not None:
                save_ok = self.save_image(image, filepath, additional_metadata)
                if save_ok:
                    info(f"USB still saved: {filepath}")
                else:
                    error(f"USB still save failed: {filepath}")
            else:
                save_ok = False

            if on_complete is not None:
                on_complete(save_ok)

        return self.capture_still(
            resolution_index=resolution_index,
            timeout_ms=timeout_ms,
            on_captured=on_captured,
            on_complete=_on_complete,
        )

    def capture_and_save_stream(
        self,
        filepath: Path,
        additional_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Capture the current stream frame and save it."""
        frame = self._grab_frame()
        if frame is None:
            error("USB stream capture: no frame available")
            return False

        success = self.save_image(frame, filepath, additional_metadata)

        if success:
            info(f"USB stream frame saved: {filepath}")
        else:
            error(f"USB stream frame save failed: {filepath}")

        return success

    def _grab_frame(self) -> np.ndarray | None:
        """Return the latest buffered frame as RGB.

        When the capture loop is running, reads from the shared buffer only —
        never touches _cap directly to avoid concurrent OpenCV access.
        Falls back to a direct cap read only when the loop is not running.
        """
        with self._frame_lock:
            if self._frame_buffer is not None:
                return cv2.cvtColor(self._frame_buffer, cv2.COLOR_BGR2RGB)

        if self._running.is_set():
            return None

        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Resolution changes (must pause the capture loop while applying)
    # ------------------------------------------------------------------

    def set_preview_resolution(self, value: str, index: int | None = None) -> bool:
        """Set the preview resolution, pausing the capture loop while applying.

        Many drivers (especially MSMF) silently ignore CAP_PROP_FRAME_WIDTH /
        CAP_PROP_FRAME_HEIGHT when a grab is in-flight. Pausing the loop before
        the write and resuming after ensures the property change lands cleanly.
        """
        self._paused.set()
        result = self.settings.set_preview_resolution(value, index)
        self._paused.clear()
        return result

    # ------------------------------------------------------------------
    # Probe (blocking — call from a worker thread)
    # ------------------------------------------------------------------

    def probe_and_cache(self) -> None:
        """Run resolution and property-range probes, then persist the results.

        Intentionally slow — must be called once on a background thread after
        the first open. Settings are saved to disk afterward so subsequent opens
        find a populated cache and skip both probes entirely.
        """
        self.settings.probe_and_cache()
        self.save_settings()
        debug(f"Probe complete and settings saved for {self.model}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def current_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            if self._frame_buffer is None:
                return None
            return cv2.cvtColor(self._frame_buffer, cv2.COLOR_BGR2RGB)