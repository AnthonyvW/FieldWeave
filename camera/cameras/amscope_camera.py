"""
Amscope camera implementation using the amcam SDK.
Now with integrated settings management.
"""

from __future__ import annotations

from typing import Callable, Any
from types import SimpleNamespace
from pathlib import Path
import ctypes
import time
import threading
import gc

import numpy as np

from camera.cameras.base_camera import BaseCamera, CameraResolution
from camera.settings.amscope_settings import AmscopeSettings
from common.logger import info, debug, error, exception, warning

_amcam = None

_HRESULT_NAMES: dict[int, str] = {
    0x00000000: "S_OK",
    0x00000001: "S_FALSE",
    0x8000FFFF: "E_UNEXPECTED (catastrophic failure)",
    0x80004001: "E_NOTIMPL (not supported on this camera)",
    0x80070005: "E_ACCESSDENIED (permission denied)",
    0x8007000E: "E_OUTOFMEMORY (out of memory)",
    0x80070057: "E_INVALIDARG (invalid argument)",
    0x80004003: "E_POINTER (null pointer)",
    0x80004005: "E_FAIL (generic failure)",
    0x8001010E: "E_WRONG_THREAD (called from wrong thread)",
    0x8007001F: "E_GEN_FAILURE (device not functioning — check cable/USB/power)",
    0x800700AA: "E_BUSY (camera already in use)",
    0x8000000A: "E_PENDING (no data available yet)",
    0x8001011F: "E_TIMEOUT (operation timed out)",
    0x80072743: "E_UNREACH (network unreachable — check IP/firewall)",
    0x800704C7: "E_CANCELLED (operation cancelled)",
}


def _format_hresult(e: Exception) -> str:
    hr: int | None = getattr(e, "hr", None)
    if hr is None:
        return str(e)
    hr_unsigned = hr & 0xFFFFFFFF
    name = _HRESULT_NAMES.get(hr_unsigned, f"0x{hr_unsigned:08X}")
    return f"HRESULT {name}"


def _get_amcam():
    global _amcam
    if _amcam is not None:
        return _amcam

    from camera.sdk_loaders.amscope_sdk_loader import load_amscope_sdk  # pylint: disable=import-outside-toplevel

    _amcam = load_amscope_sdk()
    if _amcam is not None:
        info("Amscope SDK loaded successfully")
        return _amcam

    info("Attempting fallback to direct import...")
    try:
        import amcam as amcam_module  # pylint: disable=import-outside-toplevel
        _amcam = amcam_module
        info("Amscope SDK loaded via direct import")
    except ImportError as e:
        error(f"Direct import also failed: {e}")

    return _amcam


class AmscopeCamera(BaseCamera):
    """
    Amscope camera implementation using the amcam SDK.

    Includes integrated settings management with Amscope-specific
    settings like fan control, TEC, low noise mode, etc.
    """

    def __init__(self, model: str):
        super().__init__(model=model)

        self._hcam = None
        self._frame_buffer: ctypes.Array | None = None
        self._still_buffer_a: ctypes.Array | None = None
        self._still_buffer_b: ctypes.Array | None = None
        self._still_buffer_active: int = 0
        self._still_buffer_width: int = 0
        self._still_buffer_height: int = 0
        self._still_buffer_lock = threading.Lock()
        self._pending_still_resolution_index: int = 0
        self._dfc_completion_callback: Callable[[], None] | None = None

    @property
    def settings(self) -> AmscopeSettings:
        if self._settings is None:
            raise RuntimeError("Settings not initialized. Call initialize_settings() first.")
        return self._settings

    def _get_settings_class(self) -> type[AmscopeSettings]:
        return AmscopeSettings

    # -------------------------
    # Event Constants
    # -------------------------

    @classmethod
    def get_event_constants(cls):
        amcam = _get_amcam()
        return SimpleNamespace(
            IMAGE=amcam.AMCAM_EVENT_IMAGE,
            EXPOSURE=amcam.AMCAM_EVENT_EXPOSURE,
            TEMPTINT=amcam.AMCAM_EVENT_TEMPTINT,
            STILLIMAGE=amcam.AMCAM_EVENT_STILLIMAGE,
            ERROR=amcam.AMCAM_EVENT_ERROR,
            DISCONNECTED=amcam.AMCAM_EVENT_DISCONNECTED
        )

    @property
    def EVENT_IMAGE(self):
        return _get_amcam().AMCAM_EVENT_IMAGE

    @property
    def EVENT_EXPOSURE(self):
        return _get_amcam().AMCAM_EVENT_EXPOSURE

    @property
    def EVENT_TEMPTINT(self):
        return _get_amcam().AMCAM_EVENT_TEMPTINT

    @property
    def EVENT_STILLIMAGE(self):
        return _get_amcam().AMCAM_EVENT_STILLIMAGE

    @property
    def EVENT_ERROR(self):
        return _get_amcam().AMCAM_EVENT_ERROR

    @property
    def EVENT_DISCONNECTED(self):
        return _get_amcam().AMCAM_EVENT_DISCONNECTED

    @property
    def handle(self):
        return self._hcam

    # -------------------------
    # Camera Control
    # -------------------------

    def open(self, camera_id: str) -> bool:
        amcam = _get_amcam()
        try:
            self._hcam = amcam.Amcam.Open(camera_id)
            if self._hcam:
                self._hcam.put_Option(amcam.AMCAM_OPTION_BYTEORDER, 0)
                self.initialize_settings()
                self._is_open = True
                return True
            return False
        except Exception as e:
            error(f"Failed to open camera: {_format_hresult(e)}")
            return False

    def close(self):
        if self._hcam:
            self._hcam.Close()
            self._hcam = None
        self._is_open = False
        self._callback = None
        self._callback_context = None
        self._frame_buffer = None
        self._still_buffer_a = None
        self._still_buffer_b = None
        self._still_buffer_active = 0
        self._still_buffer_width = 0
        self._still_buffer_height = 0
        if self._settings is not None:
            self._settings._histogram_enabled = False
            self._settings._preview_histogram = None
            self._settings._still_histogram = None

    def _reallocate_frame_buffer(self):
        try:
            width, height = self._hcam.get_Size()
            buffer_size = self.calculate_buffer_size(width, height, 24)
            self._frame_buffer = ctypes.create_string_buffer(buffer_size)
            info(f"Reallocated frame buffer: {width}x{height}, size={buffer_size}")
        except Exception as e:
            error(f"Failed to reallocate frame buffer: {e}")

    def start_capture(self, callback: Callable, context: Any) -> bool:
        if not self._hcam:
            return False

        amcam = _get_amcam()
        try:
            max_size = 0
            try:
                for res in self.settings.get_still_resolutions():
                    s = amcam.TDIBWIDTHBYTES(res.width * 24) * res.height
                    if s > max_size:
                        max_size = s
            except Exception:
                pass

            if max_size == 0:
                try:
                    fw, fh = self._hcam.get_FinalSize()
                except Exception:
                    _, fw, fh = self.get_current_preview_resolution()
                max_size = amcam.TDIBWIDTHBYTES(fw * 24) * fh

            self._frame_buffer = ctypes.create_string_buffer(max_size)
            self._callback = callback
            self._callback_context = context
            self._hcam.StartPullModeWithCallback(self._event_callback_wrapper, self)
            return True
        except Exception as e:
            error(f"Failed to start capture: {_format_hresult(e)}")
            return False

    def stop_capture(self):
        if self._hcam:
            try:
                self._hcam.Stop()
            except Exception:
                pass

    def get_frame_buffer(self) -> tuple[ctypes.Array, int, int] | None:
        """
        Return the camera's internal preview frame buffer and its dimensions.

        Populated by ``_event_callback_wrapper`` on every ``EVENT_IMAGE`` via
        ``PullImageV4``.  The frame is already pulled by the time the manager's
        event handler runs, so callers should read from this buffer rather than
        calling ``pull_image`` / ``WaitImageV4`` which would block waiting for
        a second pull on an already-drained frame.

        Returns:
            ``(buffer, width, height)`` if streaming has started, or ``None``
            if the buffer has not been allocated yet.
        """
        if self._frame_buffer is None:
            return None
        width, height = self.settings.get_output_dimensions()
        if width == 0 or height == 0:
            return None
        return self._frame_buffer, width, height

    def pull_image(self, buffer: ctypes.Array, bits_per_pixel: int = 24, timeout_ms: int = 1000) -> bool:
        if not self._hcam:
            error("Cannot pull image: camera handle is None")
            return False

        amcam = _get_amcam()
        try:
            self._hcam.WaitImageV4(timeout_ms, buffer, 0, bits_per_pixel, 0, None)
            return True
        except amcam.HRESULTException as e:
            error(f"Failed to pull image: {_format_hresult(e)}")
            return False

    def snap_image(self, resolution_index: int = 0) -> bool:
        if not self._hcam:
            return False
        try:
            self._hcam.Snap(resolution_index)
            self._pending_still_resolution_index = resolution_index
            return True
        except Exception as e:
            error(f"snap_image failed: {_format_hresult(e)}")
            return False

    # -------------------------
    # Resolution Management
    # -------------------------

    def get_preview_resolutions(self) -> list[CameraResolution]:
        return self.settings.get_preview_resolutions()

    def get_current_preview_resolution(self) -> tuple[int, int, int]:
        return self.settings.get_current_preview_resolution()

    def get_still_resolutions(self) -> list[CameraResolution]:
        return self.settings.get_still_resolutions()

    # -------------------------
    # Image Capture and Saving
    # -------------------------

    def _on_histogram(self, aHist: int, nFlag: int, is_still: bool) -> None:
        try:
            bins = 1 << (nFlag & 0x0F)
            channels = 1 if (nFlag & 0x8000) else 3
            total_bins = bins * channels
            arr_type = ctypes.c_float * total_bins
            arr = arr_type.from_address(aHist)
            raw = np.ctypeslib.as_array(arr).astype(np.float64).reshape(channels, bins)
            totals = raw.sum(axis=1, keepdims=True)
            histogram = np.where(totals > 0, raw / totals, raw)
            if is_still:
                self.settings._still_histogram = histogram
            else:
                self.settings._preview_histogram = histogram
        except Exception:
            exception("Failed to process histogram callback")

    def _request_histogram(self, is_still: bool) -> None:
        if not self._hcam:
            return
        try:
            self._hcam.GetHistogram(
                lambda aHist, nFlag: self._on_histogram(aHist, nFlag, is_still),
                None,
            )
        except Exception:
            exception("Failed to request histogram")

    def get_still_buffer(self) -> tuple[ctypes.Array, int, int] | None:
        with self._still_buffer_lock:
            buf = self._still_buffer_a if self._still_buffer_active == 0 else self._still_buffer_b
            if buf is None or self._still_buffer_width == 0:
                return None
            return buf, self._still_buffer_width, self._still_buffer_height

    def pull_still_image(self, buffer: ctypes.Array, bits_per_pixel: int = 24) -> tuple[bool, int, int]:
        if not self._hcam:
            return False, 0, 0

        amcam = _get_amcam()
        try:
            w, h = self._hcam.get_StillResolution(0)
            self._hcam.PullStillImageV2(buffer, bits_per_pixel, None)
            return True, w, h
        except amcam.HRESULTException:
            return False, 0, 0

    def capture_still(
        self,
        resolution_index: int | None = None,
        timeout_ms: int = 5000,
        on_captured: Callable[[], None] | None = None,
        on_complete: Callable[[bool, np.ndarray | None], None] | None = None,
    ) -> bool:
        """
        Capture a still image and return it as a numpy array without saving.

        Blocks only until the hardware has finished exposing and the raw pixel
        data has been pulled, then spawns a background thread to convert the
        buffer and call on_complete.

        Args:
            resolution_index:  Still-resolution index. Defaults to settings value.
            timeout_ms:        Maximum time to wait for the frame (milliseconds).
            on_captured:       Zero-argument callback fired as soon as the raw
                               frame is pulled — use this to unblock automation.
            on_complete:       ``(success: bool, image: np.ndarray | None) -> None``
                               fired once conversion is done.

        Returns:
            True if the snap and pull succeeded.
            False if the camera is not open, snap fails, or times out.
        """
        if not self._hcam:
            error("Camera not open")
            return False

        if resolution_index is None:
            resolution_index = self.settings.get_still_resolution_index()

        amcam = _get_amcam()
        try:
            t_start = time.perf_counter()

            width, height = self._hcam.get_StillResolution(resolution_index)
            buffer_size = amcam.TDIBWIDTHBYTES(width * 24) * height
            pData = bytearray(buffer_size)

            still_ready = threading.Event()
            capture_success: dict[str, Any] = {"success": False, "width": 0, "height": 0}

            original_callback = self._callback
            original_context = self._callback_context

            def still_callback(event: int, ctx: Any) -> None:
                if event == self.EVENT_STILLIMAGE:
                    with self._still_buffer_lock:
                        read_buf = (
                            self._still_buffer_a
                            if self._still_buffer_active == 0
                            else self._still_buffer_b
                        )
                        w = self._still_buffer_width
                        h = self._still_buffer_height

                    if read_buf is not None and w > 0:
                        try:
                            n = min(len(pData), len(read_buf))
                            pData[:n] = read_buf[:n]
                            capture_success.update(success=True, width=w, height=h)
                        except Exception as copy_err:
                            error(f"Failed to copy still buffer: {copy_err}")
                            capture_success["success"] = False
                    else:
                        error("Still buffer not yet populated; cannot capture still")
                        capture_success["success"] = False
                    still_ready.set()

                if original_callback:
                    original_callback(event, original_context)

            self._callback = still_callback
            self._callback_context = None

            if not self.snap_image(resolution_index):
                error("Failed to trigger still capture")
                self._callback = original_callback
                self._callback_context = original_context
                return False

            if not still_ready.wait(timeout_ms / 1000.0):
                error(f"Still capture timed out after {timeout_ms}ms")
                self._callback = original_callback
                self._callback_context = original_context
                return False

            self._callback = original_callback
            self._callback_context = original_context

            if not capture_success["success"]:
                error("Failed to pull still image")
                return False

            t_captured = time.perf_counter()
            snap_ms = (t_captured - t_start) * 1000
            debug(f"Still capture (snap + pull): {snap_ms:.1f} ms")

            if on_captured is not None:
                try:
                    on_captured()
                except Exception as cb_err:
                    exception(f"Error in on_captured callback: {cb_err}")

            def _process() -> None:
                nonlocal pData
                t_proc_start = time.perf_counter()
                image: np.ndarray | None = None
                try:
                    w = capture_success["width"]
                    h = capture_success["height"]
                    stride = amcam.TDIBWIDTHBYTES(w * 24)
                    image = (
                        np.frombuffer(pData, dtype=np.uint8)
                        .reshape((h, stride))[:, : w * 3]
                        .reshape((h, w, 3))
                        .copy()
                    )
                    del pData
                    process_ms = (time.perf_counter() - t_proc_start) * 1000
                    total_ms = (time.perf_counter() - t_start) * 1000
                    debug(
                        f"Still capture processing (numpy): {process_ms:.1f} ms | "
                        f"Total (snap={snap_ms:.1f} ms, "
                        f"process={process_ms:.1f} ms, "
                        f"total={total_ms:.1f} ms)"
                    )
                except Exception:
                    exception("Failed to process still image buffer")
                finally:
                    gc.collect()

                if on_complete is not None:
                    try:
                        on_complete(image is not None, image)
                    except Exception as cb_err:
                        exception(f"Error in on_complete callback: {cb_err}")

            threading.Thread(target=_process, daemon=True, name="CameraStillProcess").start()
            return True

        except Exception:
            exception("Failed to capture still image")
            return False

    def capture_and_save_still(
        self,
        filepath: Path,
        resolution_index: int | None = None,
        additional_metadata: dict[str, Any] | None = None,
        timeout_ms: int = 5000,
        on_captured: Callable[[], None] | None = None,
        on_image: Callable[[np.ndarray], None] | None = None,
        on_complete: Callable[[bool], None] | None = None,
    ) -> bool:
        """
        Capture a still image and save it with metadata.

        Blocks only until the camera hardware has finished exposing and the raw
        pixel data has been pulled from the SDK. Then calls ``on_captured()`` so
        the automation pipeline can continue, and spawns a background thread to
        convert and save the file before calling ``on_complete(success)``.

        Args:
            filepath:            Destination path for the saved image.
            resolution_index:    Still-resolution index.
            additional_metadata: Extra key/value pairs embedded in the file.
            timeout_ms:          Maximum time to wait for the camera to deliver
                                 the still frame (milliseconds).
            on_captured:         Zero-argument callback fired as soon as the raw
                                 frame has been pulled from the SDK.
            on_image:            ``(image: np.ndarray) -> None`` fired on the
                                 background save thread after conversion but
                                 before the file is written. The array is the
                                 same object passed to ``save_image``; do not
                                 retain a reference beyond the callback if
                                 memory usage matters.
            on_complete:         ``(success: bool) -> None`` fired once the file
                                 has been written (or the save failed).

        Returns:
            True if the snap and pull succeeded. Note: a True return does not
            mean the file has been saved yet — saving happens asynchronously.
            Returns False if the camera is not open, snap fails, or times out.
        """
        if not self._hcam:
            error("Camera not open")
            return False

        if resolution_index is None:
            resolution_index = self.settings.get_still_resolution_index()
            debug(
                f"Still resolution index resolved from settings: {resolution_index} ({self.settings.still_resolution})")

        amcam = _get_amcam()
        try:
            t_start = time.perf_counter()

            width, height = self._hcam.get_StillResolution(resolution_index)
            buffer_size = amcam.TDIBWIDTHBYTES(width * 24) * height
            pData = bytearray(buffer_size)

            still_ready = threading.Event()
            capture_success: dict[str, Any] = {
                'success': False,
                'width': 0,
                'height': 0,
            }

            original_callback = self._callback
            original_context = self._callback_context

            def still_callback(event: int, ctx: Any) -> None:
                if event == self.EVENT_STILLIMAGE:
                    with self._still_buffer_lock:
                        if self._still_buffer_active == 0:
                            read_buf = self._still_buffer_a
                        else:
                            read_buf = self._still_buffer_b
                        w = self._still_buffer_width
                        h = self._still_buffer_height

                    if read_buf is not None and w > 0:
                        try:
                            n = min(len(pData), len(read_buf))
                            pData[:n] = read_buf[:n]
                            capture_success['success'] = True
                            capture_success['width'] = w
                            capture_success['height'] = h
                        except Exception as copy_err:
                            error(f"Failed to copy still buffer: {copy_err}")
                            capture_success['success'] = False
                    else:
                        error("Still buffer not yet populated; cannot capture still")
                        capture_success['success'] = False
                    still_ready.set()

                if original_callback:
                    original_callback(event, original_context)

            self._callback = still_callback
            self._callback_context = None

            if not self.snap_image(resolution_index):
                error("Failed to trigger still capture")
                self._callback = original_callback
                self._callback_context = original_context
                return False

            if not still_ready.wait(timeout_ms / 1000.0):
                error(f"Still capture timed out after {timeout_ms}ms")
                self._callback = original_callback
                self._callback_context = original_context
                return False

            self._callback = original_callback
            self._callback_context = original_context

            if not capture_success['success']:
                error("Failed to pull still image")
                return False

            t_captured = time.perf_counter()
            snap_ms = (t_captured - t_start) * 1000
            debug(f"Still capture (snap + pull): {snap_ms:.1f} ms")

            if on_captured is not None:
                try:
                    on_captured()
                except Exception as cb_err:
                    exception(f"Error in on_captured callback: {cb_err}")

            def _process_and_save() -> None:
                nonlocal pData

                t_proc_start = time.perf_counter()

                try:
                    w = capture_success['width']
                    h = capture_success['height']
                    stride = amcam.TDIBWIDTHBYTES(w * 24)
                    image_data = (
                        np.frombuffer(pData, dtype=np.uint8)
                        .reshape((h, stride))[:, : w * 3]
                        .reshape((h, w, 3))
                        .copy()
                    )
                    del pData

                    t_processed = time.perf_counter()
                    process_ms = (t_processed - t_proc_start) * 1000
                    debug(f"Still image processing (numpy): {process_ms:.1f} ms")

                    if on_image is not None:
                        try:
                            on_image(image_data)
                        except Exception as cb_err:
                            exception(f"Error in on_image callback: {cb_err}")

                    save_ok = self.save_image(image_data, filepath, additional_metadata)

                    t_saved = time.perf_counter()
                    save_ms = (t_saved - t_processed) * 1000
                    total_ms = (t_saved - t_start) * 1000
                    debug(
                        f"Still image save: {save_ms:.1f} ms | "
                        f"Total (snap={snap_ms:.1f} ms, "
                        f"process={process_ms:.1f} ms, "
                        f"save={save_ms:.1f} ms, "
                        f"total={total_ms:.1f} ms): {filepath.name}"
                    )

                    del image_data
                    gc.collect()

                    if save_ok:
                        info(f"Still image captured and saved: {filepath}")
                    else:
                        error(f"Failed to save still image: {filepath}")

                except Exception:
                    save_ok = False
                    exception(f"Failed to process/save still image: {filepath}")

                if on_complete is not None:
                    try:
                        on_complete(save_ok)
                    except Exception as cb_err:
                        exception(f"Error in on_complete callback: {cb_err}")

            threading.Thread(target=_process_and_save, daemon=True, name="CameraStillSave").start()
            return True

        except Exception:
            exception(f"Failed to capture and save still image: {filepath}")
            return False

    def capture_and_save_stream(
        self,
        filepath: Path,
        additional_metadata: dict[str, Any] | None = None,
        on_image: Callable[[np.ndarray], None] | None = None,
    ) -> bool:
        """Capture current frame from live stream and save it."""
        if not self._hcam or not self._is_open:
            error("Camera not in capture mode")
            return False

        if self._frame_buffer is None:
            error("No frame buffer available")
            return False

        try:
            _, width, height = self.get_current_preview_resolution()
            amcam = _get_amcam()
            stride = amcam.TDIBWIDTHBYTES(width * 24)
            image_data = np.frombuffer(self._frame_buffer, dtype=np.uint8).reshape(
                (height, stride))[:, :width*3].reshape((height, width, 3)).copy()
            image_data = image_data[:, :, ::-1].copy()

            if on_image is not None:
                try:
                    on_image(image_data)
                except Exception as cb_err:
                    exception(f"Error in on_image callback: {cb_err}")

            success = self.save_image(image_data, filepath, additional_metadata)
            del image_data
            gc.collect()

            if success:
                info(f"Stream frame captured and saved: {filepath}")
            else:
                error(f"Failed to save stream frame: {filepath}")

            return success

        except Exception:
            exception(f"Failed to capture and save stream frame: {filepath}")
            return False

    # -------------------------
    # Utility Methods
    # -------------------------

    @staticmethod
    def calculate_buffer_size(width: int, height: int, bits_per_pixel: int = 24) -> int:
        return _get_amcam().TDIBWIDTHBYTES(width * bits_per_pixel) * height

    @staticmethod
    def calculate_stride(width: int, bits_per_pixel: int = 24) -> int:
        return _get_amcam().TDIBWIDTHBYTES(width * bits_per_pixel)

    @staticmethod
    def enable_gige(callback: Callable | None = None, context: Any = None):
        _get_amcam().Amcam.GigeEnable(callback, context)

    def _event_callback_wrapper(self, event: int, context: Any):
        amcam = _get_amcam()

        if event == self.EVENT_IMAGE and self._frame_buffer is not None:
            try:
                self._hcam.PullImageV4(self._frame_buffer, 0, 24, 0, None)
            except Exception:
                pass
            if self._settings is not None and self._settings._histogram_enabled:
                self._request_histogram(is_still=False)
        elif event == self.EVENT_STILLIMAGE:
            try:
                still_resolutions = self.settings.get_still_resolutions()
                idx = self._pending_still_resolution_index
                if still_resolutions and idx < len(still_resolutions):
                    raw_w = still_resolutions[idx].width
                    raw_h = still_resolutions[idx].height
                else:
                    raw_w, raw_h = self._hcam.get_Size()

                rotate = self.settings.rotate if self.settings else 0
                if rotate in (90, 270):
                    sw, sh = raw_h, raw_w
                else:
                    sw, sh = raw_w, raw_h

                with self._still_buffer_lock:
                    required = amcam.TDIBWIDTHBYTES(sw * 24) * sh

                    if self._still_buffer_a is None or len(self._still_buffer_a) < required:
                        self._still_buffer_a = ctypes.create_string_buffer(required)
                        self._still_buffer_b = ctypes.create_string_buffer(required)
                        debug(f"Still double-buffer allocated: {sw}x{sh} ({required} bytes each)")

                    next_active = 1 - self._still_buffer_active
                    write_buf = self._still_buffer_b if next_active == 1 else self._still_buffer_a

                    self._hcam.PullImageV4(write_buf, 1, 24, 0, None)

                    self._still_buffer_active = next_active
                    self._still_buffer_width = sw
                    self._still_buffer_height = sh
                    debug(f"Still frame pulled into buffer {next_active}: {sw}x{sh}")
            except Exception as e:
                error(f"Failed to pull still frame into still buffer: {_format_hresult(e)}")
            if self._settings is not None and self._settings._histogram_enabled:
                self._request_histogram(is_still=True)
        elif event == amcam.AMCAM_EVENT_DFC:
            debug("DFC event received")
            if self._dfc_completion_callback is not None:
                self._dfc_completion_callback()

        if self._callback:
            self._callback(event, self._callback_context)