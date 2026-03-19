"""
Amscope camera implementation using the amcam SDK.
Now with integrated settings management.
"""

from __future__ import annotations

from typing import Callable, Any
from types import SimpleNamespace
from pathlib import Path
import ctypes
import numpy as np
import threading
import gc
import time

from camera.cameras.base_camera import BaseCamera, CameraResolution
from common.logger import info, debug, error, exception, warning
from camera.settings.amscope_settings import AmscopeSettings

# Module-level reference to the loaded SDK
_amcam = None


class AmscopeCamera(BaseCamera):
    """
    Amscope camera implementation using the amcam SDK.
    
    Now includes integrated settings management with Amscope-specific
    settings like fan control, TEC, low noise mode, etc.
    
    The SDK must be loaded before using this class:
        AmscopeCamera.ensure_sdk_loaded()
    
    Or it will be loaded automatically on first use.
    """
    
    # Class-level flag to track SDK loading
    _sdk_loaded = False
    
    
    def __init__(self, model: str):
        """
        Initialize Amscope camera.
        
        Args:
            model: Camera model name (default "Amscope")
        """
        super().__init__(model=model)

        # Set Settings class
        self._settings_class = AmscopeSettings
        
        self._hcam = None  # Will be amcam.Amcam after SDK loads

        # Ensure SDK is loaded before instantiating
        if not AmscopeCamera._sdk_loaded:
            AmscopeCamera.ensure_sdk_loaded()

        self._frame_buffer: ctypes.Array | None = None
        self._still_buffer: ctypes.Array | None = None
        self._still_buffer_width: int = 0
        self._still_buffer_height: int = 0
        self._pending_still_resolution_index: int = 0
        self._dfc_completion_callback = None  # Callback for DFC completion
    
    def _get_settings_class(self):
        """
        Get the settings class for Amscope cameras.
        
        Returns:
            AmscopeSettings class
        """
        from camera.settings.amscope_settings import AmscopeSettings
        return AmscopeSettings
    
    @property
    def settings(self) -> AmscopeSettings:
        """
        Get settings with proper type hint for Amscope.
        
        Returns:
            AmscopeSettings object
        """
        if self._settings is None:
            raise RuntimeError("Settings not initialized. Call initialize_settings() first.")
        return self._settings
    
    # -------------------------
    # SDK Management
    # -------------------------
    
    @classmethod
    def ensure_sdk_loaded(cls, sdk_path: Path | None = None) -> bool:
        """
        Ensure the Amscope SDK is loaded and ready to use.
        
        Args:
            sdk_path: Optional path to SDK base directory.
                     If None, auto-detects from project structure.
        
        Returns:
            True if SDK loaded successfully, False otherwise
        """
        global _amcam
        
        if cls._sdk_loaded and _amcam is not None:
            return True
        
        try:
            from camera.sdk_loaders.amscope_sdk_loader import AmscopeSdkLoader
            
            loader = AmscopeSdkLoader(sdk_path)
            _amcam = loader.load()
            
            cls._sdk_loaded = True
            info("Amscope SDK loaded successfully")
            return True
            
        except Exception as e:
            error(f"Failed to load Amscope SDK: {e}")
            info("Attempting fallback to direct import...")
            
            try:
                # Fallback to direct import if loader fails
                import amcam as amcam_module
                _amcam = amcam_module
                cls._sdk_loaded = True
                info("Amscope SDK loaded via direct import")
                return True
            except ImportError as ie:
                error(f"Direct import also failed: {ie}")
                return False
    
    @staticmethod
    def _get_sdk():
        """Get the loaded SDK module"""
        global _amcam
        if _amcam is None:
            raise RuntimeError(
                "Amscope SDK not loaded. Call AmscopeCamera.ensure_sdk_loaded() first."
            )
        return _amcam
    
    @classmethod
    def _get_sdk_static(cls):
        """Static version of _get_sdk for class methods"""
        return cls._get_sdk()
    
    # -------------------------
    # Event Constants
    # -------------------------
    
    @classmethod
    def get_event_constants(cls):
        """Get event constants as a namespace object."""
        amcam = cls._get_sdk_static()
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
        return self._get_sdk().AMCAM_EVENT_IMAGE
    
    @property
    def EVENT_EXPOSURE(self):
        return self._get_sdk().AMCAM_EVENT_EXPOSURE
    
    @property
    def EVENT_TEMPTINT(self):
        return self._get_sdk().AMCAM_EVENT_TEMPTINT
    
    @property
    def EVENT_STILLIMAGE(self):
        return self._get_sdk().AMCAM_EVENT_STILLIMAGE
    
    @property
    def EVENT_ERROR(self):
        return self._get_sdk().AMCAM_EVENT_ERROR
    
    @property
    def EVENT_DISCONNECTED(self):
        return self._get_sdk().AMCAM_EVENT_DISCONNECTED
    
    @property
    def handle(self):
        """Get the underlying amcam handle"""
        return self._hcam
    
    # -------------------------
    # Camera Control
    # -------------------------
    
    def open(self, camera_id: str) -> bool:
        """Open connection to Amscope camera"""
        amcam = self._get_sdk()
        try:
            self._hcam = amcam.Amcam.Open(camera_id)
            if self._hcam:
                # Set RGB byte order for Qt compatibility
                self._hcam.put_Option(_amcam.AMCAM_OPTION_BYTEORDER, 0)
                # Initialize settings
                self.initialize_settings()
                self._is_open = True
                return True
            return False
        except self._get_sdk().HRESULTException:
            return False
    
    def close(self):
        """Close camera connection"""
        if self._hcam:
            self._hcam.Close()
            self._hcam = None
        self._is_open = False
        self._callback = None
        self._callback_context = None
        self._frame_buffer = None
        self._still_buffer = None
        self._still_buffer_width = 0
        self._still_buffer_height = 0
        if self._settings is not None:
            self._settings._histogram_enabled = False
            self._settings._preview_histogram = None
            self._settings._still_histogram = None
    
    def _reallocate_frame_buffer(self):
        """Reallocate frame buffer based on current resolution."""
        try:
            width, height = self._hcam.get_Size()
            buffer_size = self.calculate_buffer_size(width, height, 24)
            self._frame_buffer = ctypes.create_string_buffer(buffer_size)
            info(f"Reallocated frame buffer: {width}x{height}, size={buffer_size}")
        except Exception as e:
            error(f"Failed to reallocate frame buffer: {e}")
    
    def start_capture(self, callback: Callable, context: Any) -> bool:
        """Start capturing frames with callback"""
        if not self._hcam:
            return False
        
        amcam = self._get_sdk()
        try:
            # Get current resolution to allocate preview frame buffer
            res_index, width, height = self.get_current_preview_resolution()
            # Allocate preview frame buffer using the post-rotation output size.
            # get_FinalSize() accounts for AMCAM_OPTION_ROTATE so the buffer
            # is correctly sized for 90/270° rotations where width and height
            # are transposed relative to the raw sensor resolution.
            try:
                fw, fh = self._hcam.get_FinalSize()
            except Exception:
                _, fw, fh = self.get_current_preview_resolution()
            buffer_size = amcam.TDIBWIDTHBYTES(fw * 24) * fh
            self._frame_buffer = ctypes.create_string_buffer(buffer_size)
            
            self._callback = callback
            self._callback_context = context
            self._hcam.StartPullModeWithCallback(self._event_callback_wrapper, self)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def stop_capture(self):
        """Stop capturing frames"""
        if self._hcam:
            try:
                self._hcam.Stop()
            except:
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
        """
        Pull the latest image into buffer (expects ctypes.create_string_buffer)
        
        Args:
            buffer: ctypes buffer to receive image data
            bits_per_pixel: Bits per pixel (typically 24)
            timeout_ms: Timeout in milliseconds to wait for frame
            
        Returns:
            True if successful, False otherwise
        """
        if not self._hcam:
            error("Cannot pull image: camera handle is None")
            return False
        
        amcam = self._get_sdk()
        try:
            # Use WaitImageV4 to wait for a frame (bStill=0 for video stream)
            # This is more reliable than PullImageV4 which may fail if no frame is ready
            self._hcam.WaitImageV4(timeout_ms, buffer, 0, bits_per_pixel, 0, None)
            return True
        except amcam.HRESULTException as e:
            # If timeout or no frame available, log the error
            error(f"Failed to pull image: {e}")
            return False
    
    def snap_image(self, resolution_index: int = 0) -> bool:
        """Capture a still image at specified resolution"""
        if not self._hcam:
            return False

        try:
            self._hcam.Snap(resolution_index)
            self._pending_still_resolution_index = resolution_index
            return True
        except:
            return False
    
    # -------------------------
    # Resolution Management
    # -------------------------
    
    def get_preview_resolutions(self) -> list[CameraResolution]:
        """Get available preview resolutions"""
        return self.settings.get_preview_resolutions()

    def get_current_preview_resolution(self) -> tuple[int, int, int]:
        """Get current resolution index, width, and height"""
        return self.settings.get_current_preview_resolution()

    def get_still_resolutions(self) -> list[CameraResolution]:
        """Get available still image resolutions"""
        return self.settings.get_still_resolutions()

    # -------------------------
    # Image Capture and Saving
    # -------------------------
        """
        Submit a single histogram request to the SDK.

        We call Amcam_GetHistogramV2 directly rather than through the SDK's
        GetHistogram wrapper because the wrapper drops nFlag before forwarding
        to our callback, making it impossible to know the array size.  By going
        directly we receive (aHist: c_void_p, nFlag: c_uint) and can reconstruct
        the typed pointer ourselves.

        A closure is used so that ``self`` and ``is_still`` are both available
        inside the raw ctypes callback without relying on the ctx py_object
        (which the SDK passes through but we don't need for routing).

        Args:
            is_still: True to tag this request as belonging to a still capture,
                      False for a preview frame.
        """
        if not self._hcam:
            return
        amcam = self._get_sdk()

        # Capture self and is_still in a closure so the raw callback can route
        # back to the instance method.
        instance = self

        def _cb(aHist: int, nFlag: int, ctx: object) -> None:
            instance._on_histogram(aHist, nFlag, is_still)

        try:
            # Keep a strong reference on self so the callback is not GC'd
            # before the SDK fires it.
            self._histogram_callback = amcam.Amcam._Amcam__HISTOGRAM_CALLBACK(_cb)
            amcam.Amcam._Amcam__lib.Amcam_GetHistogramV2(
                self._hcam._Amcam__h,
                self._histogram_callback,
                ctypes.py_object(None),
            )
        except Exception as e:
            error(f"Failed to request histogram: {e}")

    def _on_histogram(self, aHist: int, nFlag: int, is_still: bool) -> None:
        """
        Process a raw histogram delivered by Amcam_GetHistogramV2.

        Args:
            aHist:    Raw pointer address (int) to the histogram float array.
            nFlag:    SDK flag encoding bit-depth and channel count.
            is_still: True if triggered by a still capture, False for preview.
        """
        try:
            bins     = 1 << (nFlag & 0x0F)          # e.g. 256 or 65536
            channels = 1 if (nFlag & 0x8000) else 3

            # Cast the void pointer to a typed float array of the correct length
            total_bins = bins * channels
            arr_type = ctypes.c_float * total_bins
            arr = arr_type.from_address(aHist)
            raw = np.ctypeslib.as_array(arr).astype(np.float64).reshape(channels, bins)

            # Normalise each channel independently
            totals = raw.sum(axis=1, keepdims=True)
            histogram = np.where(totals > 0, raw / totals, raw)

            if is_still:
                self.settings._still_histogram = histogram
            else:
                self.settings._preview_histogram = histogram
        except Exception:
            exception("Failed to process histogram callback")

    def get_still_buffer(self) -> tuple[ctypes.Array, int, int] | None:
        """
        Return the camera's internal still frame buffer and its dimensions.

        Returns:
            ``(buffer, width, height)`` if a still has been captured since
            streaming started, or ``None`` if the buffer has not been
            allocated yet.
        """
        if self._still_buffer is None or self._still_buffer_width == 0 or self._still_buffer_height == 0:
            return None
        return self._still_buffer, self._still_buffer_width, self._still_buffer_height

    def pull_still_image(self, buffer: ctypes.Array, bits_per_pixel: int = 24) -> tuple[bool, int, int]:
        """
        Pull a still image into buffer
        
        Args:
            buffer: Buffer to receive image data (ctypes.create_string_buffer)
            bits_per_pixel: Bits per pixel (typically 24)
            
        Returns:
            Tuple of (success, width, height)
        """
        if not self._hcam:
            return False, 0, 0
        
        amcam = self._get_sdk()
        try:
            # Get still resolution to return dimensions
            w, h = self._hcam.get_StillResolution(0)
            # Use PullStillImageV2 which works with ctypes.create_string_buffer
            self._hcam.PullStillImageV2(buffer, bits_per_pixel, None)
            return True, w, h
        except amcam.HRESULTException:
            return False, 0, 0
    
    # -------------------------
    # Image Capture and Saving
    # -------------------------
    
    def capture_and_save_still(
        self,
        filepath: Path,
        resolution_index: int | None = None,
        additional_metadata: dict[str, Any] | None = None,
        timeout_ms: int = 5000,
        on_captured: Callable[[], None] | None = None,
        on_complete: Callable[[bool], None] | None = None,
    ) -> bool:
        """
        Capture a still image and save it with metadata.

        The method blocks only until the camera hardware has finished exposing
        and the raw pixel data has been pulled from the SDK.  As soon as that
        is done it:

        1. Calls ``on_captured()`` (if provided) so that the caller (e.g. the
           automation stage-mover) can continue to the next position immediately.
        2. Spawns a lightweight daemon thread that converts the raw buffer to a
           numpy array, saves the file, and calls ``on_complete(success)`` when
           finished.

        Timing information is written to the debug log at each phase:
        - how long the hardware snap/pull took
        - how long numpy conversion took
        - how long the file save took
        - total elapsed time

        Args:
            filepath:            Destination path for the saved image.
            resolution_index:    Still-resolution index
            additional_metadata: Extra key/value pairs embedded in the file.
            timeout_ms:          Maximum time to wait for the camera to deliver
                                 the still frame (milliseconds).
            on_captured:         Optional zero-argument callback fired on the
                                 camera thread as soon as the raw frame has been
                                 pulled from the SDK.  Use this to unblock the
                                 automation pipeline.
            on_complete:         Optional callback ``(success: bool) -> None``
                                 fired on the background save thread once the
                                 file has been written (or the save failed).

        Returns:
            True if the snap and pull succeeded (i.e. the image was captured).
            Note: a True return does *not* mean the file has been saved yet —
            saving happens asynchronously after this method returns.
            Returns False if the camera is not open, snap fails, or times out.
        """
        if not self._hcam:
            error("Camera not open")
            return False

        if resolution_index is None:
            resolution_index = self.settings.get_still_resolution_index()
            debug(f"Still resolution index resolved from settings: {resolution_index} ({self.settings.still_resolution})")

        amcam = self._get_sdk()
        try:

            t_start = time.perf_counter()

            # Allocate buffer for still image
            width, height = self._hcam.get_StillResolution(resolution_index)
            buffer_size = amcam.TDIBWIDTHBYTES(width * 24) * height
            pData = bytearray(buffer_size)

            # Threading primitives for snap phase
            still_ready = threading.Event()
            capture_success: dict[str, Any] = {
                'success': False,
                'width': 0,
                'height': 0,
            }

            # Save original callback so we can restore it afterwards
            original_callback = self._callback
            original_context = self._callback_context

            def still_callback(event: int, ctx: Any) -> None:
                if event == self.EVENT_STILLIMAGE:
                    if self._still_buffer is not None and self._still_buffer_width > 0:
                        try:
                            n = min(len(pData), len(self._still_buffer))
                            pData[:n] = self._still_buffer[:n]
                            capture_success['success'] = True
                            capture_success['width'] = self._still_buffer_width
                            capture_success['height'] = self._still_buffer_height
                        except Exception as copy_err:
                            error(f"Failed to copy still buffer: {copy_err}")
                            capture_success['success'] = False
                    else:
                        error("Still buffer not yet populated; cannot capture still")
                        capture_success['success'] = False
                    still_ready.set()

                # Forward to the original callback so live preview keeps working
                if original_callback:
                    original_callback(event, original_context)

            # Temporarily replace callback
            self._callback = still_callback
            self._callback_context = None

            # Trigger the hardware snap
            if not self.snap_image(resolution_index):
                error("Failed to trigger still capture")
                self._callback = original_callback
                self._callback_context = original_context
                return False

            # Block until the SDK delivers the still frame (or we time out)
            if not still_ready.wait(timeout_ms / 1000.0):
                error(f"Still capture timed out after {timeout_ms}ms")
                self._callback = original_callback
                self._callback_context = original_context
                return False

            # Restore original callback
            self._callback = original_callback
            self._callback_context = original_context

            if not capture_success['success']:
                error("Failed to pull still image")
                return False

            t_captured = time.perf_counter()
            snap_ms = (t_captured - t_start) * 1000
            debug(f"Still capture (snap + pull): {snap_ms:.1f} ms")

            # Notify caller that the hardware is free — automation can move now
            if on_captured is not None:
                try:
                    on_captured()
                except Exception as cb_err:
                    exception(f"Error in on_captured callback: {cb_err}")

            # Everything from here (numpy conversion + file save) runs in a
            # daemon thread so this method returns immediately.
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

            save_thread = threading.Thread(
                target=_process_and_save,
                daemon=True,
                name="CameraStillSave",
            )
            save_thread.start()

            # Return True immediately — the image was captured successfully.
            # Saving is in progress on the background thread.
            return True

        except Exception:
            exception(f"Failed to capture and save still image: {filepath}")
            return False
    
    def capture_and_save_stream(
        self,
        filepath: Path,
        additional_metadata: dict[str, Any] | None = None
    ) -> bool:
        """Capture current frame from live stream and save it."""
        if not self._hcam or not self._is_open:
            error("Camera not in capture mode")
            return False
        
        if not hasattr(self, '_frame_buffer') or self._frame_buffer is None:
            error("No frame buffer available")
            return False
        
        try:
            # Get current resolution
            res_index, width, height = self.get_current_preview_resolution()
            
            # Copy from frame buffer
            amcam = self._get_sdk()
            stride = amcam.TDIBWIDTHBYTES(width * 24)
            
            # Create numpy array from buffer
            image_data = np.frombuffer(self._frame_buffer, dtype=np.uint8).reshape((height, stride))[:, :width*3].reshape((height, width, 3)).copy()
            
            # Convert BGR to RGB
            image_data = image_data[:, :, ::-1].copy()
            
            # Save with metadata
            success = self.save_image(image_data, filepath, additional_metadata)
            
            del image_data
            gc.collect()
            
            if success:
                info(f"Stream frame captured and saved: {filepath}")
            else:
                error(f"Failed to save stream frame: {filepath}")
            
            return success
            
        except Exception as e:
            exception(f"Failed to capture and save stream frame: {filepath}")
            return False
    
    # -------------------------
    # Utility Methods
    # -------------------------
    
    @staticmethod
    def calculate_buffer_size(width: int, height: int, bits_per_pixel: int = 24) -> int:
        """Calculate required buffer size for image data"""
        amcam = AmscopeCamera._get_sdk_static()
        return amcam.TDIBWIDTHBYTES(width * bits_per_pixel) * height
    
    @staticmethod
    def calculate_stride(width: int, bits_per_pixel: int = 24) -> int:
        """Calculate image stride (bytes per row)"""
        amcam = AmscopeCamera._get_sdk_static()
        return amcam.TDIBWIDTHBYTES(width * bits_per_pixel)
    
    @classmethod
    def enable_gige(cls, callback: Callable | None = None, context: Any = None):
        """Enable GigE camera support"""
        if not cls._sdk_loaded:
            cls.ensure_sdk_loaded()
        
        amcam = cls._get_sdk_static()
        amcam.Amcam.GigeEnable(callback, context)
    
    def _event_callback_wrapper(self, event: int, context: Any):
        """Internal wrapper for camera events."""
        amcam = self._get_sdk()
        
        # Update frame buffer on IMAGE events
        if event == self.EVENT_IMAGE and hasattr(self, '_frame_buffer') and self._frame_buffer is not None:
            try:
                self._hcam.PullImageV4(self._frame_buffer, 0, 24, 0, None)
            except:
                pass
            # Request a fresh preview histogram for this frame
            if self._settings is not None and self._settings._histogram_enabled:
                self._request_histogram(is_still=False)
        elif event == self.EVENT_STILLIMAGE:
            try:
                # Determine the raw sensor dimensions for the snapped resolution.
                still_resolutions = self.settings.get_still_resolutions()
                idx = self._pending_still_resolution_index
                if still_resolutions and idx < len(still_resolutions):
                    raw_w = still_resolutions[idx].width
                    raw_h = still_resolutions[idx].height
                else:
                    raw_w, raw_h = self._hcam.get_Size()

                # Apply the same rotation swap the SDK applies to the pixel data
                # so that width/height match the actual layout in the buffer.
                rotate = self.settings.rotate if self.settings else 0
                if rotate in (90, 270):
                    sw, sh = raw_h, raw_w
                else:
                    sw, sh = raw_w, raw_h

                required = amcam.TDIBWIDTHBYTES(sw * 24) * sh
                if self._still_buffer is None or len(self._still_buffer) < required:
                    debug(f"Still buffer reallocated: {sw}x{sh}")
                    self._still_buffer = ctypes.create_string_buffer(required)

                self._hcam.PullImageV4(self._still_buffer, 1, 24, 0, None)
                self._still_buffer_width = sw
                self._still_buffer_height = sh
                debug(f"Still frame pulled: {sw}x{sh}")
            except Exception as e:
                error(f"Failed to pull still frame into still buffer: {e}")
            if self._settings is not None and self._settings._histogram_enabled:
                self._request_histogram(is_still=True)
        elif event == amcam.AMCAM_EVENT_DFC:
            # DFC event received - call completion callback if registered
            debug("DFC event received")
            if hasattr(self, '_dfc_completion_callback') and self._dfc_completion_callback:
                self._dfc_completion_callback()
        
        # Call registered callback
        if self._callback:
            self._callback(event, self._callback_context)