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

from camera.cameras.base_camera import BaseCamera, CameraResolution, CameraInfo
from logger import info, debug, error, exception, warning
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

        self._camera_info = None  # Must be set via set_camera_info() before opening
        self._frame_buffer = None
    
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
        self._camera_info = None
        self._frame_buffer = None
    
    def _reallocate_frame_buffer(self):
        """Reallocate frame buffer based on current resolution."""
        try:
            width, height = self._hcam.get_Size()
            buffer_size = self.calculate_buffer_size(width, height, 24)
            self._frame_buffer = bytes(buffer_size)
            info(f"Reallocated frame buffer: {width}x{height}, size={buffer_size}")
        except Exception as e:
            error(f"Failed to reallocate frame buffer: {e}")
    
    def start_capture(self, callback: Callable, context: Any) -> bool:
        """Start capturing frames with callback"""
        if not self._hcam:
            return False
        
        amcam = self._get_sdk()
        try:
            # Get current resolution to allocate frame buffer
            res_index, width, height = self.get_current_resolution()
            
            # Create persistent frame buffer
            buffer_size = amcam.TDIBWIDTHBYTES(width * 24) * height
            self._frame_buffer = bytearray(buffer_size)
            
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
            return True
        except:
            return False
    
    # -------------------------
    # Resolution Management
    # -------------------------
    
    def set_camera_info(self, info: CameraInfo):
        """Set camera information (needed before get_resolutions works)"""
        self._camera_info = info
    
    def get_resolutions(self) -> list[CameraResolution]:
        """Get available preview resolutions"""
        if not self._camera_info or not self._camera_info.model:
            return []
        
        resolutions = []
        for i in range(self._camera_info.model.preview):
            res = self._camera_info.model.res[i]
            resolutions.append(CameraResolution(width=res.width, height=res.height))
        
        return resolutions
    
    def get_current_resolution(self) -> Tuple[int, int, int]:
        """Get current resolution index, width, and height"""
        if not self._hcam or not self._camera_info:
            return 0, 0, 0
        
        res_index = self._hcam.get_eSize()
        res = self._camera_info.model.res[res_index]
        return res_index, res.width, res.height
    
    def set_resolution(self, resolution_index: int) -> bool:
        """Set camera resolution"""
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_eSize(resolution_index)
            return True
        except:
            return False
    
    def supports_still_capture(self) -> bool:
        """Check if camera supports separate still image capture"""
        if not self._camera_info or not self._camera_info.model:
            return False
        
        return self._camera_info.model.still > 0
    
    def get_still_resolutions(self) -> list[CameraResolution]:
        """Get available still image resolutions"""
        if not self._camera_info or not self._camera_info.model:
            return []
        
        resolutions = []
        for i in range(self._camera_info.model.still):
            res = self._camera_info.model.res[i]
            resolutions.append(CameraResolution(width=res.width, height=res.height))
        
        return resolutions
    
    def pull_still_image(self, buffer: ctypes.Array, bits_per_pixel: int = 24) -> Tuple[bool, int, int]:
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
    # Metadata
    # -------------------------
    
    def get_camera_metadata(self) -> dict[str, Any]:
        """Get current camera metadata for image saving"""
        metadata = {
            'model': self.model,
        }
        
        # Get metadata from settings if available
        if self._settings is not None:
            metadata['exposure_time_us'] = self._settings.get_exposure_time()
            metadata['gain_percent'] = self._settings.get_gain()
            metadata['temperature'] = self._settings.temp
            metadata['tint'] = self._settings.tint
        
        # Add serial number if available
        try:
            if self._hcam:
                metadata['serial'] = self._hcam.get_SerialNumber()
        except:
            pass
        
        return metadata
    
    # -------------------------
    # Image Capture and Saving
    # -------------------------
    
    def capture_and_save_still(
        self,
        filepath: Path,
        resolution_index: int = 0,
        additional_metadata: dict[str, Any] | None = None,
        timeout_ms: int = 5000
    ) -> bool:
        """Capture a still image and save it with metadata."""
        if not self._hcam:
            error("Camera not open")
            return False
        
        amcam = self._get_sdk()
        
        try:
            # Allocate buffer for still image
            width, height = self._hcam.get_StillResolution(resolution_index)
            buffer_size = amcam.TDIBWIDTHBYTES(width * 24) * height
            pData = bytes(buffer_size)
            
            # Setup threading for still capture
            still_ready = threading.Event()
            capture_success = {'success': False, 'width': 0, 'height': 0}
            
            # Save original callback
            original_callback = self._callback
            original_context = self._callback_context
            
            def still_callback(event, ctx):
                if event == self.EVENT_STILLIMAGE:
                    # Pull the still image
                    info_struct = amcam.AmcamFrameInfoV3()
                    try:
                        self._hcam.PullImageV3(pData, 1, 24, 0, info_struct)
                        capture_success['success'] = True
                        capture_success['width'] = info_struct.width
                        capture_success['height'] = info_struct.height
                    except Exception as e:
                        error(f"Failed to pull still image: {e}")
                        capture_success['success'] = False
                    still_ready.set()
                
                # Call original callback if exists
                if original_callback:
                    original_callback(event, original_context)
            
            # Temporarily replace callback
            self._callback = still_callback
            self._callback_context = None
            
            # Trigger still capture
            if not self.snap_image(resolution_index):
                error("Failed to trigger still capture")
                self._callback = original_callback
                self._callback_context = original_context
                return False
            
            # Wait for still image
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
            
            # Convert to numpy array
            w = capture_success['width']
            h = capture_success['height']
            stride = amcam.TDIBWIDTHBYTES(w * 24)
            image_data = np.frombuffer(pData, dtype=np.uint8).reshape((h, stride))[:, :w*3].reshape((h, w, 3)).copy()
            
            # Convert BGR to RGB
            image_data = image_data[:, :, ::-1].copy()
            
            del pData
            
            # Save with metadata
            success = self.save_image(image_data, filepath, additional_metadata)
            
            del image_data
            gc.collect()
            
            if success:
                info(f"Still image captured and saved: {filepath}")
            else:
                error(f"Failed to save still image: {filepath}")
            
            return success
            
        except Exception as e:
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
            res_index, width, height = self.get_current_resolution()
            
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
        # Update frame buffer on IMAGE events
        if event == self.EVENT_IMAGE and hasattr(self, '_frame_buffer') and self._frame_buffer is not None:
            try:
                self._hcam.PullImageV4(self._frame_buffer, 0, 24, 0, None)
            except:
                pass
        
        # Call registered callback
        if self._callback:
            self._callback(event, self._callback_context)