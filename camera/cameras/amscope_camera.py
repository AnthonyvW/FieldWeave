"""
Amscope camera implementation using the amcam SDK.
"""

from typing import Tuple, Callable, Any, Optional, Dict, TYPE_CHECKING
from types import SimpleNamespace
from pathlib import Path
import ctypes
import numpy as np
import threading
import gc
from camera.cameras.base_camera import BaseCamera, CameraResolution, CameraInfo
from logger import get_logger

# Module-level reference to the loaded SDK
_amcam = None

# Type hints for IDE support (won't execute at runtime when checking types)
if TYPE_CHECKING:
    import amcam  # This is just for type hints, won't actually import


class AmscopeCamera(BaseCamera):
    """
    Amscope camera implementation using the amcam SDK.
    Wraps the amcam library to conform to the BaseCamera interface.
    
    The SDK must be loaded before using this class:
        AmscopeCamera.ensure_sdk_loaded()
    
    Or it will be loaded automatically on first use.
    """
    
    # Class-level flag to track SDK loading
    _sdk_loaded = False
    
    def __init__(self):
        super().__init__()
        
        # Ensure SDK is loaded before instantiating
        if not AmscopeCamera._sdk_loaded:
            AmscopeCamera.ensure_sdk_loaded()
        
        self._hcam: Optional[Any] = None  # Will be amcam.Amcam after SDK loads
        self._camera_info: Optional[CameraInfo] = None
    
    @classmethod
    def ensure_sdk_loaded(cls, sdk_path: Optional[Path] = None) -> bool:
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
        
        logger = get_logger()
        
        try:
            from camera.sdk_loaders.amscope_sdk_loader import AmscopeSdkLoader
            
            loader = AmscopeSdkLoader(sdk_path)
            _amcam = loader.load()
            
            cls._sdk_loaded = True
            logger.info("Amscope SDK loaded successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load Amscope SDK: {e}")
            logger.info("Attempting fallback to direct import...")
            
            try:
                # Fallback to direct import if loader fails
                import amcam as amcam_module
                _amcam = amcam_module
                cls._sdk_loaded = True
                logger.info("Amscope SDK loaded via direct import")
                return True
            except ImportError as ie:
                logger.error(f"Direct import also failed: {ie}")
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
    
    # Class-level event constant accessors
    @classmethod
    def get_event_constants(cls):
        """
        Get event constants as a namespace object.
        Useful for accessing events without a camera instance.
        
        Returns:
            SimpleNamespace with event constants
        """
        amcam = cls._get_sdk_static()
        return SimpleNamespace(
            IMAGE=amcam.AMCAM_EVENT_IMAGE,
            EXPOSURE=amcam.AMCAM_EVENT_EXPOSURE,
            TEMPTINT=amcam.AMCAM_EVENT_TEMPTINT,
            STILLIMAGE=amcam.AMCAM_EVENT_STILLIMAGE,
            ERROR=amcam.AMCAM_EVENT_ERROR,
            DISCONNECTED=amcam.AMCAM_EVENT_DISCONNECTED
        )
    
    # Event type constants - these are properties since SDK loads dynamically
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
    def handle(self) -> Optional[Any]:
        """Get the underlying amcam handle"""
        return self._hcam
    
    def open(self, camera_id: str) -> bool:
        """Open connection to Amscope camera"""
        amcam = self._get_sdk()
        try:
            self._hcam = amcam.Amcam.Open(camera_id)
            if self._hcam:
                self._is_open = True
                # Set RGB byte order for Qt compatibility
                self._hcam.put_Option(amcam.AMCAM_OPTION_BYTEORDER, 0)
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
    
    def start_capture(self, callback: Callable, context: Any) -> bool:
        """Start capturing frames with callback"""
        if not self._hcam:
            return False
        
        amcam = self._get_sdk()
        try:
            # Get current resolution to allocate frame buffer
            res_index, width, height = self.get_current_resolution()
            
            # Create persistent frame buffer (like manufacturer's self.pData)
            # This will be continuously updated by the event callback
            buffer_size = amcam.TDIBWIDTHBYTES(width * 24) * height
            self._frame_buffer = bytearray(buffer_size)  # Use bytearray so it's mutable
            
            self._callback = callback
            self._callback_context = context
            self._hcam.StartPullModeWithCallback(self._event_callback_wrapper, self)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def stop_capture(self):
        """Stop capturing frames"""
        if self._hcam:
            amcam = self._get_sdk()
            try:
                self._hcam.Stop()
            except self._get_sdk().HRESULTException:
                pass
    
    def pull_image(self, buffer: ctypes.Array, bits_per_pixel: int = 24, timeout_ms: int = 1000) -> bool:
        """
        Pull the latest image into buffer (expects ctypes.create_string_buffer)
        
        Args:
            buffer: ctypes buffer to receive image data
            bits_per_pixel: Bits per pixel (typically 24)
            timeout_ms: Timeout in milliseconds to wait for frame (default 1000ms)
            
        Returns:
            True if successful, False otherwise
        """
        if not self._hcam:
            logger = get_logger()
            logger.error("Cannot pull image: camera handle is None")
            return False
        
        amcam = self._get_sdk()
        try:
            # Use WaitImageV4 to wait for a frame (bStill=0 for video stream)
            # This is more reliable than PullImageV2 which may fail if no frame is ready
            self._hcam.WaitImageV4(timeout_ms, buffer, 0, bits_per_pixel, 0, None)
            return True
        except self._get_sdk().HRESULTException as e:
            # If timeout or no frame available, log the error
            logger = get_logger()
            logger.error(f"Failed to pull image: {e}")
            return False
    
    def snap_image(self, resolution_index: int = 0) -> bool:
        """Capture a still image"""
        if not self._hcam:
            return False
        
        amcam = self._get_sdk()
        try:
            self._hcam.Snap(resolution_index)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def pull_still_image(self, buffer: ctypes.Array, bits_per_pixel: int = 24) -> Tuple[bool, int, int]:
        """
        Pull a still image into buffer
        
        Args:
            buffer: Buffer to receive image data (ctypes.create_string_buffer, should be large enough)
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
        except self._get_sdk().HRESULTException:
            return False, 0, 0
    
    def get_resolutions(self) -> list[CameraResolution]:
        """Get available preview resolutions"""
        if not self._camera_info or not self._camera_info.model:
            return []
        
        resolutions = []
        for i in range(self._camera_info.model.preview):
            res = self._camera_info.model.res[i]
            resolutions.append(CameraResolution(res.width, res.height))
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
        
        amcam = self._get_sdk()
        try:
            self._hcam.put_eSize(resolution_index)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_exposure_range(self) -> Tuple[int, int, int]:
        """Get exposure time range (min, max, default) in microseconds"""
        if not self._hcam:
            return 0, 0, 0
        
        amcam = self._get_sdk()
        try:
            return self._hcam.get_ExpTimeRange()
        except self._get_sdk().HRESULTException:
            return 0, 0, 0
    
    def get_exposure_time(self) -> int:
        """Get current exposure time in microseconds"""
        amcam = self._get_sdk()
        if not self._hcam:
            return 0
        
        try:
            return self._hcam.get_ExpoTime()
        except self._get_sdk().HRESULTException:
            return 0
    
    def set_exposure_time(self, time_us: int) -> bool:
        """Set exposure time in microseconds"""
        amcam = self._get_sdk()
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_ExpoTime(time_us)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_gain_range(self) -> Tuple[int, int, int]:
        """Get gain range (min, max, default) in percent"""
        amcam = self._get_sdk()
        if not self._hcam:
            return 0, 0, 0
        
        try:
            return self._hcam.get_ExpoAGainRange()
        except self._get_sdk().HRESULTException:
            return 0, 0, 0
    
    def get_gain(self) -> int:
        """Get current gain in percent"""
        amcam = self._get_sdk()
        if not self._hcam:
            return 0
        
        try:
            return self._hcam.get_ExpoAGain()
        except self._get_sdk().HRESULTException:
            return 0
    
    def set_gain(self, gain_percent: int) -> bool:
        """Set gain in percent"""
        amcam = self._get_sdk()
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_ExpoAGain(gain_percent)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_auto_exposure(self) -> bool:
        """Get auto exposure state"""
        amcam = self._get_sdk()
        if not self._hcam:
            return False
        
        try:
            return self._hcam.get_AutoExpoEnable() == 1
        except self._get_sdk().HRESULTException:
            return False
    
    def set_auto_exposure(self, enabled: bool) -> bool:
        """Set auto exposure state"""
        amcam = self._get_sdk()
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_AutoExpoEnable(1 if enabled else 0)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def supports_white_balance(self) -> bool:
        """Check if camera supports white balance (not monochrome)"""
        if not self._camera_info or not self._camera_info.model:
            return False
        
        amcam = self._get_sdk()
        return (self._camera_info.model.flag & amcam.AMCAM_FLAG_MONO) == 0
    
    def get_white_balance_range(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get white balance range ((temp_min, temp_max), (tint_min, tint_max))"""
        amcam = self._get_sdk()
        return ((amcam.AMCAM_TEMP_MIN, amcam.AMCAM_TEMP_MAX),
                (amcam.AMCAM_TINT_MIN, amcam.AMCAM_TINT_MAX))
    
    def get_white_balance(self) -> Tuple[int, int]:
        """Get current white balance (temperature, tint)"""
        amcam = self._get_sdk()
        if not self._hcam:
            return amcam.AMCAM_TEMP_DEF, amcam.AMCAM_TINT_DEF
        
        try:
            return self._hcam.get_TempTint()
        except self._get_sdk().HRESULTException:
            return amcam.AMCAM_TEMP_DEF, amcam.AMCAM_TINT_DEF
    
    def set_white_balance(self, temperature: int, tint: int) -> bool:
        """Set white balance"""
        amcam = self._get_sdk()
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_TempTint(temperature, tint)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def auto_white_balance(self) -> bool:
        """Perform one-time auto white balance"""
        amcam = self._get_sdk()
        if not self._hcam:
            return False
        
        try:
            self._hcam.AwbOnce()
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    # ========================================================================
    # Image Processing Parameters
    # ========================================================================
    
    def get_hue(self) -> int:
        """
        Get hue value.
        
        Returns:
            Hue value in range [-180, 180]
            
        Raises:
            RuntimeError: If camera is not initialized
        """
        if not self._hcam:
            raise RuntimeError("Camera not initialized")
        
        try:
            return self._hcam.get_Hue()
        except self._get_sdk().HRESULTException as e:
            raise RuntimeError(f"Failed to get hue: {e}") from e
    
    def set_hue(self, hue: int) -> bool:
        """
        Set hue value.
        
        Args:
            hue: Hue value in range [-180, 180]
            
        Returns:
            True if successful, False otherwise
        """
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_Hue(hue)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_saturation(self) -> int:
        """
        Get saturation value.
        
        Returns:
            Saturation value in range [0, 255]
            
        Raises:
            RuntimeError: If camera is not initialized
        """
        if not self._hcam:
            raise RuntimeError("Camera not initialized")
        
        try:
            return self._hcam.get_Saturation()
        except self._get_sdk().HRESULTException as e:
            raise RuntimeError(f"Failed to get saturation: {e}") from e
    
    def set_saturation(self, saturation: int) -> bool:
        """
        Set saturation value.
        
        Args:
            saturation: Saturation value in range [0, 255]
            
        Returns:
            True if successful, False otherwise
        """
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_Saturation(saturation)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_brightness(self) -> int:
        """
        Get brightness value.
        
        Returns:
            Brightness value in range [-64, 64]
            
        Raises:
            RuntimeError: If camera is not initialized
        """
        if not self._hcam:
            raise RuntimeError("Camera not initialized")
        
        try:
            return self._hcam.get_Brightness()
        except self._get_sdk().HRESULTException as e:
            raise RuntimeError(f"Failed to get brightness: {e}") from e
    
    def set_brightness(self, brightness: int) -> bool:
        """
        Set brightness value.
        
        Args:
            brightness: Brightness value in range [-64, 64]
            
        Returns:
            True if successful, False otherwise
        """
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_Brightness(brightness)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_contrast(self) -> int:
        """
        Get contrast value.
        
        Returns:
            Contrast value in range [-100, 100]
            
        Raises:
            RuntimeError: If camera is not initialized
        """
        if not self._hcam:
            raise RuntimeError("Camera not initialized")
        
        try:
            return self._hcam.get_Contrast()
        except self._get_sdk().HRESULTException as e:
            raise RuntimeError(f"Failed to get contrast: {e}") from e
    
    def set_contrast(self, contrast: int) -> bool:
        """
        Set contrast value.
        
        Args:
            contrast: Contrast value in range [-100, 100]
            
        Returns:
            True if successful, False otherwise
        """
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_Contrast(contrast)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_gamma(self) -> int:
        """
        Get gamma value.
        
        Returns:
            Gamma value in range [20, 180]
            
        Raises:
            RuntimeError: If camera is not initialized
        """
        if not self._hcam:
            raise RuntimeError("Camera not initialized")
        
        try:
            return self._hcam.get_Gamma()
        except self._get_sdk().HRESULTException as e:
            raise RuntimeError(f"Failed to get gamma: {e}") from e
    
    def set_gamma(self, gamma: int) -> bool:
        """
        Set gamma value.
        
        Args:
            gamma: Gamma value in range [20, 180]
            
        Returns:
            True if successful, False otherwise
        """
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_Gamma(gamma)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_auto_exposure_target(self) -> int:
        """
        Get auto exposure target brightness.
        
        Returns:
            Auto exposure target in range [16, 235]
            
        Raises:
            RuntimeError: If camera is not initialized
        """
        if not self._hcam:
            raise RuntimeError("Camera not initialized")
        
        try:
            return self._hcam.get_AutoExpoTarget()
        except self._get_sdk().HRESULTException as e:
            raise RuntimeError(f"Failed to get auto exposure target: {e}") from e
    
    def set_auto_exposure_target(self, target: int) -> bool:
        """
        Set auto exposure target brightness.
        
        Args:
            target: Auto exposure target in range [16, 235]
            
        Returns:
            True if successful, False otherwise
        """
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_AutoExpoTarget(target)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_white_balance_gain(self) -> Tuple[int, int, int]:
        """
        Get RGB white balance gain values.
        
        Returns:
            Tuple of (R, G, B) gain values in range [-127, 127]
            
        Raises:
            RuntimeError: If camera is not initialized or not supported
            
        Note:
            Only works in RGB Gain mode.
        """
        if not self._hcam:
            raise RuntimeError("Camera not initialized")
        
        try:
            return self._hcam.get_WhiteBalanceGain()
        except self._get_sdk().HRESULTException as e:
            raise RuntimeError(f"Failed to get white balance gain (may not be supported in Temp/Tint mode): {e}") from e
    
    def set_white_balance_gain(self, r: int, g: int, b: int) -> bool:
        """
        Set RGB white balance gain values.
        
        Args:
            r: Red gain in range [-127, 127]
            g: Green gain in range [-127, 127]
            b: Blue gain in range [-127, 127]
            
        Returns:
            True if successful, False otherwise
            
        Note:
            Only works in RGB Gain mode.
        """
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_WhiteBalanceGain([r, g, b])
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_level_range(self) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Get level range (low and high) for RGBA channels.
        
        Returns:
            Tuple of ((R_low, G_low, B_low, A_low), (R_high, G_high, B_high, A_high))
            Each value in range [0, 255]
            
        Raises:
            RuntimeError: If camera is not initialized
        """
        if not self._hcam:
            raise RuntimeError("Camera not initialized")
        
        try:
            low, high = self._hcam.get_LevelRange()
            return (tuple(low), tuple(high))
        except self._get_sdk().HRESULTException as e:
            raise RuntimeError(f"Failed to get level range: {e}") from e
    
    def set_level_range(
        self,
        low: Tuple[int, int, int, int],
        high: Tuple[int, int, int, int]
    ) -> bool:
        """
        Set level range (low and high) for RGBA channels.
        
        Args:
            low: Tuple of (R_low, G_low, B_low, A_low), each in range [0, 255]
            high: Tuple of (R_high, G_high, B_high, A_high), each in range [0, 255]
            
        Returns:
            True if successful, False otherwise
        """
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_LevelRange(list(low), list(high))
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def auto_level_range(self) -> bool:
        """
        Perform automatic level range adjustment.
        
        Returns:
            True if successful, False otherwise
        """
        if not self._hcam:
            return False
        
        try:
            self._hcam.LevelRangeAuto()
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_option(self, option: int) -> int:
        """
        Get a camera option value.
        
        Args:
            option: Option ID (use AMCAM_OPTION_* constants)
            
        Returns:
            Option value
            
        Raises:
            RuntimeError: If camera is not initialized
        """
        if not self._hcam:
            raise RuntimeError("Camera not initialized")
        
        try:
            return self._hcam.get_Option(option)
        except self._get_sdk().HRESULTException as e:
            raise RuntimeError(f"Failed to get option {option}: {e}") from e
    
    def set_option(self, option: int, value: int) -> bool:
        """
        Set a camera option value.
        
        Args:
            option: Option ID (use AMCAM_OPTION_* constants)
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_Option(option, value)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_sharpening(self) -> Tuple[int, int, int]:
        """
        Get sharpening parameters.
        
        Returns:
            Tuple of (strength, radius, threshold):
            - strength: [0, 500], 0 = disabled
            - radius: [1, 10]
            - threshold: [0, 255]
            
        Raises:
            RuntimeError: If camera is not initialized
        """
        amcam = self._get_sdk()
        if not self._hcam:
            raise RuntimeError("Camera not initialized")
        
        try:
            # Get sharpening option value
            val = self._hcam.get_Option(amcam.AMCAM_OPTION_SHARPENING)
            
            # Extract components: (threshold << 24) | (radius << 16) | strength
            strength = val & 0xFFFF
            radius = (val >> 16) & 0xFF
            threshold = (val >> 24) & 0xFF
            
            return (strength, radius, threshold)
        except self._get_sdk().HRESULTException as e:
            raise RuntimeError(f"Failed to get sharpening: {e}") from e
    
    def set_sharpening(self, strength: int, radius: int = 2, threshold: int = 0) -> bool:
        """
        Set sharpening parameters.
        
        Args:
            strength: Sharpening strength [0, 500], 0 = disabled
            radius: Sharpening radius [1, 10], default 2
            threshold: Sharpening threshold [0, 255], default 0
            
        Returns:
            True if successful, False otherwise
        """
        amcam = self._get_sdk()
        if not self._hcam:
            return False
        
        try:
            # Combine into single value: (threshold << 24) | (radius << 16) | strength
            val = (threshold << 24) | (radius << 16) | strength
            self._hcam.put_Option(amcam.AMCAM_OPTION_SHARPENING, val)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_linear_tone_mapping(self) -> bool:
        """
        Get linear tone mapping state.
        
        Returns:
            True if enabled, False if disabled
            
        Raises:
            RuntimeError: If camera is not initialized
        """
        amcam = self._get_sdk()
        if not self._hcam:
            raise RuntimeError("Camera not initialized")
        
        try:
            val = self._hcam.get_Option(amcam.AMCAM_OPTION_LINEAR)
            return val == 1
        except self._get_sdk().HRESULTException as e:
            raise RuntimeError(f"Failed to get linear tone mapping: {e}") from e
    
    def set_linear_tone_mapping(self, enabled: bool) -> bool:
        """
        Set linear tone mapping on/off.
        
        Args:
            enabled: True to enable, False to disable
            
        Returns:
            True if successful, False otherwise
        """
        amcam = self._get_sdk()
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_Option(amcam.AMCAM_OPTION_LINEAR, 1 if enabled else 0)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_curve_tone_mapping(self) -> int:
        """
        Get curve tone mapping setting.
        
        Returns:
            0 = off, 1 = polynomial, 2 = logarithmic
            
        Raises:
            RuntimeError: If camera is not initialized
        """
        amcam = self._get_sdk()
        if not self._hcam:
            raise RuntimeError("Camera not initialized")
        
        try:
            return self._hcam.get_Option(amcam.AMCAM_OPTION_CURVE)
        except self._get_sdk().HRESULTException as e:
            raise RuntimeError(f"Failed to get curve tone mapping: {e}") from e
    
    def set_curve_tone_mapping(self, curve_type: int) -> bool:
        """
        Set curve tone mapping.
        
        Args:
            curve_type: 0 = off, 1 = polynomial, 2 = logarithmic
            
        Returns:
            True if successful, False otherwise
        """
        amcam = self._get_sdk()
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_Option(amcam.AMCAM_OPTION_CURVE, curve_type)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    # ========================================================================
    # End of Image Processing Parameters
    # ========================================================================
    
    def get_frame_rate(self) -> Tuple[int, int, int]:
        """Get frame rate info (frames_in_period, time_period_ms, total_frames)"""
        amcam = self._get_sdk()
        if not self._hcam:
            return 0, 0, 0
        
        try:
            return self._hcam.get_FrameRate()
        except self._get_sdk().HRESULTException:
            return 0, 0, 0
    
    @staticmethod
    def _get_sdk_static():
        """Static method to get SDK (for use in classmethods)"""
        global _amcam
        if _amcam is None:
            raise RuntimeError(
                "Amscope SDK not loaded. Call AmscopeCamera.ensure_sdk_loaded() first."
            )
        return _amcam
    
    def set_camera_info(self, info: CameraInfo):
        """Set camera information (needed before opening)"""
        self._camera_info = info
    
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
            resolutions.append(CameraResolution(res.width, res.height))
        return resolutions
    
    def get_camera_metadata(self) -> Dict[str, Any]:
        """
        Get current camera settings as metadata
        
        Returns:
            Dictionary containing current camera settings including:
            - Camera identification (name, model, id)
            - Resolution settings
            - Exposure settings (time, gain, auto-exposure state)
            - White balance settings (if supported)
            - Image processing parameters (hue, saturation, brightness, etc.)
            - Frame rate information
            
        Note:
            If camera is not initialized or a parameter cannot be read,
            that parameter will be omitted from the metadata dictionary.
        """
        metadata = {}
        
        # Camera identification
        if self._camera_info:
            metadata["camera_name"] = self._camera_info.displayname
            metadata["camera_id"] = self._camera_info.id
            if self._camera_info.model:
                metadata["model_name"] = getattr(self._camera_info.model, 'name', 'Unknown')
        
        # Helper function to safely get values
        def safe_get(getter_func, key, default=None):
            try:
                return getter_func()
            except (RuntimeError, Exception):
                return default
        
        # Resolution
        res_index, width, height = self.get_current_resolution()
        metadata["resolution_index"] = res_index
        metadata["width"] = width
        metadata["height"] = height
        metadata["resolution"] = f"{width}x{height}"
        
        # Exposure settings
        metadata["exposure_time_us"] = self.get_exposure_time()
        metadata["gain_percent"] = self.get_gain()
        metadata["auto_exposure_enabled"] = self.get_auto_exposure()
        
        target = safe_get(self.get_auto_exposure_target, "auto_exposure_target")
        if target is not None:
            metadata["auto_exposure_target"] = target
        
        # Exposure range info
        exp_min, exp_max, exp_def = self.get_exposure_range()
        metadata["exposure_range_us"] = {
            "min": exp_min,
            "max": exp_max,
            "default": exp_def
        }
        
        # Gain range info
        gain_min, gain_max, gain_def = self.get_gain_range()
        metadata["gain_range_percent"] = {
            "min": gain_min,
            "max": gain_max,
            "default": gain_def
        }
        
        # White balance (if supported)
        if self.supports_white_balance():
            temp, tint = self.get_white_balance()
            metadata["white_balance_temperature"] = temp
            metadata["white_balance_tint"] = tint
            
            (temp_min, temp_max), (tint_min, tint_max) = self.get_white_balance_range()
            metadata["white_balance_range"] = {
                "temperature": {"min": temp_min, "max": temp_max},
                "tint": {"min": tint_min, "max": tint_max}
            }
            
            # RGB gain mode (may not work in Temp/Tint mode)
            rgb_gain = safe_get(self.get_white_balance_gain, "white_balance_rgb_gain")
            if rgb_gain is not None:
                r_gain, g_gain, b_gain = rgb_gain
                metadata["white_balance_rgb_gain"] = {
                    "red": r_gain,
                    "green": g_gain,
                    "blue": b_gain
                }
        else:
            metadata["monochrome"] = True
        
        # Image processing parameters
        hue = safe_get(self.get_hue, "hue")
        if hue is not None:
            metadata["hue"] = hue
            
        saturation = safe_get(self.get_saturation, "saturation")
        if saturation is not None:
            metadata["saturation"] = saturation
            
        brightness = safe_get(self.get_brightness, "brightness")
        if brightness is not None:
            metadata["brightness"] = brightness
            
        contrast = safe_get(self.get_contrast, "contrast")
        if contrast is not None:
            metadata["contrast"] = contrast
            
        gamma = safe_get(self.get_gamma, "gamma")
        if gamma is not None:
            metadata["gamma"] = gamma
        
        # Level range
        level_range = safe_get(self.get_level_range, "level_range")
        if level_range is not None:
            low, high = level_range
            metadata["level_range_low"] = {
                "red": low[0],
                "green": low[1],
                "blue": low[2],
                "alpha": low[3]
            }
            metadata["level_range_high"] = {
                "red": high[0],
                "green": high[1],
                "blue": high[2],
                "alpha": high[3]
            }
        
        # Sharpening
        sharpening = safe_get(self.get_sharpening, "sharpening")
        if sharpening is not None:
            strength, radius, threshold = sharpening
            metadata["sharpening"] = {
                "strength": strength,
                "radius": radius,
                "threshold": threshold
            }
        
        # Tone mapping
        linear = safe_get(self.get_linear_tone_mapping, "linear_tone_mapping")
        if linear is not None:
            metadata["linear_tone_mapping"] = linear
            
        curve = safe_get(self.get_curve_tone_mapping, "curve_tone_mapping")
        if curve is not None:
            curve_names = {0: "off", 1: "polynomial", 2: "logarithmic"}
            metadata["curve_tone_mapping"] = curve_names.get(curve, "unknown")
            metadata["curve_tone_mapping_value"] = curve
        
        # Frame rate
        frames, period_ms, total = self.get_frame_rate()
        if period_ms > 0:
            metadata["frame_rate_fps"] = round(frames * 1000 / period_ms, 2)
        metadata["frame_rate_info"] = {
            "frames_in_period": frames,
            "period_ms": period_ms,
            "total_frames": total
        }
        
        # SDK version if available
        amcam = self._get_sdk()
        try:
            metadata["sdk_version"] = amcam.Amcam.Version()
        except Exception:
            metadata["sdk_version"] = "unknown"
        
        return metadata
    
    def capture_and_save_still(
        self,
        filepath: Path,
        resolution_index: int = 0,
        additional_metadata: Optional[Dict[str, Any]] = None,
        timeout_ms: int = 5000
    ) -> bool:
        """
        Capture a still image and save it to disk with metadata.
        
        This method handles the complete workflow:
        1. Triggers still image capture at specified resolution
        2. Waits for image to be ready (with timeout)
        3. Pulls image data and converts to numpy array
        4. Saves with full metadata
        
        Args:
            filepath: Path where image should be saved
            resolution_index: Resolution index for still capture (0 = highest)
            additional_metadata: Optional dictionary of additional metadata to save
            timeout_ms: Timeout in milliseconds to wait for capture (default 5000)
            
        Returns:
            True if successful, False otherwise
        """
        logger = get_logger()
        
        if not self._hcam:
            logger.error("Camera not open")
            return False
        
        if not self.supports_still_capture():
            logger.error("Camera does not support still capture")
            logger.info(f"Camera model: {self._camera_info.model.name if self._camera_info else 'Unknown'}")
            logger.info(f"Still resolution count: {self._camera_info.model.still if self._camera_info else 0}")
            return False
        
        try:
            # Get resolution for this still index
            still_resolutions = self.get_still_resolutions()
            if resolution_index >= len(still_resolutions):
                logger.error(f"Invalid resolution index: {resolution_index}")
                return False
            
            res = still_resolutions[resolution_index]
            width, height = res.width, res.height
            logger.debug(f"Still capture target resolution: {width}x{height}")
            
            # Use Python bytes instead of ctypes buffer
            amcam = self._get_sdk()
            buffer_size = amcam.TDIBWIDTHBYTES(width * 24) * height
            pData = bytes(buffer_size)
            
            # Event to signal still image is ready
            still_ready = threading.Event()
            capture_success = {'success': False, 'width': 0, 'height': 0}
            
            # Store original callback
            original_callback = self._callback
            original_context = self._callback_context
            
            logger.debug(f"Original callback: {original_callback is not None}, context: {original_context is not None}")
            
            def still_callback(event, ctx):
                logger.debug(f"Still callback received event: {event}, STILLIMAGE={self.EVENT_STILLIMAGE}, IMAGE={self.EVENT_IMAGE}")
                if event == self.EVENT_STILLIMAGE:
                    # Pull the still image using PullImageV3
                    info = amcam.AmcamFrameInfoV3()
                    try:
                        logger.debug("Attempting to pull still image...")
                        self._hcam.PullImageV3(pData, 1, 24, 0, info)
                        capture_success['success'] = True
                        capture_success['width'] = info.width
                        capture_success['height'] = info.height
                        logger.debug(f"Still image pulled successfully: {info.width}x{info.height}")
                    except Exception as e:
                        logger.error(f"Failed to pull still image: {e}")
                        capture_success['success'] = False
                    still_ready.set()
                
                # Also call original callback if it exists
                if original_callback:
                    original_callback(event, original_context)
            
            # Temporarily replace callback
            self._callback = still_callback
            self._callback_context = None
            
            logger.debug("Triggering still capture...")
            # Trigger still capture
            if not self.snap_image(resolution_index):
                logger.error("Failed to trigger still capture")
                self._callback = original_callback
                self._callback_context = original_context
                return False
            
            logger.debug(f"Waiting for still image (timeout: {timeout_ms}ms)...")
            # Wait for still image with timeout
            if not still_ready.wait(timeout_ms / 1000.0):
                logger.error(f"Still capture timed out after {timeout_ms}ms")
                logger.error("STILLIMAGE event never received")
                self._callback = original_callback
                self._callback_context = original_context
                return False
            
            logger.debug("Still image event received!")
            
            # Restore original callback
            self._callback = original_callback
            self._callback_context = original_context
            
            if not capture_success['success']:
                logger.error("Failed to pull still image")
                return False
            
            # Convert to numpy array - creates a copy
            w = capture_success['width']
            h = capture_success['height']
            stride = amcam.TDIBWIDTHBYTES(w * 24)
            image_data = np.frombuffer(pData, dtype=np.uint8).reshape((h, stride))[:, :w*3].reshape((h, w, 3)).copy()
            
            # Convert BGR to RGB
            image_data = image_data[:, :, ::-1].copy()
            
            # Delete pData immediately
            del pData
            
            # Save with metadata
            success = self.save_image(image_data, filepath, additional_metadata)
            
            # Explicitly delete and force GC
            del image_data
            gc.collect()
            
            if success:
                logger.info(f"Still image captured and saved: {filepath}")
            else:
                logger.error(f"Failed to save still image: {filepath}")
            
            return success
            
        except Exception as e:
            logger.exception(f"Failed to capture and save still image: {filepath}")
            return False
    
    def capture_and_save_stream(
        self,
        filepath: Path,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Capture current frame from live stream and save it to disk with metadata.
        
        Uses the manufacturer's approach: directly saves the most recent frame
        from the continuously-updated buffer (no waiting or pausing needed).
        
        Args:
            filepath: Path where image should be saved
            additional_metadata: Optional dictionary of additional metadata to save
            
        Returns:
            True if successful, False otherwise
            
        Note:
            Camera must be in capture mode (start_capture() must have been called)
        """
        logger = get_logger()
        
        if not self._hcam:
            logger.error("Camera not open")
            return False
        
        if not self._is_open:
            logger.error("Camera not in capture mode")
            return False
        
        # Check if we have a frame buffer (set during start_capture)
        if not hasattr(self, '_frame_buffer') or self._frame_buffer is None:
            logger.error("No frame buffer available - camera may not be streaming")
            return False
        
        try:
            # Get current resolution
            res_index, width, height = self.get_current_resolution()
            
            # Simply copy from the current frame buffer (updated continuously by callback)
            # This is the manufacturer's approach - no waiting or pausing needed!
            amcam = self._get_sdk()
            stride = amcam.TDIBWIDTHBYTES(width * 24)
            
            # Create numpy array from the persistent buffer
            image_data = np.frombuffer(self._frame_buffer, dtype=np.uint8).reshape((height, stride))[:, :width*3].reshape((height, width, 3)).copy()
            
            # Convert BGR to RGB
            image_data = image_data[:, :, ::-1].copy()
            
            # Save with metadata
            success = self.save_image(image_data, filepath, additional_metadata)
            
            # Explicitly delete image_data and force GC
            del image_data
            gc.collect()
            
            if success:
                logger.info(f"Stream frame captured and saved: {filepath}")
            else:
                logger.error(f"Failed to save stream frame: {filepath}")
            
            return success
            
        except Exception as e:
            logger.exception(f"Failed to capture and save stream frame: {filepath}")
            return False
            
        except Exception as e:
            logger.exception(f"Failed to capture and save stream frame: {filepath}")
            return False
    
    @staticmethod
    def calculate_buffer_size(width: int, height: int, bits_per_pixel: int = 24) -> int:
        """
        Calculate required buffer size for image data
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            bits_per_pixel: Bits per pixel (typically 24 for RGB)
            
        Returns:
            Buffer size in bytes
        """
        amcam = AmscopeCamera._get_sdk_static()
        return amcam.TDIBWIDTHBYTES(width * bits_per_pixel) * height
    
    @staticmethod
    def calculate_stride(width: int, bits_per_pixel: int = 24) -> int:
        """
        Calculate image stride (bytes per row)
        
        Args:
            width: Image width in pixels
            bits_per_pixel: Bits per pixel (typically 24 for RGB)
            
        Returns:
            Stride in bytes
        """
        amcam = AmscopeCamera._get_sdk_static()
        return amcam.TDIBWIDTHBYTES(width * bits_per_pixel)
    
    @classmethod
    def enable_gige(cls, callback: Optional[Callable] = None, context: Any = None):
        """
        Enable GigE camera support
        
        Args:
            callback: Optional callback for GigE events
            context: Optional context for callback
        """
        # Ensure SDK is loaded
        if not cls._sdk_loaded:
            cls.ensure_sdk_loaded()
        
        amcam = cls._get_sdk_static()
        amcam.Amcam.GigeEnable(callback, context)
    
    def _event_callback_wrapper(self, event: int, context: Any):
        """
        Internal wrapper for camera events.
        Translates amcam events to the callback registered with start_capture.
        Also updates the persistent frame buffer on IMAGE events (manufacturer's approach).
        """
        # Update persistent frame buffer on IMAGE events
        # This is how the manufacturer's example works - continuous buffer updates
        if event == self.EVENT_IMAGE and hasattr(self, '_frame_buffer') and self._frame_buffer is not None:
            try:
                # Pull the latest frame into our persistent buffer
                self._hcam.PullImageV4(self._frame_buffer, 0, 24, 0, None)
            except:
                pass  # Silently ignore pull errors in callback
        
        # IMPORTANT: Always call the registered callback if it exists
        # Don't check _callback_context because it might be None during still capture
        if self._callback:
            self._callback(event, self._callback_context)