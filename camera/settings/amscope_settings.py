"""
AmScope camera settings implementation.

Provides settings management for AmScope cameras with hardware-specific
controls like fan, TEC, low noise mode, and demosaic settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from camera.settings.camera_settings import (
    CameraSettings,
    SettingMetadata,
    SettingType,
    RGBALevel,
    RGBGain,
    FileFormat,
)
from logger import info, debug, error, exception

if TYPE_CHECKING:
    from base_camera import BaseCamera, CameraResolution


@dataclass
class AmscopeSettings(CameraSettings):
    """
    Settings for AmScope cameras.
    
    Extends base CameraSettings with AmScope-specific hardware controls:
    - Fan control for cooling
    - TEC (Thermoelectric Cooler) control and target temperature
    - Low noise mode and high full-well capacity
    - Test pattern for diagnostics
    - Demosaic algorithm selection
    """
    
    # AmScope-specific hardware controls
    fan_enabled: bool = field(default=False)
    tec_enabled: bool = field(default=False)
    tec_target: int = field(default=-10)  # Target temperature in Celsius
    low_noise_mode: bool = field(default=False)
    high_fullwell: bool = field(default=False)
    test_pattern: bool = field(default=False)
    demosaic_algorithm: int = field(default=0)  # 0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG
    
    # Internal camera reference (not serialized to YAML)
    _camera: BaseCamera | None = field(default=None, init=False, repr=False, compare=False)
    
    @classmethod
    def get_metadata(cls) -> list[SettingMetadata]:
        """
        Get metadata for all AmScope settings.
        
        This is the SINGLE SOURCE OF TRUTH for:
        - GUI generation (widget types, labels, groups)
        - Validation (min/max ranges)
        - Organization (grouping related settings)
        
        Returns:
            List of SettingMetadata for all settings
        """
        return [
            # Exposure settings
            SettingMetadata(
                name="auto_expo",
                display_name="Auto Exposure",
                setting_type=SettingType.BOOL,
                description="Enable automatic exposure control",
                group="Exposure",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="exposure",
                display_name="Exposure Target",
                setting_type=SettingType.RANGE,
                description="Target brightness for auto exposure (16-235)",
                min_value=16,
                max_value=235,
                group="Exposure",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="exposure_time_us",
                display_name="Exposure Time (µs)",
                setting_type=SettingType.RANGE,
                description="Manual exposure time in microseconds",
                min_value=1,
                max_value=1000000,
                group="Exposure",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="gain_percent",
                display_name="Gain (%)",
                setting_type=SettingType.RANGE,
                description="Sensor gain in percent (100-1600)",
                min_value=100,
                max_value=1600,
                group="Exposure",
                runtime_changeable=True,
            ),
            
            # White balance settings
            SettingMetadata(
                name="temp",
                display_name="Color Temperature",
                setting_type=SettingType.RANGE,
                description="White balance temperature in Kelvin (2000-15000)",
                min_value=2000,
                max_value=15000,
                group="White Balance",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="tint",
                display_name="Tint",
                setting_type=SettingType.RANGE,
                description="White balance tint adjustment (200-2500)",
                min_value=200,
                max_value=2500,
                group="White Balance",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="wbgain",
                display_name="RGB Gain",
                setting_type=SettingType.RGB_GAIN,
                description="Fine-tune RGB white balance gains (-127 to 127)",
                group="White Balance",
                runtime_changeable=True,
            ),
            
            # Color and image quality
            SettingMetadata(
                name="hue",
                display_name="Hue",
                setting_type=SettingType.RANGE,
                description="Color hue adjustment (-180 to 180)",
                min_value=-180,
                max_value=180,
                group="Color",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="saturation",
                display_name="Saturation",
                setting_type=SettingType.RANGE,
                description="Color saturation (0-255)",
                min_value=0,
                max_value=255,
                group="Color",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="brightness",
                display_name="Brightness",
                setting_type=SettingType.RANGE,
                description="Image brightness adjustment (-64 to 64)",
                min_value=-64,
                max_value=64,
                group="Color",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="contrast",
                display_name="Contrast",
                setting_type=SettingType.RANGE,
                description="Image contrast adjustment (-100 to 100)",
                min_value=-100,
                max_value=100,
                group="Color",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="gamma",
                display_name="Gamma",
                setting_type=SettingType.RANGE,
                description="Gamma correction (0-180)",
                min_value=0,
                max_value=180,
                group="Color",
                runtime_changeable=True,
            ),
            
            # Level range
            SettingMetadata(
                name="levelrange_low",
                display_name="Black Point",
                setting_type=SettingType.RGBA_LEVEL,
                description="Output level for darkest input values (0-255)",
                group="Levels",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="levelrange_high",
                display_name="White Point",
                setting_type=SettingType.RGBA_LEVEL,
                description="Output level for brightest input values (0-255)",
                group="Levels",
                runtime_changeable=True,
            ),
            
            # Resolution
            SettingMetadata(
                name="resolution_index",
                display_name="Resolution",
                setting_type=SettingType.RANGE,
                description="Camera resolution index",
                min_value=0,
                max_value=10,  # Will be validated against actual camera resolutions
                group="Capture",
                runtime_changeable=False,  # Requires restart
            ),
            
            # File format
            SettingMetadata(
                name="fformat",
                display_name="File Format",
                setting_type=SettingType.DROPDOWN,
                description="Default file format for saved images",
                choices=["png", "tiff", "jpeg", "bmp"],
                group="Capture",
                runtime_changeable=True,
            ),
            
            # AmScope-specific hardware controls
            SettingMetadata(
                name="fan_enabled",
                display_name="Cooling Fan",
                setting_type=SettingType.BOOL,
                description="Enable camera cooling fan",
                group="Hardware",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="tec_enabled",
                display_name="TEC Cooler",
                setting_type=SettingType.BOOL,
                description="Enable thermoelectric cooler",
                group="Hardware",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="tec_target",
                display_name="TEC Target (°C)",
                setting_type=SettingType.RANGE,
                description="Target temperature for TEC in Celsius (-40 to 20)",
                min_value=-40,
                max_value=20,
                group="Hardware",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="low_noise_mode",
                display_name="Low Noise Mode",
                setting_type=SettingType.BOOL,
                description="Enable low noise mode (reduces read noise)",
                group="Hardware",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="high_fullwell",
                display_name="High Full-Well",
                setting_type=SettingType.BOOL,
                description="Enable high full-well capacity mode",
                group="Hardware",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="test_pattern",
                display_name="Test Pattern",
                setting_type=SettingType.BOOL,
                description="Enable test pattern for diagnostics",
                group="Advanced",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="demosaic_algorithm",
                display_name="Demosaic Algorithm",
                setting_type=SettingType.RANGE,
                description="Bayer demosaic algorithm (0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG)",
                min_value=0,
                max_value=3,
                group="Advanced",
                runtime_changeable=True,
            ),
        ]
    
    def _validate_and_set(self, param_name: str, value: int | bool) -> None:
        """
        Validate a parameter value against metadata ranges.
        
        Args:
            param_name: Name of the parameter
            value: Value to validate
            
        Raises:
            ValueError: If value is outside valid range
        """
        metadata_dict = {m.name: m for m in self.get_metadata()}
        
        if param_name not in metadata_dict:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        meta = metadata_dict[param_name]
        
        # Validate based on type
        if meta.setting_type == SettingType.RANGE:
            if not isinstance(value, (int, float)):
                raise ValueError(f"{param_name} must be numeric")
            if not (meta.min_value <= value <= meta.max_value):
                raise ValueError(
                    f"{param_name} = {value} is outside valid range "
                    f"[{meta.min_value}, {meta.max_value}]"
                )
        elif meta.setting_type == SettingType.BOOL:
            if not isinstance(value, bool):
                raise ValueError(f"{param_name} must be boolean")
    
    def _apply_to_sdk(self, param_name: str, value) -> None:
        """
        Apply a setting to the camera SDK.
        
        Args:
            param_name: Parameter name
            value: Value to apply
        """
        if self._camera is None or not hasattr(self._camera, '_hcam'):
            debug(f"Camera not available, skipping SDK update for {param_name}")
            return
        
        hcam = self._camera._hcam
        if hcam is None:
            debug(f"Camera handle not available, skipping SDK update for {param_name}")
            return
        
        try:
            # Map parameter names to SDK calls
            if param_name == "auto_expo":
                hcam.put_AutoExpoEnable(1 if value else 0)
            elif param_name == "exposure":
                hcam.put_AutoExpoTarget(value)
            elif param_name == "exposure_time_us":
                hcam.put_ExpoTime(value)
            elif param_name == "gain_percent":
                hcam.put_ExpoAGain(value)
            elif param_name == "temp" and hasattr(self, 'tint'):
                hcam.put_TempTint(value, self.tint)
            elif param_name == "tint" and hasattr(self, 'temp'):
                hcam.put_TempTint(self.temp, value)
            elif param_name == "hue":
                hcam.put_Hue(value)
            elif param_name == "saturation":
                hcam.put_Saturation(value)
            elif param_name == "brightness":
                hcam.put_Brightness(value)
            elif param_name == "contrast":
                hcam.put_Contrast(value)
            elif param_name == "gamma":
                hcam.put_Gamma(value)
            elif param_name == "wbgain":
                hcam.put_WhiteBalanceGain([value.r, value.g, value.b])
            elif param_name in ["levelrange_low", "levelrange_high"]:
                low = self.levelrange_low
                high = self.levelrange_high
                hcam.put_LevelRange([low.r, low.g, low.b, low.a], 
                                   [high.r, high.g, high.b, high.a])
            elif param_name == "fan_enabled":
                hcam.put_Option(0x0a, 1 if value else 0)  # OPTION_FAN
            elif param_name == "tec_enabled":
                hcam.put_Option(0x08, 1 if value else 0)  # OPTION_TEC
            elif param_name == "tec_target":
                hcam.put_Option(0x0c, value)  # OPTION_TECTARGET
            elif param_name == "low_noise_mode":
                hcam.put_Option(0x53, 1 if value else 0)  # OPTION_LOW_NOISE
            elif param_name == "high_fullwell":
                hcam.put_Option(0x51, 1 if value else 0)  # OPTION_HIGH_FULLWELL
            elif param_name == "test_pattern":
                hcam.put_Option(0x2c, 1 if value else 0)  # OPTION_TESTPATTERN
            elif param_name == "demosaic_algorithm":
                hcam.put_Option(0x5a, value)  # OPTION_DEMOSAIC
            
            debug(f"Applied {param_name} = {value} to camera SDK")
            
        except Exception as e:
            error(f"Failed to apply {param_name} to camera: {e}")
    
    # Required abstract method implementations
    
    def set_auto_exposure(self, enabled: bool) -> None:
        """Enable or disable automatic exposure."""
        self._validate_and_set("auto_expo", enabled)
        self.auto_expo = enabled
        self._apply_to_sdk("auto_expo", enabled)
    
    def set_exposure(self, value: int) -> None:
        """Set auto exposure target value."""
        self._validate_and_set("exposure", value)
        self.exposure = value
        self._apply_to_sdk("exposure", value)
    
    def set_temp(self, value: int) -> None:
        """Set white balance temperature."""
        self._validate_and_set("temp", value)
        self.temp = value
        self._apply_to_sdk("temp", value)
    
    def set_tint(self, value: int) -> None:
        """Set white balance tint."""
        self._validate_and_set("tint", value)
        self.tint = value
        self._apply_to_sdk("tint", value)
    
    def set_white_balance_gain(self, gain: RGBGain) -> None:
        """Set RGB white balance gains."""
        gain.validate()
        self.wbgain = gain
        self._apply_to_sdk("wbgain", gain)
    
    def set_hue(self, value: int) -> None:
        """Set hue adjustment."""
        self._validate_and_set("hue", value)
        self.hue = value
        self._apply_to_sdk("hue", value)
    
    def set_saturation(self, value: int) -> None:
        """Set saturation."""
        self._validate_and_set("saturation", value)
        self.saturation = value
        self._apply_to_sdk("saturation", value)
    
    def set_brightness(self, value: int) -> None:
        """Set brightness."""
        self._validate_and_set("brightness", value)
        self.brightness = value
        self._apply_to_sdk("brightness", value)
    
    def set_contrast(self, value: int) -> None:
        """Set contrast."""
        self._validate_and_set("contrast", value)
        self.contrast = value
        self._apply_to_sdk("contrast", value)
    
    def set_gamma(self, value: int) -> None:
        """Set gamma correction."""
        self._validate_and_set("gamma", value)
        self.gamma = value
        self._apply_to_sdk("gamma", value)
    
    def set_level_range(self, low: RGBALevel, high: RGBALevel) -> None:
        """Set level range mapping."""
        low.validate()
        high.validate()
        self.levelrange_low = low
        self.levelrange_high = high
        self._apply_to_sdk("levelrange_low", low)
    
    # Resolution methods
    
    def get_resolutions(self) -> list[CameraResolution]:
        """Get available camera resolutions."""
        if self._camera is None or not hasattr(self._camera, '_hcam'):
            return []
        
        from base_camera import CameraResolution
        
        try:
            resolutions = []
            hcam = self._camera._hcam
            
            # AmScope cameras typically have multiple resolutions
            count = hcam.ResolutionNumber
            for i in range(count):
                width, height = hcam.get_Resolution(i)
                resolutions.append(CameraResolution(width=width, height=height))
            
            return resolutions
        except Exception as e:
            error(f"Failed to get resolutions: {e}")
            return []
    
    def get_current_resolution(self) -> tuple[int, int, int]:
        """Get current resolution as (index, width, height)."""
        if self._camera is None or not hasattr(self._camera, '_hcam'):
            return (0, 0, 0)
        
        try:
            hcam = self._camera._hcam
            index = hcam.get_eSize()
            width, height = hcam.get_Size()
            return (index, width, height)
        except Exception as e:
            error(f"Failed to get current resolution: {e}")
            return (0, 0, 0)
    
    def set_resolution(self, resolution_index: int) -> bool:
        """Set camera resolution."""
        if self._camera is None or not hasattr(self._camera, '_hcam'):
            return False
        
        try:
            hcam = self._camera._hcam
            
            # Validate index
            if not (0 <= resolution_index < hcam.ResolutionNumber):
                error(f"Invalid resolution index: {resolution_index}")
                return False
            
            # Apply resolution
            hcam.put_eSize(resolution_index)
            self.resolution_index = resolution_index
            info(f"Resolution set to index {resolution_index}")
            return True
            
        except Exception as e:
            error(f"Failed to set resolution: {e}")
            return False
    
    def get_still_resolutions(self) -> list[CameraResolution]:
        """Get available still image resolutions."""
        if self._camera is None or not hasattr(self._camera, '_hcam'):
            return []
        
        from base_camera import CameraResolution
        
        try:
            resolutions = []
            hcam = self._camera._hcam
            
            count = hcam.StillResolutionNumber
            for i in range(count):
                width, height = hcam.get_StillResolution(i)
                resolutions.append(CameraResolution(width=width, height=height))
            
            return resolutions
        except Exception as e:
            error(f"Failed to get still resolutions: {e}")
            return []
    
    # Exposure time methods
    
    def get_exposure_time(self) -> int:
        """Get current exposure time in microseconds."""
        if self._camera is None or not hasattr(self._camera, '_hcam'):
            return self.exposure_time_us
        
        try:
            return self._camera._hcam.get_ExpoTime()
        except Exception as e:
            error(f"Failed to get exposure time: {e}")
            return self.exposure_time_us
    
    def set_exposure_time(self, time_us: int) -> bool:
        """Set exposure time in microseconds."""
        self._validate_and_set("exposure_time_us", time_us)
        self.exposure_time_us = time_us
        self._apply_to_sdk("exposure_time_us", time_us)
        return True
    
    # Gain methods
    
    def get_gain(self) -> int:
        """Get current gain in percent."""
        if self._camera is None or not hasattr(self._camera, '_hcam'):
            return self.gain_percent
        
        try:
            return self._camera._hcam.get_ExpoAGain()
        except Exception as e:
            error(f"Failed to get gain: {e}")
            return self.gain_percent
    
    def set_gain(self, gain_percent: int) -> bool:
        """Set gain in percent."""
        self._validate_and_set("gain_percent", gain_percent)
        self.gain_percent = gain_percent
        self._apply_to_sdk("gain_percent", gain_percent)
        return True
    
    # AmScope-specific hardware control methods
    
    def set_fan(self, enabled: bool) -> None:
        """Enable or disable cooling fan."""
        self._validate_and_set("fan_enabled", enabled)
        self.fan_enabled = enabled
        self._apply_to_sdk("fan_enabled", enabled)
        info(f"Cooling fan {'enabled' if enabled else 'disabled'}")
    
    def set_tec(self, enabled: bool) -> None:
        """Enable or disable TEC cooler."""
        self._validate_and_set("tec_enabled", enabled)
        self.tec_enabled = enabled
        self._apply_to_sdk("tec_enabled", enabled)
        info(f"TEC cooler {'enabled' if enabled else 'disabled'}")
    
    def set_tec_target(self, temperature: int) -> None:
        """Set TEC target temperature in Celsius."""
        self._validate_and_set("tec_target", temperature)
        self.tec_target = temperature
        self._apply_to_sdk("tec_target", temperature)
        info(f"TEC target temperature set to {temperature}°C")
    
    def set_low_noise_mode(self, enabled: bool) -> None:
        """Enable or disable low noise mode."""
        self._validate_and_set("low_noise_mode", enabled)
        self.low_noise_mode = enabled
        self._apply_to_sdk("low_noise_mode", enabled)
        info(f"Low noise mode {'enabled' if enabled else 'disabled'}")
    
    def set_high_fullwell(self, enabled: bool) -> None:
        """Enable or disable high full-well capacity mode."""
        self._validate_and_set("high_fullwell", enabled)
        self.high_fullwell = enabled
        self._apply_to_sdk("high_fullwell", enabled)
        info(f"High full-well mode {'enabled' if enabled else 'disabled'}")
    
    def set_test_pattern(self, enabled: bool) -> None:
        """Enable or disable test pattern."""
        self._validate_and_set("test_pattern", enabled)
        self.test_pattern = enabled
        self._apply_to_sdk("test_pattern", enabled)
        info(f"Test pattern {'enabled' if enabled else 'disabled'}")
    
    def set_demosaic_algorithm(self, algorithm: int) -> None:
        """Set demosaic algorithm (0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG)."""
        self._validate_and_set("demosaic_algorithm", algorithm)
        self.demosaic_algorithm = algorithm
        self._apply_to_sdk("demosaic_algorithm", algorithm)
        info(f"Demosaic algorithm set to {algorithm}")
    
    # Apply and refresh methods
    
    def apply_to_camera(self, camera: BaseCamera) -> None:
        """
        Apply all settings to camera hardware.
        
        Args:
            camera: Camera instance to apply settings to
        """
        self._camera = camera
        info(f"Applying all settings to camera {camera.model}")
        
        try:
            # Apply all settings in logical order
            self.set_auto_exposure(self.auto_expo)
            self.set_exposure(self.exposure)
            self.set_exposure_time(self.exposure_time_us)
            self.set_gain(self.gain_percent)
            
            self.set_temp(self.temp)
            self.set_tint(self.tint)
            self.set_white_balance_gain(self.wbgain)
            
            self.set_hue(self.hue)
            self.set_saturation(self.saturation)
            self.set_brightness(self.brightness)
            self.set_contrast(self.contrast)
            self.set_gamma(self.gamma)
            
            self.set_level_range(self.levelrange_low, self.levelrange_high)
            
            # AmScope-specific hardware controls
            self.set_fan(self.fan_enabled)
            self.set_tec(self.tec_enabled)
            self.set_tec_target(self.tec_target)
            self.set_low_noise_mode(self.low_noise_mode)
            self.set_high_fullwell(self.high_fullwell)
            self.set_test_pattern(self.test_pattern)
            self.set_demosaic_algorithm(self.demosaic_algorithm)
            
            info("All settings applied successfully")
            
        except Exception as e:
            exception(f"Failed to apply settings to camera")
    
    def refresh_from_camera(self, camera: BaseCamera) -> None:
        """
        Read all current settings from camera hardware.
        
        Args:
            camera: Camera instance to read from
        """
        self._camera = camera
        info(f"Refreshing settings from camera {camera.model}")
        
        if not hasattr(camera, '_hcam') or camera._hcam is None:
            error("Camera not available for refresh")
            return
        
        hcam = camera._hcam
        
        try:
            # Read exposure settings
            self.auto_expo = bool(hcam.get_AutoExpoEnable())
            self.exposure = hcam.get_AutoExpoTarget()
            self.exposure_time_us = hcam.get_ExpoTime()
            self.gain_percent = hcam.get_ExpoAGain()
            
            # Read white balance
            temp, tint = hcam.get_TempTint()
            self.temp = temp
            self.tint = tint
            
            wb_gains = hcam.get_WhiteBalanceGain()
            self.wbgain = RGBGain(r=wb_gains[0], g=wb_gains[1], b=wb_gains[2])
            
            # Read color adjustments
            self.hue = hcam.get_Hue()
            self.saturation = hcam.get_Saturation()
            self.brightness = hcam.get_Brightness()
            self.contrast = hcam.get_Contrast()
            self.gamma = hcam.get_Gamma()
            
            # Read level range
            low, high = hcam.get_LevelRange()
            self.levelrange_low = RGBALevel(r=low[0], g=low[1], b=low[2], a=low[3])
            self.levelrange_high = RGBALevel(r=high[0], g=high[1], b=high[2], a=high[3])
            
            # Read resolution
            self.resolution_index = hcam.get_eSize()
            
            # Read AmScope-specific hardware settings
            self.fan_enabled = bool(hcam.get_Option(0x0a))  # OPTION_FAN
            self.tec_enabled = bool(hcam.get_Option(0x08))  # OPTION_TEC
            self.tec_target = hcam.get_Option(0x0c)  # OPTION_TECTARGET
            self.low_noise_mode = bool(hcam.get_Option(0x53))  # OPTION_LOW_NOISE
            self.high_fullwell = bool(hcam.get_Option(0x51))  # OPTION_HIGH_FULLWELL
            self.test_pattern = bool(hcam.get_Option(0x2c))  # OPTION_TESTPATTERN
            self.demosaic_algorithm = hcam.get_Option(0x5a)  # OPTION_DEMOSAIC
            
            info("Successfully refreshed all settings from camera")
            
        except Exception as e:
            exception(f"Failed to refresh settings from camera")