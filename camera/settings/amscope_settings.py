from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from camera.settings.camera_settings import (
    CameraSettings,
    SettingMetadata,
    SettingType,
    RGBALevel,
    FileFormat,
)
from logger import info, error, exception, debug

if TYPE_CHECKING:
    from camera.cameras.base_camera import BaseCamera, CameraResolution


@dataclass
class AmscopeSettings(CameraSettings):
    version: str = "0"
    auto_exposure: bool = True
    exposure: int = 128
    exposure_time_us: int = 50000
    resolution: int = 0
    temp: int = 6500
    tint: int = 1000
    contrast: int = 0
    hue: int = 0
    saturation: int = 128
    brightness: int = 0
    gamma: int = 100
    gain_percent: int = 100
    levelrange_low: RGBALevel = RGBALevel(0, 0, 0, 0)
    levelrange_high: RGBALevel = RGBALevel(255, 255, 255, 255)
    fformat: FileFormat = FileFormat.TIFF
    
    fan: bool = field(default=False)
    high_fullwell: bool = field(default=False)
    
    _camera: BaseCamera | None = field(default=None, repr=False, compare=False)
    
    def __post_init__(self) -> None:
        super().__post_init__()
    
    def get_metadata(self) -> list[SettingMetadata]:
        """
        Get metadata for all settings with dynamically populated resolution choices.
        """
        # Get available resolutions from camera
        resolutions = self.get_resolutions()
        resolution_choices = [f"{res.width}x{res.height}" 
                            for idx, res in enumerate(resolutions)] if resolutions else []
        
        return [
            SettingMetadata(
                name="auto_exposure",
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
                description="Target brightness for auto exposure",
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
                description="Sensor gain in percent",
                min_value=100,
                max_value=1600,
                group="Exposure",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="temp",
                display_name="Color Temperature",
                setting_type=SettingType.RANGE,
                description="White balance temperature in Kelvin",
                min_value=2000,
                max_value=15000,
                group="White Balance",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="tint",
                display_name="Tint",
                setting_type=SettingType.RANGE,
                description="White balance tint adjustment",
                min_value=200,
                max_value=2500,
                group="White Balance",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="hue",
                display_name="Hue",
                setting_type=SettingType.RANGE,
                description="Color hue adjustment",
                min_value=-180,
                max_value=180,
                group="Color",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="saturation",
                display_name="Saturation",
                setting_type=SettingType.RANGE,
                description="Color saturation",
                min_value=0,
                max_value=255,
                group="Color",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="brightness",
                display_name="Brightness",
                setting_type=SettingType.RANGE,
                description="Image brightness adjustment",
                min_value=-64,
                max_value=64,
                group="Color",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="contrast",
                display_name="Contrast",
                setting_type=SettingType.RANGE,
                description="Image contrast adjustment",
                min_value=-100,
                max_value=100,
                group="Color",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="gamma",
                display_name="Gamma",
                setting_type=SettingType.RANGE,
                description="Gamma correction",
                min_value=0,
                max_value=180,
                group="Color",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="levelrange_low",
                display_name="Black Point",
                setting_type=SettingType.RGBA_LEVEL,
                description="Output level for darkest input values",
                group="Levels",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="levelrange_high",
                display_name="White Point",
                setting_type=SettingType.RGBA_LEVEL,
                description="Output level for brightest input values",
                group="Levels",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="resolution",
                display_name="Resolution",
                setting_type=SettingType.DROPDOWN,
                description="Camera resolution",
                choices=resolution_choices,  # Dynamically populated from camera
                group="Capture",
                runtime_changeable=False,
            ),
            SettingMetadata(
                name="fformat",
                display_name="File Format",
                setting_type=SettingType.DROPDOWN,
                description="Default file format for saved images",
                choices=self._file_formats,
                group="Capture",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="fan",
                display_name="Cooling Fan",
                setting_type=SettingType.BOOL,
                description="Enable cooling fan",
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
        ]
    
    def _get_metadata_map(self) -> dict[str, SettingMetadata]:
        return {m.name: m for m in self.get_metadata()}
    
    def _validate_range(self, name: str, value: int) -> None:
        meta = self._get_metadata_map().get(name)
        if meta and meta.setting_type == SettingType.RANGE:
            if not (meta.min_value <= value <= meta.max_value):
                raise ValueError(
                    f"{name} must be in [{meta.min_value}, {meta.max_value}], got {value}"
                )
    
    def set_auto_exposure(self, enabled: bool) -> None:
        self.auto_exposure = enabled
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_AutoExpoEnable(1 if enabled else 0)
    
    def set_exposure(self, value: int) -> None:
        self._validate_range("exposure", value)
        self.exposure = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_AutoExpoTarget(value)
    
    def set_exposure_time(self, time_us: int) -> bool:
        self._validate_range("exposure_time_us", time_us)
        self.exposure_time_us = time_us
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_ExpoTime(time_us)
        return True
    
    def set_gain(self, gain_percent: int) -> None:
        self._validate_range("gain_percent", gain_percent)
        self.gain_percent = gain_percent
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_ExpoAGain(gain_percent)
    
    def set_temp(self, value: int) -> None:
        self._validate_range("temp", value)
        self.temp = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_TempTint(value, self.tint)
    
    def set_tint(self, value: int) -> None:
        self._validate_range("tint", value)
        self.tint = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_TempTint(self.temp, value)
    
    def set_temp_tint(self, temp: int, tint: int) -> None:
        self._validate_range("temp", temp)
        self._validate_range("tint", tint)
        self.temp = temp
        self.tint = tint
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_TempTint(temp, tint)
    
    def set_hue(self, value: int) -> None:
        self._validate_range("hue", value)
        self.hue = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_Hue(value)
    
    def set_saturation(self, value: int) -> None:
        self._validate_range("saturation", value)
        self.saturation = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_Saturation(value)
    
    def set_brightness(self, value: int) -> None:
        self._validate_range("brightness", value)
        self.brightness = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_Brightness(value)
    
    def set_contrast(self, value: int) -> None:
        self._validate_range("contrast", value)
        self.contrast = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_Contrast(value)
    
    def set_gamma(self, value: int) -> None:
        self._validate_range("gamma", value)
        self.gamma = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_Gamma(value)
    
    def set_level_range(self, low: RGBALevel, high: RGBALevel) -> None:
        low.validate()
        high.validate()
        self.levelrange_low = low
        self.levelrange_high = high
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_LevelRange(
                (low.r, low.g, low.b, low.a),
                (high.r, high.g, high.b, high.a)
            )
    
    def set_fan(self, enabled: bool) -> None:
        self.fan = enabled
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_Option(0x0a, 1 if enabled else 0)
    
    def set_high_fullwell(self, enabled: bool) -> None:
        self.high_fullwell = enabled
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_Option(0x51, 1 if enabled else 0)
    
    def get_resolutions(self) -> list[CameraResolution]:
        if self._camera is None or not hasattr(self._camera, '_hcam'):
            return []
        
        from camera.cameras.base_camera import CameraResolution
        
        try:
            resolutions = []
            hcam = self._camera._hcam
            count = hcam.ResolutionNumber()
            for i in range(count):
                width, height = hcam.get_Resolution(i)
                resolutions.append(CameraResolution(width=width, height=height))
            return resolutions
        except Exception as e:
            error(f"Failed to get resolutions: {e}")
            return []
    
    def get_current_resolution(self) -> tuple[int, int, int]:
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
    
    def set_resolution(self, index: int, value: str = "") -> bool:
        """Set camera resolution. Requires camera restart."""
        try:
            if not (0 <= index < len(self.get_resolutions())):
                error(f"Invalid resolution index: {index}")
                return False
            
            camera_was_open = self._camera.is_open
            saved_callback = self._camera._callback
            saved_context = self._camera._callback_context
            
            if camera_was_open:
                info("Camera is open, stopping to set resolution")
                self._camera.stop_capture()
            
            # Set resolution on the underlying camera
            self._camera._hcam.put_eSize(index)
            self.resolution = index
                        
            if camera_was_open:
                info("Restarting camera to set resolution")
                self._camera.start_capture(saved_callback, saved_context)
            
            debug(f"Successfully changed resolution to index {index}")
            return True
        except Exception as e:
            error(f"Failed to set resolution: {e}")
            return False
    
    def get_still_resolutions(self) -> list[CameraResolution]:
        if self._camera is None or not hasattr(self._camera, '_hcam'):
            return []
        
        from camera.cameras.base_camera import CameraResolution
        
        try:
            resolutions = []
            hcam = self._camera._hcam
            count = hcam.StillResolutionNumber()
            for i in range(count):
                width, height = hcam.get_StillResolution(i)
                resolutions.append(CameraResolution(width=width, height=height))
            return resolutions
        except Exception as e:
            error(f"Failed to get still resolutions: {e}")
            return []
    
    def get_exposure_time(self) -> int:
        if self._camera is None or not hasattr(self._camera, '_hcam'):
            return self.exposure_time_us
        
        try:
            return self._camera._hcam.get_ExpoTime()
        except Exception as e:
            error(f"Failed to get exposure time: {e}")
            return self.exposure_time_us
    
    def apply_to_camera(self, camera: BaseCamera) -> None:
        self._camera = camera
        info(f"Applying settings to camera {camera.model}")
        
        try:
            self.set_resolution(self.resolution)
            self.set_auto_exposure(self.auto_exposure)
            self.set_exposure(self.exposure)
            self.set_exposure_time(self.exposure_time_us)
            self.set_gain(self.gain_percent)
            
            self.set_temp_tint(self.temp, self.tint)
            
            self.set_hue(self.hue)
            self.set_saturation(self.saturation)
            self.set_brightness(self.brightness)
            self.set_contrast(self.contrast)
            self.set_gamma(self.gamma)
            
            self.set_level_range(self.levelrange_low, self.levelrange_high)
            
            self.set_fan(self.fan)
            self.set_high_fullwell(self.high_fullwell)
            
            info("Successfully applied all settings to camera")
        except Exception as e:
            exception(f"Failed to apply settings to camera: {e}")
            raise
    
    def refresh_from_camera(self, camera: BaseCamera) -> None:
        self._camera = camera
        info(f"Refreshing settings from camera {camera.model}")
        
        if not hasattr(camera, '_hcam') or camera._hcam is None:
            error("Camera not available for refresh")
            return
        
        hcam = camera._hcam
        
        try:
            self.auto_exposure = bool(hcam.get_AutoExpoEnable())
            self.exposure = hcam.get_AutoExpoTarget()
            self.exposure_time_us = hcam.get_ExpoTime()
            self.gain_percent = hcam.get_ExpoAGain()
            
            temp, tint = hcam.get_TempTint()
            self.temp = temp
            self.tint = tint
            
            self.hue = hcam.get_Hue()
            self.saturation = hcam.get_Saturation()
            self.brightness = hcam.get_Brightness()
            self.contrast = hcam.get_Contrast()
            self.gamma = hcam.get_Gamma()
            
            low, high = hcam.get_LevelRange()
            self.levelrange_low = RGBALevel(r=low[0], g=low[1], b=low[2], a=low[3])
            self.levelrange_high = RGBALevel(r=high[0], g=high[1], b=high[2], a=high[3])
            
            self.resolution = hcam.get_eSize()
            
            self.fan = bool(hcam.get_Option(0x0a))
            self.high_fullwell = bool(hcam.get_Option(0x51))
            
            info("Successfully refreshed all settings from camera")
        except Exception as e:
            exception(f"Failed to refresh settings from camera: {e}")