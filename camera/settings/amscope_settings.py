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
    exposure_time: int = 50000
    preview_resolution: str = ""
    still_resolution: str = ""
    temp: int = 6500
    tint: int = 1000
    contrast: int = 0
    hue: int = 0
    saturation: int = 128
    brightness: int = 0
    gamma: int = 100
    gain: int = 100
    level_range_low: RGBALevel = RGBALevel(0, 0, 0, 0)
    level_range_high: RGBALevel = RGBALevel(255, 255, 255, 255)
    fformat: FileFormat = FileFormat.TIFF
    rotate: int = 0
    hflip: bool = False
    vflip: bool = False
    
    _camera: BaseCamera | None = field(default=None, repr=False, compare=False)
    
    def __post_init__(self) -> None:
        super().__post_init__()
    
    def get_metadata(self) -> list[SettingMetadata]:
        """
        Get metadata for all settings with dynamically populated resolution choices.
        """
        # Get available resolutions from camera
        resolutions = self.get_resolutions()
        resolution_choices = [f"{res.width}x{res.height}" for res in resolutions]

        still_resolutions = self.get_still_resolutions()
        still_resolution_choices = [f"{res.width}x{res.height}" for res in still_resolutions]

        return [
            SettingMetadata(
                name="preview_resolution",
                display_name="Preview Resolution",
                setting_type=SettingType.DROPDOWN,
                description="Camera preview resolution",
                choices=resolution_choices,
                group="Capture",
                runtime_changeable=False,
            ),
            SettingMetadata(
                name="still_resolution",
                display_name="Still Resolution",
                setting_type=SettingType.DROPDOWN,
                description="Resolution used when capturing a still image",
                choices=still_resolution_choices,
                group="Capture",
                runtime_changeable=True,
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
                name="rotate",
                display_name="Rotation",
                setting_type=SettingType.DROPDOWN,
                description="Rotate the camera image clockwise. Requires camera restart to apply.",
                choices=["0", "90", "180", "270"],
                group="Capture",
                runtime_changeable=False,
            ),
            SettingMetadata(
                name="hflip",
                display_name="Flip Horizontal",
                setting_type=SettingType.BOOL,
                description="Mirror the image horizontally",
                group="Capture",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="vflip",
                display_name="Flip Vertical",
                setting_type=SettingType.BOOL,
                description="Mirror the image vertically",
                group="Capture",
                runtime_changeable=True,
            ),
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
                controlled_by="auto_exposure",
                controlled_when=False,
            ),
            SettingMetadata(
                name="exposure_time",
                display_name="Exposure Time (µs)",
                setting_type=SettingType.RANGE,
                description="Manual exposure time in microseconds",
                min_value=1,
                max_value=1000000,
                group="Exposure",
                runtime_changeable=True,
                controlled_by="auto_exposure",
            ),
            SettingMetadata(
                name="gain",
                display_name="Gain",
                setting_type=SettingType.RANGE,
                description="Sensor gain",
                min_value=100,
                max_value=300,
                group="Exposure",
                runtime_changeable=True,
                controlled_by="auto_exposure",
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
                name="level_range_low",
                display_name="Black Point",
                setting_type=SettingType.RGBA_LEVEL,
                description="Output level for darkest input values",
                group="Levels",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="level_range_high",
                display_name="White Point",
                setting_type=SettingType.RGBA_LEVEL,
                description="Output level for brightest input values",
                group="Levels",
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

    # ------------------------------------------------------------------
    # Live-value protocol
    # ------------------------------------------------------------------

    def get_live_values(self) -> dict[str, int]:
        """Return live hardware exposure_time and gain while auto_exposure is on."""
        if not self.auto_exposure:
            return {}
        if not (self._camera and hasattr(self._camera, '_hcam')):
            return {}
        try:
            hcam = self._camera._hcam
            return {
                "exposure_time": hcam.get_ExpoTime(),
                "gain": hcam.get_ExpoAGain(),
            }
        except Exception as e:
            error(f"Failed to read live exposure values: {e}")
            return {}

    def on_controller_disabled(self, controller_name: str) -> None:
        """Flush live exposure_time / gain into stored settings when auto_exposure turns off."""
        if controller_name != "auto_exposure":
            super().on_controller_disabled(controller_name)
            return

        if not (self._camera and hasattr(self._camera, '_hcam')):
            return
        try:
            hcam = self._camera._hcam
            self.exposure_time = hcam.get_ExpoTime()
            self.gain = hcam.get_ExpoAGain()
            debug(
                f"Flushed auto-exposure values: exposure_time={self.exposure_time}, gain={self.gain}"
            )
        except Exception as e:
            error(f"Failed to flush live exposure values: {e}")

    # ------------------------------------------------------------------
    
    def set_auto_exposure(self, enabled: bool) -> None:
        if not enabled and self.auto_exposure:
            # Flush hardware values before turning off so stored settings are up-to-date.
            self.on_controller_disabled("auto_exposure")
        self.auto_exposure = enabled
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_AutoExpoEnable(1 if enabled else 0)
    
    def set_exposure(self, value: int) -> None:
        self._validate_range("exposure", value)
        self.exposure = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_AutoExpoTarget(value)
    
    def set_exposure_time(self, time_us: int) -> bool:
        self._validate_range("exposure_time", time_us)
        self.exposure_time = time_us
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_ExpoTime(time_us)
        return True
    
    def set_gain(self, gain: int) -> None:
        self._validate_range("gain", gain)
        self.gain = gain
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_ExpoAGain(gain)
    
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
        self.level_range_low = low
        self.level_range_high = high
        if self._camera:
            self._camera._hcam.put_LevelRange(
                (low.r, low.g, low.b, low.a),
                (high.r, high.g, high.b, high.a)
            )

    def set_level_range_low(self, low: RGBALevel) -> None:
        low.validate()
        if self._camera:
            high = self.level_range_high
            self._camera._hcam.put_LevelRange(
                (low.r, low.g, low.b, low.a),
                (high.r, high.g, high.b, high.a)
            )
            self.level_range_low = low

    def set_level_range_high(self, high: RGBALevel) -> None:
        high.validate()
        if self._camera:
            low = self.level_range_low
            self._camera._hcam.put_LevelRange(
                (low.r, low.g, low.b, low.a),
                (high.r, high.g, high.b, high.a)
            )
            self.level_range_high = high
    
    def set_rotate(self, value: int | str, index: int | None = None) -> bool:
        """
        Set the camera image rotation (0, 90, 180, 270 degrees clockwise).

        AMCAM_OPTION_ROTATE cannot be changed while the camera is running, so
        this method follows the same stop/restart pattern as set_preview_resolution.
        The dropdown supplies both the string label (e.g. "90") and the index of
        that label in the choices list.  When ``index`` is provided it is used
        directly; otherwise it is derived from ``value``.
        """
        valid_degrees = [0, 90, 180, 270]

        if index is not None:
            if not (0 <= index < len(valid_degrees)):
                error(f"Invalid rotation index: {index}. Valid range: 0-{len(valid_degrees) - 1}")
                return False
            degrees = valid_degrees[index]
        else:
            try:
                degrees = int(value)
            except (ValueError, TypeError):
                error(f"Invalid rotation value: {value!r}. Must be one of {valid_degrees}")
                return False
            if degrees not in valid_degrees:
                error(f"Invalid rotation value: {degrees}. Must be one of {valid_degrees}")
                return False

        try:
            self.rotate = degrees

            if not (self._camera and hasattr(self._camera, '_hcam')):
                return True

            camera_was_open = self._camera.is_open
            saved_callback = self._camera._callback
            saved_context = self._camera._callback_context

            if camera_was_open:
                debug("Camera is open, stopping to set rotation")
                self._camera.stop_capture()

            amcam = self._camera._get_sdk()
            self._camera._hcam.put_Option(amcam.AMCAM_OPTION_ROTATE, degrees)

            if camera_was_open:
                debug("Restarting camera after rotation change")
                self._camera.start_capture(saved_callback, saved_context)

            debug(f"Successfully changed rotation to {degrees} degrees")
            return True
        except Exception as e:
            error(f"Failed to set rotation: {e}")
            return False

    def set_hflip(self, enabled: bool) -> None:
        """Flip the image horizontally."""
        self.hflip = enabled
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_HFlip(1 if enabled else 0)

    def set_vflip(self, enabled: bool) -> None:
        """Flip the image vertically."""
        self.vflip = enabled
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_VFlip(1 if enabled else 0)

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

    def get_output_dimensions(self) -> tuple[int, int]:
        """
        Return the final (width, height) of frames delivered by the SDK.
        """
        if self._camera is None or not hasattr(self._camera, '_hcam'):
            return (0, 0)
        try:
            width, height = self._camera._hcam.get_FinalSize()
            return (width, height)
        except Exception:
            pass
        # Fallback: raw sensor resolution (no rotation compensation)
        _, width, height = self.get_current_resolution()
        return (width, height)
    
    def set_preview_resolution(self, value: str, index: int | None = None) -> bool:
        """
        Set camera preview resolution. Requires camera restart.
        """
        try:
            resolutions = self.get_resolutions()
            choices = [f"{r.width}x{r.height}" for r in resolutions]

            if index is None:
                if value not in choices:
                    error(f"Invalid resolution value: {value!r}. Available: {choices}")
                    return False
                index = choices.index(value)
            else:
                if not (0 <= index < len(choices)):
                    error(f"Invalid resolution index: {index}. Valid range: 0-{len(choices) - 1}")
                    return False
                value = choices[index]

            camera_was_open = self._camera.is_open
            saved_callback = self._camera._callback
            saved_context = self._camera._callback_context

            if camera_was_open:
                debug("Camera is open, stopping to set resolution")
                self._camera.stop_capture()

            self._camera._hcam.put_eSize(index)
            self.preview_resolution = value

            if camera_was_open:
                debug("Restarting camera after resolution change")
                self._camera.start_capture(saved_callback, saved_context)

            debug(f"Successfully changed preview resolution to {value} (index {index})")
            return True
        except Exception as e:
            error(f"Failed to set resolution: {e}")
            return False

    def set_still_resolution(self, value: str, index: int | None = None) -> bool:
        """
        Set the still-capture resolution.

        The dropdown supplies both the string label (e.g. "2592x1944") and the index
        of that label in the choices list.  When ``index`` is provided it is used
        directly; otherwise it is derived from ``value``.
        """
        try:
            still_resolutions = self.get_still_resolutions()
            choices = [f"{r.width}x{r.height}" for r in still_resolutions]

            if not choices:
                self.still_resolution = value
                return True

            if index is None:
                if value not in choices:
                    error(f"Invalid still resolution value: {value!r}. Available: {choices}")
                    return False
                index = choices.index(value)
            else:
                if not (0 <= index < len(choices)):
                    error(f"Invalid still resolution index: {index}. Valid range: 0-{len(choices) - 1}")
                    return False
                value = choices[index]

            self.still_resolution = value
            debug(f"Successfully changed still resolution to {value} (index {index})")
            return True
        except Exception as e:
            error(f"Failed to set still resolution: {e}")
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
            return self.exposure_time
        
        try:
            return self._camera._hcam.get_ExpoTime()
        except Exception as e:
            error(f"Failed to get exposure time: {e}")
            return self.exposure_time
    
    def apply_to_camera(self, camera: BaseCamera) -> None:
        self._camera = camera
        info(f"Applying settings to camera {camera.model}")
        
        try:
            if self.preview_resolution:
                self.set_preview_resolution(self.preview_resolution)
            if self.still_resolution:
                self.set_still_resolution(self.still_resolution)
            self.set_auto_exposure(self.auto_exposure)
            self.set_exposure(self.exposure)
            self.set_exposure_time(self.exposure_time)
            self.set_gain(self.gain)
            
            self.set_temp_tint(self.temp, self.tint)
            
            self.set_hue(self.hue)
            self.set_saturation(self.saturation)
            self.set_brightness(self.brightness)
            self.set_contrast(self.contrast)
            self.set_gamma(self.gamma)
            
            self.set_level_range(self.level_range_low, self.level_range_high)
            
            self.set_rotate(self.rotate)
            self.set_hflip(self.hflip)
            self.set_vflip(self.vflip)
            
            debug("Successfully applied all settings to camera")
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
            self.exposure_time = hcam.get_ExpoTime()
            self.gain = hcam.get_ExpoAGain()
            temp, tint = hcam.get_TempTint()
            self.temp = temp
            self.tint = tint
            
            self.hue = hcam.get_Hue()
            self.saturation = hcam.get_Saturation()
            self.brightness = hcam.get_Brightness()
            self.contrast = hcam.get_Contrast()
            self.gamma = hcam.get_Gamma()
            
            low, high = hcam.get_LevelRange()
            self.level_range_low = RGBALevel(r=low[0], g=low[1], b=low[2], a=low[3])
            self.level_range_high = RGBALevel(r=high[0], g=high[1], b=high[2], a=high[3])
            
            index = hcam.get_eSize()
            resolutions = self.get_resolutions()
            if 0 <= index < len(resolutions):
                r = resolutions[index]
                self.preview_resolution = f"{r.width}x{r.height}"
            else:
                self.preview_resolution = ""

            still_resolutions = self.get_still_resolutions()
            if still_resolutions:
                r = still_resolutions[0]
                self.still_resolution = f"{r.width}x{r.height}"

            rotate_raw = hcam.get_Option(camera._get_sdk().AMCAM_OPTION_ROTATE)
            self.rotate = rotate_raw if rotate_raw in (0, 90, 180, 270) else 0
            self.hflip = bool(hcam.get_HFlip())
            self.vflip = bool(hcam.get_VFlip())
            
            info("Successfully refreshed all settings from camera")
        except Exception as e:
            exception(f"Failed to refresh settings from camera: {e}")