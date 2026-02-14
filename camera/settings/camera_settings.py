from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, TYPE_CHECKING

from generic_config import ConfigManager
from logger import info, debug, exception, error

from common.setting_types import FileFormat, RGBALevel, SettingType, SettingMetadata

if TYPE_CHECKING:
    from camera.cameras.base_camera import BaseCamera, CameraResolution

@dataclass
class CameraSettings(ABC):
    """
    Abstract base camera settings class with validation and hardware manipulation.

    Live-value protocol (for settings driven by automatic hardware control):
    - Mark controlled fields with ``controlled_by="<bool_field_name>"`` in
      SettingMetadata.
    - Override ``get_live_values()`` to return {field_name: current_hw_value} for
      all fields currently being driven by hardware.  Return an empty dict when no
      field is under hardware control.
    - Override ``on_controller_disabled()`` if you need custom flush logic; the
      default calls ``get_live_values()`` and writes each value to self.
    - The GUI polls ``get_live_values()`` on a timer, updates display widgets only,
      and calls ``on_controller_disabled()`` when the controlling boolean turns off.
    """
    
    version: str
    auto_exposure: bool
    exposure: int
    exposure_time: int
    preview_resolution: str
    still_resolution: str
    tint: int
    contrast: int
    hue: int
    saturation: int
    brightness: int
    gamma: int
    level_range_low: RGBALevel
    level_range_high: RGBALevel
    fformat: FileFormat
    
    _camera: BaseCamera | None = field(default=None, repr=False, compare=False)
    _file_formats: tuple[str] = (f.value for f in FileFormat)
    
    def __post_init__(self) -> None:
        if isinstance(self.fformat, str):
            self.fformat = FileFormat(self.fformat)
        
        if isinstance(self.level_range_low, (tuple, list)):
            self.level_range_low = RGBALevel(*self.level_range_low)
        if isinstance(self.level_range_high, (tuple, list)):
            self.level_range_high = RGBALevel(*self.level_range_high)
    
    def validate(self) -> None:
        metadata_list = self.get_metadata()
        metadata_by_name = {m.name: m for m in metadata_list}
        
        for name, meta in metadata_by_name.items():
            if meta.setting_type == SettingType.RANGE:
                value = getattr(self, name, None)
                if value is not None and meta.min_value is not None and meta.max_value is not None:
                    if not (meta.min_value <= value <= meta.max_value):
                        raise ValueError(
                            f"{name} = {value} is outside valid range [{meta.min_value}, {meta.max_value}]"
                        )
        
        try:
            self.level_range_low.validate()
        except ValueError as e:
            raise ValueError(f"level_range_low invalid: {e}") from e
        
        try:
            self.level_range_high.validate()
        except ValueError as e:
            raise ValueError(f"level_range_high invalid: {e}") from e
                
        if not isinstance(self.fformat, FileFormat):
            raise ValueError(f"fformat must be a FileFormat enum, got {type(self.fformat)}")
    
    @abstractmethod
    def get_metadata(cls) -> list[SettingMetadata]:
        pass

    # ------------------------------------------------------------------
    # Live-value protocol
    # ------------------------------------------------------------------

    def get_live_values(self) -> dict[str, int]:
        """Return the current hardware values for any fields under automatic control.

        Returns a mapping of ``{field_name: current_hardware_value}`` for fields
        whose controlling boolean is currently True.  Return an empty dict when no
        field is actively being driven by hardware.

        The GUI polls this on a short interval and updates display widgets without
        writing back to the stored settings object.
        """
        return {}

    def on_controller_disabled(self, controller_name: str) -> None:
        """Flush current hardware values for fields controlled by *controller_name*.

        Called by the GUI immediately after the user turns off a controlling
        boolean (e.g. ``auto_exposure``).  The default implementation reads
        ``get_live_values()`` and writes any value whose metadata ``controlled_by``
        matches *controller_name* back into self, so that the stored settings
        reflect the actual hardware state the moment control was released.

        Subclasses may override for clamping, extra register reads, etc.
        """
        live = self.get_live_values()
        metadata_map = {m.name: m for m in self.get_metadata()}
        for field_name, value in live.items():
            meta = metadata_map.get(field_name)
            if (
                meta
                and meta.controlled_by == controller_name
                and meta.controlled_when  # only flush live-value fields (controlled_when=True)
                and hasattr(self, field_name)
            ):
                setattr(self, field_name, value)
                debug(f"Flushed live value {field_name}={value} after {controller_name} disabled")

    # ------------------------------------------------------------------
    
    def apply_to_camera(self, camera: BaseCamera) -> None:
        self._camera = camera
        info(f"Applying settings to camera {camera.model}")
        
        try:
            self.set_auto_exposure(self.auto_exposure)
            self.set_exposure(self.exposure)
            self.set_tint(self.tint)
            self.set_contrast(self.contrast)
            self.set_hue(self.hue)
            self.set_saturation(self.saturation)
            self.set_brightness(self.brightness)
            self.set_gamma(self.gamma)
            self.set_level_range(self.level_range_low, self.level_range_high)
            
            debug("Successfully applied all settings to camera")
            
        except Exception as e:
            exception(f"Failed to apply settings to camera: {e}")
            raise
    
    @abstractmethod
    def set_auto_exposure(self, enabled: bool) -> None:
        pass
    
    @abstractmethod
    def set_exposure(self, value: int) -> None:
        pass
    
    @abstractmethod
    def set_tint(self, value: int) -> None:
        pass
    
    @abstractmethod
    def set_contrast(self, value: int) -> None:
        pass
    
    @abstractmethod
    def set_hue(self, value: int) -> None:
        pass
    
    @abstractmethod
    def set_saturation(self, value: int) -> None:
        pass
    
    @abstractmethod
    def set_brightness(self, value: int) -> None:
        pass
    
    @abstractmethod
    def set_gamma(self, value: int) -> None:
        pass
    
    @abstractmethod
    def set_level_range(self, low: RGBALevel, high: RGBALevel) -> None:
        pass
    
    def set_fformat(self, value: str, index: int | None = None) -> None:        
        try:
            format_enum = FileFormat(value)
            self.fformat = format_enum
        except ValueError as e:
            raise ValueError(f"Invalid file format: {value}. Must be one of: png, tiff, jpeg") from e
    
    @abstractmethod
    def get_resolutions(self) -> list['CameraResolution']:
        pass
    
    @abstractmethod
    def get_current_resolution(self) -> tuple[int, int, int]:
        pass
    
    @abstractmethod
    def set_preview_resolution(self, value: str, index: int | None = None) -> bool:
        pass

    @abstractmethod
    def set_still_resolution(self, value: str, index: int | None = None) -> bool:
        pass
    
    def get_still_resolutions(self) -> list['CameraResolution']:
        return []
    
    @abstractmethod
    def get_exposure_time(self) -> int:
        pass
    
    @abstractmethod
    def set_exposure_time(self, time_us: int) -> bool:
        pass
    
    @abstractmethod
    def refresh_from_camera(self, camera: BaseCamera) -> None:
        pass


class CameraSettingsManager(ConfigManager[CameraSettings]):
    """
    Settings manager for camera configurations.
    
    Manages camera-specific settings directories and handles serialization
    of camera settings with custom types (RGBALevel, RGBGain, FileFormat).
    """
    
    def __init__(self, model: str, settings_class: type[CameraSettings]):
        self.model = model
        self.settings_class = settings_class
        
        root_dir = Path("./config/cameras") / model
        
        super().__init__(
            config_type=f"camera_settings_{model}",
            root_dir=root_dir
        )
        
        debug(f"Initialized CameraSettingsManager for model '{model}' at {self.root_dir}")
    
    def from_dict(self, data: dict[str, Any]) -> CameraSettings:
        processed_data = data.copy()
        
        if 'level_range_low' in processed_data and isinstance(processed_data['level_range_low'], dict):
            processed_data['level_range_low'] = RGBALevel(**processed_data['level_range_low'])
        
        if 'level_range_high' in processed_data and isinstance(processed_data['level_range_high'], dict):
            processed_data['level_range_high'] = RGBALevel(**processed_data['level_range_high'])
        
        settings = self.settings_class(**processed_data)
        return settings
    
    def to_dict(self, settings: CameraSettings) -> dict[str, Any]:
        data = {}
        
        for field_name in settings.__dataclass_fields__:
            if field_name.startswith('_'):
                continue
            
            value = getattr(settings, field_name)
            
            if isinstance(value, RGBALevel):
                data[field_name] = value._asdict()
            elif isinstance(value, FileFormat):
                data[field_name] = value.value
            elif isinstance(value, Enum):
                data[field_name] = value.value
            else:
                data[field_name] = value
        
        return data