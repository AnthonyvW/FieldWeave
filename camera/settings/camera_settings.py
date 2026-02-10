from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, TYPE_CHECKING

from generic_config import ConfigManager
from logger import info, debug, exception, error

if TYPE_CHECKING:
    from camera.cameras.base_camera import BaseCamera, CameraResolution


class FileFormat(str, Enum):
    PNG = 'png'
    TIFF = 'tiff'
    JPEG = 'jpeg'


class RGBALevel(NamedTuple):
    r: int
    g: int
    b: int
    a: int
    
    def validate(self) -> None:
        for name, value in [('r', self.r), ('g', self.g), ('b', self.b), ('a', self.a)]:
            if not (0 <= value <= 255):
                raise ValueError(f"RGBALevel.{name} must be in range [0, 255], got {value}")


class SettingType(str, Enum):
    BOOL = "bool"
    RANGE = "range"
    DROPDOWN = "dropdown"
    RGBA_LEVEL = "rgba_level"


@dataclass
class SettingMetadata:
    name: str
    display_name: str
    setting_type: SettingType
    description: str = ""
    min_value: int | None = None
    max_value: int | None = None
    choices: list[str] | None = None
    group: str = "General"
    runtime_changeable: bool = True


@dataclass
class CameraSettings(ABC):
    """
    Abstract base camera settings class with validation and hardware manipulation.
    
    This is an abstract base class that MUST be subclassed for each camera type.
    Subclasses must implement all abstract methods and provide camera-specific configuration.
    
    Architecture:
    - CameraSettings owns settings storage, validation, and hardware access
    - For cameras with SDKs (like AmScope): subclass accesses camera._sdk directly
    - For cameras without SDKs (like USB): subclass implements direct hardware access
    - BaseCamera provides camera operations, SDK loading, and settings management
    - CameraSettingsManager handles loading/saving settings from YAML files
    
    Responsibilities of CameraSettings:
    1. Storage of settings values (dataclass fields)
    2. Validation of settings (using metadata from get_metadata())
    3. High-level API (set_* methods with validation)
    4. Low-level hardware access (directly to SDK/hardware via abstract methods)
    5. Applying settings to camera hardware
    6. Reading settings from camera hardware
    7. Providing metadata for GUI generation and validation (single source of truth)
    
    Requirements for Subclasses:
    
    1. Implement get_metadata() class method (SINGLE SOURCE OF TRUTH):
       - Return SettingMetadata list for GUI generation AND validation
       - Include ranges, descriptions, groups, types for ALL settings
       - This replaces the old get_ranges() method - no duplication needed
       - Example:
         @classmethod
         def get_metadata(cls) -> list[SettingMetadata]:
             return [
                 SettingMetadata(
                     name="exposure",
                     display_name="Exposure Target",
                     setting_type=SettingType.RANGE,
                     description="Target brightness for auto exposure",
                     min_value=16,
                     max_value=220,
                     group="Exposure",
                 ),
                 # ... all other settings
             ]
    
    2. Implement all abstract setter methods (set_exposure, set_temp, etc.):
       - Get validation ranges from get_metadata()
       - Validate input against those ranges
       - Update the corresponding dataclass field
       - Access hardware directly to apply the setting
       - Example:
         def set_exposure(self, value: int) -> None:
             metadata = {m.name: m for m in self.get_metadata()}
             meta = metadata['exposure']
             if not (meta.min_value <= value <= meta.max_value):
                 raise ValueError(f"exposure must be in [{meta.min_value}, {meta.max_value}]")
             
             self.exposure = value
             if self._camera and hasattr(self._camera, '_sdk'):
                 self._camera._sdk.put_AutoExpoTarget(self._camera._device, value)
    
    3. Implement refresh_from_camera():
       - Read all current settings from camera hardware
       - Update all dataclass fields with hardware values
    
    4. Implement resolution and exposure methods:
       - get_resolutions(), set_resolution(), get_current_resolution()
       - get_still_resolutions(), set_still_resolution()
       - get_exposure_time(), set_exposure_time()
    
    Note on Default Values:
    - Default values are loaded from YAML files by CameraSettingsManager
    - Subclasses do NOT need a create_default() method
    - The YAML file serves as the default configuration
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