from __future__ import annotations

from abc import ABC, abstractmethod

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, Union, TYPE_CHECKING
from pathlib import Path

from generic_config import ConfigManager
from logger import info, debug, exception

if TYPE_CHECKING:
    from camera.cameras.base_camera import BaseCamera, CameraResolution


# -------------------------
# Enums for type safety
# -------------------------

class FileFormat(str, Enum):
    """Supported image file formats."""
    PNG = 'png'
    TIFF = 'tiff'
    JPEG = 'jpeg'
    BMP = 'bmp'


# -------------------------
# Type-safe tuples
# -------------------------
class RGBALevel(NamedTuple):
    """RGBA level range values (0-255 each)."""
    r: int
    g: int
    b: int
    a: int
    
    def validate(self) -> None:
        """Ensure all values are in valid range."""
        for name, value in [('r', self.r), ('g', self.g), ('b', self.b), ('a', self.a)]:
            if not (0 <= value <= 255):
                raise ValueError(f"RGBALevel.{name} must be in range [0, 255], got {value}")


class RGBGain(NamedTuple):
    """RGB white balance gain values (-127 to 127 each)."""
    r: int
    g: int
    b: int
    
    def validate(self) -> None:
        """Ensure all values are in valid range."""
        for name, value in [('r', self.r), ('g', self.g), ('b', self.b)]:
            if not (-127 <= value <= 127):
                raise ValueError(f"RGBGain.{name} must be in range [-127, 127], got {value}")


# -------------------------
# GUI Metadata System
# -------------------------
class SettingType(str, Enum):
    """Types of settings for GUI rendering."""
    BOOL = "bool"           # Checkbox
    RANGE = "range"         # Slider with min/max
    DROPDOWN = "dropdown"   # Combo box with choices
    RGBA_LEVEL = "rgba_level"  # Custom RGBA widget
    RGB_GAIN = "rgb_gain"      # Custom RGB widget


@dataclass
class SettingMetadata:
    """
    Metadata describing a setting for GUI generation.
    
    This allows the GUI to automatically create appropriate controls
    for each setting without hardcoding knowledge of the settings.
    """
    name: str                           # Parameter name (e.g., "exposure")
    display_name: str                   # Human-readable name (e.g., "Exposure")
    setting_type: SettingType           # Type of control to render
    description: str = ""               # Tooltip/help text
    
    # For RANGE type
    min_value: int | None = None
    max_value: int | None = None
    
    # For DROPDOWN type
    choices: list[str] | None = None
    
    # Grouping for organized GUI
    group: str = "General"
    
    # Whether this setting can be changed while camera is running
    runtime_changeable: bool = True


# -------------------------
# Settings dataclass
# -------------------------
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
                 SettingMetadata(
                     name="temp",
                     display_name="Color Temperature",
                     setting_type=SettingType.RANGE,
                     description="White balance temperature in Kelvin",
                     min_value=2000,
                     max_value=15000,
                     group="White Balance",
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
             # Get range from metadata
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
       - Example:
         def refresh_from_camera(self, camera: BaseCamera) -> None:
             self._camera = camera
             if hasattr(camera, '_device'):
                 self.auto_expo = bool(camera._sdk.get_AutoExpoEnable(camera._device))
                 self.exposure = camera._sdk.get_AutoExpoTarget(camera._device)
                 # ... read all other settings
    
    4. Implement resolution and exposure/gain methods:
       - get_resolutions(), set_resolution(), get_current_resolution()
       - get_exposure_time(), set_exposure_time()
       - get_gain(), set_gain()
    
    Note on Default Values:
    - Default values are loaded from YAML files by CameraSettingsManager
    - Subclasses do NOT need a create_default() method
    - The YAML file serves as the default configuration
    
    Example Complete Subclass:
        class AmScopeSettings(CameraSettings):
            @classmethod
            def get_metadata(cls) -> list[SettingMetadata]:
                return [
                    SettingMetadata(name="exposure", display_name="Exposure", ...),
                    # ... all settings with ranges
                ]
            
            def set_exposure(self, value: int) -> None:
                # Validate using metadata, update field, access SDK
                pass
            
            def refresh_from_camera(self, camera: BaseCamera) -> None:
                # Read all settings from SDK
                pass
            
            # ... implement all other abstract methods
    """
    
    # Version tracking
    version: str
    
    # Image processing parameters (subclasses must provide defaults via factory methods)
    auto_expo: bool
    exposure: int                      # Auto Exposure Target
    exposure_time_us: int              # Manual exposure time in microseconds
    gain_percent: int                  # Gain in percent
    resolution_index: int              # Selected resolution index
    temp: int                          # White balance temperature
    tint: int                          # White balance tint
    contrast: int
    hue: int
    saturation: int
    brightness: int
    gamma: int
    
    # Complex parameters (subclasses must provide defaults via factory methods)
    levelrange_low: RGBALevel
    levelrange_high: RGBALevel
    wbgain: RGBGain
    
    # Tone mapping and format (subclasses must provide defaults via factory methods)
    fformat: FileFormat
    
    # Private - reference to camera (set by apply_to_camera)
    _camera: BaseCamera | None = field(default=None, repr=False, compare=False)
    
    @classmethod
    @abstractmethod
    def get_metadata(cls) -> list[SettingMetadata]:
        """
        Get metadata for all settings to enable GUI generation and validation.
        
        This is the SINGLE SOURCE OF TRUTH for setting information including:
        - Display names and descriptions
        - Valid ranges (min/max values)
        - Setting types (bool, range, dropdown, etc.)
        - Grouping and organization
        
        Subclasses MUST override this method to provide metadata specific to their camera model.
        
        Returns:
            List of SettingMetadata objects describing each setting
        
        Example implementation in AmScopeSettings:
            @classmethod
            def get_metadata(cls) -> list[SettingMetadata]:
                return [
                    SettingMetadata(
                        name="auto_expo",
                        display_name="Auto Exposure",
                        setting_type=SettingType.BOOL,
                        description="Enable automatic exposure control",
                        group="Exposure",
                    ),
                    SettingMetadata(
                        name="exposure",
                        display_name="Exposure Target",
                        setting_type=SettingType.RANGE,
                        description="Target brightness for auto exposure",
                        min_value=16,
                        max_value=220,
                        group="Exposure",
                    ),
                    SettingMetadata(
                        name="temp",
                        display_name="Color Temperature",
                        setting_type=SettingType.RANGE,
                        description="White balance temperature in Kelvin",
                        min_value=2000,
                        max_value=15000,
                        group="White Balance",
                    ),
                    # ... all other settings
                ]
        """
        pass
    
    def validate(self) -> None:
        """
        Validate all settings are within acceptable ranges.
        
        Uses get_metadata() as the single source of truth for valid ranges.
        
        Raises:
            ValueError: If any parameter is outside its valid range
        """
        metadata_list = self.get_metadata()
        metadata_by_name = {m.name: m for m in metadata_list}
        
        # Validate simple numeric parameters
        for name, meta in metadata_by_name.items():
            if meta.setting_type == SettingType.RANGE:
                value = getattr(self, name, None)
                if value is not None and meta.min_value is not None and meta.max_value is not None:
                    if not (meta.min_value <= value <= meta.max_value):
                        raise ValueError(
                            f"{name} = {value} is outside valid range [{meta.min_value}, {meta.max_value}]"
                        )
        
        # Validate complex types
        try:
            self.levelrange_low.validate()
        except ValueError as e:
            raise ValueError(f"levelrange_low invalid: {e}") from e
        
        try:
            self.levelrange_high.validate()
        except ValueError as e:
            raise ValueError(f"levelrange_high invalid: {e}") from e
        
        try:
            self.wbgain.validate()
        except ValueError as e:
            raise ValueError(f"wbgain invalid: {e}") from e
        
        # Validate enum types
        if not isinstance(self.fformat, FileFormat):
            raise ValueError(f"fformat must be a FileFormat enum, got {type(self.fformat)}")
    
    def __post_init__(self) -> None:
        """
        Post-initialization hook to ensure enums are converted from strings.
        """
        # Convert string values to enums if needed
        if isinstance(self.fformat, str):
            self.fformat = FileFormat(self.fformat)
        
        # Convert tuples/lists to NamedTuples if needed
        if isinstance(self.levelrange_low, (tuple, list)):
            self.levelrange_low = RGBALevel(*self.levelrange_low)
        if isinstance(self.levelrange_high, (tuple, list)):
            self.levelrange_high = RGBALevel(*self.levelrange_high)
        if isinstance(self.wbgain, (tuple, list)):
            self.wbgain = RGBGain(*self.wbgain)
    
    # -------------------------
    # Camera Manipulation
    # -------------------------
    
    def apply_to_camera(self, camera: BaseCamera) -> None:
        """
        Apply all settings to the camera hardware.
        
        This is the main entry point for pushing settings to the camera.
        It calls individual setter methods which handle the low-level
        camera API calls.
        
        Args:
            camera: The camera instance to apply settings to
            
        Example:
            >>> settings = manager.load()
            >>> settings.apply_to_camera(camera)
        """
        self._camera = camera
        info(f"Applying settings to camera {camera.model}")
        
        try:
            # Apply each setting in logical order
            self.set_auto_exposure(self.auto_expo)
            self.set_exposure(self.exposure)
            self.set_temperature(self.temp)
            self.set_tint(self.tint)
            self.set_wb_gain(self.wbgain)
            self.set_contrast(self.contrast)
            self.set_hue(self.hue)
            self.set_saturation(self.saturation)
            self.set_brightness(self.brightness)
            self.set_gamma(self.gamma)
            self.set_level_range(self.levelrange_low, self.levelrange_high)
            
            info("Successfully applied all settings to camera")
            
        except Exception as e:
            exception(f"Failed to apply settings to camera: {e}")
            raise
    
    # Abstract setter methods - subclasses MUST implement these
    # Each method should:
    # 1. Validate the input value
    # 2. Update the corresponding field
    # 3. Access the SDK/hardware directly to apply the change
    # 4. Log the change with debug()
    
    @abstractmethod
    def set_auto_exposure(self, enabled: bool) -> None:
        """
        Enable/disable auto exposure.
        
        Subclasses must implement to access SDK/hardware directly.
        
        Example:
            if self._camera and hasattr(self._camera, '_sdk'):
                self._camera._sdk.put_AutoExpoEnable(self._camera._device, enabled)
        """
        pass
    
    @abstractmethod
    def set_exposure(self, value: int) -> None:
        """
        Set exposure target.
        
        Subclasses must:
        1. Validate value against ranges from get_metadata()
        2. Update field: self.exposure = value
        3. Access hardware directly
        """
        pass
    
    @abstractmethod
    def set_temperature(self, value: int) -> None:
        """
        Set white balance temperature.
        
        Subclasses must:
        1. Validate value against ranges from get_metadata()
        2. Update field: self.temp = value
        3. Access hardware directly
        """
        pass
    
    @abstractmethod
    def set_tint(self, value: int) -> None:
        """
        Set white balance tint.
        
        Subclasses must:
        1. Validate value against ranges from get_metadata()
        2. Update field: self.tint = value
        3. Access hardware directly
        """
        pass
    
    @abstractmethod
    def set_wb_gain(self, gain: RGBGain) -> None:
        """
        Set RGB white balance gains.
        
        Subclasses must:
        1. Validate: gain.validate()
        2. Update field: self.wbgain = gain
        3. Access hardware directly
        """
        pass
    
    @abstractmethod
    def set_contrast(self, value: int) -> None:
        """
        Set contrast.
        
        Subclasses must:
        1. Validate value against ranges from get_metadata()
        2. Update field: self.contrast = value
        3. Access hardware directly
        """
        pass
    
    @abstractmethod
    def set_hue(self, value: int) -> None:
        """
        Set hue.
        
        Subclasses must:
        1. Validate value against ranges from get_metadata()
        2. Update field: self.hue = value
        3. Access hardware directly
        """
        pass
    
    @abstractmethod
    def set_saturation(self, value: int) -> None:
        """
        Set saturation.
        
        Subclasses must:
        1. Validate value against ranges from get_metadata()
        2. Update field: self.saturation = value
        3. Access hardware directly
        """
        pass
    
    @abstractmethod
    def set_brightness(self, value: int) -> None:
        """
        Set brightness.
        
        Subclasses must:
        1. Validate value against ranges from get_metadata()
        2. Update field: self.brightness = value
        3. Access hardware directly
        """
        pass
    
    @abstractmethod
    def set_gamma(self, value: int) -> None:
        """
        Set gamma correction.
        
        Subclasses must:
        1. Validate value against ranges from get_metadata()
        2. Update field: self.gamma = value
        3. Access hardware directly
        """
        pass
    
    @abstractmethod
    def set_level_range(self, low: RGBALevel, high: RGBALevel) -> None:
        """
        Set level range mapping.
        
        Subclasses must:
        1. Validate: low.validate() and high.validate()
        2. Update fields: self.levelrange_low = low, self.levelrange_high = high
        3. Access hardware directly
        """
        pass
    
    # Resolution settings
    
    @abstractmethod
    def get_resolutions(self) -> list['CameraResolution']:
        """
        Get available camera resolutions.
        
        Returns:
            List of available resolutions
            
        Example:
            return [
                CameraResolution(width=2592, height=1944),
                CameraResolution(width=1920, height=1080),
                CameraResolution(width=1280, height=720),
            ]
        """
        pass
    
    @abstractmethod
    def get_current_resolution(self) -> tuple[int, int, int]:
        """
        Get current resolution.
        
        Returns:
            Tuple of (resolution_index, width, height)
        """
        pass
    
    @abstractmethod
    def set_resolution(self, resolution_index: int) -> bool:
        """
        Set camera resolution.
        
        Args:
            resolution_index: Index of resolution to use
            
        Returns:
            True if successful, False otherwise
            
        Subclasses must:
        1. Validate resolution_index is valid
        2. Update field: self.resolution_index = resolution_index
        3. Access hardware to change resolution
        """
        pass
    
    def get_still_resolutions(self) -> list['CameraResolution']:
        """
        Get available still image resolutions.
        
        For cameras that support separate still image capture at
        different resolutions than the video stream.
        
        Returns:
            List of available still resolutions
        """
        return []
    
    # Exposure time settings (manual exposure)
    
    @abstractmethod
    def get_exposure_time(self) -> int:
        """
        Get current exposure time.
        
        Returns:
            Current exposure time in microseconds
        """
        pass
    
    @abstractmethod
    def set_exposure_time(self, time_us: int) -> bool:
        """
        Set exposure time (manual exposure control).
        
        Args:
            time_us: Exposure time in microseconds
            
        Returns:
            True if successful, False otherwise
            
        Subclasses must:
        1. Validate time_us against ranges from get_metadata()
        2. Update field: self.exposure_time_us = time_us
        3. Access hardware to set exposure time
        """
        pass
    
    # Gain settings
    
    @abstractmethod
    def get_gain(self) -> int:
        """
        Get current gain.
        
        Returns:
            Current gain in percent
        """
        pass
    
    @abstractmethod
    def set_gain(self, gain_percent: int) -> bool:
        """
        Set gain.
        
        Args:
            gain_percent: Gain in percent
            
        Returns:
            True if successful, False otherwise
            
        Subclasses must:
        1. Validate gain_percent against ranges from get_metadata()
        2. Update field: self.gain_percent = gain_percent
        3. Access hardware to set gain
        """
        pass
    
    # Getter methods - read current values from camera
    
    @abstractmethod
    def refresh_from_camera(self, camera: BaseCamera) -> None:
        """
        Read all current settings from camera hardware.
        
        Subclasses MUST override this method to read settings from their SDK/hardware.
        
        This is useful to sync the settings object with the actual
        camera state, for example after manual adjustments or after
        camera initialization.
        
        Args:
            camera: The camera instance to read from
            
        Example implementation in AmScopeSettings:
            def refresh_from_camera(self, camera: BaseCamera) -> None:
                self._camera = camera
                info(f"Refreshing settings from camera {camera.model}")
                
                if hasattr(camera, '_device'):
                    self.auto_expo = bool(camera._sdk.get_AutoExpoEnable(camera._device))
                    self.exposure = camera._sdk.get_AutoExpoTarget(camera._device)
                    temp, tint = camera._sdk.get_TempTint(camera._device)
                    self.temp = temp
                    self.tint = tint
                    # ... read all other settings from SDK
                    
                info("Successfully refreshed all settings from camera")
        """
        pass


# -------------------------
# Specialized manager
# -------------------------
class CameraSettingsManager(ConfigManager[CameraSettings]):
    """
    Specialized configuration manager for a single camera model.
    
    Each camera model should have its own manager instance.
    This ensures settings don't bleed between incompatible models.
    
    Example usage:
        >>> # Create manager for MU500
        >>> manager = CameraSettingsManager(model="MU500")
        >>> 
        >>> # Load settings and apply to camera
        >>> settings = manager.load()
        >>> settings.apply_to_camera(camera)
        >>> 
        >>> # User changes settings via GUI...
        >>> settings.set_exposure(150)
        >>> settings.set_contrast(10)
        >>> 
        >>> # Save when user clicks "Save"
        >>> manager.save(settings)
        >>> 
        >>> # Reset to saved settings
        >>> settings = manager.load()
        >>> settings.apply_to_camera(camera)
        >>> 
        >>> # Reset to factory defaults
        >>> settings = manager.restore_defaults()
        >>> settings.apply_to_camera(camera)
    """
    
    def __init__(
        self,
        *,
        model: str,
        base_dir: Union[str, Path] = "./config/cameras",
        default_filename: str = "default_settings.yaml",
        backup_dirname: str = "backups",
        backup_keep: int = 5,
    ) -> None:
        # Set root_dir to the model-specific directory
        model_dir = Path(base_dir) / model
        
        super().__init__(
            CameraSettings,
            root_dir=model_dir,
            default_filename=default_filename,
            backup_dirname=backup_dirname,
            backup_keep=backup_keep,
        )
        
        self.model = model
        info(f"Initialized CameraSettingsManager for model '{model}' at {model_dir}")