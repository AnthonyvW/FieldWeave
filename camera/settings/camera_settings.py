from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, TYPE_CHECKING

import numpy as np

from common.generic_config import ConfigManager
from common.logger import info, debug, exception, error

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
    contrast: int
    hue: int
    saturation: int
    brightness: int
    gamma: int
    fformat: FileFormat

    _camera: BaseCamera | None = field(default=None, repr=False, compare=False)
    _file_formats: tuple[str, ...] = tuple(f.value for f in FileFormat)
    _ui_update_callback: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.fformat, str):
            self.fformat = FileFormat(self.fformat)
    
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
            self.set_contrast(self.contrast)
            self.set_hue(self.hue)
            self.set_saturation(self.saturation)
            self.set_brightness(self.brightness)
            self.set_gamma(self.gamma)
            
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
    
    def set_fformat(self, value: str, index: int | None = None) -> None:
        normalized = value.lower()
        format_enum = FileFormat(normalized) if normalized in FileFormat._value2member_map_ else None
        if format_enum is None:
            raise ValueError(f"Invalid file format: {value}. Must be one of: {', '.join(f.value for f in FileFormat)}")
        self.fformat = format_enum
        if self._ui_update_callback is not None:
            self._ui_update_callback("fformat", format_enum)
    
    @abstractmethod
    def get_preview_resolutions(self) -> list['CameraResolution']:
        pass
    
    @abstractmethod
    def get_current_preview_resolution(self) -> tuple[int, int, int]:
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
    def supports_still_capture(self) -> bool:
        """
        Check if the camera supports separate still image capture.

        Returns:
            True if supported, False otherwise.
        """
        pass

    def get_camera_metadata(self) -> dict[str, Any]:
        """
        Get camera metadata for image saving.

        Retrieves current settings to be embedded in saved images. Walks
        get_metadata() rather than the raw dataclass fields, since that is
        the authoritative list of settings a given camera exposes — a
        subclass's fields (e.g. Amscope fan/TEC controls) only show up here
        if get_metadata() declares them.

        Returns:
            Dictionary containing camera metadata including the model name
            (if a camera is attached) and every setting registered via
            get_metadata().
        """
        metadata: dict[str, Any] = {}

        if self._camera is not None:
            metadata["model"] = self._camera.model

        for meta in self.get_metadata():
            if not hasattr(self, meta.name):
                continue

            value = getattr(self, meta.name)
            if hasattr(value, "_asdict"):
                value = value._asdict()
            elif isinstance(value, Enum):
                value = value.value

            metadata[meta.name] = value

        return metadata

    def field_default(self, name: str) -> Any:
        """The dataclass-declared default for *name*, or None if there isn't one.

        Used as the "normal" baseline for direction_code() — the factory
        default rather than the middle of the field's valid range, since a
        hardware-probed range (see UsbCameraSettings) isn't necessarily
        centered on where the camera actually ships.
        """
        for f in fields(self):
            if f.name == name:
                return f.default
        return None

    def direction_code(self, name: str) -> int:
        """EXIF Contrast/Saturation-style code (0=Normal, 1=Low, 2=High) for
        field *name*, relative to its dataclass default.
        """
        value = getattr(self, name)
        default = self.field_default(name)
        if value > default:
            return 2
        if value < default:
            return 1
        return 0

    def supports_histogram(self) -> bool:
        """
        Check if this camera supports histogram retrieval.

        Subclasses that support histograms should override this to return True.

        Returns:
            True if histogram retrieval is supported, False otherwise.
        """
        return False

    def set_histogram_enabled(self, enabled: bool) -> bool:
        """
        Enable or disable automatic per-frame histogram capture.

        When enabled, implementations should capture a histogram alongside each
        preview frame and each still frame, storing the results so they can be
        retrieved instantly via ``get_preview_histogram()`` /
        ``get_still_histogram()`` without any further round-trip to the camera.

        Only valid if ``supports_histogram()`` returns True.  Subclasses that
        support histograms must override this method.

        Args:
            enabled: True to enable histogram capture, False to disable.

        Returns:
            True if the operation succeeded, False otherwise.

        Raises:
            NotImplementedError: If this camera does not support histograms.
        """
        if not self.supports_histogram():
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support histogram control"
            )
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement set_histogram_enabled()"
        )

    def get_preview_histogram(self) -> np.ndarray | None:
        """
        Return the most recently captured preview-frame histogram, or None if
        histogram capture is disabled or no frame has arrived yet.

        Implementations store a histogram with each incoming preview frame so
        this method returns immediately without any camera round-trip.

        Returns:
            A copy of the latest histogram as a float64 numpy array with shape
            ``(channels, bins)`` and values normalised to [0, 1], or None.

        Raises:
            NotImplementedError: If this camera does not support histograms.
        """
        if not self.supports_histogram():
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support histograms"
            )
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_preview_histogram()"
        )

    def get_still_histogram(self) -> np.ndarray | None:
        """
        Return the most recently captured still-image histogram, or None if no
        still has been taken with histogram capture enabled.

        Returns:
            A copy of the latest still histogram as a float64 numpy array with
            shape ``(channels, bins)`` and values normalised to [0, 1], or None.

        Raises:
            NotImplementedError: If this camera does not support histograms.
        """
        if not self.supports_histogram():
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support histograms"
            )
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_still_histogram()"
        )

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
        hints = self.settings_class.__dataclass_fields__
        for field_name, field_obj in hints.items():
            if field_name not in processed_data:
                continue
            value = processed_data[field_name]
            if isinstance(value, dict) and field_obj.type in ("RGBALevel", RGBALevel):
                processed_data[field_name] = RGBALevel(**value)
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