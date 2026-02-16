"""
Base camera class that defines the interface for camera operations.
All specific camera implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Callable, Any, TYPE_CHECKING
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image, ExifTags
from PIL.Image import Exif
from PIL import PngImagePlugin
import json

from common.logger import info, debug, error, exception
from camera.settings.camera_settings import CameraSettings, CameraSettingsManager


@dataclass
class CameraResolution:
    """Represents a camera resolution"""
    width: int
    height: int
    
    def __str__(self):
        return f"{self.width}*{self.height}"


class BaseCamera(ABC):
    """
    Abstract base class for camera operations.
    Defines the interface that all camera implementations must follow.
    """
    
    # Class-level flag to track if SDK has been loaded
    _sdk_loaded = False
    
    def __init__(self, model: str):
        """
        Initialize camera base class.
        
        Args:
            model: Camera model identifier (e.g., "MU500", "MU3000")
        """
        self.model = model
        self._is_open = False
        self._callback = None
        self._callback_context = None
        
        # Settings management (initialized after camera is opened)
        self._settings_manager: CameraSettingsManager | None = None
        self._settings: CameraSettings | None = None

    @property
    def is_open(self) -> bool:
        """Check if camera is currently open"""
        return self._is_open
    
    @classmethod
    @abstractmethod
    def ensure_sdk_loaded(cls, sdk_path: Path | None = None) -> bool:
        """
        Ensure the camera SDK is loaded and ready to use.
        
        This method should be called before any camera operations.
        Implementations should handle:
        - Loading vendor SDK libraries
        - Platform-specific initialization
        - Setting up library search paths
        - Extracting SDK files if needed
        
        Args:
            sdk_path: Optional path to SDK location. If None, use default location.
            
        Returns:
            True if SDK is loaded successfully, False otherwise
            
        Note:
            This is a class method so it can be called before instantiating cameras.
            Most implementations should track SDK load state to avoid reloading.
        """
        pass
    
    @classmethod
    def is_sdk_loaded(cls) -> bool:
        """
        Check if SDK has been loaded.
        
        Returns:
            True if SDK is loaded, False otherwise
        """
        return cls._sdk_loaded
    
    @abstractmethod
    def _get_settings_class(self) -> type[CameraSettings]:
        """
        Get the appropriate settings class for this camera.
        
        This method must be implemented by subclasses to return their
        concrete settings class (e.g., AmscopeSettings, ToupcamSettings).
        
        Returns:
            Concrete CameraSettings subclass for this camera type
            
        Example:
            In AmscopeCamera:
            >>> def _get_settings_class(self):
            ...     from camera.settings.amscope_settings import AmscopeSettings
            ...     return AmscopeSettings
        """
        pass

    def initialize_settings(self) -> None:
        """
        Initialize the settings system for this camera.
        
        This should be called after the camera is opened.
        It creates a settings manager specific to this camera model,
        loads the saved settings (or defaults if none exist), and
        applies them to the camera hardware.
        
        Note:
            The settings manager expects a CameraSettings subclass specific
            to this camera model. The subclass must implement all abstract
            methods from CameraSettings and provide metadata via get_metadata().
        
        Example:
            >>> camera = MU500Camera()
            >>> camera.open("camera_id")
            >>> camera.initialize_settings()
            >>> # Now camera.settings is available
        """
        
        info(f"Initializing settings for {self.model}")
        
        # Create model-specific settings manager
        self._settings_manager = CameraSettingsManager(
            model=self.model,
            settings_class=self._get_settings_class()
        )
        
        # Load saved settings or create defaults
        self._settings = self._settings_manager.load()
        
        # Then apply settings to camera hardware
        self._settings.apply_to_camera(self)
    
    @property
    def settings(self) -> CameraSettings:
        """
        Get the current settings object.
        
        The GUI can use this to read and modify settings.
        
        Returns:
            CameraSettings object for this camera
            
        Raises:
            RuntimeError: If settings haven't been initialized yet
            
        Example:
            >>> # GUI code
            >>> settings = camera.settings
            >>> settings.set_exposure(150)
            >>> settings.set_contrast(10)
            >>> # Changes are immediately applied to camera hardware
        """
        if self._settings is None:
            raise RuntimeError(
                "Settings not initialized. Call initialize_settings() first."
            )
        return self._settings
    
    def save_settings(self) -> None:
        """
        Save current settings to config file.
        
        This creates a backup of the previous settings before saving.
        Call this when the user clicks "Save" or "Apply" in the GUI.
        
        Example:
            >>> # User adjusted settings via GUI
            >>> camera.settings.set_exposure(150)
            >>> camera.settings.set_contrast(10)
            >>> # User clicks "Save"
            >>> camera.save_settings()
        """
        if self._settings is None or self._settings_manager is None:
            raise RuntimeError("Settings not initialized")
        
        info(f"Saving settings for {self.model}")
        self._settings_manager.save(self._settings)
        info("Settings saved successfully")
    
    def load_settings(self, filepath: Path | str | None = None) -> None:
        """
        Load settings from file and apply to camera.
        
        Args:
            filepath: Optional path to load from. If None, loads from default location.
            
        Example:
            >>> # Load from default location
            >>> camera.load_settings()
            >>> 
            >>> # Load from specific file
            >>> camera.load_settings("./saved_configs/night_mode.yaml")
        """
        if self._settings_manager is None:
            raise RuntimeError("Settings not initialized")
        
        info(f"Loading settings for {self.model}")
        
        if filepath is None:
            # Load from default location
            self._settings = self._settings_manager.load()
        else:
            # Load from specific file
            self._settings = self._settings_manager.load_from_file(filepath)
        
        # Refresh to ensure we have camera reference
        self._settings.refresh_from_camera(self)
        
        # Apply to camera hardware
        self._settings.apply_to_camera(self)
        
        info("Settings loaded and applied to camera")
    
    def reset_settings(self) -> None:
        """
        Reset settings to last saved state and apply to camera.
        
        Call this when the user clicks "Cancel" or "Reset" in the GUI.
        
        Example:
            >>> # User made changes but wants to discard them
            >>> camera.reset_settings()
        """
        if self._settings_manager is None:
            raise RuntimeError("Settings not initialized")
        
        info(f"Resetting settings for {self.model}")
        
        # Reload from disk
        self._settings = self._settings_manager.load()
        
        # Refresh to ensure camera reference
        self._settings.refresh_from_camera(self)
        
        # Re-apply to camera
        self._settings.apply_to_camera(self)
        
        info("Settings reset to saved state")
    
    def reset_to_defaults(self) -> None:
        """
        Reset settings to factory defaults and apply to camera.
        
        This also saves the defaults as the current settings.
        
        Example:
            >>> # User wants factory defaults
            >>> camera.reset_to_defaults()
        """
        if self._settings_manager is None:
            raise RuntimeError("Settings not initialized")
        
        info(f"Resetting to factory defaults for {self.model}")
        
        # Restore defaults (this also saves them)
        self._settings = self._settings_manager.restore_defaults()
        
        # Refresh to ensure camera reference
        self._settings.refresh_from_camera(self)
        
        # Apply to camera
        self._settings.apply_to_camera(self)
        
        info("Factory defaults restored and applied")
    
    def refresh_settings_from_camera(self) -> None:
        """
        Read current camera state and update settings object.
        
        Useful if the camera was adjusted outside of the settings system
        (e.g., via hardware buttons or external software).
        
        Example:
            >>> # Camera was adjusted externally
            >>> camera.refresh_settings_from_camera()
            >>> # Now settings object matches camera hardware
        """
        if self._settings is None:
            raise RuntimeError("Settings not initialized")
        
        info("Refreshing settings from camera hardware")
        self._settings.refresh_from_camera(self)
        info("Settings refreshed")
    
    @abstractmethod
    def open(self, camera_id: str) -> bool:
        """
        Open camera connection
        
        Args:
            camera_id: Identifier for the camera to open
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close camera connection and cleanup resources"""
        pass
    
    @abstractmethod
    def start_capture(self, callback: Callable, context: Any) -> bool:
        """
        Start capturing frames
        
        Args:
            callback: Function to call when events occur
            context: Context object to pass to callback
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def stop_capture(self):
        """Stop capturing frames"""
        pass
    
    @abstractmethod
    def pull_image(self, buffer: bytes, bits_per_pixel: int = 24, timeout_ms: int = 1000) -> bool:
        """
        Pull the latest image into provided buffer
        
        Args:
            buffer: Pre-allocated buffer to receive image data
            bits_per_pixel: Bits per pixel (typically 24 for RGB)
            timeout_ms: Timeout in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def snap_image(self, resolution_index: int = 0) -> bool:
        """
        Capture a still image at specified resolution
        
        Args:
            resolution_index: Index of resolution to use
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_camera_metadata(self) -> dict[str, Any]:
        """
        Get camera metadata for image saving.
        
        This method retrieves current camera settings and information
        to be embedded in saved images.
        
        Returns:
            Dictionary containing camera metadata including:
            - model: Camera model name
            - All other camera settings from the settings object
        """
        metadata = {
            "model": self.model,
        }

        # Get all dataclass fields as a dictionary
        settings_dict = asdict(self._settings)
        
        # Remove internal fields and complex types that don't serialize well
        settings_dict.pop("version", None)
        
        # Convert NamedTuples to dicts for better serialization
        for key, value in settings_dict.items():
            if hasattr(value, "_asdict"):
                settings_dict[key] = value._asdict()
        
        # Merge with metadata
        metadata.update(settings_dict)
        
        return metadata
    
    @abstractmethod
    def supports_still_capture(self) -> bool:
        """
        Check if camera supports separate still image capture
        
        Returns:
            True if supported, False otherwise
        """
        pass
    
    @abstractmethod
    def capture_and_save_still(
        self,
        filepath: Path,
        resolution_index: int = 0,
        additional_metadata: dict[str, Any] | None = None,
        timeout_ms: int = 5000
    ) -> bool:
        """
        Capture a still image and save it with metadata.
        
        Args:
            filepath: Path where image should be saved
            resolution_index: Camera resolution to use (0 = highest)
            additional_metadata: Optional dict of extra metadata to save
            timeout_ms: Timeout for capture in milliseconds
            
        Returns:
            True if successful, False otherwise
            
        """
        pass
    
    @abstractmethod
    def capture_and_save_stream(
        self,
        filepath: Path,
        additional_metadata: dict[str, Any] | None = None
    ) -> bool:
        """
        Capture current stream frame and save it with metadata.
        
        Args:
            filepath: Path where image should be saved
            additional_metadata: Optional dict of extra metadata to save
            
        Returns:
            True if successful, False otherwise
            
        """
        pass
    
    @abstractmethod
    def calculate_buffer_size(width: int, height: int, bits_per_pixel: int) -> int:
        pass

    @abstractmethod
    def calculate_stride(width: int, bits_per_pixel: int) -> int:
        pass

    def save_image(
        self,
        image_data: np.ndarray,
        filepath: Path,
        additional_metadata: dict[str, Any] | None = None
    ) -> bool:
        """
        Save image data with embedded metadata.
        
        Args:
            image_data: Image as numpy array (height, width, channels) or (height, width)
            filepath: Path where image should be saved
            additional_metadata: Optional dictionary of additional metadata to save
            
        Returns:
            True if successful, False otherwise
            
        Note:
            - TIFF/TIF: Metadata saved in TIFF tags and as JSON in UserComment
            - JPG/JPEG: Metadata saved in EXIF UserComment as JSON
            - PNG: Metadata saved in PNG text chunks
        """
        pil_image = None
        try:
            # Ensure filepath is a Path object
            filepath = Path(filepath)
            
            # Get camera metadata
            camera_metadata = self.get_camera_metadata()
            
            # Combine with additional metadata
            full_metadata = {
                "timestamp": datetime.now().isoformat(),
                "camera": camera_metadata
            }
            
            if additional_metadata:
                full_metadata["additional"] = additional_metadata
            
            # Convert to PIL Image
            if image_data.dtype != np.uint8:
                # Normalize to uint8 if needed
                if image_data.max() > 255:
                    image_data = (image_data / image_data.max() * 255).astype(np.uint8)
                else:
                    image_data = image_data.astype(np.uint8)
            
            # Handle grayscale vs RGB
            if len(image_data.shape) == 2:
                pil_image = Image.fromarray(image_data, mode='L')
            elif image_data.shape[2] == 3:
                pil_image = Image.fromarray(image_data, mode='RGB')
            elif image_data.shape[2] == 4:
                pil_image = Image.fromarray(image_data, mode='RGBA')
            else:
                error(f"Unsupported image shape: {image_data.shape}")
                return False
            
            # Get file extension
            ext = filepath.suffix.lower()
            
            # Save with format-specific metadata
            if ext in ['.tif', '.tiff']:
                self._save_tiff_with_metadata(pil_image, filepath, full_metadata)
            elif ext in ['.jpg', '.jpeg']:
                self._save_jpeg_with_metadata(pil_image, filepath, full_metadata)
            elif ext == '.png':
                self._save_png_with_metadata(pil_image, filepath, full_metadata)
            else:
                error(f"Unsupported file format: {ext}")
                return False
            
            debug(f"Image saved successfully: {filepath}")
            return True
            
        except Exception as e:
            exception(f"Failed to save image to {filepath}")
            return False
        finally:
            # Explicitly close and delete PIL image to free memory
            if pil_image is not None:
                pil_image.close()
                del pil_image
    
    def _save_tiff_with_metadata(
        self,
        pil_image: Image.Image,
        filepath: Path,
        metadata: dict[str, Any]
    ):
        """Save TIFF with metadata in EXIF tags and UserComment"""
        # Get tag mappings from Base enum
        base_tags = {tag.name: tag.value for tag in ExifTags.Base}
        
        # Create Exif object
        exif = Exif()
        
        # Add software information
        from common.app_context import get_app_context
        exif[base_tags['Software']] = f"FieldWeave - v{get_app_context().settings.version}"
        
        # Add timestamp
        timestamp = metadata.get("timestamp", datetime.now().isoformat())
        exif[base_tags['DateTime']] = datetime.fromisoformat(timestamp).strftime("%Y:%m:%d %H:%M:%S")
        
        # Add camera metadata if available
        camera_meta = metadata.get("camera", {})
        
        # Camera Model
        if "model" in camera_meta:
            exif[base_tags['Model']] = str(camera_meta["model"])
        
        # Get the EXIF IFD to add camera-specific tags
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        
        # Exposure time
        if "exposure_time_us" in camera_meta:
            exposure_sec = camera_meta["exposure_time_us"] / 1_000_000
            exif_ifd[base_tags['ExposureTime']] = (int(exposure_sec * 1_000_000), 1_000_000)
        
        # ISO Speed (using gain as proxy)
        if "gain_percent" in camera_meta:
            iso_value = camera_meta["gain_percent"]
            exif_ifd[base_tags['ISOSpeedRatings']] = iso_value
        
        # Add timestamp to EXIF IFD
        exif_ifd[base_tags['DateTimeOriginal']] = datetime.fromisoformat(timestamp).strftime("%Y:%m:%d %H:%M:%S")
        exif_ifd[base_tags['DateTimeDigitized']] = datetime.fromisoformat(timestamp).strftime("%Y:%m:%d %H:%M:%S")
        
        # Image description from user metadata
        additional_meta = metadata.get("additional", {})
        description_parts = []
        
        if "description" in additional_meta:
            description_parts.append(str(additional_meta["description"]))
        if "sample_id" in additional_meta:
            description_parts.append(f"Sample: {additional_meta['sample_id']}")
        
        if description_parts:
            exif[base_tags['ImageDescription']] = " | ".join(description_parts)
        
        # Store complete metadata as JSON in UserComment
        metadata_json = json.dumps(metadata, indent=2)
        exif_ifd[base_tags['UserComment']] = metadata_json.encode('utf-16')
        
        # Save with EXIF
        pil_image.save(filepath, format='TIFF', exif=exif, compression='tiff_deflate')
        debug(f"TIFF with EXIF metadata saved to {filepath}")
    
    def _save_jpeg_with_metadata(
        self,
        pil_image: Image.Image,
        filepath: Path,
        metadata: dict[str, Any]
    ):
        """Save JPEG with metadata in EXIF tags"""
        # Get tag mappings from Base enum
        base_tags = {tag.name: tag.value for tag in ExifTags.Base}
        
        # Create Exif object
        exif = Exif()
        
        # Add software information
        from common.app_context import get_app_context
        exif[base_tags['Software']] = f"FieldWeave - v{get_app_context().settings.version}"
        
        # Add timestamp
        timestamp = metadata.get("timestamp", datetime.now().isoformat())
        exif[base_tags['DateTime']] = datetime.fromisoformat(timestamp).strftime("%Y:%m:%d %H:%M:%S")
        
        # Add camera metadata
        camera_meta = metadata.get("camera", {})
        
        if "model" in camera_meta:
            exif[base_tags['Model']] = str(camera_meta["model"])
        
        # Image description from additional metadata
        additional_meta = metadata.get("additional", {})
        if "description" in additional_meta:
            exif[base_tags['ImageDescription']] = str(additional_meta["description"])
        elif "sample_id" in additional_meta:
            exif[base_tags['ImageDescription']] = f"Sample: {additional_meta['sample_id']}"
        
        # Get the EXIF IFD
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        
        # Exposure time
        if "exposure_time_us" in camera_meta:
            exposure_sec = camera_meta["exposure_time_us"] / 1_000_000
            exif_ifd[base_tags['ExposureTime']] = (int(exposure_sec * 1_000_000), 1_000_000)
        
        # ISO Speed
        if "gain_percent" in camera_meta:
            iso_value = camera_meta["gain_percent"]
            exif_ifd[base_tags['ISOSpeedRatings']] = iso_value
        
        # Add timestamp to EXIF IFD
        exif_ifd[base_tags['DateTimeOriginal']] = datetime.fromisoformat(timestamp).strftime("%Y:%m:%d %H:%M:%S")
        exif_ifd[base_tags['DateTimeDigitized']] = datetime.fromisoformat(timestamp).strftime("%Y:%m:%d %H:%M:%S")
        
        # Store complete metadata as JSON in UserComment
        metadata_json = json.dumps(metadata, indent=2)
        exif_ifd[base_tags['UserComment']] = metadata_json.encode('utf-16')
        
        # Save with EXIF
        pil_image.save(filepath, format='JPEG', exif=exif, quality=95)
        debug(f"JPEG with EXIF metadata saved to {filepath}")
    
    def _save_png_with_metadata(
        self,
        pil_image: Image.Image,
        filepath: Path,
        metadata: dict[str, Any]
    ):
        """Save PNG with metadata in text chunks"""
        
        # Create PNG info
        pnginfo = PngImagePlugin.PngInfo()
        
        # Add software info
        from common.app_context import get_app_context
        pnginfo.add_text("Software", f"FieldWeave - v{get_app_context().settings.version}")
        pnginfo.add_text("Metadata", json.dumps(metadata, indent=2))
        
        # Add individual camera settings as separate chunks
        camera_meta = metadata.get("camera", {})
        for key, value in camera_meta.items():
            pnginfo.add_text(f"Camera.{key}", str(value))
        
        # Save with metadata
        pil_image.save(filepath, format='PNG', pnginfo=pnginfo)
        debug(f"PNG metadata saved to {filepath}")