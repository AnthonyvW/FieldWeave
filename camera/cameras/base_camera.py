"""
Base camera class that defines the interface for camera operations.
All specific camera implementations should inherit from this class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

from common.logger import info, debug, error, exception
from camera.settings.camera_settings import CameraSettings, CameraSettingsManager
from camera.cameras import exif_metadata


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

    def __init__(self, model: str):
        self.model = model
        self._is_open = False
        self._callback = None
        self._callback_context = None
        self._settings_manager: CameraSettingsManager | None = None
        self._settings: CameraSettings | None = None

    @property
    def is_open(self) -> bool:
        """Check if camera is currently open"""
        return self._is_open
    
    @abstractmethod
    def _get_settings_class(self) -> type[CameraSettings]:
        pass

    def initialize_settings(self) -> None:
        """
        Initialize the settings system for this camera.
        Should be called after the camera is opened.
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
    
    def save_settings(self) -> bool:
        """
        Save current settings to config file.

        This creates a backup of the previous settings before saving.
        Call this when the user clicks "Save" or "Apply" in the GUI.

        Returns:
            True if settings were saved successfully, False otherwise

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
        saved = self._settings_manager.save(self._settings)
        if saved:
            info("Settings saved successfully")
        else:
            error("Failed to save settings")
        return saved
    
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
    def capture_still(
        self,
        resolution_index: int | None = None,
        timeout_ms: int = 5000,
        on_captured: Callable[[], None] | None = None,
        on_complete: Callable[[bool, np.ndarray | None], None] | None = None,
    ) -> bool:
        """
        Capture a still image and return it as a numpy array without saving.

        Args:
            resolution_index: Still-resolution index.
            timeout_ms: Maximum time to wait for the frame (milliseconds).
            on_captured: Zero-argument callback fired as soon as the raw frame is ready.
            on_complete: ``(success: bool, image: np.ndarray | None) -> None`` fired
                         once conversion is done.

        Returns:
            True if the snap and pull succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def capture_and_save_still(
        self,
        filepath: Path,
        resolution_index: int | None = None,
        additional_metadata: dict[str, Any] | None = None,
        timeout_ms: int = 5000,
        on_captured: Callable[[], None] | None = None,
        on_image: Callable[[np.ndarray], None] | None = None,
        on_complete: Callable[[bool], None] | None = None,
    ) -> bool:
        """
        Capture a still image and save it with metadata.

        Args:
            filepath: Path where image should be saved.
            resolution_index: Camera resolution to use (0 = highest).
            additional_metadata: Optional dict of extra metadata to save.
            timeout_ms: Timeout for capture in milliseconds.
            on_captured: Zero-argument callback fired as soon as the raw frame is ready.
            on_image: ``(image: np.ndarray) -> None`` fired after conversion but before
                      the file is written.
            on_complete: ``(success: bool) -> None`` fired once the file has been written.

        Returns:
            True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def capture_and_save_stream(
        self,
        filepath: Path,
        additional_metadata: dict[str, Any] | None = None,
        on_image: Callable[[np.ndarray], None] | None = None,
    ) -> bool:
        """
        Capture current stream frame and save it with metadata.

        Args:
            filepath: Path where image should be saved.
            additional_metadata: Optional dict of extra metadata to save.
            on_image: ``(image: np.ndarray) -> None`` fired after the frame is captured
                      but before the file is written.

        Returns:
            True if successful, False otherwise.
        """
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
            See camera.cameras.exif_metadata for exactly how fields are
            distributed across tags/chunks per format.
        """
        pil_image = None
        try:
            # Ensure filepath is a Path object
            filepath = Path(filepath)
            
            # Get camera metadata
            camera_metadata = self._settings.get_camera_metadata()
            
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
                exif_metadata.save_tiff_with_metadata(pil_image, filepath, full_metadata)
            elif ext in ['.jpg', '.jpeg']:
                exif_metadata.save_jpeg_with_metadata(pil_image, filepath, full_metadata)
            elif ext == '.png':
                exif_metadata.save_png_with_metadata(pil_image, filepath, full_metadata)
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