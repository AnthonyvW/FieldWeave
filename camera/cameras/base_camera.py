"""
Base camera class that defines the interface for camera operations.
All specific camera implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable, Any, Dict
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image, ExifTags
from PIL.Image import Exif
from PIL.TiffImagePlugin import ImageFileDirectory_v2
import json


@dataclass
class CameraResolution:
    """Represents a camera resolution"""
    width: int
    height: int
    
    def __str__(self):
        return f"{self.width}*{self.height}"


@dataclass
class CameraInfo:
    """Basic camera information"""
    id: str
    displayname: str
    model: Any  # Model-specific information


class BaseCamera(ABC):
    """
    Abstract base class for camera operations.
    Defines the interface that all camera implementations must follow.
    
    Each camera implementation should handle its own SDK loading in the
    ensure_sdk_loaded() method. This is typically called once before any
    camera operations.
    """
    
    # Class-level flag to track if SDK has been loaded
    _sdk_loaded = False
    
    def __init__(self):
        self._is_open = False
        self._callback = None
        self._callback_context = None
        
    @property
    def is_open(self) -> bool:
        """Check if camera is currently open"""
        return self._is_open
    
    @classmethod
    @abstractmethod
    def ensure_sdk_loaded(cls, sdk_path: Optional[Path] = None) -> bool:
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
    def get_resolutions(self) -> list[CameraResolution]:
        """
        Get available camera resolutions
        
        Returns:
            List of available resolutions
        """
        pass
    
    @abstractmethod
    def get_current_resolution(self) -> Tuple[int, int, int]:
        """
        Get current resolution
        
        Returns:
            Tuple of (resolution_index, width, height)
        """
        pass
    
    @abstractmethod
    def set_resolution(self, resolution_index: int) -> bool:
        """
        Set camera resolution
        
        Args:
            resolution_index: Index of resolution to use
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_exposure_range(self) -> Tuple[int, int, int]:
        """
        Get exposure time range
        
        Returns:
            Tuple of (min, max, default) values
        """
        pass
    
    @abstractmethod
    def get_exposure_time(self) -> int:
        """
        Get current exposure time
        
        Returns:
            Current exposure time in microseconds
        """
        pass
    
    @abstractmethod
    def set_exposure_time(self, time_us: int) -> bool:
        """
        Set exposure time
        
        Args:
            time_us: Exposure time in microseconds
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_gain_range(self) -> Tuple[int, int, int]:
        """
        Get gain range
        
        Returns:
            Tuple of (min, max, default) values in percent
        """
        pass
    
    @abstractmethod
    def get_gain(self) -> int:
        """
        Get current gain
        
        Returns:
            Current gain in percent
        """
        pass
    
    @abstractmethod
    def set_gain(self, gain_percent: int) -> bool:
        """
        Set gain
        
        Args:
            gain_percent: Gain in percent
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_auto_exposure(self) -> bool:
        """
        Get auto exposure state
        
        Returns:
            True if auto exposure is enabled, False otherwise
        """
        pass
    
    @abstractmethod
    def set_auto_exposure(self, enabled: bool) -> bool:
        """
        Set auto exposure state
        
        Args:
            enabled: True to enable, False to disable
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def supports_white_balance(self) -> bool:
        """
        Check if camera supports white balance
        
        Returns:
            True if white balance is supported, False otherwise
        """
        pass
    
    @abstractmethod
    def get_white_balance_range(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Get white balance range
        
        Returns:
            Tuple of ((temp_min, temp_max), (tint_min, tint_max))
        """
        pass
    
    @abstractmethod
    def get_white_balance(self) -> Tuple[int, int]:
        """
        Get current white balance
        
        Returns:
            Tuple of (temperature, tint)
        """
        pass
    
    @abstractmethod
    def set_white_balance(self, temperature: int, tint: int) -> bool:
        """
        Set white balance
        
        Args:
            temperature: Color temperature value
            tint: Tint value
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def auto_white_balance(self) -> bool:
        """
        Perform one-time auto white balance
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_frame_rate(self) -> Tuple[int, int, int]:
        """
        Get current frame rate information
        
        Returns:
            Tuple of (frames_in_period, time_period_ms, total_frames)
        """
        pass
    
    @abstractmethod
    def get_camera_metadata(self) -> Dict[str, Any]:
        """
        Get current camera settings as metadata
        
        Returns:
            Dictionary of camera settings (exposure, gain, white balance, etc.)
        """
        pass
    
    @abstractmethod
    def supports_still_capture(self) -> bool:
        """
        Check if camera supports separate still image capture
        
        Returns:
            True if supported, False otherwise
        """
        pass
    
    @abstractmethod
    def get_still_resolutions(self) -> list[CameraResolution]:
        """
        Get available still image resolutions
        
        Returns:
            List of available still resolutions
        """
        pass
    
    @abstractmethod
    def capture_and_save_still(
        self,
        filepath: Path,
        resolution_index: int = 0,
        additional_metadata: Optional[Dict[str, Any]] = None,
        timeout_ms: int = 5000
    ) -> bool:
        """
        Capture a still image and save it to disk with metadata.
        
        This is a convenience method that handles the complete workflow:
        1. Triggers still image capture
        2. Waits for image to be ready
        3. Pulls image data
        4. Saves with metadata
        
        Args:
            filepath: Path where image should be saved
            resolution_index: Resolution index for still capture (0 = highest)
            additional_metadata: Optional dictionary of additional metadata to save
            timeout_ms: Timeout in milliseconds to wait for capture
            
        Returns:
            True if successful, False otherwise
            
        Note:
            Only works if supports_still_capture() returns True
        """
        pass
    
    @abstractmethod
    def capture_and_save_stream(
        self,
        filepath: Path,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Capture current frame from live stream and save it to disk with metadata.
        
        This is a convenience method that handles the complete workflow:
        1. Pulls current frame from live stream
        2. Converts to numpy array
        3. Saves with metadata
        
        Args:
            filepath: Path where image should be saved
            additional_metadata: Optional dictionary of additional metadata to save
            
        Returns:
            True if successful, False otherwise
            
        Note:
            Camera must be in capture mode (start_capture() must have been called)
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
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save image to disk with camera and optional additional metadata.
        
        Supports formats: TIFF, TIF, JPG, JPEG, PNG
        
        Args:
            image_data: Image as numpy array (height, width, channels) or (height, width)
            filepath: Path where image should be saved
            additional_metadata: Optional dictionary of additional metadata to save
            
        Returns:
            True if successful, False otherwise
            
        Note:
            - TIFF/TIF: Metadata saved in TIFF tags and as JSON in ImageDescription
            - JPG/JPEG: Metadata saved in EXIF UserComment as JSON
            - PNG: Metadata saved in PNG text chunks
        """
        pil_image = None
        try:
            from logger import get_logger
            logger = get_logger()
            
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
                logger.error(f"Unsupported image shape: {image_data.shape}")
                return False
            
            # Get file extension
            ext = filepath.suffix.lower()
            
            # Save with format-specific metadata
            if ext in ['.tif', '.tiff']:
                self._save_tiff_with_metadata(pil_image, filepath, full_metadata, logger)
            elif ext in ['.jpg', '.jpeg']:
                self._save_jpeg_with_metadata(pil_image, filepath, full_metadata, logger)
            elif ext == '.png':
                self._save_png_with_metadata(pil_image, filepath, full_metadata, logger)
            else:
                logger.error(f"Unsupported file format: {ext}")
                return False
            
            logger.info(f"Image saved successfully: {filepath}")
            return True
            
        except Exception as e:
            try:
                from logger import get_logger
                logger = get_logger()
                logger.exception(f"Failed to save image to {filepath}")
            except:
                print(f"Failed to save image to {filepath}: {e}")
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
        metadata: Dict[str, Any],
        logger
    ):
        """Save TIFF with metadata in EXIF tags and ImageDescription"""
        # Get tag mappings from Base enum
        base_tags = {tag.name: tag.value for tag in ExifTags.Base}
        
        # Create Exif object
        exif = Exif()
        
        # Add software information - placeholder for version
        exif[base_tags['Software']] = "Forge - v{VERSION_PLACEHOLDER}"
        
        # Add timestamp
        timestamp = metadata.get("timestamp", datetime.now().isoformat())
        exif[base_tags['DateTime']] = datetime.fromisoformat(timestamp).strftime("%Y:%m:%d %H:%M:%S")
        
        # Add camera metadata if available
        camera_meta = metadata.get("camera", {})
        
        # Camera Make and Model
        if "model" in camera_meta:
            exif[base_tags['Model']] = str(camera_meta["model"])
        
        # Get the EXIF IFD to add camera-specific tags
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        
        # Exposure time (tag ExposureTime)
        if "exposure_time_us" in camera_meta:
            exposure_sec = camera_meta["exposure_time_us"] / 1_000_000
            # Store as rational (numerator, denominator)
            exif_ifd[base_tags['ExposureTime']] = (int(exposure_sec * 1_000_000), 1_000_000)
        
        # ISO Speed (tag ISOSpeedRatings)
        if "gain_percent" in camera_meta:
            iso_value = camera_meta["gain_percent"]
            exif_ifd[base_tags['ISOSpeedRatings']] = iso_value
        
        # Add timestamp to EXIF IFD as well
        exif_ifd[base_tags['DateTimeOriginal']] = datetime.fromisoformat(timestamp).strftime("%Y:%m:%d %H:%M:%S")
        exif_ifd[base_tags['DateTimeDigitized']] = datetime.fromisoformat(timestamp).strftime("%Y:%m:%d %H:%M:%S")
        
        # Image description from user-provided metadata only
        additional_meta = metadata.get("additional", {})
        description_parts = []
        
        if "description" in additional_meta:
            description_parts.append(str(additional_meta["description"]))
        if "sample_id" in additional_meta:
            description_parts.append(f"Sample: {additional_meta['sample_id']}")
        
        # Only set ImageDescription if user provided a description
        if description_parts:
            exif[base_tags['ImageDescription']] = " | ".join(description_parts)
        
        # Store complete metadata as JSON in UserComment instead
        metadata_json = json.dumps(metadata, indent=2)
        exif_ifd[base_tags['UserComment']] = metadata_json.encode('utf-16')
        
        # Save with EXIF
        pil_image.save(filepath, format='TIFF', exif=exif, compression='tiff_deflate')
        logger.debug(f"TIFF with EXIF metadata saved to {filepath}")
    
    def _save_jpeg_with_metadata(
        self,
        pil_image: Image.Image,
        filepath: Path,
        metadata: Dict[str, Any],
        logger
    ):
        """Save JPEG with metadata in proper EXIF tags"""
        # Get tag mappings from Base enum
        base_tags = {tag.name: tag.value for tag in ExifTags.Base}
        
        # Create Exif object
        exif = Exif()
        
        # Add software information - placeholder for version
        exif[base_tags['Software']] = "Forge - v{VERSION_PLACEHOLDER}"
        
        # Add timestamp
        timestamp = metadata.get("timestamp", datetime.now().isoformat())
        exif[base_tags['DateTime']] = datetime.fromisoformat(timestamp).strftime("%Y:%m:%d %H:%M:%S")
        
        # Add camera metadata if available
        camera_meta = metadata.get("camera", {})
        
        # Camera Make and Model
        if "model" in camera_meta:
            exif[base_tags['Model']] = str(camera_meta["model"])
        
        # Image description from additional metadata
        additional_meta = metadata.get("additional", {})
        if "description" in additional_meta:
            exif[base_tags['ImageDescription']] = str(additional_meta["description"])
        elif "sample_id" in additional_meta:
            exif[base_tags['ImageDescription']] = f"Sample: {additional_meta['sample_id']}"
        
        # Get the EXIF IFD to add camera-specific tags
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        
        # Exposure time
        if "exposure_time_us" in camera_meta:
            exposure_sec = camera_meta["exposure_time_us"] / 1_000_000
            # Store as rational (numerator, denominator)
            exif_ifd[base_tags['ExposureTime']] = (int(exposure_sec * 1_000_000), 1_000_000)
        
        # ISO Speed
        if "gain_percent" in camera_meta:
            # Map gain percent to ISO-like value
            iso_value = camera_meta["gain_percent"]
            exif_ifd[base_tags['ISOSpeedRatings']] = iso_value
        
        # Add timestamp to EXIF IFD
        exif_ifd[base_tags['DateTimeOriginal']] = datetime.fromisoformat(timestamp).strftime("%Y:%m:%d %H:%M:%S")
        exif_ifd[base_tags['DateTimeDigitized']] = datetime.fromisoformat(timestamp).strftime("%Y:%m:%d %H:%M:%S")
        
        # Store complete metadata as JSON in UserComment for full data preservation
        metadata_json = json.dumps(metadata, indent=2)
        exif_ifd[base_tags['UserComment']] = metadata_json.encode('utf-16')
        
        # Save with EXIF
        pil_image.save(filepath, format='JPEG', exif=exif, quality=95)
        logger.debug(f"JPEG with EXIF metadata saved to {filepath}")
    
    def _save_png_with_metadata(
        self,
        pil_image: Image.Image,
        filepath: Path,
        metadata: Dict[str, Any],
        logger
    ):
        """Save PNG with metadata in text chunks"""
        from PIL import PngImagePlugin
        
        # Create PNG info
        pnginfo = PngImagePlugin.PngInfo()
        
        # Add metadata as text chunks
        pnginfo.add_text("Software", "Forge - v{VERSION_PLACEHOLDER}")
        pnginfo.add_text("Metadata", json.dumps(metadata, indent=2))
        
        # Add individual camera settings as separate chunks for easier access
        camera_meta = metadata.get("camera", {})
        for key, value in camera_meta.items():
            pnginfo.add_text(f"Camera.{key}", str(value))
        
        # Save with metadata
        pil_image.save(filepath, format='PNG', pnginfo=pnginfo)
        logger.debug(f"PNG metadata saved to {filepath}")
