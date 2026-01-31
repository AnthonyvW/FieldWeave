"""
Camera enumeration system with plugin architecture.
Supports multiple camera types through enumerator plugins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
from logger import error, exception, debug


class CameraType(Enum):
    """Supported camera types"""
    AMSCOPE = "amscope"
    GENERIC_USB = "generic_usb"


@dataclass
class CameraInfo:
    """
    Information about an available camera.
    Lightweight object returned by enumeration before camera instantiation.
    """
    camera_type: CameraType
    device_id: str
    display_name: str
    model: Optional[str] = None
    manufacturer: Optional[str] = None
    serial_number: Optional[str] = None
    max_resolution: Optional[tuple[int, int]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        parts = [self.display_name]
        if self.model:
            parts.append(f"({self.model})")
        if self.serial_number:
            parts.append(f"SN:{self.serial_number}")
        return " ".join(parts)
    
    def __repr__(self) -> str:
        return f"CameraInfo({self.camera_type.value}, {self.display_name})"


class CameraEnumerator(ABC):
    """
    Base class for camera enumerators.
    Each camera type implements this to provide enumeration capability.
    """
    
    @abstractmethod
    def enumerate(self) -> List[CameraInfo]:
        """
        Enumerate all cameras of this type.
        
        Returns:
            List of CameraInfo objects for available cameras
        """
        pass
    
    @abstractmethod
    def get_camera_type(self) -> CameraType:
        """
        Get the camera type this enumerator handles.
        
        Returns:
            CameraType enum value
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this camera type is available (SDK loaded, etc).
        
        Returns:
            True if this camera type can be enumerated
        """
        pass


class AmscopeEnumerator(CameraEnumerator):
    """Enumerator for Amscope cameras"""
    
    def __init__(self):
        self._sdk_loaded = False
        self._sdk = None
        
    def get_camera_type(self) -> CameraType:
        return CameraType.AMSCOPE
    
    def is_available(self) -> bool:
        """Check if Amscope SDK is available"""
        if self._sdk_loaded:
            return self._sdk is not None
            
        try:
            from camera.amscope_camera import AmscopeCamera
            
            # Ensure SDK is loaded
            debug("Loading Amscope SDK...")
            load_result = AmscopeCamera.ensure_sdk_loaded()
            
            if not load_result:
                error("AmscopeCamera.ensure_sdk_loaded() returned False")
                self._sdk_loaded = True
                self._sdk = None
                return False
            
            # Get SDK instance using the private method
            self._sdk = AmscopeCamera._get_sdk()
            self._sdk_loaded = True
            
            if self._sdk is None:
                error("Amscope SDK loaded but _get_sdk() returned None")
                return False
            
            debug("Amscope SDK loaded successfully")
            return True
            
        except ImportError as ie:
            exception(f"Failed to import AmscopeCamera: {ie}")
            self._sdk_loaded = True
            self._sdk = None
            return False
        except RuntimeError as re:
            exception(f"Runtime error loading Amscope SDK: {re}")
            self._sdk_loaded = True
            self._sdk = None
            return False
        except Exception as e:
            exception(f"Unexpected error loading Amscope SDK: {e}")
            self._sdk_loaded = True
            self._sdk = None
            return False
    
    def enumerate(self) -> List[CameraInfo]:
        """Enumerate Amscope cameras"""
        # Ensure SDK is available before enumerating
        if not self.is_available():
            error("Amscope SDK not available, cannot enumerate cameras")
            return []
        
        cameras = []
        
        try:
            # Get SDK (should be loaded now)
            from camera.amscope_camera import AmscopeCamera
            sdk = AmscopeCamera._get_sdk()
            
            if sdk is None:
                error("SDK is None during enumeration")
                return []
            
            # Enumerate devices
            device_list = sdk.Amcam.EnumV2()
            debug(f"Amscope enumerator found {len(device_list)} camera(s)")
            
            for idx, device in enumerate(device_list):
                try:
                    # Get model info
                    model_name = device.model.name if device.model else "Unknown"
                    
                    # Get max resolution
                    max_res = None
                    if device.model and device.model.res and len(device.model.res) > 0:
                        # First resolution is typically the highest
                        max_res = (device.model.res[0].width, device.model.res[0].height)
                    
                    # Create camera info
                    camera_info = CameraInfo(
                        camera_type=CameraType.AMSCOPE,
                        device_id=device.id,
                        display_name=device.displayname or f"Amscope Camera {idx}",
                        model=model_name,
                        manufacturer="Amscope",
                        serial_number=None,  # Could extract from device.id if needed
                        max_resolution=max_res,
                        metadata={
                            'device_index': idx,
                            'model_info': device.model,
                        }
                    )
                    
                    cameras.append(camera_info)
                    
                except Exception as e:
                    exception(f"Error processing Amscope device {idx}: {e}")
                    continue
                    
        except Exception as e:
            exception(f"Error enumerating Amscope cameras: {e}")
        
        return cameras


class GenericUSBEnumerator(CameraEnumerator):
    """
    Enumerator for generic USB cameras (future implementation).
    Placeholder for now.
    """
    
    def get_camera_type(self) -> CameraType:
        return CameraType.GENERIC_USB
    
    def is_available(self) -> bool:
        """Check if OpenCV or other generic USB support is available"""
        try:
            import cv2
            return True
        except ImportError:
            return False
    
    def enumerate(self) -> List[CameraInfo]:
        """Enumerate generic USB cameras (placeholder)"""
        # For now, return empty list
        # Future: Implement using OpenCV or platform-specific APIs
        debug("Generic USB camera enumeration not yet implemented")
        return []
