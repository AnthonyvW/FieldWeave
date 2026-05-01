"""
Camera enumeration system with plugin architecture.
Supports multiple camera types through enumerator plugins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any
from common.logger import error, exception, debug

from camera.cameras.amscope_camera import AmscopeCamera, _get_amcam

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
    model: str | None = None
    manufacturer: str | None = None
    serial_number: str | None = None
    max_resolution: tuple[int, int] | None = None
    metadata: dict[str, Any] | None = None
    
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
    def enumerate(self) -> list[CameraInfo]:
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
        self._sdk = None

    def get_camera_type(self) -> CameraType:
        return CameraType.AMSCOPE

    def is_available(self) -> bool:
        if self._sdk is not None:
            return True
        self._sdk = _get_amcam()
        if self._sdk is None:
            error("Failed to load Amscope SDK")
        return self._sdk is not None

    def enumerate(self) -> list[CameraInfo]:
        if not self.is_available():
            error("Amscope SDK not available, cannot enumerate cameras")
            return []

        cameras = []
        try:
            device_list = self._sdk.Amcam.EnumV2()
            debug(f"Amscope enumerator found {len(device_list)} camera(s)")

            for idx, device in enumerate(device_list):
                try:
                    model_name = device.model.name if device.model else "Unknown"
                    max_res = None
                    if device.model and device.model.res and len(device.model.res) > 0:
                        max_res = (device.model.res[0].width, device.model.res[0].height)

                    cameras.append(CameraInfo(
                        camera_type=CameraType.AMSCOPE,
                        device_id=device.id,
                        display_name=device.displayname or f"Amscope Camera {idx}",
                        model=model_name,
                        manufacturer="Amscope",
                        serial_number=None,
                        max_resolution=max_res,
                        metadata={
                            'device_index': idx,
                            'model_info': device.model,
                        }
                    ))
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
    
    def enumerate(self) -> list[CameraInfo]:
        """Enumerate generic USB cameras (placeholder)"""
        # For now, return empty list
        # Future: Implement using OpenCV or platform-specific APIs
        debug("Generic USB camera enumeration not yet implemented")
        return []