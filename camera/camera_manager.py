"""
Camera manager for handling camera enumeration, selection, and lifecycle.
Provides plugin architecture for multiple camera types.
"""

from __future__ import annotations

from typing import Optional, List, Callable
from PySide6.QtCore import QObject, Signal

from camera.cameras.base_camera import BaseCamera
from camera.cameras.amscope_camera import AmscopeCamera
from camera.threaded_camera import ThreadedCamera
from camera.camera_enumerator import (
    CameraEnumerator,
    CameraInfo,
    CameraType,
    AmscopeEnumerator,
    GenericUSBEnumerator
)
from logger import info, error, warning, exception, debug


class CameraManager(QObject):
    """
    Manages camera enumeration, selection, and lifecycle.
    
    Signals:
        camera_list_changed: Emitted when available cameras change
        active_camera_changed: Emitted when active camera changes (camera_info or None)
        enumeration_complete: Emitted when camera enumeration completes (camera_count)
    """
    
    camera_list_changed = Signal()
    active_camera_changed = Signal(object)  # CameraInfo or None
    enumeration_complete = Signal(int)  # count
    
    def __init__(self):
        super().__init__()
        
        # Available camera enumerators (plugin architecture)
        self._enumerators: List[CameraEnumerator] = [
            AmscopeEnumerator(),
            GenericUSBEnumerator(),
            # Future: Add more enumerators here
        ]
        
        # Available cameras (from last enumeration)
        self._available_cameras: List[CameraInfo] = []
        
        # Active camera
        self._active_camera: Optional[BaseCamera] = None
        self._active_camera_info: Optional[CameraInfo] = None
        self._camera_thread_started = False
        
        info("Camera manager initialized")
    
    @property
    def available_cameras(self) -> List[CameraInfo]:
        """Get list of available cameras from last enumeration"""
        return self._available_cameras.copy()
    
    @property
    def active_camera(self) -> Optional[BaseCamera]:
        """Get the currently active camera (may be None)"""
        return self._active_camera
    
    @property
    def active_camera_info(self) -> Optional[CameraInfo]:
        """Get info about the currently active camera"""
        return self._active_camera_info
    
    @property
    def has_active_camera(self) -> bool:
        """Check if there is an active camera"""
        return self._active_camera is not None
    
    def enumerate_cameras(self) -> List[CameraInfo]:
        """
        Enumerate all available cameras across all enumerators.
        
        Returns:
            List of CameraInfo objects for all available cameras
        """
        cameras = []
        
        for enumerator in self._enumerators:
            enumerator_type = enumerator.get_camera_type().value
            
            try:
                if enumerator.is_available():
                    enum_cameras = enumerator.enumerate()
                    cameras.extend(enum_cameras)
                else:
                    debug(f"{enumerator_type} enumerator not available")
            except Exception as e:
                exception(f"Error in {enumerator_type} enumerator: {e}")
                continue
        
        self._available_cameras = cameras
        
        # Single clean summary log
        if cameras:
            info(f"Found {len(cameras)} camera(s):")
            for idx, cam in enumerate(cameras):
                info(f"  [{idx}] {cam.display_name} ({cam.model})")
        else:
            info("No cameras found")
        
        self.camera_list_changed.emit()
        self.enumeration_complete.emit(len(cameras))
        
        return cameras
    
    def get_camera_by_id(self, device_id: str) -> Optional[CameraInfo]:
        """
        Find a camera by its device ID.
        
        Args:
            device_id: The device ID to search for
            
        Returns:
            CameraInfo if found, None otherwise
        """
        for camera_info in self._available_cameras:
            if camera_info.device_id == device_id:
                return camera_info
        return None
    
    def get_cameras_by_type(self, camera_type: CameraType) -> List[CameraInfo]:
        """
        Get all cameras of a specific type.
        
        Args:
            camera_type: The camera type to filter by
            
        Returns:
            List of CameraInfo objects matching the type
        """
        return [cam for cam in self._available_cameras if cam.camera_type == camera_type]
    
    def switch_camera(self, camera_info: CameraInfo) -> bool:
        """
        Switch to a different camera.
        Closes the current camera if any, then opens the new one.
        
        Args:
            camera_info: Information about the camera to switch to
            
        Returns:
            True if switch was successful, False otherwise
        """
        info(f"Switching to camera: {camera_info}")
        
        # Close current camera if any
        if self._active_camera is not None:
            info("Closing current camera before switching")
            self.close_camera()
        
        # Create new camera
        camera = self._create_camera_instance(camera_info)
        if camera is None:
            error(f"Failed to create camera instance for {camera_info}")
            return False
        
        # Set camera info if the camera supports it
        if hasattr(camera, 'set_camera_info'):
            # Create the old-style CameraInfo from our new CameraInfo
            from camera.cameras.base_camera import CameraInfo as OldCameraInfo
            old_camera_info = OldCameraInfo(
                id=camera_info.device_id,
                displayname=camera_info.display_name,
                model=camera_info.metadata.get('model_info') if camera_info.metadata else None
            )
            camera.set_camera_info(old_camera_info)
        
        # Wrap in threaded camera
        threaded_camera = ThreadedCamera(camera)
        threaded_camera.start_thread()
        self._camera_thread_started = True
        
        # Open the camera with the device_id
        try:
            info(f"Opening camera: {camera_info.display_name}")
            
            # Call open with device_id and wait=True to ensure it completes
            success, _ = threaded_camera.open(camera_info.device_id, wait=True)
            
            if not success:
                error(f"Failed to open camera: {camera_info}")
                threaded_camera.stop_thread(wait=True)
                return False
            
            # Set as active camera
            self._active_camera = threaded_camera
            self._active_camera_info = camera_info
            
            debug(f"Successfully switched to camera: {camera_info}")
            self.active_camera_changed.emit(camera_info)
            return True
            
        except Exception as e:
            exception(f"Error opening camera: {e}")
            try:
                threaded_camera.stop_thread(wait=True)
            except Exception as stop_error:
                exception(f"Error stopping thread: {stop_error}")
            return False
    
    def open_first_available(self) -> bool:
        """
        Convenience method to enumerate and open the first available camera.
        
        Returns:
            True if a camera was opened, False otherwise
        """
        cameras = self.enumerate_cameras()
        
        if not cameras:
            warning("No cameras available to open")
            return False
        
        # Try to open the first camera
        return self.switch_camera(cameras[0])
    
    def close_camera(self) -> bool:
        """
        Close the currently active camera.
        
        Returns:
            True if successful, False otherwise
        """
        if self._active_camera is None:
            info("No active camera to close")
            return True
        
        info(f"Closing camera: {self._active_camera_info}")
        
        try:
            # Close the camera
            result = self._active_camera.close(wait=True)
            
            if result is not None:
                success, _ = result
                if not success:
                    warning("Camera close returned failure")
            
            # Stop the thread
            if self._camera_thread_started:
                self._active_camera.stop_thread(wait=True)
                self._camera_thread_started = False
            
            # Clear active camera
            self._active_camera = None
            prev_info = self._active_camera_info
            self._active_camera_info = None
            
            info(f"Camera closed: {prev_info}")
            self.active_camera_changed.emit(None)
            return True
            
        except Exception as e:
            exception(f"Error closing camera: {e}")
            
            # Try to stop thread anyway
            try:
                if self._camera_thread_started and self._active_camera:
                    self._active_camera.stop_thread(wait=True)
            except:
                pass
            
            # Clear state
            self._active_camera = None
            self._active_camera_info = None
            self._camera_thread_started = False
            
            self.active_camera_changed.emit(None)
            return False
    
    def _create_camera_instance(self, camera_info: CameraInfo) -> Optional[BaseCamera]:
        """
        Factory method to create camera instance based on camera info.
        
        Note: This only creates the camera instance. The camera must be
        opened separately using camera.open(device_id).
        
        Args:
            camera_info: Information about the camera to create
            
        Returns:
            Camera instance or None if creation failed
        """
        try:
            if camera_info.camera_type == CameraType.AMSCOPE:
                # Create camera instance (does not open it yet)
                camera = AmscopeCamera(camera_info.model)
                return camera
            
            elif camera_info.camera_type == CameraType.GENERIC_USB:
                # Future: Create generic USB camera
                error("Generic USB camera not yet implemented")
                return None
            
            else:
                error(f"Unsupported camera type: {camera_info.camera_type}")
                return None
                
        except Exception as e:
            exception(f"Error creating camera instance: {e}")
            return None
    
    def cleanup(self):
        """Cleanup camera manager resources"""
        info("Cleaning up camera manager")
        
        # Close active camera
        self.close_camera()
        
        # Clear available cameras
        self._available_cameras.clear()
        self.camera_list_changed.emit()
