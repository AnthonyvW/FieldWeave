"""
Camera manager for handling camera enumeration, selection, and lifecycle.
Provides plugin architecture for multiple camera types and manages frame acquisition.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

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
from common.logger import info, error, warning, exception, debug


class CameraManager(QObject):
    """
    Manages camera enumeration, selection, lifecycle, and frame acquisition.

    Signals:
        camera_list_changed: Emitted when available cameras change
        active_camera_changed: Emitted when active camera changes (camera_info or None)
        enumeration_complete: Emitted when camera enumeration completes (camera_count)
        preview_frame_ready: Emitted when a new *preview* frame is available (width, height)
        still_frame_ready: Emitted when a new *still* frame is available (width, height)
        streaming_started: Emitted when camera streaming starts (width, height)
        streaming_stopped: Emitted when camera streaming stops
        camera_error: Emitted when a camera error occurs
        camera_disconnected: Emitted when camera is disconnected
    """
    
    camera_list_changed = Signal()
    active_camera_changed = Signal(object)  # CameraInfo or None
    enumeration_complete = Signal(int)  # count
    preview_frame_ready = Signal(int, int)       # preview: width, height
    still_frame_ready = Signal(int, int) # still:   width, height
    streaming_started = Signal(int, int)  # width, height
    streaming_stopped = Signal()
    camera_error = Signal()
    camera_disconnected = Signal()
    
    # Internal signal for forwarding camera events to UI thread
    _camera_event = Signal(int)
    
    def __init__(self):
        super().__init__()
        
        # Available camera enumerators (plugin architecture)
        self._enumerators: list[CameraEnumerator] = [
            AmscopeEnumerator(),
            GenericUSBEnumerator(),
            # Future: Add more enumerators here
        ]
        
        # Available cameras (from last enumeration)
        self._available_cameras: list[CameraInfo] = []
        
        # Active camera
        self._active_camera: BaseCamera | None = None
        self._active_camera_info: CameraInfo | None = None
        self._camera_thread_started = False
        
        # Preview frame state
        self._current_frame_buffer: bytes | None = None
        self._frame_width = 0
        self._frame_height = 0
        self._preview_frame_seq: int = 0

        # Still frame state
        self._current_still_buffer: bytes | None = None
        self._still_frame_width: int = 0
        self._still_frame_height: int = 0
        self._still_frame_seq: int = 0

        self._is_streaming = False
        
        # Connect internal camera event signal
        self._camera_event.connect(self._on_camera_event)
        
        info("Camera manager initialized")
    
    @property
    def available_cameras(self) -> list[CameraInfo]:
        """Get list of available cameras from last enumeration"""
        return self._available_cameras.copy()
    
    @property
    def active_camera(self) -> BaseCamera | None:
        """Get the currently active camera (may be None)"""
        return self._active_camera
    
    @property
    def active_camera_info(self) -> CameraInfo | None:
        """Get info about the currently active camera"""
        return self._active_camera_info
    
    @property
    def has_active_camera(self) -> bool:
        """Check if there is an active camera"""
        return self._active_camera is not None
    
    @property
    def is_streaming(self) -> bool:
        """Check if camera is currently streaming"""
        return self._is_streaming
    
    @property
    def frame_dimensions(self) -> tuple[int, int]:
        """Get current preview frame dimensions (width, height)"""
        return (self._frame_width, self._frame_height)

    @property
    def still_frame_dimensions(self) -> tuple[int, int]:
        """Get current still frame dimensions (width, height)"""
        return (self._still_frame_width, self._still_frame_height)

    @property
    def preview_frame_seq(self) -> int:
        """
        Monotonically increasing counter incremented on every preview frame.

        Compare with ``still_frame_seq`` to determine which type is more
        recent: the higher value was updated last.  Both counters start at 0
        and are reset to 0 when streaming stops or the camera is closed.
        """
        return self._preview_frame_seq

    @property
    def still_frame_seq(self) -> int:
        """
        Monotonically increasing counter incremented on every still frame.

        Compare with ``preview_frame_seq`` to determine which type is more
        recent: the higher value was updated last.  Both counters start at 0
        and are reset to 0 when streaming stops or the camera is closed.
        """
        return self._still_frame_seq
    
    def enumerate_cameras(self) -> list[CameraInfo]:
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
    
    def get_camera_by_id(self, device_id: str) -> CameraInfo | None:
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
    
    def get_cameras_by_type(self, camera_type: CameraType) -> list[CameraInfo]:
        """
        Get all cameras of a specific type.
        
        Args:
            camera_type: The camera type to filter by
            
        Returns:
            List of CameraInfo objects matching the type
        """
        return [cam for cam in self._available_cameras if cam.camera_type == camera_type]
    
    def switch_camera(self, camera_info: CameraInfo, start_streaming: bool = True) -> bool:
        """
        Switch to a different camera.
        Closes the current camera if any, then opens the new one.
        
        Args:
            camera_info: Information about the camera to switch to
            start_streaming: If True, automatically start streaming after opening
            
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
            
            # Start streaming if requested
            if start_streaming:
                self.start_streaming()
            
            return True
            
        except Exception as e:
            exception(f"Error opening camera: {e}")
            try:
                threaded_camera.stop_thread(wait=True)
            except Exception as stop_error:
                exception(f"Error stopping thread: {stop_error}")
            return False
    
    def open_first_available(self, start_streaming: bool = True) -> bool:
        """
        Convenience method to enumerate and open the first available camera.
        
        Args:
            start_streaming: If True, automatically start streaming after opening
        
        Returns:
            True if a camera was opened, False otherwise
        """
        cameras = self.enumerate_cameras()
        
        if not cameras:
            warning("No cameras available to open")
            return False
        
        # Try to open the first camera
        return self.switch_camera(cameras[0], start_streaming=start_streaming)
    
    def start_streaming(self) -> bool:
        """
        Start streaming from the active camera.
        
        Returns:
            True if streaming started successfully, False otherwise
        """
        if not self._active_camera:
            error("Cannot start streaming - no active camera")
            return False
        
        if self._is_streaming:
            debug("Streaming already active")
            return True
        
        try:
            # Get underlying camera
            base_camera = self._active_camera.underlying_camera
            
            # Get current resolution from underlying camera
            res_index, width, height = base_camera.get_current_resolution()
            
            # If no resolution set (0x0), set to first resolution
            if width == 0 or height == 0:
                info("Setting default resolution...")
                
                # Get available resolutions
                resolutions = base_camera.get_resolutions()
                if not resolutions:
                    error("No resolutions available")
                    return False
                
                # Get resolution again after setting
                res_index, width, height = base_camera.get_current_resolution()
            
            # Use final (post-rotation) dimensions for buffer.
            width, height = self._active_camera.settings.get_output_dimensions()
            self._frame_width = width
            self._frame_height = height
            
            # Start capture - use underlying camera directly
            success = base_camera.start_capture(
                self._camera_callback,
                self
            )
            
            if not success:
                error("start_capture returned False")
                return False
            
            self._is_streaming = True
            info(f"Streaming started ({width}x{height})")
            self.streaming_started.emit(width, height)
            return True
            
        except Exception as e:
            error(f"Camera start streaming error: {e}")
            import traceback
            error(traceback.format_exc())
            return False
    
    def stop_streaming(self) -> bool:
        """
        Stop streaming from the active camera.
        
        Returns:
            True if streaming stopped successfully, False otherwise
        """
        if not self._is_streaming:
            debug("Streaming not active")
            return True
        
        try:
            if self._active_camera and self._active_camera.underlying_camera.is_open:
                info("Stopping camera streaming...")
                self._active_camera.underlying_camera.stop_capture()
            
            self._is_streaming = False
            self._current_frame_buffer = None
            self._preview_frame_seq = 0
            self._current_still_buffer = None
            self._still_frame_width = 0
            self._still_frame_height = 0
            self._still_frame_seq = 0
            info("Streaming stopped")
            self.streaming_stopped.emit()
            return True
            
        except Exception as e:
            error(f"Error stopping streaming: {e}")
            return False
    
    def get_current_frame(self) -> bytes | None:
        """
        Get the current preview frame buffer.
        
        Returns:
            Frame buffer as bytes, or None if no preview frame is available
        """
        return self._current_frame_buffer

    def get_current_still_frame(self) -> bytes | None:
        """ Get the most recently captured still frame buffer. """
        return self._current_still_buffer
    
    def copy_current_frame_to_numpy(self) -> np.ndarray | None:
        """
        Copy the current preview frame to a numpy array.
        
        Returns:
            Frame as numpy array (height, width, 3) or None if no frame available
        """
        if not self._current_frame_buffer or self._frame_width == 0 or self._frame_height == 0:
            return None
        
        try:
            # Create numpy array from buffer
            # Calculate stride
            base_camera = self._active_camera.underlying_camera
            base_camera_class = type(base_camera)
            stride = base_camera_class.calculate_stride(self._frame_width, 24)
            
            # Create view of buffer
            arr = np.frombuffer(self._current_frame_buffer, dtype=np.uint8)
            
            # Reshape to image dimensions
            # Note: stride may be larger than width*3 due to alignment
            bytes_per_pixel = 3
            if stride == self._frame_width * bytes_per_pixel:
                # No padding, simple reshape
                return arr.reshape((self._frame_height, self._frame_width, bytes_per_pixel)).copy()
            else:
                # Has padding, need to account for it
                # Reshape to include stride, then slice off padding
                arr_2d = arr.reshape((self._frame_height, stride))
                return arr_2d[:, :self._frame_width * bytes_per_pixel].reshape(
                    (self._frame_height, self._frame_width, bytes_per_pixel)
                ).copy()
                
        except Exception as e:
            error(f"Error converting frame to numpy: {e}")
            return None

    def copy_current_still_frame_to_numpy(self) -> np.ndarray | None:
        """
        Copy the most recently captured still frame to a numpy array.

        Still frames may have different (typically larger) dimensions than
        preview frames, so this method uses ``_still_frame_width/height``
        rather than the preview dimensions.

        Returns:
            Still frame as numpy array (height, width, 3), or None if no still
            has been captured since streaming started.
        """
        if (
            not self._current_still_buffer
            or self._still_frame_width == 0
            or self._still_frame_height == 0
        ):
            return None

        try:
            base_camera = self._active_camera.underlying_camera
            base_camera_class = type(base_camera)
            stride = base_camera_class.calculate_stride(self._still_frame_width, 24)

            arr = np.frombuffer(self._current_still_buffer, dtype=np.uint8)

            bytes_per_pixel = 3
            if stride == self._still_frame_width * bytes_per_pixel:
                return arr.reshape(
                    (self._still_frame_height, self._still_frame_width, bytes_per_pixel)
                ).copy()
            else:
                arr_2d = arr.reshape((self._still_frame_height, stride))
                return arr_2d[:, : self._still_frame_width * bytes_per_pixel].reshape(
                    (self._still_frame_height, self._still_frame_width, bytes_per_pixel)
                ).copy()

        except Exception as e:
            error(f"Error converting still frame to numpy: {e}")
            return None
    
    @staticmethod
    def _camera_callback(event: int, context: Any):
        """
        Camera event callback (called from camera thread).
        Forward to UI thread via signal.
        """
        if isinstance(context, CameraManager):
            # Emit signal to forward to UI thread
            context._camera_event.emit(event)
    
    @Slot(int)
    def _on_camera_event(self, event: int):
        """Handle camera events in UI thread"""
        if not self._active_camera:
            return
        
        # Get underlying camera
        base_camera = self._active_camera.underlying_camera
        
        # Check if camera is open
        if not base_camera.is_open:
            return
        
        # Get event constants from camera
        events = base_camera.get_event_constants()
        
        if event == events.IMAGE:
            self._handle_image_event()
        elif event == events.STILLIMAGE:
            self._handle_still_image_event()
        elif event == events.ERROR:
            self._handle_error()
        elif event == events.DISCONNECTED:
            self._handle_disconnected()
    
    def _handle_image_event(self):
        """Handle new preview image from camera."""
        if not self._active_camera:
            return

        try:
            base_camera = self._active_camera.underlying_camera

            result = base_camera.get_frame_buffer()
            if result is None:
                debug("Preview image event received but frame buffer not yet allocated")
                return

            frame_buf, current_width, current_height = result

            if current_width != self._frame_width or current_height != self._frame_height:
                info(f"Preview resolution changed from {self._frame_width}x{self._frame_height} to {current_width}x{current_height}")
                self._frame_width = current_width
                self._frame_height = current_height

            self._current_frame_buffer = bytes(frame_buf)
            self._preview_frame_seq += 1
            self.preview_frame_ready.emit(self._frame_width, self._frame_height)

        except Exception as e:
            error(f"Error handling preview image event: {e}")

    def _handle_still_image_event(self):
        """Handle a still image event from the camera."""
        if not self._active_camera:
            return

        try:
            base_camera = self._active_camera.underlying_camera

            result = base_camera.get_still_buffer()
            if result is None:
                debug("Still image event received but still buffer not yet allocated")
                return

            still_buf, still_w, still_h = result

            if still_w != self._still_frame_width or still_h != self._still_frame_height:
                info(f"Still resolution changed from {self._still_frame_width}x{self._still_frame_height} to {still_w}x{still_h}")
                self._still_frame_width = still_w
                self._still_frame_height = still_h

            self._current_still_buffer = bytes(still_buf)
            self._still_frame_seq += 1
            self.still_frame_ready.emit(still_w, still_h)

        except Exception as e:
            error(f"Error handling still image event: {e}")
    
    def _handle_error(self):
        """Handle camera error"""
        error("Camera error occurred")
        self.camera_error.emit()
        self.stop_streaming()
    
    def _handle_disconnected(self):
        """Handle camera disconnection"""
        warning("Camera disconnected")
        self.camera_disconnected.emit()
        self.stop_streaming()
    
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
        
        # Stop streaming first
        self.stop_streaming()
        
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
    
    def _create_camera_instance(self, camera_info: CameraInfo) -> BaseCamera | None:
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
        
        # Stop streaming
        self.stop_streaming()
        
        # Close active camera
        self.close_camera()
        
        # Clear available cameras
        self._available_cameras.clear()
        self.camera_list_changed.emit()