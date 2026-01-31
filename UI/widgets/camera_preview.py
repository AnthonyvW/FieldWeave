from __future__ import annotations

from typing import Optional, Any
import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget, QSizePolicy

from app_context import get_app_context
from camera.base_camera import BaseCamera
from logger import info, error, warning

class CameraPreview(QFrame):
    """
    Camera-agnostic Preview Area with live streaming.
    """
    
    # Signal for camera events (thread-safe)
    camera_event = Signal(int)
    
    # Signal when new frame is available for capture
    frame_ready = Signal(np.ndarray)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        # Camera state
        self._camera: BaseCamera | None = None
        self._camera_info = None
        self._img_width = 0
        self._img_height = 0
        self._img_buffer: Optional[bytes] = None
        self._is_streaming = False
        self._no_camera_logged = False
        
        # UI elements
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setScaledContents(False)
        self._video_label.setMinimumSize(1, 1)
        self._video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._video_label.setStyleSheet("color: #888; font-size: 16px;")
        self._video_label.setText("Initializing camera...")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._video_label, 1)
        
        self.setStyleSheet("QFrame { background: #000000; }")
        
        # Timer for checking camera availability
        self._init_timer = QTimer(self)
        self._init_timer.timeout.connect(self._try_initialize_camera)
        
        # Connect camera event signal
        self.camera_event.connect(self._on_camera_event)
        
        # Start initialization
        self._init_timer.start(500)
    
    @Slot()
    def _try_initialize_camera(self):
        """Try to initialize and connect to camera"""
        self._init_timer.stop()
        
        # Get camera from app context
        ctx = get_app_context()
        self._camera: BaseCamera | None = ctx.camera
        
        if self._camera is None:
            self._video_label.setText("No camera available - SDK not loaded")
            error("No camera available - SDK not loaded")
            return
        
        # Get the underlying camera to access class methods
        base_camera_class = type(self._camera.underlying_camera)
        
        # Try to enumerate and connect to first camera
        try:
            cameras = base_camera_class.enumerate_cameras()
            
            if len(cameras) == 0:
                self._video_label.setText("No camera detected")
                if not self._no_camera_logged:
                    warning("No camera connected")
                    self._no_camera_logged = True
                # Retry in a few seconds
                self._init_timer.start(3000)
                return
            
            # Camera found, reset flag
            self._no_camera_logged = False
            
            # Use first camera
            self._camera_info = cameras[0]
            info(f"Found camera: {self._camera_info.displayname}")
            self._open_camera()
            
        except Exception as e:
            self._video_label.setText(f"Camera error: {str(e)}")
            error(f"Camera initialization error: {e}")
            import traceback
            error(traceback.format_exc())
    
    def _open_camera(self):
        """Open and start streaming from camera"""
        if not self._camera or not self._camera_info:
            return
        
        try:
            # Set camera info on underlying camera
            if hasattr(self._camera.underlying_camera, 'set_camera_info'):
                self._camera.underlying_camera.set_camera_info(self._camera_info)
            
            # Open camera (async with callback)
            def on_open_complete(success: bool, result):
                if not success:
                    self._video_label.setText("Failed to open camera")
                    error("Failed to open camera")
                    return
                
                # Camera opened successfully
                self._start_streaming()
            
            # Use async open
            self._camera.open(self._camera_info.id, on_complete=on_open_complete)
            
        except Exception as e:
            self._video_label.setText(f"Error: {str(e)}")
            error(f"Camera open error: {e}")
            import traceback
            error(traceback.format_exc())
    
    def _start_streaming(self):
        """Start camera streaming after camera is opened"""
        if not self._camera:
            return
        
        try:
            # Get current resolution from underlying camera
            res_index, width, height = self._camera.underlying_camera.get_current_resolution()
            
            self._img_width = width
            self._img_height = height
            
            # Calculate buffer size using base camera class method
            base_camera_class = type(self._camera.underlying_camera)
            buffer_size = base_camera_class.calculate_buffer_size(width, height, 24)
            self._img_buffer = bytes(buffer_size)
            
            info("Starting camera stream...")
            # Start capture - use underlying camera directly
            success = self._camera.underlying_camera.start_capture(
                self._camera_callback,
                self
            )
            
            if not success:
                error("start_capture returned False")
                self._camera.underlying_camera.close()
                self._video_label.setText("Failed to start camera stream")
                return
            
            self._is_streaming = True
            # Clear text when streaming starts
            self._video_label.setText("")
            info(f"Streaming started: {self._camera_info.displayname} ({width}x{height})")
            
        except Exception as e:
            self._video_label.setText(f"Error: {str(e)}")
            error(f"Camera start streaming error: {e}")
            import traceback
            error(traceback.format_exc())
    
    @staticmethod
    def _camera_callback(event: int, context: Any):
        """
        Camera event callback (called from camera thread).
        Forward to UI thread via signal.
        """
        if isinstance(context, CameraPreview):
            context.camera_event.emit(event)
    
    @Slot(int)
    def _on_camera_event(self, event: int):
        """Handle camera events in UI thread"""
        if not self._camera:
            return
        
        # Get underlying camera
        base_camera = self._camera.underlying_camera
        
        # Check if camera is open
        if not base_camera.is_open:
            return
        
        # Get event constants from camera
        events = base_camera.get_event_constants()
        
        if event == events.IMAGE:
            self._handle_image_event()
        elif event == events.ERROR:
            self._handle_error()
        elif event == events.DISCONNECTED:
            self._handle_disconnected()
    
    def _handle_image_event(self):
        """Handle new image from camera"""
        if not self._camera or not self._img_buffer:
            return
        
        try:
            # Pull image into buffer from underlying camera
            if self._camera.underlying_camera.pull_image(self._img_buffer, 24):
                # Calculate stride using base camera class method
                base_camera_class = type(self._camera.underlying_camera)
                stride = base_camera_class.calculate_stride(self._img_width, 24)
                
                # Create QImage from buffer
                image = QImage(
                    self._img_buffer,
                    self._img_width,
                    self._img_height,
                    stride,
                    QImage.Format.Format_RGB888
                )
                
                # Make a deep copy for display
                image = image.copy()
                
                # Scale to fit label while maintaining aspect ratio
                if self._video_label.width() > 0 and self._video_label.height() > 0:
                    scaled_image = image.scaled(
                        self._video_label.width(),
                        self._video_label.height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.FastTransformation
                    )
                    self._video_label.setPixmap(QPixmap.fromImage(scaled_image))
        except Exception as e:
            error(f"Error handling image: {e}")
    
    def _handle_error(self):
        """Handle camera error"""
        self._video_label.setText("Camera error occurred")
        error("Camera error occurred")
        self._close_camera()
        # Try to reconnect
        self._init_timer.start(3000)
    
    def _handle_disconnected(self):
        """Handle camera disconnection"""
        self._video_label.setText("Camera disconnected")
        warning("Camera disconnected")
        self._close_camera()
        # Try to reconnect
        self._init_timer.start(3000)
    
    def _close_camera(self):
        """Close camera and cleanup"""
        self._is_streaming = False
        
        if self._camera:
            try:
                # Stop capture first (use underlying camera for immediate effect)
                info("Stopping camera capture...")
                if self._camera.underlying_camera.is_open:
                    self._camera.underlying_camera.stop_capture()
                
                # Close camera (async is fine, we're shutting down)
                info("Closing camera...")
                self._camera.close()
                
            except Exception as e:
                error(f"Error closing camera: {e}")
        
        self._img_buffer = None
    
    def closeEvent(self, event):
        """Handle widget close event"""
        self._close_camera()
        super().closeEvent(event)
    
    def cleanup(self):
        """Cleanup resources when widget is being destroyed"""
        info("Preview cleanup starting...")
        
        # Stop the initialization timer first
        self._init_timer.stop()
        
        # Close camera
        self._close_camera()
        
        info("Preview cleanup complete")