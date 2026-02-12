from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget, QSizePolicy

from app_context import get_app_context
from logger import info, error, warning


class CameraPreview(QFrame):
    """
    Camera preview widget that displays frames from the camera manager.
    This widget only handles display - it does not manage camera lifecycle.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        # Display state
        self._current_width = 0
        self._current_height = 0
        
        # UI elements
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setScaledContents(False)
        self._video_label.setMinimumSize(1, 1)
        self._video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._video_label.setStyleSheet("color: #888; font-size: 16px;")
        self._video_label.setText("No camera stream")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._video_label, 1)
        
        self.setStyleSheet("QFrame { background: #000000; }")
        
        # Connect to camera manager signals
        self._connect_to_camera_manager()
    
    def _connect_to_camera_manager(self):
        """Connect to camera manager signals"""
        ctx = get_app_context()
        camera_manager = ctx.camera_manager
        
        # Connect to frame ready signal
        camera_manager.frame_ready.connect(self._on_frame_ready)
        
        # Connect to streaming status signals
        camera_manager.streaming_started.connect(self._on_streaming_started)
        camera_manager.streaming_stopped.connect(self._on_streaming_stopped)
        
        # Connect to camera status signals
        camera_manager.camera_error.connect(self._on_camera_error)
        camera_manager.camera_disconnected.connect(self._on_camera_disconnected)
        camera_manager.active_camera_changed.connect(self._on_active_camera_changed)
        
        # Update initial state
        if camera_manager.is_streaming:
            width, height = camera_manager.frame_dimensions
            self._on_streaming_started(width, height)
        elif camera_manager.has_active_camera:
            self._video_label.setText("Camera ready - not streaming")
        else:
            self._video_label.setText("No camera connected")
    
    @Slot(int, int)
    def _on_frame_ready(self, width: int, height: int):
        """Handle new frame available from camera manager"""
        ctx = get_app_context()
        camera_manager = ctx.camera_manager
        
        # Get frame buffer from camera manager
        frame_buffer = camera_manager.get_current_frame()
        if not frame_buffer:
            return
        
        try:
            # Check if dimensions changed
            if width != self._current_width or height != self._current_height:
                self._current_width = width
                self._current_height = height
            
            # Calculate stride
            camera = camera_manager.active_camera
            if not camera:
                return
            
            base_camera = camera.underlying_camera
            base_camera_class = type(base_camera)
            stride = base_camera_class.calculate_stride(width, 24)
            
            # Create QImage from buffer
            image = QImage(
                frame_buffer,
                width,
                height,
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
            error(f"Preview: Error displaying frame: {e}")
    
    @Slot(int, int)
    def _on_streaming_started(self, width: int, height: int):
        """Handle streaming started signal"""
        info(f"Preview: Streaming started ({width}x{height})")
        self._current_width = width
        self._current_height = height
        self._video_label.setText("")  # Clear any text when streaming starts
    
    @Slot()
    def _on_streaming_stopped(self):
        """Handle streaming stopped signal"""
        info("Preview: Streaming stopped")
        self._video_label.setText("Camera stream stopped")
    
    @Slot()
    def _on_camera_error(self):
        """Handle camera error"""
        self._video_label.setText("Camera error occurred")
        error("Preview: Camera error occurred")
    
    @Slot()
    def _on_camera_disconnected(self):
        """Handle camera disconnection"""
        self._video_label.setText("Camera disconnected")
        warning("Preview: Camera disconnected")
    
    @Slot(object)
    def _on_active_camera_changed(self, camera_info):
        """Handle active camera changed"""
        if camera_info is None:
            self._video_label.setText("No camera connected")
            info("Preview: No active camera")
        else:
            info(f"Preview: Active camera changed to {camera_info.display_name}")
            # Don't clear text yet - wait for streaming to start
            ctx = get_app_context()
            if not ctx.camera_manager.is_streaming:
                self._video_label.setText("Camera ready - not streaming")
    
    def cleanup(self):
        """Cleanup resources when widget is being destroyed"""
        info("Preview: cleanup starting...")
        
        # Disconnect from camera manager signals
        try:
            ctx = get_app_context()
            camera_manager = ctx.camera_manager
            
            camera_manager.frame_ready.disconnect(self._on_frame_ready)
            camera_manager.streaming_started.disconnect(self._on_streaming_started)
            camera_manager.streaming_stopped.disconnect(self._on_streaming_stopped)
            camera_manager.camera_error.disconnect(self._on_camera_error)
            camera_manager.camera_disconnected.disconnect(self._on_camera_disconnected)
            camera_manager.active_camera_changed.disconnect(self._on_active_camera_changed)
        except Exception as e:
            error(f"Preview: Error disconnecting signals: {e}")
        
        info("Preview cleanup complete")