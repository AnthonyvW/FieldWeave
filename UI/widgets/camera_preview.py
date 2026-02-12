from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QFrame, QLabel, QVBoxLayout, QWidget, QSizePolicy, 
    QPushButton, QHBoxLayout
)

from app_context import get_app_context
from logger import info, error, warning


class OverlayLabel(QLabel):
    """Custom QLabel that can draw overlays on top of the image"""
    
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.show_grid = False
        self.show_crosshair = False
    
    def paintEvent(self, event):
        """Override paint event to draw overlays"""
        # First draw the base image
        super().paintEvent(event)
        
        # Only draw overlays if we have a pixmap
        if self.pixmap() is None or self.pixmap().isNull():
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get the actual image rect (considering aspect ratio)
        pixmap = self.pixmap()
        if pixmap.width() == 0 or pixmap.height() == 0:
            return
        
        # Calculate the displayed image rect
        widget_rect = self.rect()
        pixmap_rect = pixmap.rect()
        
        # Calculate scaled rect maintaining aspect ratio
        scale = min(
            widget_rect.width() / pixmap_rect.width(),
            widget_rect.height() / pixmap_rect.height()
        )
        
        scaled_width = int(pixmap_rect.width() * scale)
        scaled_height = int(pixmap_rect.height() * scale)
        
        x = (widget_rect.width() - scaled_width) // 2
        y = (widget_rect.height() - scaled_height) // 2
        
        image_rect = QRect(x, y, scaled_width, scaled_height)
        
        # Set up pen for drawing overlays
        pen = QPen(QColor(0, 0, 0, 180))  # Black with transparency
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Draw grid if enabled
        if self.show_grid:
            self._draw_grid(painter, image_rect)
        
        # Draw crosshair if enabled
        if self.show_crosshair:
            self._draw_crosshair(painter, image_rect)
        
        painter.end()
    
    def _draw_grid(self, painter: QPainter, rect: QRect):
        """Draw a 3x3 grid"""
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        
        # Vertical lines
        for i in range(1, 3):
            x_pos = x + (w * i // 3)
            painter.drawLine(x_pos, y, x_pos, y + h)
        
        # Horizontal lines
        for i in range(1, 3):
            y_pos = y + (h * i // 3)
            painter.drawLine(x, y_pos, x + w, y_pos)
    
    def _draw_crosshair(self, painter: QPainter, rect: QRect):
        """Draw a crosshair at the center"""
        center_x = rect.x() + rect.width() // 2
        center_y = rect.y() + rect.height() // 2
        
        # Draw horizontal line - smaller (1/24 of smaller dimension)
        line_length = min(rect.width(), rect.height()) // 24
        painter.drawLine(center_x - line_length, center_y, center_x + line_length, center_y)
        
        # Draw vertical line
        painter.drawLine(center_x, center_y - line_length, center_x, center_y + line_length)


class CameraPreview(QFrame):
    """
    Camera preview widget that displays frames from the camera manager.
    This widget only handles display - it does not manage camera lifecycle.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setObjectName("CameraPreview")
        
        # Display state
        self._current_width = 0
        self._current_height = 0
        
        # UI elements - use custom overlay label
        self._video_label = OverlayLabel()
        self._video_label.setObjectName("VideoLabel")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setScaledContents(False)
        self._video_label.setMinimumSize(1, 1)
        self._video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._video_label.setText("No camera stream")
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._video_label, 1)
        
        # Create overlay control buttons as direct children (true overlay)
        self._crosshair_button = QPushButton("+", self)
        self._crosshair_button.setObjectName("CrosshairButton")
        self._crosshair_button.setCheckable(True)
        self._crosshair_button.setFixedSize(30, 30)
        self._crosshair_button.setToolTip("Toggle Crosshair")
        self._crosshair_button.clicked.connect(self._toggle_crosshair)
        self._crosshair_button.move(10, 10)  # Position in top left
        self._crosshair_button.raise_()  # Ensure it's on top
        
        self._grid_button = QPushButton("⌗", self)
        self._grid_button.setObjectName("OverlayButton")
        self._grid_button.setCheckable(True)
        self._grid_button.setFixedSize(30, 30)
        self._grid_button.setToolTip("Toggle Grid")
        self._grid_button.clicked.connect(self._toggle_grid)
        self._grid_button.move(10, 45)  # Position below crosshair button (10 + 30 + 5)
        self._grid_button.raise_()  # Ensure it's on top
        
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
    
    @Slot(bool)
    def _toggle_crosshair(self, checked: bool):
        """Toggle crosshair overlay"""
        self._video_label.show_crosshair = checked
        self._video_label.update()  # Trigger repaint
        info(f"Preview: Crosshair {'enabled' if checked else 'disabled'}")
    
    @Slot(bool)
    def _toggle_grid(self, checked: bool):
        """Toggle grid overlay"""
        self._video_label.show_grid = checked
        self._video_label.update()  # Trigger repaint
        info(f"Preview: Grid {'enabled' if checked else 'disabled'}")
    
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