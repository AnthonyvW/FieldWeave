from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QFrame, QLabel, QVBoxLayout, QWidget, QSizePolicy,
    QPushButton, QHBoxLayout, QCheckBox
)

from common.app_context import get_app_context
from common.logger import info, error, warning


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
        
        # Channel filter state
        self._show_red = True
        self._show_green = True
        self._show_blue = True
        self._show_grayscale = False
        
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
        self._crosshair_button = QPushButton("⌖", self)
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
        
        # Create focus button with custom overlaid text
        self._focus_button = QPushButton(self)
        self._focus_button.setObjectName("FocusButton")
        self._focus_button.setCheckable(True)
        self._focus_button.setFixedSize(30, 30)
        self._focus_button.setToolTip("Toggle Focus Overlay")
        self._focus_button.clicked.connect(self._toggle_focus)
        self._focus_button.move(10, 80)  # Position below grid button (45 + 30 + 5)
        
        # Create labels for the overlaid symbols - each line separate
        focus_top_corners = QLabel("⌜⌝", self._focus_button)
        focus_top_corners.setObjectName("FocusOverlayLabel")
        focus_top_corners.setAlignment(Qt.AlignmentFlag.AlignCenter)
        focus_top_corners.setGeometry(0, -2, 30, 30)
        focus_top_corners.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        focus_bottom_corners = QLabel("⌞⌟", self._focus_button)
        focus_bottom_corners.setObjectName("FocusOverlayLabel")
        focus_bottom_corners.setAlignment(Qt.AlignmentFlag.AlignCenter)
        focus_bottom_corners.setGeometry(0, 2, 30, 30)
        focus_bottom_corners.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        focus_center = QLabel("⌖", self._focus_button)
        focus_center.setObjectName("FocusOverlayLabel")
        focus_center.setAlignment(Qt.AlignmentFlag.AlignCenter)
        focus_center.setGeometry(0, 0, 30, 30)
        focus_center.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        self._focus_button.raise_()  # Ensure it's on top
        
        # Create channel filter button (venn diagram: 3 overlapping circles)
        self._channel_button = QPushButton(self)
        self._channel_button.setObjectName("ChannelButton")
        self._channel_button.setCheckable(True)
        self._channel_button.setFixedSize(30, 30)
        self._channel_button.setToolTip("Channel Filters")
        self._channel_button.clicked.connect(self._toggle_channel_menu)
        self._channel_button.move(10, 115)  # Position below focus button (80 + 30 + 5)
        self._channel_button.setProperty("ChannelFiltered", False)
        
        # Three overlapping circles for venn diagram icon
        venn_top_left = QLabel("○", self._channel_button)
        venn_top_left.setObjectName("VennOverlayLabel")
        venn_top_left.setAlignment(Qt.AlignmentFlag.AlignCenter)
        venn_top_left.setGeometry(-5, -4, 30, 30)
        venn_top_left.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        venn_top_right = QLabel("○", self._channel_button)
        venn_top_right.setObjectName("VennOverlayLabel")
        venn_top_right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        venn_top_right.setGeometry(5, -4, 30, 30)
        venn_top_right.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        venn_bottom = QLabel("○", self._channel_button)
        venn_bottom.setObjectName("VennOverlayLabel")
        venn_bottom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        venn_bottom.setGeometry(0, 4, 30, 30)
        venn_bottom.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        self._channel_button.raise_()
        
        # Channel filter flyout menu (hidden by default)
        self._channel_menu = QFrame(self)
        self._channel_menu.setObjectName("ChannelMenu")
        self._channel_menu.setFixedWidth(110)
        self._channel_menu.setAutoFillBackground(True)
        self._channel_menu.setFrameShape(QFrame.Shape.StyledPanel)
        self._channel_menu.setFrameShadow(QFrame.Shadow.Raised)
        self._channel_menu.hide()
        
        channel_layout = QVBoxLayout(self._channel_menu)
        channel_layout.setContentsMargins(6, 6, 6, 6)
        channel_layout.setSpacing(4)
        
        self._cb_red = QCheckBox("Red", self._channel_menu)
        self._cb_red.setObjectName("ChannelCheckRed")
        self._cb_red.setChecked(True)
        self._cb_red.toggled.connect(self._on_channel_changed)
        
        self._cb_green = QCheckBox("Green", self._channel_menu)
        self._cb_green.setObjectName("ChannelCheckGreen")
        self._cb_green.setChecked(True)
        self._cb_green.toggled.connect(self._on_channel_changed)
        
        self._cb_blue = QCheckBox("Blue", self._channel_menu)
        self._cb_blue.setObjectName("ChannelCheckBlue")
        self._cb_blue.setChecked(True)
        self._cb_blue.toggled.connect(self._on_channel_changed)
        
        self._cb_gray = QCheckBox("Grayscale", self._channel_menu)
        self._cb_gray.setObjectName("ChannelCheckGray")
        self._cb_gray.setChecked(False)
        self._cb_gray.toggled.connect(self._on_channel_changed)
        
        channel_layout.addWidget(self._cb_red)
        channel_layout.addWidget(self._cb_green)
        channel_layout.addWidget(self._cb_blue)
        channel_layout.addWidget(self._cb_gray)
        self._channel_menu.adjustSize()
        self._channel_menu.raise_()
        
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
    
    @Slot(bool)
    def _toggle_focus(self, checked: bool):
        """Toggle focus overlay"""
        info(f"Focus Overlay Toggled {'on' if checked else 'off'}")
    
    @Slot(bool)
    def _toggle_channel_menu(self, checked: bool):
        """Show or hide the channel filter flyout menu"""
        if checked:
            # Position the menu to the right of the channel button
            btn_pos = self._channel_button.pos()
            self._channel_menu.move(btn_pos.x() + 35, btn_pos.y())
            self._channel_menu.show()
            self._channel_menu.raise_()
        else:
            self._channel_menu.hide()
            # Restore highlight state based purely on filter activity
            self._update_channel_button_highlight()
    
    @Slot()
    def _on_channel_changed(self):
        """Handle channel filter checkbox changes"""
        self._show_red = self._cb_red.isChecked()
        self._show_green = self._cb_green.isChecked()
        self._show_blue = self._cb_blue.isChecked()
        self._show_grayscale = self._cb_gray.isChecked()
        info(
            f"Preview: Channels R={self._show_red} G={self._show_green} "
            f"B={self._show_blue} Gray={self._show_grayscale}"
        )
        # Only update the highlight when the menu is closed; while open the
        # button's checked state already provides visual feedback.
        if not self._channel_menu.isVisible():
            self._update_channel_button_highlight()
    
    def _filters_are_default(self) -> bool:
        """Return True when all channel filters are at their default values."""
        return (
            self._show_red
            and self._show_green
            and self._show_blue
            and not self._show_grayscale
        )
    
    def _update_channel_button_highlight(self) -> None:
        """Set the ChannelFiltered property so stylesheets can highlight the button."""
        active = not self._filters_are_default()
        self._channel_button.setProperty("ChannelFiltered", active)
        # Force the style to re-evaluate the dynamic property
        self._channel_button.style().unpolish(self._channel_button)
        self._channel_button.style().polish(self._channel_button)
    
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
            
            # Apply channel filters if any are disabled or grayscale is on
            needs_filter = (
                self._show_grayscale
                or not self._show_red
                or not self._show_green
                or not self._show_blue
            )
            if needs_filter:
                image = self._apply_channel_filter(image)
            
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
    
    def _apply_channel_filter(self, image: QImage) -> QImage:
        """Return a new QImage with channel filters applied.
        
        Channel masking is applied first, then grayscale conversion operates
        only on the surviving channels so it acts as a modifier rather than
        overriding the channel selection.
        """
        width = image.width()
        height = image.height()
        
        # Convert QImage to numpy array (RGB888 = 3 bytes per pixel)
        ptr = image.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3)).copy()
        
        # Step 1: zero out disabled channels
        if not self._show_red:
            arr[:, :, 0] = 0
        if not self._show_green:
            arr[:, :, 1] = 0
        if not self._show_blue:
            arr[:, :, 2] = 0
        
        # Step 2: if grayscale, compute luminance from the surviving channels
        # and write it to all three output channels unconditionally so the
        # result is always a neutral grey image.  The channel mask above only
        # controls which channels contribute to the luminance calculation.
        if self._show_grayscale:
            r_w = 0.299 if self._show_red else 0.0
            g_w = 0.587 if self._show_green else 0.0
            b_w = 0.114 if self._show_blue else 0.0
            total = r_w + g_w + b_w
            if total > 0:
                r_w, g_w, b_w = r_w / total, g_w / total, b_w / total
            gray = (
                r_w * arr[:, :, 0].astype(np.float32)
                + g_w * arr[:, :, 1].astype(np.float32)
                + b_w * arr[:, :, 2].astype(np.float32)
            ).astype(np.uint8)
            arr[:, :, 0] = gray
            arr[:, :, 1] = gray
            arr[:, :, 2] = gray
        
        filtered = QImage(
            arr.tobytes(),
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        )
        return filtered.copy()
    
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