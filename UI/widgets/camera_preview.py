from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QWheelEvent
from PySide6.QtWidgets import (
    QFrame, QLabel, QVBoxLayout, QWidget, QSizePolicy,
)

from common.app_context import get_app_context
from common.logger import info, error, warning
from UI.widgets.preview_overlay.channel import ChannelButton, ChannelOverlay
from UI.widgets.preview_overlay.crosshair import CrosshairButton, CrosshairOverlay
from UI.widgets.preview_overlay.focus import FocusButton, FocusOverlay
from UI.widgets.preview_overlay.grid import GridButton, GridOverlay
from UI.widgets.preview_overlay.overlay_base import Overlay


class OverlayLabel(QLabel):
    """QLabel that drives a list of Overlay instances on each paint and frame."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._overlays: list[Overlay] = []

    def add_overlay(self, overlay: Overlay) -> None:
        self._overlays.append(overlay)

    def notify_full(self, frame: np.ndarray) -> None:
        """Forward the full-resolution frame to every enabled overlay."""
        for overlay in self._overlays:
            if overlay.enabled:
                overlay.update_full(frame)

    def notify_scaled(self, frame: np.ndarray) -> None:
        """Forward the display-resolution frame to every enabled overlay."""
        for overlay in self._overlays:
            if overlay.enabled:
                overlay.update_scaled(frame)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)

        if self.pixmap() is None or self.pixmap().isNull():
            return

        pixmap = self.pixmap()
        if pixmap.width() == 0 or pixmap.height() == 0:
            return

        image_rect = self._image_rect(pixmap)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(QColor(0, 0, 0, 180))
        pen.setWidth(2)
        painter.setPen(pen)

        for overlay in self._overlays:
            if overlay.enabled:
                overlay.draw(painter, image_rect)

        painter.end()

    def _image_rect(self, pixmap: QPixmap) -> QRect:
        widget_rect = self.rect()
        pixmap_rect = pixmap.rect()
        scale = min(
            widget_rect.width() / pixmap_rect.width(),
            widget_rect.height() / pixmap_rect.height(),
        )
        scaled_width = int(pixmap_rect.width() * scale)
        scaled_height = int(pixmap_rect.height() * scale)
        x = (widget_rect.width() - scaled_width) // 2
        y = (widget_rect.height() - scaled_height) // 2
        return QRect(x, y, scaled_width, scaled_height)


class CameraPreview(QFrame):
    """
    Camera preview widget that displays frames from the camera manager.
    This widget only handles display - it does not manage camera lifecycle.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setObjectName("CameraPreview")

        self._current_width = 0
        self._current_height = 0

        self._video_label = OverlayLabel()
        self._video_label.setObjectName("VideoLabel")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setScaledContents(False)
        self._video_label.setMinimumSize(1, 1)
        self._video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._video_label.setText("No camera stream")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._video_label, 1)

        # Overlays
        self._crosshair_overlay = CrosshairOverlay()
        self._grid_overlay = GridOverlay()
        self._focus_overlay = FocusOverlay()
        self._channel_overlay = ChannelOverlay()

        self._video_label.add_overlay(self._crosshair_overlay)
        self._video_label.add_overlay(self._grid_overlay)
        self._video_label.add_overlay(self._focus_overlay)

        # Buttons
        self._crosshair_button = CrosshairButton(self)
        self._crosshair_button.move(10, 10)
        self._crosshair_button.raise_()
        self._crosshair_button.toggled_crosshair.connect(self._crosshair_overlay.set_enabled)
        self._crosshair_button.toggled_crosshair.connect(self._video_label.update)

        self._grid_button = GridButton(self)
        self._grid_button.move(10, 45)
        self._grid_button.raise_()
        self._grid_button.toggled_grid.connect(self._grid_overlay.set_enabled)
        self._grid_button.toggled_grid.connect(self._video_label.update)

        self._focus_button = FocusButton(self)
        self._focus_button.move(10, 80)
        self._focus_button.raise_()
        self._focus_button.toggled_focus.connect(self._focus_overlay.set_enabled)
        self._focus_button.toggled_focus.connect(self._video_label.update)

        self._channel_button = ChannelButton(self)
        self._channel_button.move(10, 115)
        self._channel_button.raise_()
        self._channel_button.menu.raise_()
        self._channel_button.channel_changed.connect(self._on_channel_changed)

        self._connect_to_camera_manager()

    def _connect_to_camera_manager(self) -> None:
        ctx = get_app_context()
        camera_manager = ctx.camera_manager

        camera_manager.frame_ready.connect(self._on_frame_ready)
        camera_manager.streaming_started.connect(self._on_streaming_started)
        camera_manager.streaming_stopped.connect(self._on_streaming_stopped)
        camera_manager.camera_error.connect(self._on_camera_error)
        camera_manager.camera_disconnected.connect(self._on_camera_disconnected)
        camera_manager.active_camera_changed.connect(self._on_active_camera_changed)

        if camera_manager.is_streaming:
            width, height = camera_manager.frame_dimensions
            self._on_streaming_started(width, height)
        elif camera_manager.has_active_camera:
            self._video_label.setText("Camera ready - not streaming")
        else:
            self._video_label.setText("No camera connected")

    @Slot(bool, bool, bool, bool)
    def _on_channel_changed(
        self,
        show_red: bool,
        show_green: bool,
        show_blue: bool,
        show_grayscale: bool,
    ) -> None:
        self._channel_overlay.show_red = show_red
        self._channel_overlay.show_green = show_green
        self._channel_overlay.show_blue = show_blue
        self._channel_overlay.show_grayscale = show_grayscale

    @Slot(int, int)
    def _on_frame_ready(self, width: int, height: int) -> None:
        ctx = get_app_context()
        camera_manager = ctx.camera_manager

        frame_buffer = camera_manager.get_current_frame()
        if not frame_buffer:
            return

        try:
            if width != self._current_width or height != self._current_height:
                self._current_width = width
                self._current_height = height

            camera = camera_manager.active_camera
            if not camera:
                return

            base_camera_class = type(camera.underlying_camera)
            stride = base_camera_class.calculate_stride(width, 24)

            image = QImage(
                frame_buffer, width, height, stride, QImage.Format.Format_RGB888
            ).copy()

            if self._channel_overlay.needs_filter:
                image = self._channel_overlay.apply(image)

            # Notify overlays at full camera resolution before scaling.
            ptr = image.bits()
            full_arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
                (image.height(), image.bytesPerLine())
            )[:, : image.width() * 3].reshape(
                (image.height(), image.width(), 3)
            ).copy()
            self._video_label.notify_full(full_arr)

            if self._video_label.width() > 0 and self._video_label.height() > 0:
                scaled = image.scaled(
                    self._video_label.width(),
                    self._video_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )

                # Notify overlays at display resolution before setting the pixmap.
                # QImage rows may be padded, so reshape using bytesPerLine() as
                # the row stride then slice back to the actual pixel columns.
                ptr = scaled.bits()
                scaled_arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
                    (scaled.height(), scaled.bytesPerLine())
                )[:, : scaled.width() * 3].reshape(
                    (scaled.height(), scaled.width(), 3)
                ).copy()
                self._video_label.notify_scaled(scaled_arr)

                self._video_label.setPixmap(QPixmap.fromImage(scaled))

        except Exception as e:
            error(f"Preview: Error displaying frame: {e}")

    @Slot(int, int)
    def _on_streaming_started(self, width: int, height: int) -> None:
        info(f"Preview: Streaming started ({width}x{height})")
        self._current_width = width
        self._current_height = height
        self._video_label.setText("")

    @Slot()
    def _on_streaming_stopped(self) -> None:
        info("Preview: Streaming stopped")
        self._video_label.setText("Camera stream stopped")

    @Slot()
    def _on_camera_error(self) -> None:
        self._video_label.setText("Camera error occurred")
        error("Preview: Camera error occurred")

    @Slot()
    def _on_camera_disconnected(self) -> None:
        self._video_label.setText("Camera disconnected")
        warning("Preview: Camera disconnected")

    @Slot(object)
    def _on_active_camera_changed(self, camera_info) -> None:
        if camera_info is None:
            self._video_label.setText("No camera connected")
            info("Preview: No active camera")
        else:
            info(f"Preview: Active camera changed to {camera_info.display_name}")
            ctx = get_app_context()
            if not ctx.camera_manager.is_streaming:
                self._video_label.setText("Camera ready - not streaming")

    _SCROLL_STEP_NM: int = 40_000  # 0.04 mm in nanometres

    def wheelEvent(self, event: QWheelEvent) -> None:
        ctx = get_app_context()
        if ctx.motion is None:
            warning("CameraPreview: scroll Z ignored — motion controller not ready")
            event.accept()
            return

        delta = event.angleDelta().y()
        if delta == 0:
            event.accept()
            return

        direction = 1 if delta > 0 else -1
        ctx.motion.move("z", self._SCROLL_STEP_NM * direction)
        event.accept()

    def cleanup(self) -> None:
        info("Preview: cleanup starting...")
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