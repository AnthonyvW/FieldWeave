from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot, QRect, QPoint, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QWheelEvent, QMouseEvent, QPainterPath
from PySide6.QtWidgets import (
    QFrame, QLabel, QPushButton, QVBoxLayout, QWidget, QSizePolicy,
)

from common.app_context import get_app_context
from common.logger import info, error, warning
from UI.widgets.preview_overlay.channel import ChannelButton, ChannelOverlay
from UI.widgets.preview_overlay.machine_vision import MachineVisionButton
from UI.widgets.preview_overlay.click_to_move import ClickToMoveOverlay
from UI.widgets.preview_overlay.crosshair import CrosshairButton, CrosshairOverlay
from UI.widgets.preview_overlay.focus import FocusOverlay
from UI.widgets.preview_overlay.inspect_calibration import InspectCalibrationOverlay
from UI.widgets.preview_overlay.grid import GridButton, GridOverlay
from UI.widgets.preview_overlay.overlay_base import Overlay
from UI.widgets.preview_overlay.red_mark_detection_overlay import RedMarkDetectionOverlay
from UI.widgets.preview_overlay.background_detection import BackgroundDetectionOverlay
from UI.widgets.preview_overlay.focus_stack_preview import FocusStackPreviewOverlay


class EyeToggleButton(QPushButton):
    """
    Overlay button that draws a monochrome eye icon and an optional diagonal
    slash using QPainter — no emoji, so color is fully controlled.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._slashed: bool = False
        self.setObjectName("HidePreviewButton")
        self.setFixedSize(30, 30)
        self.setToolTip("Hide Preview")

    @property
    def slashed(self) -> bool:
        return self._slashed

    @slashed.setter
    def slashed(self, value: bool) -> None:
        if self._slashed != value:
            self._slashed = value
            self.setToolTip("Show Preview" if value else "Hide Preview")
            self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2

        pen = QPen(QColor(0, 0, 0))
        pen.setWidth(1)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        ew, eh = w * 0.72, h * 0.42

        path = QPainterPath()
        path.moveTo(cx - ew / 2, cy)
        path.quadTo(cx, cy - eh, cx + ew / 2, cy)
        path.quadTo(cx, cy + eh, cx - ew / 2, cy)
        painter.drawPath(path)

        pr = h * 0.13
        painter.setBrush(QColor(0, 0, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(cx - pr, cy - pr, pr * 2, pr * 2))

        if self._slashed:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            slash_pen = QPen(QColor(0, 0, 0))
            slash_pen.setWidth(1)
            slash_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(slash_pen)
            margin = 5
            painter.drawLine(
                QPoint(w - margin, margin),
                QPoint(margin, h - margin),
            )

        painter.end()


class OverlayLabel(QLabel):
    """QLabel that drives a list of Overlay instances on each paint and frame."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._overlays: list[Overlay] = []
        self._click_handler: ClickToMoveOverlay | None = None

    def add_overlay(self, overlay: Overlay) -> None:
        self._overlays.append(overlay)

    def set_click_handler(self, handler: ClickToMoveOverlay | None) -> None:
        """
        Register the overlay that should receive mouse clicks.

        Pass None to stop forwarding clicks (e.g. when click-to-move is
        disabled). The label enables Qt.WA_Cursor only while a handler
        is active so the cursor gives visual feedback.
        """
        self._click_handler = handler
        self.setCursor(
            Qt.CursorShape.CrossCursor if handler is not None
            else Qt.CursorShape.ArrowCursor
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (
            self._click_handler is not None
            and self._click_handler.enabled
            and event.button() == Qt.MouseButton.LeftButton
            and self.pixmap() is not None
            and not self.pixmap().isNull()
        ):
            image_rect = self._image_rect(self.pixmap())
            parent = self.parent()
            full_w = getattr(parent, "_current_full_width", 0)
            full_h = getattr(parent, "_current_full_height", 0)
            if full_w > 0 and full_h > 0:
                self._click_handler.handle_click(
                    event.position().toPoint().x(),
                    event.position().toPoint().y(),
                    image_rect,
                    full_w,
                    full_h,
                )
                event.accept()
                return
        super().mousePressEvent(event)

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


class OverlayController:
    """
    Programmatic control surface for CameraPreview overlays.

    Obtained via CameraPreview.overlays or get_app_context().camera_preview.overlays.
    Each setter mirrors the corresponding toolbar button so external modules
    can drive overlay state without importing widget internals.
    """

    def __init__(self, preview: CameraPreview) -> None:
        self._preview = preview

    @property
    def crosshair(self) -> bool:
        return self._preview._crosshair_overlay.enabled

    @crosshair.setter
    def crosshair(self, enabled: bool) -> None:
        self._preview._crosshair_button.setChecked(enabled)
        self._preview._crosshair_overlay.set_enabled(enabled)
        self._preview._video_label.update()

    @property
    def grid(self) -> bool:
        return self._preview._grid_overlay.enabled

    @grid.setter
    def grid(self, enabled: bool) -> None:
        self._preview._grid_button.setChecked(enabled)
        self._preview._grid_overlay.set_enabled(enabled)
        self._preview._video_label.update()

    @property
    def focus(self) -> bool:
        return self._preview._focus_overlay.enabled

    @focus.setter
    def focus(self, enabled: bool) -> None:
        self._preview._focus_overlay.set_enabled(enabled)
        self._preview._video_label.update()

    @property
    def inspect_calibration(self) -> bool:
        return self._preview._inspect_calibration_overlay.enabled

    @inspect_calibration.setter
    def inspect_calibration(self, enabled: bool) -> None:
        self._preview._inspect_calibration_overlay.set_enabled(enabled)
        self._preview._video_label.update()

    @property
    def red_mark(self) -> bool:
        return self._preview._red_mark_overlay.enabled

    @red_mark.setter
    def red_mark(self, enabled: bool) -> None:
        self._preview._red_mark_overlay.set_enabled(enabled)
        self._preview._video_label.update()

    @property
    def background(self) -> bool:
        return self._preview._background_overlay.enabled

    @background.setter
    def background(self, enabled: bool) -> None:
        self._preview._background_overlay.set_enabled(enabled)
        self._preview._video_label.update()

    @property
    def click_to_move(self) -> bool:
        return self._preview._click_to_move_overlay.enabled

    @click_to_move.setter
    def click_to_move(self, enabled: bool) -> None:
        self._preview._click_to_move_overlay.set_enabled(enabled)
        self._preview._video_label.update()

    @property
    def focus_stack_preview(self) -> FocusStackPreviewOverlay:
        """Direct access to the focus stack preview overlay."""
        return self._preview._focus_stack_preview_overlay

    def set_channel(
        self,
        *,
        red: bool = True,
        green: bool = True,
        blue: bool = True,
        grayscale: bool = False,
    ) -> None:
        """Set channel filter state directly, bypassing the toolbar menu."""
        overlay = self._preview._channel_overlay
        overlay.show_red = red
        overlay.show_green = green
        overlay.show_blue = blue
        overlay.show_grayscale = grayscale


class CameraPreview(QFrame):
    """
    Camera preview widget that displays frames from the camera manager.

    Whichever frame type has the higher sequence number (preview_frame_seq
    vs still_frame_seq on the camera manager) is considered more recent
    and is shown on screen. This means the display keeps updating from still
    frames during automation even when the live preview stream is paused.

    This widget only handles display - it does not manage camera lifecycle.

    A single instance should be created by the main window and registered with
    AppContext via register_camera_preview(). Other pages embed it directly
    rather than constructing their own instances. Overlay state is accessible
    from any module through the overlays property or via AppContext:

        get_app_context().camera_preview.overlays.crosshair = True
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setObjectName("CameraPreview")

        self._preview_buf: bytearray = bytearray()
        self._preview_width: int = 0
        self._preview_height: int = 0
        self._preview_seq: int = 0

        self._still_buf: bytearray = bytearray()
        self._still_width: int = 0
        self._still_height: int = 0
        self._still_seq: int = 0

        self._current_full_width: int = 0
        self._current_full_height: int = 0

        self._preview_hidden: bool = False

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

        self._hidden_label = QLabel("Camera preview disabled", self)
        self._hidden_label.setObjectName("PreviewHiddenLabel")
        self._hidden_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hidden_label.hide()

        self._reenable_button = QPushButton("Enable Preview", self)
        self._reenable_button.setObjectName("PreviewReenableButton")
        self._reenable_button.clicked.connect(self._show_preview)
        self._reenable_button.hide()

        self._crosshair_overlay = CrosshairOverlay()
        self._grid_overlay = GridOverlay()
        self._focus_overlay = FocusOverlay()
        self._inspect_calibration_overlay = InspectCalibrationOverlay()
        self._red_mark_overlay = RedMarkDetectionOverlay()
        self._background_overlay = BackgroundDetectionOverlay()
        self._channel_overlay = ChannelOverlay()
        self._click_to_move_overlay = ClickToMoveOverlay()
        self._focus_stack_preview_overlay = FocusStackPreviewOverlay()

        self._video_label.add_overlay(self._crosshair_overlay)
        self._video_label.add_overlay(self._grid_overlay)
        self._video_label.add_overlay(self._focus_overlay)
        self._video_label.add_overlay(self._inspect_calibration_overlay)
        self._video_label.add_overlay(self._red_mark_overlay)
        self._video_label.add_overlay(self._background_overlay)
        self._video_label.add_overlay(self._click_to_move_overlay)
        self._video_label.add_overlay(self._focus_stack_preview_overlay)

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

        self._machine_vision_button = MachineVisionButton(self)
        self._machine_vision_button.move(10, 80)
        self._machine_vision_button.raise_()
        self._machine_vision_button.menu.raise_()
        self._machine_vision_button.vision_mode_changed.connect(self._on_vision_mode_changed)

        self._channel_button = ChannelButton(self)
        self._channel_button.move(10, 115)
        self._channel_button.raise_()
        self._channel_button.menu.raise_()
        self._channel_button.channel_changed.connect(self._on_channel_changed)

        self._hide_preview_button = EyeToggleButton(self)
        self._hide_preview_button.move(10, 150)
        self._hide_preview_button.raise_()
        self._hide_preview_button.clicked.connect(self._toggle_preview_visibility)

        self._overlays = OverlayController(self)

        self._video_label.set_click_handler(self._click_to_move_overlay)

        self._focus_overlay._relay.result_ready.connect(self._video_label.update)
        self._inspect_calibration_overlay._relay.result_ready.connect(self._video_label.update)
        self._red_mark_overlay._relay.result_ready.connect(self._video_label.update)
        self._background_overlay._relay.result_ready.connect(self._video_label.update)

        self._connect_to_camera_manager()

    @property
    def overlays(self) -> OverlayController:
        """
        Programmatic control surface for overlay state.

        Use this from other modules instead of manipulating overlay objects directly:

            preview = get_app_context().camera_preview
            if preview is not None:
                preview.overlays.crosshair = True
        """
        return self._overlays

    def _connect_to_camera_manager(self) -> None:
        ctx = get_app_context()
        camera_manager = ctx.camera_manager

        camera_manager.preview_frame_ready.connect(self._on_frame_ready)
        camera_manager.still_frame_ready.connect(self._on_still_frame_ready)
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

    @Slot(bool, bool, bool, bool, bool)
    def _on_vision_mode_changed(
        self,
        focus: bool,
        focus_region: bool,
        red_mark: bool,
        scale: bool,
        background: bool,
    ) -> None:
        self._focus_overlay.set_region_mode(focus_region)
        self._focus_overlay.set_enabled(focus or focus_region)
        self._red_mark_overlay.set_enabled(red_mark)
        self._inspect_calibration_overlay.set_enabled(scale)
        self._background_overlay.set_enabled(background)
        self._video_label.update()

    def _toggle_preview_visibility(self) -> None:
        if self._preview_hidden:
            self._show_preview()
        else:
            self._hide_preview()

    def _hide_preview(self) -> None:
        self._preview_hidden = True
        self._video_label.hide()
        self._hidden_label.show()
        self._reenable_button.show()
        self._hide_preview_button.slashed = True
        self._reposition_overlay_widgets()

    def _show_preview(self) -> None:
        self._preview_hidden = False
        self._video_label.show()
        self._hidden_label.hide()
        self._reenable_button.hide()
        self._hide_preview_button.slashed = False
        self._reposition_overlay_widgets()

    def _reposition_overlay_widgets(self) -> None:
        w = self.width()
        h = self.height()
        label_hint = self._hidden_label.sizeHint()
        self._hidden_label.setGeometry(
            (w - label_hint.width()) // 2,
            (h - label_hint.height()) // 2 - 20,
            label_hint.width(),
            label_hint.height(),
        )
        btn_hint = self._reenable_button.sizeHint()
        self._reenable_button.setGeometry(
            (w - btn_hint.width()) // 2,
            (h - label_hint.height()) // 2 + label_hint.height() - 10,
            btn_hint.width(),
            btn_hint.height(),
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._preview_hidden:
            self._reposition_overlay_widgets()

    # ------------------------------------------------------------------
    # Frame slots
    # ------------------------------------------------------------------

    @Slot(int, int)
    def _on_frame_ready(self, width: int, height: int) -> None:
        """Receive a new preview frame from the camera manager."""
        ctx = get_app_context()
        camera_manager = ctx.camera_manager

        src = camera_manager.get_current_frame()
        if not src:
            return

        stride = camera_manager.preview_frame_stride
        if stride == 0:
            return
        required = stride * height

        if width != self._preview_width or height != self._preview_height:
            self._preview_buf = bytearray(required)
            self._preview_width = width
            self._preview_height = height

        self._preview_buf[:required] = src[:required]
        self._preview_seq = camera_manager.preview_frame_seq

        if self._preview_seq >= self._still_seq:
            self._render_display(self._preview_buf, width, height, stride)

    @Slot(int, int)
    def _on_still_frame_ready(self, width: int, height: int) -> None:
        """Receive a new still frame from the camera manager."""
        ctx = get_app_context()
        camera_manager = ctx.camera_manager
        src = camera_manager.get_current_still_frame()
        if not src:
            return

        stride = camera_manager.still_frame_stride
        if stride == 0:
            return
        required = stride * height

        if width != self._still_width or height != self._still_height:
            self._still_buf = bytearray(required)
            self._still_width = width
            self._still_height = height

        self._still_buf[:required] = src[:required]
        self._still_seq = camera_manager.still_frame_seq

        self._render_display(self._still_buf, width, height, stride)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_display(
        self,
        buf: bytearray,
        width: int,
        height: int,
        stride: int,
    ) -> None:
        if self._preview_hidden:
            return

        self._current_full_width = width
        self._current_full_height = height

        # QImage(buf, ...) does not copy the data — it holds a raw pointer
        # into buf.  Call .copy() immediately so the QImage owns its memory
        # and cannot be invalidated if buf is reassigned elsewhere.
        image = QImage(buf, width, height, stride, QImage.Format.Format_RGB888).copy()

        if self._channel_overlay.needs_filter:
            image = self._channel_overlay.apply(image)

        # image.bits() returns a raw pointer; keep image alive in a local
        # so the GC cannot collect it while ptr is still being read.
        ptr = image.bits()
        full_arr = (
            np.frombuffer(ptr, dtype=np.uint8)
            .reshape((image.height(), image.bytesPerLine()))
            [:, : image.width() * 3]
            .reshape((image.height(), image.width(), 3))
            .copy()
        )
        del ptr
        self._video_label.notify_full(full_arr)

        lw = self._video_label.width()
        lh = self._video_label.height()
        if lw > 0 and lh > 0:
            scaled = image.scaled(
                lw, lh,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )

            scaled_ptr = scaled.bits()
            scaled_arr = (
                np.frombuffer(scaled_ptr, dtype=np.uint8)
                .reshape((scaled.height(), scaled.bytesPerLine()))
                [:, : scaled.width() * 3]
                .reshape((scaled.height(), scaled.width(), 3))
                .copy()
            )
            del scaled_ptr
            self._video_label.notify_scaled(scaled_arr)
            self._video_label.setPixmap(QPixmap.fromImage(scaled))

    # ------------------------------------------------------------------
    # Camera manager state slots
    # ------------------------------------------------------------------

    @Slot(int, int)
    def _on_streaming_started(self, width: int, height: int) -> None:
        info(f"Preview: Streaming started ({width}x{height})")
        self._video_label.setText("")

    @Slot()
    def _on_streaming_stopped(self) -> None:
        info("Preview: Streaming stopped")
        self._preview_buf = bytearray()
        self._preview_width = 0
        self._preview_height = 0
        self._preview_seq = 0
        self._still_buf = bytearray()
        self._still_width = 0
        self._still_height = 0
        self._still_seq = 0
        self._video_label.setText("Camera stream stopped")

    @Slot()
    def _on_camera_error(self, description: str) -> None:
        self._video_label.setText(f"Camera error: {description}")
        error(f"Preview: Camera error occurred: {description}")

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

    _SCROLL_STEP_NM: int = 40_000

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

        ctx = get_app_context()
        camera_manager = ctx.camera_manager

        camera_manager.preview_frame_ready.disconnect(self._on_frame_ready)
        camera_manager.still_frame_ready.disconnect(self._on_still_frame_ready)
        camera_manager.streaming_started.disconnect(self._on_streaming_started)
        camera_manager.streaming_stopped.disconnect(self._on_streaming_stopped)
        camera_manager.camera_error.disconnect(self._on_camera_error)
        camera_manager.camera_disconnected.disconnect(self._on_camera_disconnected)
        camera_manager.active_camera_changed.disconnect(self._on_active_camera_changed)

        info("Preview cleanup complete")