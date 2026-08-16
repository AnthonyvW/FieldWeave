from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QPushButton, QWidget

from common.logger import info
from UI.style import ZOOM_PREVIEW_VIEWPORT_COLOR
from UI.widgets.preview_overlay.overlay_base import Overlay


class ZoomPreviewOverlay(Overlay):
    """
    Displays a scroll-zoomable, click-and-drag-pannable crop of the camera
    feed over the normal display.

    Latches onto the full-resolution frame via ``update_full`` and draws a
    scaled crop of it in ``draw``. Zoom and pan are driven externally by
    ``zoom()`` and ``begin_drag`` / ``drag_to`` / ``end_drag``, which
    ``OverlayLabel`` and ``CameraPreview`` call only while this overlay is
    ``enabled`` — the same enabled check is what those callers use to
    suppress z-axis scroll and click-to-move while zoom preview is active.
    """

    _MIN_ZOOM: float = 1.0
    _MAX_ZOOM: float = 8.0
    _ZOOM_STEP: float = 1.15

    _MINIMAP_MAX_WIDTH: int = 120
    _MINIMAP_MARGIN: int = 10

    def __init__(self) -> None:
        super().__init__()
        self._frame: np.ndarray | None = None
        self._zoom: float = self._MIN_ZOOM
        self._center_x: float = 0.5
        self._center_y: float = 0.5
        self._dragging: bool = False
        self._drag_last: QPoint | None = None

    def set_enabled(self, enabled: bool) -> None:
        super().set_enabled(enabled)
        if not enabled:
            self.reset()

    def reset(self) -> None:
        self._zoom = self._MIN_ZOOM
        self._center_x = 0.5
        self._center_y = 0.5
        self._dragging = False
        self._drag_last = None

    def update_full(self, frame: np.ndarray) -> None:
        self._frame = frame

    def zoom(self, steps: int) -> None:
        """Zoom in (steps > 0) or out (steps < 0) around the current center."""
        if steps == 0 or self._frame is None:
            return
        factor = self._ZOOM_STEP ** steps
        self._zoom = min(self._MAX_ZOOM, max(self._MIN_ZOOM, self._zoom * factor))
        self._clamp_center()

    def begin_drag(self, pos: QPoint) -> None:
        self._dragging = True
        self._drag_last = pos

    def drag_to(self, pos: QPoint, rect: QRect) -> None:
        if not self._dragging or self._drag_last is None or self._frame is None:
            return
        if rect.width() <= 0 or rect.height() <= 0:
            return

        dx = pos.x() - self._drag_last.x()
        dy = pos.y() - self._drag_last.y()
        self._drag_last = pos

        norm_dx = (dx / rect.width()) / self._zoom
        norm_dy = (dy / rect.height()) / self._zoom
        self._center_x -= norm_dx
        self._center_y -= norm_dy
        self._clamp_center()

    def end_drag(self) -> None:
        self._dragging = False
        self._drag_last = None

    def _clamp_center(self) -> None:
        half_w = 0.5 / self._zoom
        half_h = 0.5 / self._zoom
        self._center_x = min(1.0 - half_w, max(half_w, self._center_x))
        self._center_y = min(1.0 - half_h, max(half_h, self._center_y))

    def draw(self, painter: QPainter, rect: QRect) -> None:
        if self._frame is None:
            return

        h, w = self._frame.shape[:2]
        if w == 0 or h == 0:
            return

        crop_w = int(w / self._zoom)
        crop_h = int(h / self._zoom)
        if crop_w <= 0 or crop_h <= 0:
            return

        x0 = min(max(int(self._center_x * w - crop_w / 2), 0), w - crop_w)
        y0 = min(max(int(self._center_y * h - crop_h / 2), 0), h - crop_h)
        crop = np.ascontiguousarray(self._frame[y0:y0 + crop_h, x0:x0 + crop_w])

        q_image = QImage(crop.data, crop_w, crop_h, crop_w * 3, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(
            rect.width(),
            rect.height(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(rect.x(), rect.y(), scaled)

        self._draw_minimap(painter, rect, w, h, x0, y0, crop_w, crop_h)

    def _draw_minimap(
        self,
        painter: QPainter,
        rect: QRect,
        full_w: int,
        full_h: int,
        crop_x: int,
        crop_y: int,
        crop_w: int,
        crop_h: int,
    ) -> None:
        """Draw a small thumbnail of the full frame with a box marking the current viewport."""
        mini_w = min(self._MINIMAP_MAX_WIDTH, full_w)
        mini_h = int(mini_w * full_h / full_w)
        if mini_w <= 0 or mini_h <= 0:
            return

        thumb_image = QImage(self._frame.data, full_w, full_h, full_w * 3, QImage.Format.Format_RGB888)
        thumb = QPixmap.fromImage(thumb_image).scaled(
            mini_w,
            mini_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        mx = rect.right() - mini_w - self._MINIMAP_MARGIN
        my = rect.top() + self._MINIMAP_MARGIN
        painter.drawPixmap(mx, my, thumb)

        scale_x = mini_w / full_w
        scale_y = mini_h / full_h
        vp_x = mx + int(crop_x * scale_x)
        vp_y = my + int(crop_y * scale_y)
        vp_w = max(1, int(crop_w * scale_x))
        vp_h = max(1, int(crop_h * scale_y))

        painter.save()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        border_pen = QPen(QColor(0, 0, 0, 200))
        border_pen.setWidth(1)
        painter.setPen(border_pen)
        painter.drawRect(mx, my, mini_w, mini_h)

        viewport_pen = QPen(ZOOM_PREVIEW_VIEWPORT_COLOR)
        viewport_pen.setWidth(2)
        painter.setPen(viewport_pen)
        painter.drawRect(vp_x, vp_y, vp_w, vp_h)
        painter.restore()


class ZoomPreviewButton(QPushButton):
    """
    Checkable overlay button that toggles pan/zoom preview mode.

    Draws a magnifier-with-crosshair icon via QPainter, the same approach
    ``EyeToggleButton`` uses, so the icon renders identically everywhere
    instead of depending on a font glyph.
    """

    toggled_zoom_preview = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ZoomPreviewButton")
        self.setCheckable(True)
        self.setFixedSize(30, 30)
        self.setToolTip("Toggle Zoom Preview")
        self.clicked.connect(self._on_clicked)

    def _on_clicked(self, checked: bool) -> None:
        info(f"Preview: Zoom preview {'enabled' if checked else 'disabled'}")
        self.toggled_zoom_preview.emit(checked)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(QColor(0, 0, 0))
        pen.setWidth(2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        cx, cy, r = 13.0, 13.5, 6.0
        painter.drawEllipse(QPointF(cx, cy), r, r)
        painter.drawLine(QPointF(cx, cy - 2.3), QPointF(cx, cy + 2.3))
        painter.drawLine(QPointF(cx - 2.3, cy), QPointF(cx + 2.3, cy))

        diag = 0.7071
        handle_start = QPointF(cx + r * diag, cy + r * diag)
        handle_end = QPointF(cx + (r + 5.5) * diag, cy + (r + 5.5) * diag)
        painter.drawLine(handle_start, handle_end)

        painter.end()