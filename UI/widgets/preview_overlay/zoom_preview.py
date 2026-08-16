from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QTransform
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
    ``OverlayLabel`` calls only while this overlay is ``enabled`` — the
    same enabled check is what ``CameraPreview`` uses to suppress z-axis
    scroll while zoom preview is active. Click-to-move remains active in
    this mode: ``OverlayLabel`` distinguishes a click from a drag by
    movement distance and, for a click, maps it to a full-resolution pixel
    via ``widget_pos_to_full_pixel`` instead of the plain un-zoomed scale.
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
        self._crop: tuple[int, int, int, int] | None = None

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
        self._crop = None

    def update_full(self, frame: np.ndarray) -> None:
        self._frame = frame

    def zoom(self, steps: int, widget_rect: QRect) -> None:
        """Zoom in (steps > 0) or out (steps < 0) around the current center."""
        if steps == 0 or self._frame is None:
            return
        factor = self._ZOOM_STEP ** steps
        self._zoom = min(self._MAX_ZOOM, max(self._MIN_ZOOM, self._zoom * factor))
        self._clamp_center(widget_rect)

    def begin_drag(self, pos: QPoint) -> None:
        self._dragging = True
        self._drag_last = pos

    def drag_to(self, pos: QPoint, widget_rect: QRect) -> None:
        if not self._dragging or self._drag_last is None or self._frame is None:
            return

        crop = self._current_crop(widget_rect)
        if crop is None:
            return
        _, _, crop_w, crop_h = crop
        h, w = self._frame.shape[:2]
        display_rect = self._fit_rect(crop_w, crop_h, widget_rect)
        if display_rect.width() <= 0 or display_rect.height() <= 0:
            return

        dx = pos.x() - self._drag_last.x()
        dy = pos.y() - self._drag_last.y()
        self._drag_last = pos

        # A screen-pixel delta covers (delta / display_rect size) of the
        # crop, which itself covers (crop size / sensor size) of the full
        # frame — chain the two to get the actual sensor-fraction moved.
        # (crop_w/w is not simply 1/zoom: see ``_crop_size``.)
        norm_dx = (dx / display_rect.width()) * (crop_w / w)
        norm_dy = (dy / display_rect.height()) * (crop_h / h)
        self._center_x -= norm_dx
        self._center_y -= norm_dy
        self._clamp_center(widget_rect)

    def end_drag(self) -> None:
        self._dragging = False
        self._drag_last = None

    def _crop_size(self, widget_rect: QRect) -> tuple[int, int] | None:
        """
        Return (crop_w, crop_h) in full-resolution pixels for the current
        zoom level.

        At zoom == 1, this is always the entire, un-cropped sensor frame:
        whichever dimension the sensor already fully fills relative to
        *widget_rect*'s aspect ratio (the one that would letterbox in the
        plain, un-zoomed view) keeps shrinking by ``self._zoom`` exactly as
        before. The other dimension grows to try to match *widget_rect*'s
        aspect ratio, capped at the sensor's own extent — it can only do so
        once the shrinking dimension has freed up enough of the sensor's
        remaining field of view, which is why bars shrink progressively as
        zoom increases rather than disappearing (or appearing) all at once.
        """
        if self._frame is None or widget_rect.width() <= 0 or widget_rect.height() <= 0:
            return None

        h, w = self._frame.shape[:2]
        if w == 0 or h == 0:
            return None

        target_aspect = widget_rect.width() / widget_rect.height()
        sensor_aspect = w / h

        if sensor_aspect <= target_aspect:
            crop_h = h / self._zoom
            crop_w = min(w, crop_h * target_aspect)
        else:
            crop_w = w / self._zoom
            crop_h = min(h, crop_w / target_aspect)

        return max(1, int(crop_w)), max(1, int(crop_h))

    def _current_crop(self, widget_rect: QRect) -> tuple[int, int, int, int] | None:
        """Return (x0, y0, crop_w, crop_h) of the current viewport in full-frame pixel space."""
        size = self._crop_size(widget_rect)
        if size is None:
            return None
        crop_w, crop_h = size
        h, w = self._frame.shape[:2]

        x0 = min(max(int(self._center_x * w - crop_w / 2), 0), w - crop_w)
        y0 = min(max(int(self._center_y * h - crop_h / 2), 0), h - crop_h)
        return x0, y0, crop_w, crop_h

    def _clamp_center(self, widget_rect: QRect) -> None:
        size = self._crop_size(widget_rect)
        if size is None:
            return
        crop_w, crop_h = size
        h, w = self._frame.shape[:2]

        half_w_frac = min(0.5, crop_w / (2 * w))
        half_h_frac = min(0.5, crop_h / (2 * h))
        self._center_x = min(1.0 - half_w_frac, max(half_w_frac, self._center_x))
        self._center_y = min(1.0 - half_h_frac, max(half_h_frac, self._center_y))

    @staticmethod
    def _fit_rect(content_w: int, content_h: int, container: QRect) -> QRect:
        """Contain-fit a content_w x content_h aspect ratio into container, centered."""
        if content_w <= 0 or content_h <= 0 or container.width() <= 0 or container.height() <= 0:
            return container
        scale = min(container.width() / content_w, container.height() / content_h)
        scaled_w = int(content_w * scale)
        scaled_h = int(content_h * scale)
        x = container.left() + (container.width() - scaled_w) // 2
        y = container.top() + (container.height() - scaled_h) // 2
        return QRect(x, y, scaled_w, scaled_h)

    def display_rect(self, widget_rect: QRect) -> QRect | None:
        """
        Compute (and remember) the current crop for *widget_rect* — the
        full, un-letterboxed label rect — and return the rect within it
        that the cropped image should be drawn into.

        ``OverlayLabel`` must call this once per paint cycle, before
        drawing any overlay, to get the rect to pass to every overlay's
        ``draw()`` (including this one's) and to ``paint_transform``: both
        reuse the crop computed and cached here rather than recomputing
        it. Returns None when disabled or there's no frame yet, in which
        case callers should fall back to the plain, un-zoomed letterboxed
        display.
        """
        if not self.enabled:
            self._crop = None
            return None
        self._crop = self._current_crop(widget_rect)
        if self._crop is None:
            return None
        _, _, crop_w, crop_h = self._crop
        return self._fit_rect(crop_w, crop_h, widget_rect)

    def paint_transform(self, rect: QRect) -> QTransform | None:
        """
        Return the affine transform that maps ordinary, un-zoomed drawing
        coordinates within *rect* (the rect returned by ``display_rect``)
        onto the current pan/zoom viewport.

        ``OverlayLabel`` applies this around every other overlay's
        ``draw()`` call, so this is the single place that owns pan/zoom
        math — other overlays keep drawing against the plain ``rect``
        exactly as if zoom preview didn't exist. Uses the crop cached by
        the preceding ``display_rect`` call. Returns None when disabled or
        there's no frame yet, in which case ``OverlayLabel`` should skip
        applying any transform.
        """
        if not self.enabled or self._crop is None:
            return None

        x0, y0, crop_w, crop_h = self._crop
        h, w = self._frame.shape[:2]

        crop_frac_x0 = x0 / w
        crop_frac_y0 = y0 / h
        crop_frac_w = crop_w / w
        crop_frac_h = crop_h / h

        scale_x = 1.0 / crop_frac_w
        scale_y = 1.0 / crop_frac_h
        translate_x = rect.left() * (1 - scale_x) - crop_frac_x0 * rect.width() * scale_x
        translate_y = rect.top() * (1 - scale_y) - crop_frac_y0 * rect.height() * scale_y

        transform = QTransform()
        transform.translate(translate_x, translate_y)
        transform.scale(scale_x, scale_y)
        return transform

    def widget_pos_to_full_pixel(self, pos: QPoint, widget_rect: QRect) -> tuple[float, float, int, int] | None:
        """
        Map a widget-space position to a full camera-resolution pixel
        coordinate, accounting for the current pan/zoom viewport. Returns
        ``(full_px, full_py, full_width, full_height)``, or None if there
        is no frame yet.
        """
        crop = self._current_crop(widget_rect)
        if crop is None:
            return None
        x0, y0, crop_w, crop_h = crop
        h, w = self._frame.shape[:2]

        display_rect = self._fit_rect(crop_w, crop_h, widget_rect)
        if display_rect.width() <= 0 or display_rect.height() <= 0:
            return None

        rel_x = (pos.x() - display_rect.x()) / display_rect.width()
        rel_y = (pos.y() - display_rect.y()) / display_rect.height()

        full_px = x0 + rel_x * crop_w
        full_py = y0 + rel_y * crop_h
        return full_px, full_py, w, h

    def current_view_center_full_pixel(self, widget_rect: QRect) -> tuple[float, float] | None:
        """
        Return the full-resolution pixel currently shown at the centre of
        the display — the click-to-move reference centre while zoom
        preview is panned/zoomed, since that's what the operator is
        actually looking at rather than the sensor's absolute centre.
        """
        crop = self._current_crop(widget_rect)
        if crop is None:
            return None
        x0, y0, crop_w, crop_h = crop
        return x0 + crop_w / 2, y0 + crop_h / 2

    def draw(self, painter: QPainter, rect: QRect) -> None:
        if self._crop is None:
            return
        x0, y0, crop_w, crop_h = self._crop

        crop_arr = np.ascontiguousarray(self._frame[y0:y0 + crop_h, x0:x0 + crop_w])

        q_image = QImage(crop_arr.data, crop_w, crop_h, crop_w * 3, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(
            rect.width(),
            rect.height(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(rect.x(), rect.y(), scaled)

    def draw_foreground(self, painter: QPainter, rect: QRect) -> None:
        """
        Draw the viewport minimap on top of every other overlay's ``draw``
        output — see ``Overlay.draw_foreground``. Without this, an overlay
        drawn after this one (e.g. FocusOverlay's heatmap) would paint
        over the minimap.
        """
        if self._crop is None:
            return
        x0, y0, crop_w, crop_h = self._crop
        h, w = self._frame.shape[:2]
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