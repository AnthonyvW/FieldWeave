from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QTransform
from PySide6.QtWidgets import QPushButton, QWidget

from UI.style import ZOOM_PREVIEW_VIEWPORT_COLOR
from UI.widgets.preview_overlay.loaded_image_overlay import LoadedImageOverlay
from UI.widgets.preview_overlay.overlay_base import Overlay


class _PanZoomState:
    """
    Mutable pan/zoom position for one frame source.

    ZoomPreviewOverlay keeps one of these per source (live feed, loaded
    image) rather than a single shared position, since the two can differ
    wildly in resolution and aspect ratio — reusing one position across
    both would leave the zoom level and center fraction computed for one
    source's dimensions applied to the other's, cropping the wrong region
    entirely once the loaded image is disabled again.
    """

    def __init__(self, min_zoom: float) -> None:
        self._min_zoom = min_zoom
        self.zoom: float = min_zoom
        self.center_x: float = 0.5
        self.center_y: float = 0.5
        self.dragging: bool = False
        self.drag_last: QPoint | None = None

    def reset(self) -> None:
        self.zoom = self._min_zoom
        self.center_x = 0.5
        self.center_y = 0.5
        self.dragging = False
        self.drag_last = None


class ZoomPreviewOverlay(Overlay):
    """
    Displays a zoomable, click-and-drag-pannable crop of the camera feed
    over the normal display.

    Latches onto the full-resolution frame via ``update_full``, which
    ``OverlayLabel`` calls unconditionally so a frame is always available
    for ``zoom()`` regardless of zoom state, and draws a scaled crop of
    it in ``draw``.

    ``zoom()`` is driven by ``ZoomStepButton`` and by ctrl+scroll (see
    ``CameraPreview.wheelEvent``). ``active`` is true once zoomed past
    the fully-zoomed-out level: it's what actually drives drawing the
    crop and minimap — see ``display_rect`` / ``paint_transform`` /
    ``draw_foreground`` — and what ``OverlayLabel`` gates click-and-drag
    panning on. Click-to-move remains active too: ``OverlayLabel``
    distinguishes a click from a drag by movement distance and, for a
    click, maps it to a full-resolution pixel via
    ``widget_pos_to_full_pixel`` instead of the plain un-zoomed scale.

    ``_frame`` normally holds the last live camera frame pushed via
    ``update_full``. When a ``LoadedImageOverlay`` is registered via
    ``set_loaded_image_overlay`` and enabled, ``_frame`` switches to
    that overlay's ``full_array`` instead — this overlay draws itself on
    top of ``LoadedImageOverlay`` (see ``OverlayLabel._paint_overlays``),
    so without the switch a zoomed live frame would occlude the loaded
    image the operator actually wants to inspect. ``_state`` switches
    alongside it to the matching ``_PanZoomState``, so the two sources'
    zoom/pan positions never bleed into one another.
    """

    _MIN_ZOOM: float = 1.0
    _MAX_ZOOM: float = 8.0
    _ZOOM_STEP: float = 1.15

    _MINIMAP_MAX_WIDTH: int = 120
    _MINIMAP_MARGIN: int = 10

    def __init__(self) -> None:
        super().__init__()
        self._live_frame: np.ndarray | None = None
        self._loaded_image_overlay: LoadedImageOverlay | None = None
        self._live_state = _PanZoomState(self._MIN_ZOOM)
        self._loaded_state = _PanZoomState(self._MIN_ZOOM)
        self._crop: tuple[int, int, int, int] | None = None

    @property
    def zoomed(self) -> bool:
        """True once zoomed in past the fully-zoomed-out level."""
        return self._state.zoom > self._MIN_ZOOM

    @property
    def active(self) -> bool:
        """True whenever the crop/zoom viewport should be drawn and panned — i.e. whenever the view is zoomed."""
        return self.zoomed

    def reset(self) -> None:
        """Reset the pan/zoom position of whichever source is currently shown."""
        self._state.reset()
        self._crop = None

    def update_full(self, frame: np.ndarray) -> None:
        self._live_frame = frame

    def set_loaded_image_overlay(self, overlay: LoadedImageOverlay | None) -> None:
        """
        Register the overlay to crop/zoom into instead of the live feed
        while it's enabled — see the class docstring's note on ``_frame``.
        """
        self._loaded_image_overlay = overlay

    @property
    def _loaded_active(self) -> bool:
        return self._loaded_image_overlay is not None and self._loaded_image_overlay.enabled

    @property
    def _frame(self) -> np.ndarray | None:
        if self._loaded_active:
            return self._loaded_image_overlay.full_array
        return self._live_frame

    @property
    def _state(self) -> _PanZoomState:
        return self._loaded_state if self._loaded_active else self._live_state

    def zoom(self, steps: int, widget_rect: QRect, anchor: QPoint | None = None) -> None:
        """
        Zoom in (steps > 0) or out (steps < 0).

        Without an anchor, the current center stays fixed — used by the
        step buttons and keyboard shortcut. With an anchor — a
        widget-space position, e.g. the mouse position for ctrl+scroll —
        the full-resolution pixel under that position is kept fixed on
        screen instead, so zooming feels like it's happening at the
        cursor rather than recentering.
        """
        if steps == 0 or self._frame is None:
            return

        state = self._state
        anchor_state = self._pos_to_full_pixel_and_rel(anchor, widget_rect) if anchor is not None else None

        factor = self._ZOOM_STEP ** steps
        state.zoom = min(self._MAX_ZOOM, max(self._MIN_ZOOM, state.zoom * factor))

        if anchor_state is not None:
            full_px, full_py, rel_x, rel_y, _, _ = anchor_state
            size = self._crop_size(widget_rect)
            if size is not None:
                crop_w, crop_h = size
                h, w = self._frame.shape[:2]
                state.center_x = (full_px - crop_w * (rel_x - 0.5)) / w
                state.center_y = (full_py - crop_h * (rel_y - 0.5)) / h

        self._clamp_center(widget_rect)

    def begin_drag(self, pos: QPoint) -> None:
        state = self._state
        state.dragging = True
        state.drag_last = pos

    def drag_to(self, pos: QPoint, widget_rect: QRect) -> None:
        state = self._state
        if not state.dragging or state.drag_last is None or self._frame is None:
            return

        crop = self._current_crop(widget_rect)
        if crop is None:
            return
        _, _, crop_w, crop_h = crop
        h, w = self._frame.shape[:2]
        display_rect = self._fit_rect(crop_w, crop_h, widget_rect)
        if display_rect.width() <= 0 or display_rect.height() <= 0:
            return

        dx = pos.x() - state.drag_last.x()
        dy = pos.y() - state.drag_last.y()
        state.drag_last = pos

        # A screen-pixel delta covers (delta / display_rect size) of the
        # crop, which itself covers (crop size / sensor size) of the full
        # frame — chain the two to get the actual sensor-fraction moved.
        # (crop_w/w is not simply 1/zoom: see ``_crop_size``.)
        norm_dx = (dx / display_rect.width()) * (crop_w / w)
        norm_dy = (dy / display_rect.height()) * (crop_h / h)
        state.center_x -= norm_dx
        state.center_y -= norm_dy
        self._clamp_center(widget_rect)

    def end_drag(self) -> None:
        state = self._state
        state.dragging = False
        state.drag_last = None

    def _crop_size(self, widget_rect: QRect) -> tuple[int, int] | None:
        """
        Return (crop_w, crop_h) in full-resolution pixels for the current
        zoom level.

        At zoom == 1, this is always the entire, un-cropped sensor frame:
        whichever dimension the sensor already fully fills relative to
        *widget_rect*'s aspect ratio (the one that would letterbox in the
        plain, un-zoomed view) keeps shrinking by ``self._state.zoom``
        exactly as before. The other dimension grows to try to match *widget_rect*'s
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
            crop_h = h / self._state.zoom
            crop_w = min(w, crop_h * target_aspect)
        else:
            crop_w = w / self._state.zoom
            crop_h = min(h, crop_w / target_aspect)

        return max(1, int(crop_w)), max(1, int(crop_h))

    def _current_crop(self, widget_rect: QRect) -> tuple[int, int, int, int] | None:
        """Return (x0, y0, crop_w, crop_h) of the current viewport in full-frame pixel space."""
        size = self._crop_size(widget_rect)
        if size is None:
            return None
        crop_w, crop_h = size
        h, w = self._frame.shape[:2]
        state = self._state

        x0 = min(max(int(state.center_x * w - crop_w / 2), 0), w - crop_w)
        y0 = min(max(int(state.center_y * h - crop_h / 2), 0), h - crop_h)
        return x0, y0, crop_w, crop_h

    def _clamp_center(self, widget_rect: QRect) -> None:
        size = self._crop_size(widget_rect)
        if size is None:
            return
        crop_w, crop_h = size
        h, w = self._frame.shape[:2]
        state = self._state

        half_w_frac = min(0.5, crop_w / (2 * w))
        half_h_frac = min(0.5, crop_h / (2 * h))
        state.center_x = min(1.0 - half_w_frac, max(half_w_frac, state.center_x))
        state.center_y = min(1.0 - half_h_frac, max(half_h_frac, state.center_y))

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
        if not self.active:
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
        if not self.active or self._crop is None:
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
        state = self._pos_to_full_pixel_and_rel(pos, widget_rect)
        if state is None:
            return None
        full_px, full_py, _, _, full_w, full_h = state
        return full_px, full_py, full_w, full_h

    def _pos_to_full_pixel_and_rel(
        self, pos: QPoint, widget_rect: QRect
    ) -> tuple[float, float, float, float, int, int] | None:
        """
        Map a widget-space position to a full-resolution pixel under the
        crop shown right now, plus that same position's fraction across
        the on-screen display rect. Returns
        ``(full_px, full_py, rel_x, rel_y, full_width, full_height)``, or
        None if there is no frame yet. Shared by ``widget_pos_to_full_pixel``
        (click-to-move) and ``zoom``'s anchor handling.
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
        return full_px, full_py, rel_x, rel_y, w, h

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

        Only shown once actually zoomed in: at the fully-zoomed-out level
        the viewport box would cover the whole minimap anyway.
        """
        if self._crop is None or not self.zoomed:
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
        """
        Draw a small thumbnail of the full frame with a box marking the
        current viewport.

        Sized to fit both ``_MINIMAP_MAX_WIDTH`` and a third of *rect* —
        the camera viewport — in each dimension, so the minimap never
        crowds out the view it's overlaid on even in a small or narrow
        preview pane.
        """
        if full_w <= 0 or full_h <= 0 or rect.width() <= 0 or rect.height() <= 0:
            return

        max_w = min(self._MINIMAP_MAX_WIDTH, full_w, rect.width() / 3)
        max_h = rect.height() / 3
        scale = min(max_w / full_w, max_h / full_h)
        mini_w = int(full_w * scale)
        mini_h = int(full_h * scale)
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


class ZoomStepButton(QPushButton):
    """
    Plain button that steps the camera zoom in or out by one increment.

    Drives ``ZoomPreviewOverlay.zoom()`` directly. Does not take keyboard
    focus on click, so the video label keeps it and the +/- keyboard
    shortcuts keep working right after a button press.
    """

    zoom_step = Signal(int)

    def __init__(self, step: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._step = step
        self.setObjectName("ZoomStepButton")
        self.setFixedSize(30, 30)
        self.setToolTip("Zoom In (+)" if step > 0 else "Zoom Out (-)")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.clicked.connect(self._on_clicked)

    def _on_clicked(self) -> None:
        self.zoom_step.emit(self._step)

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
        painter.drawLine(QPointF(cx - 2.3, cy), QPointF(cx + 2.3, cy))
        if self._step > 0:
            painter.drawLine(QPointF(cx, cy - 2.3), QPointF(cx, cy + 2.3))

        diag = 0.7071
        handle_start = QPointF(cx + r * diag, cy + r * diag)
        handle_end = QPointF(cx + (r + 5.5) * diag, cy + (r + 5.5) * diag)
        painter.drawLine(handle_start, handle_end)

        painter.end()


class ZoomResetButton(QPushButton):
    """
    Plain button that resets zoom and pan back to the fully zoomed-out,
    centered view.

    Drives ``ZoomPreviewOverlay.reset()`` directly. Drawn as a magnifier
    whose lens is an open arc with a counter-clockwise arrowhead at its
    end, rather than a closed circle, so the glyph reads as "reset"
    instead of plain zoom in/out.
    """

    _ARC_START_DEG: float = -144.0
    _ARC_GAP_DEG: float = 57
    _ARROW_LENGTH: float = 4.3
    _ARROW_BOTTOM_ANGLE_DEG: float = 58.6
    _ARROW_TOP_ANGLE_DEG: float = 32.8

    reset_zoom = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ZoomStepButton")
        self.setFixedSize(30, 30)
        self.setToolTip("Reset Zoom")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.clicked.connect(self._on_clicked)

    def _on_clicked(self) -> None:
        self.reset_zoom.emit()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(QColor(0, 0, 0))
        pen.setWidth(2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        cx, cy, r = 13.0, 13.5, 6.0
        lens_rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
        arc_end_deg = self._ARC_START_DEG + (360.0 - self._ARC_GAP_DEG)
        painter.drawArc(lens_rect, int(self._ARC_START_DEG * 16), int((arc_end_deg - self._ARC_START_DEG) * 16))

        theta = math.radians(arc_end_deg)
        e_x = cx + r * math.cos(theta)
        e_y = cy - r * math.sin(theta)
        backward_x, backward_y = math.sin(theta), math.cos(theta)
        outward_x, outward_y = -backward_y, backward_x

        bottom_angle = math.radians(self._ARROW_BOTTOM_ANGLE_DEG)
        top_angle = math.radians(self._ARROW_TOP_ANGLE_DEG)

        bottom_dir_x = backward_x * math.cos(bottom_angle) + outward_x * math.sin(bottom_angle)
        bottom_dir_y = backward_y * math.cos(bottom_angle) + outward_y * math.sin(bottom_angle)
        top_dir_x = backward_x * math.cos(top_angle) - outward_x * math.sin(top_angle)
        top_dir_y = backward_y * math.cos(top_angle) - outward_y * math.sin(top_angle)

        e = QPointF(e_x, e_y)
        bottom_end = QPointF(e_x + bottom_dir_x * self._ARROW_LENGTH, e_y + bottom_dir_y * self._ARROW_LENGTH)
        top_end = QPointF(e_x + top_dir_x * self._ARROW_LENGTH, e_y + top_dir_y * self._ARROW_LENGTH)

        painter.drawLine(bottom_end, e)
        painter.drawLine(top_end, e)

        diag = 0.7071
        handle_start = QPointF(cx + r * diag, cy + r * diag)
        handle_end = QPointF(cx + (r + 5.5) * diag, cy + (r + 5.5) * diag)
        painter.drawLine(handle_start, handle_end)

        painter.end()