from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QBrush, QFont, QFontMetricsF, QPainter, QPen

from UI.widgets.measurements.measurement_style import (
    OVERLAY_DASH_GAP,
    OVERLAY_DASH_LENGTH,
    OVERLAY_ENDPOINT_HIT_RADIUS,
    OVERLAY_ENDPOINT_RADIUS,
    OVERLAY_LABEL_CORNER_RADIUS,
    OVERLAY_LABEL_FONT_SIZE,
    OVERLAY_LABEL_OFFSET,
    OVERLAY_LABEL_PADDING_X,
    OVERLAY_LABEL_PADDING_Y,
    OVERLAY_LINE_COLOR,
    OVERLAY_LINE_WIDTH,
    OVERLAY_OUTLINE_COLOR,
    OVERLAY_OUTLINE_WIDTH,
)
from UI.widgets.measurements.units import MeasurementUnit, format_length
from UI.widgets.preview_overlay.loaded_image_overlay import LoadedImageOverlay
from UI.widgets.preview_overlay.overlay_base import Overlay
from UI.widgets.preview_overlay.zoom_preview import ZoomPreviewOverlay

# How many points each kind needs before it auto-finalizes on a click.
# "Arbitrary Line" is intentionally absent — it keeps accumulating points
# until cancelled (see MeasurementOverlay.place_point/cancel_placement).
CALIBRATION_KIND = "Calibration Line"
_REQUIRED_POINTS = {
    "Point": 1,
    "Horizontal Line": 2,
    "Vertical Line": 2,
    "Radius Circle": 2,
    "Diameter": 2,
    "3 Point Circle": 3,
    CALIBRATION_KIND: 2,
}

PLACEABLE_KINDS = (*_REQUIRED_POINTS, "Arbitrary Line")
_LINE_KINDS = ("Arbitrary Line", "Horizontal Line", "Vertical Line", CALIBRATION_KIND)
_CIRCLE_KINDS = ("Radius Circle", "Diameter", "3 Point Circle")

# Points closer together than this (in frame fractions) are treated as a
# single point rather than a real second click.
_DEGENERATE_EPSILON = 1e-12


class Measurement(NamedTuple):
    kind: str
    points: tuple[tuple[float, float], ...]


def _collinear(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> bool:
    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    return abs(cross) < _DEGENERATE_EPSILON


def _circumcircle(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
) -> tuple[tuple[float, float], float] | None:
    """Center and radius of the circle through three points, or None if they're collinear (no unique circle)."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < _DEGENERATE_EPSILON:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    return (ux, uy), math.hypot(ax - ux, ay - uy)


class MeasurementOverlay(Overlay):
    """
    Draws user-placed measurements — points, lines, and circles — over
    the camera preview.

    Points are stored as fractions (0-1) of the full source frame rather
    than widget or display pixels, so a measurement stays anchored to the
    same spot on the image regardless of window size, letterboxing, or
    the current zoom/pan viewport. ``OverlayLabel._paint_overlays`` wraps
    every overlay's ``draw`` in ``ZoomPreviewOverlay.paint_transform`` and
    expects coordinates given as if *rect* were the whole un-zoomed frame
    — see that method — so drawing straight from fractions of *rect* is
    enough for pan and zoom to fall out automatically, with no zoom-aware
    math needed here for a measurement's actual position or size. Fixed
    on-screen sizes (endpoint markers, stroke widths) are a separate
    concern — see ``ZoomPreviewOverlay.current_scale_xy``.

    Live feed and loaded image each keep their own measurement list;
    which one is active follows ``LoadedImageOverlay.enabled`` the same
    way ``ZoomPreviewOverlay`` keeps separate pan/zoom state per source.

    Placement is click, then click again, for however many points the
    active kind needs (see ``_REQUIRED_POINTS``) — a single click for
    "Point", two for a line or 2-point circle, three for "3 Point
    Circle". ``place_point`` handles every click; it starts a new draft
    on the first, finalizes it once enough points have arrived, and
    (for "Point") finalizes immediately since it only ever needs one.
    "Arbitrary Line" is the exception: it has no fixed point count and
    keeps growing into a multi-segment polyline until ``cancel_placement``
    (a right-click) ends it — dropped entirely if fewer than two points
    were placed, kept as a finished polyline otherwise. ``update_preview``
    drives the dashed in-progress preview on every mouse move between
    clicks.

    Once placed, any point can be dragged to a new position via
    ``begin_endpoint_drag``/``update_endpoint_drag``/``end_endpoint_drag``
    — see ``hit_test_endpoint`` for how a click is matched to one.

    This only records placed measurements and the in-progress draft — no
    actual measuring (length, calibration, etc.) happens here yet.
    """

    def __init__(self) -> None:
        super().__init__()
        self._zoom_handler: ZoomPreviewOverlay | None = None
        self._loaded_image_overlay: LoadedImageOverlay | None = None

        self._live_measurements: list[Measurement] = []
        self._loaded_measurements: list[Measurement] = []

        self._live_dpi: float | None = None
        self._loaded_dpi: float | None = None
        self._unit: MeasurementUnit = MeasurementUnit.MM

        # The camera's live preview stream runs at a lower resolution
        # than its still capture — DPI is calibrated against the still
        # resolution (that's what actually gets saved with a capture's
        # metadata), so live-view length math must scale fraction deltas
        # by *this*, not by whatever frame size the zoom handler happens
        # to have loaded for panning/zooming — see set_live_reference_dims.
        self._live_reference_dims: tuple[int, int] | None = None

        # Manual DPI calibration's single reference line. Kept entirely
        # separate from ``measurements`` — it's a working reference, not
        # a placed measurement. Applies to whichever source is active
        # when placed; CaptureControlWidget cancels it on any mode
        # switch, so it never survives a source change. See
        # start_calibration_placement/calibration_line_length_px.
        self._calibration_line: tuple[tuple[float, float], tuple[float, float]] | None = None

        self._active_type: str | None = None
        self._draft_type: str | None = None
        self._draft_points: list[tuple[float, float]] | None = None
        self._draft_preview: tuple[float, float] | None = None

        self._drag_measurement_index: int | None = None
        self._drag_point_index: int | None = None

    def set_zoom_handler(self, handler: ZoomPreviewOverlay | None) -> None:
        """Register the overlay used to convert clicks into frame-fraction coordinates."""
        self._zoom_handler = handler

    def set_loaded_image_overlay(self, overlay: LoadedImageOverlay | None) -> None:
        self._loaded_image_overlay = overlay

    @property
    def _loaded_active(self) -> bool:
        return self._loaded_image_overlay is not None and self._loaded_image_overlay.enabled

    @property
    def measurements(self) -> list[Measurement]:
        """Placed measurements for whichever source (live feed or loaded image) is currently shown."""
        return self._loaded_measurements if self._loaded_active else self._live_measurements

    def clear(self) -> None:
        """Remove every placed measurement for the currently active source."""
        self.measurements.clear()

    @property
    def has_loaded_measurements(self) -> bool:
        """Whether the loaded-image source specifically has any placed measurements, regardless of which source is active right now."""
        return bool(self._loaded_measurements)

    def clear_loaded(self) -> None:
        """Remove every placed measurement for the loaded-image source specifically."""
        self._loaded_measurements.clear()

    @property
    def dpi(self) -> float | None:
        """DPI for whichever source (live feed or loaded image) is currently shown, or None if unresolved."""
        return self._loaded_dpi if self._loaded_active else self._live_dpi

    def set_live_dpi(self, dpi: float | None) -> None:
        self._live_dpi = dpi if dpi and dpi > 0 else None

    def set_loaded_dpi(self, dpi: float | None) -> None:
        self._loaded_dpi = dpi if dpi and dpi > 0 else None

    def set_unit(self, unit: MeasurementUnit) -> None:
        """Set the unit measurement labels are displayed in. Applies to both sources — it's a display preference, not per-source state like DPI."""
        self._unit = unit

    def set_live_reference_dims(self, dims: tuple[int, int] | None) -> None:
        self._live_reference_dims = dims

    # ------------------------------------------------------------------
    # Manual DPI calibration — a single reference line, placed the same
    # way any 2-point measurement is (see place_point), but finalizing
    # into ``_calibration_line`` instead of ``measurements``. The caller
    # (CaptureControlWidget) reads its pixel length back via
    # calibration_line_length_px, combines it with a user-entered
    # real-world length, and writes the resulting DPI to
    # machine_vision.settings — this overlay only handles placement and
    # geometry, never machine_vision itself.
    # ------------------------------------------------------------------

    def start_calibration_placement(self) -> None:
        self._calibration_line = None
        self.set_active_type(CALIBRATION_KIND)

    def cancel_calibration_placement(self) -> None:
        self._calibration_line = None
        if self._active_type == CALIBRATION_KIND:
            self.set_active_type(None)

    def clear_calibration_line(self) -> None:
        self._calibration_line = None

    @property
    def has_calibration_line(self) -> bool:
        return self._calibration_line is not None

    def calibration_line_length_px(self) -> float | None:
        """
        Pixel length of the placed calibration line at whichever
        resolution DPI actually applies to for the active source — the
        still-capture resolution for live view (see
        _live_reference_dims), or the loaded image's own true resolution
        for a loaded image (the zoom handler's current frame already is
        that, unlike live view's preview-stream frame). None if
        unavailable.
        """
        if self._calibration_line is None:
            return None
        if self._loaded_active:
            dims = self._zoom_handler.current_frame_dims() if self._zoom_handler is not None else None
        else:
            dims = self._live_reference_dims
        if dims is None:
            return None
        length = self._polyline_length_px(self._calibration_line, dims)
        return length if length > 0 else None

    @property
    def active_type(self) -> str | None:
        return self._active_type

    def set_active_type(self, kind: str | None) -> None:
        """Set which kind new clicks should place, or None to disable placement."""
        self._active_type = kind
        self._draft_type = None
        self._draft_points = None
        self._draft_preview = None

    @property
    def drawing_enabled(self) -> bool:
        return self._active_type in PLACEABLE_KINDS

    @property
    def in_progress(self) -> bool:
        return self._draft_points is not None

    # ------------------------------------------------------------------
    # Placement — place_point handles every click, whether it's the
    # first point of a new draft, a middle point, or the one that
    # completes it. Each method returns whether it did anything so
    # callers can decide whether to accept the event without needing
    # try/except.
    # ------------------------------------------------------------------

    def place_point(self, pos: QPoint, widget_rect: QRect) -> bool:
        point = self._to_fraction(pos, widget_rect)
        if point is None:
            return False

        if self._draft_points is None:
            if not self.drawing_enabled:
                return False
            if self._active_type == "Point":
                self.measurements.append(Measurement(self._active_type, (point,)))
                return True
            self._draft_type = self._active_type
            self._draft_points = [point]
            self._draft_preview = None
            return True

        self._draft_points.append(point)
        self._draft_preview = None

        required = _REQUIRED_POINTS.get(self._draft_type)
        if required is None or len(self._draft_points) < required:
            # "Arbitrary Line": no fixed count, keeps accumulating points.
            return True

        kind = self._draft_type
        points = self._draft_points
        self._draft_type = None
        self._draft_points = None
        resolved = self._resolve_measurement(kind, points)
        if resolved is None:
            return False
        if kind == CALIBRATION_KIND:
            self._calibration_line = resolved
        else:
            self.measurements.append(Measurement(kind, resolved))
        return True

    def update_preview(self, pos: QPoint, widget_rect: QRect) -> bool:
        if self._draft_points is None:
            return False
        point = self._to_fraction(pos, widget_rect)
        if point is None:
            return False
        self._draft_preview = point
        return True

    def cancel_placement(self) -> None:
        """
        Drop the in-progress draft — a right-click. An "Arbitrary Line"
        with two or more points already placed is kept as a finished
        polyline instead of discarded, so a multi-point line is finished
        by cancelling once its shape is right rather than needing a
        fixed number of clicks.
        """
        if self._draft_points is not None and self._draft_type == "Arbitrary Line" and len(self._draft_points) >= 2:
            self.measurements.append(Measurement(self._draft_type, tuple(self._draft_points)))
        self._draft_type = None
        self._draft_points = None
        self._draft_preview = None

    def discard_draft(self) -> None:
        """
        Drop the in-progress draft outright, never finalizing it —
        unlike cancel_placement's Arbitrary Line special case. Used when
        switching between live view and a loaded image: a draft belongs
        to whichever source it was started against, so it shouldn't
        silently turn into a finished measurement attributed to the
        source being switched away from.
        """
        self._draft_type = None
        self._draft_points = None
        self._draft_preview = None

    def _to_fraction(self, pos: QPoint, widget_rect: QRect) -> tuple[float, float] | None:
        if self._zoom_handler is None:
            return None
        result = self._zoom_handler.widget_pos_to_full_pixel(pos, widget_rect)
        if result is None:
            return None
        full_px, full_py, full_w, full_h = result
        if full_w <= 0 or full_h <= 0:
            return None
        return full_px / full_w, full_py / full_h

    def _resolve_measurement(
        self,
        kind: str,
        points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], ...] | None:
        """
        Turn the raw clicked points into the tuple actually stored, per
        *kind*, or None if they don't describe a valid measurement (e.g.
        two clicks landed on the same spot, or three on a straight line).
        A horizontal/vertical line holds its first click's row/column
        fixed and spans only to the second click, not the full
        width/height. Circle kinds keep their raw points as clicked —
        the actual center/radius is derived at draw time (see
        ``_circle_geometry``) so a later endpoint drag recomputes it for
        free.
        """
        if kind == "Point":
            return (points[0],) if points else None
        if kind == "Horizontal Line":
            if len(points) < 2 or points[0] == points[1]:
                return None
            y = points[0][1]
            x0, x1 = sorted((points[0][0], points[1][0]))
            return (x0, y), (x1, y)
        if kind == "Vertical Line":
            if len(points) < 2 or points[0] == points[1]:
                return None
            x = points[0][0]
            y0, y1 = sorted((points[0][1], points[1][1]))
            return (x, y0), (x, y1)
        if kind == "Arbitrary Line":
            return tuple(points) if len(points) >= 2 else None
        if kind == CALIBRATION_KIND:
            if len(points) < 2 or points[0] == points[1]:
                return None
            return points[0], points[1]
        if kind in ("Radius Circle", "Diameter"):
            if len(points) < 2 or points[0] == points[1]:
                return None
            return points[0], points[1]
        if kind == "3 Point Circle":
            if len(points) < 3:
                return None
            p1, p2, p3 = points[0], points[1], points[2]
            if p1 == p2 or p2 == p3 or p1 == p3 or _collinear(p1, p2, p3):
                return None
            return p1, p2, p3
        return None

    # ------------------------------------------------------------------
    # Endpoint dragging — moving a point on an already-placed
    # measurement. begin_endpoint_drag hit-tests against the on-screen
    # position every point is actually drawn at (via
    # ZoomPreviewOverlay.widget_pos_for_rect_point), so grabbing works
    # correctly whether zoomed or not.
    # ------------------------------------------------------------------

    @property
    def dragging_endpoint(self) -> bool:
        return self._drag_measurement_index is not None

    def hit_test_endpoint(self, pos: QPoint, rect: QRect, widget_rect: QRect) -> tuple[int, int] | None:
        """Return (measurement index, point index) for the closest placed point within grabbing distance of *pos*, or None."""
        if self._zoom_handler is None:
            return None
        best: tuple[int, int] | None = None
        best_dist_sq = float(OVERLAY_ENDPOINT_HIT_RADIUS * OVERLAY_ENDPOINT_HIT_RADIUS)
        for m_index, measurement in enumerate(self.measurements):
            for p_index, fraction in enumerate(measurement.points):
                screen = self._zoom_handler.widget_pos_for_rect_point(
                    self._to_point(rect, fraction), rect, widget_rect
                )
                dx = screen.x() - pos.x()
                dy = screen.y() - pos.y()
                dist_sq = dx * dx + dy * dy
                if dist_sq <= best_dist_sq:
                    best_dist_sq = dist_sq
                    best = (m_index, p_index)
        return best

    def begin_endpoint_drag(self, pos: QPoint, rect: QRect, widget_rect: QRect) -> bool:
        hit = self.hit_test_endpoint(pos, rect, widget_rect)
        if hit is None:
            return False
        self._drag_measurement_index, self._drag_point_index = hit
        return True

    def update_endpoint_drag(self, pos: QPoint, widget_rect: QRect) -> bool:
        if self._drag_measurement_index is None or self._drag_point_index is None:
            return False
        point = self._to_fraction(pos, widget_rect)
        if point is None:
            return False
        self._move_point(self._drag_measurement_index, self._drag_point_index, point)
        return True

    def end_endpoint_drag(self) -> None:
        self._drag_measurement_index = None
        self._drag_point_index = None

    def _move_point(self, m_index: int, p_index: int, new_point: tuple[float, float]) -> None:
        """
        Apply a dragged point's new position to the measurement at
        *m_index*. Horizontal/vertical lines and a radius circle's
        center get special handling so the shape stays the kind it is;
        every other point (polyline points, circle-edge points, a lone
        "Point") just moves freely.
        """
        measurements = self.measurements
        if m_index < 0 or m_index >= len(measurements):
            return
        measurement = measurements[m_index]
        points = list(measurement.points)
        if p_index < 0 or p_index >= len(points):
            return

        if measurement.kind == "Horizontal Line" and len(points) == 2:
            y = new_point[1]
            other = points[1 - p_index]
            x0, x1 = sorted((new_point[0], other[0]))
            points = [(x0, y), (x1, y)]
        elif measurement.kind == "Vertical Line" and len(points) == 2:
            x = new_point[0]
            other = points[1 - p_index]
            y0, y1 = sorted((new_point[1], other[1]))
            points = [(x, y0), (x, y1)]
        elif measurement.kind == "Radius Circle" and p_index == 0:
            # Dragging the center translates the whole circle rather
            # than just the center point, so the radius doesn't change
            # out from under the (undragged) edge point.
            dx = new_point[0] - points[0][0]
            dy = new_point[1] - points[0][1]
            points = [new_point, (points[1][0] + dx, points[1][1] + dy)]
        else:
            points[p_index] = new_point

        measurements[m_index] = Measurement(measurement.kind, tuple(points))

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _circle_geometry(
        kind: str,
        points: tuple[tuple[float, float], ...],
        full_dims: tuple[int, int],
    ) -> tuple[tuple[float, float], float] | None:
        """
        Return (center_fraction, radius_in_full_pixels) for a circle
        kind, or None if *points* don't yet describe one. Converts to
        true source-pixel coordinates first — points are stored as
        independent x/y fractions of the frame, and the frame is rarely
        square, so a radius computed directly in fraction space would be
        wrong whenever width and height scale differently.
        """
        full_w, full_h = full_dims
        if full_w <= 0 or full_h <= 0:
            return None

        def to_px(p: tuple[float, float]) -> tuple[float, float]:
            return p[0] * full_w, p[1] * full_h

        if kind == "Radius Circle" and len(points) >= 2:
            center_px = to_px(points[0])
            edge_px = to_px(points[1])
            radius = math.hypot(edge_px[0] - center_px[0], edge_px[1] - center_px[1])
            return (points[0], radius) if radius > 0 else None

        if kind == "Diameter" and len(points) >= 2:
            a_px = to_px(points[0])
            b_px = to_px(points[1])
            radius = math.hypot(b_px[0] - a_px[0], b_px[1] - a_px[1]) / 2
            if radius <= 0:
                return None
            center_px = ((a_px[0] + b_px[0]) / 2, (a_px[1] + b_px[1]) / 2)
            return (center_px[0] / full_w, center_px[1] / full_h), radius

        if kind == "3 Point Circle" and len(points) >= 3:
            result = _circumcircle(to_px(points[0]), to_px(points[1]), to_px(points[2]))
            if result is None:
                return None
            center_px, radius = result
            return (center_px[0] / full_w, center_px[1] / full_h), radius

        return None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, painter: QPainter, rect: QRect) -> None:
        scale_x, scale_y = self._zoom_handler.current_scale_xy() if self._zoom_handler is not None else (1.0, 1.0)
        stroke_scale = (scale_x + scale_y) / 2
        full_dims = self._zoom_handler.current_frame_dims() if self._zoom_handler is not None else None

        for measurement in self.measurements:
            self._draw_measurement(painter, rect, measurement.kind, measurement.points, stroke_scale, full_dims)
            for point in measurement.points:
                self._draw_endpoint(painter, self._to_point(rect, point), scale_x, scale_y)
            self._draw_measurement_label(painter, rect, measurement.kind, measurement.points, stroke_scale, full_dims)

        if self._calibration_line is not None:
            self._draw_polyline(painter, rect, self._calibration_line, stroke_scale, dashed=False)
            for point in self._calibration_line:
                self._draw_endpoint(painter, self._to_point(rect, point), scale_x, scale_y)
            self._draw_label(painter, rect, self._midpoint(self._calibration_line), "Calibration", stroke_scale)

        self._draw_draft(painter, rect, stroke_scale, scale_x, scale_y, full_dims)

    def _draw_measurement(
        self,
        painter: QPainter,
        rect: QRect,
        kind: str,
        points: tuple[tuple[float, float], ...],
        stroke_scale: float,
        full_dims: tuple[int, int] | None,
        *,
        dashed: bool = False,
    ) -> None:
        if kind in _LINE_KINDS:
            self._draw_polyline(painter, rect, points, stroke_scale, dashed=dashed)
        elif kind in _CIRCLE_KINDS and full_dims is not None:
            self._draw_circle(painter, rect, kind, points, stroke_scale, full_dims, dashed=dashed)

    def _draw_measurement_label(
        self,
        painter: QPainter,
        rect: QRect,
        kind: str,
        points: tuple[tuple[float, float], ...],
        stroke_scale: float,
        full_dims: tuple[int, int] | None,
    ) -> None:
        """
        Draw the real-world length/diameter next to a finalized
        measurement — the whole point of DPI. Silently does nothing
        until both a reference frame size and a DPI are available, so a
        shape placed before DPI is set just shows up unlabeled rather
        than blocking placement (see ``drawing_enabled``).

        For live view, *full_dims* (the zoom handler's current frame —
        the preview stream, sized for panning/zooming) is not what DPI
        was calibrated against, so ``_live_reference_dims`` (the still
        capture resolution) is used instead when available. A loaded
        image's *full_dims* is already its own true resolution, so it's
        used as-is.
        """
        dims = full_dims if self._loaded_active else (self._live_reference_dims or full_dims)
        if dims is None or self.dpi is None:
            return
        text_and_anchor = self._measurement_label(kind, points, dims)
        if text_and_anchor is None:
            return
        text, anchor = text_and_anchor
        self._draw_label(painter, rect, anchor, text, stroke_scale)

    def _measurement_label(
        self,
        kind: str,
        points: tuple[tuple[float, float], ...],
        full_dims: tuple[int, int],
    ) -> tuple[str, tuple[float, float]] | None:
        if kind == CALIBRATION_KIND:
            return None

        if kind in _LINE_KINDS:
            length_px = self._polyline_length_px(points, full_dims)
            if length_px <= 0:
                return None
            return format_length(length_px, self.dpi, self._unit), self._midpoint(points)

        if kind in _CIRCLE_KINDS:
            geometry = self._circle_geometry(kind, points, full_dims)
            if geometry is None:
                return None
            center, radius_px = geometry
            return f"\u00d8 {format_length(radius_px * 2, self.dpi, self._unit)}", center

        return None

    @staticmethod
    def _polyline_length_px(points: tuple[tuple[float, float], ...], full_dims: tuple[int, int]) -> float:
        full_w, full_h = full_dims
        total = 0.0
        for i in range(len(points) - 1):
            dx = (points[i + 1][0] - points[i][0]) * full_w
            dy = (points[i + 1][1] - points[i][1]) * full_h
            total += math.hypot(dx, dy)
        return total

    @staticmethod
    def _midpoint(points: tuple[tuple[float, float], ...]) -> tuple[float, float]:
        """
        Anchor for a polyline's label. A straight 2-point line has one
        segment and anchors at its midpoint as before. A multi-segment
        "Arbitrary Line" instead anchors at whichever segment is
        middlemost — its own midpoint if there's a single middle segment
        (an odd segment count), or the joint shared by the two middle
        segments if there's an even count — so the label tracks whichever
        interior point actually governs that segment when it's dragged,
        rather than staying fixed to the two endpoints regardless of what
        moved in between.
        """
        segment_count = len(points) - 1
        if segment_count <= 1:
            first, last = points[0], points[-1]
            return (first[0] + last[0]) / 2, (first[1] + last[1]) / 2
        if segment_count % 2 == 0:
            return points[segment_count // 2]
        a, b = points[segment_count // 2], points[segment_count // 2 + 1]
        return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2

    def _draw_label(
        self,
        painter: QPainter,
        rect: QRect,
        anchor: tuple[float, float],
        text: str,
        stroke_scale: float,
    ) -> None:
        """
        White rounded-rect background with a thin black outline, dark
        text on top — reads clearly regardless of what's under it, unlike
        the outlined-stroke technique used for lines and endpoints (a
        thin outline around small text gets muddy at typical font sizes).
        Sized and offset in the same fixed-on-screen-size terms as the
        rest of the overlay, divided by *stroke_scale* so it doesn't
        grow with zoom.
        """
        point = self._to_point(rect, anchor)

        font = QFont(painter.font())
        font.setPixelSize(max(1, round(OVERLAY_LABEL_FONT_SIZE / stroke_scale)))
        metrics = QFontMetricsF(font)
        pad_x = OVERLAY_LABEL_PADDING_X / stroke_scale
        pad_y = OVERLAY_LABEL_PADDING_Y / stroke_scale
        box_w = metrics.horizontalAdvance(text) + pad_x * 2
        box_h = metrics.height() + pad_y * 2

        box = QRectF(
            point.x() - box_w / 2,
            point.y() - OVERLAY_LABEL_OFFSET / stroke_scale - box_h,
            box_w,
            box_h,
        )

        painter.setFont(font)
        painter.setPen(QPen(OVERLAY_OUTLINE_COLOR, OVERLAY_OUTLINE_WIDTH / stroke_scale))
        painter.setBrush(QBrush(OVERLAY_LINE_COLOR))
        painter.drawRoundedRect(box, OVERLAY_LABEL_CORNER_RADIUS / stroke_scale, OVERLAY_LABEL_CORNER_RADIUS / stroke_scale)

        painter.setPen(QPen(OVERLAY_OUTLINE_COLOR))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_draft(
        self,
        painter: QPainter,
        rect: QRect,
        stroke_scale: float,
        scale_x: float,
        scale_y: float,
        full_dims: tuple[int, int] | None,
    ) -> None:
        if self._draft_points is None:
            return
        kind = self._draft_type
        points = self._draft_points
        preview = self._draft_preview

        if kind == "Arbitrary Line":
            # Confirmed segments are drawn solid — they're placed, just
            # not yet finalized into a stored Measurement — and only the
            # pending segment to the cursor dashes, so it's clear which
            # part of the shape is still being decided.
            if len(points) >= 2:
                self._draw_polyline(painter, rect, points, stroke_scale, dashed=False)
            if preview is not None:
                self._draw_polyline(painter, rect, (points[-1], preview), stroke_scale, dashed=True)

            label_points = (*points, preview) if preview is not None else tuple(points)
            if len(label_points) >= 2:
                self._draw_measurement_label(painter, rect, kind, label_points, stroke_scale, full_dims)
        else:
            preview_points = (*points, preview) if preview is not None else tuple(points)
            display_points = preview_points

            if kind in ("Horizontal Line", "Vertical Line") and len(preview_points) >= 2:
                # Draw the same row/column-constrained shape the click
                # would actually finalize, not the raw (diagonal) two
                # points — see _resolve_measurement.
                resolved = self._resolve_measurement(kind, list(preview_points))
                if resolved is not None:
                    display_points = resolved

            self._draw_measurement(painter, rect, kind, display_points, stroke_scale, full_dims, dashed=True)

            if kind in _CIRCLE_KINDS and len(preview_points) >= 2:
                required = _REQUIRED_POINTS.get(kind, len(preview_points))
                if len(preview_points) < required:
                    # Not enough points for a circle yet — a straight
                    # guide between what's placed so far is still useful
                    # feedback (e.g. two of three "3 Point Circle" clicks).
                    self._draw_polyline(painter, rect, preview_points, stroke_scale, dashed=True)

            if kind == CALIBRATION_KIND:
                if len(display_points) >= 2:
                    self._draw_label(painter, rect, self._midpoint(tuple(display_points)), "Calibration", stroke_scale)
            elif len(display_points) >= 2:
                self._draw_measurement_label(painter, rect, kind, tuple(display_points), stroke_scale, full_dims)

            points = list(display_points)

        for point in points:
            self._draw_endpoint(painter, self._to_point(rect, point), scale_x, scale_y)
        if preview is not None and kind == "Arbitrary Line":
            self._draw_endpoint(painter, self._to_point(rect, preview), scale_x, scale_y)

    def _draw_polyline(
        self,
        painter: QPainter,
        rect: QRect,
        points: tuple[tuple[float, float], ...],
        stroke_scale: float,
        *,
        dashed: bool,
    ) -> None:
        for i in range(len(points) - 1):
            self._draw_stroke(painter, rect, points[i], points[i + 1], stroke_scale, dashed=dashed)

    def _draw_circle(
        self,
        painter: QPainter,
        rect: QRect,
        kind: str,
        points: tuple[tuple[float, float], ...],
        stroke_scale: float,
        full_dims: tuple[int, int],
        *,
        dashed: bool,
    ) -> None:
        geometry = self._circle_geometry(kind, points, full_dims)
        if geometry is None:
            return
        center, radius_px = geometry
        full_w, full_h = full_dims

        # The circle's shape is genuine image content, not fixed UI
        # chrome, so — unlike the endpoint markers — it's sized directly
        # from the frame's own aspect ratio here rather than counter-
        # scaled against zoom; the ambient paint transform (see the
        # class docstring) does the rest, the same as a line's length.
        center_point = self._to_point(rect, center)
        rx = radius_px * (rect.width() / full_w)
        ry = radius_px * (rect.height() / full_h)

        outline_width = OVERLAY_LINE_WIDTH + OVERLAY_OUTLINE_WIDTH * 2
        outline = QPen(OVERLAY_OUTLINE_COLOR)
        outline.setWidthF(outline_width / stroke_scale)
        self._apply_dash(outline, outline_width, dashed)
        painter.setPen(outline)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center_point, rx, ry)

        fill = QPen(OVERLAY_LINE_COLOR)
        fill.setWidthF(OVERLAY_LINE_WIDTH / stroke_scale)
        self._apply_dash(fill, OVERLAY_LINE_WIDTH, dashed)
        painter.setPen(fill)
        painter.drawEllipse(center_point, rx, ry)

        if kind in ("Radius Circle", "Diameter") and dashed:
            # Matches the tile icon while placing — a straight line from
            # center to edge (Radius Circle) or all the way across
            # (Diameter) — but once finalized the circle's own outline
            # already shows the same thing, so the guide line is dropped;
            # only the point markers (drawn separately, per point, in
            # draw()) stick around, including Radius Circle's center.
            self._draw_stroke(painter, rect, points[0], points[1], stroke_scale, dashed=dashed)

    def _draw_stroke(
        self,
        painter: QPainter,
        rect: QRect,
        start: tuple[float, float],
        end: tuple[float, float],
        stroke_scale: float,
        *,
        dashed: bool,
    ) -> None:
        """
        A white stroke over a slightly wider black one gives the line a
        thin outline that reads against both light and dark backgrounds,
        rather than a plain fill picked to contrast one or the other.
        Widths are divided by *stroke_scale* — the average of
        ``ZoomPreviewOverlay``'s current x/y paint-transform scale — so
        they stay a constant size on screen instead of growing with
        zoom. *dashed* draws the in-progress preview in this same style,
        solidifying only once the measurement is actually placed.
        """
        p1 = self._to_point(rect, start)
        p2 = self._to_point(rect, end)
        outline_width = OVERLAY_LINE_WIDTH + OVERLAY_OUTLINE_WIDTH * 2

        outline = QPen(OVERLAY_OUTLINE_COLOR)
        outline.setWidthF(outline_width / stroke_scale)
        outline.setCapStyle(Qt.PenCapStyle.RoundCap)
        self._apply_dash(outline, outline_width, dashed)
        painter.setPen(outline)
        painter.drawLine(p1, p2)

        fill = QPen(OVERLAY_LINE_COLOR)
        fill.setWidthF(OVERLAY_LINE_WIDTH / stroke_scale)
        fill.setCapStyle(Qt.PenCapStyle.RoundCap)
        self._apply_dash(fill, OVERLAY_LINE_WIDTH, dashed)
        painter.setPen(fill)
        painter.drawLine(p1, p2)

    @staticmethod
    def _apply_dash(pen: QPen, stroke_width: float, dashed: bool) -> None:
        """
        Qt's dash pattern is specified in multiples of the pen's own
        width, so the same pattern values on the wider outline pen and
        the thinner fill pen drawn over it produce different absolute
        dash lengths — misaligning the two. Dividing OVERLAY_DASH_LENGTH/
        _GAP (screen-pixel targets) by each pen's own *stroke_width*
        first cancels that out, so both strokes dash at the same
        absolute, on-screen length regardless of their width.
        """
        if not dashed:
            pen.setStyle(Qt.PenStyle.SolidLine)
            return
        pen.setStyle(Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([OVERLAY_DASH_LENGTH / stroke_width, OVERLAY_DASH_GAP / stroke_width])

    def _draw_endpoint(self, painter: QPainter, point: QPointF, scale_x: float, scale_y: float) -> None:
        """
        Uses *scale_x*/*scale_y* independently, rather than a single
        averaged scale, so the drawn ellipse is a true circle on screen
        even when the zoom overlay's current stretch isn't uniform
        between the two axes — see ``ZoomPreviewOverlay.current_scale_xy``.
        """
        rx = OVERLAY_ENDPOINT_RADIUS / scale_x
        ry = OVERLAY_ENDPOINT_RADIUS / scale_y
        pen = QPen(OVERLAY_OUTLINE_COLOR)
        pen.setWidthF(OVERLAY_OUTLINE_WIDTH / ((scale_x + scale_y) / 2))
        painter.setPen(pen)
        painter.setBrush(QBrush(OVERLAY_LINE_COLOR))
        painter.drawEllipse(point, rx, ry)

    @staticmethod
    def _to_point(rect: QRect, fraction: tuple[float, float]) -> QPointF:
        fx, fy = fraction
        return QPointF(rect.x() + fx * rect.width(), rect.y() + fy * rect.height())


class MeasurementOverlayController:
    """
    Measurement/DPI/calibration control surface — exposed to the rest of
    the app as ``CameraPreview.overlays.measurement``.

    Kept apart from OverlayController's much broader, unrelated surface
    (crosshair, grid, focus, channel filters, ...) so measurement- and
    calibration-specific control logic lives next to the overlay it
    actually drives, rather than scattered through camera_preview.py.
    Every mutator repaints via *repaint* afterward the same way
    OverlayController's own setters do — CameraPreview passes its video
    label's ``update`` method in.
    """

    def __init__(self, overlay: MeasurementOverlay, repaint: Callable[[], None]) -> None:
        self._overlay = overlay
        self._repaint = repaint

    @property
    def type(self) -> str | None:
        return self._overlay.active_type

    @type.setter
    def type(self, kind: str | None) -> None:
        """Set which measurement kind new clicks on the preview should place, or None to disable placement."""
        self._overlay.set_active_type(kind)
        self._repaint()

    @property
    def dpi(self) -> float | None:
        """DPI for whichever source (live feed or loaded image) the overlay is currently showing."""
        return self._overlay.dpi

    def set_live_dpi(self, dpi: float | None) -> None:
        self._overlay.set_live_dpi(dpi)
        self._repaint()

    def set_live_reference_dims(self, dims: tuple[int, int] | None) -> None:
        self._overlay.set_live_reference_dims(dims)
        self._repaint()

    def set_loaded_dpi(self, dpi: float | None) -> None:
        self._overlay.set_loaded_dpi(dpi)
        self._repaint()

    def set_unit(self, unit: MeasurementUnit) -> None:
        self._overlay.set_unit(unit)
        self._repaint()

    def discard_draft(self) -> None:
        self._overlay.discard_draft()
        self._repaint()

    def start_calibration_placement(self) -> None:
        self._overlay.start_calibration_placement()
        self._repaint()

    def cancel_calibration_placement(self) -> None:
        self._overlay.cancel_calibration_placement()
        self._repaint()

    def clear_calibration_line(self) -> None:
        self._overlay.clear_calibration_line()
        self._repaint()

    @property
    def has_calibration_line(self) -> bool:
        return self._overlay.has_calibration_line

    def calibration_line_length_px(self) -> float | None:
        return self._overlay.calibration_line_length_px()