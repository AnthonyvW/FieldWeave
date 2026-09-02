from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QTransform,
)

from UI.widgets.measurements.lines import CALIBRATION_KIND, LINE_KINDS, arrow_dims, arrow_head_path, open_arrow_barbs_path
from UI.widgets.measurements.measurement_meta import DEFAULT_META, MeasurementMeta
from UI.widgets.measurements.measurement_style import (
    OVERLAY_DASH_GAP,
    OVERLAY_DASH_LENGTH,
    OVERLAY_DELETE_BG_COLOR,
    OVERLAY_DELETE_GLYPH_COLOR,
    OVERLAY_DELETE_MARGIN,
    OVERLAY_DELETE_SIZE,
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
    OVERLAY_POINT_RADIUS,
    OVERLAY_TAG_HOVER_COLOR,
)
from UI.widgets.measurements.units import MeasurementUnit, format_length
from UI.widgets.preview_overlay.loaded_image_overlay import LoadedImageOverlay
from UI.widgets.preview_overlay.overlay_base import Overlay
from UI.widgets.preview_overlay.zoom_preview import ZoomPreviewOverlay

# How many points each kind needs before it auto-finalizes on a click.
# "Arbitrary Line" is intentionally absent — it keeps accumulating points
# until cancelled (see MeasurementOverlay.place_point/cancel_placement).
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
_CIRCLE_KINDS = ("Radius Circle", "Diameter", "3 Point Circle")

# Raw screen-pixel dash/gap values at a fixed small reference width —
# used only to render the customize menu's style-picker icons (see
# MeasurementCustomizeMenu / _dash_style_icon), which draw at their own
# fixed small pen width and don't need width-aware scaling. Exposed as
# MEASUREMENT_DASH_STYLES so the menu can build its option list from the
# same names rather than duplicating them, and MEASUREMENT_DASH_PATTERNS
# for the icons themselves.
_DASH_PATTERNS: dict[str, list[float] | None] = {
    "solid": None,
    "dash": [4, 3],
    "dot": [1, 2],
    "dash_dot": [4, 2, 1, 2],
    "dash_dot_dot": [4, 2, 1, 2, 1, 2],
    "long_dash": [9, 4],
}
MEASUREMENT_DASH_STYLES = tuple(_DASH_PATTERNS.keys())
MEASUREMENT_DASH_PATTERNS = _DASH_PATTERNS

# Dash/gap values as multiples of the line's own chosen thickness,
# for the actual overlay rendering (see resolve_dash_pattern) — a fixed
# pixel gap reads fine at the default thickness but nearly vanishes
# under a thick round-capped stroke, since the round caps on either
# side of the gap grow with it and can bridge straight across a small
# fixed gap. Scaling by thickness keeps the gaps (and dots, which
# otherwise fuse into the stroke entirely) legible at any thickness.
_DASH_MULTIPLIERS: dict[str, list[float] | None] = {
    "solid": None,
    "dash": [4.0, 3.0],
    "dot": [1.2, 3.0],
    "dash_dot": [4.0, 2.5, 1.2, 2.5],
    "dash_dot_dot": [4.0, 2.0, 1.2, 2.0, 1.2, 2.0],
    "long_dash": [7.0, 3.5],
}
_DASH_REFERENCE_MIN = 3.0  # keeps a thin line's dashes/gaps from shrinking below a legible size


def resolve_dash_pattern(dash_style: str, line_width: float) -> list[float] | None:
    """Screen-pixel dash pattern for *dash_style* at *line_width*, or None for solid/unrecognized — see _DASH_MULTIPLIERS."""
    multipliers = _DASH_MULTIPLIERS.get(dash_style)
    if not multipliers:
        return None
    reference = max(line_width, _DASH_REFERENCE_MIN)
    return [value * reference for value in multipliers]


# Points closer together than this (in frame fractions) are treated as a
# single point rather than a real second click.
_DEGENERATE_EPSILON = 1e-12


class Measurement(NamedTuple):
    kind: str
    points: tuple[tuple[float, float], ...]
    meta: MeasurementMeta = DEFAULT_META


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

        # Template applied to a measurement's meta as it's finalized —
        # set from MeasurementsWidget's "Customize Measurements" section
        # (see set_default_meta) so newly placed measurements pick up
        # whatever description/unit the user has set up ahead of time.
        self._default_meta: MeasurementMeta = DEFAULT_META
        self._kind_counts: dict[str, int] = {}

        # Tags' own label boxes from the most recent draw() call, in the
        # same pre-paint-transform coordinate frame the boxes were drawn
        # in — kept so hover/click hit-testing (which runs against real
        # screen coordinates from mouse events, not paint calls) can map
        # them through the zoom handler the same way hit_test_endpoint
        # maps a placed point. Rebuilt from scratch on every draw(), so a
        # measurement that stops being labeled (or is removed) naturally
        # drops out.
        self._label_boxes: dict[int, QRectF] = {}
        self._delete_boxes: dict[int, QRectF] = {}
        self._hovered_index: int | None = None
        self._hover_delete: bool = False

        # Which measurement's anchor points are drawn — only the one
        # the cursor is currently near (see update_proximity), plus
        # whichever one has an endpoint actively being dragged, so
        # placed points don't otherwise clutter the view.
        self._near_index: int | None = None

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
                self.measurements.append(Measurement(self._active_type, (point,), self._resolve_meta(self._active_type)))
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
            self.measurements.append(Measurement(kind, resolved, self._resolve_meta(kind)))
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
            self.measurements.append(
                Measurement(self._draft_type, tuple(self._draft_points), self._resolve_meta(self._draft_type))
            )
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

    # ------------------------------------------------------------------
    # Customization — title/description/unit per placed measurement.
    # set_default_meta is the template new placements pick up (see
    # _resolve_meta); the rest edit an already-placed measurement, driven
    # by the on-canvas tag menu.
    # ------------------------------------------------------------------

    def set_default_meta(self, meta: MeasurementMeta) -> None:
        self._default_meta = meta

    def _resolve_meta(self, kind: str) -> MeasurementMeta:
        """
        No title by default — only number one in once the sidebar's
        default title template is actually set (e.g. "Wingspan 1",
        "Wingspan 2", ...), so a measurement placed with nothing
        configured stays untitled rather than falling back to its own
        kind name.
        """
        if not self._default_meta.title:
            return self._default_meta
        self._kind_counts[kind] = self._kind_counts.get(kind, 0) + 1
        return self._default_meta._replace(title=f"{self._default_meta.title} {self._kind_counts[kind]}")

    def measurement_meta(self, index: int) -> MeasurementMeta | None:
        measurements = self.measurements
        if index < 0 or index >= len(measurements):
            return None
        return measurements[index].meta

    def measurement_kind(self, index: int) -> str | None:
        measurements = self.measurements
        if index < 0 or index >= len(measurements):
            return None
        return measurements[index].kind

    def set_measurement_meta(self, index: int, meta: MeasurementMeta) -> bool:
        measurements = self.measurements
        if index < 0 or index >= len(measurements):
            return False
        measurements[index] = measurements[index]._replace(meta=meta)
        return True

    def remove_measurement(self, index: int) -> bool:
        measurements = self.measurements
        if index < 0 or index >= len(measurements):
            return False
        del measurements[index]
        if self._hovered_index == index:
            self._hovered_index = None
            self._hover_delete = False
        self._label_boxes.pop(index, None)
        self._delete_boxes.pop(index, None)
        return True

    # ------------------------------------------------------------------
    # Tag hover/hit-testing — mirrors hit_test_endpoint's pattern of
    # mapping a pre-paint-transform point through the zoom handler, but
    # against the label boxes recorded by the most recent draw() rather
    # than a placed point.
    # ------------------------------------------------------------------

    @property
    def hovered_index(self) -> int | None:
        return self._hovered_index

    @property
    def hover_delete(self) -> bool:
        return self._hover_delete

    def clear_hover(self) -> bool:
        changed = self._hovered_index is not None
        self._hovered_index = None
        self._hover_delete = False
        return changed

    def update_hover(self, pos: QPoint, rect: QRect, widget_rect: QRect) -> bool:
        """Recompute which tag (if any) *pos* is over, and whether it's over that tag's delete glyph. Returns True if either changed, so the caller knows whether to repaint."""
        index, over_delete = self._hit_test_tag(pos, rect, widget_rect)
        changed = index != self._hovered_index or over_delete != self._hover_delete
        self._hovered_index, self._hover_delete = index, over_delete
        return changed

    def label_screen_rect(self, index: int, rect: QRect, widget_rect: QRect) -> QRectF | None:
        """Screen-space box of measurement *index*'s tag from the most recent draw(), or None if it isn't currently tagged — used to anchor MeasurementCustomizeMenu under the actual tag rather than wherever within it the opening click happened to land."""
        box = self._label_boxes.get(index)
        if box is None or self._zoom_handler is None:
            return None
        return self._screen_rect(box, rect, widget_rect)

    def _hit_test_tag(self, pos: QPoint, rect: QRect, widget_rect: QRect) -> tuple[int | None, bool]:
        if self._zoom_handler is None:
            return None, False
        for index, box in self._label_boxes.items():
            screen_box = self._screen_rect(box, rect, widget_rect)
            if not screen_box.contains(QPointF(pos)):
                continue
            delete_box = self._delete_boxes.get(index)
            over_delete = delete_box is not None and self._screen_rect(delete_box, rect, widget_rect).contains(QPointF(pos))
            return index, over_delete
        return None, False

    def _screen_rect(self, box: QRectF, rect: QRect, widget_rect: QRect) -> QRectF:
        top_left = self._zoom_handler.widget_pos_for_rect_point(box.topLeft(), rect, widget_rect)
        bottom_right = self._zoom_handler.widget_pos_for_rect_point(box.bottomRight(), rect, widget_rect)
        return QRectF(QPointF(top_left), QPointF(bottom_right))

    def _screen_point(self, point: QPointF, rect: QRect, widget_rect: QRect) -> QPointF:
        return QPointF(self._zoom_handler.widget_pos_for_rect_point(point, rect, widget_rect))

    # ------------------------------------------------------------------
    # Proximity — which measurement's anchor points (if any) should be
    # drawn this frame. Kept separate from tag hover: a measurement's
    # points can be far from its tag (a long line's tag sits at its
    # midpoint), so being near one doesn't imply being near the other.
    # ------------------------------------------------------------------

    @property
    def near_index(self) -> int | None:
        return self._near_index

    def clear_proximity(self) -> bool:
        changed = self._near_index is not None
        self._near_index = None
        return changed

    def update_proximity(self, pos: QPoint, rect: QRect, widget_rect: QRect) -> bool:
        """Recompute which measurement (if any) *pos* is close enough to that its anchor points should show. Returns True if it changed, so the caller knows whether to repaint."""
        index = self._hit_test_proximity(pos, rect, widget_rect)
        changed = index != self._near_index
        self._near_index = index
        return changed

    def _hit_test_proximity(self, pos: QPoint, rect: QRect, widget_rect: QRect) -> int | None:
        if self._zoom_handler is None:
            return None
        cursor = QPointF(pos)
        full_dims = self._zoom_handler.current_frame_dims()
        for index, measurement in enumerate(self.measurements):
            screen_points = [self._screen_point(self._to_point(rect, p), rect, widget_rect) for p in measurement.points]
            if any(self._distance(cursor, sp) <= OVERLAY_ENDPOINT_HIT_RADIUS for sp in screen_points):
                return index
            if measurement.kind in LINE_KINDS:
                for a, b in zip(screen_points, screen_points[1:]):
                    if self._distance_to_segment(cursor, a, b) <= OVERLAY_ENDPOINT_HIT_RADIUS:
                        return index
            elif measurement.kind in _CIRCLE_KINDS and full_dims is not None:
                edge_points = self._circle_edge_screen_points(
                    measurement.kind, measurement.points, rect, widget_rect, full_dims
                )
                for a, b in zip(edge_points, edge_points[1:] + edge_points[:1]):
                    if self._distance_to_segment(cursor, a, b) <= OVERLAY_ENDPOINT_HIT_RADIUS:
                        return index
        return None

    _CIRCLE_EDGE_SAMPLES = 32

    def _circle_edge_screen_points(
        self,
        kind: str,
        points: tuple[tuple[float, float], ...],
        rect: QRect,
        widget_rect: QRect,
        full_dims: tuple[int, int],
    ) -> list[QPointF]:
        """
        *_CIRCLE_EDGE_SAMPLES* screen points sampled evenly around the
        circle *kind*/*points* actually describes — reusing the same
        _to_point/_screen_point pipeline draw() and every other
        proximity check goes through, rather than reasoning about the
        ellipse and zoom/pan transform analytically, so a click near the
        drawn boundary anywhere along it (not just at one of its own
        defining points) counts as being near this measurement, the
        same way a point anywhere along a line segment already does.
        """
        geometry = self._circle_geometry(kind, points, full_dims)
        if geometry is None:
            return []
        (cx, cy), radius_px = geometry
        full_w, full_h = full_dims
        result = []
        for i in range(self._CIRCLE_EDGE_SAMPLES):
            angle = 2 * math.pi * i / self._CIRCLE_EDGE_SAMPLES
            fraction = (
                cx + (radius_px * math.cos(angle)) / full_w,
                cy + (radius_px * math.sin(angle)) / full_h,
            )
            result.append(self._screen_point(self._to_point(rect, fraction), rect, widget_rect))
        return result

    @staticmethod
    def _distance(p: QPointF, q: QPointF) -> float:
        return math.hypot(p.x() - q.x(), p.y() - q.y())

    @staticmethod
    def _distance_to_segment(p: QPointF, a: QPointF, b: QPointF) -> float:
        dx, dy = b.x() - a.x(), b.y() - a.y()
        length_sq = dx * dx + dy * dy
        if length_sq <= 0:
            return math.hypot(p.x() - a.x(), p.y() - a.y())
        t = max(0.0, min(1.0, ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / length_sq))
        proj_x, proj_y = a.x() + t * dx, a.y() + t * dy
        return math.hypot(p.x() - proj_x, p.y() - proj_y)

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

        measurements[m_index] = measurement._replace(points=tuple(points))

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

        self._label_boxes = {}
        self._delete_boxes = {}

        for index, measurement in enumerate(self.measurements):
            meta = measurement.meta
            line_color = self._resolve_color(meta.line_color, OVERLAY_LINE_COLOR)
            line_width = meta.line_thickness or OVERLAY_LINE_WIDTH
            outline_color = self._resolve_color(meta.outline_color, OVERLAY_OUTLINE_COLOR)
            # A disabled outline is drawn at zero width rather than
            # skipped outright: its pass then paints the exact same
            # shape as the fill pass drawn right after it (same dash
            # pattern, same cap shapes — see _cap_shapes/_stroke_path),
            # which fully covers it, rather than needing a second code
            # path through every draw method just to omit one pass.
            outline_width = (meta.outline_thickness or OVERLAY_OUTLINE_WIDTH) if meta.outline_enabled else 0.0
            self._draw_measurement(
                painter, rect, measurement.kind, measurement.points, stroke_scale, scale_x, scale_y, full_dims,
                line_color=line_color, line_width=line_width,
                outline_color=outline_color, outline_width=outline_width,
                dash_style=meta.line_dash_style, start_cap=meta.line_start_cap, end_cap=meta.line_end_cap,
            )
            if index == self._near_index or index == self._drag_measurement_index:
                for point in measurement.points:
                    self._draw_endpoint(painter, self._to_point(rect, point), scale_x, scale_y)
            self._draw_measurement_label(painter, rect, index, measurement, scale_x, scale_y, full_dims)

        if self._calibration_line is not None:
            self._draw_polyline(painter, rect, self._calibration_line, stroke_scale, dashed=False)
            for point in self._calibration_line:
                self._draw_endpoint(painter, self._to_point(rect, point), scale_x, scale_y)
            self._draw_label(painter, rect, self._midpoint(self._calibration_line), "Calibration", scale_x, scale_y)

        self._draw_draft(painter, rect, stroke_scale, scale_x, scale_y, full_dims)

    def _draw_measurement(
        self,
        painter: QPainter,
        rect: QRect,
        kind: str,
        points: tuple[tuple[float, float], ...],
        stroke_scale: float,
        scale_x: float,
        scale_y: float,
        full_dims: tuple[int, int] | None,
        *,
        dashed: bool = False,
        line_color: QColor = OVERLAY_LINE_COLOR,
        line_width: float = OVERLAY_LINE_WIDTH,
        outline_color: QColor = OVERLAY_OUTLINE_COLOR,
        outline_width: float = OVERLAY_OUTLINE_WIDTH,
        dash_style: str = "solid",
        start_cap: str = "curved",
        end_cap: str = "curved",
    ) -> None:
        if kind in LINE_KINDS:
            self._draw_polyline(
                painter, rect, points, stroke_scale, dashed=dashed,
                line_color=line_color, line_width=line_width, outline_color=outline_color, outline_width=outline_width,
                dash_style=dash_style, start_cap=start_cap, end_cap=end_cap,
            )
        elif kind in _CIRCLE_KINDS and full_dims is not None:
            self._draw_circle(
                painter, rect, kind, points, stroke_scale, full_dims,
                dashed=dashed, line_color=line_color, line_width=line_width,
                outline_color=outline_color, outline_width=outline_width, dash_style=dash_style,
            )
        elif kind == "Point" and points:
            self._draw_point_marker(
                painter, rect, points[0], scale_x, scale_y,
                line_color=line_color, line_width=line_width,
                outline_color=outline_color, outline_width=outline_width,
            )

    def _draw_point_marker(
        self,
        painter: QPainter,
        rect: QRect,
        point: tuple[float, float],
        scale_x: float,
        scale_y: float,
        *,
        line_color: QColor = OVERLAY_LINE_COLOR,
        line_width: float = OVERLAY_LINE_WIDTH,
        outline_color: QColor = OVERLAY_OUTLINE_COLOR,
        outline_width: float = OVERLAY_OUTLINE_WIDTH,
    ) -> None:
        """
        A filled dot for a placed "Point" measurement — the one kind
        _draw_measurement handled nowhere before, so a placed point was
        only ever visible while hovered/dragged (see draw()'s own
        endpoint markers) rather than on its own. *line_width* sets the
        dot's radius, scaled off OVERLAY_POINT_RADIUS the same
        proportion OVERLAY_LINE_WIDTH would scale a line's stroke — a
        point has no length for "thickness" to describe, but "Line
        Thickness" still does something sensible: a bigger dot.

        Uses *scale_x*/*scale_y* independently for the dot itself,
        exactly like _draw_endpoint, rather than a single averaged
        scale — at zoom levels where the crop's aspect ratio hasn't yet
        caught up to the widget's (see ZoomPreviewOverlay.current_scale_xy),
        those two differ, and a radius corrected by their average comes
        out stretched into an ellipse along whichever axis is scaled
        less than the other. The thin outline ring's *width* still uses
        the average, same as every other kind's outline pen — only the
        dot's own two-dimensional shape needs both axes to stay round.

        Same outline-then-fill layering as every other kind (a wider
        outline-colored disc under a narrower line-colored one) so
        outline_enabled/outline_color/outline_thickness behave
        identically here, including a disabled outline fully covering
        itself at zero width (see draw()'s own outline_width comment).
        """
        center = self._to_point(rect, point)
        stroke_scale = (scale_x + scale_y) / 2
        base_radius = OVERLAY_POINT_RADIUS * (line_width / OVERLAY_LINE_WIDTH)
        outline_extra = outline_width / stroke_scale
        rx, ry = base_radius / scale_x, base_radius / scale_y

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(outline_color))
        painter.drawEllipse(center, rx + outline_extra, ry + outline_extra)
        painter.setBrush(QBrush(line_color))
        painter.drawEllipse(center, rx, ry)

    def _draw_measurement_label(
        self,
        painter: QPainter,
        rect: QRect,
        index: int,
        measurement: Measurement,
        scale_x: float,
        scale_y: float,
        full_dims: tuple[int, int] | None,
    ) -> None:
        """
        Draw this measurement's tag — its title, plus the real-world
        length/diameter once both a reference frame size and a DPI are
        available (the whole point of DPI). A shape placed before DPI is
        set still gets a tag, just without a length suffix yet — see
        drawing_enabled and _length_suffix.

        The drawn box (and, while hovered, its delete glyph's own
        sub-rect) is recorded into _label_boxes/_delete_boxes so a later
        mouse event can hit-test against it — see _hit_test_tag.
        """
        dims = self._reference_dims(full_dims)
        text_and_anchor = self._measurement_label(measurement.kind, measurement.points, measurement.meta, dims)
        if text_and_anchor is None:
            return
        text, anchor = text_and_anchor
        meta = measurement.meta
        box, delete_box = self._draw_label(
            painter, rect, anchor, text, scale_x, scale_y,
            show_delete=index == self._hovered_index,
            bg_color=self._resolve_color(meta.tag_background_color, OVERLAY_LINE_COLOR),
            text_color=self._resolve_color(meta.tag_text_color, OVERLAY_OUTLINE_COLOR),
            description=meta.description if meta.always_show_description else None,
        )
        self._label_boxes[index] = box
        if delete_box is not None:
            self._delete_boxes[index] = delete_box

    @staticmethod
    def _resolve_color(hex_color: str, default: QColor) -> QColor:
        if not hex_color:
            return default
        color = QColor(hex_color)
        return color if color.isValid() else default

    def _reference_dims(self, full_dims: tuple[int, int] | None) -> tuple[int, int] | None:
        """
        For live view, *full_dims* (the zoom handler's current frame —
        the preview stream, sized for panning/zooming) is not what DPI
        was calibrated against, so ``_live_reference_dims`` (the still
        capture resolution) is used instead when available. A loaded
        image's *full_dims* is already its own true resolution, so it's
        used as-is.
        """
        return full_dims if self._loaded_active else (self._live_reference_dims or full_dims)

    def _draw_draft_label(
        self,
        painter: QPainter,
        rect: QRect,
        kind: str,
        points: tuple[tuple[float, float], ...],
        scale_x: float,
        scale_y: float,
        full_dims: tuple[int, int] | None,
    ) -> None:
        """
        Same tag rendering as _draw_measurement_label, for the
        in-progress draft. Uses the default-meta template rather than a
        placed measurement's own meta (the draft isn't one yet), and
        never hoverable/deletable — nothing is recorded into
        _label_boxes/_delete_boxes.
        """
        dims = self._reference_dims(full_dims)
        text_and_anchor = self._measurement_label(kind, points, self._default_meta, dims)
        if text_and_anchor is None:
            return
        text, anchor = text_and_anchor
        self._draw_label(painter, rect, anchor, text, scale_x, scale_y)

    def _measurement_label(
        self,
        kind: str,
        points: tuple[tuple[float, float], ...],
        meta: MeasurementMeta,
        full_dims: tuple[int, int] | None,
    ) -> tuple[str, tuple[float, float]] | None:
        """
        Text and anchor for *kind*'s tag: its title (only if one was
        actually set — see _resolve_meta) plus a length or diameter
        suffix once DPI and a reference frame size are both available
        (see _length_suffix). No tag is drawn at all once title and
        suffix both come up empty, same as the original DPI-gated
        behavior — a "Point" with no title never has anything to show,
        and a line/circle placed before DPI is set stays untagged until
        either a title or DPI arrives.
        """
        if kind == CALIBRATION_KIND:
            return None

        unit = meta.unit if meta.unit is not None else self._unit

        if kind in LINE_KINDS:
            suffix = self._length_suffix(points, full_dims, unit)
            text = self._compose_label_text(meta.title, suffix)
            return (text, self._midpoint(points)) if text else None

        if kind in _CIRCLE_KINDS:
            if full_dims is None:
                return None
            geometry = self._circle_geometry(kind, points, full_dims)
            if geometry is None:
                return None
            center, radius_px = geometry
            suffix = f"\u00d8 {format_length(radius_px * 2, self.dpi, unit)}" if self.dpi is not None else None
            text = self._compose_label_text(meta.title, suffix)
            return (text, center) if text else None

        if not points:
            return None
        text = self._compose_label_text(meta.title, None)
        return (text, points[0]) if text else None

    @staticmethod
    def _compose_label_text(title: str, suffix: str | None) -> str:
        if title and suffix:
            return f"{title} \u00b7 {suffix}"
        return title or suffix or ""

    def _length_suffix(
        self,
        points: tuple[tuple[float, float], ...],
        full_dims: tuple[int, int] | None,
        unit: MeasurementUnit,
    ) -> str | None:
        if full_dims is None or self.dpi is None:
            return None
        length_px = self._polyline_length_px(points, full_dims)
        if length_px <= 0:
            return None
        return format_length(length_px, self.dpi, unit)

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

    @staticmethod
    def _right_rounded_path(rect: QRectF, radius: float) -> QPainterPath:
        """Rect with only its right two corners rounded (matching the tag's own rounding where the delete strip meets the tag's edge) and square left corners (where it meets the tag's text, not an outer edge)."""
        r = min(radius, rect.width() / 2, rect.height() / 2)
        path = QPainterPath()
        path.moveTo(rect.left(), rect.top())
        path.lineTo(rect.right() - r, rect.top())
        path.arcTo(rect.right() - 2 * r, rect.top(), 2 * r, 2 * r, 90, -90)
        path.lineTo(rect.right(), rect.bottom() - r)
        path.arcTo(rect.right() - 2 * r, rect.bottom() - 2 * r, 2 * r, 2 * r, 0, -90)
        path.lineTo(rect.left(), rect.bottom())
        path.closeSubpath()
        return path

    def _draw_label(
        self,
        painter: QPainter,
        rect: QRect,
        anchor: tuple[float, float],
        text: str,
        scale_x: float,
        scale_y: float,
        *,
        show_delete: bool = False,
        bg_color: QColor = OVERLAY_LINE_COLOR,
        text_color: QColor = OVERLAY_OUTLINE_COLOR,
        description: str | None = None,
    ) -> tuple[QRectF, QRectF | None]:
        """
        Rounded-rect background (bg_color, defaulting to white) with a
        thin outline and text (text_color, defaulting to near-black) on
        top — reads clearly regardless of what's under it, unlike the
        outlined-stroke technique used for lines and endpoints (a thin
        outline around small text gets muddy at typical font sizes).
        *description*, if given, is drawn as a smaller second line
        rather than a separate box, growing the same box downward.

        Drawn in a local frame — translated to the anchor, then scaled
        by 1/scale_x, 1/scale_y — that cancels out the ambient paint
        transform's own zoom scale entirely, rather than sizing
        everything in "rect" units divided by a scale factor the way
        the rest of the overlay does. That older approach works for a
        stroked line or a hand-built ellipse, whose points can be placed
        anywhere and so can be pre-distorted to cancel a non-uniform
        transform, but not for font glyphs, which distort along with
        whatever non-uniform scale the transform applies no matter how
        the surrounding box is sized — at zoom levels where the crop's
        aspect ratio hasn't yet caught up to the widget's (see
        ZoomPreviewOverlay.current_scale_xy), scale_x and scale_y
        differ, and text sized off their average came out squished
        along whichever axis was scaled less than the other. Inside this
        local frame, every size below is a plain, undistorted screen
        pixel value — no more dividing by a scale factor.

        Returns the drawn box, and — only while *show_delete* — the
        delete glyph's own sub-rect within it, both converted back into
        the same pre-paint-transform "rect" frame *anchor* was given in
        (see _local_rect_to_rect_space) so callers and _hit_test_tag
        keep working with the coordinate system they already expect.
        The delete strip spans the box's full height and sits flush
        against its right edge, rather than floating as a small chip, so
        it reads as the whole right end of the tag rather than a
        separate control on top of it. ``_draw_measurement_label``
        records the box/delete rects per measurement index so a later
        mouse event can hit-test against them (see ``_hit_test_tag``);
        calibration's own use of this method never passes *show_delete*,
        so its label isn't hoverable or deletable this way.
        """
        point = self._to_point(rect, anchor)

        font = QFont(painter.font())
        font.setPixelSize(max(1, round(OVERLAY_LABEL_FONT_SIZE)))
        metrics = QFontMetricsF(font)
        pad_x = OVERLAY_LABEL_PADDING_X
        pad_y = OVERLAY_LABEL_PADDING_Y
        delete_width = (OVERLAY_DELETE_SIZE + OVERLAY_DELETE_MARGIN) if show_delete else 0.0

        text_line_height = metrics.height() + pad_y * 2
        text_w = metrics.horizontalAdvance(text)
        box_w = text_w + pad_x * 2 + delete_width
        text_area_w = text_w  # description wraps within the title's own width rather than widening the box

        desc_font: QFont | None = None
        desc_block_height = 0.0
        desc_wrap_rect = QRectF()
        if description:
            desc_font = QFont(font)
            desc_font.setPixelSize(max(1, round(OVERLAY_LABEL_FONT_SIZE - 2)))
            desc_metrics = QFontMetricsF(desc_font)
            desc_wrap_rect = desc_metrics.boundingRect(
                QRectF(0, 0, max(text_area_w, 1.0), 10_000), Qt.TextFlag.TextWordWrap, description
            )
            # Extra padding above and below (not just the usual pad_y)
            # so the description reads as its own block under the
            # divider rather than crowding it.
            desc_block_height = desc_wrap_rect.height() + pad_y * 3

        box_h = text_line_height + desc_block_height

        local_box = QRectF(-box_w / 2, -OVERLAY_LABEL_OFFSET - box_h, box_w, box_h)

        painter.save()
        painter.translate(point)
        if scale_x > 0 and scale_y > 0:
            painter.scale(1.0 / scale_x, 1.0 / scale_y)

        painter.setFont(font)
        painter.setPen(QPen(OVERLAY_TAG_HOVER_COLOR if show_delete else OVERLAY_OUTLINE_COLOR, OVERLAY_OUTLINE_WIDTH))
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(local_box, OVERLAY_LABEL_CORNER_RADIUS, OVERLAY_LABEL_CORNER_RADIUS)

        content_w = local_box.width() - delete_width
        painter.setPen(QPen(text_color))
        painter.drawText(QRectF(local_box.x(), local_box.y(), content_w, text_line_height), Qt.AlignmentFlag.AlignCenter, text)

        if description:
            divider_y = local_box.top() + text_line_height
            divider_pen = QPen(text_color)
            divider_pen.setWidthF(1.0)
            painter.setPen(divider_pen)
            painter.drawLine(QPointF(local_box.left(), divider_y), QPointF(local_box.right(), divider_y))

            muted = QColor(text_color)
            muted.setAlpha(180)
            painter.setFont(desc_font)
            painter.setPen(QPen(muted))
            desc_box = QRectF(
                local_box.x() + (content_w - desc_wrap_rect.width()) / 2,
                divider_y + pad_y * 1.5,
                desc_wrap_rect.width(),
                desc_wrap_rect.height(),
            )
            painter.drawText(desc_box, Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap, description)
            painter.setFont(font)

        local_delete_box: QRectF | None = None
        if show_delete:
            local_delete_box = QRectF(local_box.right() - delete_width, local_box.top(), delete_width, local_box.height())
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(OVERLAY_DELETE_BG_COLOR))
            painter.drawPath(self._right_rounded_path(local_delete_box, OVERLAY_LABEL_CORNER_RADIUS))

            glyph_size = min(OVERLAY_DELETE_SIZE, local_delete_box.width(), local_delete_box.height()) * 0.7
            glyph_box = QRectF(
                local_delete_box.center().x() - glyph_size / 2,
                local_delete_box.center().y() - glyph_size / 2,
                glyph_size,
                glyph_size,
            )
            glyph_inset = glyph_size * 0.22
            glyph_pen = QPen(OVERLAY_DELETE_GLYPH_COLOR)
            glyph_pen.setWidthF(max(1.0, glyph_size / 5))
            glyph_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(glyph_pen)
            painter.drawLine(
                QPointF(glyph_box.left() + glyph_inset, glyph_box.top() + glyph_inset),
                QPointF(glyph_box.right() - glyph_inset, glyph_box.bottom() - glyph_inset),
            )
            painter.drawLine(
                QPointF(glyph_box.right() - glyph_inset, glyph_box.top() + glyph_inset),
                QPointF(glyph_box.left() + glyph_inset, glyph_box.bottom() - glyph_inset),
            )

        painter.restore()

        box = self._local_rect_to_rect_space(local_box, point, scale_x, scale_y)
        delete_box = (
            self._local_rect_to_rect_space(local_delete_box, point, scale_x, scale_y)
            if local_delete_box is not None
            else None
        )
        return box, delete_box

    @staticmethod
    def _local_rect_to_rect_space(local: QRectF, point: QPointF, scale_x: float, scale_y: float) -> QRectF:
        """The inverse of _draw_label's painter.scale(1/scale_x, 1/scale_y) — converts a rect from its local, undistorted-pixel frame (origin at *point*) back into the "rect" coordinate space _label_boxes/_delete_boxes (and everything that reads them, e.g. _hit_test_tag) are built to expect."""
        sx = scale_x if scale_x > 0 else 1.0
        sy = scale_y if scale_y > 0 else 1.0
        return QRectF(
            point.x() + local.x() / sx,
            point.y() + local.y() / sy,
            local.width() / sx,
            local.height() / sy,
        )

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
                self._draw_draft_label(painter, rect, kind, label_points, scale_x, scale_y, full_dims)
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

            self._draw_measurement(painter, rect, kind, display_points, stroke_scale, scale_x, scale_y, full_dims, dashed=True)

            if kind in _CIRCLE_KINDS and len(preview_points) >= 2:
                required = _REQUIRED_POINTS.get(kind, len(preview_points))
                if len(preview_points) < required:
                    # Not enough points for a circle yet — a straight
                    # guide between what's placed so far is still useful
                    # feedback (e.g. two of three "3 Point Circle" clicks).
                    self._draw_polyline(painter, rect, preview_points, stroke_scale, dashed=True)

            if kind == CALIBRATION_KIND:
                if len(display_points) >= 2:
                    self._draw_label(painter, rect, self._midpoint(tuple(display_points)), "Calibration", scale_x, scale_y)
            elif len(display_points) >= 2:
                self._draw_draft_label(painter, rect, kind, tuple(display_points), scale_x, scale_y, full_dims)

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
        line_color: QColor = OVERLAY_LINE_COLOR,
        line_width: float = OVERLAY_LINE_WIDTH,
        outline_color: QColor = OVERLAY_OUTLINE_COLOR,
        outline_width: float = OVERLAY_OUTLINE_WIDTH,
        dash_style: str = "solid",
        start_cap: str = "curved",
        end_cap: str = "curved",
    ) -> None:
        last = len(points) - 2
        for i in range(len(points) - 1):
            self._draw_stroke(
                painter, rect, points[i], points[i + 1], stroke_scale, dashed=dashed,
                line_color=line_color, line_width=line_width, outline_color=outline_color, outline_width=outline_width,
                dash_style=dash_style,
                start_cap=start_cap if i == 0 else "curved",
                end_cap=end_cap if i == last else "curved",
            )

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
        line_color: QColor = OVERLAY_LINE_COLOR,
        line_width: float = OVERLAY_LINE_WIDTH,
        outline_color: QColor = OVERLAY_OUTLINE_COLOR,
        outline_width: float = OVERLAY_OUTLINE_WIDTH,
        dash_style: str = "solid",
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

        total_outline_width = line_width + outline_width * 2
        pattern = None if dashed else resolve_dash_pattern(dash_style, line_width)
        outline = QPen(outline_color)
        outline.setWidthF(total_outline_width / stroke_scale)
        self._apply_dash(outline, total_outline_width, dashed, pattern)
        painter.setPen(outline)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center_point, rx, ry)

        fill = QPen(line_color)
        fill.setWidthF(line_width / stroke_scale)
        self._apply_dash(fill, line_width, dashed, pattern)
        painter.setPen(fill)
        painter.drawEllipse(center_point, rx, ry)

        if kind in ("Radius Circle", "Diameter") and dashed:
            # Matches the tile icon while placing — a straight line from
            # center to edge (Radius Circle) or all the way across
            # (Diameter) — but once finalized the circle's own outline
            # already shows the same thing, so the guide line is dropped;
            # only the point markers (drawn separately, per point, in
            # draw()) stick around, including Radius Circle's center.
            self._draw_stroke(
                painter, rect, points[0], points[1], stroke_scale, dashed=dashed,
                line_color=line_color, line_width=line_width, outline_color=outline_color, outline_width=outline_width,
            )

    def _draw_stroke(
        self,
        painter: QPainter,
        rect: QRect,
        start: tuple[float, float],
        end: tuple[float, float],
        stroke_scale: float,
        *,
        dashed: bool,
        line_color: QColor = OVERLAY_LINE_COLOR,
        line_width: float = OVERLAY_LINE_WIDTH,
        outline_color: QColor = OVERLAY_OUTLINE_COLOR,
        outline_width: float = OVERLAY_OUTLINE_WIDTH,
        dash_style: str = "solid",
        start_cap: str = "curved",
        end_cap: str = "curved",
    ) -> None:
        """
        A stroke in *line_color* over a slightly wider *outline_color*
        one gives the line a thin outline that reads against both light
        and dark backgrounds, rather than a plain fill picked to
        contrast one or the other. Widths are divided by *stroke_scale*
        — the average of ``ZoomPreviewOverlay``'s current x/y
        paint-transform scale — so they stay a constant size on screen
        instead of growing with zoom. *dashed* draws the in-progress
        preview in this same style, solidifying only once the
        measurement is actually placed — while it's still a preview,
        *dash_style*/*start_cap*/*end_cap* are skipped entirely so the
        placement animation always reads the same regardless of what
        the eventual measurement will look like.

        A "square"/"arrow"/"curved" *start_cap*/*end_cap* shortens the
        drawn body (for "arrow" only — see _cap_reach) and unions the
        cap's own filled shape onto the body's before either is
        painted — one shape means one outline, so a cap's border traces
        its own silhouette into the shaft rather than a
        separately-stroked cap drawing a second border straight across
        the shaft where the two meet. "curved"'s shape is just a round
        nub the size its old RoundCap end would have drawn — the shaft
        itself is always flat-ended (see _stroke_path) so it never
        bulges past a true endpoint on its own. "arrow_open" is the one
        style with no shape to union: its shaft runs the full length to
        the tip, flat-ended and undecorated, and the open barbs are
        drawn over it afterward so the point stays sharp rather than
        rounded.
        """
        p1 = self._to_point(rect, start)
        p2 = self._to_point(rect, end)
        lw = line_width / stroke_scale
        ow = outline_width / stroke_scale
        total_outline_width = line_width + outline_width * 2
        pattern = None if dashed else resolve_dash_pattern(dash_style, line_width)

        if dashed:
            outline = QPen(outline_color)
            outline.setWidthF(total_outline_width / stroke_scale)
            outline.setCapStyle(Qt.PenCapStyle.RoundCap)
            self._apply_dash(outline, total_outline_width, dashed, pattern)
            painter.setPen(outline)
            painter.drawLine(p1, p2)

            fill = QPen(line_color)
            fill.setWidthF(lw)
            fill.setCapStyle(Qt.PenCapStyle.RoundCap)
            self._apply_dash(fill, line_width, dashed, pattern)
            painter.setPen(fill)
            painter.drawLine(p1, p2)
            return

        body_p1 = self._point_along(p1, p2, self._cap_reach(start_cap, lw))
        body_p2 = self._point_along(p2, p1, self._cap_reach(end_cap, lw))

        # Qt dash patterns are in multiples of the stroking width, so
        # each pass normalizes pattern's screen-pixel targets against its
        # own (unscaled) width — matching _apply_dash's math — to keep
        # the outline and fill dashes aligned despite their different
        # widths.
        outline_pattern = [value / total_outline_width for value in pattern] if pattern else None
        fill_pattern = [value / line_width for value in pattern] if pattern else None
        outline_path = self._stroke_path(body_p1, body_p2, total_outline_width / stroke_scale, outline_pattern)
        fill_path = self._stroke_path(body_p1, body_p2, lw, fill_pattern)
        for origin, tip, cap in ((p2, p1, start_cap), (p1, p2, end_cap)):
            cap_outline, cap_fill = self._cap_shapes(origin, tip, cap, lw, ow)
            if cap_outline is not None:
                outline_path = outline_path.united(cap_outline)
                fill_path = fill_path.united(cap_fill)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(outline_color))
        painter.drawPath(outline_path)
        painter.setBrush(QBrush(line_color))
        painter.drawPath(fill_path)

    @staticmethod
    def _point_along(origin: QPointF, toward: QPointF, distance: float) -> QPointF:
        """*origin* moved *distance* px toward *toward* — used to shorten a stroke's drawn body to make room for a cap decoration at *origin*. Never overshoots past *toward* itself, so a segment shorter than the requested reach just collapses to a point rather than reversing direction."""
        if distance <= 0:
            return origin
        dx, dy = toward.x() - origin.x(), toward.y() - origin.y()
        length = math.hypot(dx, dy)
        if length <= 0:
            return origin
        t = min(distance, length) / length
        return QPointF(origin.x() + dx * t, origin.y() + dy * t)

    @classmethod
    def _cap_reach(cls, cap: str, lw: float) -> float:
        """How far the "arrow" cap's own shape extends back from the true endpoint — the body is shortened by this much so its flat end sits at the arrowhead's base instead of poking past the tip once the two are unioned. Every other style unions its shape directly onto the shaft's own flat, unshortened end: "square" and "curved" are no wider than the shaft (their disc/rect never reaches past the true endpoint), and "arrow_open" has no shape to make room for at all."""
        if cap == "arrow":
            return arrow_dims(lw)[0]
        return 0.0

    @staticmethod
    def _stroke_path(p1: QPointF, p2: QPointF, width: float, dash_pattern: list[float] | None = None) -> QPainterPath:
        """A filled flat-ended band from *p1* to *p2*, *width* wide (or that band split into dashes, given a *dash_pattern* already normalized to multiples of *width*) — the shaft as a shape rather than a stroked QPen line, so it can be unioned with a cap's own shape into one seamless path. Flat rather than round-ended so it never bulges past a true endpoint on its own; "curved" gets its round look from an explicit disc unioned in by _cap_shapes instead, the same way "square"/"arrow" get theirs."""
        line = QPainterPath()
        line.moveTo(p1)
        line.lineTo(p2)
        stroker = QPainterPathStroker()
        stroker.setWidth(width)
        stroker.setCapStyle(Qt.PenCapStyle.FlatCap)
        if dash_pattern:
            stroker.setDashPattern(dash_pattern)
        return stroker.createStroke(line)

    @classmethod
    def _cap_shapes(
        cls, origin: QPointF, tip: QPointF, cap: str, lw: float, ow: float
    ) -> tuple[QPainterPath | None, QPainterPath | None]:
        """
        (outline shape, fill shape) for *cap* at *tip*, pointing away
        from *origin*, already placed in the segment's coordinate frame.
        Every recognized style is a shape unioned onto the flat shaft
        here, "arrow_open"'s open barbs included — building them as
        shapes and unioning them in, rather than stroking them as
        separate paint operations after the shaft is already painted,
        is what keeps their own outline from cutting across the shaft's
        fill where the two meet. "curved" is likewise just a round nub
        the size a RoundCap end would have drawn, moved here so it,
        too, never overlaps a neighboring cap's own outline.

        "arrow" and "arrow_open"'s base shapes (arrow_head_path,
        open_arrow_barbs_path) come from UI.widgets.measurements.lines
        rather than being built here — an arrowhead's geometry is a
        line-measurement concern, this method's job is just placing
        and outlining whichever cap shape it's handed.

        Each shape is built at the size matching its own pass: the
        outline shape is the fill shape inflated outward by *ow* so the
        two, drawn outline-then-fill like the shaft itself, leave a
        uniform border. "square"'s and "curved"'s inflated shapes
        already are that outward shape directly. "arrow"'s and
        "arrow_open"'s inflated shapes come from unioning the plain
        shape with a stroke of its own outline — stroking alone would
        carve *ow* back in from each edge as well as out, but since only
        the outward half ever shows past the narrower fill shape drawn
        on top, the inward half is harmless to include.
        """
        dx, dy = tip.x() - origin.x(), tip.y() - origin.y()
        length = math.hypot(dx, dy)
        if length <= 0:
            return None, None
        angle = math.degrees(math.atan2(dy, dx))

        if cap == "square":
            half = lw / 2
            fill_shape = QPainterPath()
            fill_shape.addRect(QRectF(0, -half, half, 2 * half))
            outline_shape = QPainterPath()
            outline_shape.addRect(QRectF(0, -(half + ow), half + ow, 2 * (half + ow)))
        elif cap == "arrow":
            fill_shape = arrow_head_path(lw)
            outline_shape = cls._inflate(fill_shape, ow, Qt.PenJoinStyle.RoundJoin)
        elif cap == "arrow_open":
            barbs = open_arrow_barbs_path(lw)

            fill_stroker = QPainterPathStroker()
            fill_stroker.setWidth(lw)
            fill_stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
            fill_stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            fill_shape = fill_stroker.createStroke(barbs)

            outline_stroker = QPainterPathStroker()
            outline_stroker.setWidth(lw + ow * 2)
            outline_stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
            outline_stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            outline_shape = outline_stroker.createStroke(barbs)
        else:  # "curved" (and any unrecognized style, as a safe default)
            fill_shape = QPainterPath()
            fill_shape.addEllipse(QPointF(0, 0), lw / 2, lw / 2)
            outline_shape = QPainterPath()
            outline_shape.addEllipse(QPointF(0, 0), lw / 2 + ow, lw / 2 + ow)

        transform = QTransform()
        transform.translate(tip.x(), tip.y())
        transform.rotate(angle)
        return transform.map(outline_shape), transform.map(fill_shape)

    @staticmethod
    def _inflate(shape: QPainterPath, amount: float, join_style: Qt.PenJoinStyle) -> QPainterPath:
        """*shape* grown outward by *amount* on every edge — used to derive a cap's outline-pass shape from its fill-pass shape. Stroking *shape*'s own boundary and unioning that stroke back in expands it outward (the inward half of the stroke lands back inside *shape*, contributing nothing new); a no-op when *amount* is 0."""
        if amount <= 0:
            return QPainterPath(shape)
        stroker = QPainterPathStroker()
        stroker.setWidth(amount * 2)
        stroker.setJoinStyle(join_style)
        return shape.united(stroker.createStroke(shape))

    @staticmethod
    def _apply_dash(pen: QPen, stroke_width: float, dashed: bool, pattern: list[float] | None = None) -> None:
        """
        Qt's dash pattern is specified in multiples of the pen's own
        width, so the same pattern values on the wider outline pen and
        the thinner fill pen drawn over it produce different absolute
        dash lengths — misaligning the two. Dividing *pattern*'s values
        (screen-pixel targets — see resolve_dash_pattern) by each pen's
        own *stroke_width* first cancels that out, so both strokes dash
        at the same absolute, on-screen length regardless of their
        width.

        *dashed* (the in-progress placement animation) always wins over
        *pattern* (a finalized measurement's own chosen style) — the two
        are never meaningful at once, since a dash style only applies
        once a measurement is actually placed.
        """
        if dashed:
            pen.setStyle(Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([OVERLAY_DASH_LENGTH / stroke_width, OVERLAY_DASH_GAP / stroke_width])
            return
        if not pattern:
            pen.setStyle(Qt.PenStyle.SolidLine)
            return
        pen.setStyle(Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([value / stroke_width for value in pattern])

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

    def set_default_meta(self, meta: MeasurementMeta) -> None:
        """Template a newly placed measurement's meta is built from — see MeasurementsWidget's "Customize Measurements" section."""
        self._overlay.set_default_meta(meta)

    def measurement_meta(self, index: int) -> MeasurementMeta | None:
        return self._overlay.measurement_meta(index)

    def measurement_kind(self, index: int) -> str | None:
        return self._overlay.measurement_kind(index)

    def set_measurement_meta(self, index: int, meta: MeasurementMeta) -> bool:
        applied = self._overlay.set_measurement_meta(index, meta)
        if applied:
            self._repaint()
        return applied

    def remove_measurement(self, index: int) -> bool:
        removed = self._overlay.remove_measurement(index)
        if removed:
            self._repaint()
        return removed

    @property
    def hovered_index(self) -> int | None:
        return self._overlay.hovered_index

    @property
    def hover_delete(self) -> bool:
        return self._overlay.hover_delete

    def update_hover(self, pos: QPoint, rect: QRect, widget_rect: QRect) -> bool:
        return self._overlay.update_hover(pos, rect, widget_rect)

    def clear_hover(self) -> bool:
        return self._overlay.clear_hover()

    def label_screen_rect(self, index: int, rect: QRect, widget_rect: QRect) -> QRectF | None:
        return self._overlay.label_screen_rect(index, rect, widget_rect)