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

from UI.widgets.measurements.lines import (
    CALIBRATION_KIND, arrow_dims, arrow_head_path, bracket_path, circle_head_path, circle_head_radius,
    diamond_head_path, open_arrow_barbs_path,
)
from UI.widgets.measurements.measurement_io import (
    DeserializeResult,
    deserialize_measurements,
    load_measurements_from_file,
    save_measurements_to_file,
    serialize_measurements,
)
from UI.widgets.measurements.measurement_kind import DEFAULT_REGISTRY
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
from UI.widgets.measurements.units import MeasurementUnit, format_area, format_length
from UI.widgets.preview_overlay.loaded_image_overlay import LoadedImageOverlay
from UI.widgets.preview_overlay.overlay_base import Overlay
from UI.widgets.preview_overlay.coordinate_space import CoordinateSpace


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
    "round_dot": [0.1, 2.5],
    "dash_dot": [4, 2, 1, 2],
    "dash_dot_dot": [4, 2, 1, 2, 1, 2],
    "long_dash": [9, 4],
}
MEASUREMENT_DASH_STYLES = tuple(_DASH_PATTERNS.keys())
MEASUREMENT_DASH_PATTERNS = _DASH_PATTERNS

# Dash styles whose dashes are drawn with round caps, turning each short
# on-segment into a circular dot rather than a flat-ended tick.
ROUND_CAP_DASH_STYLES = frozenset({"round_dot"})

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
    "round_dot": [0.1, 2.6],
    "dash_dot": [4.0, 2.5, 1.2, 2.5],
    "dash_dot_dot": [4.0, 2.0, 1.2, 2.0, 1.2, 2.0],
    "long_dash": [7.0, 3.5],
}
_DASH_REFERENCE_MIN = 3.0  # keeps a thin line's dashes/gaps from shrinking below a legible size

# Screen-pixel/degree tolerance below which a value is treated as zero —
# used only by _draw_angle_indicator's own screen-space geometry (a
# coincident near-point/anchor, or a near-zero sweep), distinct from
# MeasurementKind's fraction/pixel-space _DEGENERATE_EPSILON.
_INDICATOR_EPSILON = 1e-6

# Minimum wrap width for a tag's description block, so a short title (or
# none) doesn't force it down to one word per line — see
# MeasurementOverlay._draw_label.
_DESC_MIN_WRAP_WIDTH = 150.0


def resolve_dash_pattern(dash_style: str, line_width: float) -> list[float] | None:
    """Screen-pixel dash pattern for *dash_style* at *line_width*, or None for solid/unrecognized — see _DASH_MULTIPLIERS."""
    multipliers = _DASH_MULTIPLIERS.get(dash_style)
    if not multipliers:
        return None
    reference = max(line_width, _DASH_REFERENCE_MIN)
    return [value * reference for value in multipliers]


class Measurement(NamedTuple):
    kind: str
    points: tuple[tuple[float, float], ...]
    meta: MeasurementMeta = DEFAULT_META


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
    active kind needs (see ``MeasurementKind.required_points`` in
    measurement_kind.py) — a single click for "Point", two for a line or
    2-point circle, three for "3 Point Circle". ``place_point`` handles
    every click; it starts a new draft on the first, finalizes it once
    enough points have arrived, and finalizes immediately for any kind
    that only ever needs one (e.g. "Point").
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
        self._zoom_handler: CoordinateSpace | None = None
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

        # Tag dragging — moving a measurement's tag away from its anchor
        # by accumulating a fraction-space offset into its meta (feature
        # 4). Kept apart from endpoint dragging: a tag drag moves only the
        # label, never the measurement's geometry.
        self._tag_drag_index: int | None = None
        self._tag_drag_start_fraction: tuple[float, float] | None = None
        self._tag_drag_start_offset: tuple[float, float] = (0.0, 0.0)

        # Same, for a dragged secondary (extra_measures) tag — keyed by
        # (measurement index, extra index).
        self._extra_drag_key: tuple[int, int] | None = None
        self._extra_drag_start_fraction: tuple[float, float] | None = None
        self._extra_drag_start_offset: tuple[float, float] = (0.0, 0.0)

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
        # Secondary (extra_measures) tag boxes, keyed by (measurement
        # index, extra index), so hovering one and pressing Delete can
        # dismiss just that tag — see delete_hovered_extra.
        self._extra_label_boxes: dict[tuple[int, int], QRectF] = {}
        self._hovered_index: int | None = None
        self._hover_delete: bool = False
        self._hovered_extra: tuple[int, int] | None = None

        # Which measurement's anchor points are drawn — only the one
        # the cursor is currently near (see update_proximity), plus
        # whichever one has an endpoint actively being dragged, so
        # placed points don't otherwise clutter the view.
        self._near_index: int | None = None

    def set_zoom_handler(self, handler: CoordinateSpace | None) -> None:
        """Register the coordinate space used to convert clicks into frame-fraction coordinates — see CoordinateSpace for the contract; ZoomPreviewOverlay is the live preview's implementation."""
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
        return self._active_type in DEFAULT_REGISTRY

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
            entry = DEFAULT_REGISTRY.get(self._active_type)
            if entry is not None and entry.required_points == 1:
                resolved = entry.resolve([point])
                if resolved is None:
                    return False
                self.measurements.append(Measurement(self._active_type, resolved, self._resolve_meta(self._active_type)))
                return True
            self._draft_type = self._active_type
            self._draft_points = [point]
            self._draft_preview = None
            return True

        self._draft_points.append(point)
        self._draft_preview = None

        entry = DEFAULT_REGISTRY.get(self._draft_type)
        required = entry.required_points if entry is not None else None
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
            self.measurements.append(Measurement(kind, self._apply_snap(kind, resolved), self._resolve_meta(kind)))
        return True

    def _apply_snap(
        self, kind: str, points: tuple[tuple[float, float], ...]
    ) -> tuple[tuple[float, float], ...]:
        """Snap a kind's derived control points onto its drawn geometry — see MeasurementKind.snap_points. A no-op for kinds without one."""
        entry = DEFAULT_REGISTRY.get(kind)
        if entry is None or entry.snap_points is None:
            return points
        full_dims = self._zoom_handler.current_frame_dims() if self._zoom_handler is not None else None
        return entry.snap_points(tuple(points), full_dims)

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
        entry = DEFAULT_REGISTRY.get(self._draft_type)
        if (
            self._draft_points is not None
            and entry is not None
            and entry.required_points is None
            and self._draft_type != CALIBRATION_KIND
            and len(self._draft_points) >= entry.min_points
        ):
            resolved = entry.resolve(list(self._draft_points))
            if resolved is not None:
                self.measurements.append(
                    Measurement(self._draft_type, resolved, self._resolve_meta(self._draft_type))
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

    @property
    def default_meta(self) -> MeasurementMeta:
        """The template new placements pick up — also the target a measurement's "Reset Style" resets to."""
        return self._default_meta

    def _resolve_meta(self, kind: str) -> MeasurementMeta:
        """
        No title by default — only number one in once the sidebar's
        default title template is actually set (e.g. "Wingspan 1",
        "Wingspan 2", ...), so a measurement placed with nothing
        configured stays untitled rather than falling back to its own
        kind name. A kind with its own ``meta_preset`` (see "Arrow"/
        "Bracket" in measurement_kind.py) always gets those fields
        forced, on top of whatever the panel's own defaults are.
        """
        base = self._default_meta
        entry = DEFAULT_REGISTRY.get(kind)
        if entry is not None and entry.meta_preset:
            base = base._replace(**entry.meta_preset)
        if not base.title:
            return base
        self._kind_counts[kind] = self._kind_counts.get(kind, 0) + 1
        return base._replace(title=f"{base.title} {self._kind_counts[kind]}")

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

    @property
    def hovered_extra(self) -> tuple[int, int] | None:
        """Which secondary (extra_measures) tag the cursor is over, as (measurement index, extra index), or None."""
        return self._hovered_extra

    def clear_hover(self) -> bool:
        changed = self._hovered_index is not None or self._hovered_extra is not None
        self._hovered_index = None
        self._hover_delete = False
        self._hovered_extra = None
        return changed

    def update_hover(self, pos: QPoint, rect: QRect, widget_rect: QRect) -> bool:
        """Recompute which tag (if any) *pos* is over, and whether it's over that tag's delete glyph. Returns True if anything changed, so the caller knows whether to repaint."""
        index, over_delete = self._hit_test_tag(pos, rect, widget_rect)
        # A primary tag wins over a secondary one where they overlap.
        extra = self._hit_test_extra(pos, rect, widget_rect) if index is None else None
        changed = (
            index != self._hovered_index
            or over_delete != self._hover_delete
            or extra != self._hovered_extra
        )
        self._hovered_index, self._hover_delete, self._hovered_extra = index, over_delete, extra
        return changed

    def _hit_test_extra(self, pos: QPoint, rect: QRect, widget_rect: QRect) -> tuple[int, int] | None:
        if self._zoom_handler is None:
            return None
        for key, box in self._extra_label_boxes.items():
            if self._screen_rect(box, rect, widget_rect).contains(QPointF(pos)):
                return key
        return None

    def delete_hovered_extra(self) -> bool:
        """Dismiss the hovered secondary tag by recording its index in the measurement's meta.hidden_extra. Returns True if one was hovered and dismissed."""
        if self._hovered_extra is None:
            return False
        m_index, extra_index = self._hovered_extra
        meta = self.measurement_meta(m_index)
        if meta is None:
            return False
        if extra_index not in meta.hidden_extra:
            self.set_measurement_meta(
                m_index, meta._replace(hidden_extra=tuple(sorted(set(meta.hidden_extra) | {extra_index})))
            )
        self._hovered_extra = None
        return True

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
            entry = DEFAULT_REGISTRY.get(measurement.kind)
            if entry is None:
                continue
            screen_points = [self._screen_point(self._to_point(rect, p), rect, widget_rect) for p in measurement.points]
            if any(self._distance(cursor, sp) <= OVERLAY_ENDPOINT_HIT_RADIUS for sp in screen_points):
                return index
            if entry.category == "line":
                for a, b in zip(screen_points, screen_points[1:]):
                    if self._distance_to_segment(cursor, a, b) <= OVERLAY_ENDPOINT_HIT_RADIUS:
                        return index
            elif entry.category == "circle" and full_dims is not None:
                edge_points = self._circle_edge_screen_points(
                    measurement.kind, measurement.points, rect, widget_rect, full_dims
                )
                for a, b in zip(edge_points, edge_points[1:] + edge_points[:1]):
                    if self._distance_to_segment(cursor, a, b) <= OVERLAY_ENDPOINT_HIT_RADIUS:
                        return index
            elif entry.category == "ellipse" and full_dims is not None:
                edge_points = self._ellipse_edge_screen_points(
                    measurement.kind, measurement.points, rect, widget_rect, full_dims
                )
                for a, b in zip(edge_points, edge_points[1:] + edge_points[:1]):
                    if self._distance_to_segment(cursor, a, b) <= OVERLAY_ENDPOINT_HIT_RADIUS:
                        return index
            elif entry.category in ("angle", "line_pair") and entry.segment_pairs is not None:
                segments = list(entry.segment_pairs(measurement.points, full_dims))
                if entry.connector_segments is not None:
                    segments += entry.connector_segments(measurement.points, full_dims)
                for pa, pb in segments:
                    a = self._screen_point(self._to_point(rect, pa), rect, widget_rect)
                    b = self._screen_point(self._to_point(rect, pb), rect, widget_rect)
                    if self._distance_to_segment(cursor, a, b) <= OVERLAY_ENDPOINT_HIT_RADIUS:
                        return index
            elif entry.category == "arc" and full_dims is not None:
                arc_points = self._arc_sample_points(measurement.kind, measurement.points, full_dims)
                arc_screen = [self._screen_point(self._to_point(rect, p), rect, widget_rect) for p in arc_points]
                for a, b in zip(arc_screen, arc_screen[1:]):
                    if self._distance_to_segment(cursor, a, b) <= OVERLAY_ENDPOINT_HIT_RADIUS:
                        return index
            elif entry.category == "curve" and entry.curve_points is not None:
                curve = entry.curve_points(measurement.points)
                if curve is not None:
                    curve_screen = [self._screen_point(self._to_point(rect, p), rect, widget_rect) for p in curve]
                    for a, b in zip(curve_screen, curve_screen[1:]):
                        if self._distance_to_segment(cursor, a, b) <= OVERLAY_ENDPOINT_HIT_RADIUS:
                            return index
            elif entry.category == "polygon" and entry.polygon_points is not None and full_dims is not None:
                verts = entry.polygon_points(measurement.points, full_dims)
                if verts is not None and len(verts) >= 2:
                    screen = [self._screen_point(self._to_point(rect, p), rect, widget_rect) for p in verts]
                    for a, b in zip(screen, screen[1:] + screen[:1]):
                        if self._distance_to_segment(cursor, a, b) <= OVERLAY_ENDPOINT_HIT_RADIUS:
                            return index
            elif entry.category == "annulus" and entry.annulus_geometry is not None and full_dims is not None:
                geometry = entry.annulus_geometry(measurement.points, full_dims)
                if geometry is not None and self._near_annulus(cursor, geometry, rect, widget_rect, full_dims):
                    return index
            elif entry.category == "two_circle" and entry.two_circle_geometry is not None and full_dims is not None:
                geometry = entry.two_circle_geometry(measurement.points, full_dims)
                if geometry is not None:
                    c1, r1, c2, r2 = geometry
                    for center, radius in ((c1, r1), (c2, r2)):
                        if self._near_circle_edge(cursor, center, radius, rect, widget_rect, full_dims):
                            return index
        return None

    def _near_circle_edge(
        self, cursor: QPointF, center: tuple[float, float], radius_px: float,
        rect: QRect, widget_rect: QRect, full_dims: tuple[int, int],
    ) -> bool:
        full_w, full_h = full_dims
        edge = []
        for i in range(self._CIRCLE_EDGE_SAMPLES):
            angle = 2 * math.pi * i / self._CIRCLE_EDGE_SAMPLES
            fraction = (center[0] + (radius_px * math.cos(angle)) / full_w, center[1] + (radius_px * math.sin(angle)) / full_h)
            edge.append(self._screen_point(self._to_point(rect, fraction), rect, widget_rect))
        return any(
            self._distance_to_segment(cursor, a, b) <= OVERLAY_ENDPOINT_HIT_RADIUS
            for a, b in zip(edge, edge[1:] + edge[:1])
        )

    def _near_annulus(
        self, cursor: QPointF, geometry: tuple[tuple[float, float], float, float],
        rect: QRect, widget_rect: QRect, full_dims: tuple[int, int],
    ) -> bool:
        center, outer_r, inner_r = geometry
        return (
            self._near_circle_edge(cursor, center, outer_r, rect, widget_rect, full_dims)
            or self._near_circle_edge(cursor, center, inner_r, rect, widget_rect, full_dims)
        )

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

    def _ellipse_edge_screen_points(
        self,
        kind: str,
        points: tuple[tuple[float, float], ...],
        rect: QRect,
        widget_rect: QRect,
        full_dims: tuple[int, int],
    ) -> list[QPointF]:
        """Same sampling approach as _circle_edge_screen_points, generalized to a rotated ellipse."""
        geometry = self._ellipse_geometry(kind, points, full_dims)
        if geometry is None:
            return []
        (cx, cy), rx_px, ry_px, rotation_deg = geometry
        full_w, full_h = full_dims
        rot = math.radians(rotation_deg)
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        result = []
        for i in range(self._CIRCLE_EDGE_SAMPLES):
            angle = 2 * math.pi * i / self._CIRCLE_EDGE_SAMPLES
            local_x, local_y = rx_px * math.cos(angle), ry_px * math.sin(angle)
            px = local_x * cos_r - local_y * sin_r
            py = local_x * sin_r + local_y * cos_r
            fraction = (cx + px / full_w, cy + py / full_h)
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
        *kind* — see ``MeasurementKind.resolve`` in measurement_kind.py —
        or None if they don't describe a valid measurement (e.g. two
        clicks landed on the same spot, or three on a straight line).
        Circle kinds keep their raw points as clicked — the actual
        center/radius is derived at draw time (see ``_circle_geometry``)
        so a later endpoint drag recomputes it for free.
        """
        entry = DEFAULT_REGISTRY.get(kind)
        return entry.resolve(points) if entry is not None else None

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

    @property
    def drag_measurement_index(self) -> int | None:
        """Which measurement an endpoint drag is currently moving, if any — read by OverlayLabel to tell a plain click on a point (which opens the customize menu) from a drag (which moves it)."""
        return self._drag_measurement_index

    # ------------------------------------------------------------------
    # Tag dragging — see feature 4. Moves only a measurement's tag,
    # accumulating a fraction-space offset into its meta so the tag stays
    # put relative to the anchor through pans/zooms and endpoint edits.
    # ------------------------------------------------------------------

    @property
    def dragging_tag(self) -> bool:
        return self._tag_drag_index is not None

    def begin_tag_drag(self, pos: QPoint, widget_rect: QRect, index: int) -> bool:
        fraction = self._to_fraction(pos, widget_rect)
        meta = self.measurement_meta(index)
        if fraction is None or meta is None:
            return False
        self._tag_drag_index = index
        self._tag_drag_start_fraction = fraction
        self._tag_drag_start_offset = (meta.tag_offset_x, meta.tag_offset_y)
        return True

    def update_tag_drag(self, pos: QPoint, widget_rect: QRect) -> bool:
        if self._tag_drag_index is None or self._tag_drag_start_fraction is None:
            return False
        fraction = self._to_fraction(pos, widget_rect)
        meta = self.measurement_meta(self._tag_drag_index)
        if fraction is None or meta is None:
            return False
        offset_x = self._tag_drag_start_offset[0] + (fraction[0] - self._tag_drag_start_fraction[0])
        offset_y = self._tag_drag_start_offset[1] + (fraction[1] - self._tag_drag_start_fraction[1])
        self.set_measurement_meta(
            self._tag_drag_index, meta._replace(tag_offset_x=offset_x, tag_offset_y=offset_y)
        )
        return True

    def end_tag_drag(self) -> None:
        self._tag_drag_index = None
        self._tag_drag_start_fraction = None

    # ------------------------------------------------------------------
    # Secondary-tag dragging — the extra_measures tags, each moved by its
    # own offset in meta.extra_offsets (keyed by extra index).
    # ------------------------------------------------------------------

    @property
    def dragging_extra_tag(self) -> bool:
        return self._extra_drag_key is not None

    def begin_extra_tag_drag(self, pos: QPoint, widget_rect: QRect, key: tuple[int, int]) -> bool:
        fraction = self._to_fraction(pos, widget_rect)
        meta = self.measurement_meta(key[0])
        if fraction is None or meta is None:
            return False
        self._extra_drag_key = key
        self._extra_drag_start_fraction = fraction
        self._extra_drag_start_offset = self._extra_offset(meta, key[1])
        return True

    def update_extra_tag_drag(self, pos: QPoint, widget_rect: QRect) -> bool:
        if self._extra_drag_key is None or self._extra_drag_start_fraction is None:
            return False
        fraction = self._to_fraction(pos, widget_rect)
        meta = self.measurement_meta(self._extra_drag_key[0])
        if fraction is None or meta is None:
            return False
        offset_x = self._extra_drag_start_offset[0] + (fraction[0] - self._extra_drag_start_fraction[0])
        offset_y = self._extra_drag_start_offset[1] + (fraction[1] - self._extra_drag_start_fraction[1])
        self.set_measurement_meta(
            self._extra_drag_key[0], self._set_extra_offset(meta, self._extra_drag_key[1], offset_x, offset_y)
        )
        return True

    def end_extra_tag_drag(self) -> None:
        self._extra_drag_key = None
        self._extra_drag_start_fraction = None

    @staticmethod
    def _extra_offset(meta: MeasurementMeta, extra_index: int) -> tuple[float, float]:
        for i, dx, dy in meta.extra_offsets:
            if i == extra_index:
                return (dx, dy)
        return (0.0, 0.0)

    @staticmethod
    def _set_extra_offset(meta: MeasurementMeta, extra_index: int, dx: float, dy: float) -> MeasurementMeta:
        others = [t for t in meta.extra_offsets if t[0] != extra_index]
        return meta._replace(extra_offsets=tuple(others + [(extra_index, dx, dy)]))

    def _move_point(self, m_index: int, p_index: int, new_point: tuple[float, float]) -> None:
        """
        Apply a dragged point's new position to the measurement at
        *m_index* — see ``MeasurementKind.move_point`` in
        measurement_kind.py for the kind-specific drag constraints
        (horizontal/vertical lines staying axis-locked, a radius
        circle's center translating the whole shape); every other point
        (polyline points, circle-edge points, a lone "Point") just moves
        freely by default.
        """
        measurements = self.measurements
        if m_index < 0 or m_index >= len(measurements):
            return
        measurement = measurements[m_index]
        points = list(measurement.points)
        if p_index < 0 or p_index >= len(points):
            return

        entry = DEFAULT_REGISTRY.get(measurement.kind)
        if entry is not None:
            new_points = entry.move_point(points, p_index, new_point)
        else:
            points[p_index] = new_point
            new_points = points
        measurements[m_index] = measurement._replace(points=self._apply_snap(measurement.kind, tuple(new_points)))

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
        kind, or None if *points* don't yet describe one — see
        ``MeasurementKind.circle_geometry`` in measurement_kind.py, which
        converts to true source-pixel coordinates first since points are
        stored as independent x/y fractions of the frame, and the frame
        is rarely square, so a radius computed directly in fraction
        space would be wrong whenever width and height scale differently.
        """
        entry = DEFAULT_REGISTRY.get(kind)
        if entry is None or entry.circle_geometry is None:
            return None
        return entry.circle_geometry(points, full_dims)

    @staticmethod
    def _ellipse_geometry(
        kind: str,
        points: tuple[tuple[float, float], ...],
        full_dims: tuple[int, int],
    ) -> tuple[tuple[float, float], float, float, float] | None:
        """(center_fraction, rx_px, ry_px, rotation_deg) for an ellipse kind — see MeasurementKind.ellipse_geometry, the ellipse counterpart of _circle_geometry above."""
        entry = DEFAULT_REGISTRY.get(kind)
        if entry is None or entry.ellipse_geometry is None:
            return None
        return entry.ellipse_geometry(points, full_dims)

    @staticmethod
    def _arc_geometry(
        kind: str,
        points: tuple[tuple[float, float], ...],
        full_dims: tuple[int, int],
    ) -> tuple[tuple[float, float], float, float, float] | None:
        """(center_fraction, radius_px, start_deg, sweep_deg) for an arc kind — see MeasurementKind.arc_geometry, the arc counterpart of _circle_geometry above."""
        entry = DEFAULT_REGISTRY.get(kind)
        if entry is None or entry.arc_geometry is None:
            return None
        return entry.arc_geometry(points, full_dims)

    _ARC_SAMPLES_PER_DEGREE = 1 / 6  # one sample every 6 degrees of sweep
    _ARC_MIN_SAMPLES = 8

    def _arc_sample_points(
        self, kind: str, points: tuple[tuple[float, float], ...], full_dims: tuple[int, int]
    ) -> list[tuple[float, float]]:
        """Points (fraction space) evenly sampled along the arc *kind*/*points* describes, from its start angle through its full sweep — the arc drawn/hit-tested as a many-segment polyline rather than reasoned about via Qt's own arc-angle conventions (which this sidesteps entirely)."""
        geometry = self._arc_geometry(kind, points, full_dims)
        if geometry is None:
            return []
        (cx, cy), radius_px, start_deg, sweep_deg = geometry
        full_w, full_h = full_dims
        if full_w <= 0 or full_h <= 0:
            return []
        samples = max(self._ARC_MIN_SAMPLES, round(abs(sweep_deg) * self._ARC_SAMPLES_PER_DEGREE))
        result = []
        for i in range(samples + 1):
            angle = math.radians(start_deg + sweep_deg * i / samples)
            result.append((
                cx + (radius_px * math.cos(angle)) / full_w,
                cy + (radius_px * math.sin(angle)) / full_h,
            ))
        return result

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, painter: QPainter, rect: QRect) -> None:
        self._draw_placed_measurements(painter, rect)

        scale_x, scale_y = self._zoom_handler.current_scale_xy() if self._zoom_handler is not None else (1.0, 1.0)
        stroke_scale = (scale_x + scale_y) / 2
        full_dims = self._zoom_handler.current_frame_dims() if self._zoom_handler is not None else None

        if self._calibration_line is not None:
            self._draw_polyline(painter, rect, self._calibration_line, stroke_scale, dashed=False)
            for point in self._calibration_line:
                self._draw_endpoint(painter, self._to_point(rect, point), scale_x, scale_y)
            self._draw_label(painter, rect, self._midpoint(self._calibration_line), "Calibration", scale_x, scale_y)

        self._draw_draft(painter, rect, stroke_scale, scale_x, scale_y, full_dims)

    def _draw_placed_measurements(self, painter: QPainter, rect: QRect) -> None:
        """
        The finalized measurements only — no in-progress draft, no
        manual-calibration reference line (neither is a placed
        measurement). Split out from draw() so an export renderer can
        burn in exactly what's actually been measured, nothing ephemeral.
        """
        scale_x, scale_y = self._zoom_handler.current_scale_xy() if self._zoom_handler is not None else (1.0, 1.0)
        stroke_scale = (scale_x + scale_y) / 2
        full_dims = self._zoom_handler.current_frame_dims() if self._zoom_handler is not None else None

        self._label_boxes = {}
        self._delete_boxes = {}
        self._extra_label_boxes = {}

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
            # Opacity fades the line fill and its border together
            # (feature 10) by wrapping both passes at once.
            painter.save()
            painter.setOpacity(max(0.0, min(1.0, meta.opacity)))
            indicator_color = self._resolve_color(meta.indicator_color, line_color)
            self._draw_measurement(
                painter, rect, measurement.kind, measurement.points, stroke_scale, scale_x, scale_y, full_dims,
                line_color=line_color, line_width=line_width,
                outline_color=outline_color, outline_width=outline_width,
                dash_style=meta.line_dash_style, start_cap=meta.line_start_cap, end_cap=meta.line_end_cap,
                cap_size=meta.cap_size_scale, fill_color=self._resolve_fill(meta),
                indicator_enabled=meta.indicator_enabled, indicator_color=indicator_color,
                indicator_opacity=meta.indicator_opacity, indicator_dash_style=meta.indicator_dash_style,
            )
            entry = DEFAULT_REGISTRY.get(measurement.kind)
            if entry is not None and entry.category == "line" and meta.midpoint_style != "none":
                self._draw_midpoint_marker(
                    painter, rect, measurement.points, meta.midpoint_style, stroke_scale,
                    line_color=line_color, line_width=line_width,
                    outline_color=outline_color, outline_width=outline_width,
                )
            if entry is not None and entry.category == "angle" and meta.indicator_enabled:
                self._draw_angle_indicator(
                    painter, rect, measurement.kind, measurement.points, stroke_scale,
                    indicator_color, line_width, meta.indicator_dash_style, meta.indicator_opacity,
                )
            painter.restore()
            if index == self._near_index or index == self._drag_measurement_index:
                for point in measurement.points:
                    self._draw_endpoint(painter, self._to_point(rect, point), scale_x, scale_y)
            if meta.always_show_center and full_dims is not None:
                center = self._measurement_center(measurement.kind, measurement.points, full_dims)
                if center is not None:
                    self._draw_endpoint(painter, self._to_point(rect, center), scale_x, scale_y)

        # Tags are drawn in a second pass, after every measurement's
        # geometry — otherwise a later measurement's interior fill (or any
        # opaque geometry) would paint over an earlier one's tag. meta.hidden
        # hides only the tag, not the geometry drawn above.
        for index, measurement in enumerate(self.measurements):
            self._draw_measurement_label(painter, rect, index, measurement, scale_x, scale_y, full_dims)
            self._draw_extra_measure_labels(painter, rect, index, measurement, scale_x, scale_y, full_dims)

    def _draw_extra_measure_labels(
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
        Secondary distance tags a kind exposes via
        MeasurementKind.extra_measures — every gap between an arbitrary
        parallel's lines, or the perpendicular leg of a 3pt perp. Each is
        recorded into _extra_label_boxes so hovering one and pressing
        Delete can dismiss it (meta.hidden_extra); one already dismissed
        is skipped entirely.
        """
        meta = measurement.meta
        if meta.hidden or full_dims is None:
            return
        entry = DEFAULT_REGISTRY.get(measurement.kind)
        if entry is None or entry.extra_measures is None:
            return
        dims = self._reference_dims(full_dims)
        unit = meta.unit if meta.unit is not None else self._unit
        if dims is None or not self._can_measure(unit):
            return
        decimals = meta.decimal_places
        offsets = {i: (dx, dy) for i, dx, dy in meta.extra_offsets}
        for extra_index, (anchor, distance_px) in enumerate(entry.extra_measures(measurement.points, dims)):
            if extra_index in meta.hidden_extra:
                continue
            off = offsets.get(extra_index, (0.0, 0.0))
            anchor = (anchor[0] + off[0], anchor[1] + off[1])
            text = format_length(distance_px, self.dpi or 1.0, unit, decimals)
            box, _ = self._draw_label(
                painter, rect, anchor, text, scale_x, scale_y,
                bg_color=self._resolve_color(meta.tag_background_color, OVERLAY_LINE_COLOR),
                text_color=self._resolve_color(meta.tag_text_color, OVERLAY_OUTLINE_COLOR),
                transparent_bg=meta.tag_background_transparent,
                font_family=meta.font_family, font_size=meta.font_size, tag_width=meta.tag_width,
            )
            self._extra_label_boxes[(index, extra_index)] = box

    def _measurement_center(
        self, kind: str, points: tuple[tuple[float, float], ...], full_dims: tuple[int, int]
    ) -> tuple[float, float] | None:
        """Center point (fraction space) of a circle/ellipse/"Radius Arc" measurement, or None for any other kind — see meta.always_show_center."""
        entry = DEFAULT_REGISTRY.get(kind)
        if entry is None:
            return None
        if entry.category == "circle":
            geometry = self._circle_geometry(kind, points, full_dims)
            return geometry[0] if geometry is not None else None
        if entry.category == "ellipse":
            geometry = self._ellipse_geometry(kind, points, full_dims)
            return geometry[0] if geometry is not None else None
        if kind == "Radius Arc":
            geometry = self._arc_geometry(kind, points, full_dims)
            return geometry[0] if geometry is not None else None
        return None

    def draw_placed_measurements_with_coordinate_space(
        self, painter: QPainter, rect: QRect, coords: CoordinateSpace
    ) -> None:
        """
        Draw only the finalized measurements against *coords* instead of
        whatever coordinate space is normally registered (see
        set_zoom_handler) — used for exporting a full-resolution image,
        where there's no live pan/zoom viewport at all, just a 1:1
        IdentityCoordinateSpace sized to the export target.
        """
        previous = self._zoom_handler
        self._zoom_handler = coords
        try:
            self._draw_placed_measurements(painter, rect)
        finally:
            self._zoom_handler = previous

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
        cap_size: float = 1.0,
        fill_color: QColor | None = None,
        indicator_enabled: bool = True,
        indicator_color: QColor | None = None,
        indicator_opacity: float = 1.0,
        indicator_dash_style: str = "dash",
    ) -> None:
        entry = DEFAULT_REGISTRY.get(kind)
        if entry is None:
            return
        if entry.category == "line":
            self._draw_polyline(
                painter, rect, points, stroke_scale, dashed=dashed,
                line_color=line_color, line_width=line_width, outline_color=outline_color, outline_width=outline_width,
                dash_style=dash_style, start_cap=start_cap, end_cap=end_cap, cap_size=cap_size,
            )
        elif entry.category == "circle" and full_dims is not None:
            self._draw_circle(
                painter, rect, kind, points, stroke_scale, full_dims,
                dashed=dashed, line_color=line_color, line_width=line_width,
                outline_color=outline_color, outline_width=outline_width, dash_style=dash_style, fill_color=fill_color,
            )
        elif entry.category == "ellipse" and full_dims is not None:
            self._draw_ellipse(
                painter, rect, kind, points, stroke_scale, full_dims,
                dashed=dashed, line_color=line_color, line_width=line_width,
                outline_color=outline_color, outline_width=outline_width, dash_style=dash_style, fill_color=fill_color,
            )
        elif entry.category == "polygon" and entry.polygon_points is not None:
            verts = entry.polygon_points(points, full_dims)
            if verts is not None and len(verts) >= 2:
                closed = tuple(verts) + (verts[0],)
                if fill_color is not None and len(verts) >= 3:
                    self._fill_polygon(painter, rect, verts, fill_color)
                self._draw_polyline(
                    painter, rect, closed, stroke_scale, dashed=dashed,
                    line_color=line_color, line_width=line_width,
                    outline_color=outline_color, outline_width=outline_width, dash_style=dash_style,
                )
        elif entry.category == "annulus" and full_dims is not None and entry.annulus_geometry is not None:
            self._draw_annulus(
                painter, rect, kind, points, stroke_scale, full_dims,
                dashed=dashed, line_color=line_color, line_width=line_width,
                outline_color=outline_color, outline_width=outline_width, dash_style=dash_style, fill_color=fill_color,
            )
        elif entry.category == "two_circle" and full_dims is not None and entry.two_circle_geometry is not None:
            self._draw_two_circle(
                painter, rect, kind, points, stroke_scale, full_dims,
                dashed=dashed, line_color=line_color, line_width=line_width,
                outline_color=outline_color, outline_width=outline_width, dash_style=dash_style, fill_color=fill_color,
            )
        elif entry.category in ("angle", "line_pair") and entry.segment_pairs is not None:
            if entry.connected_segments:
                # The segments share an endpoint (e.g. "3 Point Angle"'s
                # vertex) — draw as one polyline over the raw points, the
                # same proper-joint union "Arbitrary Line" uses, rather
                # than two independently-stroked segments that would
                # double up the stroke/outline right at the joint.
                self._draw_polyline(
                    painter, rect, points, stroke_scale, dashed=dashed,
                    line_color=line_color, line_width=line_width,
                    outline_color=outline_color, outline_width=outline_width,
                    dash_style=dash_style, start_cap=start_cap, end_cap=end_cap, cap_size=cap_size,
                )
            else:
                # Genuinely disconnected segments (e.g. "4 Point Angle"'s
                # two independent lines, or a parallel/perpendicular pair's
                # lines) — drawing them as one polyline would wrongly join
                # them at the middle. line_pair lines stay solid even while
                # placing (only their indicator connectors dash); angle
                # segments still dash during the placement preview.
                seg_dashed = dashed and entry.category == "angle"
                for pair in entry.segment_pairs(points, full_dims):
                    self._draw_polyline(
                        painter, rect, pair, stroke_scale, dashed=seg_dashed,
                        line_color=line_color, line_width=line_width,
                        outline_color=outline_color, outline_width=outline_width,
                        dash_style=dash_style, start_cap=start_cap, end_cap=end_cap, cap_size=cap_size,
                    )
            # The indicator line: parallel/perpendicular dimension
            # connectors and midlines. A customizable dashed guide (color,
            # opacity, dash style, on/off — see meta.indicator_*).
            if entry.connector_segments is not None and indicator_enabled:
                for pair in entry.connector_segments(points, full_dims):
                    self._draw_indicator_line(
                        painter, rect, pair, stroke_scale,
                        indicator_color if indicator_color is not None else line_color,
                        indicator_opacity, indicator_dash_style, line_width,
                    )
        elif entry.category == "arc" and full_dims is not None:
            # Drawn as a many-segment polyline sampled along the arc —
            # reuses the same stroke/dash/cap machinery as any other
            # line rather than reasoning about Qt's own arc-angle
            # conventions (see _arc_sample_points).
            arc_points = self._arc_sample_points(kind, points, full_dims)
            if len(arc_points) >= 2:
                self._draw_polyline(
                    painter, rect, arc_points, stroke_scale, dashed=dashed,
                    line_color=line_color, line_width=line_width,
                    outline_color=outline_color, outline_width=outline_width,
                    dash_style=dash_style, start_cap=start_cap, end_cap=end_cap, cap_size=cap_size,
                )
        elif entry.category == "curve" and entry.curve_points is not None:
            # Sampled points along the Bezier curve, drawn the same way
            # an arc's sampled points are — reuses the ordinary polyline
            # stroke/dash/cap machinery rather than a bespoke Bezier
            # painter path, so it picks up caps/outline/dash for free.
            curve = entry.curve_points(points)
            if curve is not None and len(curve) >= 2:
                self._draw_polyline(
                    painter, rect, curve, stroke_scale, dashed=dashed,
                    line_color=line_color, line_width=line_width,
                    outline_color=outline_color, outline_width=outline_width,
                    dash_style=dash_style, start_cap=start_cap, end_cap=end_cap, cap_size=cap_size,
                )
        elif entry.category == "point" and points:
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

    def _draw_midpoint_marker(
        self,
        painter: QPainter,
        rect: QRect,
        points: tuple[tuple[float, float], ...],
        style: str,
        stroke_scale: float,
        *,
        line_color: QColor,
        line_width: float,
        outline_color: QColor,
        outline_width: float,
    ) -> None:
        """
        A small tick (a single stroke perpendicular to the line) or x (a
        pair of crossed strokes) centered on the line's midpoint, oriented
        from the overall first-to-last direction so a tick reads as square
        across the line at any angle. Sized in fixed on-screen pixels
        (counter-scaled by *stroke_scale*) and drawn line-color over a
        slightly wider outline-color pass, the same legible-on-any-
        background treatment as the stroke itself (feature 8).
        """
        if len(points) < 2:
            return
        mid = self._to_point(rect, self._midpoint(points))
        start = self._to_point(rect, points[0])
        end = self._to_point(rect, points[-1])
        dx, dy = end.x() - start.x(), end.y() - start.y()
        length = math.hypot(dx, dy)
        if length <= 0:
            return
        ux, uy = dx / length, dy / length
        px, py = -uy, ux  # unit perpendicular
        half = (OVERLAY_POINT_RADIUS * (line_width / OVERLAY_LINE_WIDTH) + 2.0) / stroke_scale

        if style == "x":
            axes = ((ux + px, uy + py), (ux - px, uy - py))
        else:  # "tick"
            axes = ((px, py),)

        segments = []
        for ax, ay in axes:
            norm = math.hypot(ax, ay) or 1.0
            hx, hy = ax / norm * half, ay / norm * half
            segments.append((QPointF(mid.x() - hx, mid.y() - hy), QPointF(mid.x() + hx, mid.y() + hy)))

        for width, color in (
            ((line_width + outline_width * 2) / stroke_scale, outline_color),
            (line_width / stroke_scale, line_color),
        ):
            pen = QPen(color)
            pen.setWidthF(width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            for a, b in segments:
                painter.drawLine(a, b)

    def _draw_indicator_line(
        self,
        painter: QPainter,
        rect: QRect,
        segment: tuple[tuple[float, float], tuple[float, float]],
        stroke_scale: float,
        color: QColor,
        opacity: float,
        dash_style: str,
        line_width: float,
    ) -> None:
        """One indicator/dimension guide segment — a single styled line
        (color, opacity, dash style), fixed on-screen width like the rest
        of the chrome. See meta.indicator_* and _draw_measurement."""
        a = self._to_point(rect, segment[0])
        b = self._to_point(rect, segment[1])
        width = max(line_width, 1.0)
        pen = QPen(color)
        pen.setWidthF(width / stroke_scale)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pattern = resolve_dash_pattern(dash_style, width)
        if pattern:
            pen.setStyle(Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([value / width for value in pattern])
        else:
            pen.setStyle(Qt.PenStyle.SolidLine)
        painter.save()
        painter.setOpacity(max(0.0, min(1.0, opacity)))
        painter.setPen(pen)
        painter.drawLine(a, b)
        painter.restore()

    def _draw_angle_indicator(
        self, painter: QPainter, rect: QRect, kind: str, points: tuple[tuple[float, float], ...],
        stroke_scale: float, color: QColor, line_width: float = OVERLAY_LINE_WIDTH,
        dash_style: str = "dash", opacity: float = 1.0,
    ) -> None:
        """
        A dashed guide from each leg's own nearer end out to the angle's
        anchor (skipped for "3 Point Angle", whose legs already meet
        there), plus a small curved arc at the anchor sweeping between
        the two legs' directions — see meta.indicator_enabled.

        The guide/arc pens track the measurement's own *line_width* so a
        thicker line gets a proportionally heavier indicator; *dash_style*
        picks the guide's dash pattern (see resolve_dash_pattern),
        defaulting to the classic evenly-spaced dashes.

        Fixed on-screen size UI chrome, not image content, so — like
        ``_draw_midpoint_marker`` — everything here is computed directly
        from screen ("rect") positions, with only the radius counter-
        scaled by *stroke_scale* to hold it constant across zoom; no
        separate true-pixel-space angle math is needed, since ``rect``
        space shares the source frame's own aspect ratio.
        """
        entry = DEFAULT_REGISTRY.get(kind)
        if entry is None or entry.angle_anchor is None or entry.segment_pairs is None:
            return
        legs = entry.segment_pairs(points, None)  # angle kinds never need full_dims for their own segments
        if len(legs) < 2:
            return
        anchor = self._to_point(rect, entry.angle_anchor(points))

        # Fold the indicator's own opacity into the pen color's alpha
        # (rather than painter.setOpacity, which the method's early returns
        # would leave unbalanced against a save()).
        color = QColor(color)
        color.setAlphaF(max(0.0, min(1.0, opacity)))
        guide_width = line_width / stroke_scale
        far_points = []
        guide_pen = QPen(color)
        guide_pen.setWidthF(guide_width)
        guide_pen.setStyle(Qt.PenStyle.CustomDashLine)
        pattern = resolve_dash_pattern(dash_style, line_width) or [OVERLAY_DASH_LENGTH, OVERLAY_DASH_GAP]
        guide_pen.setDashPattern([value / guide_width for value in pattern] if guide_width > 0 else pattern)
        for leg_a, leg_b in legs:
            a_pt, b_pt = self._to_point(rect, leg_a), self._to_point(rect, leg_b)
            near, far = (a_pt, b_pt) if self._distance(anchor, a_pt) <= self._distance(anchor, b_pt) else (b_pt, a_pt)
            far_points.append(far)
            if not entry.connected_segments and self._distance(anchor, near) > _INDICATOR_EPSILON:
                painter.setPen(guide_pen)
                painter.drawLine(near, anchor)

        if len(far_points) < 2:
            return
        angle1 = math.degrees(math.atan2(far_points[0].y() - anchor.y(), far_points[0].x() - anchor.x()))
        angle2 = math.degrees(math.atan2(far_points[1].y() - anchor.y(), far_points[1].x() - anchor.x()))
        sweep = ((angle2 - angle1 + 180) % 360) - 180
        if abs(sweep) < _INDICATOR_EPSILON:
            return

        radius = (OVERLAY_POINT_RADIUS * 2.2) / stroke_scale
        samples = max(6, round(abs(sweep) / 8))
        arc_pen = QPen(color)
        arc_pen.setWidthF(line_width * 0.9 / stroke_scale)
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(arc_pen)
        prev = None
        for i in range(samples + 1):
            a = math.radians(angle1 + sweep * i / samples)
            p = QPointF(anchor.x() + radius * math.cos(a), anchor.y() + radius * math.sin(a))
            if prev is not None:
                painter.drawLine(prev, p)
            prev = p

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
        meta = measurement.meta
        # "Hide measurement" hides the tag only (feature 11 clarified);
        # the geometry is drawn regardless, by the caller.
        if meta.hidden:
            return
        dims = self._reference_dims(full_dims)
        text_and_anchor = self._measurement_label(measurement.kind, measurement.points, meta, full_dims, dims)
        if text_and_anchor is None:
            return
        text, anchor = text_and_anchor
        # Whether the tag exists at all (independent of hover — a tag
        # that doesn't draw a box has no hit-test region, so it could
        # never become hovered in the first place): no title and no
        # value suffix (e.g. a bare point, or before DPI is set) means
        # there's nothing worth a box unless a description is set to
        # always show.
        always_show = meta.description if meta.always_show_description else None
        if not text and not always_show:
            return
        # The description also reveals on hover, same as always-show,
        # so hovering a tag whose box already exists shows its
        # description without needing the checkbox on.
        hovered = index == self._hovered_index
        description = meta.description if (meta.always_show_description or hovered) else None
        box, delete_box = self._draw_label(
            painter, rect, anchor, text, scale_x, scale_y,
            show_delete=hovered,
            bg_color=self._resolve_color(meta.tag_background_color, OVERLAY_LINE_COLOR),
            text_color=self._resolve_color(meta.tag_text_color, OVERLAY_OUTLINE_COLOR),
            description=description,
            transparent_bg=meta.tag_background_transparent,
            font_family=meta.font_family, font_size=meta.font_size, tag_width=meta.tag_width,
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

    @staticmethod
    def _resolve_fill(meta: MeasurementMeta) -> QColor | None:
        """Interior fill color (with its own alpha) for an enclosed shape, or None when no fill is set — see meta.fill_color."""
        if not meta.fill_color:
            return None
        color = QColor(meta.fill_color)
        if not color.isValid():
            return None
        color.setAlphaF(max(0.0, min(1.0, meta.fill_opacity)))
        return color

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
        text_and_anchor = self._measurement_label(kind, points, self._default_meta, full_dims, dims)
        if text_and_anchor is None:
            return
        text, anchor = text_and_anchor
        if not text:
            return
        self._draw_label(painter, rect, anchor, text, scale_x, scale_y)

    def _measurement_label(
        self,
        kind: str,
        points: tuple[tuple[float, float], ...],
        meta: MeasurementMeta,
        full_dims: tuple[int, int] | None,
        dims: tuple[int, int] | None,
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

        *full_dims* (raw, matching whatever the shape was actually drawn
        against — see _draw_circle) anchors a circle's tag; *dims* (the
        DPI reference resolution — see _reference_dims) sizes the
        length/diameter suffix. The two differ for live view (the
        preview stream vs. the still-capture resolution DPI was
        calibrated against) — a circle's center is a nonlinear function
        of aspect ratio for "3 Point Circle" (not for "Radius Circle"/
        "Diameter", whose center formula is aspect-invariant), so
        anchoring against one dims and measuring against the other
        visibly mislocates the tag.
        """
        entry = DEFAULT_REGISTRY.get(kind)
        if entry is None or not entry.has_label:
            return None

        unit = meta.unit if meta.unit is not None else self._unit
        decimals = meta.decimal_places

        if entry.category == "line":
            suffix = self._length_suffix(points, dims, unit, decimals)
            anchor = self._midpoint(points)
        elif entry.category == "circle":
            if full_dims is None:
                return None
            geometry = self._circle_geometry(kind, points, full_dims)
            if geometry is None:
                return None
            anchor, _radius_px = geometry
            suffix = None
            if self._can_measure(unit) and dims is not None:
                suffix_geometry = self._circle_geometry(kind, points, dims)
                if suffix_geometry is not None:
                    _, suffix_radius_px = suffix_geometry
                    suffix = f"\u00d8 {format_length(suffix_radius_px * 2, self.dpi or 1.0, unit, decimals)}"
                    if meta.show_area:
                        area_unit = meta.area_unit if meta.area_unit is not None else unit
                        area_px = math.pi * suffix_radius_px * suffix_radius_px
                        suffix = f"{suffix} \u00b7 {format_area(area_px, self.dpi or 1.0, area_unit, decimals)}"
        elif entry.category == "ellipse":
            if full_dims is None:
                return None
            geometry = self._ellipse_geometry(kind, points, full_dims)
            if geometry is None:
                return None
            anchor, _rx_px, _ry_px, _rotation = geometry
            suffix = None
            if self._can_measure(unit) and dims is not None:
                suffix_geometry = self._ellipse_geometry(kind, points, dims)
                if suffix_geometry is not None:
                    _, suffix_rx_px, suffix_ry_px, _ = suffix_geometry
                    major = format_length(suffix_rx_px * 2, self.dpi or 1.0, unit, decimals)
                    minor = format_length(suffix_ry_px * 2, self.dpi or 1.0, unit, decimals)
                    suffix = f"{major} \u00d7 {minor}"
                    if meta.show_area:
                        area_unit = meta.area_unit if meta.area_unit is not None else unit
                        area_px = math.pi * suffix_rx_px * suffix_ry_px
                        suffix = f"{suffix} \u00b7 {format_area(area_px, self.dpi or 1.0, area_unit, decimals)}"
        elif entry.category == "angle":
            if entry.angle_anchor is None or not points:
                return None
            anchor = entry.angle_anchor(points)
            suffix = None
            if full_dims is not None and entry.angle_value is not None:
                angle_deg = entry.angle_value(points, full_dims)
                if angle_deg is not None:
                    suffix = f"{angle_deg:.{decimals}f}\u00b0"
            if meta.show_leg_lengths and dims is not None and entry.segment_pairs is not None:
                # Only once both legs are actually placed (all 2
                # expected segments present) \u2014 a partial draft, still
                # being placed, skips this rather than showing just one
                # leg's length.
                legs = entry.segment_pairs(points, full_dims)
                if len(legs) >= 2:
                    leg_lengths = [self._length_suffix(pair, dims, unit, decimals) for pair in legs]
                    if all(leg is not None for leg in leg_lengths):
                        legs_text = " / ".join(leg_lengths)
                        suffix = f"{suffix} \u00b7 {legs_text}" if suffix else legs_text
        elif entry.category == "line_pair":
            if entry.segment_pairs is None:
                return None
            legs = entry.segment_pairs(points, full_dims)
            if not legs:
                return None
            # A parallel/perpendicular pair tags the perpendicular gap (or
            # the derived line's length) between its lines via
            # pair_distance; lacking that (e.g. arbitrary parallel), it
            # falls back to the reference segment's own length.
            pair = entry.pair_distance(points, dims) if (entry.pair_distance is not None and dims is not None) else None
            if pair is not None and self._can_measure(unit):
                anchor, distance_px = pair
                suffix = format_length(distance_px, self.dpi or 1.0, unit, decimals)
            else:
                anchor = self._midpoint(legs[0])
                suffix = self._length_suffix(legs[0], dims, unit, decimals)
        elif entry.category == "curve":
            if entry.curve_points is None:
                return None
            curve = entry.curve_points(points)
            if curve is None or len(curve) < 2:
                return None
            anchor = curve[len(curve) // 2]
            suffix = self._length_suffix(tuple(curve), dims, unit, decimals)
        elif entry.category == "arc":
            if full_dims is None:
                return None
            geometry = self._arc_geometry(kind, points, full_dims)
            if geometry is None:
                return None
            center, radius_px, start_deg, sweep_deg = geometry
            mid_angle = math.radians(start_deg + sweep_deg / 2)
            full_w, full_h = full_dims
            # Anchor at the arc's own midpoint, not the raw center point,
            # which would place the tag away from the arc itself.
            anchor = (
                center[0] + (radius_px * math.cos(mid_angle)) / full_w,
                center[1] + (radius_px * math.sin(mid_angle)) / full_h,
            )
            suffix = None
            if self._can_measure(unit) and dims is not None:
                suffix_geometry = self._arc_geometry(kind, points, dims)
                if suffix_geometry is not None:
                    _, suffix_radius_px, _start, suffix_sweep = suffix_geometry
                    radius_text = format_length(suffix_radius_px, self.dpi or 1.0, unit, decimals)
                    suffix = f"R {radius_text} \u00b7 {abs(suffix_sweep):.{decimals}f}\u00b0"
        elif entry.category == "polygon":
            if entry.polygon_points is None or full_dims is None:
                return None
            verts = entry.polygon_points(points, full_dims)
            if verts is None or len(verts) < 2:
                return None
            anchor = self._polygon_centroid(verts)
            suffix = None
            if self._can_measure(unit) and dims is not None and len(verts) >= 3:
                area_unit = meta.area_unit if meta.area_unit is not None else unit
                area_px = self._polygon_area_px(verts, dims)
                suffix = format_area(area_px, self.dpi or 1.0, area_unit, decimals)
        elif entry.category == "annulus":
            if entry.annulus_geometry is None or full_dims is None:
                return None
            geometry = entry.annulus_geometry(points, full_dims)
            if geometry is None:
                return None
            anchor = geometry[0]
            suffix = None
            if self._can_measure(unit) and dims is not None:
                suffix_geometry = entry.annulus_geometry(points, dims)
                if suffix_geometry is not None:
                    _, outer_r, inner_r = suffix_geometry
                    outer = format_length(outer_r * 2, self.dpi or 1.0, unit, decimals)
                    inner = format_length(inner_r * 2, self.dpi or 1.0, unit, decimals)
                    suffix = f"\u00d8 {outer} / {inner}"
                    if meta.show_area:
                        area_unit = meta.area_unit if meta.area_unit is not None else unit
                        ring_px = math.pi * (outer_r * outer_r - inner_r * inner_r)
                        suffix = f"{suffix} \u00b7 {format_area(ring_px, self.dpi or 1.0, area_unit, decimals)}"
        elif entry.category == "two_circle":
            if entry.two_circle_geometry is None or full_dims is None:
                return None
            geometry = entry.two_circle_geometry(points, full_dims)
            if geometry is None:
                return None
            c1, _r1, c2, _r2 = geometry
            anchor = self._midpoint((c1, c2))
            suffix = None
            if self._can_measure(unit) and dims is not None:
                suffix = self._length_suffix((c1, c2), dims, unit, decimals)
        else:
            if not points:
                return None
            anchor = points[0]
            suffix = None

        text = self._compose_label_text(meta.title, suffix)
        anchor = (anchor[0] + meta.tag_offset_x, anchor[1] + meta.tag_offset_y)
        return text, anchor

    @staticmethod
    def _compose_label_text(title: str, suffix: str | None) -> str:
        if title and suffix:
            return f"{title} \u00b7 {suffix}"
        return title or suffix or ""

    def _can_measure(self, unit: MeasurementUnit) -> bool:
        """Whether a real-world length can be shown: always for the DPI-independent pixel unit, otherwise only once a DPI is known."""
        return unit is MeasurementUnit.PX or self.dpi is not None

    def _length_suffix(
        self,
        points: tuple[tuple[float, float], ...],
        full_dims: tuple[int, int] | None,
        unit: MeasurementUnit,
        decimals: int = 2,
    ) -> str | None:
        if full_dims is None or not self._can_measure(unit):
            return None
        length_px = self._polyline_length_px(points, full_dims)
        if length_px <= 0:
            return None
        return format_length(length_px, self.dpi or 1.0, unit, decimals)

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
    def _polygon_centroid(verts: list[tuple[float, float]]) -> tuple[float, float]:
        return sum(v[0] for v in verts) / len(verts), sum(v[1] for v in verts) / len(verts)

    @staticmethod
    def _polygon_area_px(verts: list[tuple[float, float]], full_dims: tuple[int, int]) -> float:
        """Shoelace area of the closed polygon in true square pixels — computed in pixel space since an anisotropic frame scales x and y differently."""
        full_w, full_h = full_dims
        pts = [(x * full_w, y * full_h) for x, y in verts]
        total = 0.0
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            total += x0 * y1 - x1 * y0
        return abs(total) / 2

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
    def _right_rounded_path(rect: QRectF, radius: float, round_bottom: bool = True) -> QPainterPath:
        """
        Rect with its right corners rounded (matching the tag's own
        rounding where the delete strip meets the tag's edge) and square
        left corners (where it meets the tag's text, not an outer edge).
        *round_bottom* rounds the bottom-right corner too — true when
        *rect* is the tag's full height, false when it's just the header
        row (a description block continues below it — see feature 3's
        "delete only in the header" — so that corner is an interior seam,
        not a real edge, and stays square).
        """
        r = min(radius, rect.width() / 2, rect.height() / 2)
        path = QPainterPath()
        path.moveTo(rect.left(), rect.top())
        path.lineTo(rect.right() - r, rect.top())
        path.arcTo(rect.right() - 2 * r, rect.top(), 2 * r, 2 * r, 90, -90)
        if round_bottom:
            path.lineTo(rect.right(), rect.bottom() - r)
            path.arcTo(rect.right() - 2 * r, rect.bottom() - 2 * r, 2 * r, 2 * r, 0, -90)
        else:
            path.lineTo(rect.right(), rect.bottom())
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
        transparent_bg: bool = False,
        font_family: str = "",
        font_size: float = 0.0,
        tag_width: float = 0.0,
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
        The delete strip spans the header row's height (the whole tag's
        height if there's no header — see local_delete_box) and sits
        flush against its right edge, rather than floating as a small
        chip, so it reads as the whole right end of the header rather
        than a
        separate control on top of it. ``_draw_measurement_label``
        records the box/delete rects per measurement index so a later
        mouse event can hit-test against them (see ``_hit_test_tag``);
        calibration's own use of this method never passes *show_delete*,
        so its label isn't hoverable or deletable this way.
        """
        point = self._to_point(rect, anchor)

        base_size = font_size if font_size > 0 else OVERLAY_LABEL_FONT_SIZE
        font = QFont(painter.font())
        if font_family:
            font.setFamily(font_family)
        font.setPixelSize(max(1, round(base_size)))
        metrics = QFontMetricsF(font)
        pad_x = OVERLAY_LABEL_PADDING_X
        pad_y = OVERLAY_LABEL_PADDING_Y
        delete_width = (OVERLAY_DELETE_SIZE + OVERLAY_DELETE_MARGIN) if show_delete else 0.0

        # With no title line (a description-only tag), there's no title
        # row or divider. A description wraps within the title's width,
        # but never narrower than _DESC_MIN_WRAP_WIDTH — otherwise a
        # short title forces the description down to one word per line —
        # and the box widens to fit the description when it's wider than
        # the title.
        has_title = bool(text)
        text_line_height = (metrics.height() + pad_y * 2) if has_title else 0.0
        text_w = metrics.horizontalAdvance(text) if has_title else 0.0
        # The delete strip lives in the header row only (see below), so
        # it only needs to widen the box on the title's own account; a
        # title-less tag has no header row for it to sit in, so it falls
        # back to spanning the tag's full height instead — see
        # local_delete_box below — and needs to widen the description
        # row it then overlaps.
        header_w = text_w + delete_width if has_title else 0.0

        desc_font: QFont | None = None
        desc_block_height = 0.0
        desc_wrap_rect = QRectF()
        if description:
            desc_font = QFont(font)
            desc_font.setPixelSize(max(1, round(base_size - 2)))
            desc_metrics = QFontMetricsF(desc_font)
            wrap_width = max(text_w, _DESC_MIN_WRAP_WIDTH)
            desc_wrap_rect = desc_metrics.boundingRect(
                QRectF(0, 0, wrap_width, 10_000), Qt.TextFlag.TextWordWrap, description
            )
            # Extra padding above and below (not just the usual pad_y)
            # so the description reads as its own block under the
            # divider rather than crowding it.
            desc_block_height = desc_wrap_rect.height() + (pad_y * 3 if has_title else pad_y * 2)

        desc_w = desc_wrap_rect.width() + (delete_width if not has_title else 0.0)
        content_width = max(header_w, desc_w)
        # A set tag_width forces a minimum box width (text stays centered).
        box_w = max(content_width + pad_x * 2, tag_width)
        box_h = text_line_height + desc_block_height

        local_box = QRectF(-box_w / 2, -OVERLAY_LABEL_OFFSET - box_h, box_w, box_h)

        painter.save()
        painter.translate(point)
        if scale_x > 0 and scale_y > 0:
            painter.scale(1.0 / scale_x, 1.0 / scale_y)

        painter.setFont(font)
        # A transparent tag paints no fill (feature 9), and drops its
        # border too unless hovered — the hover outline is the click
        # affordance and always shows regardless.
        if show_delete:
            painter.setPen(QPen(OVERLAY_TAG_HOVER_COLOR, OVERLAY_OUTLINE_WIDTH))
        elif transparent_bg:
            painter.setPen(Qt.PenStyle.NoPen)
        else:
            painter.setPen(QPen(OVERLAY_OUTLINE_COLOR, OVERLAY_OUTLINE_WIDTH))
        painter.setBrush(Qt.BrushStyle.NoBrush if transparent_bg else QBrush(bg_color))
        painter.drawRoundedRect(local_box, OVERLAY_LABEL_CORNER_RADIUS, OVERLAY_LABEL_CORNER_RADIUS)

        # The header row (title) makes room for the delete strip; the
        # description row below it doesn't need to — the strip lives only
        # in the header (see local_delete_box below) — except when there
        # is no header at all, in which case the strip spans the tag's
        # full height and the description row must dodge it too.
        header_content_w = local_box.width() - delete_width
        desc_content_w = local_box.width() - (delete_width if not has_title else 0.0)
        if has_title:
            painter.setPen(QPen(text_color))
            painter.drawText(
                QRectF(local_box.x(), local_box.y(), header_content_w, text_line_height),
                Qt.AlignmentFlag.AlignCenter, text,
            )

        if description:
            desc_top = local_box.top() + text_line_height
            if has_title:
                divider_pen = QPen(text_color)
                divider_pen.setWidthF(1.0)
                painter.setPen(divider_pen)
                painter.drawLine(QPointF(local_box.left(), desc_top), QPointF(local_box.right(), desc_top))
                desc_top += pad_y * 1.5
            else:
                desc_top += pad_y

            muted = QColor(text_color)
            muted.setAlpha(180)
            painter.setFont(desc_font)
            painter.setPen(QPen(muted))
            desc_box = QRectF(
                local_box.x() + (desc_content_w - desc_wrap_rect.width()) / 2,
                desc_top,
                desc_wrap_rect.width(),
                desc_wrap_rect.height(),
            )
            painter.drawText(desc_box, Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap, description)
            painter.setFont(font)

        local_delete_box: QRectF | None = None
        if show_delete:
            # Confined to the header row (feature: "only in a tag's
            # header") — or, lacking one, the tag's full height, since
            # there's no header/body split to confine it to.
            delete_height = text_line_height if has_title else local_box.height()
            local_delete_box = QRectF(local_box.right() - delete_width, local_box.top(), delete_width, delete_height)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(OVERLAY_DELETE_BG_COLOR))
            painter.drawPath(self._right_rounded_path(local_delete_box, OVERLAY_LABEL_CORNER_RADIUS, round_bottom=not has_title))

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
        entry = DEFAULT_REGISTRY.get(kind)
        if entry is None:
            return

        if entry.required_points is None and entry.category == "line":
            # Unbounded polyline kinds (e.g. "Arbitrary Line"/"Multipoint
            # Line") keep accumulating points until cancelled. Confirmed
            # segments are drawn solid — they're placed, just not yet
            # finalized into a stored Measurement — and only the pending
            # segment to the cursor dashes, so it's clear which part of
            # the shape is still being decided.
            if len(points) >= 2:
                self._draw_polyline(painter, rect, points, stroke_scale, dashed=False)
            if preview is not None:
                self._draw_polyline(painter, rect, (points[-1], preview), stroke_scale, dashed=True)

            label_points = (*points, preview) if preview is not None else tuple(points)
            if len(label_points) >= 2:
                self._draw_draft_label(painter, rect, kind, label_points, scale_x, scale_y, full_dims)
        elif entry.required_points is None:
            # Unbounded non-polyline kinds (arbitrary parallel/perp) —
            # draw the whole in-progress shape, including the line the
            # cursor is currently placing, via the kind's own dispatch.
            preview_points = (*points, preview) if preview is not None else tuple(points)
            self._draw_measurement(
                painter, rect, kind, preview_points, stroke_scale, scale_x, scale_y, full_dims, dashed=True
            )
            if len(preview_points) >= 2:
                self._draw_draft_label(painter, rect, kind, preview_points, scale_x, scale_y, full_dims)
        else:
            preview_points = (*points, preview) if preview is not None else tuple(points)
            display_points = preview_points

            if len(preview_points) >= 2:
                # Draw the same shape a click would actually finalize
                # (e.g. a horizontal/vertical line's row/column lock)
                # rather than the raw clicked points, whenever resolve()
                # already accepts what's been placed so far — see
                # _resolve_measurement.
                resolved = entry.resolve(list(preview_points))
                if resolved is not None:
                    display_points = resolved

            self._draw_measurement(painter, rect, kind, display_points, stroke_scale, scale_x, scale_y, full_dims, dashed=True)

            # A capped line kind (e.g. an Arrow) shows its head while
            # placing too — the dashed preview shaft skips caps, so the
            # cap shapes are drawn solid on top from the kind's own preset.
            if entry.category == "line" and len(display_points) >= 2:
                preset = entry.meta_preset or {}
                start_cap = preset.get("line_start_cap", self._default_meta.line_start_cap)
                end_cap = preset.get("line_end_cap", self._default_meta.line_end_cap)
                self._draw_preview_caps(
                    painter, rect, tuple(display_points), stroke_scale, start_cap, end_cap,
                    self._default_meta.cap_size_scale,
                )

            # Annulus and two-circle aren't listed here: they preview each
            # sub-circle on its own as its points arrive (see their
            # partial geometry), so a straight guide would just clutter.
            if entry.category in ("circle", "ellipse", "arc") and len(preview_points) >= 2:
                required = entry.required_points or len(preview_points)
                if len(preview_points) < required:
                    # Not enough points for a circle/ellipse/arc yet — a
                    # straight guide between what's placed so far is
                    # still useful feedback (e.g. two of three "3 Point
                    # Circle"/"3 Point Ellipse"/"3 Point Arc" clicks, or
                    # the first few of a "5 Point Ellipse").
                    self._draw_polyline(painter, rect, preview_points, stroke_scale, dashed=True)

            if kind == CALIBRATION_KIND:
                # Calibration's own fixed label, never the generic
                # title/length tag (has_label=False) — genuinely unique
                # to this pseudo-kind rather than a general capability.
                if len(display_points) >= 2:
                    self._draw_label(painter, rect, self._midpoint(tuple(display_points)), "Calibration", scale_x, scale_y)
            elif len(display_points) >= 2:
                self._draw_draft_label(painter, rect, kind, tuple(display_points), scale_x, scale_y, full_dims)

            points = list(display_points)

        for point in points:
            self._draw_endpoint(painter, self._to_point(rect, point), scale_x, scale_y)
        if preview is not None and entry.required_points is None:
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
        cap_size: float = 1.0,
    ) -> None:
        last = len(points) - 2
        if dashed:
            for i in range(len(points) - 1):
                self._draw_stroke(
                    painter, rect, points[i], points[i + 1], stroke_scale, dashed=dashed,
                    line_color=line_color, line_width=line_width, outline_color=outline_color, outline_width=outline_width,
                    dash_style=dash_style,
                    start_cap=start_cap if i == 0 else "curved",
                    end_cap=end_cap if i == last else "curved",
                    cap_size=cap_size,
                )
            return

        # Finalized (non-preview) polylines union every segment's paths
        # into one running outline/fill path and paint each exactly once
        # — painting per-segment instead would composite two independent
        # antialiased passes on top of each other at every interior
        # joint (each segment unions an identical "curved" cap disc
        # there), reading as a doubled border ring.
        outline_path = QPainterPath()
        fill_path = QPainterPath()
        for i in range(len(points) - 1):
            seg_outline, seg_fill = self._stroke_paths(
                rect, points[i], points[i + 1], stroke_scale,
                line_color=line_color, line_width=line_width, outline_color=outline_color, outline_width=outline_width,
                dash_style=dash_style,
                start_cap=start_cap if i == 0 else "curved",
                end_cap=end_cap if i == last else "curved",
                cap_size=cap_size,
            )
            outline_path = outline_path.united(seg_outline)
            fill_path = fill_path.united(seg_fill)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(outline_color))
        painter.drawPath(outline_path)
        painter.setBrush(QBrush(line_color))
        painter.drawPath(fill_path)

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
        fill_color: QColor | None = None,
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

        if fill_color is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(fill_color))
            painter.drawEllipse(center_point, rx, ry)

        total_outline_width = line_width + outline_width * 2
        pattern = None if dashed else resolve_dash_pattern(dash_style, line_width)
        dash_cap = Qt.PenCapStyle.RoundCap if dash_style in ROUND_CAP_DASH_STYLES else Qt.PenCapStyle.FlatCap
        outline = QPen(outline_color)
        outline.setWidthF(total_outline_width / stroke_scale)
        outline.setCapStyle(dash_cap)
        self._apply_dash(outline, total_outline_width, dashed, pattern)
        painter.setPen(outline)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center_point, rx, ry)

        fill = QPen(line_color)
        fill.setWidthF(line_width / stroke_scale)
        fill.setCapStyle(dash_cap)
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

    def _draw_ellipse(
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
        fill_color: QColor | None = None,
    ) -> None:
        """
        Same treatment as _draw_circle — genuine image content, sized
        from the frame's own true pixel geometry and left for the
        ambient paint transform to pan/zoom — generalized to a rotated
        ellipse via painter.rotate() rather than drawEllipse's own
        (axis-aligned only) rect.
        """
        geometry = self._ellipse_geometry(kind, points, full_dims)
        if geometry is None:
            return
        center, rx_px, ry_px, rotation_deg = geometry
        full_w, full_h = full_dims

        center_point = self._to_point(rect, center)
        rx = rx_px * (rect.width() / full_w)
        ry = ry_px * (rect.height() / full_h)

        total_outline_width = line_width + outline_width * 2
        pattern = None if dashed else resolve_dash_pattern(dash_style, line_width)
        dash_cap = Qt.PenCapStyle.RoundCap if dash_style in ROUND_CAP_DASH_STYLES else Qt.PenCapStyle.FlatCap

        painter.save()
        painter.translate(center_point)
        painter.rotate(rotation_deg)
        ellipse_rect = QRectF(-rx, -ry, 2 * rx, 2 * ry)

        if fill_color is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(fill_color))
            painter.drawEllipse(ellipse_rect)

        outline = QPen(outline_color)
        outline.setWidthF(total_outline_width / stroke_scale)
        outline.setCapStyle(dash_cap)
        self._apply_dash(outline, total_outline_width, dashed, pattern)
        painter.setPen(outline)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(ellipse_rect)

        fill = QPen(line_color)
        fill.setWidthF(line_width / stroke_scale)
        fill.setCapStyle(dash_cap)
        self._apply_dash(fill, line_width, dashed, pattern)
        painter.setPen(fill)
        painter.drawEllipse(ellipse_rect)
        painter.restore()

    def _draw_circle_at(
        self,
        painter: QPainter,
        rect: QRect,
        center: tuple[float, float],
        radius_px: float,
        full_dims: tuple[int, int],
        stroke_scale: float,
        *,
        dashed: bool,
        line_color: QColor,
        line_width: float,
        outline_color: QColor,
        outline_width: float,
        dash_style: str,
        fill_color: QColor | None = None,
    ) -> None:
        """One circle by center-fraction and true-pixel radius — the shared outline/fill pass _draw_annulus and _draw_two_circle build on, mirroring _draw_circle's own pen treatment."""
        full_w, full_h = full_dims
        if full_w <= 0 or full_h <= 0 or radius_px <= 0:
            return
        center_point = self._to_point(rect, center)
        rx = radius_px * (rect.width() / full_w)
        ry = radius_px * (rect.height() / full_h)
        if fill_color is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(fill_color))
            painter.drawEllipse(center_point, rx, ry)
        total_outline_width = line_width + outline_width * 2
        pattern = None if dashed else resolve_dash_pattern(dash_style, line_width)
        dash_cap = Qt.PenCapStyle.RoundCap if dash_style in ROUND_CAP_DASH_STYLES else Qt.PenCapStyle.FlatCap
        outline = QPen(outline_color)
        outline.setWidthF(total_outline_width / stroke_scale)
        outline.setCapStyle(dash_cap)
        self._apply_dash(outline, total_outline_width, dashed, pattern)
        painter.setPen(outline)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center_point, rx, ry)
        fill = QPen(line_color)
        fill.setWidthF(line_width / stroke_scale)
        fill.setCapStyle(dash_cap)
        self._apply_dash(fill, line_width, dashed, pattern)
        painter.setPen(fill)
        painter.drawEllipse(center_point, rx, ry)

    def _fill_polygon(
        self, painter: QPainter, rect: QRect, verts: list[tuple[float, float]], fill_color: QColor
    ) -> None:
        path = QPainterPath()
        path.moveTo(self._to_point(rect, verts[0]))
        for v in verts[1:]:
            path.lineTo(self._to_point(rect, v))
        path.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(fill_color))
        painter.drawPath(path)

    def _draw_annulus(
        self,
        painter: QPainter,
        rect: QRect,
        kind: str,
        points: tuple[tuple[float, float], ...],
        stroke_scale: float,
        full_dims: tuple[int, int],
        *,
        dashed: bool,
        line_color: QColor,
        line_width: float,
        outline_color: QColor,
        outline_width: float,
        dash_style: str,
        fill_color: QColor | None = None,
    ) -> None:
        entry = DEFAULT_REGISTRY.get(kind)
        geometry = entry.annulus_geometry(points, full_dims) if entry is not None and entry.annulus_geometry is not None else None
        if geometry is None:
            return
        center, outer_r, inner_r = geometry
        if fill_color is not None:
            full_w, full_h = full_dims
            cp = self._to_point(rect, center)
            orx, ory = outer_r * (rect.width() / full_w), outer_r * (rect.height() / full_h)
            irx, iry = inner_r * (rect.width() / full_w), inner_r * (rect.height() / full_h)
            # Odd-even fill rule leaves the inner disc hollow, filling only the ring.
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.OddEvenFill)
            path.addEllipse(cp, orx, ory)
            path.addEllipse(cp, irx, iry)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(fill_color))
            painter.drawPath(path)
        for radius_px in (outer_r, inner_r):
            self._draw_circle_at(
                painter, rect, center, radius_px, full_dims, stroke_scale,
                dashed=dashed, line_color=line_color, line_width=line_width,
                outline_color=outline_color, outline_width=outline_width, dash_style=dash_style,
            )

    def _draw_two_circle(
        self,
        painter: QPainter,
        rect: QRect,
        kind: str,
        points: tuple[tuple[float, float], ...],
        stroke_scale: float,
        full_dims: tuple[int, int],
        *,
        dashed: bool,
        line_color: QColor,
        line_width: float,
        outline_color: QColor,
        outline_width: float,
        dash_style: str,
        fill_color: QColor | None = None,
    ) -> None:
        entry = DEFAULT_REGISTRY.get(kind)
        if entry is None or entry.two_circle_partial is None:
            return
        circles = entry.two_circle_partial(points, full_dims)
        for center, radius_px in circles:
            self._draw_circle_at(
                painter, rect, center, radius_px, full_dims, stroke_scale,
                dashed=dashed, line_color=line_color, line_width=line_width,
                outline_color=outline_color, outline_width=outline_width, dash_style=dash_style, fill_color=fill_color,
            )
        # A solid line between the two centers — the distance the tag
        # reports — once both circles exist.
        if len(circles) >= 2:
            self._draw_polyline(
                painter, rect, (circles[0][0], circles[1][0]), stroke_scale, dashed=dashed,
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
        cap_size: float = 1.0,
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
        if dashed:
            p1 = self._to_point(rect, start)
            p2 = self._to_point(rect, end)
            lw = line_width / stroke_scale
            total_outline_width = line_width + outline_width * 2
            pattern = None

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

        outline_path, fill_path = self._stroke_paths(
            rect, start, end, stroke_scale,
            line_color=line_color, line_width=line_width, outline_color=outline_color, outline_width=outline_width,
            dash_style=dash_style, start_cap=start_cap, end_cap=end_cap, cap_size=cap_size,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(outline_color))
        painter.drawPath(outline_path)
        painter.setBrush(QBrush(line_color))
        painter.drawPath(fill_path)

    def _stroke_paths(
        self,
        rect: QRect,
        start: tuple[float, float],
        end: tuple[float, float],
        stroke_scale: float,
        *,
        line_color: QColor = OVERLAY_LINE_COLOR,
        line_width: float = OVERLAY_LINE_WIDTH,
        outline_color: QColor = OVERLAY_OUTLINE_COLOR,
        outline_width: float = OVERLAY_OUTLINE_WIDTH,
        dash_style: str = "solid",
        start_cap: str = "curved",
        end_cap: str = "curved",
        cap_size: float = 1.0,
    ) -> tuple[QPainterPath, QPainterPath]:
        """
        (outline path, fill path) for one finalized (non-preview) segment
        — everything ``_draw_stroke``'s non-dashed branch used to paint
        immediately, now just returned so ``_draw_polyline`` can union
        several segments' paths into one before painting (see its own
        docstring for why that matters at interior joints).
        """
        p1 = self._to_point(rect, start)
        p2 = self._to_point(rect, end)
        lw = line_width / stroke_scale
        ow = outline_width / stroke_scale
        total_outline_width = line_width + outline_width * 2
        pattern = resolve_dash_pattern(dash_style, line_width)

        body_p1 = self._point_along(p1, p2, self._cap_reach(start_cap, lw, stroke_scale, cap_size))
        body_p2 = self._point_along(p2, p1, self._cap_reach(end_cap, lw, stroke_scale, cap_size))

        # Qt dash patterns are in multiples of the stroking width, so
        # each pass normalizes pattern's screen-pixel targets against its
        # own (unscaled) width — matching _apply_dash's math — to keep
        # the outline and fill dashes aligned despite their different
        # widths.
        outline_pattern = [value / total_outline_width for value in pattern] if pattern else None
        fill_pattern = [value / line_width for value in pattern] if pattern else None
        round_caps = dash_style in ROUND_CAP_DASH_STYLES
        outline_path = self._stroke_path(body_p1, body_p2, total_outline_width / stroke_scale, outline_pattern, round_caps)
        fill_path = self._stroke_path(body_p1, body_p2, lw, fill_pattern, round_caps)
        for origin, tip, cap in ((p2, p1, start_cap), (p1, p2, end_cap)):
            cap_outline, cap_fill = self._cap_shapes(origin, tip, cap, lw, ow, stroke_scale, cap_size)
            if cap_outline is not None:
                outline_path = outline_path.united(cap_outline)
                fill_path = fill_path.united(cap_fill)

        return outline_path, fill_path

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

    @staticmethod
    def _arrow_open_miter_overhang(lw: float, stroke_scale: float, cap_size: float = 1.0) -> float:
        """
        How far a mitered join of the two "arrow_open" barbs pokes out
        past their shared vertex — shared by ``_cap_reach`` (shortens the
        shaft by exactly this much) and ``_cap_shapes`` (shifts the barbs
        back by exactly this much) so the two stay in sync: shift the
        barbs alone and the shaft's unshortened flat end squares off the
        now-receded point; shorten the shaft by more than this (e.g. the
        full arrow length, as "arrow"'s solid triangle needs) and a gap
        opens up, since the barbs — unlike a solid triangle's full-width
        base — are only ``lw`` wide even at their own base, nowhere near
        the shaft's width once you back off that far.

        Derived from the standard miter-length formula, miterLength =
        (strokeWidth/2) / sin(theta) where theta is the half-angle
        between the two joined segments and their bisector — here
        sin(theta) = arrow_half/barb_len from the barb triangle's own
        sides (see arrow_dims).
        """
        arrow_len, arrow_half = arrow_dims(lw, stroke_scale, cap_size)
        barb_len = math.hypot(arrow_len, arrow_half)
        return (lw / 2) * barb_len / arrow_half

    @classmethod
    def _cap_reach(cls, cap: str, lw: float, stroke_scale: float, cap_size: float = 1.0) -> float:
        """
        How far a cap's own shape extends back from the true endpoint —
        the body is shortened by this much so its flat end sits at the
        arrowhead's base (or, for "arrow_open", just behind where its
        barbs' miter join starts to flare wider than the shaft — see
        ``_arrow_open_miter_overhang``) instead of poking past the tip,
        or squaring it off, once the two are unioned/overlaid. Every
        other style unions its shape directly onto the shaft's own flat,
        unshortened end: "square", "curved", and "bracket" are no wider
        than the shaft (their disc/rect/tick never reaches past the true
        endpoint).
        """
        # Solid heads shorten the shaft to their own back edge, minus a
        # small *lw* overlap so the shaft fill runs a hair into the head
        # rather than merely meeting it at a seam (which left a thin
        # outline-colored line between shaft and head).
        if cap == "arrow":
            return max(0.0, arrow_dims(lw, stroke_scale, cap_size)[0] - lw)
        if cap == "arrow_diamond":
            return max(0.0, arrow_dims(lw, stroke_scale, cap_size)[0] - lw)
        if cap == "arrow_circle":
            return max(0.0, 2 * circle_head_radius(lw, stroke_scale, cap_size) - lw)
        if cap == "arrow_open":
            return cls._arrow_open_miter_overhang(lw, stroke_scale, cap_size)
        return 0.0

    @staticmethod
    def _stroke_path(
        p1: QPointF, p2: QPointF, width: float, dash_pattern: list[float] | None = None, round_caps: bool = False
    ) -> QPainterPath:
        """A filled flat-ended band from *p1* to *p2*, *width* wide (or that band split into dashes, given a *dash_pattern* already normalized to multiples of *width*) — the shaft as a shape rather than a stroked QPen line, so it can be unioned with a cap's own shape into one seamless path. Flat rather than round-ended so it never bulges past a true endpoint on its own; "curved" gets its round look from an explicit disc unioned in by _cap_shapes instead, the same way "square"/"arrow" get theirs. *round_caps* rounds each dash instead (turning a near-zero on-segment into a circular dot — see ROUND_CAP_DASH_STYLES)."""
        line = QPainterPath()
        line.moveTo(p1)
        line.lineTo(p2)
        stroker = QPainterPathStroker()
        stroker.setWidth(width)
        if round_caps:
            stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
            if dash_pattern:
                stroker.setDashPattern(dash_pattern)
            return stroker.createStroke(line)
        # A dashed shaft needs each dash's own ends covered too — a
        # FlatCap here leaves the fill dash's flat end flush with the
        # outline dash's identical flat end, so the fill covers the
        # outline everywhere except the dash's long sides. SquareCap
        # over-extends both passes equally, restoring the same per-dash
        # outline a circle's plain (default-SquareCap) QPen already gets.
        # Solid shafts must stay FlatCap — _cap_shapes unions its own
        # end-cap shapes exactly at the true endpoint, which depends on
        # the shaft ending flat there.
        stroker.setCapStyle(Qt.PenCapStyle.SquareCap if dash_pattern else Qt.PenCapStyle.FlatCap)
        if dash_pattern:
            stroker.setDashPattern(dash_pattern)
        return stroker.createStroke(line)

    def _draw_preview_caps(
        self,
        painter: QPainter,
        rect: QRect,
        points: tuple[tuple[float, float], ...],
        stroke_scale: float,
        start_cap: str,
        end_cap: str,
        cap_size: float,
    ) -> None:
        """Draw just the end-cap shapes (solid) on a dashed placement preview, so an Arrow's head is visible while placing — see _draw_draft."""
        if start_cap == "curved" and end_cap == "curved":
            return
        lw = OVERLAY_LINE_WIDTH / stroke_scale
        ow = OVERLAY_OUTLINE_WIDTH / stroke_scale
        p_first = self._to_point(rect, points[0])
        p_last = self._to_point(rect, points[-1])
        for origin, tip, cap in ((p_last, p_first, start_cap), (p_first, p_last, end_cap)):
            if cap == "curved":
                continue
            cap_outline, cap_fill = self._cap_shapes(origin, tip, cap, lw, ow, stroke_scale, cap_size)
            if cap_fill is None:
                continue
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(OVERLAY_OUTLINE_COLOR))
            painter.drawPath(cap_outline)
            painter.setBrush(QBrush(OVERLAY_LINE_COLOR))
            painter.drawPath(cap_fill)

    @classmethod
    def _cap_shapes(
        cls, origin: QPointF, tip: QPointF, cap: str, lw: float, ow: float, stroke_scale: float, cap_size: float = 1.0
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
        elif cap == "bracket":
            fill_shape = bracket_path(lw, stroke_scale, cap_size)
            outline_shape = cls._inflate(fill_shape, ow, Qt.PenJoinStyle.MiterJoin)
        elif cap == "arrow":
            fill_shape = arrow_head_path(lw, stroke_scale, cap_size)
            outline_shape = cls._inflate(fill_shape, ow, Qt.PenJoinStyle.RoundJoin)
        elif cap == "arrow_diamond":
            fill_shape = diamond_head_path(lw, stroke_scale, cap_size)
            outline_shape = cls._inflate(fill_shape, ow, Qt.PenJoinStyle.MiterJoin)
        elif cap == "arrow_circle":
            fill_shape = circle_head_path(lw, stroke_scale, cap_size)
            outline_shape = cls._inflate(fill_shape, ow, Qt.PenJoinStyle.RoundJoin)
        elif cap == "arrow_open":
            barbs = open_arrow_barbs_path(lw, stroke_scale, cap_size)

            # Stroking the barbs with a miter join pushes the apex's
            # sharp point past the true endpoint (the barb centerlines
            # meet at the origin, but the miter extends the outer edge
            # beyond it). Shift the barbs back by that overhang so the
            # visible tip lands on the endpoint, matching the solid
            # arrow's apex rather than poking past it. The shaft itself
            # is separately shortened by the exact same amount (see
            # _cap_reach/_arrow_open_miter_overhang) so its flat end
            # meets the barbs' own width exactly where they start to
            # flare — connected, with no gap, and without squaring off
            # the point.
            miter_overhang = cls._arrow_open_miter_overhang(lw, stroke_scale, cap_size)
            barbs.translate(-miter_overhang, 0)

            # MiterJoin (not RoundJoin) at the barbs' shared apex so the
            # tip comes to a sharp point like the solid arrow's, rather
            # than the rounded nub a round join leaves; the outer barb
            # ends stay round.
            fill_stroker = QPainterPathStroker()
            fill_stroker.setWidth(lw)
            fill_stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
            fill_stroker.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            fill_stroker.setMiterLimit(10.0)
            fill_shape = fill_stroker.createStroke(barbs)

            outline_stroker = QPainterPathStroker()
            outline_stroker.setWidth(lw + ow * 2)
            outline_stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
            outline_stroker.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            outline_stroker.setMiterLimit(10.0)
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

    # ------------------------------------------------------------------
    # Import/export — see measurement_io.py for the actual document
    # format and validation; this just wires it to the active source's
    # own measurement list (live or loaded, same split as everywhere
    # else — see MeasurementOverlay._loaded_active).
    # ------------------------------------------------------------------

    def export_measurements(self) -> dict:
        """A plain, JSON-serializable document of the active source's placed measurements."""
        return serialize_measurements(self._overlay.measurements)

    def import_measurements(self, data: dict, *, replace: bool = True) -> DeserializeResult:
        """
        Load measurements from a document produced by ``export_measurements``
        into the active source's own list. Invalid entries are skipped
        and reported in the returned result's ``warnings`` rather than
        raising — see ``measurement_io.deserialize_measurements``.
        *replace* clears the active list first; pass False to append
        instead.
        """
        result = deserialize_measurements(data, DEFAULT_REGISTRY)
        measurements = self._overlay.measurements
        if replace:
            measurements.clear()
        measurements.extend(Measurement(kind, points, meta) for kind, points, meta in result.entries)
        self._repaint()
        return result

    def export_measurements_to_file(self, path: str) -> None:
        save_measurements_to_file(path, self._overlay.measurements)

    def import_measurements_from_file(self, path: str, *, replace: bool = True) -> DeserializeResult:
        result = load_measurements_from_file(path, DEFAULT_REGISTRY)
        measurements = self._overlay.measurements
        if replace:
            measurements.clear()
        measurements.extend(Measurement(kind, points, meta) for kind, points, meta in result.entries)
        self._repaint()
        return result