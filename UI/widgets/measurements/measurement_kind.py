from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from UI.widgets.measurements.lines import CALIBRATION_KIND

Point2D = tuple[float, float]
CircleGeometry = tuple[Point2D, float]

# Points closer together than this (in frame fractions) are treated as a
# single point rather than a real second click — mirrors
# MeasurementOverlay's own _DEGENERATE_EPSILON, kept local here since
# nothing outside this module's own resolve functions needs it.
_DEGENERATE_EPSILON = 1e-12


def _collinear(p1: Point2D, p2: Point2D, p3: Point2D) -> bool:
    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    return abs(cross) < _DEGENERATE_EPSILON


def _circumcircle(p1: Point2D, p2: Point2D, p3: Point2D) -> CircleGeometry | None:
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


def _default_move_point(points: list[Point2D], index: int, new_point: Point2D) -> list[Point2D]:
    """Just move the dragged point — correct for a polyline's own points, a circle's non-center edge points, and a lone "Point"."""
    points = list(points)
    points[index] = new_point
    return points


@dataclass(frozen=True)
class MeasurementKind:
    """
    Everything that's specific to one measurement kind, looked up by
    ``MeasurementOverlay`` instead of the string/tuple-membership if/elif
    chains this replaces (``_REQUIRED_POINTS``, ``PLACEABLE_KINDS``,
    ``LINE_KINDS``/``_CIRCLE_KINDS`` membership, and the per-kind
    branches inside ``_resolve_measurement``/``_draw_measurement``/
    ``_hit_test_proximity``/``_move_point``/``_measurement_label``).

    ``category`` picks which of ``MeasurementOverlay``'s existing
    rendering/hit-test families a kind uses — "line" (a capped, possibly
    dashed polyline via ``_draw_polyline``), "circle" (via
    ``_draw_circle``, using ``circle_geometry`` to get a center/radius
    from the kind's own raw points), or "point" (a filled dot via
    ``_draw_point_marker``). This is deliberately composition over a
    ``LineKind``/``CircleKind`` subclass hierarchy: the actual pixel-
    drawing code is the same regardless of which exact kind is being
    drawn, so a kind just needs to say which family it belongs to and
    supply the handful of things that really are kind-specific —
    ``resolve`` (raw clicks -> stored points, or None if degenerate),
    and optionally ``move_point`` (drag semantics) and
    ``circle_geometry`` (required for category "circle").

    A wholly new shape family that doesn't fit line/circle/point (e.g. a
    filled polygon) isn't served by this dataclass alone — it would need
    a new category handled explicitly in ``MeasurementOverlay``'s
    dispatch methods, the same way "line"/"circle"/"point" are today.
    """

    name: str
    required_points: int | None  # None = unbounded (e.g. Arbitrary Line)
    category: str  # "line" | "circle" | "point"
    resolve: Callable[[list[Point2D]], tuple[Point2D, ...] | None]
    move_point: Callable[[list[Point2D], int, Point2D], list[Point2D]] = _default_move_point
    circle_geometry: Callable[[tuple[Point2D, ...], tuple[int, int]], CircleGeometry | None] | None = None
    has_label: bool = True
    # MeasurementMeta field overrides applied on top of the sidebar's
    # default meta as a measurement of this kind is placed (see
    # MeasurementOverlay._resolve_meta) — how "Arrow"/"Bracket" are
    # "just the line type preset to X with a different icon": same
    # placement/resolve/rendering as any other line, just with these
    # fields forced regardless of what the Customize Measurements panel
    # currently has set.
    meta_preset: dict[str, object] | None = None


class MeasurementKindRegistry:
    """Looked up by name from MeasurementOverlay wherever a kind-specific if/elif branch used to live."""

    def __init__(self) -> None:
        self._kinds: dict[str, MeasurementKind] = {}

    def register(self, kind: MeasurementKind) -> None:
        self._kinds[kind.name] = kind

    def get(self, name: str | None) -> MeasurementKind | None:
        return self._kinds.get(name) if name is not None else None

    def __contains__(self, name: object) -> bool:
        return name in self._kinds

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._kinds)


def _point_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    return (points[0],) if points else None


def _two_point_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 2 or points[0] == points[1]:
        return None
    return points[0], points[1]


def _horizontal_line_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 2 or points[0] == points[1]:
        return None
    y = points[0][1]
    x0, x1 = sorted((points[0][0], points[1][0]))
    return (x0, y), (x1, y)


def _vertical_line_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 2 or points[0] == points[1]:
        return None
    x = points[0][0]
    y0, y1 = sorted((points[0][1], points[1][1]))
    return (x, y0), (x, y1)


def _arbitrary_line_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    return tuple(points) if len(points) >= 2 else None


def _three_point_circle_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 3:
        return None
    p1, p2, p3 = points[0], points[1], points[2]
    if p1 == p2 or p2 == p3 or p1 == p3 or _collinear(p1, p2, p3):
        return None
    return p1, p2, p3


def _horizontal_line_move(points: list[Point2D], index: int, new_point: Point2D) -> list[Point2D]:
    if len(points) != 2:
        return _default_move_point(points, index, new_point)
    y = new_point[1]
    other = points[1 - index]
    x0, x1 = sorted((new_point[0], other[0]))
    return [(x0, y), (x1, y)]


def _vertical_line_move(points: list[Point2D], index: int, new_point: Point2D) -> list[Point2D]:
    if len(points) != 2:
        return _default_move_point(points, index, new_point)
    x = new_point[0]
    other = points[1 - index]
    y0, y1 = sorted((new_point[1], other[1]))
    return [(x, y0), (x, y1)]


def _radius_circle_move(points: list[Point2D], index: int, new_point: Point2D) -> list[Point2D]:
    if index != 0 or len(points) != 2:
        return _default_move_point(points, index, new_point)
    # Dragging the center translates the whole circle rather than just
    # the center point, so the radius doesn't change out from under the
    # (undragged) edge point.
    dx = new_point[0] - points[0][0]
    dy = new_point[1] - points[0][1]
    return [new_point, (points[1][0] + dx, points[1][1] + dy)]


def _circle_geometry_from_center_edge(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> CircleGeometry | None:
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0 or len(points) < 2:
        return None
    center_px = (points[0][0] * full_w, points[0][1] * full_h)
    edge_px = (points[1][0] * full_w, points[1][1] * full_h)
    radius = math.hypot(edge_px[0] - center_px[0], edge_px[1] - center_px[1])
    return (points[0], radius) if radius > 0 else None


def _circle_geometry_from_diameter(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> CircleGeometry | None:
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0 or len(points) < 2:
        return None
    a_px = (points[0][0] * full_w, points[0][1] * full_h)
    b_px = (points[1][0] * full_w, points[1][1] * full_h)
    radius = math.hypot(b_px[0] - a_px[0], b_px[1] - a_px[1]) / 2
    if radius <= 0:
        return None
    center_px = ((a_px[0] + b_px[0]) / 2, (a_px[1] + b_px[1]) / 2)
    return (center_px[0] / full_w, center_px[1] / full_h), radius


def _circle_geometry_from_three_points(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> CircleGeometry | None:
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0 or len(points) < 3:
        return None

    def to_px(p: Point2D) -> Point2D:
        return p[0] * full_w, p[1] * full_h

    result = _circumcircle(to_px(points[0]), to_px(points[1]), to_px(points[2]))
    if result is None:
        return None
    center_px, radius = result
    return (center_px[0] / full_w, center_px[1] / full_h), radius


DEFAULT_REGISTRY = MeasurementKindRegistry()
DEFAULT_REGISTRY.register(MeasurementKind(
    name="Point", required_points=1, category="point", resolve=_point_resolve,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    name="Horizontal Line", required_points=2, category="line",
    resolve=_horizontal_line_resolve, move_point=_horizontal_line_move,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    name="Vertical Line", required_points=2, category="line",
    resolve=_vertical_line_resolve, move_point=_vertical_line_move,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    name="Arbitrary Line", required_points=None, category="line", resolve=_arbitrary_line_resolve,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # A plain 2-point line whose only difference from "Arbitrary Line" is
    # its placement default: end_cap forced to "arrow" so a single click-
    # click gives a ready-made arrow annotation without visiting the
    # customize menu — see MeasurementKind.meta_preset.
    name="Arrow", required_points=2, category="line", resolve=_two_point_resolve,
    meta_preset={"line_start_cap": "curved", "line_end_cap": "arrow"},
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # Same idea as "Arrow" — a plain 2-point line preset to bracket caps
    # on both ends, for a ready-made dimension-line-style annotation.
    name="Bracket", required_points=2, category="line", resolve=_two_point_resolve,
    meta_preset={"line_start_cap": "bracket", "line_end_cap": "bracket"},
))
DEFAULT_REGISTRY.register(MeasurementKind(
    name="Radius Circle", required_points=2, category="circle",
    resolve=_two_point_resolve, move_point=_radius_circle_move, circle_geometry=_circle_geometry_from_center_edge,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    name="Diameter", required_points=2, category="circle",
    resolve=_two_point_resolve, circle_geometry=_circle_geometry_from_diameter,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    name="3 Point Circle", required_points=3, category="circle",
    resolve=_three_point_circle_resolve, circle_geometry=_circle_geometry_from_three_points,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # Placed the same way as a 2-point line but never becomes a real
    # Measurement (see MeasurementOverlay._calibration_line) and never
    # gets the generic title/length tag — see start_calibration_placement.
    name=CALIBRATION_KIND, required_points=2, category="line", resolve=_two_point_resolve, has_label=False,
))
