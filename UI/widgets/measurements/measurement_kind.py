from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from UI.widgets.measurements.lines import CALIBRATION_KIND

Point2D = tuple[float, float]
CircleGeometry = tuple[Point2D, float]
# (center, rx_px, ry_px, rotation_deg) — rx points along *rotation_deg*
# from the +x axis, ry perpendicular to it.
EllipseGeometry = tuple[Point2D, float, float, float]

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
    category: str  # "line" | "circle" | "point" | "ellipse" | "angle" | "arc"
    resolve: Callable[[list[Point2D]], tuple[Point2D, ...] | None]
    move_point: Callable[[list[Point2D], int, Point2D], list[Point2D]] = _default_move_point
    circle_geometry: Callable[[tuple[Point2D, ...], tuple[int, int]], CircleGeometry | None] | None = None
    ellipse_geometry: Callable[[tuple[Point2D, ...], tuple[int, int]], EllipseGeometry | None] | None = None
    has_label: bool = True
    # MeasurementMeta field overrides applied on top of the sidebar's
    # default meta as a measurement of this kind is placed (see
    # MeasurementOverlay._resolve_meta) — how "Arrow"/"Bracket" are
    # "just the line type preset to X with a different icon": same
    # placement/resolve/rendering as any other line, just with these
    # fields forced regardless of what the Customize Measurements panel
    # currently has set.
    meta_preset: dict[str, object] | None = None
    # "angle" category only: the (possibly partial, while still placing
    # — see MeasurementOverlay._draw_draft) line segments to draw/hit-
    # test, and the angle (degrees) and tag anchor those points describe.
    # Segments needn't be consecutive pairs of *points* — "4 Point Angle"
    # draws two independent, disconnected segments — which is why this
    # isn't just handled generically the way "line"'s always-consecutive
    # polyline is.
    segment_pairs: Callable[[tuple[Point2D, ...]], list[tuple[Point2D, Point2D]]] | None = None
    angle_value: Callable[[tuple[Point2D, ...], tuple[int, int]], float | None] | None = None
    angle_anchor: Callable[[tuple[Point2D, ...]], Point2D] | None = None


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


# ----------------------------------------------------------------------
# Ellipses — "3 Point Ellipse" (two points mark the minor axis, a third
# marks the major-axis extent) and "5 Point Ellipse" (five points on the
# boundary, fit to a general conic). Ellipse-ness (bounded, real,
# non-degenerate) is affine-invariant — it survives the frame's own
# anisotropic aspect ratio — so ``resolve`` can validate it directly in
# fraction space without needing ``full_dims``; only the actual on-screen
# parameters (computed by ``ellipse_geometry``, which does get
# ``full_dims``) need true pixel space.
# ----------------------------------------------------------------------


def _three_point_ellipse_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 3:
        return None
    p0, p1, p2 = points[0], points[1], points[2]
    if p0 == p1 or _collinear(p0, p1, p2):
        return None
    return p0, p1, p2


def _ellipse_geometry_three_point(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> EllipseGeometry | None:
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0 or len(points) < 3:
        return None

    def to_px(p: Point2D) -> Point2D:
        return p[0] * full_w, p[1] * full_h

    a, b, c = to_px(points[0]), to_px(points[1]), to_px(points[2])
    center = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    ux, uy = b[0] - a[0], b[1] - a[1]
    minor_len = math.hypot(ux, uy)
    if minor_len < _DEGENERATE_EPSILON:
        return None
    # Unit minor-axis direction (along p0->p1) and its perpendicular
    # (the major-axis direction, along which p2's offset from center is
    # measured — see the class docstring's "minor axis with two points,
    # then the major axis with one point").
    ux, uy = ux / minor_len, uy / minor_len
    vx, vy = -uy, ux
    cx, cy = c[0] - center[0], c[1] - center[1]
    major_r = abs(cx * vx + cy * vy)
    minor_r = minor_len / 2
    if major_r < _DEGENERATE_EPSILON:
        return None
    rotation = math.degrees(math.atan2(vy, vx))
    return (center[0] / full_w, center[1] / full_h), major_r, minor_r, rotation


def _conic_fit(points_px: list[Point2D]) -> tuple[float, float, float, float, float, float] | None:
    """
    General conic ``Ax²+Bxy+Cy²+Dx+Ey+F=0`` through exactly 5
    points, via the null space of the 5×6 design matrix (each row
    ``[x², xy, y², x, y, 1]``) — the coefficients up to a
    shared scale factor, found as the right singular vector for the
    smallest singular value. None if the points don't determine a
    (numerically) unique conic — near-duplicate or otherwise degenerate
    points leave the design matrix without a clean 1-D null space.
    """
    if len(points_px) != 5:
        return None
    rows = [[x * x, x * y, y * y, x, y, 1.0] for x, y in points_px]
    matrix = np.array(rows, dtype=float)
    norm = np.linalg.norm(matrix)
    if norm < _DEGENERATE_EPSILON:
        return None
    matrix = matrix / norm  # keeps singular values in a comparable, scale-independent range
    try:
        _, singular_values, vh = np.linalg.svd(matrix)
    except np.linalg.LinAlgError:
        return None
    # svd on a 5x6 matrix returns only 5 singular values (min(5,6)) — the
    # null-space vector (the conic's coefficients, up to scale) is
    # ``vh[5]``, the one row of ``vh`` with *no* corresponding singular
    # value at all, and it's meaningful only when the matrix has full
    # row rank 5 (a genuine 1-D null space) — i.e. its smallest reported
    # singular value isn't itself close to zero. If it is, the 5 points
    # don't pin down a single conic (duplicates, or an otherwise
    # degenerate configuration) and ``vh[5]`` would just be numerical
    # noise rather than the real null direction.
    if len(singular_values) < 5 or singular_values[-1] < 1e-9 * max(singular_values[0], 1e-12):
        return None
    a, b, c, d, e, f = vh[5]
    return float(a), float(b), float(c), float(d), float(e), float(f)


def _ellipse_params_from_conic(
    a: float, b: float, c: float, d: float, e: float, f: float
) -> tuple[Point2D, float, float, float] | None:
    """
    Center, semi-axis lengths, and rotation of the real conic
    ``Ax²+Bxy+Cy²+Dx+Ey+F=0``, or None if it isn't a genuine
    bounded ellipse (a parabola/hyperbola, an imaginary "ellipse" with no
    real points, or a degenerate single point/line pair). Standard
    conic-to-ellipse reduction: recenter to kill the linear terms, then
    diagonalize the remaining quadratic form.
    """
    discriminant = b * b - 4 * a * c
    if discriminant >= 0:
        return None  # parabola or hyperbola, not an ellipse

    # Center: where both partial derivatives vanish — 2Ax+By+D=0, Bx+2Cy+E=0.
    det = 4 * a * c - b * b
    if abs(det) < _DEGENERATE_EPSILON:
        return None
    x0 = (b * e - 2 * c * d) / det
    y0 = (b * d - 2 * a * e) / det

    # Value of the conic at its own center — the "radius" scale factor;
    # zero means a degenerate single point instead of a real ellipse.
    f0 = a * x0 * x0 + b * x0 * y0 + c * y0 * y0 + d * x0 + e * y0 + f
    if abs(f0) < _DEGENERATE_EPSILON:
        return None

    # Eigen-decomposition of the symmetric [[A, B/2], [B/2, C]] quadratic
    # form — closed form for a 2×2 symmetric matrix, so this needs
    # no general eigensolver.
    trace = a + c
    diff = a - c
    radius_term = math.hypot(diff, b)
    lambda1 = (trace + radius_term) / 2
    lambda2 = (trace - radius_term) / 2
    if abs(lambda1) < _DEGENERATE_EPSILON or abs(lambda2) < _DEGENERATE_EPSILON:
        return None

    axis1_sq = -f0 / lambda1
    axis2_sq = -f0 / lambda2
    if axis1_sq <= 0 or axis2_sq <= 0:
        return None
    axis1, axis2 = math.sqrt(axis1_sq), math.sqrt(axis2_sq)

    # Rotation of the eigenvector for lambda1 (axis1's own direction) —
    # the standard closed-form principal-axis angle for a symmetric 2x2
    # quadratic form. A harmless +-180 degree ambiguity in the result
    # (atan2's branch choice) never matters: an ellipse is unchanged by
    # a 180 degree rotation.
    angle = math.atan2(b, a - c) / 2
    rx, ry, rotation = (axis1, axis2, angle) if axis1 >= axis2 else (axis2, axis1, angle + math.pi / 2)
    return (x0, y0), rx, ry, math.degrees(rotation)


def _five_point_ellipse_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 5:
        return None
    pts = points[:5]
    conic = _conic_fit(pts)
    if conic is None:
        return None
    if _ellipse_params_from_conic(*conic) is None:
        return None
    return tuple(pts)


def _ellipse_geometry_five_point(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> EllipseGeometry | None:
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0 or len(points) < 5:
        return None

    def to_px(p: Point2D) -> Point2D:
        return p[0] * full_w, p[1] * full_h

    conic = _conic_fit([to_px(p) for p in points[:5]])
    if conic is None:
        return None
    params = _ellipse_params_from_conic(*conic)
    if params is None:
        return None
    center_px, rx, ry, rotation = params
    return (center_px[0] / full_w, center_px[1] / full_h), rx, ry, rotation


# ----------------------------------------------------------------------
# Angles — "3 Point Angle" (a vertex with two connected segments, like
# "Arbitrary Line" capped at 3 points) and "4 Point Angle" (two
# independent, disconnected segments). Both report the angle between
# their two segments' directions as the tag's value instead of a length.
# ----------------------------------------------------------------------


def _angle_between_vectors(u: Point2D, v: Point2D) -> float | None:
    """Unsigned angle in degrees (0-180) between *u* and *v*, or None if either is a zero vector."""
    mag = math.hypot(*u) * math.hypot(*v)
    if mag < _DEGENERATE_EPSILON:
        return None
    cos_angle = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / mag))
    return math.degrees(math.acos(cos_angle))


def _three_point_angle_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 3:
        return None
    p0, p1, p2 = points[0], points[1], points[2]
    if p0 == p1 or p1 == p2:
        return None
    return p0, p1, p2


def _three_point_angle_segments(points: tuple[Point2D, ...]) -> list[tuple[Point2D, Point2D]]:
    """Segments to draw for whatever's been placed so far — partial while still placing (see MeasurementKind.segment_pairs)."""
    pairs = []
    if len(points) >= 2:
        pairs.append((points[0], points[1]))
    if len(points) >= 3:
        pairs.append((points[1], points[2]))
    return pairs


def _three_point_angle_value(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> float | None:
    if len(points) < 3:
        return None
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0:
        return None

    def to_px(p: Point2D) -> Point2D:
        return p[0] * full_w, p[1] * full_h

    p0, p1, p2 = to_px(points[0]), to_px(points[1]), to_px(points[2])
    return _angle_between_vectors((p0[0] - p1[0], p0[1] - p1[1]), (p2[0] - p1[0], p2[1] - p1[1]))


def _three_point_angle_anchor(points: tuple[Point2D, ...]) -> Point2D:
    return points[1]  # the vertex


def _four_point_angle_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 4:
        return None
    p0, p1, p2, p3 = points[0], points[1], points[2], points[3]
    if p0 == p1 or p2 == p3:
        return None
    return p0, p1, p2, p3


def _four_point_angle_segments(points: tuple[Point2D, ...]) -> list[tuple[Point2D, Point2D]]:
    pairs = []
    if len(points) >= 2:
        pairs.append((points[0], points[1]))
    if len(points) >= 4:
        pairs.append((points[2], points[3]))
    return pairs


def _four_point_angle_value(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> float | None:
    if len(points) < 4:
        return None
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0:
        return None

    def to_px(p: Point2D) -> Point2D:
        return p[0] * full_w, p[1] * full_h

    p0, p1, p2, p3 = (to_px(p) for p in points[:4])
    return _angle_between_vectors((p1[0] - p0[0], p1[1] - p0[1]), (p3[0] - p2[0], p3[1] - p2[1]))


def _four_point_angle_anchor(points: tuple[Point2D, ...]) -> Point2D:
    xs = [p[0] for p in points[:4]]
    ys = [p[1] for p in points[:4]]
    return sum(xs) / len(xs), sum(ys) / len(ys)


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
    # First two points mark the minor axis (its endpoints), the third
    # marks how far the ellipse extends perpendicular to it.
    name="3 Point Ellipse", required_points=3, category="ellipse",
    resolve=_three_point_ellipse_resolve, ellipse_geometry=_ellipse_geometry_three_point,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # Five points on the boundary, fit to a general conic — rejected at
    # placement (resolve returns None) unless they actually describe a
    # real, bounded ellipse.
    name="5 Point Ellipse", required_points=5, category="ellipse",
    resolve=_five_point_ellipse_resolve, ellipse_geometry=_ellipse_geometry_five_point,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # A vertex (the 2nd point) with two connected segments — same
    # placement rhythm as "Arbitrary Line" but capped at 3 points —
    # showing the angle at the vertex instead of a length.
    name="3 Point Angle", required_points=3, category="angle",
    resolve=_three_point_angle_resolve, segment_pairs=_three_point_angle_segments,
    angle_value=_three_point_angle_value, angle_anchor=_three_point_angle_anchor,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # Two independent, disconnected segments — showing the angle
    # between them instead of a length.
    name="4 Point Angle", required_points=4, category="angle",
    resolve=_four_point_angle_resolve, segment_pairs=_four_point_angle_segments,
    angle_value=_four_point_angle_value, angle_anchor=_four_point_angle_anchor,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # Placed the same way as a 2-point line but never becomes a real
    # Measurement (see MeasurementOverlay._calibration_line) and never
    # gets the generic title/length tag — see start_calibration_placement.
    name=CALIBRATION_KIND, required_points=2, category="line", resolve=_two_point_resolve, has_label=False,
))
