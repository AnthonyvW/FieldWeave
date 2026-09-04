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
# (center, radius_px, start_deg, sweep_deg) — the arc runs from
# start_deg by sweep_deg (signed: positive/negative picks the
# direction), both measured via atan2(dy,dx) in the same pixel-space
# convention used everywhere else in this module.
ArcGeometry = tuple[Point2D, float, float, float]

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
    # Optional cleanup applied to the stored points once resolved and
    # after every drag — snaps a derived-line control point onto the line
    # actually drawn (e.g. a 4pt parallel's far point onto its projected,
    # parallel-locked end) so its anchor handle sits on the geometry
    # rather than where the raw click landed. Needs full_dims.
    snap_points: Callable[[tuple[Point2D, ...], tuple[int, int] | None], tuple[Point2D, ...]] | None = None
    circle_geometry: Callable[[tuple[Point2D, ...], tuple[int, int]], CircleGeometry | None] | None = None
    ellipse_geometry: Callable[[tuple[Point2D, ...], tuple[int, int]], EllipseGeometry | None] | None = None
    arc_geometry: Callable[[tuple[Point2D, ...], tuple[int, int]], ArcGeometry | None] | None = None
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
    # Also used by "line_pair" category kinds (the parallel/perpendicular
    # families) — their derived segments come from the first line rather
    # than straight from clicked points, which is why this exists as a
    # callback instead of consecutive-point pairing.
    # full_dims is None whenever true pixel geometry isn't available yet
    # (or isn't needed — most kinds ignore it entirely; the
    # perpendicular kinds need it, since "perpendicular" is an angle
    # concept sensitive to the frame's own aspect ratio).
    segment_pairs: Callable[[tuple[Point2D, ...], tuple[int, int] | None], list[tuple[Point2D, Point2D]]] | None = None
    # "line_pair" category only: dashed guide segments drawn alongside the
    # solid ones from segment_pairs — the connectors tying a parallel
    # pair together, or the midlines an "8pt Parallel" shows between its
    # two pairs. Never carry caps or a length label of their own.
    connector_segments: Callable[[tuple[Point2D, ...], tuple[int, int] | None], list[tuple[Point2D, Point2D]]] | None = None
    # "line_pair" category only: (tag anchor fraction, distance in true
    # pixels) the tag shows instead of a plain segment length — the
    # perpendicular gap between a parallel pair, say. Needs full_dims
    # since a distance/perpendicularity is aspect-ratio sensitive. When
    # None, a line_pair tags its first segment's own length instead.
    pair_distance: Callable[[tuple[Point2D, ...], tuple[int, int]], tuple[Point2D, float] | None] | None = None
    # Extra, non-interactive distance tags beyond the main one — each an
    # (anchor fraction, distance in true pixels) — e.g. every gap between
    # an arbitrary-parallel's lines, or each leg of a perpendicular pair.
    extra_measures: Callable[[tuple[Point2D, ...], tuple[int, int]], list[tuple[Point2D, float]]] | None = None
    # Unbounded kinds (required_points is None) only: the fewest points a
    # cancel (right-click) will keep as a finished measurement rather than
    # discarding — 2 for a plain polyline, more for kinds needing a first
    # full line before their extra points mean anything.
    min_points: int = 2
    # "curve" category only: sampled points (fraction space) along the
    # curve *points* describes, for however many control points have
    # been placed so far — see MeasurementOverlay's "curve" dispatch.
    curve_points: Callable[[tuple[Point2D, ...]], list[Point2D] | None] | None = None
    angle_value: Callable[[tuple[Point2D, ...], tuple[int, int]], float | None] | None = None
    angle_anchor: Callable[[tuple[Point2D, ...]], Point2D] | None = None
    # "polygon" category (rectangles, polygons): the ordered boundary
    # vertices (fraction space) — drawn as a closed, optionally filled
    # outline, tagged with the enclosed area. Some forms (rotated
    # rectangles, squares) need full_dims to keep their right angles /
    # equal sides true under the frame's aspect ratio.
    polygon_points: Callable[[tuple[Point2D, ...], tuple[int, int] | None], list[Point2D] | None] | None = None
    # "annulus" category: (center_fraction, outer_radius_px, inner_radius_px).
    annulus_geometry: Callable[[tuple[Point2D, ...], tuple[int, int]], tuple[Point2D, float, float] | None] | None = None
    # "two_circle" category: (center1, r1_px, center2, r2_px) — two
    # circles whose center-to-center distance is the measurement.
    two_circle_geometry: Callable[[tuple[Point2D, ...], tuple[int, int]], tuple[Point2D, float, Point2D, float] | None] | None = None
    # "two_circle" category: the circles determinable from the points so
    # far (1 while the second is still being placed, 2 once complete) — so
    # each circle previews as its own points arrive rather than the whole
    # thing waiting on the last click.
    two_circle_partial: Callable[[tuple[Point2D, ...], tuple[int, int]], list[CircleGeometry]] | None = None
    # True when segment_pairs' segments are actually consecutive (share
    # an endpoint, like "3 Point Angle"'s two legs at the vertex) — drawn
    # as one polyline (proper joint, matching "Arbitrary Line") instead
    # of independently-stroked segments, which would double up the
    # stroke/outline where they meet. False (e.g. "4 Point Angle") means
    # the segments are genuinely disconnected and must stay independent.
    connected_segments: bool = False


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


def _count_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    return tuple(points) if points else None


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
    # (a,b,c,d,e,f) come from _conic_fit's SVD null vector, unit-norm as
    # a whole 6-vector — but that doesn't bound (a,b,c) alone away from
    # tiny values: for points given in real pixel coordinates (thousands,
    # not the single digits/tens a quick sanity check tends to use),
    # (a,b,c) shrink roughly as 1/L^2 of the points' own coordinate
    # magnitude L while f stays near 1. Renormalizing so the quadratic
    # part (a,b,c) alone is unit-magnitude makes every check below
    # (calibrated assuming a/b/c ~ O(1)) behave the same regardless of
    # the input's coordinate scale — without it, real (large-coordinate)
    # clicks were spuriously rejected as "degenerate" by the absolute
    # epsilon checks that used to follow.
    quad_scale = math.sqrt(a * a + b * b + c * c)
    if quad_scale < _DEGENERATE_EPSILON:
        return None
    a, b, c, d, e, f = a / quad_scale, b / quad_scale, c / quad_scale, d / quad_scale, e / quad_scale, f / quad_scale

    discriminant = b * b - 4 * a * c
    if discriminant >= 0:
        return None  # parabola or hyperbola, not an ellipse

    # Center: where both partial derivatives vanish — 2Ax+By+D=0, Bx+2Cy+E=0.
    det = 4 * a * c - b * b
    if abs(det) < _DEGENERATE_EPSILON:
        return None
    x0 = (b * e - 2 * c * d) / det
    y0 = (b * d - 2 * a * e) / det

    # Value of the conic at its own center — the "radius" scale factor.
    # Unlike (a,b,c), this is NOT checked against a fixed epsilon here:
    # its natural magnitude scales with x0/y0 (real pixel coordinates,
    # so easily in the thousands), for which a fixed absolute threshold
    # would be meaningless. A degenerate (near-zero) value is instead
    # caught below, where it forces axis1_sq/axis2_sq non-positive.
    f0 = a * x0 * x0 + b * x0 * y0 + c * y0 * y0 + d * x0 + e * y0 + f

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
    if axis1_sq <= _DEGENERATE_EPSILON or axis2_sq <= _DEGENERATE_EPSILON:
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


def _line_intersection(p0: Point2D, p1: Point2D, p2: Point2D, p3: Point2D) -> Point2D | None:
    """
    Where the infinite lines through (p0,p1) and (p2,p3) cross, or None
    if they're parallel — line-line intersection is affine-invariant
    (survives the frame's own anisotropic aspect ratio), so this works
    directly in fraction space without needing full_dims, unlike the
    angle *value* itself (see _four_point_angle_value).
    """
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2
    x4, y4 = p3
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < _DEGENERATE_EPSILON:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


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


def _three_point_angle_segments(
    points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None
) -> list[tuple[Point2D, Point2D]]:
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


def _four_point_angle_segments(
    points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None
) -> list[tuple[Point2D, Point2D]]:
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
    """The tag anchors at the two lines' own intersection ("placed by the angle") whenever they actually cross; only genuinely parallel lines fall back to the plain centroid of what's been placed."""
    if len(points) >= 4:
        intersection = _line_intersection(points[0], points[1], points[2], points[3])
        if intersection is not None:
            return intersection
    xs = [p[0] for p in points[:4]]
    ys = [p[1] for p in points[:4]]
    return sum(xs) / len(xs), sum(ys) / len(ys)


# ----------------------------------------------------------------------
# Arcs — "3 Point Arc" (start, a point the arc passes through, end —
# same circumcircle math as "3 Point Circle", but drawn as just the arc
# between start and end that actually passes through the middle point)
# and "Radius Arc" (center, a point setting both radius and the start
# angle, a third point setting only the end angle — its own distance
# from center is ignored, matching "Radius Circle"'s center+edge but
# with a further point to say where the sweep stops).
# ----------------------------------------------------------------------


def _sweep_through_angle(start_deg: float, mid_deg: float, end_deg: float) -> float:
    """Signed sweep (degrees) from *start_deg* to *end_deg* that passes through *mid_deg* along the way — whichever of the two directions (CCW/positive or CW/negative) reaches *mid_deg* first."""
    ccw_to_end = (end_deg - start_deg) % 360
    ccw_to_mid = (mid_deg - start_deg) % 360
    return ccw_to_end if ccw_to_mid <= ccw_to_end else ccw_to_end - 360


def _three_point_arc_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 3:
        return None
    p0, p1, p2 = points[0], points[1], points[2]
    if p0 == p1 or p1 == p2 or p0 == p2 or _collinear(p0, p1, p2):
        return None
    return p0, p1, p2


def _arc_geometry_three_point(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> ArcGeometry | None:
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0 or len(points) < 3:
        return None

    def to_px(p: Point2D) -> Point2D:
        return p[0] * full_w, p[1] * full_h

    a, b, c = to_px(points[0]), to_px(points[1]), to_px(points[2])
    result = _circumcircle(a, b, c)
    if result is None:
        return None
    center, radius = result
    if radius < _DEGENERATE_EPSILON:
        return None
    start_deg = math.degrees(math.atan2(a[1] - center[1], a[0] - center[0]))
    mid_deg = math.degrees(math.atan2(b[1] - center[1], b[0] - center[0]))
    end_deg = math.degrees(math.atan2(c[1] - center[1], c[0] - center[0]))
    sweep = _sweep_through_angle(start_deg, mid_deg, end_deg)
    return (center[0] / full_w, center[1] / full_h), radius, start_deg, sweep


def _radius_arc_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 3:
        return None
    p0, p1, p2 = points[0], points[1], points[2]
    if p0 == p1 or p0 == p2:
        return None
    return p0, p1, p2


def _arc_geometry_radius(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> ArcGeometry | None:
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0 or len(points) < 3:
        return None

    def to_px(p: Point2D) -> Point2D:
        return p[0] * full_w, p[1] * full_h

    center, start_pt, end_pt = to_px(points[0]), to_px(points[1]), to_px(points[2])
    radius = math.hypot(start_pt[0] - center[0], start_pt[1] - center[1])
    if radius < _DEGENERATE_EPSILON:
        return None
    start_deg = math.degrees(math.atan2(start_pt[1] - center[1], start_pt[0] - center[0]))
    end_deg = math.degrees(math.atan2(end_pt[1] - center[1], end_pt[0] - center[0]))
    # Shortest signed sweep (in (-180, 180]) — unlike "3 Point Arc" there's
    # no third "passes through" point dictating a direction, so the
    # shorter way round is the least surprising default.
    sweep = ((end_deg - start_deg + 180) % 360) - 180
    return (center[0] / full_w, center[1] / full_h), radius, start_deg, sweep


def _radius_arc_move(points: list[Point2D], index: int, new_point: Point2D) -> list[Point2D]:
    if index != 0 or len(points) != 3:
        return _default_move_point(points, index, new_point)
    # Dragging the center translates the whole arc, same reasoning as
    # "Radius Circle"'s own center drag.
    dx = new_point[0] - points[0][0]
    dy = new_point[1] - points[0][1]
    return [new_point, (points[1][0] + dx, points[1][1] + dy), (points[2][0] + dx, points[2][1] + dy)]


# ----------------------------------------------------------------------
# Curve — both ends placed first, then a third point ("the arc") shapes
# a smooth quadratic-Bezier bulge between them. Deliberately not another
# circular arc ("3 Point Arc" already covers that, with the middle point
# lying ON the curve): here the third point is a Bezier *control* point
# the curve bends toward without ever touching, giving a curve whose
# curvature isn't constrained to a single circle's.
#
# Bezier interpolation is a purely affine (lerp-based) construction, so
# — unlike a circular arc's curvature — it's unaffected by the frame's
# own anisotropic aspect ratio: sampling directly in fraction space and
# mapping each sample through the same _to_point every other overlay
# coordinate goes through is already correct, with no true-pixel-space
# detour needed the way arcs/angles require.
# ----------------------------------------------------------------------

_CURVE_SAMPLES = 24


def _curve_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 3:
        return None
    p0, p1, p2 = points[0], points[1], points[2]
    if p0 == p1:
        return None
    return p0, p1, p2


def _curve_points(points: tuple[Point2D, ...]) -> list[Point2D] | None:
    """
    Sampled points along the curve for however many of its 3 points have
    been placed so far — a straight segment (2 points) while the third
    point isn't placed yet, matching how circle/ellipse/arc kinds show a
    straight guide during placement.

    The third point lies *on* the curve at its midpoint (t=0.5), not as a
    Bezier control point the curve merely bends toward — so the derived
    control point is solved back from it: with B(0.5) = 0.25 P0 + 0.5 C +
    0.25 P1 forced to equal the placed midpoint M, C = 2M - (P0 + P1)/2.
    """
    if len(points) < 2:
        return None
    if len(points) < 3:
        return [points[0], points[1]]
    p0, p1, mid = points[0], points[1], points[2]
    control = (2 * mid[0] - (p0[0] + p1[0]) / 2, 2 * mid[1] - (p0[1] + p1[1]) / 2)
    result = []
    for i in range(_CURVE_SAMPLES + 1):
        t = i / _CURVE_SAMPLES
        mt = 1 - t
        x = mt * mt * p0[0] + 2 * mt * t * control[0] + t * t * p1[0]
        y = mt * mt * p0[1] + 2 * mt * t * control[1] + t * t * p1[1]
        result.append((x, y))
    return result


# ----------------------------------------------------------------------
# Parallel/Perpendicular Line families — "line_pair" category. All place
# a first reference line, then derive further lines from it. The overlay
# draws segment_pairs solid and connector_segments dashed, hit-tests
# both, and tags either the perpendicular gap between a parallel pair
# (pair_distance) or, lacking that, the first segment's own length.
#
# Direction, perpendicularity and distance are aspect-ratio sensitive, so
# — like the angle tools — these compute in true pixel space via
# full_dims and convert the result back to fraction space. (Plain
# parallelism alone is affine-invariant, but the *gap* between the lines
# isn't, so full_dims is needed regardless.)
# ----------------------------------------------------------------------


def _mid2(a: Point2D, b: Point2D) -> Point2D:
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def _line_gap_px(line: tuple[Point2D, Point2D], point: Point2D, full_dims: tuple[int, int]) -> float | None:
    """Perpendicular distance (true pixels) from *point* to the infinite line through *line*'s two points, or None if the line is degenerate."""
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0:
        return None
    ax, ay = line[0][0] * full_w, line[0][1] * full_h
    bx, by = line[1][0] * full_w, line[1][1] * full_h
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < _DEGENERATE_EPSILON:
        return None
    px, py = point[0] * full_w, point[1] * full_h
    cross = (px - ax) * dy - (py - ay) * dx
    return abs(cross) / length


def _parallel_second_centered(p0: Point2D, p1: Point2D, center: Point2D) -> tuple[Point2D, Point2D]:
    """The reference line p0->p1 translated so its own midpoint lands on *center* — a plain translation, affine-invariant, so no full_dims needed."""
    mid = _mid2(p0, p1)
    off = (center[0] - mid[0], center[1] - mid[1])
    return (p0[0] + off[0], p0[1] + off[1]), (p1[0] + off[0], p1[1] + off[1])


def _perp_foot(ref: tuple[Point2D, Point2D], point: Point2D, full_dims: tuple[int, int]) -> tuple[Point2D, float] | None:
    """(foot fraction, parameter t along *ref*) of the perpendicular from *point* onto *ref*'s infinite line — pixel space, since a right angle is aspect sensitive. t is 0 at ref[0], 1 at ref[1]."""
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0:
        return None
    ax, ay = ref[0][0] * full_w, ref[0][1] * full_h
    bx, by = ref[1][0] * full_w, ref[1][1] * full_h
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < _DEGENERATE_EPSILON:
        return None
    px, py = point[0] * full_w, point[1] * full_h
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    foot = ((ax + t * dx) / full_w, (ay + t * dy) / full_h)
    return foot, t


def _dimension_connectors(
    ref: tuple[Point2D, Point2D], from_point: Point2D, full_dims: tuple[int, int] | None
) -> list[tuple[Point2D, Point2D]]:
    """
    A perpendicular dashed line from *from_point* down to *ref*'s line,
    plus — when the foot lands beyond *ref*'s own segment — a second,
    collinear dashed line extending *ref* from its nearer endpoint out to
    that foot, so the perpendicular always meets a drawn line. (The
    classic dimension-line-with-extension look.)
    """
    if full_dims is None:
        return []
    result = _perp_foot(ref, from_point, full_dims)
    if result is None:
        return []
    foot, t = result
    connectors = [(from_point, foot)]
    if t < 0.0:
        connectors.append((ref[0], foot))
    elif t > 1.0:
        connectors.append((ref[1], foot))
    return connectors


def _parallel_project(
    ref: tuple[Point2D, Point2D], start: Point2D, other: Point2D, full_dims: tuple[int, int] | None
) -> tuple[Point2D, Point2D] | None:
    """A line from *start* parallel to *ref*, its far end *other* projected onto that direction — the parallel-lock shared by 4pt/8pt parallel. Pixel space."""
    if full_dims is None:
        return None
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0:
        return None
    ax, ay = ref[0][0] * full_w, ref[0][1] * full_h
    bx, by = ref[1][0] * full_w, ref[1][1] * full_h
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < _DEGENERATE_EPSILON:
        return None
    ux, uy = dx / length, dy / length
    sx, sy = start[0] * full_w, start[1] * full_h
    ox, oy = other[0] * full_w, other[1] * full_h
    proj = (ox - sx) * ux + (oy - sy) * uy
    end = ((sx + proj * ux) / full_w, (sy + proj * uy) / full_h)
    return start, end


# --- 3pt Parallel: line, then a point centering a same-length parallel copy ---


def _two_line_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 3:
        return None
    if points[0] == points[1]:
        return None
    return tuple(points[:3])


def _parallel3_segments(
    points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None
) -> list[tuple[Point2D, Point2D]]:
    pairs = []
    if len(points) >= 2:
        pairs.append((points[0], points[1]))
    if len(points) >= 3:
        pairs.append(_parallel_second_centered(points[0], points[1], points[2]))
    return pairs


def _parallel3_connectors(
    points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None
) -> list[tuple[Point2D, Point2D]]:
    if len(points) < 3:
        return []
    a, b = _parallel_second_centered(points[0], points[1], points[2])
    return _dimension_connectors((points[0], points[1]), _mid2(a, b), full_dims)


def _parallel3_distance(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> tuple[Point2D, float] | None:
    if len(points) < 3:
        return None
    connectors = _parallel3_connectors(points, full_dims)
    line2 = _parallel_second_centered(points[0], points[1], points[2])
    gap = _line_gap_px((points[0], points[1]), _mid2(line2[0], line2[1]), full_dims)
    if gap is None or not connectors:
        return None
    # Anchor on the perpendicular connector so the tag sits by the indicator line.
    return _mid2(connectors[0][0], connectors[0][1]), gap


# --- 4pt Parallel: reference line, then both ends of a parallel-locked line ---


def _four_point_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 4 or points[0] == points[1]:
        return None
    return tuple(points[:4])


def _parallel4_second(points: tuple[Point2D, ...], full_dims: tuple[int, int] | None) -> tuple[Point2D, Point2D] | None:
    """Second line: starts at p2, runs parallel to p0->p1, its far end the projection of p3 onto that direction (so all four points are placed, yet the line stays parallel). Pixel space, since a projection isn't affine-invariant."""
    if full_dims is None:
        return None
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0:
        return None
    a = (points[0][0] * full_w, points[0][1] * full_h)
    b = (points[1][0] * full_w, points[1][1] * full_h)
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < _DEGENERATE_EPSILON:
        return None
    ux, uy = dx / length, dy / length
    c = (points[2][0] * full_w, points[2][1] * full_h)
    d = (points[3][0] * full_w, points[3][1] * full_h)
    t = (d[0] - c[0]) * ux + (d[1] - c[1]) * uy
    end = (c[0] + t * ux, c[1] + t * uy)
    return points[2], (end[0] / full_w, end[1] / full_h)


def _parallel4_segments(
    points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None
) -> list[tuple[Point2D, Point2D]]:
    pairs = []
    if len(points) >= 2:
        pairs.append((points[0], points[1]))
    if len(points) >= 4:
        second = _parallel4_second(points, full_dims)
        pairs.append(second if second is not None else (points[2], points[3]))
    return pairs


def _parallel4_snap(points: tuple[Point2D, ...], full_dims: tuple[int, int] | None) -> tuple[Point2D, ...]:
    if len(points) < 4 or full_dims is None:
        return points
    second = _parallel4_second(points, full_dims)
    if second is None:
        return points
    return (points[0], points[1], second[0], second[1])


def _parallel4_connectors(
    points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None
) -> list[tuple[Point2D, Point2D]]:
    if len(points) < 4:
        return []
    second = _parallel4_second(points, full_dims)
    if second is None:
        return []
    return _dimension_connectors((points[0], points[1]), _mid2(second[0], second[1]), full_dims)


def _parallel4_distance(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> tuple[Point2D, float] | None:
    if len(points) < 4:
        return None
    connectors = _parallel4_connectors(points, full_dims)
    second = _parallel4_second(points, full_dims)
    if not connectors or second is None:
        return None
    gap = _line_gap_px((points[0], points[1]), _mid2(second[0], second[1]), full_dims)
    if gap is None:
        return None
    return _mid2(connectors[0][0], connectors[0][1]), gap


# --- 8pt Parallel: two pairs of parallel lines, gap between their midlines ---


def _eight_point_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 8:
        return None
    return tuple(points[:8])


def _midline(line_a: tuple[Point2D, Point2D], line_b: tuple[Point2D, Point2D]) -> tuple[Point2D, Point2D]:
    return _mid2(line_a[0], line_b[0]), _mid2(line_a[1], line_b[1])


def _parallel8_lines(
    points: tuple[Point2D, ...], full_dims: tuple[int, int] | None
) -> list[tuple[Point2D, Point2D]]:
    """Up to four lines: the reference (points 0-1) and each further pair locked parallel to it (its 2nd point projected onto the reference direction). Stops at whatever's been placed so far."""
    if len(points) < 2:
        return []
    ref = (points[0], points[1])
    lines = [ref]
    for i in range(1, 4):
        seg = points[2 * i:2 * i + 2]
        if len(seg) < 2:
            break
        projected = _parallel_project(ref, seg[0], seg[1], full_dims)
        lines.append(projected if projected is not None else (seg[0], seg[1]))
    return lines


def _parallel8_segments(
    points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None
) -> list[tuple[Point2D, Point2D]]:
    return _parallel8_lines(points, full_dims)


def _parallel8_snap(points: tuple[Point2D, ...], full_dims: tuple[int, int] | None) -> tuple[Point2D, ...]:
    if full_dims is None:
        return points
    lines = _parallel8_lines(points, full_dims)
    snapped = list(points)
    for i, line in enumerate(lines[1:], start=1):
        end_index = 2 * i + 1
        if end_index < len(snapped):
            snapped[end_index] = line[1]
    return tuple(snapped)


def _parallel8_connectors(
    points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None
) -> list[tuple[Point2D, Point2D]]:
    lines = _parallel8_lines(points, full_dims)
    connectors = []
    if len(lines) >= 2:
        connectors.append(_midline(lines[0], lines[1]))
    if len(lines) >= 4:
        mid_b = _midline(lines[2], lines[3])
        connectors.append(mid_b)
        # The dimension line between the two midlines only forms once all
        # four lines are placed.
        connectors += _dimension_connectors(_midline(lines[0], lines[1]), _mid2(mid_b[0], mid_b[1]), full_dims)
    return connectors


def _parallel8_distance(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> tuple[Point2D, float] | None:
    lines = _parallel8_lines(points, full_dims)
    if len(lines) < 4:
        return None
    mid_a = _midline(lines[0], lines[1])
    mid_b = _midline(lines[2], lines[3])
    gap = _line_gap_px(mid_a, _mid2(mid_b[0], mid_b[1]), full_dims)
    if gap is None:
        return None
    return _mid2(_mid2(mid_a[0], mid_a[1]), _mid2(mid_b[0], mid_b[1])), gap


# --- Arbitrary Parallel: a line, then each extra point adds a parallel copy ---


def _arbitrary_parallel_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 3 or points[0] == points[1]:
        return None
    return tuple(points)


def _arbitrary_parallel_segments(
    points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None
) -> list[tuple[Point2D, Point2D]]:
    pairs = []
    if len(points) >= 2:
        pairs.append((points[0], points[1]))
    for extra in points[2:]:
        pairs.append(_parallel_second_centered(points[0], points[1], extra))
    return pairs


# --- 3pt Perpendicular: line, then a point; the perpendicular foot->point line ---


def _perpendicular_foot(points: tuple[Point2D, ...], full_dims: tuple[int, int] | None) -> Point2D | None:
    """Foot of the perpendicular from p2 onto the infinite line through p0/p1, in fraction space — computed via true pixel space, since a right angle is aspect-ratio sensitive."""
    if full_dims is None or len(points) < 3:
        return None
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0:
        return None
    a = (points[0][0] * full_w, points[0][1] * full_h)
    b = (points[1][0] * full_w, points[1][1] * full_h)
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq < _DEGENERATE_EPSILON:
        return None
    c = (points[2][0] * full_w, points[2][1] * full_h)
    t = ((c[0] - a[0]) * dx + (c[1] - a[1]) * dy) / length_sq
    return (a[0] + t * dx) / full_w, (a[1] + t * dy) / full_h


def _perpendicular3_segments(
    points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None
) -> list[tuple[Point2D, Point2D]]:
    """Both lines solid: the reference line and the perpendicular from its foot out to the third point."""
    segments = []
    if len(points) >= 2:
        segments.append((points[0], points[1]))
    if len(points) >= 3:
        foot = _perpendicular_foot(points, full_dims)
        if foot is not None:
            segments.append((foot, points[2]))
    return segments


def _perpendicular3_measures(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> list[tuple[Point2D, float]]:
    """A separate length tag for the perpendicular leg (the reference line gets the main tag)."""
    if len(points) < 3:
        return []
    foot = _perpendicular_foot(points, full_dims)
    if foot is None:
        return []
    gap = _line_gap_px((points[0], points[1]), points[2], full_dims)
    if gap is None:
        return []
    return [(_mid2(foot, points[2]), gap)]


# --- 4pt Perpendicular: reference line, then both ends of a perpendicular line ---


def _perpendicular4_second(points: tuple[Point2D, ...], full_dims: tuple[int, int] | None) -> tuple[Point2D, Point2D] | None:
    """Second line from p2, perpendicular to p0->p1, its length the projection of p3-p2 onto that perpendicular direction — so both of the perpendicular line's ends are placed while it stays at a true right angle. Pixel space."""
    if full_dims is None:
        return None
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0:
        return None
    a = (points[0][0] * full_w, points[0][1] * full_h)
    b = (points[1][0] * full_w, points[1][1] * full_h)
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < _DEGENERATE_EPSILON:
        return None
    # Perpendicular unit direction.
    nx, ny = -dy / length, dx / length
    c = (points[2][0] * full_w, points[2][1] * full_h)
    d = (points[3][0] * full_w, points[3][1] * full_h)
    t = (d[0] - c[0]) * nx + (d[1] - c[1]) * ny
    end = (c[0] + t * nx, c[1] + t * ny)
    return points[2], (end[0] / full_w, end[1] / full_h)


def _perpendicular4_segments(
    points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None
) -> list[tuple[Point2D, Point2D]]:
    """Reference line and the perpendicular line, both solid."""
    segments = []
    if len(points) >= 2:
        segments.append((points[0], points[1]))
    if len(points) >= 4:
        second = _perpendicular4_second(points, full_dims)
        if second is not None:
            segments.append(second)
    return segments


def _perpendicular4_connectors(
    points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None
) -> list[tuple[Point2D, Point2D]]:
    """A dashed line from the perpendicular line's start down to the reference line (collinear with the perpendicular line itself), plus an extension when its foot lands past the reference segment."""
    if len(points) < 4:
        return []
    return _dimension_connectors((points[0], points[1]), points[2], full_dims)


def _perpendicular4_snap(points: tuple[Point2D, ...], full_dims: tuple[int, int] | None) -> tuple[Point2D, ...]:
    if len(points) < 4 or full_dims is None:
        return points
    second = _perpendicular4_second(points, full_dims)
    if second is None:
        return points
    return (points[0], points[1], second[0], second[1])


def _perpendicular4_distance(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> tuple[Point2D, float] | None:
    if len(points) < 4:
        return None
    second = _perpendicular4_second(points, full_dims)
    if second is None:
        return None
    full_w, full_h = full_dims
    dx = (second[1][0] - second[0][0]) * full_w
    dy = (second[1][1] - second[0][1]) * full_h
    dist = math.hypot(dx, dy)
    if dist < _DEGENERATE_EPSILON:
        return None
    return _mid2(second[0], second[1]), dist


# --- Arbitrary Perpendicular: a line, then each extra point adds a perpendicular ---


def _arbitrary_perpendicular_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 3 or points[0] == points[1]:
        return None
    return tuple(points)


def _arbitrary_perpendicular_segments(
    points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None
) -> list[tuple[Point2D, Point2D]]:
    """The reference line plus one solid perpendicular per extra point (from its foot on the reference line out to the point)."""
    segments = []
    if len(points) >= 2:
        segments.append((points[0], points[1]))
    for extra in points[2:]:
        foot = _perpendicular_foot((points[0], points[1], extra), full_dims)
        if foot is not None:
            segments.append((foot, extra))
    return segments


def _arbitrary_perpendicular_connectors(
    points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None
) -> list[tuple[Point2D, Point2D]]:
    """Dashed extensions of the reference line, one for each perpendicular whose foot lands beyond the reference segment's own ends."""
    if len(points) < 2 or full_dims is None:
        return []
    ref = (points[0], points[1])
    connectors = []
    for extra in points[2:]:
        info = _perp_foot(ref, extra, full_dims)
        if info is None:
            continue
        foot, t = info
        if t < 0.0:
            connectors.append((ref[0], foot))
        elif t > 1.0:
            connectors.append((ref[1], foot))
    return connectors


def _arbitrary_perpendicular_measures(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> list[tuple[Point2D, float]]:
    """A length tag for each perpendicular line (from its foot on the reference line out to the placed point)."""
    if len(points) < 2:
        return []
    ref = (points[0], points[1])
    measures = []
    for extra in points[2:]:
        foot = _perpendicular_foot((points[0], points[1], extra), full_dims)
        if foot is None:
            continue
        gap = _line_gap_px(ref, extra, full_dims)
        if gap is not None:
            measures.append((_mid2(foot, extra), gap))
    return measures


def _arbitrary_parallel_connectors(
    points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None
) -> list[tuple[Point2D, Point2D]]:
    """A dashed dimension connector between each consecutive pair of parallel lines, where each gap tag sits."""
    lines = _arbitrary_parallel_segments(points, full_dims)
    connectors = []
    for line_a, line_b in zip(lines, lines[1:]):
        connectors += _dimension_connectors(line_a, _mid2(line_b[0], line_b[1]), full_dims)
    return connectors


def _arbitrary_parallel_measures(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> list[tuple[Point2D, float]]:
    """A distance tag for the gap between each consecutive pair of parallel lines."""
    lines = _arbitrary_parallel_segments(points, full_dims)
    measures = []
    for line_a, line_b in zip(lines, lines[1:]):
        gap = _line_gap_px(line_a, _mid2(line_b[0], line_b[1]), full_dims)
        if gap is not None:
            anchor = _mid2(_mid2(line_a[0], line_a[1]), _mid2(line_b[0], line_b[1]))
            measures.append((anchor, gap))
    return measures


# ----------------------------------------------------------------------
# Rectangles & polygons — "polygon" category. polygon_points returns the
# ordered boundary vertices; the overlay closes and (optionally) fills
# them. Rotated forms compute in true pixel space so their right angles
# and equal sides survive the frame's aspect ratio.
# ----------------------------------------------------------------------


def _rectangle_2pt_points(points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None) -> list[Point2D] | None:
    if len(points) < 2 or points[0] == points[1]:
        return None
    (x0, y0), (x1, y1) = points[0], points[1]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _perp_unit_px(p0: Point2D, p1: Point2D, full_dims: tuple[int, int]) -> tuple[float, float, float, float, float] | None:
    """(ax_px, ay_px, unit_perp_x, unit_perp_y, edge_len_px) for the edge p0->p1, or None if degenerate. Pixel space."""
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0:
        return None
    ax, ay = p0[0] * full_w, p0[1] * full_h
    bx, by = p1[0] * full_w, p1[1] * full_h
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < _DEGENERATE_EPSILON:
        return None
    return ax, ay, -dy / length, dx / length, length


def _rectangle_3pt_points(points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None) -> list[Point2D] | None:
    if len(points) < 3 or full_dims is None:
        return None
    basis = _perp_unit_px(points[0], points[1], full_dims)
    if basis is None:
        return None
    full_w, full_h = full_dims
    _ax, _ay, nx, ny, _length = basis
    bx, by = points[1][0] * full_w, points[1][1] * full_h
    cx, cy = points[2][0] * full_w, points[2][1] * full_h
    depth = (cx - bx) * nx + (cy - by) * ny
    off = (nx * depth, ny * depth)
    corners_px = [
        (points[0][0] * full_w, points[0][1] * full_h),
        (bx, by),
        (bx + off[0], by + off[1]),
        (points[0][0] * full_w + off[0], points[0][1] * full_h + off[1]),
    ]
    return [(x / full_w, y / full_h) for x, y in corners_px]


def _square_from_edge(points: tuple[Point2D, ...], full_dims: tuple[int, int], sign: float) -> list[Point2D] | None:
    basis = _perp_unit_px(points[0], points[1], full_dims)
    if basis is None:
        return None
    full_w, full_h = full_dims
    ax, ay, nx, ny, length = basis
    bx, by = points[1][0] * full_w, points[1][1] * full_h
    off = (nx * length * sign, ny * length * sign)
    corners_px = [(ax, ay), (bx, by), (bx + off[0], by + off[1]), (ax + off[0], ay + off[1])]
    return [(x / full_w, y / full_h) for x, y in corners_px]


def _square_2pt_points(points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None) -> list[Point2D] | None:
    if len(points) < 2 or full_dims is None:
        return None
    return _square_from_edge(points, full_dims, 1.0)


def _polygon_points(points: tuple[Point2D, ...], full_dims: tuple[int, int] | None = None) -> list[Point2D] | None:
    return list(points) if len(points) >= 2 else None


def _rectangle_2pt_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 2 or points[0] == points[1]:
        return None
    return tuple(points[:2])


def _polygon_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    return tuple(points) if len(points) >= 3 else None


# ----------------------------------------------------------------------
# Annulus (ring) — "annulus" category: two concentric circles.
# ----------------------------------------------------------------------


def _annulus_pair(center: Point2D, outer_r: float, inner_r: float) -> tuple[Point2D, float, float] | None:
    """(center, larger_r, smaller_r) — or, when only the outer radius is
    known yet (inner_r ~0, still being placed), (center, outer_r, 0.0) so
    the outer circle can preview on its own. None only if even the outer
    radius is degenerate."""
    if outer_r < _DEGENERATE_EPSILON:
        return None
    if inner_r < _DEGENERATE_EPSILON:
        return center, outer_r, 0.0
    lo, hi = sorted((outer_r, inner_r))
    return center, hi, lo


def _radius_annulus_geometry(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> tuple[Point2D, float, float] | None:
    if len(points) < 2:
        return None
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0:
        return None
    cx, cy = points[0][0] * full_w, points[0][1] * full_h
    outer_r = math.hypot(points[1][0] * full_w - cx, points[1][1] * full_h - cy)
    inner_r = math.hypot(points[2][0] * full_w - cx, points[2][1] * full_h - cy) if len(points) >= 3 else 0.0
    return _annulus_pair(points[0], outer_r, inner_r)


def _diameter_annulus_geometry(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> tuple[Point2D, float, float] | None:
    if len(points) < 2:
        return None
    full_w, full_h = full_dims
    if full_w <= 0 or full_h <= 0:
        return None
    a = (points[0][0] * full_w, points[0][1] * full_h)
    b = (points[1][0] * full_w, points[1][1] * full_h)
    center_px = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    outer_r = math.hypot(b[0] - a[0], b[1] - a[1]) / 2
    center = (center_px[0] / full_w, center_px[1] / full_h)
    inner_r = math.hypot(points[2][0] * full_w - center_px[0], points[2][1] * full_h - center_px[1]) if len(points) >= 3 else 0.0
    return _annulus_pair(center, outer_r, inner_r)


def _three_point_annulus_geometry(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> tuple[Point2D, float, float] | None:
    if len(points) < 3:
        return None
    outer = _circle_geometry_from_three_points(points[:3], full_dims)
    if outer is None:
        return None
    center, outer_r = outer
    full_w, full_h = full_dims
    inner_r = math.hypot(points[3][0] * full_w - center[0] * full_w, points[3][1] * full_h - center[1] * full_h) if len(points) >= 4 else 0.0
    return _annulus_pair(center, outer_r, inner_r)


def _radius_annulus_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 3 or points[0] == points[1]:
        return None
    return tuple(points[:3])


def _diameter_annulus_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 3 or points[0] == points[1]:
        return None
    return tuple(points[:3])


def _three_point_annulus_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 4 or _collinear(points[0], points[1], points[2]):
        return None
    return tuple(points[:4])


# ----------------------------------------------------------------------
# Two Circle — "two_circle" category: two circles, tagged with the
# distance between their centers.
# ----------------------------------------------------------------------


def _two_circle_partial(
    points: tuple[Point2D, ...], full_dims: tuple[int, int], per_circle: int,
    builder: Callable[[tuple[Point2D, ...], tuple[int, int]], CircleGeometry | None],
) -> list[CircleGeometry]:
    """The circles determinable so far: each *per_circle*-point slice fed through *builder*, stopping at the first slice that isn't complete or doesn't resolve."""
    circles = []
    for start in (0, per_circle):
        slice_ = points[start:start + per_circle]
        if len(slice_) < per_circle:
            break
        circle = builder(slice_, full_dims)
        if circle is None:
            break
        circles.append(circle)
    return circles


def _radius_two_circle_partial(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> list[CircleGeometry]:
    return _two_circle_partial(points, full_dims, 2, _circle_geometry_from_center_edge)


def _diameter_two_circle_partial(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> list[CircleGeometry]:
    return _two_circle_partial(points, full_dims, 2, _circle_geometry_from_diameter)


def _three_point_two_circle_partial(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> list[CircleGeometry]:
    return _two_circle_partial(points, full_dims, 3, _circle_geometry_from_three_points)


def _two_circle_full(circles: list[CircleGeometry]) -> tuple[Point2D, float, Point2D, float] | None:
    if len(circles) < 2:
        return None
    (c1, r1), (c2, r2) = circles[0], circles[1]
    return c1, r1, c2, r2


def _radius_two_circle_geometry(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> tuple[Point2D, float, Point2D, float] | None:
    return _two_circle_full(_radius_two_circle_partial(points, full_dims))


def _diameter_two_circle_geometry(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> tuple[Point2D, float, Point2D, float] | None:
    return _two_circle_full(_diameter_two_circle_partial(points, full_dims))


def _three_point_two_circle_geometry(points: tuple[Point2D, ...], full_dims: tuple[int, int]) -> tuple[Point2D, float, Point2D, float] | None:
    return _two_circle_full(_three_point_two_circle_partial(points, full_dims))


def _two_point_pair_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 4 or points[0] == points[1] or points[2] == points[3]:
        return None
    return tuple(points[:4])


def _three_point_two_circle_resolve(points: list[Point2D]) -> tuple[Point2D, ...] | None:
    if len(points) < 6:
        return None
    if _collinear(points[0], points[1], points[2]) or _collinear(points[3], points[4], points[5]):
        return None
    return tuple(points[:6])


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
    connected_segments=True,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # Two independent, disconnected segments — showing the angle
    # between them instead of a length.
    name="4 Point Angle", required_points=4, category="angle",
    resolve=_four_point_angle_resolve, segment_pairs=_four_point_angle_segments,
    angle_value=_four_point_angle_value, angle_anchor=_four_point_angle_anchor,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # Start, a point the arc passes through, end — same circumcircle
    # math as "3 Point Circle", drawn as just the arc between start and
    # end that passes through the middle point.
    name="3 Point Arc", required_points=3, category="arc",
    resolve=_three_point_arc_resolve, arc_geometry=_arc_geometry_three_point,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # Center, a point setting radius + start angle, a third point
    # setting only the end angle (its own distance from center is
    # ignored) — the shorter way round between start and end angle.
    name="Radius Arc", required_points=3, category="arc",
    resolve=_radius_arc_resolve, move_point=_radius_arc_move, arc_geometry=_arc_geometry_radius,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # Both ends placed first, then a third point shapes a smooth
    # quadratic-Bezier bulge between them.
    name="Curve", required_points=3, category="curve",
    resolve=_curve_resolve, curve_points=_curve_points,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # First line (2 points), then a third point centering a same-length
    # parallel copy; dashed connectors tie the pair, the tag shows the
    # perpendicular gap between them.
    name="3pt Parallel", required_points=3, category="line_pair",
    resolve=_two_line_resolve, segment_pairs=_parallel3_segments,
    connector_segments=_parallel3_connectors, pair_distance=_parallel3_distance,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # Reference line, then both ends of a second line locked parallel to
    # it (its far end is p3 projected onto the parallel direction).
    name="4pt Parallel", required_points=4, category="line_pair",
    resolve=_four_point_resolve, segment_pairs=_parallel4_segments,
    connector_segments=_parallel4_connectors, pair_distance=_parallel4_distance,
    snap_points=_parallel4_snap,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # Four parallel lines (8 points); dashed midlines between each pair,
    # the tag showing the gap between those two midlines.
    name="8pt Parallel", required_points=8, category="line_pair",
    resolve=_eight_point_resolve, segment_pairs=_parallel8_segments,
    connector_segments=_parallel8_connectors, pair_distance=_parallel8_distance,
    snap_points=_parallel8_snap,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # A first line, then each additional click drops another parallel copy
    # centered on it — right-click to finish.
    name="Arbitrary Parallel", required_points=None, category="line_pair",
    resolve=_arbitrary_parallel_resolve, segment_pairs=_arbitrary_parallel_segments,
    connector_segments=_arbitrary_parallel_connectors,
    extra_measures=_arbitrary_parallel_measures, min_points=3,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # First line (2 points), then a third point; the perpendicular from
    # its foot on the line out to that point, drawn dashed, tagged with
    # its length.
    name="3pt Perp", required_points=3, category="line_pair",
    resolve=_two_line_resolve, segment_pairs=_perpendicular3_segments,
    extra_measures=_perpendicular3_measures,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # Reference line, then both ends of a perpendicular line (its length
    # is p3 projected onto the perpendicular direction).
    name="4pt Perp", required_points=4, category="line_pair",
    resolve=_four_point_resolve, segment_pairs=_perpendicular4_segments,
    connector_segments=_perpendicular4_connectors, pair_distance=_perpendicular4_distance,
    snap_points=_perpendicular4_snap,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # A first line, then each additional click drops another perpendicular
    # off it — right-click to finish.
    name="Arbitrary Perp", required_points=None, category="line_pair",
    resolve=_arbitrary_perpendicular_resolve, segment_pairs=_arbitrary_perpendicular_segments,
    connector_segments=_arbitrary_perpendicular_connectors,
    extra_measures=_arbitrary_perpendicular_measures, min_points=3,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # Same behavior as "Arbitrary Line" — an unbounded polyline — with a
    # distinct tile icon; kept separate so the menu can group it under
    # the Line category alongside the plain arbitrary line.
    name="Multipoint Line", required_points=None, category="line", resolve=_arbitrary_line_resolve,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # A 2-point line preset to arrow caps on both ends.
    name="Double Arrow", required_points=2, category="line", resolve=_two_point_resolve,
    meta_preset={"line_start_cap": "arrow", "line_end_cap": "arrow"},
))
DEFAULT_REGISTRY.register(MeasurementKind(
    name="2pt Rectangle", required_points=2, category="polygon",
    resolve=_rectangle_2pt_resolve, polygon_points=_rectangle_2pt_points,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    name="3pt Rectangle", required_points=3, category="polygon",
    resolve=_three_point_circle_resolve, polygon_points=_rectangle_3pt_points,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    name="2pt Square", required_points=2, category="polygon",
    resolve=_rectangle_2pt_resolve, polygon_points=_square_2pt_points,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # Unbounded closed polygon — right-click to finish.
    name="Polygon", required_points=None, category="polygon",
    resolve=_polygon_resolve, polygon_points=_polygon_points, min_points=3,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    name="Radius Annulus", required_points=3, category="annulus",
    resolve=_radius_annulus_resolve, annulus_geometry=_radius_annulus_geometry,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    name="3pt Annulus", required_points=4, category="annulus",
    resolve=_three_point_annulus_resolve, annulus_geometry=_three_point_annulus_geometry,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    name="Diameter Annulus", required_points=3, category="annulus",
    resolve=_diameter_annulus_resolve, annulus_geometry=_diameter_annulus_geometry,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    name="Radius 2 Circle", required_points=4, category="two_circle",
    resolve=_two_point_pair_resolve, two_circle_geometry=_radius_two_circle_geometry,
    two_circle_partial=_radius_two_circle_partial,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    name="3pt 2 Circle", required_points=6, category="two_circle",
    resolve=_three_point_two_circle_resolve, two_circle_geometry=_three_point_two_circle_geometry,
    two_circle_partial=_three_point_two_circle_partial,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    name="Diameter 2 Circle", required_points=4, category="two_circle",
    resolve=_two_point_pair_resolve, two_circle_geometry=_diameter_two_circle_geometry,
    two_circle_partial=_diameter_two_circle_partial,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # A single-point text annotation — the "title" holds its text; drawn
    # directly on the image (no length tag). Preset to white-on-black.
    name="Text", required_points=1, category="text", resolve=_point_resolve,
    # Opacity fades only the background panel (see
    # MeasurementOverlay._draw_text_annotation) — half by default so text
    # placed over live content reads over it without fully blocking it.
    meta_preset={"tag_text_color": "#ffffff", "tag_background_color": "#000000", "opacity": 0.5},
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # A 2-point arrow annotation whose length tag is hidden by default.
    name="Annotation Arrow", required_points=2, category="line", resolve=_two_point_resolve,
    meta_preset={"line_start_cap": "curved", "line_end_cap": "arrow", "hidden": True},
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # A scale bar — placed with a single click (its point only matters for
    # the "custom" position preset); white bar on a black panel by default.
    name="Scale Bar", required_points=1, category="scalebar", resolve=_point_resolve, has_label=False,
    meta_preset={"line_color": "#ffffff", "tag_background_color": "#000000"},
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # An unbounded group of numbered points — each click adds one, a
    # right-click finalizes the group (the same "keeps accumulating,
    # right-click finishes" placement "Arbitrary Line" uses — see
    # MeasurementOverlay.cancel_placement). Its numbers are drawn directly
    # on the image, not as tags — see MeasurementOverlay._draw_count_numbers.
    name="Count", required_points=None, category="count", resolve=_count_resolve,
    has_label=False, min_points=1,
))
DEFAULT_REGISTRY.register(MeasurementKind(
    # Placed the same way as a 2-point line but never becomes a real
    # Measurement (see MeasurementOverlay._calibration_line) and never
    # gets the generic title/length tag — see start_calibration_placement.
    name=CALIBRATION_KIND, required_points=2, category="line", resolve=_two_point_resolve, has_label=False,
))
