from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QPainter, QPainterPath, QPolygonF

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import ENDPOINT_RADIUS, LINE_COLOR, LINE_MARGIN

# A calibration line is placed the same way as a 2-point line but has no
# tile of its own (see MeasurementOverlay.start_calibration_placement)
# and no registered MeasurementButton — its name lives here, next to the
# tile classes below, rather than in measurement_kind.py, which
# registers it as a MeasurementKind alongside the real kinds.
CALIBRATION_KIND = "Calibration Line"

# Matches the choices in MeasurementCustomizeMenu's start/end cap
# pickers — "curved" (the default RoundCap look) is deliberately first
# since it's what an unset cap resolves to. Caps are a line-only
# decoration (MeasurementOverlay never passes start_cap/end_cap when
# drawing a circle), hence living here.
MEASUREMENT_LINE_CAPS = ("curved", "square", "arrow", "arrow_open", "bracket")

# Marker drawn at a line's midpoint (see MeasurementOverlay._draw_midpoint_marker).
# "none" is first since it's what an unset style resolves to.
MEASUREMENT_MIDPOINT_STYLES = ("none", "tick", "x")


def arrow_dims(line_width: float, stroke_scale: float = 1.0, size_scale: float = 1.0) -> tuple[float, float]:
    """
    (length, half-width) of an arrowhead sized for *line_width* (already
    counter-scaled to screen pixels, i.e. divided by *stroke_scale* the
    same way the shaft's own width is) — shared by the solid and open
    arrow caps, and (for its half-width alone) the bracket cap, so they
    all read as consistently sized against the same *size_scale*
    (``MeasurementMeta.cap_size_scale`` — see MeasurementCustomizeMenu's
    "Arrow/Bracket Size" control) rather than each having its own knob.

    The 10.0/5.0 floors keep a thin line's arrowhead from shrinking below
    a legible size, and — like every other fixed-on-screen-size value in
    this module (see OVERLAY_DASH_LENGTH, _DASH_REFERENCE_MIN) — that
    floor is itself a *screen-pixel* target, not a rect-space one. Left
    undivided, the floor would dominate at any zoom level where
    ``line_width * 4``/``* 2.2`` falls under it (true for the default 2px
    line even unzoomed), and since the whole shape is later scaled back
    up by *stroke_scale* when painted, an undivided floor grows the
    arrowhead ever larger as you zoom in instead of holding it constant.
    Dividing the floor by *stroke_scale* here cancels that the same way
    ``line_width`` itself already does. *size_scale* is applied last, to
    the whole (already zoom-corrected) result, so it scales the
    arrowhead uniformly regardless of zoom.
    """
    base_len = max(line_width * 4, 10.0 / stroke_scale)
    base_half = max(line_width * 2.2, 5.0 / stroke_scale)
    return base_len * size_scale, base_half * size_scale


def arrow_head_path(line_width: float, stroke_scale: float = 1.0, size_scale: float = 1.0) -> QPainterPath:
    """Closed triangle for a solid ("arrow") arrowhead, tip at the origin and pointing in the -x direction — the caller translates/rotates it onto a segment's actual endpoint."""
    arrow_len, arrow_half_width = arrow_dims(line_width, stroke_scale, size_scale)
    path = QPainterPath()
    path.moveTo(0, 0)
    path.lineTo(-arrow_len, -arrow_half_width)
    path.lineTo(-arrow_len, arrow_half_width)
    path.closeSubpath()
    return path


def open_arrow_barbs_path(line_width: float, stroke_scale: float = 1.0, size_scale: float = 1.0) -> QPainterPath:
    """A single connected V flaring back from the origin for an ("arrow_open") arrowhead — meant to be stroked with a miter join so its apex at the origin comes to a sharp point like arrow_head_path's tip, rather than the rounded nub two separate round-capped strokes would leave. Tip at the origin, pointing in the -x direction."""
    arrow_len, arrow_half_width = arrow_dims(line_width, stroke_scale, size_scale)
    path = QPainterPath()
    path.moveTo(-arrow_len, -arrow_half_width)
    path.lineTo(0, 0)
    path.lineTo(-arrow_len, arrow_half_width)
    return path


def bracket_path(line_width: float, stroke_scale: float = 1.0, size_scale: float = 1.0) -> QPainterPath:
    """
    A thin rect straddling the origin, spanning ``arrow_dims``' half-width
    perpendicular to the shaft — the "[" / "]" tick a "bracket" cap draws
    across a line's end, sized by the same *size_scale* the arrow caps
    use ("customize the arrow size and have that control the bracket
    size"). Sits flush at the true endpoint (its along-shaft extent is
    only *line_width* wide, no wider than the shaft itself), so unlike
    "arrow"/"arrow_open" it needs no shortening of the shaft it's unioned
    onto — same reasoning as "square".
    """
    _, arrow_half_width = arrow_dims(line_width, stroke_scale, size_scale)
    half_thickness = line_width / 2
    path = QPainterPath()
    path.addRect(QRectF(-half_thickness, -arrow_half_width, 2 * half_thickness, 2 * arrow_half_width))
    return path


class ArbitraryLineMeasurement(MeasurementButton):
    # An unbounded polyline — the icon shows a couple of joined segments
    # with the points that place them, rather than a single straight line.
    name = "Arbitrary Line"
    display_name = "Arb. Line"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        p0 = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        p1 = QPoint(rect.center().x(), rect.top() + LINE_MARGIN)
        p2 = QPoint(rect.right() - LINE_MARGIN, rect.center().y())

        self._set_pen(painter, LINE_COLOR)
        painter.drawPolyline([p0, p1, p2])
        for point in (p0, p1, p2):
            self._draw_point(painter, point, ENDPOINT_RADIUS, active)


class MultipointLineMeasurement(MeasurementButton):
    # Same unbounded polyline as "Arbitrary Line" (see its MeasurementKind)
    # — a distinct zig-zag icon is the only difference, so the two can sit
    # side by side under the Line category.
    name = "Multipoint Line"
    display_name = "Multipoint"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        p0 = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        p1 = QPoint(rect.left() + rect.width() // 3, rect.top() + LINE_MARGIN)
        p2 = QPoint(rect.left() + 2 * rect.width() // 3, rect.bottom() - LINE_MARGIN)
        p3 = QPoint(rect.right() - LINE_MARGIN, rect.top() + LINE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        painter.drawPolyline([p0, p1, p2, p3])
        for point in (p0, p1, p2, p3):
            self._draw_point(painter, point, ENDPOINT_RADIUS, active)


class ArrowMeasurement(MeasurementButton):
    # Same kind/placement as any other 2-point line — see the "Arrow"
    # MeasurementKind's meta_preset in measurement_kind.py, which is what
    # actually gives a placed one its arrowhead. This tile is only a
    # different icon and a shortcut into that preset.
    name = "Arrow"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        start = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        end = QPoint(rect.right() - LINE_MARGIN, rect.top() + LINE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(start, end)
        self._draw_point(painter, start, ENDPOINT_RADIUS, active)

        dx, dy = end.x() - start.x(), end.y() - start.y()
        length = (dx * dx + dy * dy) ** 0.5
        if length > 0:
            ux, uy = dx / length, dy / length
            px, py = -uy, ux
            head_len, head_half = 9.0, 4.0
            base_x, base_y = end.x() - ux * head_len, end.y() - uy * head_len
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(LINE_COLOR)
            painter.drawPolygon(QPolygonF([
                QPointF(end),
                QPointF(base_x + px * head_half, base_y + py * head_half),
                QPointF(base_x - px * head_half, base_y - py * head_half),
            ]))

        # The tip point sits over the arrowhead, using the same
        # orange/blue idle/selected treatment as every other anchor.
        self._draw_point(painter, end, ENDPOINT_RADIUS, active)


class DoubleArrowMeasurement(MeasurementButton):
    # A 2-point line preset to arrow caps on both ends — see the "Double
    # Arrow" MeasurementKind's meta_preset.
    name = "Double Arrow"
    display_name = "Dbl Arrow"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        start = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        end = QPoint(rect.right() - LINE_MARGIN, rect.top() + LINE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(start, end)

        dx, dy = end.x() - start.x(), end.y() - start.y()
        length = (dx * dx + dy * dy) ** 0.5
        if length > 0:
            ux, uy = dx / length, dy / length
            px, py = -uy, ux
            head_len, head_half = 9.0, 4.0
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(LINE_COLOR)
            for tip, back in ((end, (-ux, -uy)), (start, (ux, uy))):
                base_x, base_y = tip.x() + back[0] * head_len, tip.y() + back[1] * head_len
                painter.drawPolygon(QPolygonF([
                    QPointF(tip),
                    QPointF(base_x + px * head_half, base_y + py * head_half),
                    QPointF(base_x - px * head_half, base_y - py * head_half),
                ]))

        self._draw_point(painter, start, ENDPOINT_RADIUS, active)
        self._draw_point(painter, end, ENDPOINT_RADIUS, active)


class BracketMeasurement(MeasurementButton):
    # Same kind/placement as any other 2-point line — see the "Bracket"
    # MeasurementKind's meta_preset in measurement_kind.py, which is what
    # actually gives a placed one its bracket ticks. This tile is only a
    # different icon and a shortcut into that preset.
    name = "Bracket"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        start = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        end = QPoint(rect.right() - LINE_MARGIN, rect.top() + LINE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(start, end)

        dx, dy = end.x() - start.x(), end.y() - start.y()
        length = (dx * dx + dy * dy) ** 0.5
        if length > 0:
            ux, uy = dx / length, dy / length
            px, py = -uy, ux
            tick_half = 5.0
            for pt in (start, end):
                painter.drawLine(
                    QPointF(pt.x() + px * tick_half, pt.y() + py * tick_half),
                    QPointF(pt.x() - px * tick_half, pt.y() - py * tick_half),
                )
        self._draw_point(painter, start, ENDPOINT_RADIUS, active)
        self._draw_point(painter, end, ENDPOINT_RADIUS, active)


class HorizontalLineMeasurement(MeasurementButton):
    name = "Horizontal Line"
    display_name = "Horizontal"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        y = rect.center().y()
        start = QPoint(rect.left() + LINE_MARGIN, y)
        end = QPoint(rect.right() - LINE_MARGIN, y)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(start, end)
        self._draw_point(painter, start, ENDPOINT_RADIUS, active)
        self._draw_point(painter, end, ENDPOINT_RADIUS, active)


class VerticalLineMeasurement(MeasurementButton):
    name = "Vertical Line"
    display_name = "Vertical"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        x = rect.center().x()
        start = QPoint(x, rect.top() + LINE_MARGIN)
        end = QPoint(x, rect.bottom() - LINE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(start, end)
        self._draw_point(painter, start, ENDPOINT_RADIUS, active)
        self._draw_point(painter, end, ENDPOINT_RADIUS, active)
