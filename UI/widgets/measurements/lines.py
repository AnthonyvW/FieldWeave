from __future__ import annotations

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QPainter, QPainterPath

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
MEASUREMENT_LINE_CAPS = ("curved", "square", "arrow", "arrow_open")


def arrow_dims(line_width: float) -> tuple[float, float]:
    """(length, half-width) of an arrowhead sized for *line_width* (already counter-scaled to screen pixels) — shared by the solid and open arrow caps so they read as the same size arrowhead."""
    return max(line_width * 4, 10.0), max(line_width * 2.2, 5.0)


def arrow_head_path(line_width: float) -> QPainterPath:
    """Closed triangle for a solid ("arrow") arrowhead, tip at the origin and pointing in the -x direction — the caller translates/rotates it onto a segment's actual endpoint."""
    arrow_len, arrow_half_width = arrow_dims(line_width)
    path = QPainterPath()
    path.moveTo(0, 0)
    path.lineTo(-arrow_len, -arrow_half_width)
    path.lineTo(-arrow_len, arrow_half_width)
    path.closeSubpath()
    return path


def open_arrow_barbs_path(line_width: float) -> QPainterPath:
    """Two open barb strokes flaring back from the origin for an ("arrow_open") arrowhead — an open path meant to be stroked rather than filled, tip at the origin and pointing in the -x direction like arrow_head_path."""
    arrow_len, arrow_half_width = arrow_dims(line_width)
    path = QPainterPath()
    path.moveTo(0, 0)
    path.lineTo(-arrow_len, -arrow_half_width)
    path.moveTo(0, 0)
    path.lineTo(-arrow_len, arrow_half_width)
    return path


class ArbitraryLineMeasurement(MeasurementButton):
    name = "Arbitrary Line"
    display_name = "Arb. Line"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        start = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        end = QPoint(rect.right() - LINE_MARGIN, rect.top() + LINE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(start, end)
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
