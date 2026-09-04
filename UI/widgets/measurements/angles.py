from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QRectF, Qt
from PySide6.QtGui import QPainter

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import ENDPOINT_RADIUS, LINE_COLOR, LINE_MARGIN

# Radius (icon pixels) of the small curved-dotted-line indicating the
# angle itself, drawn at whichever point the two legs meet or cross —
# see MeasurementOverlay._draw_angle_indicator for the real, on-preview
# equivalent this previews.
_INDICATOR_RADIUS = 9


class ThreePointAngleMeasurement(MeasurementButton):
    name = "3 Point Angle"
    display_name = "3pt Angle"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        vertex = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        arm_a = QPoint(rect.left() + LINE_MARGIN, rect.top() + LINE_MARGIN)
        arm_b = QPoint(rect.right() - LINE_MARGIN, rect.bottom() - LINE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(vertex, arm_a)
        painter.drawLine(vertex, arm_b)

        # arm_a sits straight up from vertex (Qt angle 90) and arm_b
        # straight out to the right (Qt angle 0) — the dotted arc
        # between them previews where MeasurementOverlay would draw the
        # real angle indicator.
        dot_pen = painter.pen()
        dot_pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(dot_pen)
        indicator_rect = QRectF(
            vertex.x() - _INDICATOR_RADIUS, vertex.y() - _INDICATOR_RADIUS,
            2 * _INDICATOR_RADIUS, 2 * _INDICATOR_RADIUS,
        )
        painter.drawArc(indicator_rect, 0, 90 * 16)

        self._draw_point(painter, vertex, ENDPOINT_RADIUS, active)
        self._draw_point(painter, arm_a, ENDPOINT_RADIUS, active)
        self._draw_point(painter, arm_b, ENDPOINT_RADIUS, active)


class FourPointAngleMeasurement(MeasurementButton):
    name = "4 Point Angle"
    display_name = "4pt Angle"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        # Two diagonals spanning the tile corner-to-corner, crossing at
        # its center — a clean, unambiguous crossing point for the
        # dotted angle-indicator arc to sit at, standing in for wherever
        # two independently-placed segments happen to cross.
        a0 = QPoint(rect.left() + LINE_MARGIN, rect.top() + LINE_MARGIN)
        a1 = QPoint(rect.right() - LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        b0 = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        b1 = QPoint(rect.right() - LINE_MARGIN, rect.top() + LINE_MARGIN)
        center = rect.center()

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(a0, a1)
        painter.drawLine(b0, b1)

        dot_pen = painter.pen()
        dot_pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(dot_pen)
        indicator_rect = QRectF(
            center.x() - _INDICATOR_RADIUS, center.y() - _INDICATOR_RADIUS,
            2 * _INDICATOR_RADIUS, 2 * _INDICATOR_RADIUS,
        )
        # The right-hand wedge between the two diagonals (Qt angles -45
        # to +45, i.e. toward a1 and b1's shared corner) reads clearly
        # against the X they form.
        painter.drawArc(indicator_rect, -45 * 16, 90 * 16)

        for point in (a0, a1, b0, b1):
            self._draw_point(painter, point, ENDPOINT_RADIUS, active)
