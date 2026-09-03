from __future__ import annotations

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QPainter

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import ENDPOINT_RADIUS, LINE_COLOR, LINE_MARGIN


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
        self._draw_point(painter, vertex, ENDPOINT_RADIUS, active)
        self._draw_point(painter, arm_a, ENDPOINT_RADIUS, active)
        self._draw_point(painter, arm_b, ENDPOINT_RADIUS, active)


class FourPointAngleMeasurement(MeasurementButton):
    name = "4 Point Angle"
    display_name = "4pt Angle"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        a0 = QPoint(rect.left() + LINE_MARGIN, rect.top() + LINE_MARGIN)
        a1 = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        b0 = QPoint(rect.center().x(), rect.bottom() - LINE_MARGIN)
        b1 = QPoint(rect.right() - LINE_MARGIN, rect.top() + LINE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(a0, a1)
        painter.drawLine(b0, b1)
        for point in (a0, a1, b0, b1):
            self._draw_point(painter, point, ENDPOINT_RADIUS, active)
