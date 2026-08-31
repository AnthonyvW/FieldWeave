from __future__ import annotations

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QPainter

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import ENDPOINT_RADIUS, LINE_COLOR, LINE_MARGIN


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