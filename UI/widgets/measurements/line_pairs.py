from __future__ import annotations

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QPainter

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import ENDPOINT_RADIUS, LINE_COLOR, LINE_MARGIN


class ParallelLineMeasurement(MeasurementButton):
    name = "Parallel Line"
    display_name = "Parallel"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        offset = (rect.bottom() - rect.top()) // 4
        a0 = QPoint(rect.left() + LINE_MARGIN, rect.top() + LINE_MARGIN + offset)
        a1 = QPoint(rect.right() - LINE_MARGIN, rect.top() + LINE_MARGIN)
        b0 = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        b1 = QPoint(rect.right() - LINE_MARGIN, rect.bottom() - LINE_MARGIN - offset)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(a0, a1)
        painter.drawLine(b0, b1)
        for point in (a0, a1, b0):
            self._draw_point(painter, point, ENDPOINT_RADIUS, active)


class PerpendicularLineMeasurement(MeasurementButton):
    name = "Perpendicular Line"
    display_name = "Perpendicular"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        a0 = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        a1 = QPoint(rect.right() - LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        foot = QPoint(rect.center().x(), rect.bottom() - LINE_MARGIN)
        tip = QPoint(rect.center().x(), rect.top() + LINE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(a0, a1)
        painter.drawLine(foot, tip)
        self._draw_point(painter, a0, ENDPOINT_RADIUS, active)
        self._draw_point(painter, a1, ENDPOINT_RADIUS, active)
        self._draw_point(painter, tip, ENDPOINT_RADIUS, active)
