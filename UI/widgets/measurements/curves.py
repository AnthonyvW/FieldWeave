from __future__ import annotations

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QPainter, QPainterPath

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import ENDPOINT_RADIUS, LINE_COLOR, LINE_MARGIN


class CurveMeasurement(MeasurementButton):
    name = "Curve"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        start = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        end = QPoint(rect.right() - LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        control = QPoint(rect.center().x(), rect.top() + LINE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        path = QPainterPath()
        path.moveTo(start)
        path.quadTo(control, end)
        painter.drawPath(path)

        self._draw_point(painter, start, ENDPOINT_RADIUS, active)
        self._draw_point(painter, end, ENDPOINT_RADIUS, active)
        self._draw_point(painter, control, ENDPOINT_RADIUS, active)
