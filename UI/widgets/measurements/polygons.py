from __future__ import annotations

import math

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QPainter

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import CIRCLE_MARGIN, ENDPOINT_RADIUS, LINE_COLOR


class PolygonMeasurement(MeasurementButton):
    name = "Polygon"

    _SIDES = 5

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        cx, cy = rect.center().x(), rect.center().y()
        radius = min(rect.width(), rect.height()) / 2 - CIRCLE_MARGIN
        verts = []
        for i in range(self._SIDES):
            angle = math.radians(-90 + 360 * i / self._SIDES)
            verts.append(QPoint(round(cx + radius * math.cos(angle)), round(cy + radius * math.sin(angle))))

        self._set_pen(painter, LINE_COLOR)
        painter.drawPolyline(verts + [verts[0]])
        for vertex in verts:
            self._draw_point(painter, vertex, ENDPOINT_RADIUS, active)
