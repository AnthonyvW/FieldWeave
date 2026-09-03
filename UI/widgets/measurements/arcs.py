from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRect, QRectF
from PySide6.QtGui import QPainter, QPainterPath

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import CIRCLE_MARGIN, ENDPOINT_RADIUS, LINE_COLOR


class ThreePointArcMeasurement(MeasurementButton):
    name = "3 Point Arc"
    display_name = "3pt Arc"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        circle_rect = QRectF(rect).adjusted(CIRCLE_MARGIN, CIRCLE_MARGIN, -CIRCLE_MARGIN, -CIRCLE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        path = QPainterPath()
        path.arcMoveTo(circle_rect, 200)
        path.arcTo(circle_rect, 200, -160)
        painter.drawPath(path)

        for angle_deg in (200, 120, 40):
            angle = math.radians(angle_deg)
            cx, cy = circle_rect.center().x(), circle_rect.center().y()
            rx, ry = circle_rect.width() / 2, circle_rect.height() / 2
            point = QPointF(cx + rx * math.cos(angle), cy - ry * math.sin(angle))
            self._draw_point(painter, point, ENDPOINT_RADIUS, active)


class RadiusArcMeasurement(MeasurementButton):
    name = "Radius Arc"
    display_name = "Radius Arc"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        circle_rect = QRectF(rect).adjusted(CIRCLE_MARGIN, CIRCLE_MARGIN, -CIRCLE_MARGIN, -CIRCLE_MARGIN)
        center = circle_rect.center()

        self._set_pen(painter, LINE_COLOR)
        path = QPainterPath()
        path.arcMoveTo(circle_rect, 20)
        path.arcTo(circle_rect, 20, 100)
        painter.drawPath(path)

        rx, ry = circle_rect.width() / 2, circle_rect.height() / 2
        for angle_deg in (20, 120):
            angle = math.radians(angle_deg)
            edge = QPointF(center.x() + rx * math.cos(angle), center.y() - ry * math.sin(angle))
            painter.drawLine(center, edge)
            self._draw_point(painter, edge, ENDPOINT_RADIUS, active)
        self._draw_point(painter, center, ENDPOINT_RADIUS, active)
