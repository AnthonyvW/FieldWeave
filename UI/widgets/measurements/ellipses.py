from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRect, QRectF
from PySide6.QtGui import QPainter

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import CIRCLE_MARGIN, ENDPOINT_RADIUS, LINE_COLOR


class ThreePointEllipseMeasurement(MeasurementButton):
    name = "3 Point Ellipse"
    display_name = "3pt Ellipse"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        ellipse_rect = QRectF(rect).adjusted(CIRCLE_MARGIN, CIRCLE_MARGIN * 2, -CIRCLE_MARGIN, -CIRCLE_MARGIN * 2)
        center = ellipse_rect.center()
        minor_top = QPointF(center.x(), ellipse_rect.top())
        minor_bottom = QPointF(center.x(), ellipse_rect.bottom())
        major_edge = QPointF(ellipse_rect.right(), center.y())

        self._set_pen(painter, LINE_COLOR)
        painter.drawEllipse(ellipse_rect)

        self._draw_point(painter, minor_top, ENDPOINT_RADIUS, active)
        self._draw_point(painter, minor_bottom, ENDPOINT_RADIUS, active)
        self._draw_point(painter, major_edge, ENDPOINT_RADIUS, active)


class FivePointEllipseMeasurement(MeasurementButton):
    name = "5 Point Ellipse"
    display_name = "5pt Ellipse"

    # Spread five sample points evenly around the ellipse, matching
    # neither the fit algorithm nor any particular click order — this is
    # just an icon.
    _POINT_ANGLES_DEG = (90, 162, 234, 306, 18)

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        ellipse_rect = QRectF(rect).adjusted(CIRCLE_MARGIN, CIRCLE_MARGIN * 2, -CIRCLE_MARGIN, -CIRCLE_MARGIN * 2)
        center = ellipse_rect.center()
        rx, ry = ellipse_rect.width() / 2, ellipse_rect.height() / 2

        self._set_pen(painter, LINE_COLOR)
        painter.drawEllipse(ellipse_rect)

        for angle_deg in self._POINT_ANGLES_DEG:
            angle = math.radians(angle_deg)
            point = QPointF(center.x() + rx * math.cos(angle), center.y() - ry * math.sin(angle))
            self._draw_point(painter, point, ENDPOINT_RADIUS, active)
