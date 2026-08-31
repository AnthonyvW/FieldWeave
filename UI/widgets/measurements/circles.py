from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRect, QRectF
from PySide6.QtGui import QPainter

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import CIRCLE_MARGIN, ENDPOINT_RADIUS, LINE_COLOR


class RadiusCircleMeasurement(MeasurementButton):
    name = "Radius Circle"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        # QRectF, not QRect: QRect.center() truncates to an integer via
        # right()/bottom() (x + width - 1), which lands slightly off the
        # ellipse's true geometric center that drawEllipse(rect) actually
        # draws around — enough to visibly misalign the points below.
        circle_rect = QRectF(rect).adjusted(CIRCLE_MARGIN, CIRCLE_MARGIN, -CIRCLE_MARGIN, -CIRCLE_MARGIN)
        center = circle_rect.center()
        edge = QPointF(center.x() + circle_rect.width() / 2, center.y())

        self._set_pen(painter, LINE_COLOR)
        painter.drawEllipse(circle_rect)
        painter.drawLine(center, edge)

        self._draw_point(painter, center, ENDPOINT_RADIUS, active)
        self._draw_point(painter, edge, ENDPOINT_RADIUS, active)


class DiameterMeasurement(MeasurementButton):
    name = "Diameter"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        circle_rect = QRectF(rect).adjusted(CIRCLE_MARGIN, CIRCLE_MARGIN, -CIRCLE_MARGIN, -CIRCLE_MARGIN)
        center = circle_rect.center()
        left = QPointF(center.x() - circle_rect.width() / 2, center.y())
        right = QPointF(center.x() + circle_rect.width() / 2, center.y())

        self._set_pen(painter, LINE_COLOR)
        painter.drawEllipse(circle_rect)
        painter.drawLine(left, right)

        self._draw_point(painter, left, ENDPOINT_RADIUS, active)
        self._draw_point(painter, right, ENDPOINT_RADIUS, active)


class ThreePointCircleMeasurement(MeasurementButton):
    name = "3 Point Circle"
    display_name = "3pt Circle"

    # Spread the three sample points evenly around the circle rather than
    # matching any real fit, since this is just an icon, not a diagram of
    # the actual circumcircle algorithm.
    _POINT_ANGLES_DEG = (90, 210, 330)

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        circle_rect = QRectF(rect).adjusted(CIRCLE_MARGIN, CIRCLE_MARGIN, -CIRCLE_MARGIN, -CIRCLE_MARGIN)
        center = circle_rect.center()
        radius = circle_rect.width() / 2

        self._set_pen(painter, LINE_COLOR)
        painter.drawEllipse(circle_rect)

        for angle_deg in self._POINT_ANGLES_DEG:
            angle = math.radians(angle_deg)
            point = QPointF(center.x() + radius * math.cos(angle), center.y() - radius * math.sin(angle))
            self._draw_point(painter, point, ENDPOINT_RADIUS, active)