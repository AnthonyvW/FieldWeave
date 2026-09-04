from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRect, QRectF
from PySide6.QtGui import QPainter

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import ENDPOINT_RADIUS, LINE_COLOR


def _two_rects(rect: QRect) -> tuple[QRectF, QRectF]:
    # Smaller circles pushed to the far edges so the center line between
    # them reads clearly.
    d = min(rect.width(), rect.height()) * 0.34
    left = QRectF(rect.left() + 3, rect.center().y() - d / 2, d, d)
    right = QRectF(rect.right() - 3 - d, rect.center().y() - d / 2, d, d)
    return left, right


def _connect_centers(painter: QPainter, button: MeasurementButton, a: QRectF, b: QRectF) -> None:
    button._set_pen(painter, LINE_COLOR)
    painter.drawLine(a.center(), b.center())


class RadiusTwoCircleMeasurement(MeasurementButton):
    name = "Radius 2 Circle"
    display_name = "R 2Circle"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        a, b = _two_rects(rect)
        _connect_centers(painter, self, a, b)
        self._set_pen(painter, LINE_COLOR)
        painter.drawEllipse(a)
        painter.drawEllipse(b)
        for r in (a, b):
            self._draw_point(painter, r.center(), ENDPOINT_RADIUS, active)
            # Edge point set above the center so it doesn't sit on the
            # center line and blur into it.
            self._draw_point(painter, QPointF(r.center().x(), r.top()), ENDPOINT_RADIUS, active)


class ThreePointTwoCircleMeasurement(MeasurementButton):
    name = "3pt 2 Circle"
    display_name = "3pt 2Circle"

    _POINT_ANGLES_DEG = (60, 180, 300)

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        a, b = _two_rects(rect)
        _connect_centers(painter, self, a, b)
        self._set_pen(painter, LINE_COLOR)
        painter.drawEllipse(a)
        painter.drawEllipse(b)
        for r in (a, b):
            radius = r.width() / 2
            for angle_deg in self._POINT_ANGLES_DEG:
                angle = math.radians(angle_deg)
                self._draw_point(
                    painter,
                    QPointF(r.center().x() + radius * math.cos(angle), r.center().y() - radius * math.sin(angle)),
                    ENDPOINT_RADIUS, active,
                )


class DiameterTwoCircleMeasurement(MeasurementButton):
    name = "Diameter 2 Circle"
    display_name = "D 2Circle"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        a, b = _two_rects(rect)
        _connect_centers(painter, self, a, b)
        self._set_pen(painter, LINE_COLOR)
        painter.drawEllipse(a)
        painter.drawEllipse(b)
        for r in (a, b):
            # Vertical diameter so the endpoints sit off the horizontal
            # center line.
            top = QPointF(r.center().x(), r.top())
            bottom = QPointF(r.center().x(), r.bottom())
            self._draw_point(painter, top, ENDPOINT_RADIUS, active)
            self._draw_point(painter, bottom, ENDPOINT_RADIUS, active)
