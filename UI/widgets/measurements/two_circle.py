from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QPainter, QPen

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import ENDPOINT_RADIUS, LINE_COLOR


def _two_rects(rect: QRect) -> tuple[QRectF, QRectF]:
    d = min(rect.width(), rect.height()) * 0.42
    left = QRectF(rect.left() + 4, rect.center().y() - d / 2, d, d)
    right = QRectF(rect.right() - 4 - d, rect.center().y() - d / 2, d, d)
    return left, right


def _connect_centers(painter: QPainter, a: QRectF, b: QRectF) -> None:
    pen = QPen(LINE_COLOR)
    pen.setWidth(1)
    pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.drawLine(a.center(), b.center())


class RadiusTwoCircleMeasurement(MeasurementButton):
    name = "Radius 2 Circle"
    display_name = "R 2Circle"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        a, b = _two_rects(rect)
        _connect_centers(painter, a, b)
        self._set_pen(painter, LINE_COLOR)
        painter.drawEllipse(a)
        painter.drawEllipse(b)
        for r in (a, b):
            self._draw_point(painter, r.center(), ENDPOINT_RADIUS, active)
            self._draw_point(painter, QPointF(r.center().x() + r.width() / 2, r.center().y()), ENDPOINT_RADIUS, active)


class ThreePointTwoCircleMeasurement(MeasurementButton):
    name = "3pt 2 Circle"
    display_name = "3pt 2Circle"

    _POINT_ANGLES_DEG = (90, 210, 330)

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        a, b = _two_rects(rect)
        _connect_centers(painter, a, b)
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
        _connect_centers(painter, a, b)
        self._set_pen(painter, LINE_COLOR)
        painter.drawEllipse(a)
        painter.drawEllipse(b)
        for r in (a, b):
            left = QPointF(r.center().x() - r.width() / 2, r.center().y())
            right = QPointF(r.center().x() + r.width() / 2, r.center().y())
            self._draw_point(painter, left, ENDPOINT_RADIUS, active)
            self._draw_point(painter, right, ENDPOINT_RADIUS, active)
