from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRect, QRectF
from PySide6.QtGui import QPainter

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import CIRCLE_MARGIN, ENDPOINT_RADIUS, LINE_COLOR

_INNER_SCALE = 0.5


def _rings(rect: QRect) -> tuple[QPointF, float, float]:
    outer = QRectF(rect).adjusted(CIRCLE_MARGIN, CIRCLE_MARGIN, -CIRCLE_MARGIN, -CIRCLE_MARGIN)
    center = outer.center()
    return center, outer.width() / 2, outer.width() / 2 * _INNER_SCALE


def _draw_rings(painter: QPainter, button: MeasurementButton, center: QPointF, outer_r: float, inner_r: float) -> None:
    button._set_pen(painter, LINE_COLOR)
    painter.drawEllipse(center, outer_r, outer_r)
    painter.drawEllipse(center, inner_r, inner_r)


class RadiusAnnulusMeasurement(MeasurementButton):
    name = "Radius Annulus"
    display_name = "R Annulus"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        center, outer_r, inner_r = _rings(rect)
        _draw_rings(painter, self, center, outer_r, inner_r)
        self._draw_point(painter, center, ENDPOINT_RADIUS, active)
        self._draw_point(painter, QPointF(center.x() + outer_r, center.y()), ENDPOINT_RADIUS, active)
        self._draw_point(painter, QPointF(center.x() + inner_r, center.y()), ENDPOINT_RADIUS, active)


class ThreePointAnnulusMeasurement(MeasurementButton):
    name = "3pt Annulus"
    display_name = "3pt Annulus"

    _POINT_ANGLES_DEG = (90, 210, 330)

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        center, outer_r, inner_r = _rings(rect)
        _draw_rings(painter, self, center, outer_r, inner_r)
        for angle_deg in self._POINT_ANGLES_DEG:
            angle = math.radians(angle_deg)
            self._draw_point(
                painter, QPointF(center.x() + outer_r * math.cos(angle), center.y() - outer_r * math.sin(angle)),
                ENDPOINT_RADIUS, active,
            )
        self._draw_point(painter, QPointF(center.x() + inner_r, center.y()), ENDPOINT_RADIUS, active)


class DiameterAnnulusMeasurement(MeasurementButton):
    name = "Diameter Annulus"
    display_name = "D Annulus"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        center, outer_r, inner_r = _rings(rect)
        _draw_rings(painter, self, center, outer_r, inner_r)
        # Diagonal diameter across the inner circle only, its two ends the
        # points that place it; the outer ring gets a single point.
        diag = inner_r / math.sqrt(2)
        inner_a = QPointF(center.x() - diag, center.y() + diag)
        inner_b = QPointF(center.x() + diag, center.y() - diag)
        painter.drawLine(inner_a, inner_b)
        self._draw_point(painter, inner_a, ENDPOINT_RADIUS, active)
        self._draw_point(painter, inner_b, ENDPOINT_RADIUS, active)
        self._draw_point(painter, QPointF(center.x() + outer_r, center.y()), ENDPOINT_RADIUS, active)
