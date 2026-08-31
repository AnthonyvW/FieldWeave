from __future__ import annotations

from PySide6.QtCore import QRect
from PySide6.QtGui import QPainter

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import POINT_RADIUS


class PointMeasurement(MeasurementButton):
    name = "Point"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        self._draw_point(painter, rect.center(), POINT_RADIUS, active)