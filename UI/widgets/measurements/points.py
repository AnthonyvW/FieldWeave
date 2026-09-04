from __future__ import annotations

from PySide6.QtCore import QRect
from PySide6.QtGui import QPainter

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import POINT_RADIUS

# A placed "Point" measurement's marker shape (meta.point_style) — see
# MeasurementOverlay._draw_point_marker for how each renders.
MEASUREMENT_POINT_STYLES = ("dot", "circle", "square", "diamond", "cross", "x", "triangle")


class PointMeasurement(MeasurementButton):
    name = "Point"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        self._draw_point(painter, rect.center(), POINT_RADIUS, active)