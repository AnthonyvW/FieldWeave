from __future__ import annotations

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QPainter

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import ENDPOINT_RADIUS, LINE_COLOR, LINE_MARGIN


class CalibrationLineButton(MeasurementButton):
    """
    Same icon as Arbitrary Line — placing a manual calibration reference
    is the same click-two-points action. Checked state is driven
    explicitly by MeasurementsWidget (not the exclusive measurement-type
    button group) to signal "calibration mode is active" for as long as
    the calibration panel is open, giving a persistent visual answer to
    "did my click actually start calibrating?" — see
    MeasurementsWidget._start_manual_calibration.
    """

    name = "Manual Calibration"
    display_name = "Manual Cal."

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        start = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        end = QPoint(rect.right() - LINE_MARGIN, rect.top() + LINE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(start, end)
        self._draw_point(painter, start, ENDPOINT_RADIUS, active)
        self._draw_point(painter, end, ENDPOINT_RADIUS, active)