from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QPainter, QPainterPath, QPen

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import ENDPOINT_RADIUS, LINE_COLOR, LINE_MARGIN


class CurveMeasurement(MeasurementButton):
    name = "Curve"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        start = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        end = QPoint(rect.right() - LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        # The apex is a point *on* the curve, matching how a placed Curve
        # treats its third click; the control point is solved back from it.
        apex = QPoint(rect.center().x(), rect.top() + LINE_MARGIN)
        control = QPoint(
            2 * apex.x() - (start.x() + end.x()) // 2,
            2 * apex.y() - (start.y() + end.y()) // 2,
        )

        # A dashed straight chord between the two ends distinguishes this
        # tile from the 3-point arc, whose middle point also lies on its curve.
        dash_pen = QPen(LINE_COLOR)
        dash_pen.setWidth(1)
        dash_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(dash_pen)
        painter.drawLine(start, end)

        self._set_pen(painter, LINE_COLOR)
        path = QPainterPath()
        path.moveTo(start)
        path.quadTo(control, end)
        painter.drawPath(path)

        self._draw_point(painter, start, ENDPOINT_RADIUS, active)
        self._draw_point(painter, end, ENDPOINT_RADIUS, active)
        self._draw_point(painter, apex, ENDPOINT_RADIUS, active)
