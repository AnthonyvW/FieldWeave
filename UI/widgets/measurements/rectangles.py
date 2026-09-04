from __future__ import annotations

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QPainter

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import CIRCLE_MARGIN, ENDPOINT_RADIUS, LINE_COLOR


def _rect_frame(rect: QRect, inset: int) -> tuple[QPoint, QPoint, QPoint, QPoint]:
    tl = QPoint(rect.left() + inset, rect.top() + inset)
    tr = QPoint(rect.right() - inset, rect.top() + inset)
    br = QPoint(rect.right() - inset, rect.bottom() - inset)
    bl = QPoint(rect.left() + inset, rect.bottom() - inset)
    return tl, tr, br, bl


class TwoPointRectangleMeasurement(MeasurementButton):
    name = "2pt Rectangle"
    display_name = "2pt Rect"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        tl, tr, br, bl = _rect_frame(rect, CIRCLE_MARGIN + 2)
        self._set_pen(painter, LINE_COLOR)
        painter.drawPolyline([tl, tr, br, bl, tl])
        self._draw_point(painter, tl, ENDPOINT_RADIUS, active)
        self._draw_point(painter, br, ENDPOINT_RADIUS, active)


class ThreePointRectangleMeasurement(MeasurementButton):
    name = "3pt Rectangle"
    display_name = "3pt Rect"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        tl, tr, br, bl = _rect_frame(rect, CIRCLE_MARGIN + 2)
        self._set_pen(painter, LINE_COLOR)
        painter.drawPolyline([tl, tr, br, bl, tl])
        for point in (bl, br, tr):
            self._draw_point(painter, point, ENDPOINT_RADIUS, active)


class TwoPointSquareMeasurement(MeasurementButton):
    name = "2pt Square"
    display_name = "2pt Square"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        side = min(rect.width(), rect.height()) - 2 * (CIRCLE_MARGIN + 2)
        left = rect.center().x() - side // 2
        top = rect.center().y() - side // 2
        tl = QPoint(left, top)
        tr = QPoint(left + side, top)
        br = QPoint(left + side, top + side)
        bl = QPoint(left, top + side)
        self._set_pen(painter, LINE_COLOR)
        painter.drawPolyline([tl, tr, br, bl, tl])
        self._draw_point(painter, bl, ENDPOINT_RADIUS, active)
        self._draw_point(painter, br, ENDPOINT_RADIUS, active)


