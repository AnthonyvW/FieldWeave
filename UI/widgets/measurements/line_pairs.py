from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QPainter, QPen

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import ENDPOINT_RADIUS, LINE_COLOR, LINE_MARGIN


def _dashed(painter: QPainter, a: QPoint, b: QPoint) -> None:
    pen = QPen(LINE_COLOR)
    pen.setWidth(1)
    pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.drawLine(a, b)


class ThreePointParallelMeasurement(MeasurementButton):
    name = "3pt Parallel"
    display_name = "3pt Parallel"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        offset = (rect.bottom() - rect.top()) // 3
        a0 = QPoint(rect.left() + LINE_MARGIN, rect.top() + LINE_MARGIN + offset)
        a1 = QPoint(rect.right() - LINE_MARGIN, rect.top() + LINE_MARGIN + offset)
        b0 = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        b1 = QPoint(rect.right() - LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        center = QPoint((b0.x() + b1.x()) // 2, b0.y())

        _dashed(painter, a0, b0)
        _dashed(painter, a1, b1)
        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(a0, a1)
        painter.drawLine(b0, b1)
        self._draw_point(painter, a0, ENDPOINT_RADIUS, active)
        self._draw_point(painter, a1, ENDPOINT_RADIUS, active)
        self._draw_point(painter, center, ENDPOINT_RADIUS, active)


class FourPointParallelMeasurement(MeasurementButton):
    name = "4pt Parallel"
    display_name = "4pt Parallel"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        offset = (rect.bottom() - rect.top()) // 3
        a0 = QPoint(rect.left() + LINE_MARGIN, rect.top() + LINE_MARGIN + offset)
        a1 = QPoint(rect.right() - LINE_MARGIN, rect.top() + LINE_MARGIN + offset)
        b0 = QPoint(rect.left() + LINE_MARGIN + 4, rect.bottom() - LINE_MARGIN)
        b1 = QPoint(rect.right() - LINE_MARGIN - 4, rect.bottom() - LINE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(a0, a1)
        painter.drawLine(b0, b1)
        for point in (a0, a1, b0, b1):
            self._draw_point(painter, point, ENDPOINT_RADIUS, active)


class EightPointParallelMeasurement(MeasurementButton):
    name = "8pt Parallel"
    display_name = "8pt Parallel"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        x0 = rect.left() + LINE_MARGIN
        x1 = rect.right() - LINE_MARGIN
        top = rect.top() + LINE_MARGIN
        bottom = rect.bottom() - LINE_MARGIN
        span = bottom - top
        # Two pairs of parallel lines with a dashed midline through each.
        ys = [top, top + span // 6, top + span * 4 // 6, top + span * 5 // 6]
        self._set_pen(painter, LINE_COLOR)
        for y in ys:
            painter.drawLine(QPoint(x0, y), QPoint(x1, y))
        _dashed(painter, QPoint(x0, (ys[0] + ys[1]) // 2), QPoint(x1, (ys[0] + ys[1]) // 2))
        _dashed(painter, QPoint(x0, (ys[2] + ys[3]) // 2), QPoint(x1, (ys[2] + ys[3]) // 2))


class ArbitraryParallelMeasurement(MeasurementButton):
    name = "Arbitrary Parallel"
    display_name = "Arb. Parallel"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        x0 = rect.left() + LINE_MARGIN
        x1 = rect.right() - LINE_MARGIN
        top = rect.top() + LINE_MARGIN
        bottom = rect.bottom() - LINE_MARGIN
        span = bottom - top
        self._set_pen(painter, LINE_COLOR)
        for i in range(3):
            y = top + span * i // 2
            painter.drawLine(QPoint(x0, y), QPoint(x1, y))
        self._draw_point(painter, QPoint(x0, top), ENDPOINT_RADIUS, active)
        self._draw_point(painter, QPoint(x1, top), ENDPOINT_RADIUS, active)


class ThreePointPerpMeasurement(MeasurementButton):
    name = "3pt Perp"
    display_name = "3pt Perp"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        a0 = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        a1 = QPoint(rect.right() - LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        foot = QPoint(rect.center().x(), rect.bottom() - LINE_MARGIN)
        tip = QPoint(rect.center().x(), rect.top() + LINE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(a0, a1)
        _dashed(painter, foot, tip)
        self._draw_point(painter, a0, ENDPOINT_RADIUS, active)
        self._draw_point(painter, a1, ENDPOINT_RADIUS, active)
        self._draw_point(painter, tip, ENDPOINT_RADIUS, active)


class FourPointPerpMeasurement(MeasurementButton):
    name = "4pt Perp"
    display_name = "4pt Perp"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        a0 = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        a1 = QPoint(rect.right() - LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        x = rect.center().x()
        foot = QPoint(x, rect.bottom() - LINE_MARGIN)
        tip = QPoint(x, rect.top() + LINE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(a0, a1)
        _dashed(painter, foot, tip)
        for point in (a0, a1, foot, tip):
            self._draw_point(painter, point, ENDPOINT_RADIUS, active)


class ArbitraryPerpMeasurement(MeasurementButton):
    name = "Arbitrary Perp"
    display_name = "Arb. Perp"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        a0 = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        a1 = QPoint(rect.right() - LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        top = rect.top() + LINE_MARGIN
        base = rect.bottom() - LINE_MARGIN

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(a0, a1)
        for frac in (1, 2):
            x = rect.left() + LINE_MARGIN + (rect.width() - 2 * LINE_MARGIN) * frac // 3
            _dashed(painter, QPoint(x, base), QPoint(x, top))
        self._draw_point(painter, a0, ENDPOINT_RADIUS, active)
        self._draw_point(painter, a1, ENDPOINT_RADIUS, active)
