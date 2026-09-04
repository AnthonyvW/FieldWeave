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
        x0 = rect.left() + LINE_MARGIN
        x1 = rect.right() - LINE_MARGIN
        top_y = rect.top() + LINE_MARGIN + 2
        bottom_y = rect.bottom() - LINE_MARGIN
        a0, a1 = QPoint(x0, top_y), QPoint(x1, top_y)
        b0, b1 = QPoint(x0, bottom_y), QPoint(x1, bottom_y)
        mid = QPoint((x0 + x1) // 2, bottom_y)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(a0, a1)
        painter.drawLine(b0, b1)
        # Perpendicular dimension connector from the second line to the first.
        _dashed(painter, QPoint(mid.x(), bottom_y), QPoint(mid.x(), top_y))
        self._draw_point(painter, a0, ENDPOINT_RADIUS, active)
        self._draw_point(painter, a1, ENDPOINT_RADIUS, active)
        self._draw_point(painter, mid, ENDPOINT_RADIUS, active)


class FourPointParallelMeasurement(MeasurementButton):
    name = "4pt Parallel"
    display_name = "4pt Parallel"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        x0 = rect.left() + LINE_MARGIN
        x1 = rect.right() - LINE_MARGIN
        top_y = rect.top() + LINE_MARGIN + 2
        bottom_y = rect.bottom() - LINE_MARGIN
        a0, a1 = QPoint(x0, top_y), QPoint(x1, top_y)
        b0, b1 = QPoint(x0 + 4, bottom_y), QPoint(x1 - 4, bottom_y)
        cx = (b0.x() + b1.x()) // 2

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(a0, a1)
        painter.drawLine(b0, b1)
        _dashed(painter, QPoint(cx, bottom_y), QPoint(cx, top_y))
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
        slope = span // 6
        # Four parallel diagonal lines, each with anchor points at its ends.
        self._set_pen(painter, LINE_COLOR)
        ends = []
        for i in range(4):
            y = top + span * i // 3
            p0 = QPoint(x0, min(bottom, y + slope))
            p1 = QPoint(x1, y)
            painter.drawLine(p0, p1)
            ends.append((p0, p1))
        for p0, p1 in ends:
            self._draw_point(painter, p0, ENDPOINT_RADIUS, active)
            self._draw_point(painter, p1, ENDPOINT_RADIUS, active)


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

        # Both legs solid.
        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(a0, a1)
        painter.drawLine(foot, tip)
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
        base = rect.bottom() - LINE_MARGIN
        # The perpendicular line sits above the reference line; a dashed
        # segment drops from its lower end down to the reference line.
        seg_bottom = QPoint(x, rect.center().y())
        seg_top = QPoint(x, rect.top() + LINE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(a0, a1)
        painter.drawLine(seg_bottom, seg_top)
        _dashed(painter, QPoint(x, base), seg_bottom)
        for point in (a0, a1, seg_bottom, seg_top):
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
        # Solid perpendiculars coming off the reference line.
        for frac in (1, 2):
            x = rect.left() + LINE_MARGIN + (rect.width() - 2 * LINE_MARGIN) * frac // 3
            painter.drawLine(QPoint(x, base), QPoint(x, top))
        self._draw_point(painter, a0, ENDPOINT_RADIUS, active)
        self._draw_point(painter, a1, ENDPOINT_RADIUS, active)
