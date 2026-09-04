from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.lines import ArrowMeasurement
from UI.widgets.measurements.measurement_style import LINE_COLOR, LINE_MARGIN


class TextMeasurement(MeasurementButton):
    # Also the Annotation category's icon: a hollow box with black outline and a black capital T.
    name = "Text"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        box = QRectF(rect).adjusted(LINE_MARGIN, LINE_MARGIN, -LINE_MARGIN, -LINE_MARGIN)
        pen = QPen(QColor("#000000"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(box)

        font = QFont(painter.font())
        font.setBold(True)
        font.setPixelSize(int(box.height() * 0.8))
        painter.setFont(font)
        painter.setPen(QColor("#000000"))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, "T")


class ScaleBarMeasurement(MeasurementButton):
    name = "Scale Bar"
    display_name = "Scale Bar"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        left = rect.left() + LINE_MARGIN
        right = rect.right() - LINE_MARGIN
        y = rect.center().y()
        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(QPoint(left, y), QPoint(right, y))
        # End ticks.
        painter.drawLine(QPoint(left, y - 5), QPoint(left, y + 5))
        painter.drawLine(QPoint(right, y - 5), QPoint(right, y + 5))

        font = QFont(painter.font())
        font.setPixelSize(9)
        painter.setFont(font)
        painter.setPen(LINE_COLOR)
        painter.drawText(QRectF(left, y + 4, right - left, 12), Qt.AlignmentFlag.AlignHCenter, "10")


class AnnotationArrowMeasurement(ArrowMeasurement):
    # Same 2-point arrow as the regular "Arrow" tile (its length tag hidden
    # by default — see the MeasurementKind's meta_preset) — reuses
    # ArrowMeasurement's icon so both tiles look identical.
    name = "Annotation Arrow"
    display_name = "Arrow"
