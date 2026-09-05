from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.lines import ArrowMeasurement
from UI.widgets.measurements.measurement_style import ENDPOINT_RADIUS, LINE_COLOR, LINE_MARGIN


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

        font = QFont(painter.font())
        font.setPixelSize(9)
        painter.setFont(font)
        painter.setPen(LINE_COLOR)
        painter.drawText(QRectF(left, y + 4, right - left, 12), Qt.AlignmentFlag.AlignHCenter, "10")


class CountMeasurement(MeasurementButton):
    # An unbounded group of numbered points — click to add each one,
    # right-click to finish (same "keeps accumulating, right-click
    # finalizes" placement Arbitrary Line uses). Icon: three unconnected
    # dots, each labeled with its own number, since points aren't joined.
    name = "Count"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        # Stacked in a column (rather than scattered around the box) so
        # each point's own number always lands in the clear space to its
        # right, at a different height than the other two — no per-point
        # direction juggling needed to dodge the icon's own edges or the
        # other labels.
        x = rect.left() + LINE_MARGIN
        top = rect.top() + LINE_MARGIN
        bottom = rect.bottom() - LINE_MARGIN
        points = (
            QPoint(x, top),
            QPoint(x, round((top + bottom) / 2)),
            QPoint(x, bottom),
        )
        font = QFont(painter.font())
        font.setPixelSize(9)
        font.setBold(True)
        painter.setFont(font)
        for i, point in enumerate(points):
            self._draw_point(painter, point, ENDPOINT_RADIUS, active)
            painter.setPen(LINE_COLOR)
            painter.drawText(
                QRectF(point.x() + 5, point.y() - 6, 14, 12), Qt.AlignmentFlag.AlignLeft, str(i + 1)
            )


class AnnotationArrowMeasurement(ArrowMeasurement):
    # Same 2-point arrow as the regular "Arrow" tile (its length tag hidden
    # by default — see the MeasurementKind's meta_preset) — reuses
    # ArrowMeasurement's icon so both tiles look identical.
    name = "Annotation Arrow"
    display_name = "Arrow"
