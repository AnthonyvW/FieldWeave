from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPolygonF

from UI.widgets.measurements.base_measurement import MeasurementButton
from UI.widgets.measurements.measurement_style import ENDPOINT_RADIUS, LINE_COLOR, LINE_MARGIN


class TextMeasurement(MeasurementButton):
    # Also the Annotation category's icon: a black box with a white capital T.
    name = "Text"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        box = QRectF(rect).adjusted(LINE_MARGIN, LINE_MARGIN, -LINE_MARGIN, -LINE_MARGIN)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#000000"))
        painter.drawRect(box)

        font = QFont(painter.font())
        font.setBold(True)
        font.setPixelSize(int(box.height() * 0.8))
        painter.setFont(font)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, "T")


class AnnotationArrowMeasurement(MeasurementButton):
    # Same 2-point arrow as "Annotation Arrow" (its length tag hidden by
    # default — see the MeasurementKind's meta_preset).
    name = "Annotation Arrow"
    display_name = "Arrow"

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        start = QPoint(rect.left() + LINE_MARGIN, rect.bottom() - LINE_MARGIN)
        end = QPoint(rect.right() - LINE_MARGIN, rect.top() + LINE_MARGIN)

        self._set_pen(painter, LINE_COLOR)
        painter.drawLine(start, end)
        self._draw_point(painter, start, ENDPOINT_RADIUS, active)

        dx, dy = end.x() - start.x(), end.y() - start.y()
        length = (dx * dx + dy * dy) ** 0.5
        if length > 0:
            ux, uy = dx / length, dy / length
            px, py = -uy, ux
            head_len, head_half = 9.0, 4.0
            base_x, base_y = end.x() - ux * head_len, end.y() - uy * head_len
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(LINE_COLOR)
            painter.drawPolygon(QPolygonF([
                QPointF(end),
                QPointF(base_x + px * head_half, base_y + py * head_half),
                QPointF(base_x - px * head_half, base_y - py * head_half),
            ]))
