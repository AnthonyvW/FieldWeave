from __future__ import annotations

from PySide6.QtCore import QRect, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QPushButton, QWidget

from common.logger import info
from .overlay_base import Overlay


class CrosshairOverlay(Overlay):
    """Draws a small crosshair at the centre of the image."""

    def draw(self, painter: QPainter, rect: QRect) -> None:
        center_x = rect.x() + rect.width() // 2
        center_y = rect.y() + rect.height() // 2
        line_length = min(rect.width(), rect.height()) // 24

        painter.drawLine(center_x - line_length, center_y, center_x + line_length, center_y)
        painter.drawLine(center_x, center_y - line_length, center_x, center_y + line_length)


class CrosshairButton(QPushButton):
    """Checkable overlay button that signals when the crosshair should be toggled."""

    toggled_crosshair = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("⌖", parent)
        self.setObjectName("CrosshairButton")
        self.setCheckable(True)
        self.setFixedSize(30, 30)
        self.setToolTip("Toggle Crosshair")
        self.clicked.connect(self._on_clicked)

    def _on_clicked(self, checked: bool) -> None:
        info(f"Preview: Crosshair {'enabled' if checked else 'disabled'}")
        self.toggled_crosshair.emit(checked)