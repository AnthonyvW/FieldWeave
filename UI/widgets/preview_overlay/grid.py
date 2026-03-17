from __future__ import annotations

from PySide6.QtCore import QRect, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QPushButton, QWidget

from common.logger import info
from .overlay_base import Overlay


class GridOverlay(Overlay):
    """Draws a 3x3 rule-of-thirds grid over the image."""

    def draw(self, painter: QPainter, rect: QRect) -> None:
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

        for i in range(1, 3):
            x_pos = x + (w * i // 3)
            painter.drawLine(x_pos, y, x_pos, y + h)

        for i in range(1, 3):
            y_pos = y + (h * i // 3)
            painter.drawLine(x, y_pos, x + w, y_pos)


class GridButton(QPushButton):
    """Checkable overlay button that signals when the grid should be toggled."""

    toggled_grid = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("⌗", parent)
        self.setObjectName("OverlayButton")
        self.setCheckable(True)
        self.setFixedSize(30, 30)
        self.setToolTip("Toggle Grid")
        self.clicked.connect(self._on_clicked)

    def _on_clicked(self, checked: bool) -> None:
        info(f"Preview: Grid {'enabled' if checked else 'disabled'}")
        self.toggled_grid.emit(checked)