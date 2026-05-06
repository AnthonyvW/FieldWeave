from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox

ORANGE = "#FFA500"
NM_PER_MM = 1_000_000


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


class NoScrollSpinBox(QSpinBox):
    def wheelEvent(self, event) -> None:
        event.ignore()