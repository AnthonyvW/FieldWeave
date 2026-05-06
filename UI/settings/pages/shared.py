from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox, QGroupBox, QLabel, QSpinBox, QWidget

ORANGE = "#FFA500"
NM_PER_MM = 1_000_000


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


class NoScrollSpinBox(QSpinBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


class LabelTrackerMixin:
    """Mixin that tracks form labels and applies orange highlighting to them.

    Classes mix this in and call _register_label during construction.
    mark_label and clear_orange then handle all highlighting uniformly.
    """

    _labels: dict[str, QLabel]

    def _register_label(self, key: str, label: QLabel) -> QLabel:
        """Store a label by key and return it for immediate use in addRow calls."""
        self._labels[key] = label
        return label

    def mark_label(self, key: str, orange: bool) -> None:
        lbl = self._labels.get(key)
        if lbl:
            lbl.setStyleSheet(f"color: {ORANGE};" if orange else "")

    def clear_orange(self) -> None:
        for lbl in self._labels.values():
            lbl.setStyleSheet("")


class SettingsGroupBase(LabelTrackerMixin, QGroupBox):
    """Base class for settings group box widgets."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._labels: dict[str, QLabel] = {}