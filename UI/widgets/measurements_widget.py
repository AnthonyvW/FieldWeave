from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QAbstractButton, QButtonGroup, QGridLayout, QGroupBox, QVBoxLayout, QWidget

from UI.widgets.measurements.circles import (
    DiameterMeasurement,
    RadiusCircleMeasurement,
    ThreePointCircleMeasurement,
)
from UI.widgets.measurements.lines import (
    ArbitraryLineMeasurement,
    HorizontalLineMeasurement,
    VerticalLineMeasurement,
)
from UI.widgets.measurements.points import PointMeasurement

GRID_COLUMNS = 4

_MEASUREMENT_TYPES = (
    PointMeasurement,
    ArbitraryLineMeasurement,
    HorizontalLineMeasurement,
    VerticalLineMeasurement,
    RadiusCircleMeasurement,
    DiameterMeasurement,
    ThreePointCircleMeasurement,
)


class MeasurementsWidget(QWidget):
    """
    Grid of measurement-type tiles, GRID_COLUMNS wide, boxed the same way
    CaptureControlWidget boxes its "Photo Capture" controls. Tiles sit
    flush against each other — QToolButton#MeasurementTile is transparent
    and borderless at idle, so the group box border is the only line the
    user sees until a tile is hovered or checked. Selection is exclusive,
    except that clicking the already-selected tile deselects it — a plain
    exclusive QButtonGroup won't uncheck its only checked button on its
    own, so that case is handled by hand in _on_button_clicked.
    """

    selection_changed = Signal(object)  # str | None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._selected_button: QAbstractButton | None = None

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._button_group.buttonClicked.connect(self._on_button_clicked)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Measurement Types")
        grid = QGridLayout(group)
        grid.setContentsMargins(4, 12, 4, 4)
        grid.setSpacing(0)

        for index, measurement_cls in enumerate(_MEASUREMENT_TYPES):
            button = measurement_cls(group)
            self._button_group.addButton(button)
            row, column = divmod(index, GRID_COLUMNS)
            grid.addWidget(button, row, column)

        outer_layout.addWidget(group)

    def selected_measurement(self) -> str | None:
        button = self._button_group.checkedButton()
        return button.name if button is not None else None

    def _on_button_clicked(self, button: QAbstractButton) -> None:
        if button is self._selected_button:
            # Exclusive groups won't uncheck their only checked button by
            # themselves, so drop exclusivity for the moment it takes to
            # force it off, then restore it for the next selection.
            self._button_group.setExclusive(False)
            button.setChecked(False)
            self._button_group.setExclusive(True)
            self._selected_button = None
            self.selection_changed.emit(None)
            return

        self._selected_button = button
        self.selection_changed.emit(button.name)