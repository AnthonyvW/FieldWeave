from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from UI.widgets.measurements.calibration_line_button import CalibrationLineButton
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
from UI.widgets.measurements.units import MeasurementUnit

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
    DPI status/entry, unit selection, and a grid of measurement-type
    tiles, GRID_COLUMNS wide, boxed the same way CaptureControlWidget
    boxes its "Photo Capture" controls. Tiles sit flush against each
    other — QToolButton#MeasurementTile is transparent and borderless at
    idle, so the group box border is the only line the user sees until a
    tile is hovered or checked. Selection is exclusive, except that
    clicking the already-selected tile deselects it — a plain exclusive
    QButtonGroup won't uncheck its only checked button on its own, so
    that case is handled by hand in _on_button_clicked.

    DPI itself is resolved by CaptureControlWidget, not here — this only
    displays it and relays requests. "Calibrate DPI" just toggles the
    "DPI Calibration" panel open/closed directly — no mode-choice popup;
    live camera and loaded image both get the exact same panel, since
    manual calibration and direct entry apply equally to either. There's
    no automatic-calibration option here at all — that's one of the
    Calibration tab's own choices, not something this tab needs to
    surface a second time.

    The panel has two independent ways to set DPI:

    - Direct entry: a spinbox prefilled with whatever DPI is already
      known (set_dpi_display's *dpi*, so it reads as "here's the current
      value" rather than a blank field) plus its own "Set" button.
    - Manual calibration: click CalibrationLineButton to place a
      reference line here, enter the real-world length it represents,
      then "Finish Calibration". The tile is forced checked (orange) for
      as long as calibration mode is open — a persistent, unmissable
      answer to "am I actually in calibration mode right now?" — and
      clicking it again re-starts placement (e.g. to redo a misplaced
      line). "Finish Calibration" stays disabled until CaptureControlWidget
      reports a line is actually placed — see set_calibration_line_ready.

    Either path hides the panel again once used. expand_calibration_panel
    is the passive counterpart, used when a loaded image turns out to
    have no DPI at all — it just opens the panel without prefilling or
    forcing placement, pointing at where to fix it.

    Placement doesn't require DPI (a shape can always be placed), but
    its length label on the preview only appears once one is set — see
    MeasurementOverlay._draw_measurement_label.
    """

    selection_changed = Signal(object)  # str | None
    unit_changed = Signal(object)  # MeasurementUnit
    dpi_value_submitted = Signal(float)
    manual_calibration_started = Signal()
    calibration_dpi_submitted = Signal(float, object)  # value, MeasurementUnit
    calibration_cancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._selected_button: QAbstractButton | None = None
        self._current_dpi: float | None = None
        self._is_live = True

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._button_group.buttonClicked.connect(self._on_button_clicked)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(6)

        dpi_row = QHBoxLayout()
        self._dpi_label = QLabel("DPI: not set")
        self._calibrate_button = QPushButton("Calibrate DPI")
        self._calibrate_button.clicked.connect(self._on_calibrate_dpi_clicked)
        dpi_row.addWidget(self._dpi_label, 1)
        dpi_row.addWidget(self._calibrate_button)
        outer_layout.addLayout(dpi_row)

        self._calibration_panel = self._build_calibration_panel()
        self._calibration_panel.setVisible(False)
        outer_layout.addWidget(self._calibration_panel)

        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("Unit:"))
        self._unit_combo = QComboBox()
        for unit in MeasurementUnit:
            self._unit_combo.addItem(unit.value, unit)
        self._unit_combo.setCurrentIndex(self._unit_combo.findData(MeasurementUnit.MM))
        self._unit_combo.currentIndexChanged.connect(self._on_unit_index_changed)
        unit_row.addWidget(self._unit_combo, 1)
        outer_layout.addLayout(unit_row)

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

    def _build_calibration_panel(self) -> QGroupBox:
        panel = QGroupBox("DPI Calibration")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(6)

        dpi_entry_row = QHBoxLayout()
        dpi_entry_row.addWidget(QLabel("Known DPI:"))
        self._dpi_entry_spin = QDoubleSpinBox()
        self._dpi_entry_spin.setRange(1.0, 100000.0)
        self._dpi_entry_spin.setDecimals(2)
        self._dpi_entry_spin.setValue(1.0)
        dpi_entry_set_button = QPushButton("Set")
        dpi_entry_set_button.clicked.connect(self._on_dpi_entry_set_clicked)
        dpi_entry_row.addWidget(self._dpi_entry_spin, 1)
        dpi_entry_row.addWidget(dpi_entry_set_button)
        layout.addLayout(dpi_entry_row)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("SampleDivider")
        layout.addWidget(divider)

        tile_row = QHBoxLayout()
        self._calibration_button = CalibrationLineButton(panel)
        self._calibration_button.clicked.connect(self._on_manual_calibration_clicked)
        tile_row.addWidget(self._calibration_button)

        hint = QLabel("Click two points on the preview, enter the real distance between them, then Finish Calibration.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #000000; font-size: 13px;")
        tile_row.addWidget(hint, 1)
        layout.addLayout(tile_row)

        entry_row = QHBoxLayout()
        self._calibration_value_spin = QDoubleSpinBox()
        self._calibration_value_spin.setRange(0.001, 100000.0)
        self._calibration_value_spin.setDecimals(3)
        self._calibration_value_spin.setValue(1.0)

        self._calibration_unit_combo = QComboBox()
        for unit in MeasurementUnit:
            self._calibration_unit_combo.addItem(unit.value, unit)
        self._calibration_unit_combo.setCurrentIndex(self._calibration_unit_combo.findData(MeasurementUnit.MM))

        entry_row.addWidget(self._calibration_value_spin, 1)
        entry_row.addWidget(self._calibration_unit_combo)
        layout.addLayout(entry_row)

        action_row = QHBoxLayout()
        self._calibration_set_button = QPushButton("Finish Calibration")
        self._calibration_set_button.setEnabled(False)
        self._calibration_set_button.clicked.connect(self._on_calibration_set_clicked)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self._on_calibration_cancel_clicked)
        action_row.addWidget(self._calibration_set_button)
        action_row.addWidget(cancel_button)
        layout.addLayout(action_row)

        return panel

    def selected_measurement(self) -> str | None:
        button = self._button_group.checkedButton()
        return button.name if button is not None else None

    def current_unit(self) -> MeasurementUnit:
        return self._unit_combo.currentData()

    def set_dpi_display(self, dpi: float | None, is_live: bool) -> None:
        self._current_dpi = dpi
        self._dpi_label.setText(f"DPI: {dpi:g}" if dpi is not None else "DPI: not set")
        if is_live != self._is_live:
            self._show_calibration_panel(False)
        self._is_live = is_live

    def set_calibration_line_ready(self, ready: bool) -> None:
        self._calibration_set_button.setEnabled(ready)

    def expand_calibration_panel(self) -> None:
        """Open the DPI Calibration panel without prefilling or starting placement — used when a loaded image turns out to have no DPI, to point at where to fix it."""
        self._show_calibration_panel(True)

    def clear_selection(self) -> None:
        """Deselect whichever tile is checked, without waiting for a click — used when calibration placement takes over the preview."""
        if self._selected_button is None:
            return
        self._button_group.setExclusive(False)
        self._selected_button.setChecked(False)
        self._button_group.setExclusive(True)
        self._selected_button = None
        self.selection_changed.emit(None)

    def _on_unit_index_changed(self, index: int) -> None:
        self.unit_changed.emit(self._unit_combo.itemData(index))

    def _on_calibrate_dpi_clicked(self) -> None:
        if self._calibration_panel.isVisible():
            self._show_calibration_panel(False)
            self.calibration_cancelled.emit()
            return
        if self._current_dpi and self._current_dpi > 0:
            self._dpi_entry_spin.setValue(self._current_dpi)
        self._show_calibration_panel(True)

    def _show_calibration_panel(self, visible: bool) -> None:
        self._calibration_panel.setVisible(visible)
        if not visible:
            self._calibration_button.setChecked(False)

    def _start_manual_calibration(self) -> None:
        """
        (Re)start placement of the calibration reference line, and force
        the tile checked (orange) regardless of its native toggle state —
        every click here means "I want to be placing a line right now",
        never "turn calibration mode off" (Cancel/Finish handle that).
        """
        self._calibration_button.setChecked(True)
        self.clear_selection()
        self.manual_calibration_started.emit()

    def _on_manual_calibration_clicked(self) -> None:
        self._start_manual_calibration()

    def _on_dpi_entry_set_clicked(self) -> None:
        self.dpi_value_submitted.emit(self._dpi_entry_spin.value())
        self._show_calibration_panel(False)
        self.calibration_cancelled.emit()

    def _on_calibration_set_clicked(self) -> None:
        unit = self._calibration_unit_combo.currentData()
        self.calibration_dpi_submitted.emit(self._calibration_value_spin.value(), unit)
        self._show_calibration_panel(False)
        self.calibration_cancelled.emit()

    def _on_calibration_cancel_clicked(self) -> None:
        self._show_calibration_panel(False)
        self.calibration_cancelled.emit()

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