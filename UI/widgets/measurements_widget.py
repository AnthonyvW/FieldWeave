from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
    MEASUREMENT_LINE_CAPS,
    VerticalLineMeasurement,
)
from UI.widgets.measurements.measurement_meta import MeasurementMeta
from UI.widgets.measurements.measurement_style import (
    OVERLAY_LINE_COLOR, OVERLAY_LINE_WIDTH, OVERLAY_OUTLINE_COLOR, OVERLAY_OUTLINE_WIDTH,
)
from UI.widgets.measurements.points import PointMeasurement
from UI.widgets.measurements.units import MeasurementUnit
from UI.widgets.preview_overlay.measurement_customize_menu import (
    _ColorPicker, _StylePicker, _ThicknessControl, _dash_style_icon, _field_label, _line_cap_icon,
)
from UI.widgets.preview_overlay.measurement_overlay import MEASUREMENT_DASH_PATTERNS

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

    Below the tile grid, "Export"/"Import" round-trip the active source's
    placed measurements through a JSON file — see measurement_io.py for
    the format and MeasurementTab for the file dialogs and defaults this
    just requests via export_measurements_requested/
    import_measurements_requested. Below that, "Customize Measurements"
    holds defaults (title prefix, unit, and the same appearance fields
    MeasurementCustomizeMenu offers) applied to measurements as they're
    placed — see _build_customize_panel. Editing an already-placed
    measurement is separate: its tag on the preview opens
    MeasurementCustomizeMenu directly (camera_preview.py), rather than
    going through this widget at all.
    """

    selection_changed = Signal(object)  # str | None
    dpi_value_submitted = Signal(float)
    manual_calibration_started = Signal()
    calibration_dpi_submitted = Signal(float, object)  # value, MeasurementUnit
    calibration_cancelled = Signal()
    default_meta_changed = Signal(object)  # MeasurementMeta, applied to newly placed measurements
    export_measurements_requested = Signal()
    import_measurements_requested = Signal()

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

        data_group = QGroupBox("Import / Export")
        data_row = QHBoxLayout(data_group)
        self._export_measurements_button = QPushButton("Export")
        self._export_measurements_button.clicked.connect(self.export_measurements_requested)
        self._import_measurements_button = QPushButton("Import")
        self._import_measurements_button.clicked.connect(self.import_measurements_requested)
        data_row.addWidget(self._export_measurements_button)
        data_row.addWidget(self._import_measurements_button)
        outer_layout.addWidget(data_group)

        self._customize_panel = self._build_customize_panel()
        outer_layout.addWidget(self._customize_panel)

    def _build_customize_panel(self) -> QGroupBox:
        """
        Defaults applied to every new measurement as it's placed —
        every field MeasurementCustomizeMenu offers for an already-
        placed measurement, applied here as a shared starting point
        instead. An unset title stays unset (see
        MeasurementOverlay._resolve_meta), a set one becomes a numbered
        prefix ("Wingspan 1", "Wingspan 2", ...); everything else
        applies as-is. A description isn't offered here since it's
        inherently per-measurement — standardizing one across every
        placement of a kind doesn't make sense the way a shared title,
        unit, or appearance choice does. Takes effect as the fields are
        edited, with no separate apply step, since there's nothing to
        preview here — it only ever affects measurements placed after
        the fact. Editing a measurement already placed (including
        giving it its own description) is a separate, per-instance
        action — see MeasurementCustomizeMenu, opened by clicking its
        tag on the preview.
        """
        panel = QGroupBox("Customize Measurements")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(6)

        layout.addWidget(QLabel("Default Title:"))
        self._default_title_edit = QLineEdit()
        self._default_title_edit.setPlaceholderText("e.g. Wingspan")
        self._default_title_edit.textChanged.connect(self._on_default_meta_edited)
        layout.addWidget(self._default_title_edit)

        default_unit_row = QHBoxLayout()
        default_unit_row.addWidget(QLabel("Unit:"))
        self._default_unit_combo = QComboBox()
        for unit in MeasurementUnit:
            self._default_unit_combo.addItem(unit.value, unit)
        self._default_unit_combo.setCurrentIndex(self._default_unit_combo.findData(MeasurementUnit.MM))
        self._default_unit_combo.currentIndexChanged.connect(self._on_default_meta_edited)
        default_unit_row.addWidget(self._default_unit_combo, 1)
        layout.addLayout(default_unit_row)

        self._default_always_show_description_check = QCheckBox("Always show description")
        self._default_always_show_description_check.toggled.connect(self._on_default_meta_edited)
        layout.addWidget(self._default_always_show_description_check)

        self._default_tag_bg_picker = _ColorPicker("Tag Background Color", OVERLAY_LINE_COLOR.name())
        self._default_tag_bg_picker.color_changed.connect(self._on_default_meta_edited)
        layout.addWidget(self._default_tag_bg_picker)

        self._default_tag_text_picker = _ColorPicker("Tag Text Color", OVERLAY_OUTLINE_COLOR.name())
        self._default_tag_text_picker.color_changed.connect(self._on_default_meta_edited)
        layout.addWidget(self._default_tag_text_picker)

        self._default_line_color_picker = _ColorPicker("Line Color", OVERLAY_LINE_COLOR.name())
        self._default_line_color_picker.color_changed.connect(self._on_default_meta_edited)
        layout.addWidget(self._default_line_color_picker)

        layout.addWidget(_field_label("Line Thickness"))
        self._default_line_thickness_control = _ThicknessControl(0.5, 12.0)
        self._default_line_thickness_control.set_value(OVERLAY_LINE_WIDTH)
        self._default_line_thickness_control.value_changed.connect(self._on_default_meta_edited)
        layout.addWidget(self._default_line_thickness_control)

        self._default_line_style_picker = _StylePicker(
            "Line Style", [(style, _dash_style_icon(style)) for style in MEASUREMENT_DASH_PATTERNS]
        )
        self._default_line_style_picker.value_changed.connect(self._on_default_meta_edited)
        layout.addWidget(self._default_line_style_picker)

        # Caps only ever apply to line-category measurements (see
        # MeasurementKind.category in measurement_kind.py) but, unlike
        # MeasurementCustomizeMenu, there's no single placed measurement's
        # kind to hide these against here — this panel is a shared
        # template for every tile, line or not, so a circle/point
        # placement simply never reads its own line_start_cap/line_end_cap.
        self._default_start_cap_picker = _StylePicker(
            "Start", [(cap, _line_cap_icon(cap)) for cap in MEASUREMENT_LINE_CAPS]
        )
        self._default_start_cap_picker.value_changed.connect(self._on_default_meta_edited)
        layout.addWidget(self._default_start_cap_picker)

        self._default_end_cap_picker = _StylePicker(
            "End", [(cap, _line_cap_icon(cap)) for cap in MEASUREMENT_LINE_CAPS]
        )
        self._default_end_cap_picker.value_changed.connect(self._on_default_meta_edited)
        layout.addWidget(self._default_end_cap_picker)

        self._default_outline_enabled_check = QCheckBox("Enable Outline")
        self._default_outline_enabled_check.setChecked(True)
        self._default_outline_enabled_check.toggled.connect(self._on_default_outline_enabled_toggled)
        layout.addWidget(self._default_outline_enabled_check)

        self._default_outline_color_picker = _ColorPicker("Outline Color", OVERLAY_OUTLINE_COLOR.name())
        self._default_outline_color_picker.color_changed.connect(self._on_default_meta_edited)
        layout.addWidget(self._default_outline_color_picker)

        self._default_outline_thickness_label = _field_label("Outline Thickness")
        layout.addWidget(self._default_outline_thickness_label)
        self._default_outline_thickness_control = _ThicknessControl(0.0, 8.0)
        self._default_outline_thickness_control.set_value(OVERLAY_OUTLINE_WIDTH)
        self._default_outline_thickness_control.value_changed.connect(self._on_default_meta_edited)
        layout.addWidget(self._default_outline_thickness_control)

        return panel

    def _set_default_outline_controls_visible(self, visible: bool) -> None:
        self._default_outline_color_picker.setVisible(visible)
        self._default_outline_thickness_label.setVisible(visible)
        self._default_outline_thickness_control.setVisible(visible)

    def _on_default_outline_enabled_toggled(self, _enabled: bool) -> None:
        self._set_default_outline_controls_visible(self._default_outline_enabled_check.isChecked())
        self._on_default_meta_edited()

    def _build_default_meta(self) -> MeasurementMeta:
        return MeasurementMeta(
            title=self._default_title_edit.text().strip(),
            unit=self._default_unit_combo.currentData(),
            always_show_description=self._default_always_show_description_check.isChecked(),
            tag_background_color=self._default_tag_bg_picker.color(),
            tag_text_color=self._default_tag_text_picker.color(),
            line_color=self._default_line_color_picker.color(),
            line_thickness=self._default_line_thickness_control.value(),
            line_dash_style=self._default_line_style_picker.value(),
            line_start_cap=self._default_start_cap_picker.value(),
            line_end_cap=self._default_end_cap_picker.value(),
            outline_enabled=self._default_outline_enabled_check.isChecked(),
            outline_color=self._default_outline_color_picker.color(),
            outline_thickness=self._default_outline_thickness_control.value(),
        )

    def current_default_meta(self) -> MeasurementMeta:
        """The Customize Measurements panel's current template — used to seed the preview's fallback unit/default meta at startup, since this panel (not the removed top-level unit dropdown) is now the only source of a newly placed measurement's unit."""
        return self._build_default_meta()

    def _on_default_meta_edited(self, *_args: object) -> None:
        self.default_meta_changed.emit(self._build_default_meta())

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