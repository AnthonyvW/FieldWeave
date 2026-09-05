from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
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
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from UI.widgets.measurements.angles import (
    FourPointAngleMeasurement,
    ThreePointAngleMeasurement,
)
from UI.widgets.measurements.arcs import (
    RadiusArcMeasurement,
    ThreePointArcMeasurement,
)
from UI.widgets.measurements.calibration_line_button import CalibrationLineButton
from UI.widgets.measurements.circles import (
    DiameterMeasurement,
    RadiusCircleMeasurement,
    ThreePointCircleMeasurement,
)
from UI.widgets.measurements.curves import CurveMeasurement
from UI.widgets.measurements.ellipses import (
    FivePointEllipseMeasurement,
    ThreePointEllipseMeasurement,
)
from UI.widgets.measurements.line_pairs import (
    ArbitraryParallelMeasurement,
    ArbitraryPerpMeasurement,
    EightPointParallelMeasurement,
    FourPointParallelMeasurement,
    FourPointPerpMeasurement,
    ThreePointParallelMeasurement,
    ThreePointPerpMeasurement,
)
from UI.widgets.measurements.annotations import (
    AnnotationArrowMeasurement,
    CountMeasurement,
    ScaleBarMeasurement,
    TextMeasurement,
)
from UI.widgets.measurements.annulus import (
    DiameterAnnulusMeasurement,
    RadiusAnnulusMeasurement,
    ThreePointAnnulusMeasurement,
)
from UI.widgets.measurements.lines import (
    ArbitraryLineMeasurement,
    ArrowMeasurement,
    BracketMeasurement,
    DoubleArrowMeasurement,
    HorizontalLineMeasurement,
    MultipointLineMeasurement,
    VerticalLineMeasurement,
)
from UI.widgets.measurements.polygons import PolygonMeasurement
from UI.widgets.measurements.rectangles import (
    ThreePointRectangleMeasurement,
    TwoPointRectangleMeasurement,
    TwoPointSquareMeasurement,
)
from UI.widgets.measurements.two_circle import (
    DiameterTwoCircleMeasurement,
    RadiusTwoCircleMeasurement,
    ThreePointTwoCircleMeasurement,
)
from UI.widgets.measurements.measurement_meta import DEFAULT_META, MeasurementMeta
from UI.widgets.measurements.points import PointMeasurement
from UI.widgets.measurements.units import MeasurementUnit
from UI.widgets.preview_overlay.measurement_customize_menu import MeasurementCustomizeMenu, block_wheel

GRID_COLUMNS = 4

# Measurement tiles grouped into categories. Each entry is
# (category chip label, content-section title, tile classes). The tab
# shows a chip per category (bearing the first tile's icon); picking one
# reveals that category's tiles below, under its section title. Selection
# itself is still one exclusive group spanning every tile (see
# MeasurementsWidget) — categories only decide which tiles are visible.
# Each tile entry is (tile class, short label shown on the tile). The
# label strips the category-implied word — the category title supplies
# the context — while the tile's full `name` stays its tooltip.
_CATEGORY_GROUPS = (
    ("Point", "Point Measurements", ((PointMeasurement, "Point"),)),
    ("Line", "Line Measurements", (
        (ArbitraryLineMeasurement, "Arbitrary"), (MultipointLineMeasurement, "Multipoint"),
        (HorizontalLineMeasurement, "Horizontal"), (VerticalLineMeasurement, "Vertical"),
    )),
    ("Angle", "Angle Measurements", (
        (ThreePointAngleMeasurement, "3 Point"), (FourPointAngleMeasurement, "4 Point"),
    )),
    ("Arrow", "Arrow Measurements", (
        (ArrowMeasurement, "Single"), (DoubleArrowMeasurement, "Double"),
    )),
    ("Bracket", "Bracket Measurements", ((BracketMeasurement, "Bracket"),)),
    ("Circle", "Circle Measurements", (
        (RadiusCircleMeasurement, "Radius"), (DiameterMeasurement, "Diameter"),
        (ThreePointCircleMeasurement, "3 Point"),
    )),
    ("Ellipse", "Ellipse Measurements", (
        (ThreePointEllipseMeasurement, "3 Point"), (FivePointEllipseMeasurement, "5 Point"),
    )),
    ("Curves", "Curve Measurements", (
        (ThreePointArcMeasurement, "3 Point"), (RadiusArcMeasurement, "Radius"), (CurveMeasurement, "Curve"),
    )),
    ("Parallel", "Parallel Measurements", (
        (ThreePointParallelMeasurement, "3 Point"), (FourPointParallelMeasurement, "4 Point"),
        (EightPointParallelMeasurement, "8 Point"), (ArbitraryParallelMeasurement, "Arbitrary"),
    )),
    ("Perp", "Perpendicular Measurements", (
        (ThreePointPerpMeasurement, "3 Point"), (FourPointPerpMeasurement, "4 Point"),
        (ArbitraryPerpMeasurement, "Arbitrary"),
    )),
    ("Rectangle", "Rectangle Measurements", (
        (TwoPointRectangleMeasurement, "2 Point"), (ThreePointRectangleMeasurement, "3 Point"),
        (TwoPointSquareMeasurement, "Square"),
    )),
    ("Annulus", "Annulus Measurements", (
        (RadiusAnnulusMeasurement, "Radius"), (ThreePointAnnulusMeasurement, "3 Point"),
        (DiameterAnnulusMeasurement, "Diameter"),
    )),
    ("2 Circle", "Two-Circle Measurements", (
        (RadiusTwoCircleMeasurement, "Radius"), (ThreePointTwoCircleMeasurement, "3 Point"),
        (DiameterTwoCircleMeasurement, "Diameter"),
    )),
    ("Polygon", "Polygon Measurements", ((PolygonMeasurement, "Polygon"),)),
    ("Annotate", "Annotations", (
        (TextMeasurement, "Text"), (AnnotationArrowMeasurement, "Arrow"), (ScaleBarMeasurement, "Scale Bar"),
        (CountMeasurement, "Count"),
    )),
)


def _delete_all_icon(size: int = 14) -> QIcon:
    """A plain black X, sized like any other category chip's icon, for the Delete All chip."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor("#000000"))
    pen.setWidth(max(2, round(size / 12)))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    inset = size * 0.2
    painter.drawLine(QPointF(inset, inset), QPointF(size - inset, size - inset))
    painter.drawLine(QPointF(size - inset, inset), QPointF(inset, size - inset))
    painter.end()
    return QIcon(pixmap)


class _CategoryChip(QToolButton):
    """
    A category selector button carrying its category's first tile icon.
    Swaps between the tile's idle (orange-point) and active (blue-point)
    icon on hover/checked, mirroring MeasurementButton, so a checked
    chip's points don't vanish into the orange checked background.
    """

    def __init__(self, label, idle_icon, active_icon, icon_size, size, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("MeasurementTile")
        self.setCheckable(True)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.setIconSize(icon_size)
        self.setFixedSize(size)
        self.setText(label)
        self._idle_icon = idle_icon
        self._active_icon = active_icon
        self.setIcon(idle_icon)
        self.toggled.connect(self._sync_icon)

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self._sync_icon()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self._sync_icon()

    def _sync_icon(self, *_args: object) -> None:
        active = self.isChecked() or self.underMouse()
        self.setIcon(self._active_icon if active else self._idle_icon)


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
    default_meta_changed = Signal(str, object)  # kind, MeasurementMeta — applied to newly placed measurements of kind
    export_measurements_requested = Signal()
    import_measurements_requested = Signal()
    delete_all_requested = Signal()  # user confirmed clearing every placed measurement

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._selected_button: QAbstractButton | None = None
        # Whichever tile was selected right before manual calibration took
        # over placement, so finishing/cancelling it can resume that tile
        # instead of leaving placement disarmed — see _start_manual_calibration.
        self._pre_calibration_button: QAbstractButton | None = None
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

        outer_layout.addWidget(self._build_measurement_types())

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
        # _build_measurement_types (above) already selected the first tile
        # and emitted selection_changed before this panel existed to hear
        # it, so seed it manually; every later change is caught live.
        self.selection_changed.connect(self._sync_defaults_panel_to_selection)
        self._sync_defaults_panel_to_selection(self._selected_button.name if self._selected_button else None)

    def _build_measurement_types(self) -> QGroupBox:
        """
        Two stacked sections: a row of category chips (each showing the
        icon of its category's first tile) and, below, the selected
        category's own tiles under its section title. Every tile across
        every category still shares one exclusive selection group — the
        categories only decide which tiles are on screen.
        """
        group = QGroupBox("Measurement Types")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(6)

        self._category_group = QButtonGroup(self)
        self._category_group.setExclusive(True)
        chip_grid = QGridLayout()
        chip_grid.setSpacing(0)
        self._content_stack = QStackedWidget()
        # Per-category tile buttons and the last tile selected within each,
        # so switching to a category re-selects where the user left off
        # (or its first tile) — see _on_category_selected.
        self._category_tiles: list[list[QAbstractButton]] = []
        self._category_last: dict[int, QAbstractButton] = {}
        self._button_category: dict[QAbstractButton, int] = {}

        for index, (chip_label, _title, tile_entries) in enumerate(_CATEGORY_GROUPS):
            sample = tile_entries[0][0]()
            chip = _CategoryChip(
                chip_label, sample._idle_icon, sample._active_icon, sample.iconSize(), sample.size(),
            )
            self._category_group.addButton(chip, index)
            row, column = divmod(index, GRID_COLUMNS)
            chip_grid.addWidget(chip, row, column)

            page = QWidget()
            page_grid = QGridLayout(page)
            page_grid.setContentsMargins(0, 0, 0, 0)
            page_grid.setSpacing(0)
            tiles: list[QAbstractButton] = []
            for tile_index, (tile_cls, short_label) in enumerate(tile_entries):
                button = tile_cls(page)
                button.setText(short_label)
                button.setToolTip(button.name)
                self._button_group.addButton(button)
                self._button_category[button] = index
                tiles.append(button)
                tile_row, tile_col = divmod(tile_index, GRID_COLUMNS)
                page_grid.addWidget(button, tile_row, tile_col, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            # A trailing stretchy row/column packs the tiles to the top-left
            # rather than spreading them across the page.
            page_grid.setColumnStretch(GRID_COLUMNS, 1)
            page_grid.setRowStretch(page_grid.rowCount(), 1)
            self._content_stack.addWidget(page)
            self._category_tiles.append(tiles)

        # Delete All sits in the chip row itself, sized and styled like any
        # other category chip, rather than inside the content box — it acts
        # on every placed measurement, not just the selected category.
        self._delete_all_button = QToolButton()
        self._delete_all_button.setObjectName("MeasurementTile")
        self._delete_all_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self._delete_all_button.setIconSize(sample.iconSize())
        self._delete_all_button.setFixedSize(sample.size())
        self._delete_all_button.setIcon(_delete_all_icon(sample.iconSize().width()))
        self._delete_all_button.setText("Delete All")
        self._delete_all_button.setToolTip("Delete all placed measurements")
        self._delete_all_button.clicked.connect(self._on_delete_all)
        delete_all_index = len(_CATEGORY_GROUPS)
        row, column = divmod(delete_all_index, GRID_COLUMNS)
        chip_grid.addWidget(self._delete_all_button, row, column)

        layout.addLayout(chip_grid)

        self._content_box = QGroupBox(_CATEGORY_GROUPS[0][1])
        content_layout = QVBoxLayout(self._content_box)
        content_layout.setContentsMargins(4, 12, 4, 4)
        content_layout.setSpacing(0)
        content_layout.addWidget(self._content_stack)
        layout.addWidget(self._content_box)

        self._category_group.idClicked.connect(self._on_category_selected)
        first_chip = self._category_group.button(0)
        if first_chip is not None:
            # setChecked doesn't emit idClicked (that only fires for an
            # actual click), so the first category's own first tile would
            # otherwise never get selected the way clicking the chip does.
            first_chip.setChecked(True)
            self._on_category_selected(0)
        return group

    def _on_delete_all(self) -> None:
        confirm = QMessageBox.question(
            self, "Delete all measurements",
            "Delete all current measurements?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.delete_all_requested.emit()

    def _on_category_selected(self, index: int) -> None:
        self._content_stack.setCurrentIndex(index)
        self._content_box.setTitle(_CATEGORY_GROUPS[index][1])
        # Selecting a category activates that category's first tile, or
        # whichever tile in it the user last had selected.
        tiles = self._category_tiles[index] if index < len(self._category_tiles) else []
        if not tiles:
            return
        target = self._category_last.get(index, tiles[0])
        if target is not self._selected_button:
            target.setChecked(True)
            self._selected_button = target
            self.selection_changed.emit(target.name)

    def _build_customize_panel(self) -> QGroupBox:
        """
        Defaults applied to newly placed measurements, per kind — reuses
        MeasurementCustomizeMenu directly (embedded mode) so this panel
        shows exactly the same fields, in the same layout, an already-
        placed measurement's own popup would for that kind, rather than a
        hand-maintained subset that drifts out of sync with it. Retitles
        to "Customize Default <kind>" and reloads whenever the selected
        tile changes — see _sync_defaults_panel_to_selection, wired to
        selection_changed at the end of __init__ (after both this panel
        and the tile buttons exist).
        """
        panel = QGroupBox("Customize Default")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(6)

        self._defaults_panel = panel
        self._defaults_menu = MeasurementCustomizeMenu(panel, embedded=True)
        self._defaults_menu.defaults_changed.connect(self._on_default_meta_edited)
        layout.addWidget(self._defaults_menu)
        return panel

    def _sync_defaults_panel_to_selection(self, kind: str | None) -> None:
        """Point the embedded defaults panel at *kind*'s own template — a no-op while nothing is selected (e.g. mid-calibration), so the panel just keeps showing whichever kind was last selected rather than going blank."""
        if kind is None:
            return
        self._defaults_menu.open_defaults_for(kind)
        self._defaults_panel.setTitle(f"Customize Default {kind}")

    def current_default_meta(self) -> tuple[str | None, MeasurementMeta]:
        """The currently selected tile's own (kind, template) — used to seed the preview's fallback unit/default meta at startup, since this panel is the only source of a newly placed measurement's per-kind defaults. (None, DEFAULT_META) if nothing is selected yet."""
        if self._selected_button is None:
            return None, DEFAULT_META
        kind = self._selected_button.name
        meta = self._defaults_menu.default_meta_for(kind)
        # A kind never opened in the embedded panel yet still carries
        # DEFAULT_META's own unit=None (meaning "use the overlay's own
        # fallback") — MeasurementTab.set_unit needs a concrete unit here,
        # not None, since this seeds that very fallback at startup.
        if meta.unit is None:
            meta = meta._replace(unit=MeasurementUnit.MM)
        return kind, meta

    def _on_default_meta_edited(self, kind: str, meta: MeasurementMeta) -> None:
        self.default_meta_changed.emit(kind, meta)

    def _build_calibration_panel(self) -> QGroupBox:
        panel = QGroupBox("DPI Calibration")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(6)

        dpi_entry_row = QHBoxLayout()
        dpi_entry_row.addWidget(QLabel("Known DPI:"))
        self._dpi_entry_spin = QDoubleSpinBox()
        block_wheel(self._dpi_entry_spin)
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
        block_wheel(self._calibration_value_spin)
        self._calibration_value_spin.setRange(0.001, 100000.0)
        self._calibration_value_spin.setDecimals(3)
        self._calibration_value_spin.setValue(1.0)

        self._calibration_unit_combo = QComboBox()
        block_wheel(self._calibration_unit_combo)
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
            self._end_calibration()
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

        Remembers whichever measurement tile was selected beforehand (if
        any) so finishing or cancelling calibration can resume it — see
        _end_calibration — rather than leaving placement disarmed until
        the user picks a tool again.
        """
        # A re-click while already placing (restarting the line) shouldn't
        # overwrite the tile remembered from before calibration first
        # started with None (clear_selection already dropped it by then).
        already_active = self._calibration_button.isChecked()
        self._calibration_button.setChecked(True)
        if not already_active:
            self._pre_calibration_button = self._selected_button
        self.clear_selection()
        self.manual_calibration_started.emit()

    def _on_manual_calibration_clicked(self) -> None:
        self._start_manual_calibration()

    def _on_dpi_entry_set_clicked(self) -> None:
        self.dpi_value_submitted.emit(self._dpi_entry_spin.value())
        self._end_calibration()

    def _on_calibration_set_clicked(self) -> None:
        unit = self._calibration_unit_combo.currentData()
        self.calibration_dpi_submitted.emit(self._calibration_value_spin.value(), unit)
        self._end_calibration()

    def _on_calibration_cancel_clicked(self) -> None:
        self._end_calibration()

    def _end_calibration(self) -> None:
        """Common tail for every way calibration mode closes (Finish, Cancel, the DPI entry Set button, or toggling the panel shut) — hides the panel, tells CaptureControlWidget to drop calibration placement, then re-selects whichever measurement tile was active before manual calibration started, if any."""
        self._show_calibration_panel(False)
        self.calibration_cancelled.emit()
        button = self._pre_calibration_button
        self._pre_calibration_button = None
        if button is None:
            return
        self._selected_button = button
        button.setChecked(True)
        self.selection_changed.emit(button.name)

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
        category = self._button_category.get(button)
        if category is not None:
            self._category_last[category] = button
        self.selection_changed.emit(button.name)