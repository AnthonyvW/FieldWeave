from __future__ import annotations

import math

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QToolButton,
)
from PySide6.QtCore import Qt, QTimer

from common.app_context import get_app_context
from common.logger import warning, error
from motion.routines.z_stack_area_scan import ZStackAreaScan
from post_processing.routines.focus_stack_routine import FocusStackRoutineConfig
from UI.widgets.utilities.open_filesystem_object_button import OpenFolderButton
from UI.widgets.automation.output_folder_widget import OutputFolderWidget


# ---------------------------------------------------------------------------
# Confirmation dialog
# ---------------------------------------------------------------------------

TIME_PER_IMAGE = 1.5 # Actual time it takes is 1.3, but it takes 0.2 seconds to settle

class _ConfirmAreaScanDialog(QDialog):
    """Modal dialog summarising the area scan parameters before starting."""

    def __init__(
        self,
        x_start: float,
        x_end: float,
        x_step_mm: float,
        y_start: float,
        y_end: float,
        y_step_mm: float,
        z_start: float,
        z_end: float,
        z_step_mm: float,
        step_decimals: int,
        output_folder: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm Area Scan")
        self.setModal(True)
        self.setMinimumWidth(380)

        fmt = f".{step_decimals}f"

        x_range = abs(x_end - x_start)
        y_range = abs(y_end - y_start)
        z_range = abs(z_end - z_start)

        n_x = int(x_range / x_step_mm) + 1 if x_step_mm > 0 else 1
        n_y = int(y_range / y_step_mm) + 1 if y_step_mm > 0 else 1
        n_z = int(z_range / z_step_mm) + 1 if z_step_mm > 0 else 1

        total_stacks = n_x * n_y
        total_images = total_stacks * n_z

        # Rough estimate of how long it'll take
        total_seconds = math.ceil(total_images * TIME_PER_IMAGE + total_stacks * 1.0)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Ready to start area scan?")
        title.setObjectName("AreaScanDialogTitle")
        layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)

        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(6)

        rows: list[tuple[str, str]] = [
            ("X range", f"{x_start:{fmt}} → {x_end:{fmt}} mm  ({x_range:{fmt}} mm)"),
            ("X step / positions", f"{x_step_mm:{fmt}} mm  ({n_x} positions)"),
            ("Y range", f"{y_start:{fmt}} → {y_end:{fmt}} mm  ({y_range:{fmt}} mm)"),
            ("Y step / positions", f"{y_step_mm:{fmt}} mm  ({n_y} positions)"),
            ("Z range", f"{z_start:{fmt}} → {z_end:{fmt}} mm  ({z_range:{fmt}} mm)"),
            ("Z step / slices", f"{z_step_mm:{fmt}} mm  ({n_z} slices)"),
            ("Total XY positions", str(total_stacks)),
            ("Total images", str(total_images)),
            ("Estimated time", time_str),
            ("Output folder", output_folder),
        ]

        for label_text, value_text in rows:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            lbl = QLabel(label_text + ":")
            lbl.setObjectName("AreaScanRowLabel")
            lbl.setFixedWidth(150)
            row_layout.addWidget(lbl)

            val = QLabel(value_text)
            val.setObjectName("AreaScanRowValue")
            val.setWordWrap(True)
            row_layout.addWidget(val, 1)

            info_layout.addWidget(row)

        layout.addWidget(info_widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Start")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


# ---------------------------------------------------------------------------
# Axis range sub-widget (reused for X, Y, Z)
# ---------------------------------------------------------------------------

class _AxisRangeWidget(QWidget):
    """
    Compact group box for a single axis: Set Start / Set End buttons with
    position readouts and a step-size spin box.
    """

    def __init__(
        self,
        axis_label: str,
        step_mm: float,
        step_decimals: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._start_mm: float | None = None
        self._end_mm: float | None = None
        self._axis_label = axis_label

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        group = QGroupBox(f"{axis_label} Axis")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(10, 8, 10, 8)
        group_layout.setSpacing(6)

        # Start row
        start_row = QWidget()
        start_layout = QHBoxLayout(start_row)
        start_layout.setContentsMargins(0, 0, 0, 0)
        start_layout.setSpacing(8)

        self._set_start_btn = QPushButton(f"Set {axis_label} Start")
        self._set_start_btn.setFixedHeight(30)
        start_layout.addWidget(self._set_start_btn)

        self._start_label = QLabel("Not set")
        self._start_label.setObjectName("AreaScanAxisReadout")
        self._start_label.setMinimumWidth(110)
        self._start_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        start_layout.addWidget(self._start_label)

        group_layout.addWidget(start_row)

        # End row
        end_row = QWidget()
        end_layout = QHBoxLayout(end_row)
        end_layout.setContentsMargins(0, 0, 0, 0)
        end_layout.setSpacing(8)

        self._set_end_btn = QPushButton(f"Set {axis_label} End")
        self._set_end_btn.setFixedHeight(30)
        end_layout.addWidget(self._set_end_btn)

        self._end_label = QLabel("Not set")
        self._end_label.setObjectName("AreaScanAxisReadout")
        self._end_label.setMinimumWidth(110)
        self._end_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        end_layout.addWidget(self._end_label)

        group_layout.addWidget(end_row)

        # Step row
        step_row = QWidget()
        step_layout = QHBoxLayout(step_row)
        step_layout.setContentsMargins(0, 0, 0, 0)
        step_layout.setSpacing(8)

        step_lbl = QLabel("Step (mm):")
        step_layout.addWidget(step_lbl)

        self._step_spin = QDoubleSpinBox()
        self._step_spin.setFixedHeight(28)
        self._step_spin.setDecimals(step_decimals)
        self._step_spin.setSuffix(" mm")
        self._step_spin.setMinimum(step_mm)
        self._step_spin.setMaximum(300.0)
        self._step_spin.setSingleStep(step_mm)
        self._step_spin.setValue(step_mm)
        step_layout.addWidget(self._step_spin)

        fmt = f".{step_decimals}f"
        min_label = QLabel(f"(min: {step_mm:{fmt}} mm)")
        min_label.setObjectName("AreaScanMinLabel")
        step_layout.addWidget(min_label)
        step_layout.addStretch(1)

        group_layout.addWidget(step_row)
        outer.addWidget(group)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def connect_start(self, slot) -> None:  # type: ignore[type-arg]
        self._set_start_btn.clicked.connect(slot)

    def connect_end(self, slot) -> None:  # type: ignore[type-arg]
        self._set_end_btn.clicked.connect(slot)

    def connect_step_changed(self, slot) -> None:  # type: ignore[type-arg]
        self._step_spin.valueChanged.connect(slot)

    def set_start(self, value_mm: float) -> None:
        self._start_mm = value_mm
        decimals = self._step_spin.decimals()
        self._start_label.setText(f"{self._axis_label} = {value_mm:.{decimals}f} mm")

    def set_end(self, value_mm: float) -> None:
        self._end_mm = value_mm
        decimals = self._step_spin.decimals()
        self._end_label.setText(f"{self._axis_label} = {value_mm:.{decimals}f} mm")

    def mark_unavailable(self, which: str) -> None:
        if which == "start":
            self._start_label.setText("Unavailable")
        else:
            self._end_label.setText("Unavailable")

    @property
    def start_mm(self) -> float | None:
        return self._start_mm

    @property
    def end_mm(self) -> float | None:
        return self._end_mm

    @property
    def step_mm(self) -> float:
        return self._step_spin.value()

    @property
    def decimals(self) -> int:
        return self._step_spin.decimals()

    @property
    def is_configured(self) -> bool:
        return self._start_mm is not None and self._end_mm is not None


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class ZStackAreaScanWidget(QWidget):
    """Widget for configuring and running a area scan across an XY grid."""

    mode_name: str = "Area Scan"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_output_folder: str | None = None
        self._setup_ui()
        self._populate_from_settings()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        printer_step = self._get_printer_step_mm()
        step_decimals = self._decimals_for_step(printer_step)

        # ---- Axis range widgets ----
        self._x_axis = _AxisRangeWidget("X", printer_step, step_decimals)
        self._x_axis.connect_start(self._set_x_start)
        self._x_axis.connect_end(self._set_x_end)
        self._x_axis.connect_step_changed(self._update_summary)
        self._x_axis.connect_step_changed(
            lambda v: self._write_int_to_settings("x_step_nm", round(v * 1_000_000))
        )
        main_layout.addWidget(self._x_axis)

        self._y_axis = _AxisRangeWidget("Y", printer_step, step_decimals)
        self._y_axis.connect_start(self._set_y_start)
        self._y_axis.connect_end(self._set_y_end)
        self._y_axis.connect_step_changed(self._update_summary)
        self._y_axis.connect_step_changed(
            lambda v: self._write_int_to_settings("y_step_nm", round(v * 1_000_000))
        )
        main_layout.addWidget(self._y_axis)

        self._z_axis = _AxisRangeWidget("Z", printer_step, step_decimals)
        self._z_axis.connect_start(self._set_z_start)
        self._z_axis.connect_end(self._set_z_end)
        self._z_axis.connect_step_changed(self._update_summary)
        self._z_axis.connect_step_changed(
            lambda v: self._write_int_to_settings("z_step_nm", round(v * 1_000_000))
        )
        main_layout.addWidget(self._z_axis)

        # ---- Output folder ----
        self._output_folder = OutputFolderWidget()
        main_layout.addWidget(self._output_folder)

        # ---- Focus stack settings ----------------------------------------
        fs_group = QGroupBox("Focus Stack Settings")
        fs_layout = QVBoxLayout(fs_group)
        fs_layout.setContentsMargins(10, 8, 10, 8)
        fs_layout.setSpacing(6)

        self._fs_enable_check = QCheckBox("Run focus stack after each XY position")
        self._fs_enable_check.setChecked(False)
        self._fs_enable_check.stateChanged.connect(self._on_fs_enabled_changed)
        self._fs_enable_check.stateChanged.connect(
            lambda v: self._write_check_to_settings("run_focus_stack", v)
        )
        fs_layout.addWidget(self._fs_enable_check)

        self._fs_settings_widget = QWidget()
        fs_settings_layout = QVBoxLayout(self._fs_settings_widget)
        fs_settings_layout.setContentsMargins(0, 4, 0, 0)
        fs_settings_layout.setSpacing(6)

        self._keep_size_check = QCheckBox("Keep original size")
        self._keep_size_check.setChecked(True)
        self._keep_size_check.setToolTip(
            "Keep the output image the same size as the input images. "
            "Warps are applied in-place rather than expanding the canvas."
        )
        self._keep_size_check.stateChanged.connect(
            lambda v: self._write_check_to_settings("keep_size", v)
        )
        fs_settings_layout.addWidget(self._keep_size_check)

        self._advanced_toggle = QToolButton()
        self._advanced_toggle.setText("Advanced settings")
        self._advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._advanced_toggle.setCheckable(True)
        self._advanced_toggle.setChecked(False)
        self._advanced_toggle.setObjectName("AdvancedSettingsToggle")
        self._advanced_toggle.toggled.connect(self._on_advanced_toggled)
        fs_settings_layout.addWidget(self._advanced_toggle)

        self._advanced_widget = QWidget()
        self._advanced_widget.setVisible(False)
        advanced_layout = QVBoxLayout(self._advanced_widget)
        advanced_layout.setContentsMargins(12, 0, 0, 0)
        advanced_layout.setSpacing(6)

        self._no_align_check = QCheckBox("Skip alignment")
        self._no_align_check.setChecked(False)
        self._no_align_check.setToolTip("Skip ECC alignment. Use when images are already registered.")
        self._no_align_check.stateChanged.connect(
            lambda v: self._write_check_to_settings("no_align", v)
        )
        advanced_layout.addWidget(self._no_align_check)

        self._crop_check = QCheckBox("Crop to intersection")
        self._crop_check.setChecked(False)
        self._crop_check.setToolTip(
            "Crop the output to the largest rectangle covered by every frame after "
            "alignment. Removes border regions but shrinks the output image."
        )
        self._crop_check.stateChanged.connect(
            lambda v: self._write_check_to_settings("crop", v)
        )
        advanced_layout.addWidget(self._crop_check)

        sharpness_row = QWidget()
        sharpness_layout = QHBoxLayout(sharpness_row)
        sharpness_layout.setContentsMargins(0, 0, 0, 0)
        sharpness_layout.setSpacing(8)
        sharpness_layout.addWidget(QLabel("Sharpness:"))
        self._sharpness_spin = QDoubleSpinBox()
        self._sharpness_spin.setFixedHeight(28)
        self._sharpness_spin.setDecimals(1)
        self._sharpness_spin.setMinimum(1.0)
        self._sharpness_spin.setMaximum(8.0)
        self._sharpness_spin.setSingleStep(0.5)
        self._sharpness_spin.setValue(4.0)
        self._sharpness_spin.setToolTip(
            "Weight sharpness exponent. Higher values favour the sharpest pixel "
            "more aggressively (approaching hard selection). Lower values blend "
            "more smoothly. Useful range: 1.0 (soft) to 8.0 (near-hard)."
        )
        self._sharpness_spin.valueChanged.connect(
            lambda v: self._write_float_to_settings("sharpness", v)
        )
        sharpness_layout.addWidget(self._sharpness_spin)
        sharpness_layout.addStretch(1)
        advanced_layout.addWidget(sharpness_row)

        cull_row = QWidget()
        cull_layout = QHBoxLayout(cull_row)
        cull_layout.setContentsMargins(0, 0, 0, 0)
        cull_layout.setSpacing(8)
        self._cull_check = QCheckBox("Cull out-of-focus frames")
        self._cull_check.setChecked(False)
        self._cull_check.setToolTip(
            "Discard frames whose focus score falls below the threshold fraction "
            "of the sharpest frame. At least the two sharpest frames are always kept."
        )
        self._cull_check.stateChanged.connect(
            lambda v: self._write_check_to_settings("cull_enabled", v)
        )
        cull_layout.addWidget(self._cull_check)
        self._cull_threshold_spin = QDoubleSpinBox()
        self._cull_threshold_spin.setFixedHeight(28)
        self._cull_threshold_spin.setDecimals(2)
        self._cull_threshold_spin.setMinimum(0.0)
        self._cull_threshold_spin.setMaximum(1.0)
        self._cull_threshold_spin.setSingleStep(0.05)
        self._cull_threshold_spin.setValue(0.6)
        self._cull_threshold_spin.setToolTip(
            "Frames scoring below this fraction of the peak score are culled. "
            "Raise toward 1.0 to cull more aggressively."
        )
        self._cull_threshold_spin.valueChanged.connect(
            lambda v: self._write_float_to_settings("cull_threshold", v)
        )
        cull_layout.addWidget(self._cull_threshold_spin)
        cull_layout.addStretch(1)
        advanced_layout.addWidget(cull_row)

        self._slab_check = QCheckBox("Enable slabbing")
        self._slab_check.setChecked(False)
        self._slab_check.setToolTip(
            "Split the image set into overlapping sub-stacks, stack each "
            "independently, then fuse the results. Reduces peak RAM for large stacks."
        )
        self._slab_check.stateChanged.connect(self._on_slab_enabled_changed)
        self._slab_check.stateChanged.connect(
            lambda v: self._write_check_to_settings("slab_enabled", v)
        )
        advanced_layout.addWidget(self._slab_check)

        self._slab_params_widget = QWidget()
        self._slab_params_widget.setVisible(False)
        slab_params_layout = QHBoxLayout(self._slab_params_widget)
        slab_params_layout.setContentsMargins(20, 0, 0, 0)
        slab_params_layout.setSpacing(8)
        slab_params_layout.addWidget(QLabel("Size:"))
        self._slab_size_spin = QSpinBox()
        self._slab_size_spin.setFixedHeight(28)
        self._slab_size_spin.setMinimum(2)
        self._slab_size_spin.setMaximum(500)
        self._slab_size_spin.setValue(20)
        self._slab_size_spin.setToolTip("Number of images per sub-stack.")
        self._slab_size_spin.valueChanged.connect(
            lambda v: self._write_int_to_settings("slab_size", v)
        )
        slab_params_layout.addWidget(self._slab_size_spin)
        slab_params_layout.addWidget(QLabel("Overlap:"))
        self._slab_overlap_spin = QSpinBox()
        self._slab_overlap_spin.setFixedHeight(28)
        self._slab_overlap_spin.setMinimum(0)
        self._slab_overlap_spin.setMaximum(499)
        self._slab_overlap_spin.setValue(5)
        self._slab_overlap_spin.setToolTip(
            "Number of images shared between adjacent slabs. Must be less than size."
        )
        self._slab_overlap_spin.valueChanged.connect(
            lambda v: self._write_int_to_settings("slab_overlap", v)
        )
        slab_params_layout.addWidget(self._slab_overlap_spin)
        slab_params_layout.addStretch(1)
        advanced_layout.addWidget(self._slab_params_widget)

        workers_row = QWidget()
        workers_layout = QHBoxLayout(workers_row)
        workers_layout.setContentsMargins(0, 0, 0, 0)
        workers_layout.setSpacing(8)
        workers_layout.addWidget(QLabel("Workers:"))
        self._workers_spin = QSpinBox()
        self._workers_spin.setFixedHeight(28)
        self._workers_spin.setMinimum(0)
        self._workers_spin.setMaximum(16)
        self._workers_spin.setValue(3)
        self._workers_spin.setToolTip(
            "Number of parallel workers for stacking. 0 = no limit (use all available). "
            "Higher values are faster but increase peak RAM by ~100 MiB per additional worker."
        )
        self._workers_spin.valueChanged.connect(
            lambda v: self._write_int_to_settings("workers", v)
        )
        workers_layout.addWidget(self._workers_spin)
        workers_layout.addStretch(1)
        advanced_layout.addWidget(workers_row)

        fs_settings_layout.addWidget(self._advanced_widget)
        fs_layout.addWidget(self._fs_settings_widget)
        self._fs_settings_widget.setVisible(False)
        main_layout.addWidget(fs_group)

        # ---- Summary label ----
        self._summary_label = QLabel("")
        self._summary_label.setObjectName("AreaScanSummary")
        self._summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._summary_label.setWordWrap(True)
        main_layout.addWidget(self._summary_label)

        main_layout.addStretch(1)

        # ---- Start button ----
        self._start_btn = QPushButton("Start Automation")
        self._start_btn.setObjectName("AreaScanStart")
        self._start_btn.setFixedHeight(34)
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start_clicked)
        main_layout.addWidget(self._start_btn)

        # ---- Open Folder button (shown once a scan has started) ----
        self._open_folder_btn = OpenFolderButton()
        main_layout.addWidget(self._open_folder_btn)

        # Timer for polling routine state on the UI thread
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_routine_state)

    # ------------------------------------------------------------------
    # Settings population and write-back
    # ------------------------------------------------------------------

    def _populate_from_settings(self) -> None:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return
        a = motion.settings.z_stack_area_scan
        _NM_PER_MM = 1_000_000

        self._x_axis._step_spin.blockSignals(True)
        self._x_axis._step_spin.setValue(a.x_step_nm / _NM_PER_MM)
        self._x_axis._step_spin.blockSignals(False)

        self._y_axis._step_spin.blockSignals(True)
        self._y_axis._step_spin.setValue(a.y_step_nm / _NM_PER_MM)
        self._y_axis._step_spin.blockSignals(False)

        self._z_axis._step_spin.blockSignals(True)
        self._z_axis._step_spin.setValue(a.z_step_nm / _NM_PER_MM)
        self._z_axis._step_spin.blockSignals(False)

        for widget, checked in (
            (self._fs_enable_check,   a.run_focus_stack),
            (self._keep_size_check,   a.keep_size),
            (self._no_align_check,    a.no_align),
            (self._crop_check,        a.crop),
            (self._cull_check,        a.cull_enabled),
            (self._slab_check,        a.slab_enabled),
        ):
            widget.blockSignals(True)
            widget.setChecked(checked)
            widget.blockSignals(False)

        for spin, value in (
            (self._sharpness_spin,       a.sharpness),
            (self._cull_threshold_spin,  a.cull_threshold),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

        for spin, value in (
            (self._slab_size_spin,    a.slab_size),
            (self._slab_overlap_spin, a.slab_overlap),
            (self._workers_spin,      a.workers),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

        self._on_fs_enabled_changed()
        self._on_slab_enabled_changed()
        self._update_summary()

    def _write_float_to_settings(self, key: str, value: float) -> None:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return
        setattr(motion.settings.z_stack_area_scan, key, value)

    def _write_int_to_settings(self, key: str, value: int) -> None:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return
        setattr(motion.settings.z_stack_area_scan, key, value)

    def _write_check_to_settings(self, key: str, value: int) -> None:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return
        setattr(motion.settings.z_stack_area_scan, key, value != 0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decimals_for_step(step_mm: float) -> int:
        if step_mm <= 0:
            return 2
        decimals = max(2, -int(math.floor(math.log10(step_mm))))
        rounded = round(step_mm, decimals)
        while decimals > 2 and round(step_mm, decimals - 1) == rounded:
            decimals -= 1
            rounded = round(step_mm, decimals)
        return decimals

    def _get_printer_step_mm(self) -> float:
        motion = get_app_context().motion
        if motion is not None and motion.settings is not None:
            return motion.settings.step_size / 1_000_000.0
        return 0.04

    def _get_current_position_mm(self) -> tuple[float, float, float] | None:
        """Return (x_mm, y_mm, z_mm) or None if the motion controller is unavailable."""
        ctx = get_app_context()
        if ctx.motion is None or not ctx.motion.is_ready():
            warning("ZStackAreaScanWidget: motion controller not ready")
            return None
        return ctx.motion.get_position().to_mm()

    def _update_summary(self) -> None:
        x, y, z = self._x_axis, self._y_axis, self._z_axis

        if not (x.is_configured and y.is_configured and z.is_configured):
            self._summary_label.setText("")
            self._start_btn.setEnabled(False)
            return

        # These are guaranteed non-None by is_configured
        x_range = abs(x.end_mm - x.start_mm)  # type: ignore[operator]
        y_range = abs(y.end_mm - y.start_mm)  # type: ignore[operator]
        z_range = abs(z.end_mm - z.start_mm)  # type: ignore[operator]

        n_x = int(x_range / x.step_mm) + 1 if x.step_mm > 0 else 1
        n_y = int(y_range / y.step_mm) + 1 if y.step_mm > 0 else 1
        n_z = int(z_range / z.step_mm) + 1 if z.step_mm > 0 else 1

        total_stacks = n_x * n_y
        total_images = total_stacks * n_z

        total_seconds = math.ceil(total_images * TIME_PER_IMAGE + total_stacks * 1.0)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            time_str = f"{hours}h {minutes}m {secs}s"
        elif minutes:
            time_str = f"{minutes}m {secs}s"
        else:
            time_str = f"{secs}s"

        self._summary_label.setText(
            f"Grid: {n_x} × {n_y} positions  |  {n_z} Z slices each  |  "
            f"{total_images} images total  |  Est. time: {time_str}"
        )
        self._start_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Set-position slots
    # ------------------------------------------------------------------

    def _set_x_start(self) -> None:
        pos = self._get_current_position_mm()
        if pos is None:
            self._x_axis.mark_unavailable("start")
            return
        self._x_axis.set_start(pos[0])
        self._update_summary()

    def _set_x_end(self) -> None:
        pos = self._get_current_position_mm()
        if pos is None:
            self._x_axis.mark_unavailable("end")
            return
        self._x_axis.set_end(pos[0])
        self._update_summary()

    def _set_y_start(self) -> None:
        pos = self._get_current_position_mm()
        if pos is None:
            self._y_axis.mark_unavailable("start")
            return
        self._y_axis.set_start(pos[1])
        self._update_summary()

    def _set_y_end(self) -> None:
        pos = self._get_current_position_mm()
        if pos is None:
            self._y_axis.mark_unavailable("end")
            return
        self._y_axis.set_end(pos[1])
        self._update_summary()

    def _set_z_start(self) -> None:
        pos = self._get_current_position_mm()
        if pos is None:
            self._z_axis.mark_unavailable("start")
            return
        self._z_axis.set_start(pos[2])
        self._update_summary()

    def _set_z_end(self) -> None:
        pos = self._get_current_position_mm()
        if pos is None:
            self._z_axis.mark_unavailable("end")
            return
        self._z_axis.set_end(pos[2])
        self._update_summary()

    # ------------------------------------------------------------------
    # Focus stack slots
    # ------------------------------------------------------------------

    def _on_fs_enabled_changed(self) -> None:
        self._fs_settings_widget.setVisible(self._fs_enable_check.isChecked())

    def _on_advanced_toggled(self, checked: bool) -> None:
        self._advanced_widget.setVisible(checked)
        self._advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def _on_slab_enabled_changed(self) -> None:
        self._slab_params_widget.setVisible(self._slab_check.isChecked())

    def _build_focus_stack_config(self) -> FocusStackRoutineConfig | None:
        if not self._fs_enable_check.isChecked():
            return None
        slab: tuple[int, int] | None = None
        if self._slab_check.isChecked():
            slab = (self._slab_size_spin.value(), self._slab_overlap_spin.value())
        return FocusStackRoutineConfig(
            no_align=self._no_align_check.isChecked(),
            keep_size=self._keep_size_check.isChecked(),
            crop=self._crop_check.isChecked(),
            sharpness=self._sharpness_spin.value(),
            cull=self._cull_threshold_spin.value() if self._cull_check.isChecked() else None,
            workers=self._workers_spin.value(),
            slab=slab,
        )

    # ------------------------------------------------------------------
    # Start slot
    # ------------------------------------------------------------------

    def _on_start_clicked(self) -> None:
        x, y, z = self._x_axis, self._y_axis, self._z_axis

        if not (x.is_configured and y.is_configured and z.is_configured):
            return

        output_folder = self._output_folder.resolved_path
        if not OutputFolderWidget.confirm_if_exists(output_folder, self):
            return

        # Guaranteed non-None by is_configured
        x_start: float = x.start_mm  # type: ignore[assignment]
        x_end: float = x.end_mm  # type: ignore[assignment]
        y_start: float = y.start_mm  # type: ignore[assignment]
        y_end: float = y.end_mm  # type: ignore[assignment]
        z_start: float = z.start_mm  # type: ignore[assignment]
        z_end: float = z.end_mm  # type: ignore[assignment]

        decimals = max(x.decimals, y.decimals, z.decimals)

        dlg = _ConfirmAreaScanDialog(
            x_start=x_start,
            x_end=x_end,
            x_step_mm=x.step_mm,
            y_start=y_start,
            y_end=y_end,
            y_step_mm=y.step_mm,
            z_start=z_start,
            z_end=z_end,
            z_step_mm=z.step_mm,
            step_decimals=decimals,
            output_folder=output_folder,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.is_ready():
            error("ZStackAreaScanWidget: motion controller not ready — cannot start scan")
            return

        _NM_PER_MM = 1_000_000
        routine = ZStackAreaScan(
            motion=motion,
            x_start_nm=round(x_start * _NM_PER_MM),
            x_end_nm=round(x_end * _NM_PER_MM),
            x_step_nm=round(x.step_mm * _NM_PER_MM),
            y_start_nm=round(y_start * _NM_PER_MM),
            y_end_nm=round(y_end * _NM_PER_MM),
            y_step_nm=round(y.step_mm * _NM_PER_MM),
            z_start_nm=round(z_start * _NM_PER_MM),
            z_end_nm=round(z_end * _NM_PER_MM),
            z_step_nm=round(z.step_mm * _NM_PER_MM),
            output_folder=output_folder,
            focus_stack_config=self._build_focus_stack_config(),
        )
        motion.start_routine(routine)

        self._last_output_folder = output_folder
        self._enter_running_state()

    # ------------------------------------------------------------------
    # Routine state helpers
    # ------------------------------------------------------------------

    def _enter_running_state(self) -> None:
        self._start_btn.setEnabled(False)
        self._fs_enable_check.setEnabled(False)
        self._fs_settings_widget.setEnabled(False)
        self._open_folder_btn.set_folder(self._last_output_folder)
        self._poll_timer.start()

    def _exit_running_state(self) -> None:
        self._poll_timer.stop()
        self._fs_enable_check.setEnabled(True)
        self._fs_settings_widget.setEnabled(self._fs_enable_check.isChecked())
        self._update_summary()

    def _poll_routine_state(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.routine_running:
            self._exit_running_state()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def output_folder(self) -> str:
        return self._output_folder.resolved_path