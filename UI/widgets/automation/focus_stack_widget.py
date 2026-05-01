from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

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
from PySide6.QtCore import Qt, QTimer, QUrl, QMetaObject, Slot
from PySide6.QtGui import QDesktopServices

def _open_path(path: str) -> None:
    if sys.platform.startswith("linux"):
        subprocess.Popen(["xdg-open", path])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

from common.app_context import get_app_context
from common.logger import warning, error
from motion.routines.z_stack_scan import ZStackScan
from post_processing.routines.focus_stack_routine import FocusStackRoutineConfig, FocusStackResult
from post_processing.routines.post_processing_routine import RoutineResult
from UI.widgets.automation.output_folder_widget import OutputFolderWidget


# ---------------------------------------------------------------------------
# Confirmation dialog
# ---------------------------------------------------------------------------

class _ConfirmAutomationDialog(QDialog):
    """Modal dialog summarising the focus stack before starting."""

    def __init__(
        self,
        z_start: float,
        z_end: float,
        step_mm: float,
        step_decimals: int,
        output_folder: str,
        will_focus_stack: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm Focus Stack")
        self.setModal(True)
        self.setMinimumWidth(360)

        distance = abs(z_end - z_start)
        n_frames = int(distance / step_mm) + 1
        total_seconds = math.ceil(n_frames * 3.15)
        minutes, seconds = divmod(total_seconds, 60)

        fmt = f".{step_decimals}f"
        time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Ready to start focus stack?")
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
            ("Start Z",          f"{z_start:{fmt}} mm"),
            ("End Z",            f"{z_end:{fmt}} mm"),
            ("Range",            f"{distance:{fmt}} mm"),
            ("Step size",        f"{step_mm:{fmt}} mm"),
            ("Estimated frames", str(n_frames)),
            ("Estimated time",   time_str),
            ("Output folder",    output_folder),
            ("Focus stack",      "Yes (after capture)" if will_focus_stack else "No"),
        ]
        for label_text, value_text in rows:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            lbl = QLabel(label_text + ":")
            lbl.setObjectName("AreaScanRowLabel")
            lbl.setFixedWidth(130)
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
# Main widget
# ---------------------------------------------------------------------------

class FocusStackWidget(QWidget):
    """Widget for configuring and running a Z-axis focus stack."""

    mode_name: str = "Focus Stacking"
    _SECS_PER_FRAME: float = 3.15

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._z_start: float | None = None
        self._z_end: float | None = None
        self._routine: ZStackScan | None = None
        self._last_output_folder: str | None = None
        self._last_stacked_path: str | None = None
        self._pending_stack_result: FocusStackResult | None = None
        self._setup_ui()

        ctx = get_app_context()
        ctx.post_processing.add_routine_complete_listener(self._on_routine_complete)


    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # ---- Z positions group -------------------------------------------
        z_group = QGroupBox("Z Positions")
        z_layout = QVBoxLayout(z_group)
        z_layout.setContentsMargins(10, 8, 10, 8)
        z_layout.setSpacing(8)

        start_row = QWidget()
        start_layout = QHBoxLayout(start_row)
        start_layout.setContentsMargins(0, 0, 0, 0)
        start_layout.setSpacing(8)

        self._set_start_btn = QPushButton("Set Start Position")
        self._set_start_btn.setFixedHeight(32)
        self._set_start_btn.clicked.connect(self._set_start_position)
        start_layout.addWidget(self._set_start_btn)

        self._start_label = QLabel("Not set")
        self._start_label.setMinimumWidth(100)
        self._start_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._start_label.setObjectName("AreaScanAxisReadout")
        start_layout.addWidget(self._start_label)

        z_layout.addWidget(start_row)

        end_row = QWidget()
        end_layout = QHBoxLayout(end_row)
        end_layout.setContentsMargins(0, 0, 0, 0)
        end_layout.setSpacing(8)

        self._set_end_btn = QPushButton("Set End Position")
        self._set_end_btn.setFixedHeight(32)
        self._set_end_btn.clicked.connect(self._set_end_position)
        end_layout.addWidget(self._set_end_btn)

        self._end_label = QLabel("Not set")
        self._end_label.setMinimumWidth(100)
        self._end_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._end_label.setObjectName("AreaScanAxisReadout")
        end_layout.addWidget(self._end_label)

        z_layout.addWidget(end_row)

        main_layout.addWidget(z_group)

        # ---- Step size group ---------------------------------------------
        step_group = QGroupBox("Step Size")
        step_layout = QHBoxLayout(step_group)
        step_layout.setContentsMargins(10, 8, 10, 8)
        step_layout.setSpacing(8)

        step_label = QLabel("Step (mm):")
        step_layout.addWidget(step_label)

        printer_step = self._get_printer_step_mm()
        step_decimals = self._decimals_for_step(printer_step)

        self._step_spin = QDoubleSpinBox()
        self._step_spin.setFixedHeight(30)
        self._step_spin.setDecimals(step_decimals)
        self._step_spin.setSuffix(" mm")
        self._step_spin.setMinimum(printer_step)
        self._step_spin.setMaximum(10.0)
        self._step_spin.setSingleStep(printer_step)
        self._step_spin.setValue(0.2)
        self._step_spin.valueChanged.connect(self._update_summary)
        step_layout.addWidget(self._step_spin)

        fmt = f".{step_decimals}f"
        min_step_label = QLabel(f"(min: {printer_step:{fmt}} mm)")
        min_step_label.setObjectName("AreaScanMinLabel")
        step_layout.addWidget(min_step_label)
        step_layout.addStretch(1)

        main_layout.addWidget(step_group)

        # ---- Focus stack settings group ----------------------------------
        fs_group = QGroupBox("Focus Stack Settings")
        fs_layout = QVBoxLayout(fs_group)
        fs_layout.setContentsMargins(10, 8, 10, 8)
        fs_layout.setSpacing(6)

        self._fs_enable_check = QCheckBox("Run focus stack after capture")
        self._fs_enable_check.setChecked(True)
        self._fs_enable_check.stateChanged.connect(self._on_fs_enabled_changed)
        fs_layout.addWidget(self._fs_enable_check)

        self._fs_settings_widget = QWidget()
        fs_settings_layout = QVBoxLayout(self._fs_settings_widget)
        fs_settings_layout.setContentsMargins(0, 4, 0, 0)
        fs_settings_layout.setSpacing(6)

        # Keep original size
        self._keep_size_check = QCheckBox("Keep original size")
        self._keep_size_check.setChecked(True)
        self._keep_size_check.setToolTip(
            "Keep the output image the same size as the input images. "
            "Warps are applied in-place rather than expanding the canvas."
        )
        fs_settings_layout.addWidget(self._keep_size_check)

        # Advanced settings (collapsible)
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

        # Skip alignment
        self._no_align_check = QCheckBox("Skip alignment")
        self._no_align_check.setChecked(False)
        self._no_align_check.setToolTip(
            "Skip ECC alignment. Use when images are already registered."
        )
        self._no_align_check.stateChanged.connect(self._on_no_align_changed)
        advanced_layout.addWidget(self._no_align_check)

        # Crop to intersection
        self._crop_check = QCheckBox("Crop to intersection")
        self._crop_check.setChecked(False)
        self._crop_check.setToolTip(
            "Crop the output to the largest rectangle covered by every frame after "
            "alignment. Removes border regions but shrinks the output image."
        )
        advanced_layout.addWidget(self._crop_check)

        # Approach distance
        approach_row = QWidget()
        approach_row_layout = QHBoxLayout(approach_row)
        approach_row_layout.setContentsMargins(0, 0, 0, 0)
        approach_row_layout.setSpacing(8)
        approach_row_layout.addWidget(QLabel("Approach distance:"))

        self._approach_spin = QDoubleSpinBox()
        self._approach_spin.setFixedHeight(28)
        self._approach_spin.setDecimals(3)
        self._approach_spin.setSuffix(" mm")
        self._approach_spin.setMinimum(0.0)
        self._approach_spin.setMaximum(10.0)
        self._approach_spin.setSingleStep(0.1)
        self._approach_spin.setValue(0.4)
        self._approach_spin.setToolTip(
            "Before starting the scan, the stage will overshoot the near end by this "
            "distance, then return to it. This eliminates backlash and wobble from "
            "the direction change at the start of the sweep."
        )
        approach_row_layout.addWidget(self._approach_spin)
        approach_row_layout.addStretch(1)
        advanced_layout.addWidget(approach_row)

        # Sharpness
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
        sharpness_layout.addWidget(self._sharpness_spin)
        sharpness_layout.addStretch(1)
        advanced_layout.addWidget(sharpness_row)

        # Cull threshold
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
        cull_layout.addWidget(self._cull_threshold_spin)
        cull_layout.addStretch(1)
        advanced_layout.addWidget(cull_row)

        # Slabbing
        self._slab_check = QCheckBox("Enable slabbing")
        self._slab_check.setChecked(False)
        self._slab_check.setToolTip(
            "Split the image set into overlapping sub-stacks, stack each "
            "independently, then fuse the results. Reduces peak RAM for large stacks."
        )
        self._slab_check.stateChanged.connect(self._on_slab_enabled_changed)
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
        slab_params_layout.addWidget(self._slab_overlap_spin)
        slab_params_layout.addStretch(1)
        advanced_layout.addWidget(self._slab_params_widget)

        # Workers
        workers_row = QWidget()
        workers_layout = QHBoxLayout(workers_row)
        workers_layout.setContentsMargins(0, 0, 0, 0)
        workers_layout.setSpacing(8)
        workers_layout.addWidget(QLabel("Workers:"))

        self._workers_spin = QSpinBox()
        self._workers_spin.setFixedHeight(28)
        self._workers_spin.setMinimum(1)
        self._workers_spin.setMaximum(16)
        self._workers_spin.setValue(3)
        self._workers_spin.setToolTip(
            "Number of parallel workers for stacking. Higher values are faster "
            "but increase peak RAM by ~100 MiB per additional worker."
        )
        workers_layout.addWidget(self._workers_spin)
        workers_layout.addStretch(1)
        advanced_layout.addWidget(workers_row)

        fs_settings_layout.addWidget(self._advanced_widget)
        fs_layout.addWidget(self._fs_settings_widget)
        main_layout.addWidget(fs_group)

        # ---- Output folder group -----------------------------------------
        self._output_folder = OutputFolderWidget()
        main_layout.addWidget(self._output_folder)

        # ---- Summary label -----------------------------------------------
        self._summary_label = QLabel("")
        self._summary_label.setObjectName("AreaScanSummary")
        self._summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._summary_label.setWordWrap(True)
        self._summary_label.setVisible(False)
        main_layout.addWidget(self._summary_label)

        # ---- Post-run results row (hidden until a run completes) ---------
        self._results_widget = QWidget()
        results_layout = QHBoxLayout(self._results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)

        self._open_folder_btn = QPushButton("Open Folder")
        self._open_folder_btn.setFixedHeight(30)
        self._open_folder_btn.clicked.connect(self._on_open_folder_clicked)
        results_layout.addWidget(self._open_folder_btn, 1)

        self._view_image_btn = QPushButton("View Stacked Image")
        self._view_image_btn.setFixedHeight(30)
        self._view_image_btn.clicked.connect(self._on_view_image_clicked)
        results_layout.addWidget(self._view_image_btn, 1)

        self._results_widget.setVisible(False)
        main_layout.addWidget(self._results_widget)

        # ---- Start button ------------------------------------------------
        self._start_btn = QPushButton("Start Automation")
        self._start_btn.setObjectName("AreaScanStart")
        self._start_btn.setFixedHeight(34)
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start_clicked)
        main_layout.addWidget(self._start_btn)

        # ---- Pause / Resume / Stop row (hidden until running) ------------
        self._controls_widget = QWidget()
        controls_layout = QHBoxLayout(self._controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self._pause_resume_btn = QPushButton("Pause")
        self._pause_resume_btn.setFixedHeight(32)
        self._pause_resume_btn.setObjectName("AutomationPause")
        self._pause_resume_btn.clicked.connect(self._on_pause_resume_clicked)
        controls_layout.addWidget(self._pause_resume_btn, 1)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedHeight(32)
        self._stop_btn.setObjectName("AutomationStop")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        controls_layout.addWidget(self._stop_btn, 1)

        self._controls_widget.setVisible(False)
        main_layout.addWidget(self._controls_widget)

        main_layout.addStretch(1)

        # ---- Poll timer --------------------------------------------------
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_routine_state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decimals_for_step(step_mm: float) -> int:
        """
        Return the number of decimal places needed to represent step_mm
        without trailing zeros (minimum 2).

        Examples:
            0.04    -> 2
            0.004   -> 3
            0.0004  -> 4
            0.1     -> 2  (clamped minimum)
        """
        if step_mm <= 0:
            return 2
        decimals = max(2, -int(math.floor(math.log10(step_mm))))
        rounded = round(step_mm, decimals)
        while decimals > 2 and round(step_mm, decimals - 1) == rounded:
            decimals -= 1
            rounded = round(step_mm, decimals)
        return decimals

    def _get_printer_step_mm(self) -> float:
        """Return the printer's minimum step size in mm from settings, defaulting to 0.04 mm."""
        try:
            ctx = get_app_context()
            if ctx.settings is not None:
                step_nm: int = ctx.settings.motion.step_size
                return step_nm / 1_000_000.0
        except Exception:
            pass
        return 0.04

    def _get_current_z_mm(self) -> float | None:
        """Return current Z position in mm, or None if unavailable."""
        ctx = get_app_context()
        if ctx.motion is None or not ctx.motion.is_ready():
            warning("FocusStackWidget: motion controller not ready")
            return None
        _, _, z_mm = ctx.motion.get_position().to_mm()
        return z_mm

    def _format_z(self, z: float) -> str:
        decimals = self._step_spin.decimals()
        return f"{z:.{decimals}f} mm"

    def _resolve_output_folder(self) -> str:
        return self._output_folder.resolved_path

    def _build_focus_stack_config(self) -> FocusStackRoutineConfig | None:
        """Build a FocusStackRoutineConfig from the current widget state, or None if disabled."""
        if not self._fs_enable_check.isChecked():
            return None

        slab: tuple[int, int] | None = None
        if self._slab_check.isChecked():
            slab = (self._slab_size_spin.value(), self._slab_overlap_spin.value())

        return FocusStackRoutineConfig(
            no_align=self._no_align_check.isChecked(),
            sharpness=self._sharpness_spin.value(),
            cull=self._cull_threshold_spin.value() if self._cull_check.isChecked() else None,
            workers=self._workers_spin.value(),
            crop=self._crop_check.isChecked(),
            keep_size=self._keep_size_check.isChecked(),
            slab=slab,
        )

    def _update_summary(self) -> None:
        """Refresh the summary label and enable/disable the start button."""
        if self._z_start is None or self._z_end is None:
            self._summary_label.setVisible(False)
            self._start_btn.setEnabled(False)
            return

        distance = abs(self._z_end - self._z_start)
        step = self._step_spin.value()
        decimals = self._step_spin.decimals()
        fmt = f".{decimals}f"

        if step <= 0:
            self._summary_label.setVisible(False)
            self._start_btn.setEnabled(False)
            return

        n_frames = int(distance / step) + 1
        total_seconds = math.ceil(n_frames * self._SECS_PER_FRAME)
        minutes, secs = divmod(total_seconds, 60)
        time_str = f"{minutes}m {secs}s" if minutes else f"{secs}s"

        self._summary_label.setText(
            f"Range: {distance:{fmt}} mm\n"
            f"Frames: ~{n_frames}  |  Step: {step:{fmt}} mm  |  Est. time: {time_str}"
        )
        self._summary_label.setVisible(True)
        self._start_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_fs_enabled_changed(self) -> None:
        self._fs_settings_widget.setEnabled(self._fs_enable_check.isChecked())

    def _on_advanced_toggled(self, checked: bool) -> None:
        self._advanced_widget.setVisible(checked)
        self._advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def _on_no_align_changed(self) -> None:
        pass  # Reserved for warp model selector if added later

    def _on_slab_enabled_changed(self) -> None:
        self._slab_params_widget.setVisible(self._slab_check.isChecked())

    def _set_start_position(self) -> None:
        z = self._get_current_z_mm()
        if z is None:
            self._start_label.setText("Unavailable")
            return
        self._z_start = z
        self._start_label.setText(f"Z = {self._format_z(z)}")
        self._update_summary()

    def _set_end_position(self) -> None:
        z = self._get_current_z_mm()
        if z is None:
            self._end_label.setText("Unavailable")
            return
        self._z_end = z
        self._end_label.setText(f"Z = {self._format_z(z)}")
        self._update_summary()

    def _on_start_clicked(self) -> None:
        if self._z_start is None or self._z_end is None:
            return

        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.is_ready():
            error("FocusStackWidget: motion controller not ready — cannot start scan")
            return

        output_folder = self._resolve_output_folder()
        if not OutputFolderWidget.confirm_if_exists(output_folder, self):
            return
        step_mm = self._step_spin.value()
        focus_stack_config = self._build_focus_stack_config()

        dlg = _ConfirmAutomationDialog(
            z_start=self._z_start,
            z_end=self._z_end,
            step_mm=step_mm,
            step_decimals=self._step_spin.decimals(),
            output_folder=output_folder,
            will_focus_stack=focus_stack_config is not None,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        if ctx.motion.routine_running:
            error("FocusStackWidget: a routine is already running")
            ctx.toast.error("A routine is already running.")
            return

        _NM_PER_MM = 1_000_000
        self._routine = ZStackScan(
            motion=motion,
            z_start_nm=round(self._z_start * _NM_PER_MM),
            z_end_nm=round(self._z_end * _NM_PER_MM),
            step_nm=round(step_mm * _NM_PER_MM),
            output_folder=output_folder,
            approach_distance_nm=round(self._approach_spin.value() * _NM_PER_MM),
            focus_stack_config=focus_stack_config,
        )
        motion.start_routine(self._routine)

        self._last_output_folder = output_folder
        if focus_stack_config is not None:
            ext = focus_stack_config.output_extension
            self._last_stacked_path = str(Path(output_folder) / f"stacked.{ext}")
        else:
            self._last_stacked_path = None
        self._results_widget.setVisible(False)
        self._enter_running_state()

    def _on_open_folder_clicked(self) -> None:
        if self._last_output_folder is not None:
            _open_path(self._last_output_folder)

    def _on_view_image_clicked(self) -> None:
        if self._last_stacked_path is not None:
            _open_path(self._last_stacked_path)

    def _on_pause_resume_clicked(self) -> None:
        if self._routine is None:
            return
        if self._routine.is_paused:
            self._routine.resume()
            self._pause_resume_btn.setText("Pause")
        else:
            self._routine.pause()
            self._pause_resume_btn.setText("Resume")

    def _on_stop_clicked(self) -> None:
        if self._routine is not None:
            self._routine.stop()

    # ------------------------------------------------------------------
    # Routine state helpers
    # ------------------------------------------------------------------

    def _enter_running_state(self) -> None:
        self._start_btn.setEnabled(False)
        self._set_start_btn.setEnabled(False)
        self._set_end_btn.setEnabled(False)
        self._output_folder.setEnabled(False)
        self._fs_enable_check.setEnabled(False)
        self._fs_settings_widget.setEnabled(False)
        self._pause_resume_btn.setText("Pause")
        self._controls_widget.setVisible(True)
        self._poll_timer.start()

    def _exit_running_state(self) -> None:
        self._poll_timer.stop()
        self._start_btn.setEnabled(True)
        self._set_start_btn.setEnabled(True)
        self._set_end_btn.setEnabled(True)
        self._output_folder.setEnabled(True)
        self._fs_enable_check.setEnabled(True)
        self._fs_settings_widget.setEnabled(self._fs_enable_check.isChecked())
        self._controls_widget.setVisible(False)

        stacked_path = self._last_stacked_path
        self._routine = None
        self._update_summary()

        if self._last_output_folder is not None:
            self._view_image_btn.setVisible(
                stacked_path is not None and Path(stacked_path).exists()
            )
            self._results_widget.setVisible(True)

    def _on_routine_complete(self, result: RoutineResult) -> None:
        """Called from the routine's background thread — marshal Qt calls to main thread."""
        if not result.success:
            QMetaObject.invokeMethod(self, "_notify_failure", Qt.ConnectionType.QueuedConnection)
            return
        fs_result: FocusStackResult | None = result.get("focus_stack")
        if fs_result is not None:
            self._pending_stack_result = fs_result
        QMetaObject.invokeMethod(self, "_notify_success", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _notify_failure(self) -> None:
        ctx = get_app_context()
        if ctx.toast is not None:
            ctx.toast.error("Focus stack automation failed.")

    @Slot()
    def _notify_success(self) -> None:
        ctx = get_app_context()
        if ctx.toast is not None:
            ctx.toast.success("Focus stack automation complete.")
        fs_result = self._pending_stack_result
        self._pending_stack_result = None
        if fs_result is not None:
            preview = ctx.camera_preview
            if preview is not None:
                preview.show_static_image(fs_result.result_rgb)

    def _poll_routine_state(self) -> None:
        if self._routine is None or not self._routine.is_running:
            self._exit_running_state()
            return
        self._pause_resume_btn.setText(
            "Resume" if self._routine.is_paused else "Pause"
        )

    # ------------------------------------------------------------------
    # Public accessors (for the parent automation widget)
    # ------------------------------------------------------------------

    @property
    def z_start(self) -> float | None:
        return self._z_start

    @property
    def z_end(self) -> float | None:
        return self._z_end

    @property
    def step_mm(self) -> float:
        return self._step_spin.value()

    @property
    def output_folder(self) -> str:
        """Resolved output folder path (generates timestamp default if field is empty)."""
        return self._resolve_output_folder()