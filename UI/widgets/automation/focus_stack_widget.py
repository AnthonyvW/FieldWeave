from __future__ import annotations

import math
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
)
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices

from common.app_context import get_app_context
from common.logger import warning, error
from motion.routines.z_stack_scan import ZStackScan
from post_processing.routines.focus_stack_routine import FocusStackConfig
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
        self._setup_ui()


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
        self._step_spin.setValue(printer_step)
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

        # Depth radius
        depth_row = QWidget()
        depth_layout = QHBoxLayout(depth_row)
        depth_layout.setContentsMargins(0, 0, 0, 0)
        depth_layout.setSpacing(8)
        depth_layout.addWidget(QLabel("Depth radius:"))

        self._depth_radius_spin = QSpinBox()
        self._depth_radius_spin.setFixedHeight(28)
        self._depth_radius_spin.setMinimum(0)
        self._depth_radius_spin.setMaximum(20)
        self._depth_radius_spin.setValue(1)
        self._depth_radius_spin.setToolTip(
            "Restrict pixel selection to a window of [peak-R, peak+R] frames "
            "around each pixel's best-focus frame. 0 = disabled."
        )
        depth_layout.addWidget(self._depth_radius_spin)
        depth_layout.addStretch(1)
        fs_settings_layout.addWidget(depth_row)

        # Smooth source
        smooth_row = QWidget()
        smooth_layout = QHBoxLayout(smooth_row)
        smooth_layout.setContentsMargins(0, 0, 0, 0)
        smooth_layout.setSpacing(8)
        smooth_layout.addWidget(QLabel("Smooth source radius:"))

        self._smooth_source_spin = QSpinBox()
        self._smooth_source_spin.setFixedHeight(28)
        self._smooth_source_spin.setMinimum(0)
        self._smooth_source_spin.setMaximum(99)
        self._smooth_source_spin.setValue(15)
        self._smooth_source_spin.setToolTip(
            "Apply two passes of median filtering to the source-frame map after "
            "selection. Removes isolated outlier frame assignments. 0 = disabled."
        )
        smooth_layout.addWidget(self._smooth_source_spin)
        smooth_layout.addStretch(1)
        fs_settings_layout.addWidget(smooth_row)

        # Sigma
        sigma_row = QWidget()
        sigma_layout = QHBoxLayout(sigma_row)
        sigma_layout.setContentsMargins(0, 0, 0, 0)
        sigma_layout.setSpacing(8)
        sigma_layout.addWidget(QLabel("Focus map sigma:"))

        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setFixedHeight(28)
        self._sigma_spin.setDecimals(1)
        self._sigma_spin.setMinimum(0.5)
        self._sigma_spin.setMaximum(20.0)
        self._sigma_spin.setSingleStep(0.5)
        self._sigma_spin.setValue(5.0)
        self._sigma_spin.setToolTip(
            "Gaussian smoothing radius for focus maps. Larger values produce "
            "smoother region boundaries."
        )
        sigma_layout.addWidget(self._sigma_spin)
        sigma_layout.addStretch(1)
        fs_settings_layout.addWidget(sigma_row)

        # Warp model
        warp_row = QWidget()
        warp_layout = QHBoxLayout(warp_row)
        warp_layout.setContentsMargins(0, 0, 0, 0)
        warp_layout.setSpacing(8)
        warp_layout.addWidget(QLabel("Alignment:"))

        self._no_align_check = QCheckBox("Skip alignment")
        self._no_align_check.setChecked(True)
        self._no_align_check.setToolTip(
            "Skip ECC alignment. Enable when images are already registered."
        )
        self._no_align_check.stateChanged.connect(self._on_no_align_changed)
        warp_layout.addWidget(self._no_align_check)
        warp_layout.addStretch(1)
        fs_settings_layout.addWidget(warp_row)

        # Score power
        power_row = QWidget()
        power_layout = QHBoxLayout(power_row)
        power_layout.setContentsMargins(0, 0, 0, 0)
        power_layout.setSpacing(8)
        power_layout.addWidget(QLabel("Score power:"))

        self._score_power_spin = QDoubleSpinBox()
        self._score_power_spin.setFixedHeight(28)
        self._score_power_spin.setDecimals(1)
        self._score_power_spin.setMinimum(1.0)
        self._score_power_spin.setMaximum(5.0)
        self._score_power_spin.setSingleStep(0.5)
        self._score_power_spin.setValue(2.0)
        self._score_power_spin.setToolTip(
            "Exponent applied to the raw Tenengrad score before smoothing. "
            "Increase to 3-4 if halo contamination persists."
        )
        power_layout.addWidget(self._score_power_spin)
        power_layout.addStretch(1)
        fs_settings_layout.addWidget(power_row)

        # Save depth map
        self._save_depth_map_check = QCheckBox("Save depth map")
        self._save_depth_map_check.setChecked(True)
        self._save_depth_map_check.setToolTip(
            "Save a greyscale source-frame depth map alongside the stacked output."
        )
        fs_settings_layout.addWidget(self._save_depth_map_check)

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
        main_layout.addWidget(self._summary_label)

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
        self._pause_resume_btn.setObjectName("AreaScanSecondaryButton")
        self._pause_resume_btn.clicked.connect(self._on_pause_resume_clicked)
        controls_layout.addWidget(self._pause_resume_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedHeight(32)
        self._stop_btn.setObjectName("AreaScanStop")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        controls_layout.addWidget(self._stop_btn)

        self._controls_widget.setVisible(False)
        main_layout.addWidget(self._controls_widget)

        # ---- Status label ------------------------------------------------
        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._status_label.setObjectName("AreaScanSummary")
        main_layout.addWidget(self._status_label)

        # ---- Post-run results row (hidden until a run completes) ---------
        self._results_widget = QWidget()
        results_layout = QHBoxLayout(self._results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)

        self._open_folder_btn = QPushButton("Open Folder")
        self._open_folder_btn.setFixedHeight(30)
        self._open_folder_btn.setObjectName("AreaScanSecondaryButton")
        self._open_folder_btn.clicked.connect(self._on_open_folder_clicked)
        results_layout.addWidget(self._open_folder_btn)

        self._view_image_btn = QPushButton("View Stacked Image")
        self._view_image_btn.setFixedHeight(30)
        self._view_image_btn.setObjectName("AreaScanSecondaryButton")
        self._view_image_btn.clicked.connect(self._on_view_image_clicked)
        results_layout.addWidget(self._view_image_btn)

        results_layout.addStretch(1)
        self._results_widget.setVisible(False)
        main_layout.addWidget(self._results_widget)

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

    def _build_focus_stack_config(self) -> FocusStackConfig | None:
        """Build a FocusStackConfig from the current widget state, or None if disabled."""
        if not self._fs_enable_check.isChecked():
            return None

        cfg = FocusStackConfig()
        cfg.depth_radius = self._depth_radius_spin.value() or None
        cfg.smooth_source = self._smooth_source_spin.value() or None
        cfg.sigma = self._sigma_spin.value()
        cfg.score_power = self._score_power_spin.value()
        cfg.no_align = self._no_align_check.isChecked()
        # depth_map_path is left as None here; ZStackScan fills it in relative
        # to the output folder when focus_stack_config.depth_map_path is None.
        if not self._save_depth_map_check.isChecked():
            cfg.depth_map_path = ""  # empty string signals "don't save"
        return cfg

    def _update_summary(self) -> None:
        """Refresh the summary label and enable/disable the start button."""
        if self._z_start is None or self._z_end is None:
            self._summary_label.setText("")
            self._start_btn.setEnabled(False)
            return

        distance = abs(self._z_end - self._z_start)
        step = self._step_spin.value()
        decimals = self._step_spin.decimals()
        fmt = f".{decimals}f"

        if step <= 0:
            self._summary_label.setText("")
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
        self._start_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_fs_enabled_changed(self) -> None:
        self._fs_settings_widget.setEnabled(self._fs_enable_check.isChecked())

    def _on_no_align_changed(self) -> None:
        pass  # Reserved for warp model selector if added later

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
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._last_output_folder))

    def _on_view_image_clicked(self) -> None:
        if self._last_stacked_path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._last_stacked_path))

    def _on_pause_resume_clicked(self) -> None:
        if self._routine is None:
            return
        if self._routine.is_paused:
            self._routine.resume()
            self._pause_resume_btn.setText("Pause")
            self._status_label.setText("Running...")
        else:
            self._routine.pause()
            self._pause_resume_btn.setText("Resume")
            self._status_label.setText("Paused.")

    def _on_stop_clicked(self) -> None:
        if self._routine is not None:
            self._routine.stop()
        self._status_label.setText("Stopping...")

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
        self._status_label.setText("Running...")
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
        self._status_label.setText("Finished.")
        self._routine = None
        self._update_summary()

        if self._last_output_folder is not None:
            self._view_image_btn.setVisible(
                self._last_stacked_path is not None
                and Path(self._last_stacked_path).exists()
            )
            self._results_widget.setVisible(True)

    def _poll_routine_state(self) -> None:
        if self._routine is None or not self._routine.is_running:
            self._exit_running_state()
            return
        self._pause_resume_btn.setText(
            "Resume" if self._routine.is_paused else "Pause"
        )
        self._status_label.setText(self._routine.activity)

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