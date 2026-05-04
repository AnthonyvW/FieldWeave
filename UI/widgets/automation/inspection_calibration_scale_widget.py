from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from common.app_context import get_app_context
from common.logger import error, info
from motion.models import Position
from motion.routines.inspection_calibration_scale_routine import InspectionCalibrationScaleRoutine
from UI.widgets.automation.output_folder_widget import OutputFolderWidget, ViewImageWidget

_NM_PER_MM = 1_000_000


# ---------------------------------------------------------------------------
# Confirmation dialog
# ---------------------------------------------------------------------------

class _ConfirmDialog(QDialog):
    """Modal dialog summarising the planned calibration scale routine."""

    def __init__(
        self,
        output_path: str,
        current_x_mm: float,
        current_y_mm: float,
        current_z_mm: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm Calibration Scale Routine")
        self.setModal(True)
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Ready to start calibration scale routine?")
        title.setObjectName("CalScaleDialogTitle")
        layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setObjectName("SampleDivider")
        layout.addWidget(line)

        form = QFormLayout()
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        rows: list[tuple[str, str]] = [
            ("Start X",      f"{current_x_mm:.4f} mm"),
            ("Start Y",      f"{current_y_mm:.4f} mm"),
            ("Start Z",      f"{current_z_mm:.4f} mm"),
            ("Step size",    "0.4 mm"),
            ("Output path",  output_path),
            ("Save folder",  "raw_calibration_scale/"),
        ]
        for label_text, value_text in rows:
            lbl = QLabel(label_text + ":")
            lbl.setObjectName("CalScaleRowLabel")
            val = QLabel(value_text)
            val.setObjectName("CalScaleRowValue")
            val.setWordWrap(True)
            form.addRow(lbl, val)

        layout.addLayout(form)

        note = QLabel(
            "The routine will autofocus, detect the scale bar axis, then step "
            "along the bar saving images at each position until the end is reached."
        )
        note.setWordWrap(True)
        note.setObjectName("CalScaleNote")
        layout.addWidget(note)

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

class InspectionCalibrationScaleWidget(QWidget):
    """Widget for configuring and running the inspection calibration scale routine."""

    mode_name: str = "Calibration Scale"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._routine: InspectionCalibrationScaleRoutine | None = None
        self._last_output_path: str | None = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        self._inspect_cal_warning = QLabel(
            "Inspection calibration has not been completed. "
            "Please run Inspection Calibration before using the Calibration Scale routine."
        )
        self._inspect_cal_warning.setObjectName("CalErrorLabel")
        self._inspect_cal_warning.setWordWrap(True)
        self._inspect_cal_warning.setVisible(False)
        main_layout.addWidget(self._inspect_cal_warning)

        # ---- Saved position group ----------------------------------------
        position_group = QGroupBox("Scale Bar Start Position")
        pos_vbox = QVBoxLayout(position_group)
        pos_vbox.setContentsMargins(10, 8, 10, 8)
        pos_vbox.setSpacing(6)

        self._pos_label = QLabel("Saved position: Not set")
        self._pos_label.setObjectName("CalScalePosLabel")
        pos_vbox.addWidget(self._pos_label)

        pos_btn_row = QHBoxLayout()
        pos_btn_row.setSpacing(6)

        self._set_pos_btn = QPushButton("Set Position")
        self._set_pos_btn.setFixedHeight(30)
        self._set_pos_btn.setToolTip("Save the current stage XYZ as the scale bar start position")
        self._set_pos_btn.setObjectName("CalSecondaryButton")
        self._set_pos_btn.clicked.connect(self._on_set_position_clicked)
        pos_btn_row.addWidget(self._set_pos_btn)

        self._goto_pos_btn = QPushButton("Go to Position")
        self._goto_pos_btn.setFixedHeight(30)
        self._goto_pos_btn.setEnabled(False)
        self._goto_pos_btn.setToolTip("Move the stage to the saved scale bar start position")
        self._goto_pos_btn.setObjectName("CalSecondaryButton")
        self._goto_pos_btn.clicked.connect(self._on_goto_position_clicked)
        pos_btn_row.addWidget(self._goto_pos_btn)

        self._clear_pos_btn = QPushButton("Clear Position")
        self._clear_pos_btn.setFixedHeight(30)
        self._clear_pos_btn.setEnabled(False)
        self._clear_pos_btn.setToolTip("Remove the saved scale bar start position")
        self._clear_pos_btn.setObjectName("CalSecondaryButton")
        self._clear_pos_btn.clicked.connect(self._on_clear_position_clicked)
        pos_btn_row.addWidget(self._clear_pos_btn)

        pos_btn_row.addStretch()
        pos_vbox.addLayout(pos_btn_row)

        self._pos_status_label = QLabel("")
        self._pos_status_label.setObjectName("CalScaleStatusLabel")
        self._pos_status_label.hide()
        pos_vbox.addWidget(self._pos_status_label)

        main_layout.addWidget(position_group)

        # ---- Calibration info group --------------------------------------
        cal_info_group = QGroupBox("Calibration Info")
        cal_info_vbox = QVBoxLayout(cal_info_group)
        cal_info_vbox.setContentsMargins(10, 8, 10, 8)
        cal_info_vbox.setSpacing(4)

        self._last_calibrated_label = QLabel("Last calibrated: —")
        self._last_calibrated_label.setObjectName("CalScalePosLabel")
        cal_info_vbox.addWidget(self._last_calibrated_label)

        self._dpi_label = QLabel("DPI: —")
        self._dpi_label.setObjectName("CalScalePosLabel")
        cal_info_vbox.addWidget(self._dpi_label)

        main_layout.addWidget(cal_info_group)

        # ---- Output folder (matches area scan pattern) -------------------
        self._output_folder = OutputFolderWidget()
        main_layout.addWidget(self._output_folder)

        # ---- Post-run results row (hidden until a run completes) ---------
        self._results_widget = ViewImageWidget("View Calibration Image")
        main_layout.addWidget(self._results_widget)

        # ---- Start button ------------------------------------------------
        self._start_btn = QPushButton("Start Automation")
        self._start_btn.setFixedHeight(34)
        self._start_btn.setObjectName("CalScaleStart")
        self._start_btn.clicked.connect(self._on_start_clicked)
        main_layout.addWidget(self._start_btn)

        # ---- Pause / Resume / Stop row (hidden until running) ------------
        self._controls_widget = QWidget()
        controls_layout = QHBoxLayout(self._controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self._pause_resume_btn = QPushButton("Pause")
        self._pause_resume_btn.setFixedHeight(32)
        self._pause_resume_btn.setObjectName("CalSecondaryButton")
        self._pause_resume_btn.clicked.connect(self._on_pause_resume_clicked)
        controls_layout.addWidget(self._pause_resume_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedHeight(32)
        self._stop_btn.setObjectName("CalScaleStop")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        controls_layout.addWidget(self._stop_btn)

        self._controls_widget.setVisible(False)
        main_layout.addWidget(self._controls_widget)

        main_layout.addStretch(1)

        # ---- Poll timer --------------------------------------------------
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_routine_state)

        self._refresh_position_display()
        self._refresh_calibration_info()
        self._refresh_inspection_calibration_state()
        get_app_context().machine_vision.settings_changed.connect(self._on_settings_changed)

    # ------------------------------------------------------------------
    # showEvent — refresh calibration guard on tab switch
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._refresh_inspection_calibration_state()

    # ------------------------------------------------------------------
    # Settings change slot
    # ------------------------------------------------------------------

    def _on_settings_changed(self) -> None:
        self._refresh_calibration_info()
        self._refresh_position_display()
        self._refresh_inspection_calibration_state()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_inspection_calibrated(self) -> bool:
        return bool(get_app_context().machine_vision.settings.inspect_calibration.last_calibrated)

    def _refresh_inspection_calibration_state(self) -> None:
        calibrated = self._is_inspection_calibrated()
        self._inspect_cal_warning.setVisible(not calibrated)
        self._start_btn.setEnabled(calibrated)

    def _get_motion(self):
        return get_app_context().motion

    def _get_saved_position(self) -> tuple[int, int, int] | None:
        """Return the saved (x_nm, y_nm, z_nm) from machine vision settings, or None."""
        icp = get_app_context().machine_vision.settings.inspection_calibration_position
        if not icp.is_set:
            return None
        return icp.x_nm, icp.y_nm, icp.z_nm

    def _refresh_calibration_info(self) -> None:
        s = get_app_context().machine_vision.settings
        last_cal = s.inspect_calibration.last_calibrated
        if last_cal:
            try:
                dt = datetime.fromisoformat(last_cal)
                self._last_calibrated_label.setText(
                    f"Last calibrated: {dt.strftime('%Y-%m-%d %H:%M')}"
                )
            except ValueError:
                self._last_calibrated_label.setText(f"Last calibrated: {last_cal}")
        else:
            self._last_calibrated_label.setText("Last calibrated: —")
        if s.dpi is not None:
            self._dpi_label.setText(f"DPI: {s.dpi:.1f}")
        else:
            self._dpi_label.setText("DPI: —")

    def _refresh_position_display(self) -> None:
        saved = self._get_saved_position()
        if saved is not None:
            x_mm = saved[0] / _NM_PER_MM
            y_mm = saved[1] / _NM_PER_MM
            z_mm = saved[2] / _NM_PER_MM
            self._pos_label.setText(
                f"Saved X: {x_mm:.3f}  Y: {y_mm:.3f}  Z: {z_mm:.3f} mm"
            )
            self._goto_pos_btn.setEnabled(True)
            self._clear_pos_btn.setEnabled(True)
        else:
            self._pos_label.setText("Saved position: Not set")
            self._goto_pos_btn.setEnabled(False)
            self._clear_pos_btn.setEnabled(False)

    def _set_pos_status(self, text: str) -> None:
        self._pos_status_label.setText(text)
        self._pos_status_label.setVisible(bool(text))

    # ------------------------------------------------------------------
    # Position slots
    # ------------------------------------------------------------------

    def _on_set_position_clicked(self) -> None:
        motion = self._get_motion()
        if motion is None or not motion.is_ready():
            self._set_pos_status("Motion controller not ready.")
            return
        
        pos = motion.get_position()

        mv = get_app_context().machine_vision
        mv.settings.inspection_calibration_position.x_nm = pos.x
        mv.settings.inspection_calibration_position.y_nm = pos.y
        mv.settings.inspection_calibration_position.z_nm = pos.z
        mv.settings.inspection_calibration_position.is_set = True
        mv.save_settings()
        info(
            f"[CalibrationScaleWidget] Start position saved:"
            f" X={pos.x / _NM_PER_MM:.3f} mm"
            f" Y={pos.y / _NM_PER_MM:.3f} mm"
            f" Z={pos.z / _NM_PER_MM:.3f} mm"
        )
        self._refresh_position_display()
        self._set_pos_status(
            f"Position saved:"
            f" ({pos.x / _NM_PER_MM:.3f},"
            f" {pos.y / _NM_PER_MM:.3f},"
            f" {pos.z / _NM_PER_MM:.3f}) mm"
        )

    def _on_goto_position_clicked(self) -> None:
        motion = self._get_motion()
        if motion is None or not motion.is_ready():
            self._set_pos_status("Motion controller not ready.")
            return
        saved = self._get_saved_position()
        if saved is None:
            self._set_pos_status("No position saved.")
            return
        x_nm, y_nm, z_nm = saved

        motion.move_to_position(Position(x=x_nm, y=y_nm, z=z_nm), wait=False)

        self._set_pos_status(
            f"Moving to ({x_nm / _NM_PER_MM:.3f},"
            f" {y_nm / _NM_PER_MM:.3f},"
            f" {z_nm / _NM_PER_MM:.3f}) mm…"
        )

    def _on_clear_position_clicked(self) -> None:
        mv = get_app_context().machine_vision
        mv.settings.inspection_calibration_position.x_nm = 0
        mv.settings.inspection_calibration_position.y_nm = 0
        mv.settings.inspection_calibration_position.z_nm = 0
        mv.settings.inspection_calibration_position.is_set = False
        mv.save_settings()
        info("[CalibrationScaleWidget] Start position cleared")
        self._refresh_position_display()
        self._set_pos_status("Position cleared.")

    # ------------------------------------------------------------------
    # Output path slot
    # ------------------------------------------------------------------

    def _resolve_output_path(self) -> str:
        return self._output_folder.resolved_path

    def _find_result_image(self, output_path: str) -> str | None:
        folder = Path(output_path)
        ctx = get_app_context()
        if ctx.camera is not None:
            ext = ctx.camera.underlying_camera.settings.fformat.value
            candidate = folder / f"{folder.name}.{ext}"
            if candidate.exists():
                return str(candidate)
        fallback = folder / "stitched.jpg"
        return str(fallback) if fallback.exists() else None

    # ------------------------------------------------------------------
    # Routine slots
    # ------------------------------------------------------------------

    def _on_start_clicked(self) -> None:
        motion = self._get_motion()
        if motion is None or not motion.is_ready():
            error("InspectionCalibrationScaleWidget: motion controller not ready")
            return

        output_path = self._resolve_output_path()
        if not OutputFolderWidget.confirm_if_exists(output_path, self):
            return

        pos = self._get_saved_position()
        if pos is not None:
            current_x_mm = pos[0] / _NM_PER_MM
            current_y_mm = pos[1] / _NM_PER_MM
            current_z_mm = pos[2] / _NM_PER_MM
        else:
            stage_pos = motion.get_position()
            current_x_mm = stage_pos.x / _NM_PER_MM
            current_y_mm = stage_pos.y / _NM_PER_MM
            current_z_mm = stage_pos.z / _NM_PER_MM

        dlg = _ConfirmDialog(
            output_path=output_path,
            current_x_mm=current_x_mm,
            current_y_mm=current_y_mm,
            current_z_mm=current_z_mm,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        if pos is not None:
            start_position = Position(x=pos[0], y=pos[1], z=pos[2])
        else:
            start_position = None

        ctx = get_app_context()
        if ctx.motion.routine_running:
            error("TreeCoreWidget: a routine is already running")
            ctx.toast.error("A routine is already running.")
            return


        self._routine = InspectionCalibrationScaleRoutine(
            motion=motion,
            output_path=output_path,
            start_position=start_position,
        )
        motion.start_routine(self._routine)

        self._last_output_path = output_path
        self._results_widget.hide_result()
        self._enter_running_state()

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
        self._output_folder.setEnabled(False)
        self._set_pos_btn.setEnabled(False)
        self._goto_pos_btn.setEnabled(False)
        self._clear_pos_btn.setEnabled(False)
        self._pause_resume_btn.setText("Pause")
        self._controls_widget.setVisible(True)
        self._poll_timer.start()

    def _exit_running_state(self) -> None:
        self._poll_timer.stop()
        self._start_btn.setEnabled(self._is_inspection_calibrated())
        self._output_folder.setEnabled(True)
        self._set_pos_btn.setEnabled(True)
        self._controls_widget.setVisible(False)
        self._routine = None
        self._refresh_position_display()
        self._refresh_calibration_info()
        if self._last_output_path is not None:
            self._results_widget.show_result(self._last_output_path, self._find_result_image(self._last_output_path))

    def _poll_routine_state(self) -> None:
        if self._routine is None or not self._routine.is_running:
            self._exit_running_state()
            return
        self._pause_resume_btn.setText(
            "Resume" if self._routine.is_paused else "Pause"
        )