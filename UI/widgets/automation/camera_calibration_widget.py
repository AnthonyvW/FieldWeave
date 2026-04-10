from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from common.app_context import get_app_context
from common.logger import error
from motion.routines.camera_calibration_routine import CameraCalibrationRoutine

_NM_PER_MM = 1_000_000
_NM_PER_TICK = 10_000


# ---------------------------------------------------------------------------
# Confirmation dialog
# ---------------------------------------------------------------------------

class _ConfirmCalibrationDialog(QDialog):
    """Modal dialog summarising the planned calibration moves."""

    def __init__(
        self,
        move_x_mm: float,
        move_y_mm: float,
        current_x_mm: float,
        current_y_mm: float,
        already_calibrated: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm Camera Calibration")
        self.setModal(True)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Ready to calibrate camera?")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: rgb(200, 200, 200);")
        layout.addWidget(line)

        form = QFormLayout()
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        label_style = "font-size: 13px; color: #555;"
        value_style = "font-size: 13px; font-weight: bold;"

        rows: list[tuple[str, str]] = [
            ("Start X",   f"{current_x_mm:.4f} mm"),
            ("Start Y",   f"{current_y_mm:.4f} mm"),
            ("+X move",   f"{move_x_mm:.4f} mm"),
            ("+Y move",   f"{move_y_mm:.4f} mm"),
            ("Captures",  "3 stills (base, +X, +Y)"),
        ]
        for label_text, value_text in rows:
            lbl = QLabel(label_text + ":")
            lbl.setStyleSheet(label_style)
            val = QLabel(value_text)
            val.setStyleSheet(value_style)
            form.addRow(lbl, val)

        layout.addLayout(form)

        if already_calibrated:
            warn = QLabel("An existing calibration will be replaced.")
            warn.setStyleSheet("font-size: 12px; color: #c97020; padding-top: 4px;")
            warn.setWordWrap(True)
            layout.addWidget(warn)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Start Calibration")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class CameraCalibrationWidget(QWidget):
    """
    Widget for running the camera-to-stage spatial calibration routine.

    Displays the move distances that will be used (read from
    ``machine_vision.settings.camera_calibration`` at the time the routine
    starts), the current calibration status, and start / stop controls.
    Move distance settings are edited elsewhere (e.g. a settings page);
    this widget is intentionally read-only with respect to those values.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._routine: CameraCalibrationRoutine | None = None
        self._setup_ui()
        self._refresh_calibration_status()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        group_style = """
            QGroupBox {
                font-size: 13px;
                font-weight: normal;
                border: 1px solid rgb(180, 180, 180);
                border-radius: 0px;
                margin-top: 6px;
                padding-top: 4px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 3px;
            }
        """

        # ---- Current calibration status ---------------------------------
        status_group = QGroupBox("Current Calibration")
        status_group.setStyleSheet(group_style)
        status_form = QFormLayout(status_group)
        status_form.setContentsMargins(10, 8, 10, 8)
        status_form.setSpacing(6)

        self._cal_status_label = QLabel("Unknown")
        self._cal_status_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        status_form.addRow("Status:", self._cal_status_label)

        self._cal_dpi_label = QLabel("—")
        self._cal_dpi_label.setStyleSheet("font-size: 13px;")
        status_form.addRow("DPI:", self._cal_dpi_label)

        self._cal_ref_label = QLabel("—")
        self._cal_ref_label.setStyleSheet("font-size: 13px;")
        status_form.addRow("Reference XY:", self._cal_ref_label)

        main_layout.addWidget(status_group)

        # ---- Move distances (read-only, from settings) ------------------
        moves_group = QGroupBox("Calibration Moves (from settings)")
        moves_group.setStyleSheet(group_style)
        moves_form = QFormLayout(moves_group)
        moves_form.setContentsMargins(10, 8, 10, 8)
        moves_form.setSpacing(6)

        self._move_x_label = QLabel("—")
        self._move_x_label.setStyleSheet("font-size: 13px;")
        moves_form.addRow("+X move:", self._move_x_label)

        self._move_y_label = QLabel("—")
        self._move_y_label.setStyleSheet("font-size: 13px;")
        moves_form.addRow("+Y move:", self._move_y_label)

        main_layout.addWidget(moves_group)

        # ---- Start button -----------------------------------------------
        self._start_btn = QPushButton("Start Calibration")
        self._start_btn.setFixedHeight(34)
        self._start_btn.setStyleSheet("""
            QPushButton {
                background-color: #f28c28;
                color: white;
                border: 1px solid #c97020;
                border-radius: 0px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover   { background-color: #d97a20; }
            QPushButton:pressed { background-color: #bf6a18; }
            QPushButton:disabled {
                background-color: rgb(208, 211, 214);
                color: rgb(150, 153, 156);
                border: 1px solid rgb(170, 173, 176);
            }
        """)
        self._start_btn.clicked.connect(self._on_start_clicked)
        main_layout.addWidget(self._start_btn)

        # ---- Stop button (hidden until running) -------------------------
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedHeight(32)
        self._stop_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(200, 80, 70);
                color: white;
                border: 1px solid rgb(160, 60, 50);
                border-radius: 0px;
                font-size: 13px;
            }
            QPushButton:hover   { background-color: rgb(180, 65, 55); }
            QPushButton:pressed { background-color: rgb(160, 55, 45); }
        """)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        self._stop_btn.setVisible(False)
        main_layout.addWidget(self._stop_btn)

        # ---- Clear calibration button -----------------------------------
        self._clear_btn = QPushButton("Clear Calibration")
        self._clear_btn.setFixedHeight(30)
        self._clear_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(208, 211, 214);
                border: 1px solid rgb(150, 150, 150);
                border-radius: 0px;
                font-size: 13px;
                padding: 0 8px;
            }
            QPushButton:hover   { background-color: rgb(187, 190, 193); }
            QPushButton:pressed { background-color: rgb(170, 173, 175); }
            QPushButton:disabled {
                color: rgb(150, 153, 156);
            }
        """)
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        main_layout.addWidget(self._clear_btn)

        # ---- Routine status label ---------------------------------------
        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._status_label.setStyleSheet("font-size: 12px; color: #444; padding: 2px 0;")
        self._status_label.setWordWrap(True)
        main_layout.addWidget(self._status_label)

        main_layout.addStretch(1)

        # ---- Poll timer -------------------------------------------------
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_routine_state)

    # ------------------------------------------------------------------
    # Calibration status display
    # ------------------------------------------------------------------

    def _refresh_calibration_status(self) -> None:
        """Re-read the current calibration from the manager and update labels."""
        try:
            ctx = get_app_context()
            mv = ctx.machine_vision
            cal = mv.calibration
            cc = mv.settings.camera_calibration

            move_x_mm = cc.move_x_ticks * _NM_PER_TICK / _NM_PER_MM
            move_y_mm = cc.move_y_ticks * _NM_PER_TICK / _NM_PER_MM
            self._move_x_label.setText(
                f"{move_x_mm:.4f} mm  ({cc.move_x_ticks} ticks)"
            )
            self._move_y_label.setText(
                f"{move_y_mm:.4f} mm  ({cc.move_y_ticks} ticks)"
            )

            if cal is None:
                self._cal_status_label.setText("Not calibrated")
                self._cal_status_label.setStyleSheet(
                    "font-size: 13px; font-weight: bold; color: #999;"
                )
                self._cal_dpi_label.setText("—")
                self._cal_ref_label.setText("—")
                self._clear_btn.setEnabled(False)
            else:
                self._cal_status_label.setText("Calibrated")
                self._cal_status_label.setStyleSheet(
                    "font-size: 13px; font-weight: bold; color: #2a8a2a;"
                )
                dpi_str = f"{cal.dpi:.1f}" if cal.dpi is not None else "—"
                self._cal_dpi_label.setText(dpi_str)
                ref_x_mm = cal.ref_x * _NM_PER_TICK / _NM_PER_MM
                ref_y_mm = cal.ref_y * _NM_PER_TICK / _NM_PER_MM
                self._cal_ref_label.setText(f"{ref_x_mm:.3f} mm, {ref_y_mm:.3f} mm")
                self._clear_btn.setEnabled(True)

        except Exception:
            pass

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_start_clicked(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        mv = ctx.machine_vision

        if motion is None or not motion.is_ready():
            error("CameraCalibrationWidget: motion controller not ready")
            self._status_label.setText("Motion controller not ready.")
            return

        if not ctx.has_camera:
            error("CameraCalibrationWidget: no camera available")
            self._status_label.setText("No camera available.")
            return

        # Read current move distances from settings for the confirmation dialog.
        cc = mv.settings.camera_calibration
        move_x_mm = cc.move_x_ticks * _NM_PER_TICK / _NM_PER_MM
        move_y_mm = cc.move_y_ticks * _NM_PER_TICK / _NM_PER_MM

        try:
            pos = motion.get_position()
            current_x_mm = pos.x / _NM_PER_MM
            current_y_mm = pos.y / _NM_PER_MM
        except Exception:
            current_x_mm, current_y_mm = 0.0, 0.0

        dlg = _ConfirmCalibrationDialog(
            move_x_mm=move_x_mm,
            move_y_mm=move_y_mm,
            current_x_mm=current_x_mm,
            current_y_mm=current_y_mm,
            already_calibrated=mv.is_calibrated,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            self._routine = CameraCalibrationRoutine(motion=motion)
            self._routine.on_state_changed = self._on_routine_state_changed
            motion.start_routine(self._routine)
        except Exception as exc:
            error(f"CameraCalibrationWidget: failed to start routine — {exc}")
            self._status_label.setText(f"Failed to start: {exc}")
            return

        self._enter_running_state()

    def _on_stop_clicked(self) -> None:
        if self._routine is not None:
            self._routine.stop()
        self._status_label.setText("Stopping…")

    def _on_clear_clicked(self) -> None:
        try:
            ctx = get_app_context()
            ctx.machine_vision.clear_calibration()
        except Exception as exc:
            error(f"CameraCalibrationWidget: clear_calibration failed — {exc}")
        self._refresh_calibration_status()
        self._status_label.setText("Calibration cleared.")

    # ------------------------------------------------------------------
    # Routine state
    # ------------------------------------------------------------------

    def _on_routine_state_changed(
        self,
        job_name: str,
        activity: str,
        progress_current: int,
        progress_total: int,
        eta_seconds: int,
    ) -> None:
        """Called from the routine's background thread — update via the poll timer."""
        # Store the latest activity string; the poll timer reads it on the GUI thread.
        self._latest_activity = activity

    def _enter_running_state(self) -> None:
        self._start_btn.setEnabled(False)
        self._clear_btn.setEnabled(False)
        self._stop_btn.setVisible(True)
        self._status_label.setText("Running…")
        self._latest_activity: str = ""
        self._poll_timer.start()

    def _exit_running_state(self) -> None:
        self._poll_timer.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setVisible(False)
        self._routine = None
        self._refresh_calibration_status()

    def _poll_routine_state(self) -> None:
        """Called every 250 ms on the GUI thread to sync routine state to UI."""
        if self._routine is None or not self._routine.is_running:
            # Grab the final activity string before clearing the routine.
            final_activity = getattr(self, "_latest_activity", "")
            self._exit_running_state()
            if final_activity:
                self._status_label.setText(final_activity)
            return

        activity = getattr(self, "_latest_activity", "")
        if activity:
            prog = self._routine.progress_current
            total = self._routine.progress_total
            if total > 0:
                self._status_label.setText(f"[{prog}/{total}]  {activity}")
            else:
                self._status_label.setText(activity)