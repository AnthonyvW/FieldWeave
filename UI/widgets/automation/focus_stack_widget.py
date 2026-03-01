from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLineEdit,
    QFileDialog,
)
from PySide6.QtCore import Qt, QTimer

from common.app_context import get_app_context
from common.logger import warning, error
from motion.routines.z_stack_scan import ZStackScan


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
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: rgb(200, 200, 200);")
        layout.addWidget(line)

        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(6)

        rows: list[tuple[str, str]] = [
            ("Start Z", f"{z_start:{fmt}} mm"),
            ("End Z", f"{z_end:{fmt}} mm"),
            ("Range", f"{distance:{fmt}} mm"),
            ("Step size", f"{step_mm:{fmt}} mm"),
            ("Estimated frames", str(n_frames)),
            ("Estimated time", time_str),
            ("Output folder", output_folder),
        ]
        for label_text, value_text in rows:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            lbl = QLabel(label_text + ":")
            lbl.setStyleSheet("font-size: 13px; color: #555;")
            lbl.setFixedWidth(130)
            row_layout.addWidget(lbl)

            val = QLabel(value_text)
            val.setStyleSheet("font-size: 13px; font-weight: bold;")
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

    _SECS_PER_FRAME: float = 3.15
    _DEFAULT_OUTPUT_PLACEHOLDER: str = "Default: ./output/<timestamp>"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._z_start: float | None = None
        self._z_end: float | None = None
        self._setup_ui()

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

        # Z positions group
        z_group = QGroupBox("Z Positions")
        z_group.setStyleSheet(group_style)
        z_layout = QVBoxLayout(z_group)
        z_layout.setContentsMargins(10, 8, 10, 8)
        z_layout.setSpacing(8)

        start_row = QWidget()
        start_layout = QHBoxLayout(start_row)
        start_layout.setContentsMargins(0, 0, 0, 0)
        start_layout.setSpacing(8)

        self._set_start_btn = QPushButton("Set Start Position")
        self._set_start_btn.setFixedHeight(32)
        self._set_start_btn.setStyleSheet(self._button_style())
        self._set_start_btn.clicked.connect(self._set_start_position)
        start_layout.addWidget(self._set_start_btn)

        self._start_label = QLabel("Not set")
        self._start_label.setMinimumWidth(100)
        self._start_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._start_label.setStyleSheet("font-size: 13px; color: #555;")
        start_layout.addWidget(self._start_label)

        z_layout.addWidget(start_row)

        end_row = QWidget()
        end_layout = QHBoxLayout(end_row)
        end_layout.setContentsMargins(0, 0, 0, 0)
        end_layout.setSpacing(8)

        self._set_end_btn = QPushButton("Set End Position")
        self._set_end_btn.setFixedHeight(32)
        self._set_end_btn.setStyleSheet(self._button_style())
        self._set_end_btn.clicked.connect(self._set_end_position)
        end_layout.addWidget(self._set_end_btn)

        self._end_label = QLabel("Not set")
        self._end_label.setMinimumWidth(100)
        self._end_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._end_label.setStyleSheet("font-size: 13px; color: #555;")
        end_layout.addWidget(self._end_label)

        z_layout.addWidget(end_row)

        main_layout.addWidget(z_group)

        # Step size group
        step_group = QGroupBox("Step Size")
        step_group.setStyleSheet(group_style)
        step_layout = QHBoxLayout(step_group)
        step_layout.setContentsMargins(10, 8, 10, 8)
        step_layout.setSpacing(8)

        step_label = QLabel("Step (mm):")
        step_label.setStyleSheet("font-size: 13px;")
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
        self._step_spin.setStyleSheet("""
            QDoubleSpinBox {
                font-size: 13px;
                padding: 2px 4px;
                border: 1px solid rgb(180, 180, 180);
                border-radius: 0px;
            }
        """)
        self._step_spin.valueChanged.connect(self._update_summary)
        step_layout.addWidget(self._step_spin)

        fmt = f".{step_decimals}f"
        min_step_label = QLabel(f"(min: {printer_step:{fmt}} mm)")
        min_step_label.setStyleSheet("font-size: 11px; color: #777;")
        step_layout.addWidget(min_step_label)
        step_layout.addStretch(1)

        main_layout.addWidget(step_group)

        # Output folder group
        output_group = QGroupBox("Output Folder")
        output_group.setStyleSheet(group_style)
        output_layout = QHBoxLayout(output_group)
        output_layout.setContentsMargins(10, 8, 10, 8)
        output_layout.setSpacing(8)

        self._output_edit = QLineEdit()
        self._output_edit.setFixedHeight(30)
        self._output_edit.setPlaceholderText(self._DEFAULT_OUTPUT_PLACEHOLDER)
        self._output_edit.setStyleSheet("""
            QLineEdit {
                font-size: 13px;
                padding: 2px 4px;
                border: 1px solid rgb(180, 180, 180);
                border-radius: 0px;
            }
        """)
        output_layout.addWidget(self._output_edit, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedHeight(30)
        browse_btn.setStyleSheet(self._button_style())
        browse_btn.clicked.connect(self._browse_output_folder)
        output_layout.addWidget(browse_btn)

        main_layout.addWidget(output_group)

        # Summary label
        self._summary_label = QLabel("")
        self._summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._summary_label.setStyleSheet("font-size: 12px; color: #444; padding: 2px 0;")
        self._summary_label.setWordWrap(True)
        main_layout.addWidget(self._summary_label)

        # Start automation button
        self._start_btn = QPushButton("Start Automation")
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
            QPushButton:hover {
                background-color: #d97a20;
            }
            QPushButton:pressed {
                background-color: #bf6a18;
            }
            QPushButton:disabled {
                background-color: rgb(208, 211, 214);
                color: rgb(150, 153, 156);
                border: 1px solid rgb(170, 173, 176);
            }
        """)
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start_clicked)
        main_layout.addWidget(self._start_btn)

        # Timer for polling routine state on the UI thread
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_routine_state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _button_style() -> str:
        return """
            QPushButton {
                background-color: rgb(208, 211, 214);
                border: 1px solid rgb(150, 150, 150);
                border-radius: 0px;
                font-size: 13px;
                padding: 0 8px;
            }
            QPushButton:hover {
                background-color: rgb(187, 190, 193);
            }
            QPushButton:pressed {
                background-color: rgb(170, 173, 175);
            }
        """

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
        # Trim any unnecessary extra places
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
        return 0.04  # fallback: 40 000 nm = 0.04 mm

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
        """Return the user-specified folder, or generate the default timestamped path."""
        text = self._output_edit.text().strip()
        if text:
            return text
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(Path("output") / timestamp)

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

    def _browse_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            self._output_edit.text().strip() or "./output/",
        )
        if folder:
            self._output_edit.setText(folder)

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

        output_folder = self._resolve_output_folder()
        step_mm = self._step_spin.value()

        dlg = _ConfirmAutomationDialog(
            z_start=self._z_start,
            z_end=self._z_end,
            step_mm=step_mm,
            step_decimals=self._step_spin.decimals(),
            output_folder=output_folder,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.is_ready():
            error("FocusStackWidget: motion controller not ready — cannot start scan")
            return

        _NM_PER_MM = 1_000_000
        try:
            routine = ZStackScan(
                motion=motion,
                z_start_nm=round(self._z_start * _NM_PER_MM),
                z_end_nm=round(self._z_end * _NM_PER_MM),
                step_nm=round(step_mm * _NM_PER_MM),
                output_folder=output_folder,
            )
            motion.start_routine(routine)
        except Exception as exc:
            error(f"FocusStackWidget: failed to start routine — {exc}")
            return

        self._enter_running_state()

    # ------------------------------------------------------------------
    # Routine state helpers
    # ------------------------------------------------------------------

    def _enter_running_state(self) -> None:
        """Disable the start button while a routine is active."""
        self._start_btn.setEnabled(False)
        self._poll_timer.start()

    def _exit_running_state(self) -> None:
        """Re-enable the start button when the routine finishes."""
        self._poll_timer.stop()
        self._update_summary()

    def _poll_routine_state(self) -> None:
        """Called every 250 ms to detect when the routine has finished."""
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.routine_running:
            self._exit_running_state()

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