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
from motion.routines.z_stack_area_scan import ZStackAreaScan


# ---------------------------------------------------------------------------
# Confirmation dialog
# ---------------------------------------------------------------------------

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

        # Rough estimate: ~3.15 s/image + 1 s XY travel per stack
        total_seconds = math.ceil(total_images * 3.15 + total_stacks * 1.0)
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
            lbl.setStyleSheet("font-size: 13px; color: #555;")
            lbl.setFixedWidth(150)
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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        group = QGroupBox(f"{axis_label} Axis")
        group.setStyleSheet(group_style)
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
        self._set_start_btn.setStyleSheet(_button_style())
        start_layout.addWidget(self._set_start_btn)

        self._start_label = QLabel("Not set")
        self._start_label.setMinimumWidth(110)
        self._start_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._start_label.setStyleSheet("font-size: 13px; color: #555;")
        start_layout.addWidget(self._start_label)

        group_layout.addWidget(start_row)

        # End row
        end_row = QWidget()
        end_layout = QHBoxLayout(end_row)
        end_layout.setContentsMargins(0, 0, 0, 0)
        end_layout.setSpacing(8)

        self._set_end_btn = QPushButton(f"Set {axis_label} End")
        self._set_end_btn.setFixedHeight(30)
        self._set_end_btn.setStyleSheet(_button_style())
        end_layout.addWidget(self._set_end_btn)

        self._end_label = QLabel("Not set")
        self._end_label.setMinimumWidth(110)
        self._end_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._end_label.setStyleSheet("font-size: 13px; color: #555;")
        end_layout.addWidget(self._end_label)

        group_layout.addWidget(end_row)

        # Step row
        step_row = QWidget()
        step_layout = QHBoxLayout(step_row)
        step_layout.setContentsMargins(0, 0, 0, 0)
        step_layout.setSpacing(8)

        step_lbl = QLabel("Step (mm):")
        step_lbl.setStyleSheet("font-size: 13px;")
        step_layout.addWidget(step_lbl)

        self._step_spin = QDoubleSpinBox()
        self._step_spin.setFixedHeight(28)
        self._step_spin.setDecimals(step_decimals)
        self._step_spin.setSuffix(" mm")
        self._step_spin.setMinimum(step_mm)
        self._step_spin.setMaximum(300.0)
        self._step_spin.setSingleStep(step_mm)
        self._step_spin.setValue(step_mm)
        self._step_spin.setStyleSheet("""
            QDoubleSpinBox {
                font-size: 13px;
                padding: 2px 4px;
                border: 1px solid rgb(180, 180, 180);
                border-radius: 0px;
            }
        """)
        step_layout.addWidget(self._step_spin)

        fmt = f".{step_decimals}f"
        min_label = QLabel(f"(min: {step_mm:{fmt}} mm)")
        min_label.setStyleSheet("font-size: 11px; color: #777;")
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
# Shared style helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class ZStackAreaScanWidget(QWidget):
    """Widget for configuring and running a Z-stack area scan across an XY grid."""

    _DEFAULT_OUTPUT_PLACEHOLDER: str = "Default: ./output/<timestamp>"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

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
        main_layout.addWidget(self._x_axis)

        self._y_axis = _AxisRangeWidget("Y", printer_step, step_decimals)
        self._y_axis.connect_start(self._set_y_start)
        self._y_axis.connect_end(self._set_y_end)
        self._y_axis.connect_step_changed(self._update_summary)
        main_layout.addWidget(self._y_axis)

        self._z_axis = _AxisRangeWidget("Z", printer_step, step_decimals)
        self._z_axis.connect_start(self._set_z_start)
        self._z_axis.connect_end(self._set_z_end)
        self._z_axis.connect_step_changed(self._update_summary)
        main_layout.addWidget(self._z_axis)

        # ---- Output folder ----
        output_group_style = """
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
        output_group = QGroupBox("Output Folder")
        output_group.setStyleSheet(output_group_style)
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
        browse_btn.setStyleSheet(_button_style())
        browse_btn.clicked.connect(self._browse_output_folder)
        output_layout.addWidget(browse_btn)

        main_layout.addWidget(output_group)

        # ---- Summary label ----
        self._summary_label = QLabel("")
        self._summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._summary_label.setStyleSheet("font-size: 12px; color: #444; padding: 2px 0;")
        self._summary_label.setWordWrap(True)
        main_layout.addWidget(self._summary_label)

        main_layout.addStretch(1)

        # ---- Start button ----
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
        try:
            ctx = get_app_context()
            if ctx.settings is not None:
                step_nm: int = ctx.settings.motion.step_size
                return step_nm / 1_000_000.0
        except Exception:
            pass
        return 0.04

    def _get_current_position_mm(self) -> tuple[float, float, float] | None:
        """Return (x_mm, y_mm, z_mm) or None if the motion controller is unavailable."""
        ctx = get_app_context()
        if ctx.motion is None or not ctx.motion.is_ready():
            warning("ZStackAreaScanWidget: motion controller not ready")
            return None
        return ctx.motion.get_position().to_mm()

    def _resolve_output_folder(self) -> str:
        text = self._output_edit.text().strip()
        if text:
            return text
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(Path("output") / timestamp)

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

        total_seconds = math.ceil(total_images * 3.15 + total_stacks * 1.0)
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

    def _browse_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            self._output_edit.text().strip() or "./output/",
        )
        if folder:
            self._output_edit.setText(folder)

    # ------------------------------------------------------------------
    # Start slot
    # ------------------------------------------------------------------

    def _on_start_clicked(self) -> None:
        x, y, z = self._x_axis, self._y_axis, self._z_axis

        if not (x.is_configured and y.is_configured and z.is_configured):
            return

        output_folder = self._resolve_output_folder()

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
        try:
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
            )
            motion.start_routine(routine)
        except Exception as exc:
            error(f"ZStackAreaScanWidget: failed to start routine — {exc}")
            return

        self._enter_running_state()

    # ------------------------------------------------------------------
    # Routine state helpers
    # ------------------------------------------------------------------

    def _enter_running_state(self) -> None:
        self._start_btn.setEnabled(False)
        self._poll_timer.start()

    def _exit_running_state(self) -> None:
        self._poll_timer.stop()
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
        return self._resolve_output_folder()