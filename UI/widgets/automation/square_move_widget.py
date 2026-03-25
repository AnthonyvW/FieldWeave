"""
Square move automation widget.

Provides :class:`SquareMoveWidget`, a PySide6 widget for configuring and
running :class:`~motion.routines.square_move.SquareMove`.

The widget lets the operator set the side length, then prompts with a
confirmation dialog before any motion begins.  Pause / resume / stop
controls are shown while the routine is active.

Usage::

    from motion.widgets.square_move_widget import SquareMoveWidget

    widget = SquareMoveWidget()
    widget.show()
"""

from __future__ import annotations

import threading

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from common.app_context import get_app_context
from common.logger import error, warning
from motion.routines.square_move_routine import SquareMove


# ---------------------------------------------------------------------------
# Confirmation dialog
# ---------------------------------------------------------------------------

class _ConfirmSquareMoveDialog(QDialog):
    """Modal dialog summarising the planned square move."""

    def __init__(
        self,
        side_mm: float,
        repeats: int,
        current_x_mm: float,
        current_y_mm: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm Square Move")
        self.setModal(True)
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Ready to start square move?")
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
            ("Side length",    f"{side_mm:.4f} mm"),
            ("Repeats",        str(repeats)),
            ("Total distance", f"{side_mm * 4 * repeats:.4f} mm"),
            ("Start X",        f"{current_x_mm:.4f} mm"),
            ("Start Y",        f"{current_y_mm:.4f} mm"),
            ("Pattern",        "(+X) → (+Y) → (−X) → (−Y)"),
        ]
        for label_text, value_text in rows:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            lbl = QLabel(label_text + ":")
            lbl.setStyleSheet("font-size: 13px; color: #555;")
            lbl.setFixedWidth(110)
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

class SquareMoveWidget(QWidget):
    """Widget for configuring and running a square move automation routine."""

    _DEFAULT_SIDE_MM: float = 10.0
    _MIN_SIDE_MM: float = 0.1
    _MAX_SIDE_MM: float = 200.0
    _DEFAULT_REPEATS: int = 5
    _MIN_REPEATS: int = 1
    _MAX_REPEATS: int = 999

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Set when the routine is launched so we can poll it.
        self._routine: SquareMove | None = None
        # Threading event used to pass the operator's confirm/cancel decision
        # into the routine's background thread.
        self._confirm_event: threading.Event = threading.Event()
        self._confirm_result: bool = False
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

        # ---- Settings (side length + repeats) ---------------------------
        settings_group = QGroupBox("Settings")
        settings_group.setStyleSheet(group_style)
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setContentsMargins(10, 8, 10, 8)
        settings_layout.setSpacing(8)

        spin_style = """
            QDoubleSpinBox, QSpinBox {
                font-size: 13px;
                padding: 2px 4px;
                border: 1px solid rgb(180, 180, 180);
                border-radius: 0px;
            }
        """

        # Side length row
        side_row = QWidget()
        side_row_layout = QHBoxLayout(side_row)
        side_row_layout.setContentsMargins(0, 0, 0, 0)
        side_row_layout.setSpacing(8)

        side_label = QLabel("Side length:")
        side_label.setStyleSheet("font-size: 13px;")
        side_label.setFixedWidth(90)
        side_row_layout.addWidget(side_label)

        self._side_spin = QDoubleSpinBox()
        self._side_spin.setFixedHeight(30)
        self._side_spin.setDecimals(4)
        self._side_spin.setSuffix(" mm")
        self._side_spin.setMinimum(self._MIN_SIDE_MM)
        self._side_spin.setMaximum(self._MAX_SIDE_MM)
        self._side_spin.setSingleStep(1.0)
        self._side_spin.setValue(self._DEFAULT_SIDE_MM)
        self._side_spin.setStyleSheet(spin_style)
        self._side_spin.valueChanged.connect(self._update_summary)
        side_row_layout.addWidget(self._side_spin)
        side_row_layout.addStretch(1)

        settings_layout.addWidget(side_row)

        # Repeats row
        repeats_row = QWidget()
        repeats_row_layout = QHBoxLayout(repeats_row)
        repeats_row_layout.setContentsMargins(0, 0, 0, 0)
        repeats_row_layout.setSpacing(8)

        repeats_label = QLabel("Repeats:")
        repeats_label.setStyleSheet("font-size: 13px;")
        repeats_label.setFixedWidth(90)
        repeats_row_layout.addWidget(repeats_label)

        self._repeats_spin = QSpinBox()
        self._repeats_spin.setFixedHeight(30)
        self._repeats_spin.setMinimum(self._MIN_REPEATS)
        self._repeats_spin.setMaximum(self._MAX_REPEATS)
        self._repeats_spin.setValue(self._DEFAULT_REPEATS)
        self._repeats_spin.setStyleSheet(spin_style)
        self._repeats_spin.valueChanged.connect(self._update_summary)
        repeats_row_layout.addWidget(self._repeats_spin)
        repeats_row_layout.addStretch(1)

        settings_layout.addWidget(repeats_row)

        main_layout.addWidget(settings_group)

        # ---- Summary -----------------------------------------------------
        self._summary_label = QLabel("")
        self._summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._summary_label.setStyleSheet("font-size: 12px; color: #444; padding: 2px 0;")
        self._summary_label.setWordWrap(True)
        main_layout.addWidget(self._summary_label)

        # ---- Start button ------------------------------------------------
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
            QPushButton:hover  { background-color: #d97a20; }
            QPushButton:pressed { background-color: #bf6a18; }
            QPushButton:disabled {
                background-color: rgb(208, 211, 214);
                color: rgb(150, 153, 156);
                border: 1px solid rgb(170, 173, 176);
            }
        """)
        self._start_btn.clicked.connect(self._on_start_clicked)
        main_layout.addWidget(self._start_btn)

        # ---- Pause / Resume / Stop row (hidden until running) ------------
        self._controls_widget = QWidget()
        controls_layout = QHBoxLayout(self._controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self._pause_resume_btn = QPushButton("Pause")
        self._pause_resume_btn.setFixedHeight(32)
        self._pause_resume_btn.setStyleSheet(self._button_style())
        self._pause_resume_btn.clicked.connect(self._on_pause_resume_clicked)
        controls_layout.addWidget(self._pause_resume_btn)

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
            QPushButton:hover  { background-color: rgb(180, 65, 55); }
            QPushButton:pressed { background-color: rgb(160, 55, 45); }
        """)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        controls_layout.addWidget(self._stop_btn)

        self._controls_widget.setVisible(False)
        main_layout.addWidget(self._controls_widget)

        # ---- Status label ------------------------------------------------
        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._status_label.setStyleSheet("font-size: 12px; color: #444; padding: 2px 0;")
        main_layout.addWidget(self._status_label)

        main_layout.addStretch(1)

        # ---- Poll timer --------------------------------------------------
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_routine_state)

        self._update_summary()

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
            QPushButton:hover   { background-color: rgb(187, 190, 193); }
            QPushButton:pressed { background-color: rgb(170, 173, 175); }
        """

    def _get_current_position_mm(self) -> tuple[float, float] | None:
        """Return (x_mm, y_mm) of the current stage position, or None."""
        try:
            ctx = get_app_context()
            if ctx.motion is None or not ctx.motion.is_ready():
                return None
            pos = ctx.motion.get_position()
            return pos.x / 1_000_000.0, pos.y / 1_000_000.0
        except Exception:
            return None

    def _update_summary(self) -> None:
        side = self._side_spin.value()
        repeats = self._repeats_spin.value()
        total = side * 4 * repeats
        self._summary_label.setText(
            f"Side: {side:.4f} mm  |  Repeats: {repeats}  |  Total distance: {total:.4f} mm"
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_start_clicked(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.is_ready():
            error("SquareMoveWidget: motion controller not ready — cannot start")
            self._status_label.setText("Motion controller not ready.")
            return

        side_mm = self._side_spin.value()
        repeats = self._repeats_spin.value()
        pos = self._get_current_position_mm()
        current_x_mm, current_y_mm = pos if pos is not None else (0.0, 0.0)

        dlg = _ConfirmSquareMoveDialog(
            side_mm=side_mm,
            repeats=repeats,
            current_x_mm=current_x_mm,
            current_y_mm=current_y_mm,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # Prepare the confirmation event before creating the routine so
        # the callback is ready the moment the background thread calls it.
        self._confirm_event.clear()
        self._confirm_result = True   # operator already confirmed via dialog

        def _on_confirm(side: float) -> bool:  # noqa: ARG001
            # The UI dialog already collected confirmation; just return it.
            return self._confirm_result

        try:
            self._routine = SquareMove(
                motion=motion,
                on_confirm=_on_confirm,
                side_mm=side_mm,
                repeats=repeats,
            )
            motion.start_routine(self._routine)
        except Exception as exc:
            error(f"SquareMoveWidget: failed to start routine — {exc}")
            self._status_label.setText(f"Failed to start: {exc}")
            return

        self._enter_running_state()

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
        self._side_spin.setEnabled(False)
        self._repeats_spin.setEnabled(False)
        self._pause_resume_btn.setText("Pause")
        self._controls_widget.setVisible(True)
        self._status_label.setText("Running...")
        self._poll_timer.start()

    def _exit_running_state(self) -> None:
        self._poll_timer.stop()
        self._start_btn.setEnabled(True)
        self._side_spin.setEnabled(True)
        self._repeats_spin.setEnabled(True)
        self._controls_widget.setVisible(False)
        self._status_label.setText("Finished.")
        self._routine = None

    def _poll_routine_state(self) -> None:
        """Called every 250 ms to detect when the routine has finished."""
        if self._routine is None or not self._routine.is_running:
            self._exit_running_state()
            return

        # Keep the pause/resume label in sync if state changed externally.
        if self._routine.is_paused:
            self._pause_resume_btn.setText("Resume")
        else:
            self._pause_resume_btn.setText("Pause")

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def side_mm(self) -> float:
        return self._side_spin.value()

    @property
    def repeats(self) -> int:
        return self._repeats_spin.value()