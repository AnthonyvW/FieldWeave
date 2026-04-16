from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

_STEPS: list[tuple[str, str]] = [
    (
        "Select the first reference slot",
        "Navigate to the first sample slot in the tray using the movement "
        "controls.\n\n"
        "Centre the slot opening in the camera preview. The system will record "
        "this position as the origin reference for the slot grid.",
    ),
    (
        "Confirm slot geometry",
        "Verify that the slot boundary is clearly visible and centred in the "
        "frame.\n\n"
        "Adjust the position if necessary, then confirm. The system will "
        "measure the slot dimensions and spacing from this reference.",
    ),
    (
        "Navigate to the last reference slot",
        "Move to the last slot in the tray (opposite corner of the grid).\n\n"
        "Centre it in the camera preview. This second reference point allows "
        "the system to calculate precise positions for all intermediate slots.",
    ),
    (
        "Verify and save",
        "The system will compute the full slot map using both reference "
        "positions.\n\n"
        "A grid overlay will appear in the camera preview. Confirm the overlay "
        "aligns with the physical slots before saving the calibration.",
    ),
]

_DESCRIPTION = (
    "<b>Purpose:</b><br>"
    "Maps the position of every sample slot so the system can navigate "
    "to each one accurately and repeatably.<br><br>"
    "<b>What it does:</b><br>"
    "• Records reference positions for the slots<br>"
    "• Calculates slot spacing and geometry<br>"
    "• Enables automatic slot navigation<br><br>"
    "<b>What you need:</b><br>"
    "• An empty sample tray mounted on the machine<br>"
    "• Clear visibility of the start of each slot<br>"
    "• Approximately 5 minutes<br><br>"
    "<b>Process:</b><br>"
    "The calibration will guide you through the process of calibrating the sample slot positions."
)


class SlotCalibrationWidget(QWidget):
    """Step-through widget for the Sample Slot Position Calibration procedure."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_step: int = 0
        self._total_steps: int = len(_STEPS)
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        title = QLabel("Sample Slot Position Calibration")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #5a5a5a;")
        main_layout.addWidget(title)

        self._stack = QStackedWidget()
        main_layout.addWidget(self._stack)

        self._stack.addWidget(self._build_info_page())
        self._stack.addWidget(self._build_steps_page())
        self._stack.setCurrentIndex(0)

    def _build_info_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        description = QLabel(_DESCRIPTION)
        description.setWordWrap(True)
        description.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        description.setStyleSheet(
            "QLabel {"
            "  font-size: 14px; color: #000000;"
            "  background: #f8f8f8; padding: 20px;"
            "  border: 1px solid #e0e0e0;"
            "}"
        )
        layout.addWidget(description)
        layout.addStretch()

        start_btn = QPushButton("Start Calibration")
        start_btn.setStyleSheet(
            "QPushButton {"
            "  font-size: 16px; font-weight: bold; padding: 12px 30px;"
            "  background: #dbdbdb; border: 2px solid #b3b4b6;"
            "}"
            "QPushButton:hover { background: #b3b4b6; }"
        )
        start_btn.setMinimumHeight(45)
        start_btn.clicked.connect(self._start_calibration)
        layout.addWidget(start_btn)

        return page

    def _build_steps_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addStretch()

        steps_group = QGroupBox("Calibration Steps")
        steps_layout = QVBoxLayout(steps_group)
        steps_layout.setSpacing(10)

        self._step_indicator = QLabel()
        self._step_indicator.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #5a5a5a;"
        )
        steps_layout.addWidget(self._step_indicator)

        self._step_title = QLabel()
        self._step_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #3a3a3a;"
        )
        steps_layout.addWidget(self._step_title)

        self._step_body = QLabel()
        self._step_body.setWordWrap(True)
        self._step_body.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._step_body.setStyleSheet(
            "font-size: 13px; padding: 15px; background: #f8f8f8;"
            "border-radius: 4px; border: 1px solid #e0e0e0; color: #5a5a5a;"
        )
        self._step_body.setMinimumHeight(110)
        steps_layout.addWidget(self._step_body)

        nav_layout = QHBoxLayout()
        self._prev_btn = QPushButton("Previous")
        self._prev_btn.clicked.connect(self._previous_step)
        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(self._next_step)
        self._finish_btn = QPushButton("Finish Calibration")
        self._finish_btn.clicked.connect(self._finish)

        nav_layout.addWidget(self._prev_btn)
        nav_layout.addWidget(self._next_btn)
        nav_layout.addWidget(self._finish_btn)
        steps_layout.addLayout(nav_layout)

        layout.addWidget(steps_group)
        layout.addStretch()

        return page

    def _start_calibration(self) -> None:
        self._current_step = 0
        self._update_step_display()
        self._stack.setCurrentIndex(1)

    def _update_step_display(self) -> None:
        step_title, step_body = _STEPS[self._current_step]
        self._step_indicator.setText(f"Step {self._current_step + 1} of {self._total_steps}")
        self._step_title.setText(step_title)
        self._step_body.setText(step_body)
        self._prev_btn.setEnabled(self._current_step > 0)
        is_last = self._current_step == self._total_steps - 1
        self._next_btn.setVisible(not is_last)
        self._finish_btn.setVisible(is_last)

    def _next_step(self) -> None:
        if self._current_step < self._total_steps - 1:
            self._current_step += 1
            self._update_step_display()

    def _previous_step(self) -> None:
        if self._current_step > 0:
            self._current_step -= 1
            self._update_step_display()

    def _finish(self) -> None:
        pass

    def reset(self) -> None:
        """Reset to the info page."""
        self._stack.setCurrentIndex(0)