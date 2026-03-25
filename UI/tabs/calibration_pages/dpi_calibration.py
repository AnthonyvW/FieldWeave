from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_STEPS: list[tuple[str, str]] = [
    (
        "Position the calibration target",
        "Place a calibration target with known dimensions on the build plate.\n\n"
        "Ensure the target is flat, well-lit, and fully visible in the camera "
        "preview. Adjust lighting if necessary to eliminate shadows or glare.",
    ),
    (
        "Align the camera",
        "Use the movement controls to centre the calibration target in the "
        "camera preview.\n\n"
        "The target should fill as much of the frame as possible while remaining "
        "fully visible. Confirm the image is in sharp focus before continuing.",
    ),
    (
        "Capture and calculate",
        "The system will capture an image of the target and measure the known "
        "reference dimensions in pixels.\n\n"
        "Hold the target steady. Precise scaling factors will be calculated and "
        "saved for use in all subsequent measurements.",
    ),
]


class DpiCalibrationWidget(QWidget):
    """Step-through widget for the Precise DPI Calibration procedure."""

    # ------------------------------------------------------------------ init

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._current_step: int = 0
        self._total_steps: int = len(_STEPS)

        self._build_ui()
        self._update_step_display()

    # ---------------------------------------------------------- UI construction

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Title
        title = QLabel("Precise DPI Calibration")
        title.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #5a5a5a;"
        )
        main_layout.addWidget(title)

        # Description
        description = QLabel(
            "Fine-tunes the camera's pixels-per-millimetre ratio for accurate "
            "sample imaging and measurement."
        )
        description.setWordWrap(True)
        description.setStyleSheet("font-size: 13px; color: #7a7a7a;")
        main_layout.addWidget(description)

        main_layout.addStretch()

        # Steps group
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

        # Navigation buttons
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

        main_layout.addWidget(steps_group)
        main_layout.addStretch()

    # ---------------------------------------------------------- step control

    def _update_step_display(self) -> None:
        step_title, step_body = _STEPS[self._current_step]

        self._step_indicator.setText(
            f"Step {self._current_step + 1} of {self._total_steps}"
        )
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
        # Placeholder — wired up by the parent tab when needed.
        pass

    # ---------------------------------------------------------- public API

    def reset(self) -> None:
        """Reset to the first step (call before showing the widget again)."""
        self._current_step = 0
        self._update_step_display()