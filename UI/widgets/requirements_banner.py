from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QLabel, QWidget

from common.app_context import get_app_context
from motion.requirements import NO_REQUIREMENTS, MotionRequirements

_POLL_INTERVAL_MS = 500


class RequirementsBanner(QLabel):
    """Explains why an automation or calibration cannot run with the current axis setup.

    Polls the motion controller, stays hidden while *requirements* are met and
    emits ``availability_changed`` whenever that flips.
    """

    availability_changed = Signal(bool)

    def __init__(self, requirements: MotionRequirements = NO_REQUIREMENTS, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._requirements = requirements
        self._available = True
        self.setWordWrap(True)
        self.setStyleSheet(
            "QLabel { color: #b3261e; background: #fdecea; border: 1px solid #f5c2c0;"
            " border-radius: 4px; padding: 6px 8px; font-size: 12px; }"
        )
        self.hide()

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    @property
    def available(self) -> bool:
        return self._available

    def set_requirements(self, requirements: MotionRequirements) -> None:
        self._requirements = requirements
        self.refresh()

    def refresh(self) -> None:
        text = self._requirements.describe_problems(get_app_context().motion)
        self.setText(text)
        self.setVisible(bool(text))
        available = not text
        if available != self._available:
            self._available = available
            self.availability_changed.emit(available)
