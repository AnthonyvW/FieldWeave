from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from motion.motion_config import AXES, MotionSystemSettings
from UI.settings.pages.shared import SettingsGroupBase


class AxesSettingsWidget(SettingsGroupBase):
    """Axes group: which axes exist, which are homed, and whether to home on startup."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Axes", parent)
        self._checks: dict[str, QCheckBox] = {}
        self._saved: dict[str, bool] = {}
        self._build()

    def _build(self) -> None:
        vbox = QVBoxLayout(self)
        vbox.setSpacing(10)

        desc = QLabel(
            "Disabled axes are never moved and their navigation controls are greyed out.\n"
            "Automations and calibrations that need a disabled or unhomed axis are unavailable."
        )
        desc.setStyleSheet("color: #5f6368; font-size: 11px;")
        vbox.addWidget(desc)

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(6)
        grid.addWidget(QLabel("<b>Enabled</b>"), 0, 1, Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(QLabel("<b>Home</b>"), 0, 2, Qt.AlignmentFlag.AlignCenter)

        for row, axis in enumerate(AXES, start=1):
            label = self._register_label(axis, QLabel(f"{axis.upper()} axis:"))
            grid.addWidget(label, row, 0)

            enabled = QCheckBox()
            enabled.setToolTip(f"Allow the {axis.upper()} axis to be moved.")
            self._checks[f"{axis}_enabled"] = enabled
            grid.addWidget(enabled, row, 1, Qt.AlignmentFlag.AlignCenter)

            home = QCheckBox()
            home.setToolTip(f"Include the {axis.upper()} axis when homing.")
            self._checks[f"home_{axis}"] = home
            grid.addWidget(home, row, 2, Qt.AlignmentFlag.AlignCenter)

        grid.setColumnStretch(3, 1)
        vbox.addLayout(grid)
        # Matches the machine vision menu so Home boxes of disabled axes read as inactive.
        self.setStyleSheet(
            "QCheckBox::indicator:disabled { border: 1px solid #c8cacc; background-color: #e8eaec; }"
        )

        startup = QCheckBox("Home on startup")
        startup.setToolTip(
            "Run the home sequence automatically after connecting to the motion controller.\n"
            "When off, use the Home button before running anything that needs homed axes."
        )
        self._checks["home_on_startup"] = startup
        self._labels["home_on_startup"] = startup
        vbox.addWidget(startup)

    def connect_signals(self, on_check) -> None:
        for key, check in self._checks.items():
            check.checkStateChanged.connect(
                lambda state, k=key: on_check(k, state == Qt.CheckState.Checked)
            )

    def _sync_home_enabled(self) -> None:
        for axis in AXES:
            self._checks[f"home_{axis}"].setEnabled(self._checks[f"{axis}_enabled"].isChecked())

    def populate(self, s: MotionSystemSettings) -> None:
        for key, check in self._checks.items():
            check.blockSignals(True)
            check.setChecked(bool(getattr(s, key)))
            check.blockSignals(False)
        self._sync_home_enabled()

    def snapshot(self, s: MotionSystemSettings) -> None:
        self._saved = {key: bool(getattr(s, key)) for key in self._checks}

    def apply_to_live(self, key: str, value: bool, s: MotionSystemSettings) -> None:
        setattr(s, key, value)
        self._sync_home_enabled()

    def mark_field(self, key: str, value: bool) -> None:
        if key == "home_on_startup":
            self.mark_label(key, self._saved.get(key) != value)
            return
        axis = key[0] if key.endswith("_enabled") else key[-1]
        # Enabled and Home share the axis row label, so it is orange if either differs.
        changed = any(
            self._saved.get(k) != self._checks[k].isChecked()
            for k in (f"{axis}_enabled", f"home_{axis}")
        )
        self.mark_label(axis, changed)

    def has_changes(self) -> bool:
        return any(self._saved.get(key) != check.isChecked() for key, check in self._checks.items())
