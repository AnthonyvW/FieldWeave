"""
navigation_settings.py

Settings page for motion controller / navigation configuration.

Design
------
- Two QGroupBoxes: "Controller" (hardware parameters) and "Navigation"
  (axis inversion toggles and jog-step presets for the navigation widget).
- Modified fields turn orange exactly like AutomationSettingsWidget does.
- get_group_names() returns the top-level group names so SettingsDialog can
  add them as sidebar sub-items.
- Changes are applied to the live settings object on every widget interaction
  and persisted to disk only when Save is clicked.
"""

from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from common.app_context import get_app_context
from common.logger import info
from motion.motion_config import MotionSystemSettings, MotionSystemSettingsManager
from UI.widgets.navigation_widget import NavigationWidget

from UI.settings.pages.navigation.controller_settings import ControllerSettingsWidget
from UI.settings.pages.navigation.navigation_group_settings import NavigationGroupSettingsWidget


class NavigationSettingsWidget(QWidget):
    """Full settings page for navigation / motion controller configuration."""

    _GROUP_NAMES = ["Controller", "Navigation"]

    def __init__(self, parent_dialog=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.parent_dialog = parent_dialog
        self._settings_manager = MotionSystemSettingsManager()
        self._has_unsaved_changes: bool = False
        self._build_ui()
        self._populate_from_settings(self._current_settings())

    def _live_settings(self) -> MotionSystemSettings | None:
        motion = get_app_context().motion
        if motion is not None and motion.settings is not None:
            return motion.settings
        return None

    def _current_settings(self) -> MotionSystemSettings:
        s = self._live_settings()
        return s if s is not None else MotionSystemSettings()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        content.setObjectName("NavigationSettingsContent")
        content.setStyleSheet("QWidget#NavigationSettingsContent { background: white; }")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(10)

        title = QLabel("Navigation")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #5f6368;")
        cl.addWidget(title)

        self._controller = ControllerSettingsWidget()
        self._controller.connect_signals(self._on_controller_changed)
        cl.addWidget(self._controller)

        self._navigation = NavigationGroupSettingsWidget()
        self._navigation.connect_signals(
            self._on_nav_float,
            self._on_nav_check,
            self._on_set_current_height,
            self._on_reset_height,
        )
        cl.addWidget(self._navigation)

        if self.parent_dialog and hasattr(self.parent_dialog, "register_group_box"):
            self.parent_dialog.register_group_box("Navigation", "Controller", self._controller)
            self.parent_dialog.register_group_box("Navigation", "Navigation", self._navigation)

        cl.addStretch()

        btn_row = QHBoxLayout()
        self._save_btn = QPushButton("Save")
        self._save_btn.setEnabled(False)
        self._save_btn.setMaximumWidth(100)
        self._save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self._save_btn)
        btn_row.addStretch()
        cl.addLayout(btn_row)

        scroll.setWidget(content)
        root.addWidget(scroll)

    def _populate_from_settings(self, s: MotionSystemSettings) -> None:
        self._controller.populate(s)
        self._navigation.populate(s)
        self._controller.snapshot(s)
        self._navigation.snapshot(s)
        self._set_unsaved(False)

    def _on_controller_changed(self, key: str, value: object) -> None:
        s = self._live_settings()
        if s is not None:
            self._controller.apply_to_live(key, value, s)
        current = self._controller._w[key].value()
        self._controller.mark_field(key, current)
        self._recheck_unsaved()

    def _on_nav_float(self, key: str, value: float) -> None:
        s = self._live_settings()
        if s is not None:
            self._navigation.apply_float_to_live(key, value, s)
        self._navigation.mark_float_field(key, value)
        self._recheck_unsaved()

    def _on_nav_check(self, key: str, value: bool) -> None:
        s = self._live_settings()
        if s is not None:
            self._navigation.apply_check_to_live(key, value, s)
        self._navigation.mark_check_field(key, value)
        self._recheck_unsaved()

    @Slot()
    def _on_set_current_height(self) -> None:
        self._navigation.set_height_from_current_position()

    @Slot()
    def _on_reset_height(self) -> None:
        self._navigation.reset_height()

    def _recheck_unsaved(self) -> None:
        controller_changed = self._controller.has_changes()
        nav_changed = self._navigation.has_changes()
        has_changes = controller_changed or nav_changed

        if self.parent_dialog and hasattr(self.parent_dialog, "set_category_modified"):
            self.parent_dialog.set_category_modified("Navigation", controller_changed, "Controller")
            self.parent_dialog.set_category_modified("Navigation", nav_changed, "Navigation")

        self._set_unsaved(has_changes)

    @Slot()
    def _on_save(self) -> None:
        ctx = get_app_context()
        s = self._current_settings()
        self._settings_manager.save(s)
        self._controller.snapshot(s)
        self._navigation.snapshot(s)
        self._controller.clear_orange()
        self._navigation.clear_orange()
        self._recheck_unsaved()
        self._set_unsaved(False)
        NavigationWidget.notify_settings_changed()
        ctx.toast.success("Navigation settings saved", duration=2000)
        info("Navigation settings saved")

    def _set_unsaved(self, has_changes: bool) -> None:
        self._has_unsaved_changes = has_changes
        self._save_btn.setEnabled(has_changes)
        if self.parent_dialog:
            if hasattr(self.parent_dialog, "save_btn"):
                self.parent_dialog.save_btn.setEnabled(has_changes)
            if hasattr(self.parent_dialog, "set_category_modified"):
                self.parent_dialog.set_category_modified("Navigation", has_changes)

    def has_unsaved_changes(self) -> bool:
        return self._has_unsaved_changes

    def get_group_names(self) -> list[str]:
        return list(self._GROUP_NAMES)


def navigation_page(parent_dialog=None) -> QWidget:
    """Create and return the navigation settings page widget."""
    return NavigationSettingsWidget(parent_dialog=parent_dialog)