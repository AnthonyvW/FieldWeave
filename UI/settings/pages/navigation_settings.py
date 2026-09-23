"""
navigation_settings.py

Settings page for motion controller / navigation configuration.

Design
------
- Three QGroupBoxes: "Controller" (hardware parameters), "Axes" (enable,
  homing and inversion per axis) and "Navigation" (jog-step presets and
  starting height for the navigation widget).
- Modified fields turn orange exactly like AutomationSettingsWidget does.
- get_group_names() returns the top-level group names so SettingsDialog can
  add them as sidebar sub-items.
- Changes are applied to the live settings object on every widget interaction
  and persisted to disk only when Save is clicked.
"""

from __future__ import annotations

import threading

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
from common.logger import error, info
from motion.motion_config import MotionSystemSettings, MotionSystemSettingsManager

from UI.settings.pages.navigation.axes_settings import AxesSettingsWidget
from UI.settings.pages.navigation.controller_settings import ControllerSettingsWidget
from UI.settings.pages.navigation.navigation_group_settings import NavigationGroupSettingsWidget


class NavigationSettingsWidget(QWidget):
    """Full settings page for navigation / motion controller configuration."""

    _GROUP_NAMES = ["Controller", "Axes", "Navigation"]

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
        self._controller.connect_signals(self._on_controller_changed, self._on_connect_port)
        cl.addWidget(self._controller)

        self._axes = AxesSettingsWidget()
        self._axes.connect_signals(self._on_axes_check)
        cl.addWidget(self._axes)

        self._navigation = NavigationGroupSettingsWidget()
        self._navigation.connect_signals(
            self._on_nav_float,
            self._on_set_current_height,
            self._on_reset_height,
        )
        cl.addWidget(self._navigation)

        if self.parent_dialog and hasattr(self.parent_dialog, "register_group_box"):
            self.parent_dialog.register_group_box("Navigation", "Controller", self._controller)
            self.parent_dialog.register_group_box("Navigation", "Axes", self._axes)
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
        self._axes.populate(s)
        self._navigation.populate(s)
        self._controller.snapshot(s)
        self._axes.snapshot(s)
        self._navigation.snapshot(s)
        self._set_unsaved(False)

    def _on_controller_changed(self, key: str, value: object) -> None:
        s = self._live_settings()
        if s is not None:
            self._controller.apply_to_live(key, value, s)
        self._controller.mark_field(key, value)
        self._recheck_unsaved()

    def _on_axes_check(self, key: str, value: bool) -> None:
        s = self._live_settings()
        if s is not None:
            self._axes.apply_to_live(key, value, s)
        self._axes.mark_field(key, value)
        self._recheck_unsaved()

    def _on_nav_float(self, key: str, value: float) -> None:
        s = self._live_settings()
        if s is not None:
            self._navigation.apply_float_to_live(key, value, s)
        self._navigation.mark_float_field(key, value)
        self._recheck_unsaved()

    @Slot()
    def _on_set_current_height(self) -> None:
        self._navigation.set_height_from_current_position()

    @Slot()
    def _on_reset_height(self) -> None:
        self._navigation.reset_height()

    def _recheck_unsaved(self) -> None:
        controller_changed = self._controller.has_changes()
        axes_changed = self._axes.has_changes()
        nav_changed = self._navigation.has_changes()
        has_changes = controller_changed or axes_changed or nav_changed

        if self.parent_dialog and hasattr(self.parent_dialog, "set_category_modified"):
            self.parent_dialog.set_category_modified("Navigation", controller_changed, "Controller")
            self.parent_dialog.set_category_modified("Navigation", axes_changed, "Axes")
            self.parent_dialog.set_category_modified("Navigation", nav_changed, "Navigation")

        self._set_unsaved(has_changes)

    @Slot()
    def _on_save(self) -> None:
        ctx = get_app_context()
        s = self._current_settings()
        self._settings_manager.save(s)
        reconnect = self._controller.connection_changed(s)
        self._controller.snapshot(s)
        self._axes.snapshot(s)
        self._navigation.snapshot(s)
        self._controller.clear_orange()
        self._axes.clear_orange()
        self._navigation.clear_orange()
        self._recheck_unsaved()
        self._set_unsaved(False)
        ctx.toast.success("Navigation settings saved", duration=2000)
        info("Navigation settings saved")
        if reconnect:
            self._reconnect_motion()

    @Slot()
    def _on_connect_port(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        s = self._live_settings()
        if motion is None or s is None:
            return
        if motion.routine_running:
            ctx.toast.warning("Stop the running routine before changing the motion controller port.")
            return

        port = self._controller.selected_port()
        s.com_port = port
        # Persist only the port so other unsaved edits on this page stay pending.
        try:
            on_disk = self._settings_manager.load()
            on_disk.com_port = port
            saved = self._settings_manager.save(on_disk)
        except Exception as exc:
            error(f"Failed to save motion controller port: {exc}")
            saved = False
        if saved:
            self._controller.mark_port_saved(port)
        else:
            ctx.toast.warning("Could not save the selected port; it will only be used until FieldWeave restarts.")
        self._recheck_unsaved()

        self._reconnect_motion()

    def _reconnect_motion(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None:
            return
        if motion.routine_running:
            ctx.toast.warning(
                "A routine is running. The new connection settings will be used after FieldWeave restarts.",
            )
            return
        port = motion.settings.com_port if motion.settings is not None else ""
        ctx.toast.info(f"Connecting to {port or 'motion controller (auto-detect)'}...", duration=2000)
        info("Connection settings changed, restarting motion controller")
        threading.Thread(target=motion.restart, daemon=True, name="MotionRestart").start()

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