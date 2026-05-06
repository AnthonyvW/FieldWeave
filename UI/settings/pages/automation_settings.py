"""
automation_settings.py

Settings page for automation routines.

Design
------
- One QGroupBox per automation routine.  Currently: Tree Core.
- Within Tree Core, scalar fields (mark position, starting height/offset) sit
  in a "Run Settings" sub-group.  A "Slots" sub-group lists each slot with its
  position and offset; slots can be added or removed.
- Modified fields turn orange exactly like NavigationSettingsWidget does.
- get_group_names() returns the top-level group names so SettingsDialog can
  add them as sidebar sub-items.
- Changes are applied to the live settings object on every widget interaction
  and persisted to disk only when Save is clicked.
"""

from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from common.app_context import get_app_context
from common.logger import info, error
from motion.motion_config import MotionSystemSettings, MotionSystemSettingsManager

from UI.settings.pages.automation.general_settings import GeneralSettingsWidget
from UI.settings.pages.automation.z_stack_settings import ZStackSettingsWidget
from UI.settings.pages.automation.tree_core_settings import TreeCoreSettingsWidget
from UI.settings.pages.shared import NM_PER_MM


class AutomationSettingsWidget(QWidget):
    """Full settings page for all automation routines."""

    _GROUP_NAMES = ["General", "Z-Stack Scan", "Tree Core"]

    def __init__(self, parent_dialog=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.parent_dialog = parent_dialog
        self._settings_manager = MotionSystemSettingsManager()
        self._has_unsaved_changes: bool = False
        self._group_boxes: dict[str, QGroupBox] = {}
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
        content.setObjectName("AutomationSettingsContent")
        content.setStyleSheet("QWidget#AutomationSettingsContent { background: white; }")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(10)

        title = QLabel("Automation")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #5f6368;")
        cl.addWidget(title)

        self._general = GeneralSettingsWidget()
        self._general.connect_signals(self._on_general_changed)
        cl.addWidget(self._general)
        self._group_boxes["General"] = self._general

        self._zstack = ZStackSettingsWidget()
        self._zstack.connect_signals(
            self._on_zstack_float,
            self._on_zstack_int,
            self._on_zstack_check,
        )
        cl.addWidget(self._zstack)
        self._group_boxes["Z-Stack Scan"] = self._zstack

        self._tree_core = TreeCoreSettingsWidget()
        self._tree_core.connect_signals(self._on_run_changed, self._on_slot_changed)
        self._tree_core.set_slot_mutation_callback(lambda: self._set_unsaved(True))
        cl.addWidget(self._tree_core)
        self._group_boxes["Tree Core"] = self._tree_core

        if self.parent_dialog and hasattr(self.parent_dialog, "register_group_box"):
            for name, box in self._group_boxes.items():
                self.parent_dialog.register_group_box("Automation", name, box)

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

        if self.parent_dialog and hasattr(self.parent_dialog, "save_btn"):
            self.parent_dialog.save_btn.clicked.connect(self._on_save)

    def _populate_from_settings(self, s: MotionSystemSettings) -> None:
        self._general.populate(s)
        self._zstack.populate(s)
        self._tree_core.populate(s)
        self._general.snapshot(s)
        self._zstack.snapshot(s)
        self._tree_core.snapshot(s)
        self._set_unsaved(False)

    def _on_general_changed(self, key: str, value: object, type_: type) -> None:
        self._general.apply_to_live(key, value, type_)
        self._general.mark_field(key, type_(value))
        self._set_unsaved(True)

    def _on_zstack_float(self, key: str, value: float) -> None:
        nm_keys = {"step_nm", "approach_distance_nm"}
        stored = round(value * NM_PER_MM) if key in nm_keys else value
        self._zstack.apply_float_to_live(key, value)
        self._zstack.mark_field(key, stored)
        self._set_unsaved(True)

    def _on_zstack_int(self, key: str, value: int) -> None:
        self._zstack.apply_int_to_live(key, value)
        self._zstack.mark_field(key, value)
        self._set_unsaved(True)

    def _on_zstack_check(self, key: str, value: int) -> None:
        checked = value != 0
        self._zstack.apply_check_to_live(key, value)
        self._zstack.mark_field(key, checked)
        self._set_unsaved(True)

    def _on_run_changed(self, key: str, value_mm: float) -> None:
        value_nm = round(value_mm * NM_PER_MM)
        self._tree_core.apply_run_to_live(key, value_mm)
        self._tree_core.mark_run_field(key, value_nm)
        self._set_unsaved(True)

    def _on_slot_changed(self, index: int, key: str, value_mm: float) -> None:
        value_nm = round(value_mm * NM_PER_MM)
        self._tree_core.apply_slot_to_live(index, key, value_mm)
        self._tree_core.mark_slot_field(index, key, value_nm)
        self._set_unsaved(True)

    @Slot()
    def _on_save(self) -> None:
        ctx = get_app_context()
        s = self._current_settings()
        self._settings_manager.save(s)
        self._general.snapshot(s)
        self._zstack.snapshot(s)
        self._tree_core.snapshot(s)
        self._general.clear_orange()
        self._zstack.clear_orange()
        self._tree_core.clear_orange()
        self._set_unsaved(False)
        if ctx.toast:
            ctx.toast.success("Automation settings saved", duration=2000)
        info("Automation settings saved")

    def _set_unsaved(self, has_changes: bool) -> None:
        self._has_unsaved_changes = has_changes
        self._save_btn.setEnabled(has_changes)
        if self.parent_dialog:
            if hasattr(self.parent_dialog, "save_btn"):
                self.parent_dialog.save_btn.setEnabled(has_changes)
            if hasattr(self.parent_dialog, "set_category_modified"):
                self.parent_dialog.set_category_modified("Automation", has_changes)

    def has_unsaved_changes(self) -> bool:
        return self._has_unsaved_changes

    def get_group_names(self) -> list[str]:
        return list(self._GROUP_NAMES)


def automation_page(parent_dialog=None) -> QWidget:
    """Create and return the automation settings page widget."""
    return AutomationSettingsWidget(parent_dialog=parent_dialog)