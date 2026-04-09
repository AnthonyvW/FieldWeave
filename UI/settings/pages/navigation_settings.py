"""
navigation_settings.py

Settings page for motion controller / navigation configuration.

Design
------
- Two QGroupBoxes: "Controller" (hardware parameters) and "Navigation"
  (axis inversion toggles and jog-step presets for the navigation widget).
- Modified fields turn orange exactly like MachineVisionSettingsWidget does.
- get_group_names() returns the top-level group names so SettingsDialog can
  add them as sidebar sub-items.
- Changes are applied to the MotionSystemSettings object live on every widget
  interaction and persisted to disk only when Save is clicked.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from common.app_context import get_app_context
from common.logger import info, error
from motion.motion_config import MotionSystemSettings, MotionSystemSettingsManager
from UI.widgets.navigation_widget import NavigationWidget

_ORANGE = "#FFA500"

# Default step-size presets in millimetres.
_DEFAULT_PRESETS_MM = (0.04, 0.4, 2.0, 10.0)

# Nanometres per millimetre conversion constant.
_NM_PER_MM = 1_000_000


def _mm_to_nm(mm: float) -> int:
    return round(mm * _NM_PER_MM)


def _nm_to_mm(nm: int) -> float:
    return nm / _NM_PER_MM


# ---------------------------------------------------------------------------
# Main settings widget
# ---------------------------------------------------------------------------

class NavigationSettingsWidget(QWidget):
    """
    Full settings page for navigation / motion controller configuration.

    Embedded in the application settings dialog.
    """

    _GROUP_NAMES = ["Controller", "Navigation"]

    def __init__(self, parent_dialog=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.parent_dialog = parent_dialog

        self._settings_manager = MotionSystemSettingsManager()

        self._has_unsaved_changes: bool = False
        self._saved_values: dict[str, object] = {}
        self._group_boxes: dict[str, QGroupBox] = {}

        # Widget stores — populated in _build_*
        self._w: dict[str, QWidget] = {}

        self._build_ui()
        self._populate_from_settings(self._current_settings())
        self._connect_signals()

    # ------------------------------------------------------------------
    # Settings helpers
    # ------------------------------------------------------------------

    def _live_settings(self) -> MotionSystemSettings | None:
        """Return the live settings object from the controller, or None."""
        ctx = get_app_context()
        motion = ctx.motion
        if motion is not None and motion.settings is not None:
            return motion.settings
        return None

    def _current_settings(self) -> MotionSystemSettings:
        """Return live settings from the manager if available, else defaults."""
        s = self._live_settings()
        return s if s is not None else MotionSystemSettings()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

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

        controller_group = self._build_controller_group()
        cl.addWidget(controller_group)
        self._group_boxes["Controller"] = controller_group

        navigation_group = self._build_navigation_group()
        cl.addWidget(navigation_group)
        self._group_boxes["Navigation"] = navigation_group

        # Register group boxes with parent dialog for scroll-to support.
        if self.parent_dialog and hasattr(self.parent_dialog, "register_group_box"):
            self.parent_dialog.register_group_box("Navigation", "Controller", controller_group)
            self.parent_dialog.register_group_box("Navigation", "Navigation", navigation_group)

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

    def _build_controller_group(self) -> QGroupBox:
        """Hardware-level controller parameters."""
        group = QGroupBox("Controller")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # Baud rate
        baud_spin = QSpinBox()
        baud_spin.setMinimum(1_200)
        baud_spin.setMaximum(3_000_000)
        baud_spin.setSingleStep(9_600)
        baud_spin.setFixedWidth(110)
        baud_spin.setToolTip("Serial baud rate for communication with the motion controller.")
        form.addRow("Baud rate:", baud_spin)
        self._w["baud_rate"] = baud_spin

        # Max X
        max_x_spin = QSpinBox()
        max_x_spin.setMinimum(1)
        max_x_spin.setMaximum(10_000)
        max_x_spin.setSuffix(" mm")
        max_x_spin.setFixedWidth(110)
        max_x_spin.setToolTip("Maximum travel distance of the X axis in millimetres.")
        form.addRow("Max X:", max_x_spin)
        self._w["max_x"] = max_x_spin

        # Max Y
        max_y_spin = QSpinBox()
        max_y_spin.setMinimum(1)
        max_y_spin.setMaximum(10_000)
        max_y_spin.setSuffix(" mm")
        max_y_spin.setFixedWidth(110)
        max_y_spin.setToolTip("Maximum travel distance of the Y axis in millimetres.")
        form.addRow("Max Y:", max_y_spin)
        self._w["max_y"] = max_y_spin

        # Max Z
        max_z_spin = QSpinBox()
        max_z_spin.setMinimum(1)
        max_z_spin.setMaximum(10_000)
        max_z_spin.setSuffix(" mm")
        max_z_spin.setFixedWidth(110)
        max_z_spin.setToolTip("Maximum travel distance of the Z axis in millimetres.")
        form.addRow("Max Z:", max_z_spin)
        self._w["max_z"] = max_z_spin

        # Step size (displayed in mm, stored as nm)
        step_spin = QDoubleSpinBox()
        step_spin.setMinimum(0.001)
        step_spin.setMaximum(100.0)
        step_spin.setDecimals(4)
        step_spin.setSingleStep(0.01)
        step_spin.setSuffix(" mm")
        step_spin.setFixedWidth(130)
        step_spin.setToolTip(
            "Minimum hardware step size (motion controller resolution) in millimetres.\n"
            "This is the smallest distance the controller can reliably move."
        )
        form.addRow("Step size:", step_spin)
        self._w["step_size"] = step_spin

        return group

    def _build_navigation_group(self) -> QGroupBox:
        """Navigation widget behaviour: axis inversion and jog-step presets."""
        group = QGroupBox("Navigation")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(12)

        # ---- Axis inversion ----
        invert_box = QGroupBox("Axis Inversion")
        invert_form = QFormLayout(invert_box)
        invert_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        for axis in ("x", "y", "z"):
            check = QCheckBox()
            check.setToolTip(
                f"Invert the {axis.upper()} axis direction in the navigation widget.\n"
                "Enable if the on-screen arrow moves the stage in the wrong direction."
            )
            invert_form.addRow(f"Invert {axis.upper()}:", check)
            self._w[f"invert_{axis}"] = check

        vbox.addWidget(invert_box)

        # ---- Jog-step presets ----
        presets_box = QGroupBox("Jog-Step Presets")
        presets_form = QFormLayout(presets_box)
        presets_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        presets_label = QLabel(
            "Four step-size buttons shown in the navigation widget.\n"
            "Values are in millimetres."
        )
        presets_label.setStyleSheet("color: #5f6368; font-size: 11px;")
        presets_form.addRow(presets_label)

        for i in range(1, 5):
            preset_spin = QDoubleSpinBox()
            preset_spin.setMinimum(0.001)
            preset_spin.setMaximum(500.0)
            preset_spin.setDecimals(4)
            preset_spin.setSingleStep(0.01)
            preset_spin.setSuffix(" mm")
            preset_spin.setFixedWidth(130)
            preset_spin.setToolTip(f"Step-size preset {i} for the navigation widget jog buttons.")
            presets_form.addRow(f"Preset {i}:", preset_spin)
            self._w[f"preset_{i}"] = preset_spin

        vbox.addWidget(presets_box)

        return group

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        w = self._w

        w["baud_rate"].valueChanged.connect(
            lambda v: self._on_field_changed("baud_rate", v)
        )
        w["max_x"].valueChanged.connect(
            lambda v: self._on_field_changed("max_x", v)
        )
        w["max_y"].valueChanged.connect(
            lambda v: self._on_field_changed("max_y", v)
        )
        w["max_z"].valueChanged.connect(
            lambda v: self._on_field_changed("max_z", v)
        )
        w["step_size"].valueChanged.connect(
            lambda v: self._on_field_changed("step_size", v)
        )

        for axis in ("x", "y", "z"):
            key = f"invert_{axis}"
            w[key].checkStateChanged.connect(
                lambda state, k=key: self._on_field_changed(k, state == Qt.CheckState.Checked)
            )

        for i in range(1, 5):
            key = f"preset_{i}"
            w[key].valueChanged.connect(
                lambda v, k=key: self._on_field_changed(k, v)
            )

    # ------------------------------------------------------------------
    # Populate from settings
    # ------------------------------------------------------------------

    def _populate_from_settings(self, s: MotionSystemSettings) -> None:
        """Push all values into widgets without triggering saves."""
        self._block_all_signals(True)
        try:
            self._w["baud_rate"].setValue(s.baud_rate)
            self._w["max_x"].setValue(s.max_x)
            self._w["max_y"].setValue(s.max_y)
            self._w["max_z"].setValue(s.max_z)
            self._w["step_size"].setValue(_nm_to_mm(s.step_size))

            for axis in ("x", "y", "z"):
                self._w[f"invert_{axis}"].setChecked(
                    getattr(s, f"invert_{axis}", False)
                )

            presets_nm: list[int] = getattr(
                s,
                "step_presets",
                [_mm_to_nm(mm) for mm in _DEFAULT_PRESETS_MM],
            )
            # Pad / truncate to exactly 4 entries.
            defaults_nm = [_mm_to_nm(mm) for mm in _DEFAULT_PRESETS_MM]
            presets_nm = (presets_nm + defaults_nm)[:4]
            for i, nm in enumerate(presets_nm, start=1):
                self._w[f"preset_{i}"].setValue(_nm_to_mm(nm))

        finally:
            self._block_all_signals(False)

        self._snapshot_saved_values(s)
        self._set_unsaved(False)

    # ------------------------------------------------------------------
    # Saved-value snapshot and orange tracking
    # ------------------------------------------------------------------

    def _snapshot_saved_values(self, s: MotionSystemSettings) -> None:
        presets_nm: list[int] = getattr(
            s,
            "step_presets",
            [_mm_to_nm(mm) for mm in _DEFAULT_PRESETS_MM],
        )
        defaults_nm = [_mm_to_nm(mm) for mm in _DEFAULT_PRESETS_MM]
        presets_nm = (presets_nm + defaults_nm)[:4]

        self._saved_values = {
            "baud_rate": s.baud_rate,
            "max_x": s.max_x,
            "max_y": s.max_y,
            "max_z": s.max_z,
            "step_size": _nm_to_mm(s.step_size),
            "invert_x": getattr(s, "invert_x", False),
            "invert_y": getattr(s, "invert_y", False),
            "invert_z": getattr(s, "invert_z", False),
            "preset_1": _nm_to_mm(presets_nm[0]),
            "preset_2": _nm_to_mm(presets_nm[1]),
            "preset_3": _nm_to_mm(presets_nm[2]),
            "preset_4": _nm_to_mm(presets_nm[3]),
        }

    def _check_modified(self, key: str, current_value: object) -> bool:
        saved = self._saved_values.get(key)
        if isinstance(saved, float) and isinstance(current_value, float):
            return abs(saved - current_value) > 1e-9
        return saved != current_value

    def _apply_orange(self, widget: QWidget, orange: bool) -> None:
        color = _ORANGE if orange else ""
        if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
            widget.setStyleSheet(f"color: {color};" if orange else "")
        elif isinstance(widget, QCheckBox):
            widget.setStyleSheet(f"QCheckBox {{ color: {color}; }}" if orange else "")

    # ------------------------------------------------------------------
    # Change handler
    # ------------------------------------------------------------------

    def _on_field_changed(self, key: str, value: object) -> None:
        """Apply the changed field directly to the live settings object."""
        s = self._live_settings()
        if s is None:
            # No controller available yet — nothing to mutate, but still mark
            # the field as modified so Save is enabled and orange is shown.
            orange = self._check_modified(key, value)
            w = self._w.get(key)
            if w:
                self._apply_orange(w, orange)
            self._set_unsaved(True)
            return

        if key == "baud_rate":
            s.baud_rate = int(value)  # type: ignore[arg-type]
        elif key == "max_x":
            s.max_x = int(value)  # type: ignore[arg-type]
        elif key == "max_y":
            s.max_y = int(value)  # type: ignore[arg-type]
        elif key == "max_z":
            s.max_z = int(value)  # type: ignore[arg-type]
        elif key == "step_size":
            s.step_size = _mm_to_nm(float(value))  # type: ignore[arg-type]
        elif key in ("invert_x", "invert_y", "invert_z"):
            setattr(s, key, value)
        elif key.startswith("preset_"):
            idx = int(key[-1]) - 1
            presets: list[int] = list(
                getattr(s, "step_presets", [_mm_to_nm(mm) for mm in _DEFAULT_PRESETS_MM])
            )
            defaults_nm = [_mm_to_nm(mm) for mm in _DEFAULT_PRESETS_MM]
            presets = (presets + defaults_nm)[:4]
            presets[idx] = _mm_to_nm(float(value))  # type: ignore[arg-type]
            s.step_presets = presets  # type: ignore[attr-defined]

        orange = self._check_modified(key, value)
        w = self._w.get(key)
        if w:
            self._apply_orange(w, orange)

        self._set_unsaved(True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    @Slot()
    def _on_save(self) -> None:
        ctx = get_app_context()
        try:
            s = self._current_settings()
            self._settings_manager.save(s)
            self._snapshot_saved_values(s)
            self._clear_all_orange()
            self._set_unsaved(False)
            NavigationWidget.notify_settings_changed()
            if ctx.toast:
                ctx.toast.success("Navigation settings saved", duration=2000)
            info("Navigation settings saved")
        except Exception as exc:
            error(f"Failed to save navigation settings: {exc}")
            if ctx.toast:
                ctx.toast.error(f"Save failed: {exc}", duration=3000)

    def _clear_all_orange(self) -> None:
        for w in self._w.values():
            self._apply_orange(w, False)

    # ------------------------------------------------------------------
    # Unsaved state
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Sidebar sub-item support
    # ------------------------------------------------------------------

    def get_group_names(self) -> list[str]:
        """Return group names for SettingsDialog sidebar sub-items."""
        return list(self._GROUP_NAMES)

    # ------------------------------------------------------------------
    # Signal blocking helpers
    # ------------------------------------------------------------------

    def _block_all_signals(self, block: bool) -> None:
        for w in self._w.values():
            w.blockSignals(block)


def navigation_page(parent_dialog=None) -> QWidget:
    """Create and return the navigation settings page widget."""
    return NavigationSettingsWidget(parent_dialog=parent_dialog)
