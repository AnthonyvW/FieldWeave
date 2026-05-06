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

import math

from PySide6.QtCore import Slot
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
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from common.app_context import get_app_context
from common.logger import info, error
from motion.motion_config import MotionSystemSettings, MotionSystemSettingsManager, TreeCoreSlot

_ORANGE = "#FFA500"
_NM_PER_MM = 1_000_000


class _NoScrollDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that ignores scroll-wheel events to prevent accidental edits."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class _NoScrollSpinBox(QSpinBox):
    """QSpinBox that ignores scroll-wheel events to prevent accidental edits."""

    def wheelEvent(self, event) -> None:
        event.ignore()


# ---------------------------------------------------------------------------
# Per-slot row widget
# ---------------------------------------------------------------------------

class _SlotRow(QWidget):
    """A single row in the slots list showing position and offset spinboxes."""

    def __init__(
        self,
        index: int,
        slot: TreeCoreSlot,
        get_axis_fn,
        get_motion_fn,
        axis_max_mm: float,
    ) -> None:
        super().__init__()
        self._index = index
        self._get_axis = get_axis_fn
        self._get_motion = get_motion_fn
        self._axis_max_mm = axis_max_mm
        self._w: dict[str, QDoubleSpinBox] = {}
        self._build(slot)

    def _build(self, slot: TreeCoreSlot) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        label = QLabel(f"Slot {self._index + 1}:")
        label.setFixedWidth(50)
        row.addWidget(label)

        for key, value_nm, pos_label, tooltip in (
            ("position_nm", slot.position_nm, "Pos (mm):", "Position along the automation axis (mm)."),
            ("offset_nm",   slot.offset_nm,   "Offset (mm):", "Fine-tune offset applied on top of the slot position (mm)."),
        ):
            lbl = QLabel(pos_label)
            lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            row.addWidget(lbl)

            spin = _NoScrollDoubleSpinBox()
            spin.setMinimum(0.0)
            spin.setMaximum(self._axis_max_mm)
            spin.setSingleStep(0.001)
            spin.setDecimals(3)
            spin.setValue(value_nm / _NM_PER_MM)
            spin.setFixedWidth(110)
            spin.setToolTip(tooltip)
            row.addWidget(spin)
            self._w[key] = spin

            set_btn = QPushButton("Set")
            set_btn.setFixedWidth(38)
            set_btn.setToolTip(f"Set to current printer position ({tooltip})")
            set_btn.clicked.connect(lambda checked=False, k=key: self._on_set(k))
            row.addWidget(set_btn)

            if key == "position_nm":
                goto_btn = QPushButton("Go to")
                goto_btn.setFixedWidth(50)
                goto_btn.setToolTip(f"Move printer to this value ({tooltip})")
                goto_btn.clicked.connect(lambda checked=False, k=key: self._on_goto(k))
                row.addWidget(goto_btn)

        row.addStretch()

    def _on_set(self, key: str) -> None:
        motion = self._get_motion()
        if motion is None or not motion.is_ready():
            return
        try:
            pos = motion.get_position()
        except Exception as exc:
            error(f"SlotRow: get_position failed — {exc}")
            return
        axis = self._get_axis()
        current_nm = pos.y if axis == "x" else pos.x
        if key == "offset_nm":
            position_nm = round(self._w["position_nm"].value() * _NM_PER_MM)
            value_nm = current_nm - position_nm
        else:
            value_nm = current_nm
        spin = self._w[key]
        spin.blockSignals(True)
        spin.setValue(value_nm / _NM_PER_MM)
        spin.blockSignals(False)
        spin.valueChanged.emit(spin.value())

    def _on_goto(self, key: str) -> None:
        motion = self._get_motion()
        if motion is None or not motion.is_ready():
            return
        from motion.models import Position
        try:
            current = motion.get_position()
        except Exception as exc:
            error(f"SlotRow: get_position failed — {exc}")
            return
        value_nm = round(self._w[key].value() * _NM_PER_MM)
        axis = self._get_axis()
        if axis == "x":
            target = Position(x=current.x, y=value_nm, z=current.z)
        else:
            target = Position(x=value_nm, y=current.y, z=current.z)
        try:
            motion.move_to_position(target, wait=False)
        except Exception as exc:
            error(f"SlotRow: move_to_position failed — {exc}")

    @property
    def widgets(self) -> dict[str, QDoubleSpinBox]:
        return self._w

    @property
    def index(self) -> int:
        return self._index

    def to_slot(self) -> TreeCoreSlot:
        return TreeCoreSlot(
            position_nm=round(self._w["position_nm"].value() * _NM_PER_MM),
            offset_nm=round(self._w["offset_nm"].value() * _NM_PER_MM),
        )


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class AutomationSettingsWidget(QWidget):
    """Full settings page for all automation routines."""

    _GROUP_NAMES = ["General", "Z-Stack Scan", "Tree Core"]

    def __init__(self, parent_dialog=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.parent_dialog = parent_dialog

        self._settings_manager = MotionSystemSettingsManager()

        self._has_unsaved_changes: bool = False
        self._saved_values: dict[str, object] = {}
        self._group_boxes: dict[str, QGroupBox] = {}

        self._w_general: dict[str, QDoubleSpinBox] = {}
        self._w_general_int: dict[str, QSpinBox] = {}
        self._w_zstack: dict[str, QDoubleSpinBox] = {}
        self._w_zstack_int: dict[str, QSpinBox] = {}
        self._w_zstack_check: dict[str, QCheckBox] = {}

        # Slot row widgets — rebuilt whenever slots are added/removed.
        self._slot_rows: list[_SlotRow] = []

        self._build_ui()
        self._populate_from_settings(self._current_settings())
        self._connect_run_signals()

    # ------------------------------------------------------------------
    # Settings helpers
    # ------------------------------------------------------------------

    def _live_settings(self) -> MotionSystemSettings | None:
        """Return the live settings object from the controller, or None."""
        motion = get_app_context().motion
        if motion is not None and motion.settings is not None:
            return motion.settings
        return None

    def _current_settings(self) -> MotionSystemSettings:
        """Return live settings if available, else defaults."""
        s = self._live_settings()
        return s if s is not None else MotionSystemSettings()

    def _get_axis(self) -> str:
        s = self._live_settings()
        if s is not None:
            return s.tree_core_automation.axis.lower()
        return "x"

    def _axis_max_mm(self, axis: str) -> float:
        s = self._live_settings()
        if s is None:
            return 300.0
        if axis == "x":
            return float(s.max_x)
        if axis == "y":
            return float(s.max_y)
        return float(s.max_z)

    def _get_motion(self):
        return get_app_context().motion

    def _get_printer_step_mm(self) -> float:
        """Return the printer's minimum step size in mm from settings, defaulting to 0.04 mm."""
        motion = get_app_context().motion
        if motion is not None and motion.settings is not None:
            return motion.settings.step_size / 1_000_000.0
        return 0.04

    def _on_run_set(self, key: str, coord: str) -> None:
        motion = self._get_motion()
        if motion is None or not motion.is_ready():
            return
        try:
            pos = motion.get_position()
        except Exception as exc:
            error(f"AutomationSettings: get_position failed — {exc}")
            return
        if coord == "z":
            value_nm = pos.z
        else:
            axis = self._get_axis()
            value_nm = pos.x if axis == "x" else pos.y
        spin = self._w_run[key]
        spin.blockSignals(True)
        spin.setValue(value_nm / _NM_PER_MM)
        spin.blockSignals(False)
        spin.valueChanged.emit(spin.value())

    def _on_run_goto(self, key: str, coord: str) -> None:
        motion = self._get_motion()
        if motion is None or not motion.is_ready():
            return
        from motion.models import Position
        try:
            current = motion.get_position()
        except Exception as exc:
            error(f"AutomationSettings: get_position failed — {exc}")
            return
        value_nm = round(self._w_run[key].value() * _NM_PER_MM)
        if coord == "z":
            target = Position(x=current.x, y=current.y, z=value_nm)
        else:
            axis = self._get_axis()
            if axis == "x":
                target = Position(x=value_nm, y=current.y, z=current.z)
            else:
                target = Position(x=current.x, y=value_nm, z=current.z)
        try:
            motion.move_to_position(target, wait=False)
        except Exception as exc:
            error(f"AutomationSettings: move_to_position failed — {exc}")

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
        content.setObjectName("AutomationSettingsContent")
        content.setStyleSheet("QWidget#AutomationSettingsContent { background: white; }")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(10)

        title = QLabel("Automation")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #5f6368;")
        cl.addWidget(title)

        general_group = self._build_general_group()
        cl.addWidget(general_group)
        self._group_boxes["General"] = general_group

        if self.parent_dialog and hasattr(self.parent_dialog, "register_group_box"):
            self.parent_dialog.register_group_box("Automation", "General", general_group)

        z_stack_group = self._build_z_stack_group()
        cl.addWidget(z_stack_group)
        self._group_boxes["Z-Stack Scan"] = z_stack_group

        if self.parent_dialog and hasattr(self.parent_dialog, "register_group_box"):
            self.parent_dialog.register_group_box("Automation", "Z-Stack Scan", z_stack_group)

        tree_core_group = self._build_tree_core_group()
        cl.addWidget(tree_core_group)
        self._group_boxes["Tree Core"] = tree_core_group

        if self.parent_dialog and hasattr(self.parent_dialog, "register_group_box"):
            self.parent_dialog.register_group_box("Automation", "Tree Core", tree_core_group)

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

    def _build_general_group(self) -> QGroupBox:
        group = QGroupBox("General")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        for key, label_text, tooltip in (
            ("overlap_x_pct", "X overlap (%):", "Fraction of each frame that overlaps the next along X (0–100)."),
            ("overlap_y_pct", "Y overlap (%):", "Fraction of each frame that overlaps the next along Y (0–100)."),
        ):
            spin = _NoScrollDoubleSpinBox()
            spin.setMinimum(0.0)
            spin.setMaximum(100.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(1)
            spin.setFixedWidth(130)
            spin.setToolTip(tooltip)
            self._w_general[key] = spin
            form.addRow(QLabel(label_text), spin)

        timeout_spin = _NoScrollSpinBox()
        timeout_spin.setMinimum(100)
        timeout_spin.setMaximum(60_000)
        timeout_spin.setSingleStep(100)
        timeout_spin.setSuffix(" ms")
        timeout_spin.setFixedWidth(130)
        timeout_spin.setToolTip("How long to wait for each image capture to complete before treating it as a failure.")
        self._w_general_int["capture_timeout_ms"] = timeout_spin
        form.addRow(QLabel("Capture timeout:"), timeout_spin)

        return group

    def _build_z_stack_group(self) -> QGroupBox:
        group = QGroupBox("Z-Stack Scan")
        vbox = QVBoxLayout(group)

        note = QLabel("These defaults will take effect the next time the program is launched.")
        note.setWordWrap(True)
        vbox.addWidget(note)

        # -- Scan sub-group --------------------------------------------------
        scan_box = QGroupBox("Scan")
        scan_form = QFormLayout(scan_box)
        scan_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        step_spin = _NoScrollDoubleSpinBox()
        printer_step_mm = self._get_printer_step_mm()
        step_spin.setMinimum(printer_step_mm)
        step_spin.setMaximum(10.0)
        step_spin.setSingleStep(printer_step_mm)
        step_spin.setDecimals(max(2, -int(math.floor(math.log10(printer_step_mm)))))
        step_spin.setSuffix(" mm")
        step_spin.setFixedWidth(130)
        step_spin.setToolTip("Distance between capture positions.")
        self._w_zstack["step_nm"] = step_spin
        scan_form.addRow(QLabel("Step size:"), step_spin)

        approach_spin = _NoScrollDoubleSpinBox()
        approach_spin.setMinimum(0.0)
        approach_spin.setMaximum(10.0)
        approach_spin.setSingleStep(0.1)
        approach_spin.setDecimals(3)
        approach_spin.setSuffix(" mm")
        approach_spin.setFixedWidth(130)
        approach_spin.setToolTip(
            "Before starting the scan, the stage overshoots the near end by this "
            "distance then returns to it, eliminating backlash. 0 disables."
        )
        self._w_zstack["approach_distance_nm"] = approach_spin
        scan_form.addRow(QLabel("Approach distance:"), approach_spin)

        vbox.addWidget(scan_box)

        # -- Focus stack sub-group -------------------------------------------
        fs_box = QGroupBox("Focus Stack")
        fs_form = QFormLayout(fs_box)
        fs_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        for key, label_text, tooltip in (
            ("run_focus_stack", "Run after capture:", "Automatically run focus stacking after all frames are captured."),
            ("keep_size",       "Keep original size:", "Keep the output image the same size as the inputs. Warps are applied in-place rather than expanding the canvas."),
            ("no_align",        "Skip alignment:", "Skip ECC alignment. Use when images are already registered."),
            ("crop",            "Crop to intersection:", "Crop output to the largest rectangle covered by every frame after alignment."),
            ("cull_enabled",    "Cull out-of-focus frames:", "Discard frames whose focus score falls below the threshold fraction of the sharpest frame."),
            ("slab_enabled",    "Enable slabbing:", "Split the image set into overlapping sub-stacks, stack each independently, then fuse. Reduces peak RAM."),
        ):
            check = QCheckBox()
            check.setToolTip(tooltip)
            self._w_zstack_check[key] = check
            fs_form.addRow(QLabel(label_text), check)

        sharpness_spin = _NoScrollDoubleSpinBox()
        sharpness_spin.setMinimum(1.0)
        sharpness_spin.setMaximum(8.0)
        sharpness_spin.setSingleStep(0.5)
        sharpness_spin.setDecimals(1)
        sharpness_spin.setFixedWidth(130)
        sharpness_spin.setToolTip("Weight sharpness exponent. Higher values favour the sharpest pixel more aggressively. Useful range: 1.0 (soft) to 8.0 (near-hard).")
        self._w_zstack["sharpness"] = sharpness_spin
        fs_form.addRow(QLabel("Sharpness:"), sharpness_spin)

        cull_spin = _NoScrollDoubleSpinBox()
        cull_spin.setMinimum(0.0)
        cull_spin.setMaximum(1.0)
        cull_spin.setSingleStep(0.05)
        cull_spin.setDecimals(2)
        cull_spin.setFixedWidth(130)
        cull_spin.setToolTip("Frames scoring below this fraction of the peak score are culled. Raise toward 1.0 to cull more aggressively.")
        self._w_zstack["cull_threshold"] = cull_spin
        fs_form.addRow(QLabel("Cull threshold:"), cull_spin)

        slab_size_spin = _NoScrollSpinBox()
        slab_size_spin.setMinimum(2)
        slab_size_spin.setMaximum(500)
        slab_size_spin.setFixedWidth(130)
        slab_size_spin.setToolTip("Number of images per sub-stack.")
        self._w_zstack_int["slab_size"] = slab_size_spin
        fs_form.addRow(QLabel("Slab size:"), slab_size_spin)

        slab_overlap_spin = _NoScrollSpinBox()
        slab_overlap_spin.setMinimum(0)
        slab_overlap_spin.setMaximum(499)
        slab_overlap_spin.setFixedWidth(130)
        slab_overlap_spin.setToolTip("Number of images shared between adjacent slabs. Must be less than slab size.")
        self._w_zstack_int["slab_overlap"] = slab_overlap_spin
        fs_form.addRow(QLabel("Slab overlap:"), slab_overlap_spin)

        workers_spin = _NoScrollSpinBox()
        workers_spin.setMinimum(1)
        workers_spin.setMaximum(16)
        workers_spin.setFixedWidth(130)
        workers_spin.setToolTip("Number of parallel workers for stacking. Higher values are faster but increase peak RAM by ~100 MiB per additional worker.")
        self._w_zstack_int["workers"] = workers_spin
        fs_form.addRow(QLabel("Workers:"), workers_spin)

        vbox.addWidget(fs_box)
        return group

    def _build_tree_core_group(self) -> QGroupBox:
        group = QGroupBox("Tree Core")
        vbox = QVBoxLayout(group)

        # -- Run settings sub-group ------------------------------------------
        run_box = QGroupBox("Run Settings")
        run_form = QFormLayout(run_box)
        run_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._run_form = run_form

        self._w_run: dict[str, QDoubleSpinBox] = {}

        # Keys that get Set/Go To buttons and the coordinate they read/write.
        # "axis"  = the automation axis opposite coordinate (same logic as slots)
        # "z"     = Z axis
        _BUTTON_FIELDS: dict[str, str] = {
            "mark_reference_nm":  "axis",
            "mark_z_nm":          "z",
            "starting_height_nm": "z",
            "starting_offset_nm": "axis",
        }

        self._axis_labels: dict[str, QLabel] = {}

        axis = self._get_axis()
        axis_upper = axis.upper()
        opp_axis = "y" if axis == "x" else "x"

        # Map each run field to the axis whose limits apply.
        _FIELD_AXIS: dict[str, str] = {
            "mark_reference_nm":  axis,
            "mark_z_nm":          "z",
            "starting_height_nm": "z",
            "starting_offset_nm": axis,
            "slot_separation_nm": axis,
        }

        for key, label_text, tooltip in (
            ("mark_reference_nm",   f"Mark {axis_upper} (mm):",           "Axis coordinate of the reference mark (mm)."),
            ("mark_z_nm",           "Mark Z (mm):",                        "Z coordinate of the reference mark (mm)."),
            ("starting_height_nm",  "Starting height (mm):",               "Z height to move to at the start of the run (mm)."),
            ("starting_offset_nm",  f"Starting {axis_upper} offset (mm):", "Offset applied along the automation axis before the first slot (mm)."),
            ("slot_separation_nm",  "Slot separation (mm):",               "Distance between consecutive slots along the automation axis (mm)."),
        ):
            spin = _NoScrollDoubleSpinBox()
            spin.setMinimum(0.0)
            spin.setMaximum(self._axis_max_mm(_FIELD_AXIS[key]))
            spin.setSingleStep(0.001)
            spin.setDecimals(3)
            spin.setFixedWidth(130)
            spin.setToolTip(tooltip)
            self._w_run[key] = spin

            row_label = QLabel(label_text)
            if key in ("mark_reference_nm", "starting_offset_nm"):
                self._axis_labels[key] = row_label

            if key in _BUTTON_FIELDS:
                coord = _BUTTON_FIELDS[key]
                container = QWidget()
                h = QHBoxLayout(container)
                h.setContentsMargins(0, 0, 0, 0)
                h.setSpacing(4)
                h.addWidget(spin)

                set_btn = QPushButton("Set")
                set_btn.setFixedWidth(38)
                set_btn.setToolTip(f"Set from current printer position ({tooltip})")
                set_btn.clicked.connect(lambda checked=False, k=key, c=coord: self._on_run_set(k, c))
                h.addWidget(set_btn)

                goto_btn = QPushButton("Go to")
                goto_btn.setFixedWidth(50)
                goto_btn.setToolTip(f"Move printer to this value ({tooltip})")
                goto_btn.clicked.connect(lambda checked=False, k=key, c=coord: self._on_run_goto(k, c))
                h.addWidget(goto_btn)

                h.addStretch()
                run_form.addRow(row_label, container)
            else:
                run_form.addRow(row_label, spin)

        vbox.addWidget(run_box)

        # -- Slots sub-group -------------------------------------------------
        slots_box = QGroupBox("Slots")
        slots_vbox = QVBoxLayout(slots_box)

        self._slots_container = QWidget()
        self._slots_layout = QVBoxLayout(self._slots_container)
        self._slots_layout.setContentsMargins(0, 0, 0, 0)
        self._slots_layout.setSpacing(4)
        slots_vbox.addWidget(self._slots_container)

        slot_btn_row = QHBoxLayout()
        self._add_slot_btn = QPushButton("Add Slot")
        self._add_slot_btn.setMaximumWidth(100)
        self._add_slot_btn.clicked.connect(self._on_add_slot)
        self._remove_slot_btn = QPushButton("Remove Last")
        self._remove_slot_btn.setMaximumWidth(110)
        self._remove_slot_btn.clicked.connect(self._on_remove_slot)
        slot_btn_row.addWidget(self._add_slot_btn)
        slot_btn_row.addWidget(self._remove_slot_btn)
        slot_btn_row.addStretch()
        slots_vbox.addLayout(slot_btn_row)

        vbox.addWidget(slots_box)

        return group

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_run_signals(self) -> None:
        for key, spin in self._w_run.items():
            spin.valueChanged.connect(
                lambda v, k=key: self._on_run_field_changed(k, v)
            )
        for key, spin in self._w_general.items():
            spin.valueChanged.connect(
                lambda v, k=key: self._on_general_field_changed(k, v)
            )
        for key, spin in self._w_general_int.items():
            spin.valueChanged.connect(
                lambda v, k=key: self._on_general_int_field_changed(k, v)
            )
        for key, spin in self._w_zstack.items():
            spin.valueChanged.connect(
                lambda v, k=key: self._on_zstack_field_changed(k, v)
            )
        for key, spin in self._w_zstack_int.items():
            spin.valueChanged.connect(
                lambda v, k=key: self._on_zstack_int_field_changed(k, v)
            )
        for key, check in self._w_zstack_check.items():
            check.stateChanged.connect(
                lambda v, k=key: self._on_zstack_check_changed(k, v)
            )

    def _connect_slot_signals(self, row: _SlotRow) -> None:
        for key, spin in row.widgets.items():
            spin.valueChanged.connect(
                lambda v, r=row, k=key: self._on_slot_field_changed(r.index, k, v)
            )

    # ------------------------------------------------------------------
    # Populate from settings
    # ------------------------------------------------------------------

    def _populate_from_settings(self, s: MotionSystemSettings) -> None:
        tca = s.tree_core_automation

        self._block_general_signals(True)
        self._w_general["overlap_x_pct"].setValue(s.automation.overlap_x_pct)
        self._w_general["overlap_y_pct"].setValue(s.automation.overlap_y_pct)
        self._w_general_int["capture_timeout_ms"].setValue(s.automation.capture_timeout_ms)
        self._block_general_signals(False)

        self._block_zstack_signals(True)
        zs = s.z_stack_scan
        self._w_zstack["step_nm"].setValue(zs.step_nm / _NM_PER_MM)
        self._w_zstack["approach_distance_nm"].setValue(zs.approach_distance_nm / _NM_PER_MM)
        self._w_zstack["sharpness"].setValue(zs.sharpness)
        self._w_zstack["cull_threshold"].setValue(zs.cull_threshold)
        self._w_zstack_int["slab_size"].setValue(zs.slab_size)
        self._w_zstack_int["slab_overlap"].setValue(zs.slab_overlap)
        self._w_zstack_int["workers"].setValue(zs.workers)
        self._w_zstack_check["run_focus_stack"].setChecked(zs.run_focus_stack)
        self._w_zstack_check["keep_size"].setChecked(zs.keep_size)
        self._w_zstack_check["no_align"].setChecked(zs.no_align)
        self._w_zstack_check["crop"].setChecked(zs.crop)
        self._w_zstack_check["cull_enabled"].setChecked(zs.cull_enabled)
        self._w_zstack_check["slab_enabled"].setChecked(zs.slab_enabled)
        self._block_zstack_signals(False)

        self._block_run_signals(True)
        self._w_run["mark_reference_nm"].setValue(tca.mark_reference_nm / _NM_PER_MM)
        self._w_run["mark_z_nm"].setValue(tca.mark_z_nm / _NM_PER_MM)
        self._w_run["starting_height_nm"].setValue(tca.starting_height_nm / _NM_PER_MM)
        self._w_run["starting_offset_nm"].setValue(tca.starting_offset_nm / _NM_PER_MM)
        self._w_run["slot_separation_nm"].setValue(tca.slot_separation_nm / _NM_PER_MM)
        self._block_run_signals(False)

        axis_upper = tca.axis.upper()
        if "mark_reference_nm" in self._axis_labels:
            self._axis_labels["mark_reference_nm"].setText(f"Mark {axis_upper} (mm):")
        if "starting_offset_nm" in self._axis_labels:
            self._axis_labels["starting_offset_nm"].setText(f"Starting {axis_upper} offset (mm):")

        while len(tca.slots) < tca.num_slots:
            tca.slots.append(TreeCoreSlot())

        self._rebuild_slot_rows(tca.slots)
        self._snapshot_saved_values(s)
        self._set_unsaved(False)

    def _rebuild_slot_rows(self, slots: list[TreeCoreSlot]) -> None:
        for row in self._slot_rows:
            self._slots_layout.removeWidget(row)
            row.deleteLater()
        self._slot_rows.clear()

        for i, slot in enumerate(slots):
            axis_max = self._axis_max_mm(self._get_axis())
            row = _SlotRow(i, slot, self._get_axis, self._get_motion, axis_max)
            self._connect_slot_signals(row)
            self._slots_layout.addWidget(row)
            self._slot_rows.append(row)

        self._remove_slot_btn.setEnabled(bool(self._slot_rows))

    # ------------------------------------------------------------------
    # Saved-value snapshot and orange tracking
    # ------------------------------------------------------------------

    def _snapshot_saved_values(self, s: MotionSystemSettings) -> None:
        tca = s.tree_core_automation
        zs = s.z_stack_scan
        self._saved_values = {
            "overlap_x_pct":        s.automation.overlap_x_pct,
            "overlap_y_pct":        s.automation.overlap_y_pct,
            "capture_timeout_ms":   s.automation.capture_timeout_ms,
            "zs.step_nm":           zs.step_nm,
            "zs.approach_distance_nm": zs.approach_distance_nm,
            "zs.run_focus_stack":   zs.run_focus_stack,
            "zs.keep_size":         zs.keep_size,
            "zs.no_align":          zs.no_align,
            "zs.crop":              zs.crop,
            "zs.sharpness":         zs.sharpness,
            "zs.cull_enabled":      zs.cull_enabled,
            "zs.cull_threshold":    zs.cull_threshold,
            "zs.slab_enabled":      zs.slab_enabled,
            "zs.slab_size":         zs.slab_size,
            "zs.slab_overlap":      zs.slab_overlap,
            "zs.workers":           zs.workers,
            "mark_reference_nm":    tca.mark_reference_nm,
            "mark_z_nm":            tca.mark_z_nm,
            "starting_height_nm":   tca.starting_height_nm,
            "starting_offset_nm":   tca.starting_offset_nm,
            "slot_separation_nm":   tca.slot_separation_nm,
            "num_slots":            tca.num_slots,
        }
        for i, slot in enumerate(tca.slots):
            self._saved_values[f"slot.{i}.position_nm"] = slot.position_nm
            self._saved_values[f"slot.{i}.offset_nm"]   = slot.offset_nm

    def _check_modified(self, key: str, current_value: object) -> bool:
        return self._saved_values.get(key) != current_value

    def _apply_orange(self, widget: QWidget, orange: bool) -> None:
        widget.setStyleSheet(f"color: {_ORANGE};" if orange else "")

    def _mark_run_field(self, key: str, value: object) -> None:
        w = self._w_run.get(key)
        if w:
            self._apply_orange(w, self._check_modified(key, value))

    def _mark_slot_field(self, index: int, key: str, value: object) -> None:
        if index < len(self._slot_rows):
            w = self._slot_rows[index].widgets.get(key)
            if w:
                self._apply_orange(w, self._check_modified(f"slot.{index}.{key}", value))

    # ------------------------------------------------------------------
    # Change handlers
    # ------------------------------------------------------------------

    def _mark_general_field(self, key: str, value: object) -> None:
        w = self._w_general.get(key)
        if w:
            self._apply_orange(w, self._check_modified(key, value))

    def _mark_general_int_field(self, key: str, value: object) -> None:
        w = self._w_general_int.get(key)
        if w:
            self._apply_orange(w, self._check_modified(key, value))

    def _on_general_field_changed(self, key: str, value: float) -> None:
        s = self._live_settings()
        if s is None:
            self._mark_general_field(key, value)
            self._set_unsaved(True)
            return
        setattr(s.automation, key, value)
        self._mark_general_field(key, value)
        self._set_unsaved(True)

    def _on_general_int_field_changed(self, key: str, value: int) -> None:
        s = self._live_settings()
        if s is None:
            self._mark_general_int_field(key, value)
            self._set_unsaved(True)
            return
        setattr(s.automation, key, value)
        self._mark_general_int_field(key, value)
        self._set_unsaved(True)

    def _mark_zstack_field(self, key: str, value: object) -> None:
        w = self._w_zstack.get(key) or self._w_zstack_int.get(key) or self._w_zstack_check.get(key)
        if w:
            self._apply_orange(w, self._check_modified(f"zs.{key}", value))

    def _on_zstack_field_changed(self, key: str, value: float) -> None:
        s = self._live_settings()
        nm_keys = {"step_nm", "approach_distance_nm"}
        stored = round(value * _NM_PER_MM) if key in nm_keys else value
        if s is not None:
            if key in nm_keys:
                setattr(s.z_stack_scan, key, round(value * _NM_PER_MM))
            else:
                setattr(s.z_stack_scan, key, value)
        self._mark_zstack_field(key, stored)
        self._set_unsaved(True)

    def _on_zstack_int_field_changed(self, key: str, value: int) -> None:
        s = self._live_settings()
        if s is not None:
            setattr(s.z_stack_scan, key, value)
        self._mark_zstack_field(key, value)
        self._set_unsaved(True)

    def _on_zstack_check_changed(self, key: str, value: int) -> None:
        checked = value != 0
        s = self._live_settings()
        if s is not None:
            setattr(s.z_stack_scan, key, checked)
        self._mark_zstack_field(key, checked)
        self._set_unsaved(True)

    def _on_run_field_changed(self, key: str, value_mm: float) -> None:
        value_nm = round(value_mm * _NM_PER_MM)
        s = self._live_settings()
        if s is None:
            self._mark_run_field(key, value_nm)
            self._set_unsaved(True)
            return
        setattr(s.tree_core_automation, key, value_nm)
        self._mark_run_field(key, value_nm)
        self._set_unsaved(True)

    def _on_slot_field_changed(self, index: int, key: str, value_mm: float) -> None:
        value_nm = round(value_mm * _NM_PER_MM)
        s = self._live_settings()
        if s is None:
            self._mark_slot_field(index, key, value_nm)
            self._set_unsaved(True)
            return
        if index < len(s.tree_core_automation.slots):
            setattr(s.tree_core_automation.slots[index], key, value_nm)
            self._mark_slot_field(index, key, value_nm)
            self._set_unsaved(True)

    def _on_add_slot(self) -> None:
        s = self._live_settings()
        if s is None:
            return
        s.tree_core_automation.slots.append(TreeCoreSlot())
        self._rebuild_slot_rows(s.tree_core_automation.slots)
        self._set_unsaved(True)

    def _on_remove_slot(self) -> None:
        s = self._live_settings()
        if s is None:
            return
        if s.tree_core_automation.slots:
            s.tree_core_automation.slots.pop()
            self._rebuild_slot_rows(s.tree_core_automation.slots)
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
            if ctx.toast:
                ctx.toast.success("Automation settings saved", duration=2000)
            info("Automation settings saved")
        except Exception as exc:
            error(f"Failed to save automation settings: {exc}")
            if ctx.toast:
                ctx.toast.error(f"Save failed: {exc}", duration=3000)

    def _clear_all_orange(self) -> None:
        for w in self._w_general.values():
            self._apply_orange(w, False)
        for w in self._w_general_int.values():
            self._apply_orange(w, False)
        for w in self._w_zstack.values():
            self._apply_orange(w, False)
        for w in self._w_zstack_int.values():
            self._apply_orange(w, False)
        for w in self._w_zstack_check.values():
            self._apply_orange(w, False)
        for w in self._w_run.values():
            self._apply_orange(w, False)
        for row in self._slot_rows:
            for w in row.widgets.values():
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
                self.parent_dialog.set_category_modified("Automation", has_changes)

    def has_unsaved_changes(self) -> bool:
        return self._has_unsaved_changes

    # ------------------------------------------------------------------
    # Sidebar sub-item support
    # ------------------------------------------------------------------

    def get_group_names(self) -> list[str]:
        return list(self._GROUP_NAMES)

    # ------------------------------------------------------------------
    # Signal blocking helpers
    # ------------------------------------------------------------------

    def _block_general_signals(self, block: bool) -> None:
        for w in self._w_general.values():
            w.blockSignals(block)
        for w in self._w_general_int.values():
            w.blockSignals(block)

    def _block_zstack_signals(self, block: bool) -> None:
        for w in self._w_zstack.values():
            w.blockSignals(block)
        for w in self._w_zstack_int.values():
            w.blockSignals(block)
        for w in self._w_zstack_check.values():
            w.blockSignals(block)

    def _block_run_signals(self, block: bool) -> None:
        for w in self._w_run.values():
            w.blockSignals(block)

    def _block_all_signals(self, block: bool) -> None:
        self._block_general_signals(block)
        self._block_zstack_signals(block)
        self._block_run_signals(block)
        for row in self._slot_rows:
            for w in row.widgets.values():
                w.blockSignals(block)


def automation_page(parent_dialog=None) -> QWidget:
    """Create and return the automation settings page widget."""
    return AutomationSettingsWidget(parent_dialog=parent_dialog)