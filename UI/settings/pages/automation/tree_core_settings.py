from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from common.app_context import get_app_context
from motion.motion_config import MotionSystemSettings, TreeCoreSlot
from UI.settings.pages.shared import LabelTrackerMixin, NM_PER_MM, NoScrollDoubleSpinBox, SettingsGroupBase


class _SlotRow(LabelTrackerMixin, QWidget):
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
        self._w: dict[str, NoScrollDoubleSpinBox] = {}
        self._labels: dict[str, QLabel] = {}
        self._build(slot)

    def _build(self, slot: TreeCoreSlot) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        slot_label = QLabel(f"Slot {self._index + 1}:")
        slot_label.setFixedWidth(50)
        row.addWidget(slot_label)

        for key, value_nm, pos_label, tooltip in (
            ("position_nm", slot.position_nm, "Pos (mm):", "Position along the automation axis (mm)."),
            ("offset_nm",   slot.offset_nm,   "Offset (mm):", "Fine-tune offset applied on top of the slot position (mm)."),
        ):
            lbl = QLabel(pos_label)
            lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            row.addWidget(lbl)
            self._labels[key] = lbl

            spin = NoScrollDoubleSpinBox()
            spin.setMinimum(0.0)
            spin.setMaximum(self._axis_max_mm)
            spin.setSingleStep(0.001)
            spin.setDecimals(3)
            spin.setValue(value_nm / NM_PER_MM)
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
        pos = motion.get_position()
        if pos is None:
            return
        axis = self._get_axis()
        current_nm = pos.y if axis == "x" else pos.x
        if key == "offset_nm":
            position_nm = round(self._w["position_nm"].value() * NM_PER_MM)
            value_nm = current_nm - position_nm
        else:
            value_nm = current_nm
        spin = self._w[key]
        spin.blockSignals(True)
        spin.setValue(value_nm / NM_PER_MM)
        spin.blockSignals(False)
        spin.valueChanged.emit(spin.value())

    def _on_goto(self, key: str) -> None:
        motion = self._get_motion()
        if motion is None or not motion.is_ready():
            return
        from motion.models import Position
        current = motion.get_position()
        if current is None:
            return
        value_nm = round(self._w[key].value() * NM_PER_MM)
        axis = self._get_axis()
        if axis == "x":
            target = Position(x=current.x, y=value_nm, z=current.z)
        else:
            target = Position(x=value_nm, y=current.y, z=current.z)
        motion.move_to_position(target, wait=False)

    @property
    def widgets(self) -> dict[str, NoScrollDoubleSpinBox]:
        return self._w

    @property
    def index(self) -> int:
        return self._index

    def to_slot(self) -> TreeCoreSlot:
        return TreeCoreSlot(
            position_nm=round(self._w["position_nm"].value() * NM_PER_MM),
            offset_nm=round(self._w["offset_nm"].value() * NM_PER_MM),
        )


class TreeCoreSettingsWidget(SettingsGroupBase):
    """Tree Core automation settings group (run settings and per-slot positions)."""

    _BUTTON_FIELDS: dict[str, str] = {
        "mark_reference_nm":  "axis",
        "mark_z_nm":          "z",
        "starting_height_nm": "z",
        "starting_offset_nm": "axis",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Tree Core", parent)
        self._w_run: dict[str, NoScrollDoubleSpinBox] = {}
        self._axis_labels: dict[str, QLabel] = {}
        self._slot_rows: list[_SlotRow] = []
        self._saved: dict[str, object] = {}
        self._slots_layout: QVBoxLayout | None = None
        self._remove_slot_btn: QPushButton | None = None
        self._build()

    def _get_axis(self) -> str:
        motion = get_app_context().motion
        if motion is not None and motion.settings is not None:
            return motion.settings.tree_core_automation.axis.lower()
        return "x"

    def _axis_max_mm(self, axis: str) -> float:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return 300.0
        s = motion.settings
        if axis == "x":
            return float(s.max_x)
        if axis == "y":
            return float(s.max_y)
        return float(s.max_z)

    def _get_motion(self):
        return get_app_context().motion

    def _on_run_set(self, key: str, coord: str) -> None:
        motion = self._get_motion()
        if motion is None or not motion.is_ready():
            return
        pos = motion.get_position()
        if pos is None:
            return
        if coord == "z":
            value_nm = pos.z
        else:
            axis = self._get_axis()
            value_nm = pos.x if axis == "x" else pos.y
        spin = self._w_run[key]
        spin.blockSignals(True)
        spin.setValue(value_nm / NM_PER_MM)
        spin.blockSignals(False)
        spin.valueChanged.emit(spin.value())

    def _on_run_goto(self, key: str, coord: str) -> None:
        motion = self._get_motion()
        if motion is None or not motion.is_ready():
            return
        from motion.models import Position
        current = motion.get_position()
        if current is None:
            return
        value_nm = round(self._w_run[key].value() * NM_PER_MM)
        if coord == "z":
            target = Position(x=current.x, y=current.y, z=value_nm)
        else:
            axis = self._get_axis()
            if axis == "x":
                target = Position(x=value_nm, y=current.y, z=current.z)
            else:
                target = Position(x=current.x, y=value_nm, z=current.z)
        motion.move_to_position(target, wait=False)

    def _build(self) -> None:
        vbox = QVBoxLayout(self)

        run_box = QGroupBox("Run Settings")
        run_form = QFormLayout(run_box)
        run_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        axis = self._get_axis()
        axis_upper = axis.upper()

        field_axis: dict[str, str] = {
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
            spin = NoScrollDoubleSpinBox()
            spin.setMinimum(0.0)
            spin.setMaximum(self._axis_max_mm(field_axis[key]))
            spin.setSingleStep(0.001)
            spin.setDecimals(3)
            spin.setFixedWidth(130)
            spin.setToolTip(tooltip)
            self._w_run[key] = spin

            row_label = self._register_label(key, QLabel(label_text))
            if key in ("mark_reference_nm", "starting_offset_nm"):
                self._axis_labels[key] = row_label

            if key in self._BUTTON_FIELDS:
                coord = self._BUTTON_FIELDS[key]
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

        slots_box = QGroupBox("Slots")
        slots_vbox = QVBoxLayout(slots_box)

        self._slots_container = QWidget()
        self._slots_layout = QVBoxLayout(self._slots_container)
        self._slots_layout.setContentsMargins(0, 0, 0, 0)
        self._slots_layout.setSpacing(4)
        slots_vbox.addWidget(self._slots_container)

        slot_btn_row = QHBoxLayout()
        add_slot_btn = QPushButton("Add Slot")
        add_slot_btn.setMaximumWidth(100)
        add_slot_btn.clicked.connect(self._on_add_slot)
        self._remove_slot_btn = QPushButton("Remove Last")
        self._remove_slot_btn.setMaximumWidth(110)
        self._remove_slot_btn.clicked.connect(self._on_remove_slot)
        slot_btn_row.addWidget(add_slot_btn)
        slot_btn_row.addWidget(self._remove_slot_btn)
        slot_btn_row.addStretch()
        slots_vbox.addLayout(slot_btn_row)

        vbox.addWidget(slots_box)

    def connect_signals(self, on_run_changed, on_slot_changed) -> None:
        self._on_slot_changed_cb = on_slot_changed
        for key, spin in self._w_run.items():
            spin.valueChanged.connect(lambda v, k=key: on_run_changed(k, v))

    def _connect_slot_signals(self, row: _SlotRow) -> None:
        for key, spin in row.widgets.items():
            spin.valueChanged.connect(
                lambda v, r=row, k=key: self._on_slot_changed_cb(r.index, k, v)
            )

    def populate(self, s: MotionSystemSettings) -> None:
        tca = s.tree_core_automation

        for w in self._w_run.values():
            w.blockSignals(True)
        self._w_run["mark_reference_nm"].setValue(tca.mark_reference_nm / NM_PER_MM)
        self._w_run["mark_z_nm"].setValue(tca.mark_z_nm / NM_PER_MM)
        self._w_run["starting_height_nm"].setValue(tca.starting_height_nm / NM_PER_MM)
        self._w_run["starting_offset_nm"].setValue(tca.starting_offset_nm / NM_PER_MM)
        self._w_run["slot_separation_nm"].setValue(tca.slot_separation_nm / NM_PER_MM)
        for w in self._w_run.values():
            w.blockSignals(False)

        axis_upper = tca.axis.upper()
        if "mark_reference_nm" in self._axis_labels:
            self._axis_labels["mark_reference_nm"].setText(f"Mark {axis_upper} (mm):")
        if "starting_offset_nm" in self._axis_labels:
            self._axis_labels["starting_offset_nm"].setText(f"Starting {axis_upper} offset (mm):")

        while len(tca.slots) < tca.num_slots:
            tca.slots.append(TreeCoreSlot())

        self._rebuild_slot_rows(tca.slots)

    def snapshot(self) -> None:
        self._saved = {
            "mark_reference_nm":  round(self._w_run["mark_reference_nm"].value() * NM_PER_MM),
            "mark_z_nm":          round(self._w_run["mark_z_nm"].value() * NM_PER_MM),
            "starting_height_nm": round(self._w_run["starting_height_nm"].value() * NM_PER_MM),
            "starting_offset_nm": round(self._w_run["starting_offset_nm"].value() * NM_PER_MM),
            "slot_separation_nm": round(self._w_run["slot_separation_nm"].value() * NM_PER_MM),
            "num_slots":          len(self._slot_rows),
        }
        for i, row in enumerate(self._slot_rows):
            self._saved[f"slot.{i}.position_nm"] = round(row.widgets["position_nm"].value() * NM_PER_MM)
            self._saved[f"slot.{i}.offset_nm"]   = round(row.widgets["offset_nm"].value() * NM_PER_MM)

    def apply_run_to_live(self, key: str, value_mm: float) -> None:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return
        setattr(motion.settings.tree_core_automation, key, round(value_mm * NM_PER_MM))

    def apply_slot_to_live(self, index: int, key: str, value_mm: float) -> None:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return
        slots = motion.settings.tree_core_automation.slots
        if index < len(slots):
            setattr(slots[index], key, round(value_mm * NM_PER_MM))

    def _on_add_slot(self) -> None:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return
        motion.settings.tree_core_automation.slots.append(TreeCoreSlot())
        self._rebuild_slot_rows(motion.settings.tree_core_automation.slots)
        if hasattr(self, "_on_slot_add_cb"):
            self._on_slot_add_cb()

    def _on_remove_slot(self) -> None:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return
        slots = motion.settings.tree_core_automation.slots
        if slots:
            slots.pop()
            self._rebuild_slot_rows(slots)
            if hasattr(self, "_on_slot_add_cb"):
                self._on_slot_add_cb()

    def set_slot_mutation_callback(self, cb) -> None:
        self._on_slot_add_cb = cb

    def _rebuild_slot_rows(self, slots: list[TreeCoreSlot]) -> None:
        for row in self._slot_rows:
            self._slots_layout.removeWidget(row)
            row.deleteLater()
        self._slot_rows.clear()

        for i, slot in enumerate(slots):
            axis_max = self._axis_max_mm(self._get_axis())
            row = _SlotRow(i, slot, self._get_axis, self._get_motion, axis_max)
            if hasattr(self, "_on_slot_changed_cb"):
                self._connect_slot_signals(row)
            self._slots_layout.addWidget(row)
            self._slot_rows.append(row)

        if self._remove_slot_btn is not None:
            self._remove_slot_btn.setEnabled(bool(self._slot_rows))

    def has_changes(self) -> bool:
        for key, spin in self._w_run.items():
            if self._saved.get(key) != round(spin.value() * NM_PER_MM):
                return True
        for i, row in enumerate(self._slot_rows):
            for key, spin in row.widgets.items():
                if self._saved.get(f"slot.{i}.{key}") != round(spin.value() * NM_PER_MM):
                    return True
        return False

    def clear_orange(self) -> None:
        super().clear_orange()
        for row in self._slot_rows:
            row.clear_orange()

    def mark_run_field(self, key: str, value_nm: int) -> None:
        self.mark_label(key, self._saved.get(key) != value_nm)

    def mark_slot_field(self, index: int, key: str, value_nm: int) -> None:
        if index < len(self._slot_rows):
            self._slot_rows[index].mark_label(key, self._saved.get(f"slot.{index}.{key}") != value_nm)

    def block_run_signals(self, block: bool) -> None:
        for w in self._w_run.values():
            w.blockSignals(block)

    def block_slot_signals(self, block: bool) -> None:
        for row in self._slot_rows:
            for w in row.widgets.values():
                w.blockSignals(block)