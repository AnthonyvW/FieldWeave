from __future__ import annotations

import math

from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from common.app_context import get_app_context
from motion.motion_config import MotionSystemSettings
from UI.settings.pages.shared import NM_PER_MM, NoScrollDoubleSpinBox, NoScrollSpinBox, SettingsGroupBase


class ZStackSettingsWidget(SettingsGroupBase):
    """Z-Stack Scan settings group (scan params and focus stack options)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Z-Stack Scan", parent)
        self._w: dict[str, NoScrollDoubleSpinBox] = {}
        self._w_int: dict[str, NoScrollSpinBox] = {}
        self._w_check: dict[str, QCheckBox] = {}
        self._saved: dict[str, object] = {}
        self._build()

    def _get_printer_step_mm(self) -> float:
        motion = get_app_context().motion
        if motion is not None and motion.settings is not None:
            return motion.settings.step_size / 1_000_000.0
        return 0.04

    def _build(self) -> None:
        vbox = QVBoxLayout(self)

        note = QLabel("These defaults will take effect the next time the program is launched.")
        note.setWordWrap(True)
        vbox.addWidget(note)

        scan_box = QGroupBox("Scan")
        scan_form = QFormLayout(scan_box)
        scan_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        printer_step_mm = self._get_printer_step_mm()
        step_spin = NoScrollDoubleSpinBox()
        step_spin.setMinimum(printer_step_mm)
        step_spin.setMaximum(10.0)
        step_spin.setSingleStep(printer_step_mm)
        step_spin.setDecimals(max(2, -int(math.floor(math.log10(printer_step_mm)))))
        step_spin.setSuffix(" mm")
        step_spin.setFixedWidth(130)
        step_spin.setToolTip("Distance between capture positions.")
        self._w["step_nm"] = step_spin
        scan_form.addRow(self._register_label("step_nm", QLabel("Step size:")), step_spin)

        approach_spin = NoScrollDoubleSpinBox()
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
        self._w["approach_distance_nm"] = approach_spin
        scan_form.addRow(self._register_label("approach_distance_nm", QLabel("Approach distance:")), approach_spin)

        vbox.addWidget(scan_box)

        fs_box = QGroupBox("Focus Stack")
        fs_form = QFormLayout(fs_box)
        fs_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        for key, label_text, tooltip in (
            ("run_focus_stack", "Run after capture:", "Automatically run focus stacking after all frames are captured."),
            ("keep_size",       "Keep original size:", "Keep the output image the same size as the inputs."),
            ("no_align",        "Skip alignment:", "Skip ECC alignment. Use when images are already registered."),
            ("crop",            "Crop to intersection:", "Crop output to the largest rectangle covered by every frame after alignment."),
            ("cull_enabled",    "Cull out-of-focus frames:", "Discard frames whose focus score falls below the threshold fraction of the sharpest frame."),
            ("slab_enabled",    "Enable slabbing:", "Split the image set into overlapping sub-stacks, stack each independently, then fuse. Reduces peak RAM."),
        ):
            check = QCheckBox()
            check.setToolTip(tooltip)
            self._w_check[key] = check
            fs_form.addRow(self._register_label(key, QLabel(label_text)), check)

        sharpness_spin = NoScrollDoubleSpinBox()
        sharpness_spin.setMinimum(1.0)
        sharpness_spin.setMaximum(8.0)
        sharpness_spin.setSingleStep(0.5)
        sharpness_spin.setDecimals(1)
        sharpness_spin.setFixedWidth(130)
        sharpness_spin.setToolTip("Weight sharpness exponent. Higher values favour the sharpest pixel more aggressively. Useful range: 1.0 (soft) to 8.0 (near-hard).")
        self._w["sharpness"] = sharpness_spin
        fs_form.addRow(self._register_label("sharpness", QLabel("Sharpness:")), sharpness_spin)

        cull_spin = NoScrollDoubleSpinBox()
        cull_spin.setMinimum(0.0)
        cull_spin.setMaximum(1.0)
        cull_spin.setSingleStep(0.05)
        cull_spin.setDecimals(2)
        cull_spin.setFixedWidth(130)
        cull_spin.setToolTip("Frames scoring below this fraction of the peak score are culled.")
        self._w["cull_threshold"] = cull_spin
        fs_form.addRow(self._register_label("cull_threshold", QLabel("Cull threshold:")), cull_spin)

        slab_size_spin = NoScrollSpinBox()
        slab_size_spin.setMinimum(2)
        slab_size_spin.setMaximum(500)
        slab_size_spin.setFixedWidth(130)
        slab_size_spin.setToolTip("Number of images per sub-stack.")
        self._w_int["slab_size"] = slab_size_spin
        fs_form.addRow(self._register_label("slab_size", QLabel("Slab size:")), slab_size_spin)

        slab_overlap_spin = NoScrollSpinBox()
        slab_overlap_spin.setMinimum(0)
        slab_overlap_spin.setMaximum(499)
        slab_overlap_spin.setFixedWidth(130)
        slab_overlap_spin.setToolTip("Number of images shared between adjacent slabs. Must be less than slab size.")
        self._w_int["slab_overlap"] = slab_overlap_spin
        fs_form.addRow(self._register_label("slab_overlap", QLabel("Slab overlap:")), slab_overlap_spin)

        workers_spin = NoScrollSpinBox()
        workers_spin.setMinimum(1)
        workers_spin.setMaximum(16)
        workers_spin.setFixedWidth(130)
        workers_spin.setToolTip("Number of parallel workers for stacking.")
        self._w_int["workers"] = workers_spin
        fs_form.addRow(self._register_label("workers", QLabel("Workers:")), workers_spin)

        vbox.addWidget(fs_box)

    def connect_signals(self, on_float, on_int, on_check) -> None:
        for key, spin in self._w.items():
            spin.valueChanged.connect(lambda v, k=key: on_float(k, v))
        for key, spin in self._w_int.items():
            spin.valueChanged.connect(lambda v, k=key: on_int(k, v))
        for key, check in self._w_check.items():
            check.stateChanged.connect(lambda v, k=key: on_check(k, v))

    def populate(self, s: MotionSystemSettings) -> None:
        for w in self._w.values():
            w.blockSignals(True)
        for w in self._w_int.values():
            w.blockSignals(True)
        for w in self._w_check.values():
            w.blockSignals(True)

        zs = s.z_stack_scan
        self._w["step_nm"].setValue(zs.step_nm / NM_PER_MM)
        self._w["approach_distance_nm"].setValue(zs.approach_distance_nm / NM_PER_MM)
        self._w["sharpness"].setValue(zs.sharpness)
        self._w["cull_threshold"].setValue(zs.cull_threshold)
        self._w_int["slab_size"].setValue(zs.slab_size)
        self._w_int["slab_overlap"].setValue(zs.slab_overlap)
        self._w_int["workers"].setValue(zs.workers)
        self._w_check["run_focus_stack"].setChecked(zs.run_focus_stack)
        self._w_check["keep_size"].setChecked(zs.keep_size)
        self._w_check["no_align"].setChecked(zs.no_align)
        self._w_check["crop"].setChecked(zs.crop)
        self._w_check["cull_enabled"].setChecked(zs.cull_enabled)
        self._w_check["slab_enabled"].setChecked(zs.slab_enabled)

        for w in self._w.values():
            w.blockSignals(False)
        for w in self._w_int.values():
            w.blockSignals(False)
        for w in self._w_check.values():
            w.blockSignals(False)

    def snapshot(self) -> None:
        self._saved = {
            "zs.step_nm":              round(self._w["step_nm"].value() * NM_PER_MM),
            "zs.approach_distance_nm": round(self._w["approach_distance_nm"].value() * NM_PER_MM),
            "zs.run_focus_stack":      self._w_check["run_focus_stack"].isChecked(),
            "zs.keep_size":            self._w_check["keep_size"].isChecked(),
            "zs.no_align":             self._w_check["no_align"].isChecked(),
            "zs.crop":                 self._w_check["crop"].isChecked(),
            "zs.sharpness":            self._w["sharpness"].value(),
            "zs.cull_enabled":         self._w_check["cull_enabled"].isChecked(),
            "zs.cull_threshold":       self._w["cull_threshold"].value(),
            "zs.slab_enabled":         self._w_check["slab_enabled"].isChecked(),
            "zs.slab_size":            self._w_int["slab_size"].value(),
            "zs.slab_overlap":         self._w_int["slab_overlap"].value(),
            "zs.workers":              self._w_int["workers"].value(),
        }

    def apply_float_to_live(self, key: str, value: float) -> None:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return
        nm_keys = {"step_nm", "approach_distance_nm"}
        if key in nm_keys:
            setattr(motion.settings.z_stack_scan, key, round(value * NM_PER_MM))
        else:
            setattr(motion.settings.z_stack_scan, key, value)

    def apply_int_to_live(self, key: str, value: int) -> None:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return
        setattr(motion.settings.z_stack_scan, key, value)

    def apply_check_to_live(self, key: str, value: int) -> None:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return
        setattr(motion.settings.z_stack_scan, key, value != 0)

    def has_changes(self) -> bool:
        zs_checks = {k: self._w_check[k].isChecked() for k in self._w_check}
        return any(
            self._saved.get(f"zs.{k}") != v for k, v in zs_checks.items()
        ) or any(
            self._saved.get(f"zs.{k}") != v.value() for k, v in self._w.items()
            if k not in {"step_nm", "approach_distance_nm"}
        ) or (
            self._saved.get("zs.step_nm") != round(self._w["step_nm"].value() * NM_PER_MM)
        ) or (
            self._saved.get("zs.approach_distance_nm") != round(self._w["approach_distance_nm"].value() * NM_PER_MM)
        ) or any(
            self._saved.get(f"zs.{k}") != v.value() for k, v in self._w_int.items()
        )

    def mark_field(self, key: str, stored_value: object) -> None:
        self.mark_label(key, self._saved.get(f"zs.{key}") != stored_value)