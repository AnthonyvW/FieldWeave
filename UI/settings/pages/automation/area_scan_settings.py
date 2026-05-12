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


class AreaScanSettingsWidget(SettingsGroupBase):
    """Settings group for the Z-stack area scan routine defaults and focus stack parameters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Z-Stack Area Scan", parent)
        self._w_float: dict[str, NoScrollDoubleSpinBox] = {}
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

        scan_box = QGroupBox("Scan")
        scan_form = QFormLayout(scan_box)
        scan_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        printer_step_mm = self._get_printer_step_mm()
        step_decimals = max(2, -int(math.floor(math.log10(printer_step_mm))))

        for key, label_text, tooltip, min_mm, max_mm in (
            ("x_step_nm", "X step:", "Step size along X between grid positions.", printer_step_mm, 50.0),
            ("y_step_nm", "Y step:", "Step size along Y between grid positions.", printer_step_mm, 50.0),
            ("z_step_nm", "Z step:", "Distance between Z capture positions within each stack.", printer_step_mm, 10.0),
        ):
            spin = NoScrollDoubleSpinBox()
            spin.setMinimum(min_mm)
            spin.setMaximum(max_mm)
            spin.setSingleStep(printer_step_mm)
            spin.setDecimals(step_decimals)
            spin.setSuffix(" mm")
            spin.setFixedWidth(130)
            spin.setToolTip(tooltip)
            self._w_float[key] = spin
            scan_form.addRow(self._register_label(key, QLabel(label_text)), spin)

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

        sharpness = NoScrollDoubleSpinBox()
        sharpness.setMinimum(1.0)
        sharpness.setMaximum(8.0)
        sharpness.setSingleStep(0.5)
        sharpness.setDecimals(1)
        sharpness.setFixedWidth(130)
        sharpness.setToolTip(
            "Weight sharpness exponent. Higher values favour the sharpest pixel "
            "more aggressively. Useful range: 1.0 (soft) to 8.0 (near-hard)."
        )
        self._w_float["sharpness"] = sharpness
        fs_form.addRow(self._register_label("sharpness", QLabel("Sharpness:")), sharpness)

        cull_threshold = NoScrollDoubleSpinBox()
        cull_threshold.setMinimum(0.0)
        cull_threshold.setMaximum(1.0)
        cull_threshold.setSingleStep(0.05)
        cull_threshold.setDecimals(2)
        cull_threshold.setFixedWidth(130)
        cull_threshold.setToolTip("Frames scoring below this fraction of the peak score are culled.")
        self._w_float["cull_threshold"] = cull_threshold
        fs_form.addRow(self._register_label("cull_threshold", QLabel("Cull threshold:")), cull_threshold)

        slab_size = NoScrollSpinBox()
        slab_size.setMinimum(2)
        slab_size.setMaximum(500)
        slab_size.setFixedWidth(130)
        slab_size.setToolTip("Number of images per sub-stack.")
        self._w_int["slab_size"] = slab_size
        fs_form.addRow(self._register_label("slab_size", QLabel("Slab size:")), slab_size)

        slab_overlap = NoScrollSpinBox()
        slab_overlap.setMinimum(0)
        slab_overlap.setMaximum(499)
        slab_overlap.setFixedWidth(130)
        slab_overlap.setToolTip("Number of images shared between adjacent slabs. Must be less than slab size.")
        self._w_int["slab_overlap"] = slab_overlap
        fs_form.addRow(self._register_label("slab_overlap", QLabel("Slab overlap:")), slab_overlap)

        workers = NoScrollSpinBox()
        workers.setMinimum(1)
        workers.setMaximum(16)
        workers.setFixedWidth(130)
        workers.setToolTip("Number of parallel workers for stacking.")
        self._w_int["workers"] = workers
        fs_form.addRow(self._register_label("workers", QLabel("Workers:")), workers)

        vbox.addWidget(fs_box)

    def connect_signals(self, on_float, on_int, on_check) -> None:
        for key, spin in self._w_float.items():
            spin.valueChanged.connect(lambda v, k=key: on_float(k, v))
        for key, spin in self._w_int.items():
            spin.valueChanged.connect(lambda v, k=key: on_int(k, v))
        for key, check in self._w_check.items():
            check.stateChanged.connect(lambda v, k=key: on_check(k, v))

    def populate(self, s: MotionSystemSettings) -> None:
        for w in self._w_float.values():
            w.blockSignals(True)
        for w in self._w_int.values():
            w.blockSignals(True)
        for w in self._w_check.values():
            w.blockSignals(True)

        a = s.z_stack_area_scan
        self._w_float["x_step_nm"].setValue(a.x_step_nm / NM_PER_MM)
        self._w_float["y_step_nm"].setValue(a.y_step_nm / NM_PER_MM)
        self._w_float["z_step_nm"].setValue(a.z_step_nm / NM_PER_MM)
        self._w_float["sharpness"].setValue(a.sharpness)
        self._w_float["cull_threshold"].setValue(a.cull_threshold)
        self._w_int["slab_size"].setValue(a.slab_size)
        self._w_int["slab_overlap"].setValue(a.slab_overlap)
        self._w_int["workers"].setValue(a.workers)
        self._w_check["run_focus_stack"].setChecked(a.run_focus_stack)
        self._w_check["keep_size"].setChecked(a.keep_size)
        self._w_check["no_align"].setChecked(a.no_align)
        self._w_check["crop"].setChecked(a.crop)
        self._w_check["cull_enabled"].setChecked(a.cull_enabled)
        self._w_check["slab_enabled"].setChecked(a.slab_enabled)

        for w in self._w_float.values():
            w.blockSignals(False)
        for w in self._w_int.values():
            w.blockSignals(False)
        for w in self._w_check.values():
            w.blockSignals(False)

    def snapshot(self) -> None:
        self._saved = {
            "a.x_step_nm":       round(self._w_float["x_step_nm"].value() * NM_PER_MM),
            "a.y_step_nm":       round(self._w_float["y_step_nm"].value() * NM_PER_MM),
            "a.z_step_nm":       round(self._w_float["z_step_nm"].value() * NM_PER_MM),
            "a.sharpness":       self._w_float["sharpness"].value(),
            "a.cull_threshold":  self._w_float["cull_threshold"].value(),
            "a.slab_size":       self._w_int["slab_size"].value(),
            "a.slab_overlap":    self._w_int["slab_overlap"].value(),
            "a.workers":         self._w_int["workers"].value(),
            "a.run_focus_stack": self._w_check["run_focus_stack"].isChecked(),
            "a.keep_size":       self._w_check["keep_size"].isChecked(),
            "a.no_align":        self._w_check["no_align"].isChecked(),
            "a.crop":            self._w_check["crop"].isChecked(),
            "a.cull_enabled":    self._w_check["cull_enabled"].isChecked(),
            "a.slab_enabled":    self._w_check["slab_enabled"].isChecked(),
        }

    def apply_float_to_live(self, key: str, value: float) -> None:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return
        nm_keys = {"x_step_nm", "y_step_nm", "z_step_nm"}
        stored = round(value * NM_PER_MM) if key in nm_keys else value
        setattr(motion.settings.z_stack_area_scan, key, stored)

    def apply_int_to_live(self, key: str, value: int) -> None:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return
        setattr(motion.settings.z_stack_area_scan, key, value)

    def apply_check_to_live(self, key: str, value: int) -> None:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return
        setattr(motion.settings.z_stack_area_scan, key, value != 0)

    def has_changes(self) -> bool:
        nm_keys = {"x_step_nm", "y_step_nm", "z_step_nm"}
        return (
            any(
                self._saved.get(f"a.{k}") != round(self._w_float[k].value() * NM_PER_MM)
                for k in nm_keys
            )
            or any(
                self._saved.get(f"a.{k}") != self._w_float[k].value()
                for k in ("sharpness", "cull_threshold")
            )
            or any(
                self._saved.get(f"a.{k}") != self._w_int[k].value()
                for k in ("slab_size", "slab_overlap", "workers")
            )
            or any(
                self._saved.get(f"a.{k}") != self._w_check[k].isChecked()
                for k in ("run_focus_stack", "keep_size", "no_align", "crop", "cull_enabled", "slab_enabled")
            )
        )

    def mark_field(self, key: str, value: object) -> None:
        self.mark_label(key, self._saved.get(f"a.{key}") != value)