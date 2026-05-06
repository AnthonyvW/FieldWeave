from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from common.app_context import get_app_context
from motion.motion_config import MotionSystemSettings
from UI.settings.pages.shared import NM_PER_MM, NoScrollDoubleSpinBox, SettingsGroupBase


_DEFAULT_PRESETS_MM = (0.04, 0.4, 2.0, 10.0)


def _mm_to_nm(mm: float) -> int:
    return round(mm * NM_PER_MM)


def _nm_to_mm(nm: int) -> float:
    return nm / NM_PER_MM


class NavigationGroupSettingsWidget(SettingsGroupBase):
    """Navigation behaviour group: axis inversion, jog-step presets, starting height."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Navigation", parent)
        self._w: dict[str, NoScrollDoubleSpinBox | QCheckBox | QPushButton] = {}
        self._saved: dict[str, object] = {}
        self._build()

    def _build(self) -> None:
        vbox = QVBoxLayout(self)
        vbox.setSpacing(12)

        invert_box = QGroupBox("Axis Inversion")
        invert_form = QFormLayout(invert_box)
        invert_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        for axis in ("x", "y", "z"):
            key = f"invert_{axis}"
            check = QCheckBox()
            check.setToolTip(
                f"Invert the {axis.upper()} axis direction in the navigation widget.\n"
                "Enable if the on-screen arrow moves the stage in the wrong direction."
            )
            self._w[key] = check
            invert_form.addRow(
                self._register_label(key, QLabel(f"Invert {axis.upper()}:")),
                check,
            )

        vbox.addWidget(invert_box)

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
            key = f"preset_{i}"
            preset_spin = NoScrollDoubleSpinBox()
            preset_spin.setMinimum(0.001)
            preset_spin.setMaximum(500.0)
            preset_spin.setDecimals(4)
            preset_spin.setSingleStep(0.01)
            preset_spin.setSuffix(" mm")
            preset_spin.setFixedWidth(130)
            preset_spin.setToolTip(f"Step-size preset {i} for the navigation widget jog buttons.")
            self._w[key] = preset_spin
            presets_form.addRow(self._register_label(key, QLabel(f"Preset {i}:")), preset_spin)

        vbox.addWidget(presets_box)

        starting_height_box = QGroupBox("Starting Height")
        starting_height_vbox = QVBoxLayout(starting_height_box)
        starting_height_vbox.setSpacing(6)

        desc = QLabel(
            "Z position to move to automatically after every home sequence.\n"
            "Set to 0 to stay at the homed position."
        )
        desc.setStyleSheet("color: #5f6368; font-size: 11px;")
        starting_height_vbox.addWidget(desc)

        height_form = QFormLayout()
        height_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        height_spin = NoScrollDoubleSpinBox()
        height_spin.setMinimum(0.0)
        height_spin.setMaximum(10_000.0)
        height_spin.setDecimals(3)
        height_spin.setSingleStep(0.1)
        height_spin.setSuffix(" mm")
        height_spin.setFixedWidth(130)
        height_spin.setToolTip(
            "Z axis position (in millimetres) to move to after homing.\n"
            "0 means no post-home move."
        )
        self._w["starting_height"] = height_spin
        height_form.addRow(self._register_label("starting_height", QLabel("Height:")), height_spin)
        starting_height_vbox.addLayout(height_form)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        set_current_btn = QPushButton("Set to current height")
        set_current_btn.setToolTip("Set the starting height to the stage's current Z position.")
        btn_row.addWidget(set_current_btn)
        self._w["starting_height_set_current"] = set_current_btn

        reset_btn = QPushButton("Reset to 0")
        reset_btn.setToolTip("Clear the starting height (no post-home Z move).")
        btn_row.addWidget(reset_btn)
        self._w["starting_height_reset"] = reset_btn

        btn_row.addStretch()
        starting_height_vbox.addLayout(btn_row)

        vbox.addWidget(starting_height_box)

    def connect_signals(self, on_float, on_check, on_set_height, on_reset_height) -> None:
        for axis in ("x", "y", "z"):
            key = f"invert_{axis}"
            check: QCheckBox = self._w[key]  # type: ignore[assignment]
            check.checkStateChanged.connect(
                lambda state, k=key: on_check(k, state == Qt.CheckState.Checked)
            )

        for i in range(1, 5):
            key = f"preset_{i}"
            spin: NoScrollDoubleSpinBox = self._w[key]  # type: ignore[assignment]
            spin.valueChanged.connect(lambda v, k=key: on_float(k, v))

        height_spin: NoScrollDoubleSpinBox = self._w["starting_height"]  # type: ignore[assignment]
        height_spin.valueChanged.connect(lambda v: on_float("starting_height", v))

        self._w["starting_height_set_current"].clicked.connect(on_set_height)
        self._w["starting_height_reset"].clicked.connect(on_reset_height)

    def populate(self, s: MotionSystemSettings) -> None:
        for key, w in self._w.items():
            if isinstance(w, (NoScrollDoubleSpinBox, QCheckBox)):
                w.blockSignals(True)

        for axis in ("x", "y", "z"):
            check: QCheckBox = self._w[f"invert_{axis}"]  # type: ignore[assignment]
            check.setChecked(getattr(s, f"invert_{axis}", False))

        defaults_nm = [_mm_to_nm(mm) for mm in _DEFAULT_PRESETS_MM]
        presets_nm = list(getattr(s, "step_presets", defaults_nm))
        presets_nm = (presets_nm + defaults_nm)[:4]
        for i, nm in enumerate(presets_nm, start=1):
            spin: NoScrollDoubleSpinBox = self._w[f"preset_{i}"]  # type: ignore[assignment]
            spin.setValue(_nm_to_mm(nm))

        height_spin: NoScrollDoubleSpinBox = self._w["starting_height"]  # type: ignore[assignment]
        height_spin.setValue(_nm_to_mm(getattr(s, "starting_height_nm", 0)))

        for key, w in self._w.items():
            if isinstance(w, (NoScrollDoubleSpinBox, QCheckBox)):
                w.blockSignals(False)

    def snapshot(self, s: MotionSystemSettings) -> None:
        defaults_nm = [_mm_to_nm(mm) for mm in _DEFAULT_PRESETS_MM]
        presets_nm = list(getattr(s, "step_presets", defaults_nm))
        presets_nm = (presets_nm + defaults_nm)[:4]

        self._saved = {
            "invert_x":       getattr(s, "invert_x", False),
            "invert_y":       getattr(s, "invert_y", False),
            "invert_z":       getattr(s, "invert_z", False),
            "preset_1":       _nm_to_mm(presets_nm[0]),
            "preset_2":       _nm_to_mm(presets_nm[1]),
            "preset_3":       _nm_to_mm(presets_nm[2]),
            "preset_4":       _nm_to_mm(presets_nm[3]),
            "starting_height": _nm_to_mm(getattr(s, "starting_height_nm", 0)),
        }

    def apply_float_to_live(self, key: str, value: float, s: MotionSystemSettings) -> None:
        if key.startswith("preset_"):
            idx = int(key[-1]) - 1
            defaults_nm = [_mm_to_nm(mm) for mm in _DEFAULT_PRESETS_MM]
            presets: list[int] = list(getattr(s, "step_presets", defaults_nm))
            presets = (presets + defaults_nm)[:4]
            presets[idx] = _mm_to_nm(value)
            s.step_presets = presets  # type: ignore[attr-defined]
        elif key == "starting_height":
            s.starting_height_nm = _mm_to_nm(value)  # type: ignore[attr-defined]

    def apply_check_to_live(self, key: str, value: bool, s: MotionSystemSettings) -> None:
        setattr(s, key, value)

    def set_height_from_current_position(self) -> None:
        motion = get_app_context().motion
        if motion is None:
            return
        position = motion.get_position()
        spin: NoScrollDoubleSpinBox = self._w["starting_height"]  # type: ignore[assignment]
        spin.setValue(_nm_to_mm(position.z))

    def reset_height(self) -> None:
        spin: NoScrollDoubleSpinBox = self._w["starting_height"]  # type: ignore[assignment]
        spin.setValue(0.0)

    def mark_float_field(self, key: str, current_value: float) -> None:
        saved = self._saved.get(key)
        changed = isinstance(saved, float) and abs(saved - current_value) > 1e-9
        self.mark_label(key, changed)

    def mark_check_field(self, key: str, current_value: bool) -> None:
        self.mark_label(key, self._saved.get(key) != current_value)

    def has_changes(self) -> bool:
        for axis in ("x", "y", "z"):
            key = f"invert_{axis}"
            check: QCheckBox = self._w[key]  # type: ignore[assignment]
            if self._saved.get(key) != check.isChecked():
                return True

        for i in range(1, 5):
            key = f"preset_{i}"
            spin: NoScrollDoubleSpinBox = self._w[key]  # type: ignore[assignment]
            saved = self._saved.get(key)
            if isinstance(saved, float) and abs(saved - spin.value()) > 1e-9:
                return True

        saved_h = self._saved.get("starting_height")
        height_spin: NoScrollDoubleSpinBox = self._w["starting_height"]  # type: ignore[assignment]
        if isinstance(saved_h, float) and abs(saved_h - height_spin.value()) > 1e-9:
            return True

        return False