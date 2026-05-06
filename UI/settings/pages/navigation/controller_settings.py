from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QLabel,
    QWidget,
)

from motion.motion_config import MotionSystemSettings
from UI.settings.pages.shared import NM_PER_MM, NoScrollDoubleSpinBox, NoScrollSpinBox, SettingsGroupBase


_DEFAULT_PRESETS_MM = (0.04, 0.4, 2.0, 10.0)


def _mm_to_nm(mm: float) -> int:
    return round(mm * NM_PER_MM)


def _nm_to_mm(nm: int) -> float:
    return nm / NM_PER_MM


class ControllerSettingsWidget(SettingsGroupBase):
    """Hardware-level controller parameters group box."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Controller", parent)
        self._w: dict[str, NoScrollSpinBox | NoScrollDoubleSpinBox] = {}
        self._saved: dict[str, object] = {}
        self._build()

    def _build(self) -> None:
        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        baud_spin = NoScrollSpinBox()
        baud_spin.setMinimum(1_200)
        baud_spin.setMaximum(3_000_000)
        baud_spin.setSingleStep(9_600)
        baud_spin.setFixedWidth(110)
        baud_spin.setToolTip("Serial baud rate for communication with the motion controller.")
        self._w["baud_rate"] = baud_spin
        form.addRow(self._register_label("baud_rate", QLabel("Baud rate:")), baud_spin)

        max_x_spin = NoScrollSpinBox()
        max_x_spin.setMinimum(1)
        max_x_spin.setMaximum(10_000)
        max_x_spin.setSuffix(" mm")
        max_x_spin.setFixedWidth(110)
        max_x_spin.setToolTip("Maximum travel distance of the X axis in millimetres.")
        self._w["max_x"] = max_x_spin
        form.addRow(self._register_label("max_x", QLabel("Max X:")), max_x_spin)

        max_y_spin = NoScrollSpinBox()
        max_y_spin.setMinimum(1)
        max_y_spin.setMaximum(10_000)
        max_y_spin.setSuffix(" mm")
        max_y_spin.setFixedWidth(110)
        max_y_spin.setToolTip("Maximum travel distance of the Y axis in millimetres.")
        self._w["max_y"] = max_y_spin
        form.addRow(self._register_label("max_y", QLabel("Max Y:")), max_y_spin)

        max_z_spin = NoScrollSpinBox()
        max_z_spin.setMinimum(1)
        max_z_spin.setMaximum(10_000)
        max_z_spin.setSuffix(" mm")
        max_z_spin.setFixedWidth(110)
        max_z_spin.setToolTip("Maximum travel distance of the Z axis in millimetres.")
        self._w["max_z"] = max_z_spin
        form.addRow(self._register_label("max_z", QLabel("Max Z:")), max_z_spin)

        step_spin = NoScrollDoubleSpinBox()
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
        self._w["step_size"] = step_spin
        form.addRow(self._register_label("step_size", QLabel("Step size:")), step_spin)

    def connect_signals(self, on_change) -> None:
        for key, widget in self._w.items():
            widget.valueChanged.connect(lambda v, k=key: on_change(k, v))

    def populate(self, s: MotionSystemSettings) -> None:
        for w in self._w.values():
            w.blockSignals(True)

        self._w["baud_rate"].setValue(s.baud_rate)
        self._w["max_x"].setValue(s.max_x)
        self._w["max_y"].setValue(s.max_y)
        self._w["max_z"].setValue(s.max_z)
        self._w["step_size"].setValue(_nm_to_mm(s.step_size))

        for w in self._w.values():
            w.blockSignals(False)

    def snapshot(self, s: MotionSystemSettings) -> None:
        self._saved = {
            "baud_rate": s.baud_rate,
            "max_x":     s.max_x,
            "max_y":     s.max_y,
            "max_z":     s.max_z,
            "step_size": _nm_to_mm(s.step_size),
        }

    def apply_to_live(self, key: str, value: object, s: MotionSystemSettings) -> None:
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

    def mark_field(self, key: str, current_value: object) -> None:
        saved = self._saved.get(key)
        if isinstance(saved, float) and isinstance(current_value, float):
            changed = abs(saved - current_value) > 1e-9
        else:
            changed = saved != current_value
        self.mark_label(key, changed)

    def has_changes(self) -> bool:
        checks = {
            "baud_rate": self._w["baud_rate"].value(),
            "max_x":     self._w["max_x"].value(),
            "max_y":     self._w["max_y"].value(),
            "max_z":     self._w["max_z"].value(),
        }
        for key, val in checks.items():
            if self._saved.get(key) != val:
                return True
        saved_step = self._saved.get("step_size")
        current_step = self._w["step_size"].value()
        if isinstance(saved_step, float) and abs(saved_step - current_step) > 1e-9:
            return True
        return False