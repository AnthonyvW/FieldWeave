from __future__ import annotations

import serial.tools.list_ports
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from motion.motion_config import MotionSystemSettings
from UI.settings.pages.shared import (
    NM_PER_MM,
    NoScrollComboBox,
    NoScrollDoubleSpinBox,
    NoScrollSpinBox,
    SettingsGroupBase,
)


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
        self._last_com_port: str = ""
        self._build()

    def _build(self) -> None:
        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._port_combo = NoScrollComboBox()
        self._port_combo.setMinimumWidth(220)
        self._port_combo.setToolTip(
            "Serial port of the motion controller.\n"
            "Auto-detect probes every port, starting with the last one that worked.\n"
            "Connect uses the selected port immediately and remembers it."
        )
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setToolTip("Reconnect the motion controller using the selected port.")
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Rescan for available serial ports.")
        refresh_btn.clicked.connect(self._refresh_ports)

        port_row = QWidget()
        port_layout = QHBoxLayout(port_row)
        port_layout.setContentsMargins(0, 0, 0, 0)
        port_layout.addWidget(self._port_combo)
        port_layout.addWidget(self._connect_btn)
        port_layout.addWidget(refresh_btn)
        port_layout.addStretch()
        form.addRow(self._register_label("com_port", QLabel("Serial port:")), port_row)

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

    def _current_port(self) -> str:
        return self._port_combo.currentData() or ""

    def _fill_port_combo(self, selected: str) -> None:
        self._port_combo.blockSignals(True)
        self._port_combo.clear()

        auto_label = "Auto-detect"
        if self._last_com_port:
            auto_label += f" (last: {self._last_com_port})"
        self._port_combo.addItem(auto_label, "")

        devices: list[str] = []
        # include_links adds the stable /dev/serial/by-id/... names on Linux, which
        # keep pointing at the same controller when ttyUSB/ttyACM numbers shuffle.
        for p in sorted(serial.tools.list_ports.comports(include_links=True), key=lambda p: p.device):
            devices.append(p.device)
            has_desc = p.description and p.description not in ("n/a", p.device)
            self._port_combo.addItem(f"{p.device} - {p.description}" if has_desc else p.device, p.device)

        if selected and selected not in devices:
            self._port_combo.addItem(f"{selected} (not detected)", selected)

        index = self._port_combo.findData(selected)
        self._port_combo.setCurrentIndex(max(index, 0))
        self._port_combo.blockSignals(False)

    def _refresh_ports(self) -> None:
        self._fill_port_combo(self._current_port())

    def selected_port(self) -> str:
        return self._current_port()

    def mark_port_saved(self, port: str) -> None:
        self._saved["com_port"] = port
        self.mark_label("com_port", False)

    def connect_signals(self, on_change, on_connect) -> None:
        self._connect_btn.clicked.connect(on_connect)
        self._port_combo.currentIndexChanged.connect(
            lambda _i: on_change("com_port", self._current_port())
        )
        for key, widget in self._w.items():
            widget.valueChanged.connect(lambda v, k=key: on_change(k, v))

    def populate(self, s: MotionSystemSettings) -> None:
        for w in self._w.values():
            w.blockSignals(True)

        self._last_com_port = s.last_com_port
        self._fill_port_combo(s.com_port)
        self._w["baud_rate"].setValue(s.baud_rate)
        self._w["max_x"].setValue(s.max_x)
        self._w["max_y"].setValue(s.max_y)
        self._w["max_z"].setValue(s.max_z)
        self._w["step_size"].setValue(_nm_to_mm(s.step_size))

        for w in self._w.values():
            w.blockSignals(False)

    def connection_changed(self, s: MotionSystemSettings) -> bool:
        """True if *s* differs from the snapshot in a way that requires reconnecting."""
        return s.com_port != self._saved.get("com_port") or s.baud_rate != self._saved.get("baud_rate")

    def snapshot(self, s: MotionSystemSettings) -> None:
        self._saved = {
            "com_port":  s.com_port,
            "baud_rate": s.baud_rate,
            "max_x":     s.max_x,
            "max_y":     s.max_y,
            "max_z":     s.max_z,
            "step_size": _nm_to_mm(s.step_size),
        }

    def apply_to_live(self, key: str, value: object, s: MotionSystemSettings) -> None:
        if key == "com_port":
            s.com_port = str(value)
        elif key == "baud_rate":
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
            "com_port":  self._current_port(),
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
