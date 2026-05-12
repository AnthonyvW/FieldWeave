from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QWidget,
)

from common.app_context import get_app_context
from motion.motion_config import MotionSystemSettings
from UI.settings.pages.shared import NoScrollDoubleSpinBox, NoScrollSpinBox, SettingsGroupBase


class GeneralSettingsWidget(SettingsGroupBase):
    """General automation settings group (overlap percentages, capture timeout, settle times)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("General", parent)
        self._w: dict[str, NoScrollDoubleSpinBox] = {}
        self._w_int: dict[str, NoScrollSpinBox] = {}
        self._saved: dict[str, object] = {}
        self._build()

    def _build(self) -> None:
        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        for key, label_text, tooltip in (
            ("overlap_x_pct", "X overlap (%):", "Fraction of each frame that overlaps the next along X (0-100)."),
            ("overlap_y_pct", "Y overlap (%):", "Fraction of each frame that overlaps the next along Y (0-100)."),
        ):
            spin = NoScrollDoubleSpinBox()
            spin.setMinimum(0.0)
            spin.setMaximum(100.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(1)
            spin.setFixedWidth(130)
            spin.setToolTip(tooltip)
            self._w[key] = spin
            form.addRow(self._register_label(key, QLabel(label_text)), spin)

        timeout_spin = NoScrollSpinBox()
        timeout_spin.setMinimum(100)
        timeout_spin.setMaximum(60_000)
        timeout_spin.setSingleStep(100)
        timeout_spin.setSuffix(" ms")
        timeout_spin.setFixedWidth(130)
        timeout_spin.setToolTip("How long to wait for each image capture to complete before treating it as a failure.")
        self._w_int["capture_timeout_ms"] = timeout_spin
        form.addRow(self._register_label("capture_timeout_ms", QLabel("Capture timeout:")), timeout_spin)

        form.addRow(QLabel("Settle times:"), QLabel("Delay after a move before the next operation."))

        for key, label_text, tooltip in (
            ("settle_x_ms",      "X settle (ms):",      "Settle time after a move that changes X, in milliseconds."),
            ("settle_y_ms",      "Y settle (ms):",      "Settle time after a move that changes Y, in milliseconds."),
            ("settle_z_ms",      "Z settle (ms):",      "Settle time after a Z-only move before triggering the camera, in milliseconds."),
            ("settle_travel_ms", "Travel settle (ms):", "Settle time after a multi-axis travel move (e.g. moving to a new slot or grid position), in milliseconds."),
        ):
            spin = NoScrollSpinBox()
            spin.setMinimum(0)
            spin.setMaximum(10_000)
            spin.setSingleStep(10)
            spin.setSuffix(" ms")
            spin.setFixedWidth(130)
            spin.setToolTip(tooltip)
            self._w_int[key] = spin
            form.addRow(self._register_label(key, QLabel(label_text)), spin)

    def connect_signals(self, on_changed) -> None:
        for key, spin in self._w.items():
            spin.valueChanged.connect(lambda v, k=key: on_changed(k, v, float))
        for key, spin in self._w_int.items():
            spin.valueChanged.connect(lambda v, k=key: on_changed(k, v, int))

    def populate(self, s: MotionSystemSettings) -> None:
        for w in self._w.values():
            w.blockSignals(True)
        for w in self._w_int.values():
            w.blockSignals(True)

        self._w["overlap_x_pct"].setValue(s.automation.overlap_x_pct)
        self._w["overlap_y_pct"].setValue(s.automation.overlap_y_pct)
        self._w_int["capture_timeout_ms"].setValue(s.automation.capture_timeout_ms)
        self._w_int["settle_x_ms"].setValue(s.automation.settle_x_ms)
        self._w_int["settle_y_ms"].setValue(s.automation.settle_y_ms)
        self._w_int["settle_z_ms"].setValue(s.automation.settle_z_ms)
        self._w_int["settle_travel_ms"].setValue(s.automation.settle_travel_ms)

        for w in self._w.values():
            w.blockSignals(False)
        for w in self._w_int.values():
            w.blockSignals(False)

    def snapshot(self) -> None:
        self._saved = {
            "overlap_x_pct":      self._w["overlap_x_pct"].value(),
            "overlap_y_pct":      self._w["overlap_y_pct"].value(),
            "capture_timeout_ms": self._w_int["capture_timeout_ms"].value(),
            "settle_x_ms":        self._w_int["settle_x_ms"].value(),
            "settle_y_ms":        self._w_int["settle_y_ms"].value(),
            "settle_z_ms":        self._w_int["settle_z_ms"].value(),
            "settle_travel_ms":   self._w_int["settle_travel_ms"].value(),
        }

    def apply_to_live(self, key: str, value: object, type_: type) -> None:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return
        setattr(motion.settings.automation, key, type_(value))

    def has_changes(self) -> bool:
        return any(
            self._saved.get(key) != self._w_int[key].value()
            for key in ("capture_timeout_ms", "settle_x_ms", "settle_y_ms", "settle_z_ms", "settle_travel_ms")
        ) or (
            self._saved.get("overlap_x_pct") != self._w["overlap_x_pct"].value()
            or self._saved.get("overlap_y_pct") != self._w["overlap_y_pct"].value()
        )

    def mark_field(self, key: str, value: object) -> None:
        self.mark_label(key, self._saved.get(key) != value)