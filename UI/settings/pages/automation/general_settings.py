from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QWidget,
)

from common.app_context import get_app_context
from motion.motion_config import MotionSystemSettings
from UI.settings.pages.shared import ORANGE, NoScrollDoubleSpinBox, NoScrollSpinBox


class GeneralSettingsWidget(QGroupBox):
    """General automation settings group (overlap percentages, capture timeout)."""

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
            form.addRow(QLabel(label_text), spin)

        timeout_spin = NoScrollSpinBox()
        timeout_spin.setMinimum(100)
        timeout_spin.setMaximum(60_000)
        timeout_spin.setSingleStep(100)
        timeout_spin.setSuffix(" ms")
        timeout_spin.setFixedWidth(130)
        timeout_spin.setToolTip("How long to wait for each image capture to complete before treating it as a failure.")
        self._w_int["capture_timeout_ms"] = timeout_spin
        form.addRow(QLabel("Capture timeout:"), timeout_spin)

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

        for w in self._w.values():
            w.blockSignals(False)
        for w in self._w_int.values():
            w.blockSignals(False)

    def snapshot(self, s: MotionSystemSettings) -> None:
        self._saved = {
            "overlap_x_pct":      s.automation.overlap_x_pct,
            "overlap_y_pct":      s.automation.overlap_y_pct,
            "capture_timeout_ms": s.automation.capture_timeout_ms,
        }

    def apply_to_live(self, key: str, value: object, type_: type) -> None:
        motion = get_app_context().motion
        if motion is None or motion.settings is None:
            return
        setattr(motion.settings.automation, key, type_(value))

    def has_changes(self) -> bool:
        return (
            self._saved.get("overlap_x_pct") != self._w["overlap_x_pct"].value()
            or self._saved.get("overlap_y_pct") != self._w["overlap_y_pct"].value()
            or self._saved.get("capture_timeout_ms") != self._w_int["capture_timeout_ms"].value()
        )

    def clear_orange(self) -> None:
        for w in self._w.values():
            w.setStyleSheet("")
        for w in self._w_int.values():
            w.setStyleSheet("")

    def mark_field(self, key: str, value: object) -> None:
        w = self._w.get(key) or self._w_int.get(key)
        if w:
            modified = self._saved.get(key) != value
            w.setStyleSheet(f"color: {ORANGE};" if modified else "")