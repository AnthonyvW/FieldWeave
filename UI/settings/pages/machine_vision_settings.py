"""
machine_vision_settings.py

Settings page for the machine-vision pipeline.

Design
------
- One QGroupBox per vision algorithm.  Currently: Focus Detection.
- Within Focus Detection a method dropdown (Tenengrad / Laplacian) swaps
  the visible parameter group so each method has its own tunable controls.
- Modified fields turn orange exactly like CameraSettingsWidget does.
- get_group_names() returns the top-level group names so SettingsDialog can
  add them as sidebar sub-items.
- Changes are applied to the manager live on every widget interaction.
  The Save button persists them to disk.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from common.app_context import get_app_context
from common.logger import info, error
from machine_vision.machine_vision_config import (
    LaplacianSettings,
    MachineVisionSettings,
    TenengradSettings,
    FOCUS_METHOD_TENENGRAD,
    FOCUS_METHOD_LAPLACIAN,
)

_ORANGE = "#FFA500"
_GREY = "#aaaaaa"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_float_row(
    key: str,
    widget_store: dict[str, QWidget],
    min_val: float,
    max_val: float,
    decimals: int,
    step: float,
    tooltip: str,
) -> tuple[QWidget, QDoubleSpinBox, QSlider]:
    """
    Build a slider + spinbox pair.

    Registers '<key>' (spinbox) and '<key>_slider' (slider) in widget_store.
    Returns (container, spinbox, slider).
    """
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(1000)
    slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    slider.setToolTip(tooltip)

    spin = QDoubleSpinBox()
    spin.setMinimum(min_val)
    spin.setMaximum(max_val)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setFixedWidth(90)
    spin.setToolTip(tooltip)

    span = max_val - min_val

    def _to_slider(val: float) -> int:
        return int((val - min_val) / span * 1000) if span > 0 else 0

    def _to_spin(pos: int) -> float:
        return min_val + (pos / 1000.0) * span

    # Mutual sync — the change handlers supplied by the caller are connected
    # after this function returns; these closures handle only the widget sync.
    spin.valueChanged.connect(lambda v: (slider.blockSignals(True), slider.setValue(_to_slider(v)), slider.blockSignals(False)))
    slider.valueChanged.connect(lambda p: (spin.blockSignals(True), spin.setValue(_to_spin(p)), spin.blockSignals(False)))

    row.addWidget(slider)
    row.addWidget(spin)

    widget_store[key] = spin
    widget_store[f"{key}_slider"] = slider
    return container, spin, slider


def _make_ceiling_row(
    widget_store: dict[str, QWidget],
    tooltip: str,
) -> QWidget:
    """
    Build the score ceiling row: a spinbox, an 'Auto' checkbox, and a
    'Reset' button.  When Auto is checked the spinbox is disabled.
    """
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)

    spin = QDoubleSpinBox()
    spin.setMinimum(0.0)
    spin.setMaximum(1_000_000.0)
    spin.setDecimals(1)
    spin.setSingleStep(10.0)
    spin.setFixedWidth(110)
    spin.setToolTip(tooltip)

    auto_check = QCheckBox("Auto")
    auto_check.setToolTip("When checked, normalise each frame independently.")

    reset_btn = QPushButton("Reset")
    reset_btn.setMaximumWidth(70)
    reset_btn.setToolTip("Reset ceiling to default (15.0).")
    reset_btn.clicked.connect(lambda: spin.setValue(15.0))

    auto_check.checkStateChanged.connect(
        lambda s: spin.setEnabled(s != Qt.CheckState.Checked.value)
    )

    row.addWidget(spin)
    row.addWidget(auto_check)
    row.addWidget(reset_btn)
    row.addStretch()

    widget_store["score_ceiling"] = spin
    widget_store["auto_ceiling"] = auto_check
    return container


# ---------------------------------------------------------------------------
# Method-specific parameter panels
# ---------------------------------------------------------------------------

class _TenengradPanel(QWidget):
    """Parameter controls for the Tenengrad method."""

    def __init__(self) -> None:
        super().__init__()
        self._w: dict[str, QWidget] = {}
        self._build()

    def _build(self) -> None:
        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # kernel_size — dropdown (only valid values)
        kernel_combo = QComboBox()
        for k in (1, 3, 5, 7):
            kernel_combo.addItem(str(k), k)
        kernel_combo.setFixedWidth(90)
        kernel_combo.setToolTip(
            "Sobel kernel size.  Larger kernels are less sensitive to noise "
            "but reduce spatial resolution of the edge response."
        )
        form.addRow("Kernel size:", kernel_combo)
        self._w["kernel_size"] = kernel_combo

        # radius
        container, spin, slider = _make_float_row(
            "radius", self._w, 0.0, 32.0, 1, 0.5,
            "Gaussian/box blur radius (px) applied after gradient magnitude.  0 = no blur.",
        )
        form.addRow("Blur radius:", container)

        # threshold
        container, spin, slider = _make_float_row(
            "threshold", self._w, 0.0, 500.0, 1, 1.0,
            "Gradient values below this level are zeroed out.  0 = disabled.",
        )
        form.addRow("Threshold:", container)

        # overlay_alpha
        container, spin, slider = _make_float_row(
            "overlay_alpha", self._w, 0.0, 1.0, 2, 0.05,
            "Heatmap blend weight over the camera image.  0 = image only; 1 = heatmap only.",
        )
        form.addRow("Overlay alpha:", container)

        # score_ceiling
        ceiling_container = _make_ceiling_row(
            self._w,
            "Fixed ceiling used to normalise the score map across frames.\n"
            "Focus sharply, note the 'raw max' value, enter it here.\n"
            "Check Auto to normalise each frame independently.",
        )
        form.addRow("Score ceiling:", ceiling_container)

        # half_resolution
        half_check = QCheckBox()
        half_check.setToolTip("Process at half resolution for speed; result upscaled before display.")
        form.addRow("Half resolution:", half_check)
        self._w["half_resolution"] = half_check

    @property
    def widgets(self) -> dict[str, QWidget]:
        return self._w


class _LaplacianPanel(QWidget):
    """Parameter controls for the Laplacian method."""

    def __init__(self) -> None:
        super().__init__()
        self._w: dict[str, QWidget] = {}
        self._build()

    def _build(self) -> None:
        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # window_size
        window_spin = QSpinBox()
        window_spin.setMinimum(3)
        window_spin.setMaximum(101)
        window_spin.setSingleStep(2)
        window_spin.setFixedWidth(90)
        window_spin.setToolTip(
            "Side length (px) of the local variance window.  Must be odd.\n"
            "Larger values integrate more context but reduce heatmap resolution."
        )
        form.addRow("Window size:", window_spin)
        self._w["window_size"] = window_spin

        # radius
        container, spin, slider = _make_float_row(
            "radius", self._w, 0.0, 32.0, 1, 0.5,
            "Gaussian/box blur radius (px) applied after the variance step.  0 = no blur.",
        )
        form.addRow("Blur radius:", container)

        # threshold
        container, spin, slider = _make_float_row(
            "threshold", self._w, 0.0, 500.0, 1, 1.0,
            "Variance values below this level are zeroed out.  0 = disabled.",
        )
        form.addRow("Threshold:", container)

        # overlay_alpha
        container, spin, slider = _make_float_row(
            "overlay_alpha", self._w, 0.0, 1.0, 2, 0.05,
            "Heatmap blend weight over the camera image.  0 = image only; 1 = heatmap only.",
        )
        form.addRow("Overlay alpha:", container)

        # score_ceiling
        ceiling_container = _make_ceiling_row(
            self._w,
            "Fixed ceiling used to normalise the score map across frames.\n"
            "Focus sharply, note the 'raw max' value, enter it here.\n"
            "Check Auto to normalise each frame independently.",
        )
        form.addRow("Score ceiling:", ceiling_container)

        # half_resolution
        half_check = QCheckBox()
        half_check.setToolTip("Process at half resolution for speed; result upscaled before display.")
        form.addRow("Half resolution:", half_check)
        self._w["half_resolution"] = half_check

    @property
    def widgets(self) -> dict[str, QWidget]:
        return self._w


# ---------------------------------------------------------------------------
# Main settings widget
# ---------------------------------------------------------------------------

class MachineVisionSettingsWidget(QWidget):
    """
    Full settings page for all machine-vision algorithms.

    Embedded in the application settings dialog.
    """

    # Group names exposed for SettingsDialog sidebar sub-items.
    _GROUP_NAMES = ["Focus Detection"]

    def __init__(self, parent_dialog=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.parent_dialog = parent_dialog
        self._mv = get_app_context().machine_vision

        self._has_unsaved_changes: bool = False

        # Saved baseline values for orange-on-modify tracking.
        # Key: "<section>.<field>"  e.g. "laplacian.radius"
        self._saved_values: dict[str, object] = {}

        # Registered group boxes for scroll-to support.
        self._group_boxes: dict[str, QGroupBox] = {}

        self._tenengrad_panel = _TenengradPanel()
        self._laplacian_panel = _LaplacianPanel()

        self._build_ui()
        self._populate_from_settings(self._mv.settings)
        self._connect_panel_signals()

        self._mv.settings_changed.connect(self._on_settings_changed_externally)

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
        content.setObjectName("MachineVisionSettingsContent")
        content.setStyleSheet("QWidget#MachineVisionSettingsContent { background: white; }")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(10)

        title = QLabel("Machine Vision")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #5f6368;")
        cl.addWidget(title)

        focus_group = self._build_focus_group()
        cl.addWidget(focus_group)
        self._group_boxes["Focus Detection"] = focus_group

        # Register group boxes with parent dialog for scroll-to support.
        if self.parent_dialog and hasattr(self.parent_dialog, "register_group_box"):
            self.parent_dialog.register_group_box("Machine Vision", "Focus Detection", focus_group)

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

    def _build_focus_group(self) -> QGroupBox:
        group = QGroupBox("Focus Detection")
        vbox = QVBoxLayout(group)

        # Method selector
        method_row = QFormLayout()
        method_row.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._method_combo = QComboBox()
        self._method_combo.addItem("Laplacian", FOCUS_METHOD_LAPLACIAN)
        self._method_combo.addItem("Tenengrad", FOCUS_METHOD_TENENGRAD)
        self._method_combo.setFixedWidth(160)
        self._method_combo.setToolTip(
            "Laplacian: local variance of the Laplacian — good general-purpose measure.\n"
            "Tenengrad: Sobel gradient magnitude — faster, slightly noisier."
        )
        method_row.addRow("Method:", self._method_combo)
        vbox.addLayout(method_row)

        # Stacked panels — one per method
        self._method_stack = QStackedWidget()
        self._method_stack.addWidget(self._laplacian_panel)   # index 0 = Laplacian
        self._method_stack.addWidget(self._tenengrad_panel)   # index 1 = Tenengrad

        vbox.addWidget(self._method_stack)

        self._method_combo.currentIndexChanged.connect(self._on_method_combo_changed)

        return group

    # ------------------------------------------------------------------
    # Signal connections for parameter panels
    # ------------------------------------------------------------------

    def _connect_panel_signals(self) -> None:
        """Connect every widget in each panel to its change handler."""
        self._connect_tenengrad_signals()
        self._connect_laplacian_signals()

    def _connect_tenengrad_signals(self) -> None:
        w = self._tenengrad_panel.widgets
        w["kernel_size"].currentIndexChanged.connect(
            lambda _: self._on_field_changed("tenengrad", "kernel_size", w["kernel_size"].currentData())
        )
        w["radius"].valueChanged.connect(
            lambda v: self._on_field_changed("tenengrad", "radius", v)
        )
        w["threshold"].valueChanged.connect(
            lambda v: self._on_field_changed("tenengrad", "threshold", v)
        )
        w["overlay_alpha"].valueChanged.connect(
            lambda v: self._on_field_changed("tenengrad", "overlay_alpha", v)
        )
        w["score_ceiling"].valueChanged.connect(
            lambda v: self._on_field_changed("tenengrad", "score_ceiling", v)
        )
        w["auto_ceiling"].checkStateChanged.connect(
            lambda s: self._on_field_changed("tenengrad", "auto_ceiling", s == Qt.CheckState.Checked.value)
        )
        w["half_resolution"].checkStateChanged.connect(
            lambda s: self._on_field_changed("tenengrad", "half_resolution", s == Qt.CheckState.Checked.value)
        )

    def _connect_laplacian_signals(self) -> None:
        w = self._laplacian_panel.widgets
        w["window_size"].valueChanged.connect(self._on_window_size_changed)
        w["radius"].valueChanged.connect(
            lambda v: self._on_field_changed("laplacian", "radius", v)
        )
        w["threshold"].valueChanged.connect(
            lambda v: self._on_field_changed("laplacian", "threshold", v)
        )
        w["overlay_alpha"].valueChanged.connect(
            lambda v: self._on_field_changed("laplacian", "overlay_alpha", v)
        )
        w["score_ceiling"].valueChanged.connect(
            lambda v: self._on_field_changed("laplacian", "score_ceiling", v)
        )
        w["auto_ceiling"].checkStateChanged.connect(
            lambda s: self._on_field_changed("laplacian", "auto_ceiling", s == Qt.CheckState.Checked.value)
        )
        w["half_resolution"].checkStateChanged.connect(
            lambda s: self._on_field_changed("laplacian", "half_resolution", s == Qt.CheckState.Checked.value)
        )

    # ------------------------------------------------------------------
    # Populate from settings
    # ------------------------------------------------------------------

    def _populate_from_settings(self, settings: MachineVisionSettings) -> None:
        """Push all values into widgets without triggering saves."""
        self._block_all_signals(True)
        try:
            f = settings.focus

            # Method combo
            idx = self._method_combo.findData(f.method)
            if idx >= 0:
                self._method_combo.setCurrentIndex(idx)
            self._method_stack.setCurrentIndex(
                0 if f.method == FOCUS_METHOD_LAPLACIAN else 1
            )

            self._populate_tenengrad(f.tenengrad)
            self._populate_laplacian(f.laplacian)
        finally:
            self._block_all_signals(False)

        self._snapshot_saved_values(settings)
        self._set_unsaved(False)

    def _populate_tenengrad(self, t: TenengradSettings) -> None:
        w = self._tenengrad_panel.widgets
        idx = w["kernel_size"].findData(t.kernel_size)
        if idx >= 0:
            w["kernel_size"].setCurrentIndex(idx)
        self._set_float_row(w, "radius", t.radius, 0.0, 32.0)
        self._set_float_row(w, "threshold", t.threshold, 0.0, 500.0)
        self._set_float_row(w, "overlay_alpha", t.overlay_alpha, 0.0, 1.0)
        w["score_ceiling"].setValue(t.score_ceiling)
        w["score_ceiling"].setEnabled(not t.auto_ceiling)
        w["auto_ceiling"].setChecked(t.auto_ceiling)
        w["half_resolution"].setChecked(t.half_resolution)

    def _populate_laplacian(self, lap: LaplacianSettings) -> None:
        w = self._laplacian_panel.widgets
        w["window_size"].setValue(lap.window_size)
        self._set_float_row(w, "radius", lap.radius, 0.0, 32.0)
        self._set_float_row(w, "threshold", lap.threshold, 0.0, 500.0)
        self._set_float_row(w, "overlay_alpha", lap.overlay_alpha, 0.0, 1.0)
        w["score_ceiling"].setValue(lap.score_ceiling)
        w["score_ceiling"].setEnabled(not lap.auto_ceiling)
        w["auto_ceiling"].setChecked(lap.auto_ceiling)
        w["half_resolution"].setChecked(lap.half_resolution)

    def _set_float_row(
        self,
        w: dict[str, QWidget],
        key: str,
        value: float,
        min_val: float,
        max_val: float,
    ) -> None:
        spin = w.get(key)
        slider = w.get(f"{key}_slider")
        if isinstance(spin, QDoubleSpinBox):
            spin.setValue(value)
        span = max_val - min_val
        if isinstance(slider, QSlider) and span > 0:
            slider.setValue(int((value - min_val) / span * 1000))

    # ------------------------------------------------------------------
    # Saved-value snapshot and orange tracking
    # ------------------------------------------------------------------

    def _snapshot_saved_values(self, settings: MachineVisionSettings) -> None:
        """Record current values as the saved baseline for orange tracking."""
        f = settings.focus
        self._saved_values = {
            "method": f.method,
            "tenengrad.kernel_size": f.tenengrad.kernel_size,
            "tenengrad.radius": f.tenengrad.radius,
            "tenengrad.threshold": f.tenengrad.threshold,
            "tenengrad.half_resolution": f.tenengrad.half_resolution,
            "tenengrad.overlay_alpha": f.tenengrad.overlay_alpha,
            "tenengrad.score_ceiling": f.tenengrad.score_ceiling,
            "tenengrad.auto_ceiling": f.tenengrad.auto_ceiling,
            "laplacian.window_size": f.laplacian.window_size,
            "laplacian.radius": f.laplacian.radius,
            "laplacian.threshold": f.laplacian.threshold,
            "laplacian.half_resolution": f.laplacian.half_resolution,
            "laplacian.overlay_alpha": f.laplacian.overlay_alpha,
            "laplacian.score_ceiling": f.laplacian.score_ceiling,
            "laplacian.auto_ceiling": f.laplacian.auto_ceiling,
        }

    def _check_modified(self, key: str, current_value: object) -> bool:
        saved = self._saved_values.get(key)
        if isinstance(saved, float) and isinstance(current_value, float):
            return abs(saved - current_value) > 1e-9
        return saved != current_value

    def _apply_orange(self, widget: QWidget, orange: bool) -> None:
        """Apply or clear orange styling on a single leaf widget."""
        color = _ORANGE if orange else ""
        if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
            widget.setStyleSheet(f"color: {color};" if orange else "")
        elif isinstance(widget, QCheckBox):
            widget.setStyleSheet(f"QCheckBox {{ color: {color}; }}" if orange else "")
        elif isinstance(widget, QComboBox):
            widget.setStyleSheet(f"QComboBox {{ color: {color}; }}" if orange else "")
        elif isinstance(widget, QSlider) and orange:
            widget.setStyleSheet("""
                QSlider::handle:horizontal {
                    background: #FFA500;
                    border: 1px solid #FFA500;
                    width: 18px;
                    margin: -2px 0;
                    border-radius: 3px;
                }
            """)
        elif isinstance(widget, QSlider):
            widget.setStyleSheet("")

    def _mark_field(self, section_field: str, widget_key: str, panel_widgets: dict[str, QWidget], current_value: object) -> None:
        """Check if field is modified and orange the relevant widget(s)."""
        orange = self._check_modified(section_field, current_value)
        w = panel_widgets.get(widget_key)
        if w:
            self._apply_orange(w, orange)
        slider = panel_widgets.get(f"{widget_key}_slider")
        if slider:
            self._apply_orange(slider, orange)

    # ------------------------------------------------------------------
    # Change handlers
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_method_combo_changed(self, index: int) -> None:
        method = self._method_combo.itemData(index)
        self._method_stack.setCurrentIndex(0 if method == FOCUS_METHOD_LAPLACIAN else 1)
        s = self._mv._copy_settings()
        s.focus.method = method
        self._mv.apply_settings(s)
        orange = self._check_modified("method", method)
        self._apply_orange(self._method_combo, orange)
        self._set_unsaved(True)

    def _on_window_size_changed(self, value: int) -> None:
        # Snap to nearest odd.
        if value % 2 == 0:
            w = self._laplacian_panel.widgets["window_size"]
            w.blockSignals(True)
            w.setValue(value + 1)
            w.blockSignals(False)
            value += 1
        self._on_field_changed("laplacian", "window_size", value)

    def _on_field_changed(self, section: str, field: str, value: object) -> None:
        """Apply the changed field to the manager and update orange state."""
        s = self._mv._copy_settings()
        target = s.focus.tenengrad if section == "tenengrad" else s.focus.laplacian
        setattr(target, field, value)
        self._mv.apply_settings(s)

        panel = self._tenengrad_panel if section == "tenengrad" else self._laplacian_panel
        self._mark_field(f"{section}.{field}", field, panel.widgets, value)
        self._set_unsaved(True)

    # ------------------------------------------------------------------
    # External settings change
    # ------------------------------------------------------------------

    @Slot()
    def _on_settings_changed_externally(self) -> None:
        self._populate_from_settings(self._mv.settings)
        self._set_unsaved(True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    @Slot()
    def _on_save(self) -> None:
        ctx = get_app_context()
        try:
            self._mv.save_settings()
            self._snapshot_saved_values(self._mv.settings)
            self._clear_all_orange()
            self._set_unsaved(False)
            if ctx.toast:
                ctx.toast.success("Machine vision settings saved", duration=2000)
            info("Machine vision settings saved")
        except Exception as exc:
            error(f"Failed to save machine vision settings: {exc}")
            if ctx.toast:
                ctx.toast.error(f"Save failed: {exc}", duration=3000)

    def _clear_all_orange(self) -> None:
        for panel in (self._tenengrad_panel, self._laplacian_panel):
            for w in panel.widgets.values():
                self._apply_orange(w, False)
        self._apply_orange(self._method_combo, False)

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
                self.parent_dialog.set_category_modified("Machine Vision", has_changes)

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
        self._method_combo.blockSignals(block)
        for panel in (self._tenengrad_panel, self._laplacian_panel):
            for w in panel.widgets.values():
                w.blockSignals(block)


def machine_vision_page(parent_dialog=None) -> QWidget:
    """Create and return the machine vision settings page widget."""
    return MachineVisionSettingsWidget(parent_dialog=parent_dialog)