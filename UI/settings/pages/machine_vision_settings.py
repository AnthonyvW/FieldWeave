"""
machine_vision_settings.py

Settings page for the machine-vision pipeline.

Each vision algorithm gets its own QGroupBox section.  Currently only
Focus Detection is implemented; future algorithms (edge detection, object
detection, etc.) should each get their own _build_*_section() method and a
corresponding group box added in _build_ui().

Changes are applied to the manager immediately on every widget interaction
(live preview).  The Save button persists them to disk via
MachineVisionManager.save_settings().
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
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
    QSlider,
    QVBoxLayout,
    QWidget,
)

from common.app_context import get_app_context
from common.logger import info, error
from machine_vision.machine_vision_config import FocusDetectionSettings, MachineVisionSettings


class MachineVisionSettingsWidget(QWidget):
    """
    Full settings page for all machine-vision algorithms.

    Intended to be embedded in the application's settings dialog in the same
    way as CameraSettingsWidget.
    """

    def __init__(self, parent_dialog=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.parent_dialog = parent_dialog
        self._ctx = get_app_context()
        self._mv = self._ctx.machine_vision

        # Track unsaved state so the Save button can be enabled/disabled.
        self._has_unsaved_changes: bool = False

        # Widget references — populated in _build_focus_section().
        self._focus_widgets: dict[str, QWidget] = {}

        self._build_ui()
        self._populate_from_settings(self._mv.settings)

        # Reflect any live changes made by other code (e.g. from the overlay
        # legend "set ceiling" button that may be added later).
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
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)

        title = QLabel("Machine Vision")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #5f6368;")
        content_layout.addWidget(title)

        # ---- Algorithm sections ----------------------------------------
        # Add one group box per vision algorithm.  Future algorithms slot in
        # here as additional _build_*_section() calls.
        content_layout.addWidget(self._build_focus_section())

        content_layout.addStretch()

        # ---- Save button -----------------------------------------------
        btn_row = QHBoxLayout()
        self._save_btn = QPushButton("Save")
        self._save_btn.setEnabled(False)
        self._save_btn.setMaximumWidth(100)
        self._save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self._save_btn)
        btn_row.addStretch()
        content_layout.addLayout(btn_row)

        scroll.setWidget(content)
        root.addWidget(scroll)

        # Connect parent dialog save button if available.
        if self.parent_dialog and hasattr(self.parent_dialog, "save_btn"):
            self.parent_dialog.save_btn.clicked.connect(self._on_save)

    def _build_focus_section(self) -> QGroupBox:
        """Build the Focus Detection settings group box."""
        group = QGroupBox("Focus Detection")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # window_size — odd integer spinbox
        window_spin = QSpinBox()
        window_spin.setMinimum(3)
        window_spin.setMaximum(101)
        window_spin.setSingleStep(2)       # keep it odd
        window_spin.setFixedWidth(90)
        window_spin.setToolTip(
            "Side length (px) of the local variance window.  Must be odd.\n"
            "Larger values integrate more spatial context but reduce heatmap resolution."
        )
        window_spin.valueChanged.connect(self._on_window_size_changed)
        form.addRow("Window size:", window_spin)
        self._focus_widgets["window_size"] = window_spin

        # radius — float slider + spinbox
        form.addRow("Blur radius:", self._build_float_row(
            key="radius",
            min_val=0.0, max_val=32.0, decimals=1, step=0.5,
            tooltip=(
                "Gaussian/box blur radius (px) applied after the variance step.\n"
                "Spreads the focus signal to neighbouring regions.  0 = no blur."
            ),
            on_change=self._on_radius_changed,
        ))

        # threshold — float slider + spinbox
        form.addRow("Threshold:", self._build_float_row(
            key="threshold",
            min_val=0.0, max_val=500.0, decimals=1, step=1.0,
            tooltip=(
                "Raw variance values below this level are zeroed out before\n"
                "blurring, suppressing flat / textureless regions.  0 = disabled."
            ),
            on_change=self._on_threshold_changed,
        ))

        # overlay_alpha — float slider + spinbox
        form.addRow("Overlay alpha:", self._build_float_row(
            key="overlay_alpha",
            min_val=0.0, max_val=1.0, decimals=2, step=0.05,
            tooltip=(
                "Blend weight of the focus heatmap over the camera image.\n"
                "0.0 = camera image only;  1.0 = heatmap only."
            ),
            on_change=self._on_alpha_changed,
        ))

        # score_ceiling — float spinbox (no slider — value range is unbounded)
        ceiling_row = QHBoxLayout()
        ceiling_spin = QDoubleSpinBox()
        ceiling_spin.setMinimum(0.0)
        ceiling_spin.setMaximum(1_000_000.0)
        ceiling_spin.setDecimals(1)
        ceiling_spin.setSingleStep(10.0)
        ceiling_spin.setFixedWidth(110)
        ceiling_spin.setSpecialValueText("Auto (per-frame)")   # shown when value == 0
        ceiling_spin.setToolTip(
            "Fixed ceiling used to normalise the raw score map to [0, 1].\n"
            "\n"
            "When > 0, the same value is applied to every frame so the\n"
            "heatmap brightness is stable across a focus sweep.\n"
            "\n"
            "Set to 0 to use per-frame normalisation (always fills the full\n"
            "colour range, but prevents meaningful cross-frame comparison).\n"
            "\n"
            "Tip: focus sharply, read the 'raw max' value shown in the overlay\n"
            "legend, then enter that value here."
        )
        ceiling_spin.valueChanged.connect(self._on_ceiling_changed)
        ceiling_row.addWidget(ceiling_spin)

        reset_ceiling_btn = QPushButton("Reset to auto")
        reset_ceiling_btn.setMaximumWidth(110)
        reset_ceiling_btn.setToolTip("Set score ceiling back to 0 (per-frame normalisation).")
        reset_ceiling_btn.clicked.connect(lambda: ceiling_spin.setValue(0.0))
        ceiling_row.addWidget(reset_ceiling_btn)
        ceiling_row.addStretch()

        ceiling_container = QWidget()
        ceiling_container.setLayout(ceiling_row)
        form.addRow("Score ceiling:", ceiling_container)
        self._focus_widgets["score_ceiling"] = ceiling_spin

        # half_resolution — checkbox
        half_res_check = QCheckBox()
        half_res_check.setToolTip(
            "Process at half the input resolution for speed.\n"
            "The result is upscaled back to full resolution before display."
        )
        half_res_check.checkStateChanged.connect(self._on_half_resolution_changed)
        form.addRow("Half resolution:", half_res_check)
        self._focus_widgets["half_resolution"] = half_res_check

        return group

    def _build_float_row(
        self,
        key: str,
        min_val: float,
        max_val: float,
        decimals: int,
        step: float,
        tooltip: str,
        on_change,
    ) -> QWidget:
        """
        Build a linked horizontal slider + double spinbox for a float parameter.

        The slider maps [min_val, max_val] onto [0, 1000] integer steps.
        """
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(1000)
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        spin = QDoubleSpinBox()
        spin.setMinimum(min_val)
        spin.setMaximum(max_val)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setFixedWidth(90)
        spin.setToolTip(tooltip)
        slider.setToolTip(tooltip)

        span = max_val - min_val

        def _spin_to_slider(val: float) -> int:
            return int((val - min_val) / span * 1000) if span > 0 else 0

        def _slider_to_spin(pos: int) -> float:
            return min_val + (pos / 1000.0) * span

        def _on_spin_changed(val: float) -> None:
            slider.blockSignals(True)
            slider.setValue(_spin_to_slider(val))
            slider.blockSignals(False)
            on_change(val)

        def _on_slider_changed(pos: int) -> None:
            val = _slider_to_spin(pos)
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
            on_change(val)

        spin.valueChanged.connect(_on_spin_changed)
        slider.valueChanged.connect(_on_slider_changed)

        row.addWidget(slider)
        row.addWidget(spin)

        # Store both so _populate_from_settings can set them.
        self._focus_widgets[f"{key}_slider"] = slider
        self._focus_widgets[key] = spin

        return container

    # ------------------------------------------------------------------
    # Populate widgets from a settings object
    # ------------------------------------------------------------------

    def _populate_from_settings(self, settings: MachineVisionSettings) -> None:
        """Push all values from *settings* into the widgets without triggering saves."""
        self._block_focus_signals(True)
        try:
            f = settings.focus

            self._set_spin(self._focus_widgets.get("window_size"), f.window_size)

            self._set_float_row("radius", f.radius, 0.0, 32.0)
            self._set_float_row("threshold", f.threshold, 0.0, 500.0)
            self._set_float_row("overlay_alpha", f.overlay_alpha, 0.0, 1.0)

            self._set_spin(self._focus_widgets.get("score_ceiling"), f.score_ceiling)

            check = self._focus_widgets.get("half_resolution")
            if isinstance(check, QCheckBox):
                check.setChecked(f.half_resolution)
        finally:
            self._block_focus_signals(False)

        # After loading, nothing is "unsaved" relative to disk.
        self._set_unsaved(False)

    def _set_float_row(self, key: str, value: float, min_val: float, max_val: float) -> None:
        spin = self._focus_widgets.get(key)
        slider = self._focus_widgets.get(f"{key}_slider")
        span = max_val - min_val
        slider_pos = int((value - min_val) / span * 1000) if span > 0 else 0
        self._set_spin(spin, value)
        if isinstance(slider, QSlider):
            slider.setValue(slider_pos)

    def _set_spin(self, widget: QWidget | None, value: float | int) -> None:
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.setValue(value)

    def _block_focus_signals(self, block: bool) -> None:
        for w in self._focus_widgets.values():
            w.blockSignals(block)

    # ------------------------------------------------------------------
    # Widget change handlers — apply live, mark unsaved
    # ------------------------------------------------------------------

    def _on_window_size_changed(self, value: int) -> None:
        # Snap to nearest odd number.
        if value % 2 == 0:
            spin = self._focus_widgets.get("window_size")
            if isinstance(spin, QSpinBox):
                spin.blockSignals(True)
                spin.setValue(value + 1)
                spin.blockSignals(False)
            value += 1
        self._apply_focus_field("window_size", value)

    def _on_radius_changed(self, value: float) -> None:
        self._apply_focus_field("radius", value)

    def _on_threshold_changed(self, value: float) -> None:
        self._apply_focus_field("threshold", value)

    def _on_alpha_changed(self, value: float) -> None:
        self._apply_focus_field("overlay_alpha", value)

    def _on_ceiling_changed(self, value: float) -> None:
        self._apply_focus_field("score_ceiling", value)

    @Slot(int)
    def _on_half_resolution_changed(self, state: int) -> None:
        self._apply_focus_field("half_resolution", state == Qt.CheckState.Checked.value)

    def _apply_focus_field(self, field: str, value) -> None:
        """Copy current settings, update one focus field, apply live."""
        s = self._mv._copy_settings()
        setattr(s.focus, field, value)
        self._mv.apply_settings(s)
        self._set_unsaved(True)

    # ------------------------------------------------------------------
    # External settings change (e.g. from another widget or future API)
    # ------------------------------------------------------------------

    @Slot()
    def _on_settings_changed_externally(self) -> None:
        """
        Re-populate widgets when something outside this page changes the
        manager's settings (e.g. a future "capture ceiling" button on the
        focus overlay legend).
        """
        self._populate_from_settings(self._mv.settings)
        # Re-mark as unsaved since the new values haven't been saved yet.
        self._set_unsaved(True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    @Slot()
    def _on_save(self) -> None:
        try:
            self._mv.save_settings()
            self._set_unsaved(False)
            if self._ctx.toast:
                self._ctx.toast.success("Machine vision settings saved", duration=2000)
            info("Machine vision settings saved from settings page")
        except Exception as exc:
            error(f"Failed to save machine vision settings: {exc}")
            if self._ctx.toast:
                self._ctx.toast.error(f"Save failed: {exc}", duration=3000)

    # ------------------------------------------------------------------
    # Unsaved state helpers
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


def machine_vision_page(parent_dialog=None) -> QWidget:
    """Create and return the machine vision settings page widget."""
    return MachineVisionSettingsWidget(parent_dialog=parent_dialog)