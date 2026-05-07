"""
machine_vision_settings.py

Settings page for the machine-vision pipeline.

Design
------
- One QGroupBox per vision algorithm.  Currently: Focus Detection.
- Within Focus Detection a method dropdown (Tenengrad / Laplacian) swaps
  the visible parameter group so each method has its own tunable controls.
- A shared "Focus Region" group lets the user restrict analysis to a
  rectangular inset defined by four edge margins (% of image dimension).
- Modified fields turn orange on their labels via LabelTrackerMixin from shared.py.
- get_group_names() returns the top-level group names so SettingsDialog can
  add them as sidebar sub-items.
- Changes are applied to the manager live on every widget interaction.
  The Save button persists them to disk.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from common.app_context import get_app_context
from common.logger import info, error
from machine_vision.machine_vision_config import (
    BackgroundDetectionSettings,
    FocusRegionSettings,
    InspectCalibrationSettings,
    LaplacianSettings,
    MachineVisionSettings,
    RedMarkDetectionSettings,
    TenengradSettings,
    FOCUS_METHOD_TENENGRAD,
    FOCUS_METHOD_LAPLACIAN,
)
from motion.models import Position
from UI.settings.pages.shared import NM_PER_MM, NoScrollDoubleSpinBox, NoScrollSpinBox, LabelTrackerMixin

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
) -> tuple[QWidget, NoScrollDoubleSpinBox, QSlider]:
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

    spin = NoScrollDoubleSpinBox()
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

    spin = NoScrollDoubleSpinBox()
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
        lambda s: spin.setEnabled(s != Qt.CheckState.Checked)
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

class _TenengradPanel(LabelTrackerMixin, QWidget):
    """Parameter controls for the Tenengrad method."""

    def __init__(self) -> None:
        super().__init__()
        self._labels: dict[str, QLabel] = {}
        self._w: dict[str, QWidget] = {}
        self._build()

    def _build(self) -> None:
        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        kernel_combo = QComboBox()
        for k in (1, 3, 5, 7):
            kernel_combo.addItem(str(k), k)
        kernel_combo.setFixedWidth(90)
        kernel_combo.setToolTip(
            "Sobel kernel size.  Larger kernels are less sensitive to noise "
            "but reduce spatial resolution of the edge response."
        )
        form.addRow(self._register_label("kernel_size", QLabel("Kernel size:")), kernel_combo)
        self._w["kernel_size"] = kernel_combo

        container, spin, slider = _make_float_row(
            "radius", self._w, 0.0, 32.0, 1, 0.5,
            "Gaussian/box blur radius (px) applied after gradient magnitude.  0 = no blur.",
        )
        form.addRow(self._register_label("radius", QLabel("Blur radius:")), container)

        container, spin, slider = _make_float_row(
            "threshold", self._w, 0.0, 500.0, 1, 1.0,
            "Gradient values below this level are zeroed out.  0 = disabled.",
        )
        form.addRow(self._register_label("threshold", QLabel("Threshold:")), container)

        container, spin, slider = _make_float_row(
            "overlay_alpha", self._w, 0.0, 1.0, 2, 0.05,
            "Heatmap blend weight over the camera image.  0 = image only; 1 = heatmap only.",
        )
        form.addRow(self._register_label("overlay_alpha", QLabel("Overlay alpha:")), container)

        ceiling_container = _make_ceiling_row(
            self._w,
            "Fixed ceiling used to normalise the score map across frames.\n"
            "Focus sharply, note the 'raw max' value, enter it here.\n"
            "Check Auto to normalise each frame independently.",
        )
        form.addRow(self._register_label("score_ceiling", QLabel("Score ceiling:")), ceiling_container)

        half_check = QCheckBox()
        half_check.setToolTip("Process at half resolution for speed; result upscaled before display.")
        form.addRow(self._register_label("half_resolution", QLabel("Half resolution:")), half_check)
        self._w["half_resolution"] = half_check

    @property
    def widgets(self) -> dict[str, QWidget]:
        return self._w


class _LaplacianPanel(LabelTrackerMixin, QWidget):
    """Parameter controls for the Laplacian method."""

    def __init__(self) -> None:
        super().__init__()
        self._labels: dict[str, QLabel] = {}
        self._w: dict[str, QWidget] = {}
        self._build()

    def _build(self) -> None:
        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        window_spin = NoScrollSpinBox()
        window_spin.setMinimum(3)
        window_spin.setMaximum(101)
        window_spin.setSingleStep(2)
        window_spin.setFixedWidth(90)
        window_spin.setToolTip(
            "Side length (px) of the local variance window.  Must be odd.\n"
            "Larger values integrate more context but reduce heatmap resolution."
        )
        form.addRow(self._register_label("window_size", QLabel("Window size:")), window_spin)
        self._w["window_size"] = window_spin

        container, spin, slider = _make_float_row(
            "radius", self._w, 0.0, 32.0, 1, 0.5,
            "Gaussian/box blur radius (px) applied after the variance step.  0 = no blur.",
        )
        form.addRow(self._register_label("radius", QLabel("Blur radius:")), container)

        container, spin, slider = _make_float_row(
            "threshold", self._w, 0.0, 500.0, 1, 1.0,
            "Variance values below this level are zeroed out.  0 = disabled.",
        )
        form.addRow(self._register_label("threshold", QLabel("Threshold:")), container)

        container, spin, slider = _make_float_row(
            "overlay_alpha", self._w, 0.0, 1.0, 2, 0.05,
            "Heatmap blend weight over the camera image.  0 = image only; 1 = heatmap only.",
        )
        form.addRow(self._register_label("overlay_alpha", QLabel("Overlay alpha:")), container)

        ceiling_container = _make_ceiling_row(
            self._w,
            "Fixed ceiling used to normalise the score map across frames.\n"
            "Focus sharply, note the 'raw max' value, enter it here.\n"
            "Check Auto to normalise each frame independently.",
        )
        form.addRow(self._register_label("score_ceiling", QLabel("Score ceiling:")), ceiling_container)

        half_check = QCheckBox()
        half_check.setToolTip("Process at half resolution for speed; result upscaled before display.")
        form.addRow(self._register_label("half_resolution", QLabel("Half resolution:")), half_check)
        self._w["half_resolution"] = half_check

    @property
    def widgets(self) -> dict[str, QWidget]:
        return self._w


# ---------------------------------------------------------------------------
# Focus Region panel (shared across methods)
# ---------------------------------------------------------------------------

class _FocusRegionPanel(LabelTrackerMixin, QWidget):
    """
    Controls for the focus region of interest.

    Each margin (left, right, top, bottom) is a percentage [0 – 50] of the
    image dimension to exclude from that edge.  When the 'Enabled' checkbox
    is unchecked all margin controls are disabled and the full frame is used.
    """

    def __init__(self) -> None:
        super().__init__()
        self._labels: dict[str, QLabel] = {}
        self._w: dict[str, QWidget] = {}
        self._build()

    def _build(self) -> None:
        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        enabled_check = QCheckBox()
        enabled_check.setToolTip(
            "When checked, restrict focus analysis to the rectangle defined "
            "by the four margins below.  Pixels outside are ignored."
        )
        form.addRow(self._register_label("enabled", QLabel("Enabled:")), enabled_check)
        self._w["enabled"] = enabled_check

        _MARGIN_TOOLTIP = (
            "Percentage of the image {edge} to exclude from focus analysis [0 – 50].\n"
            "Example: 10 means the {adj} 10 % of {axis} is masked out."
        )

        for key, label, edge, adj, axis in (
            ("left",   "Left margin %:",   "left edge",   "leftmost",   "columns"),
            ("right",  "Right margin %:",  "right edge",  "rightmost",  "columns"),
            ("top",    "Top margin %:",    "top edge",    "topmost",    "rows"),
            ("bottom", "Bottom margin %:", "bottom edge", "bottommost", "rows"),
        ):
            container, _spin, _slider = _make_float_row(
                key, self._w, 0.0, 50.0, 1, 0.5,
                _MARGIN_TOOLTIP.format(edge=edge, adj=adj, axis=axis),
            )
            form.addRow(self._register_label(key, QLabel(label)), container)

        def _toggle_margins(state: Qt.CheckState) -> None:
            active = state == Qt.CheckState.Checked
            for k in ("left", "right", "top", "bottom"):
                for suffix in ("", "_slider"):
                    w = self._w.get(f"{k}{suffix}")
                    if w is not None:
                        w.setEnabled(active)

        enabled_check.checkStateChanged.connect(_toggle_margins)
        _toggle_margins(Qt.CheckState.Unchecked)

    @property
    def widgets(self) -> dict[str, QWidget]:
        return self._w


# ---------------------------------------------------------------------------
# Inspection Calibration panel
# ---------------------------------------------------------------------------

class _InspectCalibrationModePanel(LabelTrackerMixin, QWidget):
    """
    Controls for one mode (preview or snap) of the inspection calibration.
    """

    def __init__(self, label: str) -> None:
        super().__init__()
        self._labels: dict[str, QLabel] = {}
        self._label = label
        self._w: dict[str, QWidget] = {}
        self._build()

    def _build(self) -> None:
        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        ds_container = QWidget()
        ds_row = QHBoxLayout(ds_container)
        ds_row.setContentsMargins(0, 0, 0, 0)

        ds_spin = NoScrollSpinBox()
        ds_spin.setMinimum(1)
        ds_spin.setMaximum(16)
        ds_spin.setSingleStep(1)
        ds_spin.setFixedWidth(70)
        ds_spin.setToolTip(
            f"Shrink each axis by this factor before processing ({self._label} mode).\n"
            "Check 'None' to disable downscaling entirely."
        )

        ds_none = QCheckBox("None")
        ds_none.setToolTip("When checked, downscaling is disabled (equivalent to factor 1).")

        def _toggle_downsample(state: Qt.CheckState) -> None:
            enabled = state != Qt.CheckState.Checked
            ds_spin.setEnabled(enabled)
            ds_spin.setStyleSheet("" if enabled else f"color: {_GREY}; background-color: #f0f0f0;")

        ds_none.checkStateChanged.connect(_toggle_downsample)
        _toggle_downsample(Qt.CheckState.Unchecked)

        ds_row.addWidget(ds_spin)
        ds_row.addWidget(ds_none)
        ds_row.addStretch()
        form.addRow(self._register_label("downsample", QLabel("Downsample:")), ds_container)
        self._w["downsample"] = ds_spin
        self._w["downsample_none"] = ds_none

        tick_spin = NoScrollSpinBox()
        tick_spin.setMinimum(1)
        tick_spin.setMaximum(2000)
        tick_spin.setSingleStep(10)
        tick_spin.setFixedWidth(90)
        tick_spin.setToolTip(
            "Minimum perpendicular pixel run (full-resolution px) required to\n"
            "count a candidate as a tick mark. Scaled by downsample internally."
        )
        form.addRow(self._register_label("tick_min_length", QLabel("Min tick length:")), tick_spin)
        self._w["tick_min_length"] = tick_spin

    @property
    def widgets(self) -> dict[str, QWidget]:
        return self._w


# ---------------------------------------------------------------------------
# Red Mark Detection panel
# ---------------------------------------------------------------------------

class _RedMarkPanel(LabelTrackerMixin, QWidget):
    """Parameter controls for the red registration-mark detection algorithm."""

    def __init__(self) -> None:
        super().__init__()
        self._labels: dict[str, QLabel] = {}
        self._w: dict[str, QWidget] = {}
        self._build()

    def _build(self) -> None:
        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        scale_spin = NoScrollSpinBox()
        scale_spin.setMinimum(1)
        scale_spin.setMaximum(16)
        scale_spin.setSingleStep(1)
        scale_spin.setFixedWidth(90)
        scale_spin.setToolTip(
            "Downsample factor applied before processing.\n"
            "The image is resized to 1/scale in each dimension.\n"
            "Higher values reduce processing time at the cost of precision."
        )
        form.addRow(self._register_label("scale", QLabel("Scale:")), scale_spin)
        self._w["scale"] = scale_spin

        kernel_spin = NoScrollSpinBox()
        kernel_spin.setMinimum(1)
        kernel_spin.setMaximum(51)
        kernel_spin.setSingleStep(2)
        kernel_spin.setFixedWidth(90)
        kernel_spin.setToolTip(
            "Side length of the elliptical morphological opening kernel.\n"
            "Larger values remove more noise but may break up small marks."
        )
        form.addRow(self._register_label("open_kernel_size", QLabel("Open kernel size:")), kernel_spin)
        self._w["open_kernel_size"] = kernel_spin

        min_area_spin = NoScrollSpinBox()
        min_area_spin.setMinimum(0)
        min_area_spin.setMaximum(100_000)
        min_area_spin.setSingleStep(50)
        min_area_spin.setFixedWidth(90)
        min_area_spin.setToolTip("Minimum blob area in pixels. Smaller blobs are discarded as noise.")
        form.addRow(self._register_label("min_area", QLabel("Min area (px):")), min_area_spin)
        self._w["min_area"] = min_area_spin

        container, _spin, _slider = _make_float_row(
            "max_aspect_ratio", self._w, 1.0, 20.0, 1, 0.5,
            "Maximum ratio of longer to shorter side of a blob's bounding box.\n"
            "Elongated blobs (e.g. scratches) are discarded above this value.",
        )
        form.addRow(self._register_label("max_aspect_ratio", QLabel("Max aspect ratio:")), container)

        container, _spin, _slider = _make_float_row(
            "min_area_fraction", self._w, 0.0, 1.0, 2, 0.05,
            "Minimum blob area as a fraction of the largest surviving blob.\n"
            "Discards small stray marks when a large registration mark is present.",
        )
        form.addRow(self._register_label("min_area_fraction", QLabel("Min area fraction:")), container)

        hsv_label = QLabel("HSV Colour Range")
        hsv_label.setStyleSheet("font-weight: bold; margin-top: 4px;")
        form.addRow(hsv_label)

        hue_low_spin = NoScrollSpinBox()
        hue_low_spin.setMinimum(0)
        hue_low_spin.setMaximum(180)
        hue_low_spin.setSingleStep(1)
        hue_low_spin.setFixedWidth(90)
        hue_low_spin.setToolTip(
            "Lower hue boundary for the upper red range (160–180 in OpenCV HSV).\n"
            "Red wraps around 0/180, so two ranges are combined."
        )
        form.addRow(self._register_label("hue_low", QLabel("Hue low:")), hue_low_spin)
        self._w["hue_low"] = hue_low_spin

        hue_high_spin = NoScrollSpinBox()
        hue_high_spin.setMinimum(0)
        hue_high_spin.setMaximum(180)
        hue_high_spin.setSingleStep(1)
        hue_high_spin.setFixedWidth(90)
        hue_high_spin.setToolTip("Upper hue boundary for the lower red range (0–hue_high in OpenCV HSV).")
        form.addRow(self._register_label("hue_high", QLabel("Hue high:")), hue_high_spin)
        self._w["hue_high"] = hue_high_spin

        sat_spin = NoScrollSpinBox()
        sat_spin.setMinimum(0)
        sat_spin.setMaximum(255)
        sat_spin.setSingleStep(5)
        sat_spin.setFixedWidth(90)
        sat_spin.setToolTip("Minimum HSV saturation for a pixel to count as red [0–255].")
        form.addRow(self._register_label("sat_min", QLabel("Sat min:")), sat_spin)
        self._w["sat_min"] = sat_spin

        val_spin = NoScrollSpinBox()
        val_spin.setMinimum(0)
        val_spin.setMaximum(255)
        val_spin.setSingleStep(5)
        val_spin.setToolTip("Minimum HSV value for a pixel to count as red [0–255].")
        val_spin.setFixedWidth(90)
        form.addRow(self._register_label("val_min", QLabel("Val min:")), val_spin)
        self._w["val_min"] = val_spin

        smooth_label = QLabel("Centroid Smoothing")
        smooth_label.setStyleSheet("font-weight: bold; margin-top: 4px;")
        form.addRow(smooth_label)

        container, _spin, _slider = _make_float_row(
            "smoothing_alpha", self._w, 0.0, 1.0, 2, 0.05,
            "EMA weight for the stabilised centroid position [0.0–1.0].\n"
            "Lower values give more smoothing; higher values track faster.",
        )
        form.addRow(self._register_label("smoothing_alpha", QLabel("Smoothing alpha:")), container)

        container, _spin, _slider = _make_float_row(
            "deadband_px", self._w, 0.0, 50.0, 1, 0.5,
            "Changes smaller than this (in pixels) are ignored to prevent jitter.",
        )
        form.addRow(self._register_label("deadband_px", QLabel("Deadband (px):")), container)

        container, _spin, _slider = _make_float_row(
            "max_step_px", self._w, 0.0, 200.0, 1, 1.0,
            "Maximum centroid movement per frame in pixels (slew-rate limit).",
        )
        form.addRow(self._register_label("max_step_px", QLabel("Max step (px):")), container)

        container, _spin, _slider = _make_float_row(
            "jump_threshold_px", self._w, 0.0, 500.0, 1, 5.0,
            "If the raw centroid moves more than this many pixels from the smoothed\n"
            "value in a single frame, the smoothed value jumps directly to the raw value.",
        )
        form.addRow(self._register_label("jump_threshold_px", QLabel("Jump threshold (px):")), container)

        cluster_label = QLabel("Side Cluster")
        cluster_label.setStyleSheet("font-weight: bold; margin-top: 4px;")
        form.addRow(cluster_label)

        container, _spin, _slider = _make_float_row(
            "side_cluster_fraction", self._w, 0.0, 1.0, 2, 0.05,
            "Fraction of marks that must be within side_cluster_margin of one\n"
            "horizontal edge before the line orientation switches to 'horizontal'.",
        )
        form.addRow(self._register_label("side_cluster_fraction", QLabel("Cluster fraction:")), container)

        container, _spin, _slider = _make_float_row(
            "side_cluster_margin", self._w, 0.0, 0.45, 2, 0.01,
            "Fraction of image width defining the left/right edge zones used\n"
            "by the side-cluster test.",
        )
        form.addRow(self._register_label("side_cluster_margin", QLabel("Cluster margin:")), container)

    @property
    def widgets(self) -> dict[str, QWidget]:
        return self._w


# ---------------------------------------------------------------------------
# Background Detection panel
# ---------------------------------------------------------------------------

class _BackgroundPanel(LabelTrackerMixin, QWidget):
    """Parameter controls for the background detection algorithm."""

    def __init__(self) -> None:
        super().__init__()
        self._labels: dict[str, QLabel] = {}
        self._w: dict[str, QWidget] = {}
        self._build()

    def _build(self) -> None:
        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        val_median_spin = NoScrollSpinBox()
        val_median_spin.setMinimum(0)
        val_median_spin.setMaximum(255)
        val_median_spin.setSingleStep(5)
        val_median_spin.setFixedWidth(90)
        val_median_spin.setToolTip(
            "Maximum median HSV brightness to classify a frame as background [0–255].\n"
            "The plastic surface is dark-to-mid grey; bright surfaces like paper are rejected."
        )
        form.addRow(self._register_label("val_median_max", QLabel("Median max:")), val_median_spin)
        self._w["val_median_max"] = val_median_spin

        container, _spin, _slider = _make_float_row(
            "val_std_max", self._w, 0.0, 128.0, 1, 1.0,
            "Maximum standard deviation of HSV brightness to classify a frame as background [0–255].\n"
            "The plastic surface is highly uniform; textured materials are rejected.",
        )
        form.addRow(self._register_label("val_std_max", QLabel("Std max:")), container)

        scale_spin = NoScrollSpinBox()
        scale_spin.setMinimum(1)
        scale_spin.setMaximum(16)
        scale_spin.setSingleStep(1)
        scale_spin.setFixedWidth(90)
        scale_spin.setToolTip(
            "Downsample factor applied before computing statistics.\n"
            "Higher values reduce computation with negligible effect on accuracy."
        )
        form.addRow(self._register_label("scale", QLabel("Scale:")), scale_spin)
        self._w["scale"] = scale_spin

    @property
    def widgets(self) -> dict[str, QWidget]:
        return self._w


class MachineVisionSettingsWidget(LabelTrackerMixin, QWidget):
    """
    Full settings page for all machine-vision algorithms.

    Embedded in the application settings dialog.
    """

    _GROUP_NAMES = ["Focus Detection", "Camera Calibration", "Inspection Calibration", "Red Mark Detection", "Background Detection"]

    def __init__(self, parent_dialog=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._labels: dict[str, QLabel] = {}
        self.parent_dialog = parent_dialog
        self._mv = get_app_context().machine_vision

        self._has_unsaved_changes: bool = False
        self._saved_values: dict[str, object] = {}
        self._group_boxes: dict[str, QGroupBox] = {}

        self._tenengrad_panel = _TenengradPanel()
        self._laplacian_panel = _LaplacianPanel()
        self._focus_region_panel = _FocusRegionPanel()
        self._inspect_preview_panel = _InspectCalibrationModePanel("preview")
        self._inspect_snap_panel = _InspectCalibrationModePanel("snap")
        self._red_mark_panel = _RedMarkPanel()
        self._background_panel = _BackgroundPanel()

        self._build_ui()
        self._populate_from_settings(self._mv.settings)
        self._connect_panel_signals()

        self._external_state: dict[str, object] = self._read_external_state()
        self._idle_poll_timer = QTimer(self)
        self._idle_poll_timer.setInterval(1000)
        self._idle_poll_timer.timeout.connect(self._poll_external_state)
        self._idle_poll_timer.start()

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

        camera_cal_group = self._build_camera_calibration_group()
        cl.addWidget(camera_cal_group)
        self._group_boxes["Camera Calibration"] = camera_cal_group

        inspect_group = self._build_inspect_calibration_group()
        cl.addWidget(inspect_group)
        self._group_boxes["Inspection Calibration"] = inspect_group

        red_mark_group = self._build_red_mark_group()
        cl.addWidget(red_mark_group)
        self._group_boxes["Red Mark Detection"] = red_mark_group

        background_group = self._build_background_group()
        cl.addWidget(background_group)
        self._group_boxes["Background Detection"] = background_group

        # Register group boxes with parent dialog for scroll-to support.
        if self.parent_dialog and hasattr(self.parent_dialog, "register_group_box"):
            self.parent_dialog.register_group_box("Machine Vision", "Focus Detection", focus_group)
            self.parent_dialog.register_group_box("Machine Vision", "Camera Calibration", camera_cal_group)
            self.parent_dialog.register_group_box("Machine Vision", "Inspection Calibration", inspect_group)
            self.parent_dialog.register_group_box("Machine Vision", "Red Mark Detection", red_mark_group)
            self.parent_dialog.register_group_box("Machine Vision", "Background Detection", background_group)

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

        self._refresh_inspection_position_display()
        self._refresh_camera_calibration_display()
        self._refresh_scale_calibration_display()

    def _build_camera_calibration_group(self) -> QGroupBox:
        group = QGroupBox("Camera Calibration")
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(6)

        # --- DPI sub-box ---
        dpi_box = QGroupBox("DPI")
        dpi_form = QFormLayout(dpi_box)
        dpi_form.setContentsMargins(6, 6, 6, 6)
        dpi_form.setSpacing(6)
        dpi_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        dpi_input_row = QWidget()
        dpi_input_layout = QHBoxLayout(dpi_input_row)
        dpi_input_layout.setContentsMargins(0, 0, 0, 0)
        dpi_input_layout.setSpacing(6)

        self._dpi_spin = NoScrollDoubleSpinBox()
        self._dpi_spin.setMinimum(1.0)
        self._dpi_spin.setMaximum(100_000.0)
        self._dpi_spin.setDecimals(2)
        self._dpi_spin.setSingleStep(10.0)
        self._dpi_spin.setFixedWidth(110)
        self._dpi_spin.setToolTip("Enter a DPI value to store as the known optical resolution.")

        self._dpi_apply_btn = QPushButton("Set DPI")
        self._dpi_apply_btn.setMaximumWidth(80)
        self._dpi_apply_btn.setToolTip("Save the entered DPI value to settings.")
        self._dpi_apply_btn.clicked.connect(self._on_dpi_apply_clicked)

        self._dpi_clear_btn = QPushButton("Clear DPI")
        self._dpi_clear_btn.setMaximumWidth(90)
        self._dpi_clear_btn.setToolTip("Remove the stored DPI value.")
        self._dpi_clear_btn.clicked.connect(self._on_dpi_clear_clicked)

        dpi_input_layout.addWidget(self._dpi_spin)
        dpi_input_layout.addWidget(self._dpi_apply_btn)
        dpi_input_layout.addWidget(self._dpi_clear_btn)
        dpi_input_layout.addStretch()

        self._dpi_label = self._register_label("dpi", QLabel("Current DPI:"))
        self._dpi_label.setObjectName("DpiLabel")
        dpi_form.addRow(self._dpi_label, dpi_input_row)

        self._dpi_spin.valueChanged.connect(self._on_dpi_spin_changed)

        self._dpi_status = QLabel("")
        self._dpi_status.setObjectName("DpiStatusLabel")
        self._dpi_status.setWordWrap(False)
        self._dpi_status.hide()
        dpi_form.addRow(self._dpi_status)

        vbox.addWidget(dpi_box)

        # --- Camera Space Calibration sub-box ---
        cam_space_box = QGroupBox("Camera Space Calibration")
        cam_space_vbox = QVBoxLayout(cam_space_box)
        cam_space_vbox.setContentsMargins(6, 6, 6, 6)
        cam_space_vbox.setSpacing(6)

        self._cam_space_label = QLabel("Calibration: Not set")
        self._cam_space_label.setObjectName("CamSpaceLabel")
        cam_space_vbox.addWidget(self._cam_space_label)

        cam_space_btn_row = QHBoxLayout()
        cam_space_btn_row.setSpacing(6)

        self._cam_space_clear_btn = QPushButton("Clear Calibration")
        self._cam_space_clear_btn.setToolTip("Remove the saved camera space calibration matrix.")
        self._cam_space_clear_btn.setEnabled(False)
        self._cam_space_clear_btn.clicked.connect(self._on_cam_space_clear_clicked)

        cam_space_btn_row.addWidget(self._cam_space_clear_btn)
        cam_space_btn_row.addStretch()
        cam_space_vbox.addLayout(cam_space_btn_row)

        self._cam_space_status = QLabel("")
        self._cam_space_status.setObjectName("CamSpaceStatusLabel")
        self._cam_space_status.setWordWrap(False)
        self._cam_space_status.hide()
        cam_space_vbox.addWidget(self._cam_space_status)

        vbox.addWidget(cam_space_box)

        return group

    def _refresh_camera_calibration_display(self) -> None:
        settings = self._mv.settings
        dpi = settings.dpi
        self._dpi_spin.blockSignals(True)
        if dpi is not None:
            self._dpi_spin.setValue(dpi)
        else:
            self._dpi_spin.setValue(self._dpi_spin.minimum())
        self._dpi_spin.blockSignals(False)

        cal = settings.camera_calibration.calibration
        if cal is not None:
            self._cam_space_label.setText("Calibration: Set")
            self._cam_space_clear_btn.setEnabled(True)
        else:
            self._cam_space_label.setText("Calibration: Not set")
            self._cam_space_clear_btn.setEnabled(False)

    def _on_dpi_spin_changed(self, value: float) -> None:
        saved = self._saved_values.get("dpi")
        if saved is None:
            orange = True
        else:
            orange = abs(saved - value) > 1e-9
        self.mark_label("dpi", orange)

    def _set_dpi_status(self, text: str) -> None:
        self._dpi_status.setText(text)
        self._dpi_status.setVisible(bool(text))

    def _set_cam_space_status(self, text: str) -> None:
        self._cam_space_status.setText(text)
        self._cam_space_status.setVisible(bool(text))

    def _on_dpi_apply_clicked(self) -> None:
        dpi = self._dpi_spin.value()
        self._mv.settings.dpi = dpi
        self._mv.save_settings()
        self._saved_values["dpi"] = dpi
        self._external_state["dpi"] = dpi
        self.mark_label("dpi", False)
        info(f"[MachineVisionSettings] DPI set to {dpi:.2f}")
        self._refresh_camera_calibration_display()
        self._set_dpi_status(f"DPI set to {dpi:.2f}")

    def _on_dpi_clear_clicked(self) -> None:
        self._mv.settings.dpi = None
        self._mv.save_settings()
        self._saved_values["dpi"] = None
        self._external_state["dpi"] = None
        self.mark_label("dpi", False)
        info("[MachineVisionSettings] DPI cleared")
        self._refresh_camera_calibration_display()
        self._set_dpi_status("DPI cleared.")

    def _on_cam_space_clear_clicked(self) -> None:
        self._mv.settings.camera_calibration.calibration = None
        self._mv.save_settings()
        self._external_state["camera_calibration"] = False
        info("[MachineVisionSettings] Camera space calibration cleared")
        self._refresh_camera_calibration_display()
        self._set_cam_space_status("Camera space calibration cleared.")

    def _refresh_scale_calibration_display(self) -> None:
        last = self._mv.settings.inspect_calibration.last_calibrated
        if last:
            self._scale_cal_label.setText(f"Last calibrated: {last}")
            self._scale_cal_clear_btn.setEnabled(True)
        else:
            self._scale_cal_label.setText("Last calibrated: Not set")
            self._scale_cal_clear_btn.setEnabled(False)

    def _set_scale_cal_status(self, text: str) -> None:
        self._scale_cal_status.setText(text)
        self._scale_cal_status.setVisible(bool(text))

    def _on_scale_cal_clear_clicked(self) -> None:
        self._mv.settings.inspect_calibration.last_calibrated = None
        self._mv.save_settings()
        self._external_state["last_calibrated"] = None
        info("[MachineVisionSettings] Scale calibration cleared")
        self._refresh_scale_calibration_display()
        self._set_scale_cal_status("Scale calibration cleared.")

    def _build_inspect_calibration_group(self) -> QGroupBox:
        group = QGroupBox("Inspection Calibration")
        vbox = QVBoxLayout(group)

        preview_box = QGroupBox("Preview Mode")
        preview_box.setToolTip(
            "Settings used during live preview.\n"
            "Speed-optimised; typically no downscaling."
        )
        preview_vbox = QVBoxLayout(preview_box)
        preview_vbox.setContentsMargins(6, 6, 6, 6)
        preview_vbox.addWidget(self._inspect_preview_panel)
        vbox.addWidget(preview_box)

        snap_box = QGroupBox("Snap Mode")
        snap_box.setToolTip(
            "Settings used for full-image captures.\n"
            "Accuracy-optimised; typically uses downscaling."
        )
        snap_vbox = QVBoxLayout(snap_box)
        snap_vbox.setContentsMargins(6, 6, 6, 6)
        snap_vbox.addWidget(self._inspect_snap_panel)
        vbox.addWidget(snap_box)

        position_box = QGroupBox("Inspection Position")
        position_box.setToolTip(
            "Saved stage position used as the starting point for inspection calibration."
        )
        position_vbox = QVBoxLayout(position_box)
        position_vbox.setContentsMargins(6, 6, 6, 6)
        position_vbox.setSpacing(6)

        self._inspection_pos_label = QLabel("Saved position: Not set")
        self._inspection_pos_label.setObjectName("CalSavedPosLabel")
        position_vbox.addWidget(self._inspection_pos_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._inspection_set_pos_btn = QPushButton("Set Position")
        self._inspection_set_pos_btn.setToolTip("Save the current stage XYZ as the inspection calibration position")
        self._inspection_set_pos_btn.clicked.connect(self._on_inspection_set_position_clicked)
        btn_row.addWidget(self._inspection_set_pos_btn)

        self._inspection_goto_pos_btn = QPushButton("Go to Position")
        self._inspection_goto_pos_btn.setToolTip("Move the stage to the saved inspection calibration position")
        self._inspection_goto_pos_btn.setEnabled(False)
        self._inspection_goto_pos_btn.clicked.connect(self._on_inspection_goto_position_clicked)
        btn_row.addWidget(self._inspection_goto_pos_btn)

        self._inspection_clear_pos_btn = QPushButton("Clear Position")
        self._inspection_clear_pos_btn.setToolTip("Remove the saved inspection calibration position")
        self._inspection_clear_pos_btn.setEnabled(False)
        self._inspection_clear_pos_btn.clicked.connect(self._on_inspection_clear_position_clicked)
        btn_row.addWidget(self._inspection_clear_pos_btn)

        btn_row.addStretch()
        position_vbox.addLayout(btn_row)

        self._inspection_pos_status = QLabel("")
        self._inspection_pos_status.setObjectName("CalStatusLabel")
        self._inspection_pos_status.setWordWrap(False)
        self._inspection_pos_status.hide()
        position_vbox.addWidget(self._inspection_pos_status)

        vbox.addWidget(position_box)

        # --- Scale Calibration sub-box ---
        scale_cal_box = QGroupBox("Scale Calibration")
        scale_cal_box.setToolTip(
            "The accepted inspection calibration result.\n"
            "Records when the calibration bar was last successfully measured."
        )
        scale_cal_vbox = QVBoxLayout(scale_cal_box)
        scale_cal_vbox.setContentsMargins(6, 6, 6, 6)
        scale_cal_vbox.setSpacing(6)

        self._scale_cal_label = QLabel("Last calibrated: Not set")
        self._scale_cal_label.setObjectName("ScaleCalLabel")
        scale_cal_vbox.addWidget(self._scale_cal_label)

        scale_cal_btn_row = QHBoxLayout()
        scale_cal_btn_row.setSpacing(6)

        self._scale_cal_clear_btn = QPushButton("Clear Calibration")
        self._scale_cal_clear_btn.setToolTip("Remove the saved inspection scale calibration result.")
        self._scale_cal_clear_btn.setEnabled(False)
        self._scale_cal_clear_btn.clicked.connect(self._on_scale_cal_clear_clicked)
        scale_cal_btn_row.addWidget(self._scale_cal_clear_btn)
        scale_cal_btn_row.addStretch()
        scale_cal_vbox.addLayout(scale_cal_btn_row)

        self._scale_cal_status = QLabel("")
        self._scale_cal_status.setObjectName("ScaleCalStatusLabel")
        self._scale_cal_status.setWordWrap(False)
        self._scale_cal_status.hide()
        scale_cal_vbox.addWidget(self._scale_cal_status)

        vbox.addWidget(scale_cal_box)

        return group

    def _build_red_mark_group(self) -> QGroupBox:
        group = QGroupBox("Red Mark Detection")
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.addWidget(self._red_mark_panel)
        return group

    def _build_background_group(self) -> QGroupBox:
        group = QGroupBox("Background Detection")
        group.setToolTip(
            "Parameters for classifying frames as black-plastic background.\n"
            "Detection uses the median and standard deviation of the HSV brightness channel."
        )
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.addWidget(self._background_panel)
        return group

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
        method_row.addRow(self._register_label("method", QLabel("Method:")), self._method_combo)
        vbox.addLayout(method_row)

        # Stacked panels — one per method
        self._method_stack = QStackedWidget()
        self._method_stack.addWidget(self._laplacian_panel)   # index 0 = Laplacian
        self._method_stack.addWidget(self._tenengrad_panel)   # index 1 = Tenengrad

        vbox.addWidget(self._method_stack)

        self._method_combo.currentIndexChanged.connect(self._on_method_combo_changed)

        # Focus region — shared, always visible below the method panels.
        region_box = QGroupBox("Focus Region")
        region_box.setToolTip(
            "Restrict focus analysis to a rectangular region of interest.\n"
            "Each margin is a percentage of the image dimension to exclude."
        )
        region_vbox = QVBoxLayout(region_box)
        region_vbox.setContentsMargins(6, 6, 6, 6)
        region_vbox.addWidget(self._focus_region_panel)
        vbox.addWidget(region_box)

        return group

    # ------------------------------------------------------------------
    # Signal connections for parameter panels
    # ------------------------------------------------------------------

    def _connect_panel_signals(self) -> None:
        """Connect every widget in each panel to its change handler."""
        self._connect_tenengrad_signals()
        self._connect_laplacian_signals()
        self._connect_focus_region_signals()
        self._connect_inspect_calibration_signals()
        self._connect_red_mark_signals()
        self._connect_background_signals()

    def _connect_background_signals(self) -> None:
        w = self._background_panel.widgets
        w["val_median_max"].valueChanged.connect(
            lambda v: self._on_background_changed("val_median_max", v)
        )
        w["val_std_max"].valueChanged.connect(
            lambda v: self._on_background_changed("val_std_max", v)
        )
        if "val_std_max_slider" in w:
            w["val_std_max_slider"].valueChanged.connect(
                lambda _: self._on_background_changed("val_std_max", w["val_std_max"].value())
            )
        w["scale"].valueChanged.connect(
            lambda v: self._on_background_changed("scale", v)
        )

    def _connect_red_mark_signals(self) -> None:
        w = self._red_mark_panel.widgets
        for field in ("scale", "open_kernel_size", "min_area", "hue_low", "hue_high", "sat_min", "val_min"):
            w[field].valueChanged.connect(
                lambda v, f=field: self._on_red_mark_changed(f, v)
            )
        for field in ("max_aspect_ratio", "min_area_fraction", "smoothing_alpha",
                      "deadband_px", "max_step_px", "jump_threshold_px",
                      "side_cluster_fraction", "side_cluster_margin"):
            w[field].valueChanged.connect(
                lambda v, f=field: self._on_red_mark_changed(f, v)
            )
            if f"{field}_slider" in w:
                w[f"{field}_slider"].valueChanged.connect(
                    lambda _, f=field: self._on_red_mark_changed(f, w[f].value())
                )

    def _connect_inspect_calibration_signals(self) -> None:
        for mode, panel in (("preview", self._inspect_preview_panel), ("snap", self._inspect_snap_panel)):
            w = panel.widgets
            w["tick_min_length"].valueChanged.connect(
                lambda v, m=mode: self._on_inspect_calibration_changed(m, "tick_min_length", v)
            )
            w["downsample"].valueChanged.connect(
                lambda v, m=mode: self._on_inspect_calibration_changed(m, "downsample", v)
            )
            w["downsample_none"].checkStateChanged.connect(
                lambda s, m=mode: self._on_inspect_calibration_changed(
                    m, "downsample_none", s == Qt.CheckState.Checked
                )
            )

    def _connect_tenengrad_signals(self) -> None:
        w = self._tenengrad_panel.widgets
        w["kernel_size"].currentIndexChanged.connect(
            lambda _: self._on_field_changed("tenengrad", "kernel_size", w["kernel_size"].currentData())
        )
        for field in ("radius", "threshold", "overlay_alpha", "score_ceiling"):
            w[field].valueChanged.connect(
                lambda v, f=field: self._on_field_changed("tenengrad", f, v)
            )
            if f"{field}_slider" in w:
                w[f"{field}_slider"].valueChanged.connect(
                    lambda _, f=field: self._on_field_changed("tenengrad", f, w[f].value())
                )
        w["auto_ceiling"].checkStateChanged.connect(
            lambda s: self._on_field_changed("tenengrad", "auto_ceiling", s == Qt.CheckState.Checked)
        )
        w["half_resolution"].checkStateChanged.connect(
            lambda s: self._on_field_changed("tenengrad", "half_resolution", s == Qt.CheckState.Checked)
        )

    def _connect_laplacian_signals(self) -> None:
        w = self._laplacian_panel.widgets
        w["window_size"].valueChanged.connect(self._on_window_size_changed)
        for field in ("radius", "threshold", "overlay_alpha", "score_ceiling"):
            w[field].valueChanged.connect(
                lambda v, f=field: self._on_field_changed("laplacian", f, v)
            )
            if f"{field}_slider" in w:
                w[f"{field}_slider"].valueChanged.connect(
                    lambda _, f=field: self._on_field_changed("laplacian", f, w[f].value())
                )
        w["auto_ceiling"].checkStateChanged.connect(
            lambda s: self._on_field_changed("laplacian", "auto_ceiling", s == Qt.CheckState.Checked)
        )
        w["half_resolution"].checkStateChanged.connect(
            lambda s: self._on_field_changed("laplacian", "half_resolution", s == Qt.CheckState.Checked)
        )

    def _connect_focus_region_signals(self) -> None:
        w = self._focus_region_panel.widgets
        w["enabled"].checkStateChanged.connect(
            lambda s: self._on_focus_region_changed("enabled", s == Qt.CheckState.Checked)
        )
        for key in ("left", "right", "top", "bottom"):
            # Connect both the spinbox and the slider.  The slider→spinbox sync
            # inside _make_float_row blocks the spinbox signal, so dragging the
            # slider would otherwise be silent.  Connecting the slider directly
            # ensures the handler fires from both input paths.  By the time the
            # slider's valueChanged reaches our lambda the sync closure has
            # already run, so spin.value() is already up to date — we read that
            # rather than converting the raw slider int ourselves.
            w[key].valueChanged.connect(
                lambda v, k=key: self._on_focus_region_changed(k, v)
            )
            w[f"{key}_slider"].valueChanged.connect(
                lambda _, k=key: self._on_focus_region_changed(k, w[k].value())
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
            self._populate_focus_region(f.focus_region)
            self._populate_inspect_calibration(settings.inspect_calibration)
            self._populate_red_mark(settings.red_mark)
            self._populate_background(settings.background)
        finally:
            self._block_all_signals(False)

        self._refresh_camera_calibration_display()
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

    def _populate_focus_region(self, fr: FocusRegionSettings) -> None:
        w = self._focus_region_panel.widgets
        w["enabled"].setChecked(fr.enabled)
        for key in ("left", "right", "top", "bottom"):
            self._set_float_row(w, key, getattr(fr, key), 0.0, 50.0)
        # Sync enabled state of margin controls.
        for k in ("left", "right", "top", "bottom"):
            for suffix in ("", "_slider"):
                widget = w.get(f"{k}{suffix}")
                if widget is not None:
                    widget.setEnabled(fr.enabled)

    def _populate_inspect_calibration(self, ic: InspectCalibrationSettings) -> None:
        for mode, panel, mode_settings in (
            ("preview", self._inspect_preview_panel, ic.preview),
            ("snap", self._inspect_snap_panel, ic.snap),
        ):
            w = panel.widgets
            is_none = mode_settings.downsample is None
            w["downsample_none"].setChecked(is_none)
            w["downsample"].setEnabled(not is_none)
            w["downsample"].setStyleSheet("" if not is_none else f"color: {_GREY}; background-color: #f0f0f0;")
            w["downsample"].setValue(mode_settings.downsample if mode_settings.downsample is not None else 1)
            w["tick_min_length"].setValue(mode_settings.tick_min_length)

    def _populate_red_mark(self, rm: RedMarkDetectionSettings) -> None:
        w = self._red_mark_panel.widgets
        w["scale"].setValue(rm.scale)
        w["open_kernel_size"].setValue(rm.open_kernel_size)
        w["min_area"].setValue(rm.min_area)
        self._set_float_row(w, "max_aspect_ratio", rm.max_aspect_ratio, 1.0, 20.0)
        self._set_float_row(w, "min_area_fraction", rm.min_area_fraction, 0.0, 1.0)
        w["hue_low"].setValue(rm.hue_low)
        w["hue_high"].setValue(rm.hue_high)
        w["sat_min"].setValue(rm.sat_min)
        w["val_min"].setValue(rm.val_min)
        self._set_float_row(w, "smoothing_alpha", rm.smoothing_alpha, 0.0, 1.0)
        self._set_float_row(w, "deadband_px", rm.deadband_px, 0.0, 50.0)
        self._set_float_row(w, "max_step_px", rm.max_step_px, 0.0, 200.0)
        self._set_float_row(w, "jump_threshold_px", rm.jump_threshold_px, 0.0, 500.0)
        self._set_float_row(w, "side_cluster_fraction", rm.side_cluster_fraction, 0.0, 1.0)
        self._set_float_row(w, "side_cluster_margin", rm.side_cluster_margin, 0.0, 0.45)

    def _populate_background(self, bg: BackgroundDetectionSettings) -> None:
        w = self._background_panel.widgets
        w["val_median_max"].setValue(bg.val_median_max)
        self._set_float_row(w, "val_std_max", bg.val_std_max, 0.0, 128.0)
        w["scale"].setValue(bg.scale)

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
        if isinstance(spin, NoScrollDoubleSpinBox):
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
        fr = f.focus_region
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
            "focus_region.enabled": fr.enabled,
            "focus_region.left": fr.left,
            "focus_region.right": fr.right,
            "focus_region.top": fr.top,
            "focus_region.bottom": fr.bottom,
            "inspect_calibration.preview.downsample": settings.inspect_calibration.preview.downsample,
            "inspect_calibration.preview.tick_min_length": settings.inspect_calibration.preview.tick_min_length,
            "inspect_calibration.snap.downsample": settings.inspect_calibration.snap.downsample,
            "inspect_calibration.snap.tick_min_length": settings.inspect_calibration.snap.tick_min_length,
            "red_mark.scale": settings.red_mark.scale,
            "red_mark.open_kernel_size": settings.red_mark.open_kernel_size,
            "red_mark.min_area": settings.red_mark.min_area,
            "red_mark.max_aspect_ratio": settings.red_mark.max_aspect_ratio,
            "red_mark.min_area_fraction": settings.red_mark.min_area_fraction,
            "red_mark.hue_low": settings.red_mark.hue_low,
            "red_mark.hue_high": settings.red_mark.hue_high,
            "red_mark.sat_min": settings.red_mark.sat_min,
            "red_mark.val_min": settings.red_mark.val_min,
            "red_mark.smoothing_alpha": settings.red_mark.smoothing_alpha,
            "red_mark.deadband_px": settings.red_mark.deadband_px,
            "red_mark.max_step_px": settings.red_mark.max_step_px,
            "red_mark.jump_threshold_px": settings.red_mark.jump_threshold_px,
            "red_mark.side_cluster_fraction": settings.red_mark.side_cluster_fraction,
            "red_mark.side_cluster_margin": settings.red_mark.side_cluster_margin,
            "background.val_median_max": settings.background.val_median_max,
            "background.val_std_max": settings.background.val_std_max,
            "background.scale": settings.background.scale,
            "dpi": settings.dpi,
        }

    def _check_modified(self, key: str, current_value: object) -> bool:
        saved = self._saved_values.get(key)
        if isinstance(saved, float) and isinstance(current_value, float):
            return abs(saved - current_value) > 1e-9
        return saved != current_value

    def _mark_field(self, section_field: str, widget_key: str, panel: LabelTrackerMixin, current_value: object) -> None:
        panel.mark_label(widget_key, self._check_modified(section_field, current_value))

    # ------------------------------------------------------------------
    # Change handlers
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_method_combo_changed(self, index: int) -> None:
        method = self._method_combo.itemData(index)
        self._method_stack.setCurrentIndex(0 if method == FOCUS_METHOD_LAPLACIAN else 1)
        self._mv.settings.focus.method = method
        orange = self._check_modified("method", method)
        self.mark_label("method", orange)
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
        target = self._mv.settings.focus.tenengrad if section == "tenengrad" else self._mv.settings.focus.laplacian
        setattr(target, field, value)
        panel = self._tenengrad_panel if section == "tenengrad" else self._laplacian_panel
        self._mark_field(f"{section}.{field}", field, panel, value)
        self._set_unsaved(True)

    def _on_red_mark_changed(self, field: str, value: object) -> None:
        """Apply a changed red-mark field to the manager and update orange state."""
        setattr(self._mv.settings.red_mark, field, value)
        self._mark_field(f"red_mark.{field}", field, self._red_mark_panel, value)
        self._set_unsaved(True)

    def _on_background_changed(self, field: str, value: object) -> None:
        """Apply a changed background field to the manager and update orange state."""
        setattr(self._mv.settings.background, field, value)
        self._mark_field(f"background.{field}", field, self._background_panel, value)
        self._set_unsaved(True)

    def _on_focus_region_changed(self, field: str, value: object) -> None:
        """Apply a changed focus-region field to the manager and update orange state."""
        setattr(self._mv.settings.focus.focus_region, field, value)
        self._mark_field(f"focus_region.{field}", field, self._focus_region_panel, value)
        self._set_unsaved(True)

    def _on_inspect_calibration_changed(self, mode: str, field: str, value: object) -> None:
        """Apply a changed inspect-calibration field to the manager and update orange state."""
        mode_settings = self._mv.settings.inspect_calibration.preview if mode == "preview" else self._mv.settings.inspect_calibration.snap
        if field == "downsample_none":
            mode_settings.downsample = None if value else (
                (self._inspect_preview_panel if mode == "preview" else self._inspect_snap_panel)
                .widgets["downsample"].value()
            )
        else:
            setattr(mode_settings, field, value)
        saved_key = f"inspect_calibration.{mode}.{field if field != 'downsample_none' else 'downsample'}"
        panel = self._inspect_preview_panel if mode == "preview" else self._inspect_snap_panel
        current = mode_settings.downsample if field == "downsample_none" else value
        widget_key = "downsample" if field == "downsample_none" else field
        self._mark_field(saved_key, widget_key, panel, current)
        self._set_unsaved(True)

    # ------------------------------------------------------------------
    # Inspection calibration position handlers
    # ------------------------------------------------------------------

    def _refresh_inspection_position_display(self) -> None:
        try:
            icp = self._mv.settings.inspection_calibration_position
            if icp.is_set:
                x_mm = icp.x_nm / NM_PER_MM
                y_mm = icp.y_nm / NM_PER_MM
                z_mm = icp.z_nm / NM_PER_MM
                self._inspection_pos_label.setText(
                    f"Saved X: {x_mm:.3f}  Y: {y_mm:.3f}  Z: {z_mm:.3f} mm"
                )
                self._inspection_goto_pos_btn.setEnabled(True)
                self._inspection_clear_pos_btn.setEnabled(True)
            else:
                self._inspection_pos_label.setText("Saved position: Not set")
                self._inspection_goto_pos_btn.setEnabled(False)
                self._inspection_clear_pos_btn.setEnabled(False)
        except Exception:
            pass

    def _set_inspection_status(self, text: str) -> None:
        self._inspection_pos_status.setText(text)
        self._inspection_pos_status.setVisible(bool(text))

    def _on_inspection_set_position_clicked(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.is_ready():
            self._set_inspection_status("Motion controller not ready.")
            return
        pos = motion.get_position()
        if pos is None:
            self._set_inspection_status("Could not read stage position.")
            return
        icp = self._mv.settings.inspection_calibration_position
        icp.x_nm = pos.x
        icp.y_nm = pos.y
        icp.z_nm = pos.z
        icp.is_set = True
        self._mv.save_settings()
        info(
            f"[MachineVisionSettings] Inspection position saved: "
            f"X={pos.x / NM_PER_MM:.3f} mm  Y={pos.y / NM_PER_MM:.3f} mm  Z={pos.z / NM_PER_MM:.3f} mm"
        )
        self._refresh_inspection_position_display()
        self._set_inspection_status(
            f"Position saved: ({pos.x / NM_PER_MM:.3f}, {pos.y / NM_PER_MM:.3f}, {pos.z / NM_PER_MM:.3f}) mm"
        )

    def _on_inspection_goto_position_clicked(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.is_ready():
            self._set_inspection_status("Motion controller not ready.")
            return
        icp = self._mv.settings.inspection_calibration_position
        if not icp.is_set:
            self._set_inspection_status("No inspection position saved.")
            return
        try:
            motion.move_to_position(
                Position(x=icp.x_nm, y=icp.y_nm, z=icp.z_nm),
                wait=False,
            )
        except Exception as exc:
            error(f"MachineVisionSettings: move_to_position failed — {exc}")
            self._set_inspection_status("Move failed — see log.")
            return
        self._set_inspection_status(
            f"Moving to ({icp.x_nm / NM_PER_MM:.3f}, "
            f"{icp.y_nm / NM_PER_MM:.3f}, "
            f"{icp.z_nm / NM_PER_MM:.3f}) mm…"
        )

    def _on_inspection_clear_position_clicked(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None:
            return
        icp = self._mv.settings.inspection_calibration_position
        icp.x_nm = 0
        icp.y_nm = 0
        icp.z_nm = 0
        icp.is_set = False
        self._mv.save_settings()
        info("[MachineVisionSettings] Inspection calibration position cleared")
        self._refresh_inspection_position_display()
        self._set_inspection_status("Inspection position cleared.")

    # ------------------------------------------------------------------
    # External state polling
    # ------------------------------------------------------------------

    def _read_external_state(self) -> dict[str, object]:
        s = self._mv.settings
        icp = s.inspection_calibration_position
        return {
            "dpi": s.dpi,
            "last_calibrated": s.inspect_calibration.last_calibrated,
            "icp_is_set": icp.is_set,
            "icp_x": icp.x_nm,
            "icp_y": icp.y_nm,
            "icp_z": icp.z_nm,
            "camera_calibration": s.camera_calibration.calibration is not None,
        }

    def _poll_external_state(self) -> None:
        current = self._read_external_state()
        if current != self._external_state:
            self._external_state = current
            self._populate_from_settings(self._mv.settings)
            self._refresh_camera_calibration_display()
            self._refresh_scale_calibration_display()
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
        for panel in (
            self._tenengrad_panel,
            self._laplacian_panel,
            self._focus_region_panel,
            self._inspect_preview_panel,
            self._inspect_snap_panel,
            self._red_mark_panel,
            self._background_panel,
        ):
            panel.clear_orange()
        self.clear_orange()

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
        for panel in (
            self._tenengrad_panel,
            self._laplacian_panel,
            self._focus_region_panel,
            self._inspect_preview_panel,
            self._inspect_snap_panel,
            self._red_mark_panel,
            self._background_panel,
        ):
            for w in panel.widgets.values():
                w.blockSignals(block)


def machine_vision_page(parent_dialog=None) -> QWidget:
    """Create and return the machine vision settings page widget."""
    return MachineVisionSettingsWidget(parent_dialog=parent_dialog)