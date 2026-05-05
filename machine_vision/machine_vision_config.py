"""
machine_vision_config.py

Persistent configuration for the machine-vision pipeline.

Each vision algorithm gets its own nested dataclass.  The focus-detection
algorithm additionally separates Tenengrad and Laplacian parameters so each
method has its own saved state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union

from common.generic_config import ConfigManager
from common.logger import info
from machine_vision.algorithms.camera_calibration import CameraCalibration, CameraYAxisOrientation


# ---------------------------------------------------------------------------
# Focus-detection method type
# ---------------------------------------------------------------------------

FocusMethod = Literal["tenengrad", "laplacian"]
FOCUS_METHOD_TENENGRAD: FocusMethod = "tenengrad"
FOCUS_METHOD_LAPLACIAN: FocusMethod = "laplacian"


# ---------------------------------------------------------------------------
# Per-method parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TenengradSettings:
    """Parameters specific to the Tenengrad (Sobel-based) focus measure."""

    kernel_size: int = 3
    """
    Sobel kernel size (must be 1, 3, 5, or 7).  Larger kernels are less
    sensitive to noise but reduce spatial resolution of the edge response.
    """

    radius: float = 8.0
    """Gaussian/box blur radius (px) applied after gradient magnitude."""

    threshold: float = 0.0
    """Gradient values below this level are zeroed out.  0 = disabled."""

    half_resolution: bool = True
    """Process at half resolution for speed; result upscaled before display."""

    overlay_alpha: float = 0.55
    """Heatmap blend weight over the camera image [0.0 – 1.0]."""

    score_ceiling: float = 15.0
    """
    Fixed normalisation ceiling.  Applied every frame when auto_ceiling is
    False; ignored (per-frame normalisation) when auto_ceiling is True.
    """

    auto_ceiling: bool = False
    """When True, ignore score_ceiling and normalise per-frame."""

    def validate(self) -> None:
        if self.kernel_size not in (1, 3, 5, 7):
            raise ValueError("kernel_size must be 1, 3, 5, or 7")
        if self.radius < 0:
            raise ValueError("radius must be >= 0")
        if self.threshold < 0:
            raise ValueError("threshold must be >= 0")
        if not (0.0 <= self.overlay_alpha <= 1.0):
            raise ValueError("overlay_alpha must be in [0.0, 1.0]")
        if self.score_ceiling < 0:
            raise ValueError("score_ceiling must be >= 0")


@dataclass
class LaplacianSettings:
    """Parameters specific to the local Laplacian-variance focus measure."""

    window_size: int = 15
    """
    Side length (px) of the local variance window.  Must be odd.
    Larger values integrate more context but reduce heatmap resolution.
    """

    radius: float = 8.0
    """Gaussian/box blur radius (px) applied after the variance step."""

    threshold: float = 0.0
    """Variance values below this level are zeroed out.  0 = disabled."""

    half_resolution: bool = True
    """Process at half resolution for speed; result upscaled before display."""

    overlay_alpha: float = 0.55
    """Heatmap blend weight over the camera image [0.0 – 1.0]."""

    score_ceiling: float = 15.0
    """
    Fixed normalisation ceiling.  Applied every frame when auto_ceiling is
    False; ignored (per-frame normalisation) when auto_ceiling is True.
    """

    auto_ceiling: bool = False
    """When True, ignore score_ceiling and normalise per-frame."""

    def validate(self) -> None:
        if self.window_size < 3:
            raise ValueError("window_size must be >= 3")
        if self.window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        if self.radius < 0:
            raise ValueError("radius must be >= 0")
        if self.threshold < 0:
            raise ValueError("threshold must be >= 0")
        if not (0.0 <= self.overlay_alpha <= 1.0):
            raise ValueError("overlay_alpha must be in [0.0, 1.0]")
        if self.score_ceiling < 0:
            raise ValueError("score_ceiling must be >= 0")


@dataclass
class FocusRegionSettings:
    """
    Defines a rectangular region of interest for focus analysis.

    Each margin is expressed as a percentage (0–50) of the image dimension
    to exclude from that edge.  For example, left=10 means the leftmost 10%
    of columns are masked out.  When enabled=False the full frame is analysed
    and all margin values are ignored.

    The four margins must not overlap: left + right < 100 and top + bottom < 100.
    """

    enabled: bool = False
    """When False the full frame is used and all margins are ignored."""

    left: float = 0.0
    """Percentage of image width to exclude from the left edge [0–50]."""

    right: float = 0.0
    """Percentage of image width to exclude from the right edge [0–50]."""

    top: float = 0.0
    """Percentage of image height to exclude from the top edge [0–50]."""

    bottom: float = 0.0
    """Percentage of image height to exclude from the bottom edge [0–50]."""

    def validate(self) -> None:
        for name, val in (("left", self.left), ("right", self.right),
                          ("top", self.top), ("bottom", self.bottom)):
            if not (0.0 <= val <= 50.0):
                raise ValueError(f"focus_region.{name} must be in [0.0, 50.0]")
        if self.left + self.right >= 100.0:
            raise ValueError("focus_region left + right must be < 100")
        if self.top + self.bottom >= 100.0:
            raise ValueError("focus_region top + bottom must be < 100")


@dataclass
class FocusDetectionSettings:
    """
    Top-level focus-detection configuration.

    Holds the active method selector and independent parameter blocks for
    each method so that switching methods does not discard the other method's
    tuned values.
    """

    method: FocusMethod = FOCUS_METHOD_LAPLACIAN
    """Which focus measure to use."""

    tenengrad: TenengradSettings = field(default_factory=TenengradSettings)
    laplacian: LaplacianSettings = field(default_factory=LaplacianSettings)
    focus_region: FocusRegionSettings = field(default_factory=FocusRegionSettings)

    def validate(self) -> None:
        if self.method not in (FOCUS_METHOD_TENENGRAD, FOCUS_METHOD_LAPLACIAN):
            raise ValueError(f"Unknown focus method: {self.method!r}")
        self.tenengrad.validate()
        self.laplacian.validate()
        self.focus_region.validate()

    @property
    def active(self) -> TenengradSettings | LaplacianSettings:
        """Return the parameter block for the currently selected method."""
        return self.tenengrad if self.method == FOCUS_METHOD_TENENGRAD else self.laplacian


# ---------------------------------------------------------------------------
# Red-mark detection settings
# ---------------------------------------------------------------------------

@dataclass
class RedMarkDetectionSettings:
    """
    Parameters for the red registration-mark detection algorithm.

    The detector isolates red blobs in each frame using HSV colour space,
    filters by area and aspect ratio, and computes a stabilized centroid
    used to draw a reference line on the preview overlay.
    """

    scale: int = 4
    """
    Downsample factor applied before processing.  The image is resized to
    1/scale in each dimension before detection; centroids and masks are
    projected back to full resolution afterwards.  Must be >= 1.
    Higher values reduce processing time at the cost of precision.
    """

    open_kernel_size: int = 5
    """
    Side length of the elliptical morphological opening kernel applied after
    HSV thresholding.  Larger values remove more noise but may break up small
    marks.
    """

    min_area: int = 500
    """Minimum blob area in pixels.  Smaller blobs are discarded as noise."""

    max_aspect_ratio: float = 8.0
    """
    Maximum ratio of the longer side to the shorter side of a blob's bounding
    box.  Elongated blobs (e.g. scratches) are discarded above this value.
    """

    min_area_fraction: float = 0.1
    """
    Minimum blob area as a fraction of the largest surviving blob.  Discards
    small stray marks when a large registration mark is also present.
    """

    hue_low: int = 160
    """
    Lower hue boundary for the upper red range (160-180 in OpenCV HSV).
    Red wraps around 0/180, so two ranges are combined.
    """

    hue_high: int = 10
    """Upper hue boundary for the lower red range (0-hue_high in OpenCV HSV)."""

    sat_min: int = 100
    """Minimum HSV saturation for a pixel to count as red [0-255]."""

    val_min: int = 50
    """Minimum HSV value for a pixel to count as red [0-255]."""

    smoothing_alpha: float = 0.25
    """EMA weight for the stabilized centroid position [0.0 - 1.0]."""

    deadband_px: float = 2.0
    """Changes smaller than this (in pixels) are ignored to prevent jitter."""

    max_step_px: float = 30.0
    """Maximum centroid movement per frame in pixels (slew-rate limit)."""

    jump_threshold_px: float = 50.0
    """
    If the raw centroid moves more than this many pixels from the smoothed
    value in a single frame, the smoothed value jumps directly to the raw
    value instead of slewing.
    """

    side_cluster_fraction: float = 0.85
    """
    Fraction of marks that must be within ``side_cluster_margin`` of one
    horizontal edge before the line orientation switches to ``'horizontal'``.
    """

    side_cluster_margin: float = 0.18
    """
    Fraction of image width defining the left/right edge zones used by the
    side-cluster test.
    """

    def validate(self) -> None:
        if self.scale < 1:
            raise ValueError("red_mark.scale must be >= 1")
        if self.open_kernel_size < 1:
            raise ValueError("red_mark.open_kernel_size must be >= 1")
        if self.min_area < 0:
            raise ValueError("red_mark.min_area must be >= 0")
        if self.max_aspect_ratio < 1.0:
            raise ValueError("red_mark.max_aspect_ratio must be >= 1.0")
        if not (0.0 <= self.min_area_fraction <= 1.0):
            raise ValueError("red_mark.min_area_fraction must be in [0.0, 1.0]")
        if not (0 <= self.hue_low <= 180):
            raise ValueError("red_mark.hue_low must be in [0, 180]")
        if not (0 <= self.hue_high <= 180):
            raise ValueError("red_mark.hue_high must be in [0, 180]")
        if not (0 <= self.sat_min <= 255):
            raise ValueError("red_mark.sat_min must be in [0, 255]")
        if not (0 <= self.val_min <= 255):
            raise ValueError("red_mark.val_min must be in [0, 255]")
        if not (0.0 <= self.smoothing_alpha <= 1.0):
            raise ValueError("red_mark.smoothing_alpha must be in [0.0, 1.0]")
        if self.deadband_px < 0:
            raise ValueError("red_mark.deadband_px must be >= 0")
        if self.max_step_px < 0:
            raise ValueError("red_mark.max_step_px must be >= 0")
        if self.jump_threshold_px < 0:
            raise ValueError("red_mark.jump_threshold_px must be >= 0")
        if not (0.0 <= self.side_cluster_fraction <= 1.0):
            raise ValueError("red_mark.side_cluster_fraction must be in [0.0, 1.0]")
        if not (0.0 <= self.side_cluster_margin <= 0.45):
            raise ValueError("red_mark.side_cluster_margin must be in [0.0, 0.45]")


# ---------------------------------------------------------------------------
# Background detection settings
# ---------------------------------------------------------------------------

@dataclass
class BackgroundDetectionSettings:
    """
    Parameters for the black-plastic background detection algorithm.

    Detection is based on the median and standard deviation of the HSV value
    (brightness) channel.  The plastic surface is dark-to-mid grey and highly
    uniform; other materials are rejected by one or both constraints.
    """

    val_median_max: int = 128
    """
    Maximum median HSV value (brightness) to classify the frame as background
    [0-255].  Corresponds to 0.5 on GIMP's normalised [0-1] scale.  The
    plastic surface is dark-to-mid grey; bright surfaces such as paper are
    rejected by this constraint.
    """

    val_std_max: float = 25.0
    """
    Maximum standard deviation of HSV value to classify the frame as
    background [0-255].  Corresponds to 0.1 on GIMP's normalised [0-1]
    scale.  The plastic surface is highly uniform; textured materials such
    as wood grain or printed paper are rejected by this constraint.
    """

    scale: int = 4
    """
    Downsample factor applied before computing statistics.  Must be >= 1.
    Higher values reduce computation with negligible effect on accuracy.
    """

    def validate(self) -> None:
        if not (0 <= self.val_median_max <= 255):
            raise ValueError("background.val_median_max must be in [0, 255]")
        if self.val_std_max < 0:
            raise ValueError("background.val_std_max must be >= 0")
        if self.scale < 1:
            raise ValueError("background.scale must be >= 1")


# ---------------------------------------------------------------------------
# Inspect-calibration settings
# ---------------------------------------------------------------------------

@dataclass
class InspectCalibrationModeSettings:
    """
    Parameters for one mode (preview or snap) of the calibration-bar inspector.
    """

    downsample: int | None = None
    """
    Shrink each axis by this factor before processing.  ``None`` or ``1``
    disables downscaling.  Use a value like ``2`` for full-resolution captures
    where speed matters more than precision.
    """

    tick_min_length: int = 150
    """
    Minimum perpendicular pixel run (in full-resolution pixels) required to
    count a candidate as a tick mark.  Scaled by ``downsample`` internally.
    """

    def validate(self, label: str) -> None:
        if self.downsample is not None and self.downsample < 1:
            raise ValueError(f"inspect_calibration.{label}.downsample must be >= 1 or None")
        if self.tick_min_length < 1:
            raise ValueError(f"inspect_calibration.{label}.tick_min_length must be >= 1")


@dataclass
class InspectCalibrationSettings:
    """
    Parameters for the calibration-bar inspection algorithm.

    The inspector detects a ruler/scale bar in the frame, determines its
    orientation (horizontal or vertical), locates tick marks along it, and
    reports whether end-caps are present at each terminus.

    Two independent mode blocks are provided so that the live preview and
    full-image snap can be tuned separately without mutual interference.
    """

    preview: InspectCalibrationModeSettings = field(
        default_factory=lambda: InspectCalibrationModeSettings(
            downsample=None,
            tick_min_length=150,
        )
    )
    """Settings used during live preview (speed-optimised; no downscaling)."""

    snap: InspectCalibrationModeSettings = field(
        default_factory=lambda: InspectCalibrationModeSettings(
            downsample=2,
            tick_min_length=200,
        )
    )
    """Settings used for full-image captures (accuracy-optimised; 2x downscaling)."""

    last_calibrated: str | None = None
    """
    ISO-8601 timestamp of the most recent accepted inspection calibration snap,
    or ``None`` if no snap has been accepted in the current configuration.
    Set by the caller after a successful snap result is accepted.
    """

    def validate(self) -> None:
        self.preview.validate("preview")
        self.snap.validate("snap")


# ---------------------------------------------------------------------------
# Camera calibration settings
# ---------------------------------------------------------------------------

@dataclass
class CameraCalibrationSettings:
    """
    Persistent camera-calibration configuration.

    ``move_x_ticks`` and ``move_y_ticks`` are the distances (in 0.01 mm tick
    units) that the stage moves during the calibration routine.  They are
    persisted here so that the UI can edit them and the printer controller can
    read them without hard-coding defaults.

    ``calibration`` holds the last successfully computed
    ``CameraCalibration``, serialised to/from a plain dict via
    ``CameraCalibration.to_dict`` / ``CameraCalibration.from_dict``.  It is
    ``None`` when no calibration has been performed yet or after
    ``clear_calibration`` is called.
    """

    move_x_ticks: int = 100
    """Distance to move in +X during calibration (0.01 mm units; 100 = 1 mm)."""

    move_y_ticks: int = 100
    """Distance to move in +Y during calibration (0.01 mm units; 100 = 1 mm)."""

    calibration: CameraCalibration | None = None
    """Most recently computed calibration, or None if uncalibrated."""

    @property
    def y_axis_orientation(self) -> CameraYAxisOrientation | None:
        """
        Which image axis the world Y axis is primarily aligned with, or
        ``None`` when no calibration is present.

        Derived from the active ``CameraCalibration.M_est``; never persisted.
        """
        if self.calibration is None:
            return None
        return self.calibration.y_axis_orientation

    def validate(self) -> None:
        if self.move_x_ticks <= 0:
            raise ValueError("move_x_ticks must be > 0")
        if self.move_y_ticks <= 0:
            raise ValueError("move_y_ticks must be > 0")


# ---------------------------------------------------------------------------
# Inspection calibration position
# ---------------------------------------------------------------------------

@dataclass
class InspectionCalibrationPosition:
    """
    Saved stage position for the inspection calibration workflow.

    Stores the XYZ coordinates (in nanometres) of the position where the
    inspection calibration slide was last centred.  When ``is_set`` is False
    the coordinate fields should be treated as undefined.
    """

    is_set: bool = False
    """True when a position has been saved and the coordinates are valid."""

    x_nm: int = 0
    """Stage X coordinate in nanometres."""

    y_nm: int = 0
    """Stage Y coordinate in nanometres."""

    z_nm: int = 0
    """Stage Z coordinate in nanometres."""


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------

@dataclass
class MachineVisionSettings:
    """Top-level machine-vision configuration."""

    dpi: float | None = None
    """
    Known optical resolution of the camera/lens assembly in dots-per-inch.
    When set this serves as a reference value for display and cross-check
    against the DPI estimated by ``CameraCalibration``.  ``None`` means
    unspecified.
    """

    focus: FocusDetectionSettings = field(default_factory=FocusDetectionSettings)
    camera_calibration: CameraCalibrationSettings = field(
        default_factory=CameraCalibrationSettings
    )
    inspect_calibration: InspectCalibrationSettings = field(
        default_factory=InspectCalibrationSettings
    )
    inspection_calibration_position: InspectionCalibrationPosition = field(
        default_factory=InspectionCalibrationPosition
    )
    """Saved stage position used as the starting point for inspection calibration."""
    red_mark: RedMarkDetectionSettings = field(default_factory=RedMarkDetectionSettings)
    """Parameters for the red registration-mark detection algorithm."""

    background: BackgroundDetectionSettings = field(default_factory=BackgroundDetectionSettings)
    """Parameters for the black-plastic background detection algorithm."""

    def validate(self) -> None:
        self.focus.validate()
        self.camera_calibration.validate()
        self.inspect_calibration.validate()
        self.red_mark.validate()
        self.background.validate()


# ---------------------------------------------------------------------------
# ConfigManager
# ---------------------------------------------------------------------------

def _load_tenengrad(d: dict[str, Any]) -> TenengradSettings:
    D = TenengradSettings
    return TenengradSettings(
        kernel_size=d.get("kernel_size", D.kernel_size),
        radius=d.get("radius", D.radius),
        threshold=d.get("threshold", D.threshold),
        half_resolution=d.get("half_resolution", D.half_resolution),
        overlay_alpha=d.get("overlay_alpha", D.overlay_alpha),
        score_ceiling=d.get("score_ceiling", D.score_ceiling),
        auto_ceiling=d.get("auto_ceiling", D.auto_ceiling),
    )


def _load_laplacian(d: dict[str, Any]) -> LaplacianSettings:
    D = LaplacianSettings
    return LaplacianSettings(
        window_size=d.get("window_size", D.window_size),
        radius=d.get("radius", D.radius),
        threshold=d.get("threshold", D.threshold),
        half_resolution=d.get("half_resolution", D.half_resolution),
        overlay_alpha=d.get("overlay_alpha", D.overlay_alpha),
        score_ceiling=d.get("score_ceiling", D.score_ceiling),
        auto_ceiling=d.get("auto_ceiling", D.auto_ceiling),
    )


def _load_focus_region(d: dict[str, Any]) -> FocusRegionSettings:
    D = FocusRegionSettings
    return FocusRegionSettings(
        enabled=d.get("enabled", D.enabled),
        left=d.get("left", D.left),
        right=d.get("right", D.right),
        top=d.get("top", D.top),
        bottom=d.get("bottom", D.bottom),
    )


def _load_red_mark(d: dict[str, Any]) -> RedMarkDetectionSettings:
    D = RedMarkDetectionSettings
    return RedMarkDetectionSettings(
        scale=d.get("scale", D.scale),
        open_kernel_size=d.get("open_kernel_size", D.open_kernel_size),
        min_area=d.get("min_area", D.min_area),
        max_aspect_ratio=d.get("max_aspect_ratio", D.max_aspect_ratio),
        min_area_fraction=d.get("min_area_fraction", D.min_area_fraction),
        hue_low=d.get("hue_low", D.hue_low),
        hue_high=d.get("hue_high", D.hue_high),
        sat_min=d.get("sat_min", D.sat_min),
        val_min=d.get("val_min", D.val_min),
        smoothing_alpha=d.get("smoothing_alpha", D.smoothing_alpha),
        deadband_px=d.get("deadband_px", D.deadband_px),
        max_step_px=d.get("max_step_px", D.max_step_px),
        jump_threshold_px=d.get("jump_threshold_px", D.jump_threshold_px),
        side_cluster_fraction=d.get("side_cluster_fraction", D.side_cluster_fraction),
        side_cluster_margin=d.get("side_cluster_margin", D.side_cluster_margin),
    )



def _load_background(d: dict[str, Any]) -> BackgroundDetectionSettings:
    D = BackgroundDetectionSettings
    return BackgroundDetectionSettings(
        val_median_max=d.get("val_median_max", D.val_median_max),
        val_std_max=d.get("val_std_max", D.val_std_max),
        scale=d.get("scale", D.scale),
    )


class MachineVisionSettingsManager(ConfigManager[MachineVisionSettings]):
    """
    Persistent configuration manager for machine-vision settings.

    Saved to ``./config/machine_vision/settings.yaml`` by default.
    """

    def __init__(
        self,
        *,
        root_dir: Union[str, Path] = "./config/machine_vision",
        backup_dirname: str = "backups",
        backup_keep: int = 5,
    ) -> None:
        super().__init__(
            config_type="machine_vision_settings",
            root_dir=root_dir,
            backup_dirname=backup_dirname,
            backup_keep=backup_keep,
        )

    def migrate(
        self,
        data: dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> dict[str, Any]:
        info(f"MachineVisionSettings: migrate {from_version} → {to_version} (no-op)")
        return data

    def from_dict(self, data: dict[str, Any]) -> MachineVisionSettings:
        if not data:
            return MachineVisionSettings()

        focus_data: dict[str, Any] = data.get("focus", {})
        focus = FocusDetectionSettings(
            method=focus_data.get("method", FocusDetectionSettings.method),
            tenengrad=_load_tenengrad(focus_data.get("tenengrad", {})),
            laplacian=_load_laplacian(focus_data.get("laplacian", {})),
            focus_region=_load_focus_region(focus_data.get("focus_region", {})),
        )

        cal_data: dict[str, Any] = data.get("camera_calibration", {})
        D = CameraCalibrationSettings
        cal_dict = cal_data.get("calibration")
        calibration: CameraCalibration | None = None
        if cal_dict:
            try:
                calibration = CameraCalibration.from_dict(cal_dict)
            except Exception:
                pass  # Corrupt saved calibration; start uncalibrated.
        camera_calibration = CameraCalibrationSettings(
            move_x_ticks=cal_data.get("move_x_ticks", D.move_x_ticks),
            move_y_ticks=cal_data.get("move_y_ticks", D.move_y_ticks),
            calibration=calibration,
        )

        ic_data: dict[str, Any] = data.get("inspect_calibration", {})
        D_preview = InspectCalibrationModeSettings(downsample=None, tick_min_length=150)
        D_snap = InspectCalibrationModeSettings(downsample=2, tick_min_length=200)

        def _load_ic_mode(d: dict[str, Any], defaults: InspectCalibrationModeSettings) -> InspectCalibrationModeSettings:
            return InspectCalibrationModeSettings(
                downsample=d.get("downsample", defaults.downsample),
                tick_min_length=d.get("tick_min_length", defaults.tick_min_length),
            )

        inspect_calibration = InspectCalibrationSettings(
            preview=_load_ic_mode(ic_data.get("preview", {}), D_preview),
            snap=_load_ic_mode(ic_data.get("snap", {}), D_snap),
            last_calibrated=ic_data.get("last_calibrated"),
        )

        icp_data: dict[str, Any] = data.get("inspection_calibration_position", {})
        inspection_calibration_position = InspectionCalibrationPosition(
            is_set=icp_data.get("is_set", False),
            x_nm=icp_data.get("x_nm", 0),
            y_nm=icp_data.get("y_nm", 0),
            z_nm=icp_data.get("z_nm", 0),
        )

        return MachineVisionSettings(
            dpi=data.get("dpi"),
            focus=focus,
            camera_calibration=camera_calibration,
            inspect_calibration=inspect_calibration,
            inspection_calibration_position=inspection_calibration_position,
            red_mark=_load_red_mark(data.get("red_mark", {})),
            background=_load_background(data.get("background", {})),
        )

    def to_dict(self, settings: MachineVisionSettings) -> dict[str, Any]:
        f = settings.focus
        t = f.tenengrad
        lap = f.laplacian
        fr = f.focus_region
        cc = settings.camera_calibration
        ic = settings.inspect_calibration
        icp = settings.inspection_calibration_position
        rm = settings.red_mark
        bg = settings.background
        return {
            "dpi": settings.dpi,
            "focus": {
                "method": f.method,
                "tenengrad": {
                    "kernel_size": t.kernel_size,
                    "radius": t.radius,
                    "threshold": t.threshold,
                    "half_resolution": t.half_resolution,
                    "overlay_alpha": t.overlay_alpha,
                    "score_ceiling": t.score_ceiling,
                    "auto_ceiling": t.auto_ceiling,
                },
                "laplacian": {
                    "window_size": lap.window_size,
                    "radius": lap.radius,
                    "threshold": lap.threshold,
                    "half_resolution": lap.half_resolution,
                    "overlay_alpha": lap.overlay_alpha,
                    "score_ceiling": lap.score_ceiling,
                    "auto_ceiling": lap.auto_ceiling,
                },
                "focus_region": {
                    "enabled": fr.enabled,
                    "left": fr.left,
                    "right": fr.right,
                    "top": fr.top,
                    "bottom": fr.bottom,
                },
            },
            "camera_calibration": {
                "move_x_ticks": cc.move_x_ticks,
                "move_y_ticks": cc.move_y_ticks,
                "calibration": cc.calibration.to_dict() if cc.calibration is not None else None,
            },
            "inspect_calibration": {
                "preview": {
                    "downsample": ic.preview.downsample,
                    "tick_min_length": ic.preview.tick_min_length,
                },
                "snap": {
                    "downsample": ic.snap.downsample,
                    "tick_min_length": ic.snap.tick_min_length,
                },
                "last_calibrated": ic.last_calibrated,
            },
            "inspection_calibration_position": {
                "is_set": icp.is_set,
                "x_nm": icp.x_nm,
                "y_nm": icp.y_nm,
                "z_nm": icp.z_nm,
            },
            "red_mark": {
                "scale": rm.scale,
                "open_kernel_size": rm.open_kernel_size,
                "min_area": rm.min_area,
                "max_aspect_ratio": rm.max_aspect_ratio,
                "min_area_fraction": rm.min_area_fraction,
                "hue_low": rm.hue_low,
                "hue_high": rm.hue_high,
                "sat_min": rm.sat_min,
                "val_min": rm.val_min,
                "smoothing_alpha": rm.smoothing_alpha,
                "deadband_px": rm.deadband_px,
                "max_step_px": rm.max_step_px,
                "jump_threshold_px": rm.jump_threshold_px,
                "side_cluster_fraction": rm.side_cluster_fraction,
                "side_cluster_margin": rm.side_cluster_margin,
            },
            "background": {
                "val_median_max": bg.val_median_max,
                "val_std_max": bg.val_std_max,
                "scale": bg.scale,
            },
        }
