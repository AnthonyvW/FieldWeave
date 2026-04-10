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
from machine_vision.camera_calibration import CameraCalibration


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

    def validate(self) -> None:
        if self.move_x_ticks <= 0:
            raise ValueError("move_x_ticks must be > 0")
        if self.move_y_ticks <= 0:
            raise ValueError("move_y_ticks must be > 0")


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------

@dataclass
class MachineVisionSettings:
    """Top-level machine-vision configuration."""

    focus: FocusDetectionSettings = field(default_factory=FocusDetectionSettings)
    camera_calibration: CameraCalibrationSettings = field(
        default_factory=CameraCalibrationSettings
    )

    def validate(self) -> None:
        self.focus.validate()
        self.camera_calibration.validate()


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


class MachineVisionSettingsManager(ConfigManager[MachineVisionSettings]):
    """
    Persistent configuration manager for machine-vision settings.

    Saved to ``./config/machine_vision/default_settings.yaml`` by default.
    """

    def __init__(
        self,
        *,
        root_dir: Union[str, Path] = "./config/machine_vision",
        default_filename: str = "default_settings.yaml",
        backup_dirname: str = "backups",
        backup_keep: int = 5,
    ) -> None:
        super().__init__(
            config_type="machine_vision_settings",
            root_dir=root_dir,
            default_filename=default_filename,
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

        return MachineVisionSettings(focus=focus, camera_calibration=camera_calibration)

    def to_dict(self, settings: MachineVisionSettings) -> dict[str, Any]:
        f = settings.focus
        t = f.tenengrad
        lap = f.laplacian
        fr = f.focus_region
        cc = settings.camera_calibration
        return {
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
        }