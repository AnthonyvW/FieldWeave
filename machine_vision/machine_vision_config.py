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

    def validate(self) -> None:
        if self.method not in (FOCUS_METHOD_TENENGRAD, FOCUS_METHOD_LAPLACIAN):
            raise ValueError(f"Unknown focus method: {self.method!r}")
        self.tenengrad.validate()
        self.laplacian.validate()

    @property
    def active(self) -> TenengradSettings | LaplacianSettings:
        """Return the parameter block for the currently selected method."""
        return self.tenengrad if self.method == FOCUS_METHOD_TENENGRAD else self.laplacian


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------

@dataclass
class MachineVisionSettings:
    """Top-level machine-vision configuration."""

    focus: FocusDetectionSettings = field(default_factory=FocusDetectionSettings)

    def validate(self) -> None:
        self.focus.validate()


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
        )
        return MachineVisionSettings(focus=focus)

    def to_dict(self, settings: MachineVisionSettings) -> dict[str, Any]:
        f = settings.focus
        t = f.tenengrad
        lap = f.laplacian
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
            },
        }