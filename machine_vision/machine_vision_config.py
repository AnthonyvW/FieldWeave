"""
machine_vision_config.py

Persistent configuration for the machine-vision pipeline.

Each vision algorithm gets its own nested dataclass so that future additions
(e.g. edge detection, object detection) can be added without breaking existing
saved files or requiring a version bump.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from common.generic_config import ConfigManager
from common.logger import info


# ---------------------------------------------------------------------------
# Setting dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FocusDetectionSettings:
    """Parameters for the Laplacian focus-detection pipeline."""

    window_size: int = 15
    """
    Side length (px) of the local variance window.  Larger values integrate
    more context but reduce spatial resolution of the heatmap.
    """

    radius: float = 8.0
    """
    Gaussian/box blur radius (px) applied after the variance step to spread
    the focus signal smoothly across neighbouring regions.
    """

    threshold: float = 0.0
    """
    Raw-variance values below this threshold are zeroed out before blurring,
    suppressing flat/textureless regions.  0.0 disables thresholding.
    """

    half_resolution: bool = True
    """
    Process at half the input resolution for speed.  The result is upscaled
    back to full resolution before display.
    """

    overlay_alpha: float = 0.55
    """
    Blend weight of the focus heatmap over the camera image [0.0 – 1.0].
    0.0 = original image only; 1.0 = heatmap only.
    """

    score_ceiling: float = 0.0
    """
    Fixed divisor used to normalise the raw score map to [0, 1].

    When > 0, the same ceiling is applied to every frame so heatmap
    brightness is stable across frames.  When 0, each frame is normalised
    to its own maximum (per-frame normalisation).
    """

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
class MachineVisionSettings:
    """Top-level machine-vision configuration."""

    focus: FocusDetectionSettings = field(default_factory=FocusDetectionSettings)

    def validate(self) -> None:
        self.focus.validate()


# ---------------------------------------------------------------------------
# ConfigManager
# ---------------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # ConfigManager interface
    # ------------------------------------------------------------------

    def migrate(
        self,
        data: dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> dict[str, Any]:
        """No migrations yet — return data unchanged."""
        info(f"MachineVisionSettings: migrate {from_version} → {to_version} (no-op)")
        return data

    def from_dict(self, data: dict[str, Any]) -> MachineVisionSettings:
        if not data:
            return MachineVisionSettings()

        focus_data: dict[str, Any] = data.get("focus", {})
        focus = FocusDetectionSettings(
            window_size=focus_data.get("window_size", FocusDetectionSettings.window_size),
            radius=focus_data.get("radius", FocusDetectionSettings.radius),
            threshold=focus_data.get("threshold", FocusDetectionSettings.threshold),
            half_resolution=focus_data.get("half_resolution", FocusDetectionSettings.half_resolution),
            overlay_alpha=focus_data.get("overlay_alpha", FocusDetectionSettings.overlay_alpha),
            score_ceiling=focus_data.get("score_ceiling", FocusDetectionSettings.score_ceiling),
        )
        return MachineVisionSettings(focus=focus)

    def to_dict(self, settings: MachineVisionSettings) -> dict[str, Any]:
        f = settings.focus
        return {
            "focus": {
                "window_size": f.window_size,
                "radius": f.radius,
                "threshold": f.threshold,
                "half_resolution": f.half_resolution,
                "overlay_alpha": f.overlay_alpha,
                "score_ceiling": f.score_ceiling,
            },
        }