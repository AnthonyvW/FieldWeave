from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.generic_config import ConfigManager
from common.logger import info

FIELDWEAVE_VERSION = "1.3.0"


@dataclass
class StitchAndMeasureSettings:
    """Settings for the stitch-and-measure post-processing task."""

    scale_mm: float = 10.0
    tick_min_length: int = 150
    crop_borders: bool = True
    auto_rotate: bool = True
    save_debug_overlay: bool = False

    def validate(self) -> None:
        if self.scale_mm <= 0:
            raise ValueError("scale_mm must be positive")
        if self.tick_min_length <= 0:
            raise ValueError("tick_min_length must be positive")


@dataclass
class PostProcessingSettings:
    """Settings for the post-processing manager."""

    stitch_and_measure: StitchAndMeasureSettings = field(
        default_factory=StitchAndMeasureSettings
    )

    def validate(self) -> None:
        self.stitch_and_measure.validate()


@dataclass
class FieldWeaveSettings:
    """FieldWeave application settings"""

    version: str = FIELDWEAVE_VERSION
    show_patchnotes: bool = False  # Runtime flag - set when version changes, not saved
    first_start: bool = False  # Runtime flag - True when the config file didn't exist yet, not saved
    post_processing: PostProcessingSettings = field(
        default_factory=PostProcessingSettings
    )

    def validate(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty string")
        self.post_processing.validate()


class FieldWeaveSettingsManager(ConfigManager[FieldWeaveSettings]):
    """
    Configuration manager for FieldWeave application settings.

    When a version mismatch is detected during load, the migration
    updates the stored version and sets show_patchnotes flag.
    """

    def __init__(
        self,
        *,
        root_dir: str | Path = "./config/fieldweave",
        backup_dirname: str = "backups",
        backup_keep: int = 5,
    ) -> None:
        super().__init__(
            config_type="fieldweave_settings",
            root_dir=root_dir,
            backup_dirname=backup_dirname,
            backup_keep=backup_keep,
        )

    def migrate(
        self,
        data: dict[str, Any],
        from_version: str,
        to_version: str
    ) -> dict[str, Any]:
        info(f"FieldWeave version changed: {from_version} -> {to_version}")
        data["version"] = to_version
        data["_migrated"] = True
        return data

    def from_dict(self, data: dict[str, Any]) -> FieldWeaveSettings:
        if not data:
            return FieldWeaveSettings()

        migrated = data.pop("_migrated", False)

        pp_data: dict[str, Any] = data.get("post_processing", {})
        sm_data: dict[str, Any] = pp_data.get("stitch_and_measure", {})

        sm_fields = {
            "scale_mm", "tick_min_length", "crop_borders",
            "auto_rotate", "save_debug_overlay",
        }
        stitch_settings = StitchAndMeasureSettings(
            **{k: v for k, v in sm_data.items() if k in sm_fields}
        )
        post_settings = PostProcessingSettings(stitch_and_measure=stitch_settings)

        settings = FieldWeaveSettings(
            version=data.get("version", FieldWeaveSettings.version),
            post_processing=post_settings,
        )

        if migrated:
            settings.show_patchnotes = True
            info("Patch notes flag set - new version detected")

        return settings

    def to_dict(self, settings: FieldWeaveSettings) -> dict[str, Any]:
        sm = settings.post_processing.stitch_and_measure
        return {
            "version": settings.version,
            "post_processing": {
                "stitch_and_measure": {
                    "scale_mm": sm.scale_mm,
                    "tick_min_length": sm.tick_min_length,
                    "crop_borders": sm.crop_borders,
                    "auto_rotate": sm.auto_rotate,
                    "save_debug_overlay": sm.save_debug_overlay,
                },
            },
        }