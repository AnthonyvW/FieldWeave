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

    max_concurrent_focus_stacks: int = 3
    """
    How many queued post-processing routines (e.g. focus stacks) the manager
    runs at once. Higher values finish a backlog faster at the cost of more
    simultaneous CPU/RAM use; each stack's own ``workers`` setting controls
    parallelism within a single stack independently of this.
    """

    def validate(self) -> None:
        self.stitch_and_measure.validate()
        if self.max_concurrent_focus_stacks < 1:
            raise ValueError("max_concurrent_focus_stacks must be >= 1")


@dataclass
class SavedCamera:
    """Identity of a camera, stored so the same physical device can be reopened on the next run."""

    camera_type: str
    device_id: str
    display_name: str
    model: str | None = None
    vid: int | None = None
    pid: int | None = None

    def validate(self) -> None:
        if not self.camera_type:
            raise ValueError("camera_type must be a non-empty string")


@dataclass
class CameraManagerSettings:
    """Settings for the camera manager."""

    last_camera: SavedCamera | None = None

    def validate(self) -> None:
        if self.last_camera is not None:
            self.last_camera.validate()


@dataclass
class FieldWeaveSettings:
    """FieldWeave application settings"""

    version: str = FIELDWEAVE_VERSION
    show_patchnotes: bool = False  # Runtime flag - set when version changes, not saved
    first_start: bool = False  # Runtime flag - True when the config file didn't exist yet, not saved
    post_processing: PostProcessingSettings = field(
        default_factory=PostProcessingSettings
    )
    camera_manager: CameraManagerSettings = field(
        default_factory=CameraManagerSettings
    )

    def validate(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty string")
        self.post_processing.validate()
        self.camera_manager.validate()


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
        post_settings = PostProcessingSettings(
            stitch_and_measure=stitch_settings,
            max_concurrent_focus_stacks=pp_data.get(
                "max_concurrent_focus_stacks", PostProcessingSettings.max_concurrent_focus_stacks
            ),
        )

        cm_data: dict[str, Any] = data.get("camera_manager") or {}
        last_camera_data: dict[str, Any] | None = cm_data.get("last_camera")
        last_camera: SavedCamera | None = None
        if last_camera_data and last_camera_data.get("camera_type"):
            last_camera = SavedCamera(
                camera_type=str(last_camera_data["camera_type"]),
                device_id=str(last_camera_data.get("device_id", "")),
                display_name=str(last_camera_data.get("display_name", "")),
                model=last_camera_data.get("model"),
                vid=last_camera_data.get("vid"),
                pid=last_camera_data.get("pid"),
            )

        settings = FieldWeaveSettings(
            version=data.get("version", FieldWeaveSettings.version),
            post_processing=post_settings,
            camera_manager=CameraManagerSettings(last_camera=last_camera),
        )

        if migrated:
            settings.show_patchnotes = True
            info("Patch notes flag set - new version detected")

        return settings

    def to_dict(self, settings: FieldWeaveSettings) -> dict[str, Any]:
        sm = settings.post_processing.stitch_and_measure
        last_camera = settings.camera_manager.last_camera
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
                "max_concurrent_focus_stacks": settings.post_processing.max_concurrent_focus_stacks,
            },
            "camera_manager": {
                "last_camera": None if last_camera is None else {
                    "camera_type": last_camera.camera_type,
                    "device_id": last_camera.device_id,
                    "display_name": last_camera.display_name,
                    "model": last_camera.model,
                    "vid": last_camera.vid,
                    "pid": last_camera.pid,
                },
            },
        }