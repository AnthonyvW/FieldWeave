from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Union

from common.generic_config import ConfigManager, DEFAULT_FILENAME


@dataclass
class MotionSystemSettings:
    FIRMWARE_NAME: str = "Marlin"
    MACHINE_TYPE: str = "Ender-3"
    baud_rate: int = 115200
    max_x: int = 220  # Maximum X dimension in mm
    max_y: int = 220  # Maximum Y dimension in mm
    max_z: int = 220   # Maximum Z dimension in mm
    step_size: int = 40000  # Minimum distance that can be moved in 0.nanometers. 0.04mm is 40,000 nm
    sample_positions: dict[int, dict[str, float]] = field(default_factory=dict)
    calibration_pattern_position: dict[str, float] = field(default_factory=dict)  # X, Y, Z in mm

    # Sample calibration positions (for verifying X positions)
    calibration_y: float = 220.0  # Y position for calibration checks (mm)
    calibration_z: float = 26.0   # Z position for calibration checks (mm)

    # Camera calibration data
    camera_calibration: dict[str, Any] = field(default_factory=dict)  # Stores M_est, M_inv, reference position, etc.

    def validate(self) -> None:
        """
        Validate motion system settings.

        Raises:
            ValueError: If any setting is invalid
        """
        if self.baud_rate <= 0:
            raise ValueError("baud_rate must be positive")
        if self.max_x <= 0 or self.max_y <= 0 or self.max_z <= 0:
            raise ValueError("max_x, max_y, and max_z must all be positive")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")


class MotionSystemSettingsManager(ConfigManager[MotionSystemSettings]):
    """Configuration manager for motion system settings."""

    def __init__(
        self,
        *,
        root_dir: Union[str, Path] = "./config/motion_system",
        default_filename: str = DEFAULT_FILENAME,
        backup_dirname: str = "backups",
        backup_keep: int = 5,
    ) -> None:
        super().__init__(
            config_type="motion_system_settings",
            root_dir=root_dir,
            default_filename=default_filename,
            backup_dirname=backup_dirname,
            backup_keep=backup_keep,
        )

    def from_dict(self, data: dict[str, Any]) -> MotionSystemSettings:
        if not data:
            return MotionSystemSettings()

        valid_fields = {f.name for f in MotionSystemSettings.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return MotionSystemSettings(**filtered_data)

    def to_dict(self, settings: MotionSystemSettings) -> dict[str, Any]:
        return asdict(settings)