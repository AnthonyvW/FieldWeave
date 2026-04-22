from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

from common.generic_config import ConfigManager, DEFAULT_FILENAME

# Default jog-step presets in nanometres (0.04 mm, 0.4 mm, 2.0 mm, 10.0 mm).
_DEFAULT_STEP_PRESETS_NM: list[int] = [40_000, 400_000, 2_000_000, 10_000_000]


@dataclass
class CameraCalibrationPosition:
    """Saved stage position used as the starting point for camera calibration.

    Coordinates are stored in nanometres.  ``last_calibrated_iso`` holds the
    ISO-8601 timestamp of the most recent successful calibration, or an empty
    string if no calibration has been completed yet.
    """

    x_nm: int = 0
    y_nm: int = 0
    z_nm: int = 0
    last_calibrated_iso: str = ""
    is_set: bool = False

    @property
    def has_been_calibrated(self) -> bool:
        """True if a successful calibration has been recorded."""
        return bool(self.last_calibrated_iso)


@dataclass
class TreeCoreSlot:
    """A single slot in a tree core slide.

    ``position_nm`` is the coordinate along the automation axis for this slot.
    ``offset_nm`` is an additional fine-tune offset applied on top of the slot
    position, also along the automation axis.
    """

    position_nm: int = 0
    offset_nm: int = 0


@dataclass
class TreeCoreAutomationSettings:
    """Settings for automated tree core slot imaging.

    The stage steps along a fixed axis (currently Y) to image each slot in
    sequence.  A mark is imaged at a known X/Y position before the run begins.

    All positions and offsets are in nanometres.
    """

    # Axis driven during automation — hard-coded to Y for now.
    axis: str = "y"

    # Position of the reference mark imaged before the run.
    mark_reference_nm: int = 0
    mark_z_nm: int = 0

    # Z height to move to at the start of the run.
    starting_height_nm: int = 0

    # Offset applied along the automation axis before the first slot.
    starting_offset_nm: int = 0

    # Distance between consecutive slots along the automation axis.
    slot_separation_nm: int = 11_200_000

    # ISO-8601 timestamp of the most recent successful calibration, or an empty
    # string if no calibration has been completed yet.
    last_calibrated_iso: str = ""

    # Ordered list of slot definitions (defaults to 20 empty slots).
    slots: list[TreeCoreSlot] = field(default_factory=lambda: [TreeCoreSlot() for _ in range(20)])

    @property
    def num_slots(self) -> int:
        return len(self.slots)

    @property
    def has_been_calibrated(self) -> bool:
        """True if a successful calibration has been recorded."""
        return bool(self.last_calibrated_iso)


@dataclass
class MotionSystemSettings:
    FIRMWARE_NAME: str = "Marlin"
    MACHINE_TYPE: str = "Ender-3"
    baud_rate: int = 115200
    max_x: int = 220  # Maximum X dimension in mm
    max_y: int = 235  # Maximum Y dimension in mm
    max_z: int = 220   # Maximum Z dimension in mm
    step_size: int = 40000  # Minimum distance that can be moved in nanometres. 0.04 mm = 40,000 nm
    sample_positions: dict[int, dict[str, float]] = field(default_factory=dict)
    calibration_pattern_position: dict[str, float] = field(default_factory=dict)  # X, Y, Z in mm

    # Sample calibration positions (for verifying X positions)
    calibration_y: float = 220.0  # Y position for calibration checks (mm)
    calibration_z: float = 26.0   # Z position for calibration checks (mm)

    # Navigation widget — axis inversion
    invert_x: bool = False  # Invert X direction in the navigation widget
    invert_y: bool = False  # Invert Y direction in the navigation widget
    invert_z: bool = False  # Invert Z direction in the navigation widget

    # Starting height: Z position (nanometres) to move to after every home sequence.
    # 0 means stay at the homed position (no post-home move).
    starting_height_nm: int = 0

    # Navigation widget — jog-step presets (nanometres)
    # Four buttons shown in the navigation widget; default: 0.04, 0.4, 2.0, 10.0 mm.
    step_presets: list[int] = field(
        default_factory=lambda: list(_DEFAULT_STEP_PRESETS_NM)
    )

    # Saved stage position for camera calibration and last-calibrated timestamp.
    camera_calibration_position: CameraCalibrationPosition = field(
        default_factory=CameraCalibrationPosition
    )

    # Tree core slot automation settings.
    tree_core_automation: TreeCoreAutomationSettings = field(
        default_factory=TreeCoreAutomationSettings
    )

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
        if len(self.step_presets) != 4:
            raise ValueError("step_presets must contain exactly 4 values")
        if any(p <= 0 for p in self.step_presets):
            raise ValueError("all step_presets values must be positive")
        if self.starting_height_nm < 0:
            raise ValueError("starting_height_nm must be non-negative")


class MotionSystemSettingsManager(ConfigManager[MotionSystemSettings]):
    """Configuration manager for motion system settings."""

    def __init__(
        self,
        *,
        root_dir: str | Path = "./config/motion_system",
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

        valid_fields = {f.name for f in fields(MotionSystemSettings)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Ensure step_presets is always a list of exactly 4 ints, padding with
        # defaults if an older config file has fewer entries.
        raw_presets = filtered_data.get("step_presets", _DEFAULT_STEP_PRESETS_NM)
        if not isinstance(raw_presets, list):
            raw_presets = list(_DEFAULT_STEP_PRESETS_NM)
        padded = (list(raw_presets) + list(_DEFAULT_STEP_PRESETS_NM))[:4]
        filtered_data["step_presets"] = padded

        # Deserialise the nested CameraCalibrationPosition if present.
        raw_cal_pos = filtered_data.get("camera_calibration_position")
        if isinstance(raw_cal_pos, dict):
            valid_cal_fields = {f.name for f in fields(CameraCalibrationPosition)}
            filtered_data["camera_calibration_position"] = CameraCalibrationPosition(
                **{k: v for k, v in raw_cal_pos.items() if k in valid_cal_fields}
            )
        else:
            filtered_data.pop("camera_calibration_position", None)

        # Deserialise the nested TreeCoreAutomationSettings if present.
        raw_tca = filtered_data.get("tree_core_automation")
        if isinstance(raw_tca, dict):
            valid_tca_fields = {f.name for f in fields(TreeCoreAutomationSettings)}
            tca_data = {k: v for k, v in raw_tca.items() if k in valid_tca_fields}

            raw_slots = tca_data.get("slots", [])
            valid_slot_fields = {f.name for f in fields(TreeCoreSlot)}
            tca_data["slots"] = [
                TreeCoreSlot(**{k: v for k, v in s.items() if k in valid_slot_fields})
                if isinstance(s, dict) else TreeCoreSlot()
                for s in raw_slots
            ]

            filtered_data["tree_core_automation"] = TreeCoreAutomationSettings(**tca_data)
        else:
            filtered_data.pop("tree_core_automation", None)

        return MotionSystemSettings(**filtered_data)

    def to_dict(self, settings: MotionSystemSettings) -> dict[str, Any]:
        return asdict(settings)