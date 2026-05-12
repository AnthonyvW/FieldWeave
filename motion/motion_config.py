from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

from common.generic_config import ConfigManager

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
class ZStackScanSettings:
    """Persisted defaults for the Z-stack scan and focus stack routine.

    Scan parameters
    ---------------
    step_nm:
        Distance between capture positions in nanometres.
    approach_distance_nm:
        Overshoot distance before the scan begins to eliminate backlash.
        0 disables the approach move.

    Focus stack parameters
    ----------------------
    All fields correspond directly to the controls in FocusStackWidget and
    are forwarded to FocusStackRoutineConfig when a scan is started.
    """

    # Scan
    step_nm: int = 200_000
    approach_distance_nm: int = 400_000

    # Focus stack — top-level toggles
    run_focus_stack: bool = True
    keep_size: bool = True

    # Focus stack — advanced
    no_align: bool = False
    crop: bool = False
    sharpness: float = 4.0
    cull_enabled: bool = False
    cull_threshold: float = 0.6
    slab_enabled: bool = False
    slab_size: int = 20
    slab_overlap: int = 5
    workers: int = 3


@dataclass
class AreaScanSettings:
    """Persisted defaults for the Z-stack area scan routine.

    Scan parameters
    ---------------
    x_step_nm:
        Step size along X between grid positions, in nanometres.
    y_step_nm:
        Step size along Y between grid positions, in nanometres.
    z_step_nm:
        Distance between Z capture positions within each stack, in nanometres.
    z_start_nm:
        Default near end of the Z range, in nanometres.
    z_end_nm:
        Default far end of the Z range, in nanometres.

    Focus stack parameters
    ----------------------
    All fields correspond directly to the controls in FocusStackWidget and
    are forwarded to FocusStackRoutineConfig when a scan is started.
    """

    # Scan
    x_step_nm: int = 1_000_000
    y_step_nm: int = 1_000_000
    z_step_nm: int = 200_000

    # Focus stack — top-level toggles
    run_focus_stack: bool = True
    keep_size: bool = True

    # Focus stack — advanced
    no_align: bool = False
    crop: bool = False
    sharpness: float = 4.0
    cull_enabled: bool = False
    cull_threshold: float = 0.6
    slab_enabled: bool = False
    slab_size: int = 20
    slab_overlap: int = 5
    workers: int = 3


@dataclass
class AutomationSettings:
    """General settings shared across all automation routines.

    Overlap percentages express what fraction of each captured frame overlaps
    with the next frame along that axis, as a value from 0 to 100.

    Settle times are delays inserted after a move completes, before the next
    operation (capture or autofocus), to allow vibration to decay:

    settle_x_ms:
        Settle time after a move that changes X, in milliseconds.
    settle_y_ms:
        Settle time after a move that changes Y, in milliseconds.
    settle_z_ms:
        Settle time after a move that changes Z only, in milliseconds.
    settle_travel_ms:
        Settle time after a multi-axis travel move (e.g. moving to a new
        slot or returning to a start position), in milliseconds.
    """

    overlap_x_pct: float = 50.0
    overlap_y_pct: float = 50.0
    capture_timeout_ms: int = 5000
    settle_x_ms: int = 200
    settle_y_ms: int = 200
    settle_z_ms: int = 10
    settle_travel_ms: int = 200

    @property
    def overlap_x(self) -> int:
        return int(round(self.overlap_x_pct))

    @property
    def overlap_y(self) -> int:
        return int(round(self.overlap_y_pct))


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

    # General automation settings shared across routines.
    automation: AutomationSettings = field(
        default_factory=AutomationSettings
    )

    # Z-stack scan and focus stack defaults.
    z_stack_scan: ZStackScanSettings = field(
        default_factory=ZStackScanSettings
    )

    # Z-stack area scan and focus stack defaults.
    z_stack_area_scan: AreaScanSettings = field(
        default_factory=AreaScanSettings
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
        if not (0.0 <= self.automation.overlap_x_pct <= 100.0):
            raise ValueError("overlap_x_pct must be between 0 and 100")
        if not (0.0 <= self.automation.overlap_y_pct <= 100.0):
            raise ValueError("overlap_y_pct must be between 0 and 100")
        if self.automation.capture_timeout_ms <= 0:
            raise ValueError("capture_timeout_ms must be positive")
        if self.automation.settle_x_ms < 0:
            raise ValueError("settle_x_ms must be non-negative")
        if self.automation.settle_y_ms < 0:
            raise ValueError("settle_y_ms must be non-negative")
        if self.automation.settle_z_ms < 0:
            raise ValueError("settle_z_ms must be non-negative")
        if self.automation.settle_travel_ms < 0:
            raise ValueError("settle_travel_ms must be non-negative")
        if self.z_stack_scan.step_nm <= 0:
            raise ValueError("z_stack_scan.step_nm must be positive")
        if self.z_stack_scan.approach_distance_nm < 0:
            raise ValueError("z_stack_scan.approach_distance_nm must be non-negative")
        if not (1.0 <= self.z_stack_scan.sharpness <= 8.0):
            raise ValueError("z_stack_scan.sharpness must be between 1.0 and 8.0")
        if not (0.0 <= self.z_stack_scan.cull_threshold <= 1.0):
            raise ValueError("z_stack_scan.cull_threshold must be between 0.0 and 1.0")
        if self.z_stack_scan.slab_size < 2:
            raise ValueError("z_stack_scan.slab_size must be at least 2")
        if self.z_stack_scan.slab_overlap >= self.z_stack_scan.slab_size:
            raise ValueError("z_stack_scan.slab_overlap must be less than slab_size")
        if not (1 <= self.z_stack_scan.workers <= 16):
            raise ValueError("z_stack_scan.workers must be between 1 and 16")
        if self.z_stack_area_scan.x_step_nm <= 0:
            raise ValueError("z_stack_area_scan.x_step_nm must be positive")
        if self.z_stack_area_scan.y_step_nm <= 0:
            raise ValueError("z_stack_area_scan.y_step_nm must be positive")
        if self.z_stack_area_scan.z_step_nm <= 0:
            raise ValueError("z_stack_area_scan.z_step_nm must be positive")
        if not (1.0 <= self.z_stack_area_scan.sharpness <= 8.0):
            raise ValueError("z_stack_area_scan.sharpness must be between 1.0 and 8.0")
        if not (0.0 <= self.z_stack_area_scan.cull_threshold <= 1.0):
            raise ValueError("z_stack_area_scan.cull_threshold must be between 0.0 and 1.0")
        if self.z_stack_area_scan.slab_size < 2:
            raise ValueError("z_stack_area_scan.slab_size must be at least 2")
        if self.z_stack_area_scan.slab_overlap >= self.z_stack_area_scan.slab_size:
            raise ValueError("z_stack_area_scan.slab_overlap must be less than slab_size")
        if not (1 <= self.z_stack_area_scan.workers <= 16):
            raise ValueError("z_stack_area_scan.workers must be between 1 and 16")

class MotionSystemSettingsManager(ConfigManager[MotionSystemSettings]):
    """Configuration manager for motion system settings."""

    def __init__(
        self,
        *,
        root_dir: str | Path = "./config/motion_system",
        backup_dirname: str = "backups",
        backup_keep: int = 5,
    ) -> None:
        super().__init__(
            config_type="motion_system_settings",
            root_dir=root_dir,
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

        # Deserialise the nested AutomationSettings if present.
        raw_automation = filtered_data.get("automation")
        if isinstance(raw_automation, dict):
            valid_automation_fields = {f.name for f in fields(AutomationSettings)}
            filtered_data["automation"] = AutomationSettings(
                **{k: v for k, v in raw_automation.items() if k in valid_automation_fields}
            )
        else:
            filtered_data.pop("automation", None)

        # Deserialise the nested ZStackScanSettings if present.
        raw_z_stack = filtered_data.get("z_stack_scan")
        if isinstance(raw_z_stack, dict):
            valid_z_stack_fields = {f.name for f in fields(ZStackScanSettings)}
            filtered_data["z_stack_scan"] = ZStackScanSettings(
                **{k: v for k, v in raw_z_stack.items() if k in valid_z_stack_fields}
            )
        else:
            filtered_data.pop("z_stack_scan", None)

        # Deserialise the nested AreaScanSettings if present.
        raw_z_stack_area = filtered_data.get("z_stack_area_scan")
        if isinstance(raw_z_stack_area, dict):
            valid_z_stack_area_fields = {f.name for f in fields(AreaScanSettings)}
            filtered_data["z_stack_area_scan"] = AreaScanSettings(
                **{k: v for k, v in raw_z_stack_area.items() if k in valid_z_stack_area_fields}
            )
        else:
            filtered_data.pop("z_stack_area_scan", None)

        return MotionSystemSettings(**filtered_data)

    def to_dict(self, settings: MotionSystemSettings) -> dict[str, Any]:
        return asdict(settings)