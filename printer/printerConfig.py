from __future__ import annotations

from dataclasses import dataclass, field

from generic_config import ConfigManager, DEFAULT_FILENAME, ACTIVE_FILENAME

@dataclass
class PrinterSettings():
        FIRMWARE_NAME: str = "Marlin"
        MACHINE_TYPE: str = "Ender-3"
        baud_rate: int = 115200
        max_x: int = 23500  # Maximum X dimension in steps
        max_y: int = 23500  # Maximum Y dimension in steps
        max_z: int = 6000   # Maximum Z dimension in steps
        step_size: int = 4  # minimum distance that can be moved in 0.01mm
        sample_positions: dict[int, dict[str, float]] = field(default_factory=dict)
        calibration_pattern_position: dict[str, float] = field(default_factory=dict)  # X, Y, Z in mm
        
        # Sample calibration positions (for verifying X positions)
        calibration_y: float = 220.0  # Y position for calibration checks (mm)
        calibration_z: float = 26.0   # Z position for calibration checks (mm)
        
        # Camera calibration data
        camera_calibration: dict[str, any] = field(default_factory=dict)  # Stores M_est, M_inv, reference position, etc.
    

def make_printer_settings_manager(
    *,
    root_dir: str = "./config/printers",
    default_filename: str = "default_settings.yaml",
    backup_dirname: str = "backups",
    backup_keep: int = 5,
) -> ConfigManager[PrinterSettings]:
    return ConfigManager[PrinterSettings](
        PrinterSettings,
        root_dir=root_dir,
        default_filename=default_filename,
        backup_dirname=backup_dirname,
        backup_keep=backup_keep,
    )

PrinterSettingsManager = make_printer_settings_manager(
    root_dir="./config/printers",
    default_filename=DEFAULT_FILENAME,
    backup_dirname="backups",
    backup_keep=5,
)