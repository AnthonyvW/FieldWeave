from __future__ import annotations

from dataclasses import dataclass
from common.generic_config import ConfigManager, DEFAULT_FILENAME, ACTIVE_FILENAME

@dataclass
class AutomationSettings:
    # --- Machine Vision ---
    tile_size: int = 48
    stride: int = 48
    top_percent: float = 0.15
    min_score: float = 50.0
    soft_min_score: float = 35.0

    # Edge insets as % from each edge (0..1), used to restrict focus region
    inset_left_pct: float = 0.10
    inset_top_pct: float = 0.00
    inset_right_pct: float = 0.10
    inset_bottom_pct: float = 0.00
    scale_factor: float = 1.0

    # --- Image Name Formatter ---
    # Only the template string is externalized for now
    image_name_template: str = "Y{y} X{x} Z{z} F{f}"
    zero_pad: bool = True
    delimiter: str = "."

def make_automation_settings_manager(
    *,
    root_dir: str = "./config/automation",
    default_filename: str = "default_settings.yaml",
    backup_dirname: str = "backups",
    backup_keep: int = 5,
) -> ConfigManager[AutomationSettings]:
    return ConfigManager[AutomationSettings](
        AutomationSettings,
        root_dir=root_dir,
        default_filename=default_filename,
        backup_dirname=backup_dirname,
        backup_keep=backup_keep,
    )

AutomationSettingsManager = make_automation_settings_manager(
    root_dir="./config/automation",
    default_filename=DEFAULT_FILENAME,
    backup_dirname="backups",
    backup_keep=5,
)
