from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from generic_config import ConfigManager
from logger import info

@dataclass
class ForgeSettings:
    """Forge application settings"""

    version: str = "1.2"  # Version from last startup
    show_patchnotes: bool = False  # Runtime flag - set when version changes, not saved
    
    def validate(self) -> None:
        """
        Validate Forge settings.
        
        Raises:
            ValueError: If any setting is invalid
        """
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty string")


class ForgeSettingsManager(ConfigManager[ForgeSettings]):
    """
    Configuration manager for Forge application settings.

    When a version mismatch is detected during load, the migration
    updates the stored version and sets show_patchnotes flag.
    """
    
    def __init__(
        self,
        *,
        root_dir: Union[str, Path] = "./config/forge",
        default_filename: str = "default_settings.yaml",
        backup_dirname: str = "backups",
        backup_keep: int = 5,
    ) -> None:
        super().__init__(
            config_type="forge_settings",
            root_dir=root_dir,
            default_filename=default_filename,
            backup_dirname=backup_dirname,
            backup_keep=backup_keep,
        )
    
    def migrate(
        self,
        data: dict[str, Any],
        from_version: str,
        to_version: str
    ) -> dict[str, Any]:
        """
        Migrate Forge settings and update version.
        
        When version changes:
        1. Updates the stored version to current
        2. Sets show_patchnotes flag (handled after from_dict)
        
        Args:
            data: Dictionary containing settings data
            from_version: Version from the file
            to_version: Current Forge version
        
        Returns:
            Migrated dictionary with updated version
        """
        info(f"Forge version changed: {from_version} -> {to_version}")
        
        # Update version to current
        data["version"] = to_version
        
        # Add any future version-specific migrations here
        
        return data
    
    def from_dict(self, data: dict[str, Any]) -> ForgeSettings:
        """
        Convert dictionary to ForgeSettings object.
        
        Sets show_patchnotes flag if migration occurred.
        
        Args:
            data: Dictionary containing settings data
        
        Returns:
            ForgeSettings instance with show_patchnotes set if needed
        """
        # Handle empty dict (fresh instance)
        if not data:
            settings = ForgeSettings()
        else:
            # Extract only valid fields for ForgeSettings
            valid_fields = {"version"}
            filtered_data = {k: v for k, v in data.items() if k in valid_fields}
            settings = ForgeSettings(**filtered_data)
        
        # If migration happened, set the show_patchnotes flag
        if settings.version != self.get_forge_version():
            settings.show_patchnotes = True
            info("Patch notes flag set - new version detected")
            
            # Save the updated version
            self.save(settings)
        
        return settings
    
    def to_dict(self, settings: ForgeSettings) -> dict[str, Any]:
        """
        Convert ForgeSettings object to dictionary.
        
        Only includes fields that should be saved (excludes show_patchnotes).
        
        Args:
            settings: ForgeSettings instance to convert
        
        Returns:
            Dictionary representation
        """
        return {
            "version": settings.version,
        }
