from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from common.generic_config import ConfigManager
from common.logger import info


@dataclass
class FieldWeaveSettings:
    """FieldWeave application settings"""

    version: str = "1.2"  # Version from last startup
    show_patchnotes: bool = False  # Runtime flag - set when version changes, not saved

    def validate(self) -> None:
        """
        Validate FieldWeave settings.

        Raises:
            ValueError: If any setting is invalid
        """
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty string")


class FieldWeaveSettingsManager(ConfigManager[FieldWeaveSettings]):
    """
    Configuration manager for FieldWeave application settings.

    When a version mismatch is detected during load, the migration
    updates the stored version and sets show_patchnotes flag.
    """

    def __init__(
        self,
        *,
        root_dir: Union[str, Path] = "./config/fieldweave",
        default_filename: str = "default_settings.yaml",
        backup_dirname: str = "backups",
        backup_keep: int = 5,
    ) -> None:
        super().__init__(
            config_type="fieldweave_settings",
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
        Migrate FieldWeave settings and update version.

        When version changes:
        1. Updates the stored version to current
        2. Sets show_patchnotes flag (handled after from_dict)

        Args:
            data: Dictionary containing settings data
            from_version: Version from the file
            to_version: Current FieldWeave version

        Returns:
            Migrated dictionary with updated version
        """
        info(f"FieldWeave version changed: {from_version} -> {to_version}")

        # Update version to current
        data["version"] = to_version

        # Add any future version-specific migrations here

        return data

    def from_dict(self, data: dict[str, Any]) -> FieldWeaveSettings:
        """
        Convert dictionary to FieldWeaveSettings object.

        Sets show_patchnotes flag if migration occurred.

        Args:
            data: Dictionary containing settings data

        Returns:
            FieldWeaveSettings instance with show_patchnotes set if needed
        """
        # Handle empty dict (fresh instance)
        if not data:
            settings = FieldWeaveSettings()
        else:
            # Extract only valid fields for FieldWeaveSettings
            valid_fields = {"version"}
            filtered_data = {k: v for k,
                             v in data.items() if k in valid_fields}
            settings = FieldWeaveSettings(**filtered_data)

        # If migration happened, set the show_patchnotes flag
        if settings.version != self.get_fieldweave_version():
            settings.show_patchnotes = True
            info("Patch notes flag set - new version detected")

            # Save the updated version
            self.save(settings)

        return settings

    def to_dict(self, settings: FieldWeaveSettings) -> dict[str, Any]:
        """
        Convert FieldWeaveSettings object to dictionary.

        Only includes fields that should be saved (excludes show_patchnotes).

        Args:
            settings: FieldWeaveSettings instance to convert

        Returns:
            Dictionary representation
        """
        return {
            "version": settings.version,
        }
