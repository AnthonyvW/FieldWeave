# config_manager.py
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generic, Iterator, TypeVar, Union
import shutil
import time
import yaml

from common.logger import info, debug, error, warning

# File/dir names are generic—usable for ANY config
ACTIVE_FILENAME = "settings.yaml"
BACKUP_DIRNAME = "backups"
BACKUP_KEEP = 5  # keep most recent N backups

S = TypeVar("S")  # Config schema type (must be a dataclass)


class ConfigValidationError(Exception):
    """Raised when settings validation fails."""
    pass


class ConfigManager(Generic[S], ABC):
    """
    Generic YAML-backed config manager for ANY dataclass-based settings.
    Manages a single configuration directory with active settings, defaults, and backups.
    
    All config files include metadata fields:
        - config_type: Identifies which config loader to use
        - config_version: The FieldWeave version that created/last modified this config
    
    Directory structure:
        root_dir/
            settings.yaml          # Active settings
            backups/               # Timestamped backups
                settings.20250128-143052.yaml
                settings.20250128-120301.yaml
    
    Child classes must implement:
        - from_dict(data: dict[str, Any]) -> S: Convert dictionary to settings object
        - to_dict(settings: S) -> dict[str, Any]: Convert settings object to dictionary
    
    Example:
        >>> @dataclass
        ... class MySettings:
        ...     value: int = 10
        ...     def validate(self):
        ...         if self.value < 0:
        ...             raise ValueError("value must be non-negative")
        >>> 
        >>> class MySettingsManager(ConfigManager[MySettings]):
        ...     def __init__(self):
        ...         super().__init__(
        ...             config_type="my_settings",
        ...             root_dir="./config/my_component"
        ...         )
        ...     
        ...     def from_dict(self, data: dict[str, Any]) -> MySettings:
        ...         return MySettings(**data)
        ...     
        ...     def to_dict(self, settings: MySettings) -> dict[str, Any]:
        ...         return asdict(settings)
        >>> 
        >>> manager = MySettingsManager()
        >>> settings = manager.load()
        >>> settings.value = 20
        >>> manager.save(settings)
    """

    def __init__(
        self,
        config_type: str,
        *,
        root_dir: Union[str, Path] = "./config",
        backup_dirname: str = BACKUP_DIRNAME,
        backup_keep: int = BACKUP_KEEP,
    ) -> None:
        """
        Initialize the config manager.
        
        Args:
            config_type: Identifier for this config type (e.g., "camera_settings", "fieldweave_settings")
            root_dir: Directory for config files (settings, backups)
            backup_dirname: Name for the backups subdirectory
            backup_keep: Number of backup files to retain (oldest are deleted)
        
        Raises:
            ValueError: If config_type is empty
        """
        if not config_type:
            raise ValueError("config_type must be a non-empty string")
        
        self.config_type = config_type
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dirname = backup_dirname
        self.backup_keep = backup_keep
        
        debug(f"Initialized ConfigManager for '{config_type}' at {self.root_dir}")
    
    @abstractmethod
    def from_dict(self, data: dict[str, Any]) -> S:
        """
        Convert a dictionary (loaded from YAML) into a settings object.
        
        The dictionary will NOT include config_type or config_version fields,
        as those are handled separately by the base class.
        
        Args:
            data: Dictionary containing the settings data
        
        Returns:
            Settings object instance
        
        Raises:
            Any exception appropriate for conversion failures
        """
        pass
    
    @abstractmethod
    def to_dict(self, settings: S) -> dict[str, Any]:
        """
        Convert a settings object into a dictionary for YAML serialization.
        
        Do NOT include config_type or config_version in the returned dictionary,
        as those are added automatically by the base class.
        
        Args:
            settings: Settings object to convert
        
        Returns:
            Dictionary containing the settings data
        """
        pass
    
    def migrate(
        self, 
        data: dict[str, Any], 
        from_version: str, 
        to_version: str
    ) -> dict[str, Any]:
        """
        Migrate config data from one version to another.
        
        Override this method in child classes to handle version migrations.
        The base implementation does nothing (no migration).
        
        Args:
            data: Dictionary containing the config data (without metadata fields)
            from_version: Version the config was created with
            to_version: Current FieldWeave version
        
        Returns:
            Migrated dictionary (or original if no migration needed)
        """
        return data
    
    def get_fieldweave_version(self) -> str:
        """Get the current FieldWeave version."""
        from common.app_context import FIELDWEAVE_VERSION
        return FIELDWEAVE_VERSION
    
    def active_path(self) -> Path:
        """Return path to the active settings file."""
        return self.root_dir / ACTIVE_FILENAME

    def backup_dir(self) -> Path:
        """Return path to the backup directory."""
        bd = self.root_dir / self.backup_dirname
        bd.mkdir(exist_ok=True)
        return bd
    
    def _add_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Add config_type and config_version to a data dictionary.
        
        Args:
            data: Dictionary to add metadata to
        
        Returns:
            New dictionary with metadata fields added
        """
        return {
            "config_type": self.config_type,
            "config_version": self.get_fieldweave_version(),
            **data,
        }
    
    def _extract_metadata(
        self, data: dict[str, Any]
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        """
        Extract metadata and migrate if needed.
        
        Args:
            data: Dictionary loaded from YAML
        
        Returns:
            Tuple of (config_type, config_version, remaining_data)
            The remaining_data will be migrated if version mismatch detected
        """
        data = data.copy()  # Don't modify the original
        
        config_type = data.pop("config_type", None)
        config_version = data.pop("config_version", None)
        
        # Validate config_type matches if present
        if config_type is not None and config_type != self.config_type:
            warning(
                f"Config type mismatch: expected '{self.config_type}', "
                f"got '{config_type}' in file"
            )
        
        # Handle migration if version mismatch
        if config_version is not None:
            current_version = self.get_fieldweave_version()
            
            if config_version != current_version and current_version != "unknown":
                info(
                    f"Config version mismatch: file has v{config_version}, "
                    f"current is v{current_version}. Running migration..."
                )
                try:
                    data = self.migrate(data, config_version, current_version)
                    info("Migration completed successfully")
                except Exception as e:
                    error(f"Migration failed: {e}")
                    # Continue with unmigrated data - child class should handle it
            else:
                debug(f"Config version matches: v{config_version}")
        
        return config_type, config_version, data

    def _load_dict_from_file(self, path: Path) -> dict[str, Any]:
        """
        Load a dictionary from a YAML file.
        
        Args:
            path: Path to the YAML file
        
        Returns:
            Dictionary loaded from the file (empty dict if file is empty)
        
        Raises:
            IOError: If file cannot be read or parsed
        """
        
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            return data or {}
        except Exception as e:
            error(f"Failed to load YAML from {path}: {e}")
            raise IOError(f"Failed to load config from {path}") from e
    
    def _save_dict_to_file(self, data: dict[str, Any], path: Path) -> None:
        """
        Save a dictionary to a YAML file.
        
        Args:
            data: Dictionary to save
            path: Path to save to
        
        Raises:
            IOError: If file cannot be written
        """
        
        try:
            with open(path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            debug(f"Saved config to {path.name}")
        except Exception as e:
            error(f"Failed to save YAML to {path}: {e}")
            raise IOError(f"Failed to save config to {path}") from e

    # -------------------------
    # Validation
    # -------------------------
    
    def _validate(self, settings: S, context: str = "") -> None:
        """
        Validate settings if validate() method exists.
        
        Args:
            settings: Settings instance to validate
            context: Context string for logging (e.g., "after load", "before save")
        
        Raises:
            ConfigValidationError: If validation fails
        """
        if not hasattr(settings, "validate"):
            return
        
        try:
            settings.validate()
            debug(f"Validation passed{f' ({context})' if context else ''}")
        except Exception as e:
            error(
                f"Validation failed{f' ({context})' if context else ''}: {e}"
            )
            raise ConfigValidationError(f"Settings validation failed: {e}") from e

    # -------------------------
    # Backup management
    # -------------------------
    
    def _backup_if_exists(self) -> Path | None:
        """
        Create a timestamped backup of the active settings file if it exists.
        Cleans up old backups to maintain backup_keep limit.
        
        Returns:
            Path to the created backup, or None if no file to backup
        """
        src = self.active_path()
        if not src.exists():
            return None
        
        backup_dir = self.backup_dir()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_name = f"{ACTIVE_FILENAME.split('.')[0]}.{timestamp}.yaml"
        dst = backup_dir / backup_name
        
        try:
            shutil.copy2(src, dst)
            info(f"Created backup: {backup_name}")
        except Exception as e:
            warning(f"Failed to create backup: {e}")
            return None
        
        # Clean up old backups
        self._cleanup_old_backups()
        
        return dst
    
    def _cleanup_old_backups(self) -> None:
        """Remove old backups beyond the configured limit."""
        backups = self.list_backups()
        to_delete = backups[self.backup_keep:]
        
        for backup in to_delete:
            try:
                backup.unlink()
                debug(f"Deleted old backup: {backup.name}")
            except Exception as e:
                warning(f"Failed to delete old backup {backup.name}: {e}")

    # -------------------------
    # Public API
    # -------------------------
    
    def load(self) -> S:
        """
        Load settings with fallback chain: active → fresh instance.
        
        Returns:
            Settings instance (validated if validate() method exists)
        
        Raises:
            ConfigValidationError: If loaded settings fail validation
            IOError: If active settings exist but cannot be read
        """
        ap = self.active_path()
        if ap.exists():
            try:
                data_dict = self._load_dict_from_file(ap)
                _, _, clean_data = self._extract_metadata(data_dict)
                settings = self.from_dict(clean_data)
                self._validate(settings, "active settings")
                info(f"Loaded active settings from {ap.name}")
                return settings
            except ConfigValidationError:
                raise
            except Exception as e:
                error(f"Failed to load active settings from {ap}: {e}")
                raise IOError("Failed to load active settings") from e

        warning(f"No existing settings found for {self.config_type}, using fresh instance")
        settings = self.from_dict({})
        self._validate(settings, "fresh instance")
        return settings

    def load_from_file(self, path: Union[str, Path]) -> S:
        """
        Load settings from an arbitrary file path.
        
        This is useful for loading user-provided or downloaded configuration files.
        
        Args:
            path: Path to the settings file
        
        Returns:
            Settings instance (validated if validate() method exists)
        
        Raises:
            ConfigValidationError: If loaded settings fail validation
            IOError: If file cannot be read or config type mismatch
        """
        p = Path(path)
        try:
            data_dict = self._load_dict_from_file(p)
            file_config_type, _, clean_data = self._extract_metadata(data_dict)
            
            # Check if config_type matches
            if file_config_type is not None and file_config_type != self.config_type:
                error(
                    f"Config type mismatch when loading from {p}: "
                    f"expected '{self.config_type}', got '{file_config_type}'"
                )
                raise IOError(
                    f"Config type mismatch: file is '{file_config_type}', "
                    f"but this manager expects '{self.config_type}'"
                )
            
            settings = self.from_dict(clean_data)
            self._validate(settings, f"file {p.name}")
            info(f"Loaded settings from file: {p}")
            return settings
        except ConfigValidationError:
            raise
        except IOError:
            raise
        except Exception as e:
            error(f"Failed to load settings from {p}: {e}")
            raise IOError(f"Failed to load settings from {path}") from e

    def save(self, settings: S) -> bool:
        """
        Save settings to the active settings file.

        Creates a backup of existing settings before saving.
        Automatically adds config_type and config_version metadata.

        Args:
            settings: Settings instance to save

        Returns:
            True if settings were saved successfully, False otherwise
        """
        try:
            self._validate(settings, "before save")
        except ConfigValidationError:
            return False

        self._backup_if_exists()

        data = self.to_dict(settings)
        data_with_metadata = self._add_metadata(data)

        p = self.active_path()
        try:
            self._save_dict_to_file(data_with_metadata, p)
        except IOError:
            return False

        info(f"Saved settings to {p.name}")
        return True

    def restore_defaults(self) -> S:
        """
        Restore default settings as the active settings.

        Creates a backup of current active settings before restoring.

        Returns:
            The restored default settings

        Raises:
            ConfigValidationError: If default settings fail validation
            IOError: If restore operation fails
        """
        defaults = self.from_dict({})
        self._validate(defaults, "defaults")
        self._backup_if_exists()
        if not self.save(defaults):
            raise IOError("Failed to save restored defaults")
        info("Restored defaults as active settings")
        return defaults

    def list_backups(self) -> list[Path]:
        """List all backup files, sorted by last modified."""
        bd = self.backup_dir()
        try:
            backups = sorted(
                bd.glob(f"{ACTIVE_FILENAME.split('.')[0]}.*.yaml"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            debug(f"Found {len(backups)} backup(s)")
            return backups
        except Exception as e:
            warning(f"Failed to list backups: {e}")
            return []

    @contextmanager
    def edit(self) -> Iterator[S]:
        """
        Context manager for transactional settings editing.
        
        Loads settings, yields for editing, and automatically saves
        on successful exit. If an exception occurs, changes are discarded.
        
        Yields:
            Settings instance for editing
        
        Raises:
            ConfigValidationError: If edited settings fail validation
            IOError: If load or save operations fail
        
        Example:
            >>> with manager.edit() as settings:
            ...     settings.value = 150
            # Auto-saves on successful exit
        """
        debug("Starting edit transaction")
        settings = self.load()
        try:
            yield settings
        except Exception as e:
            error(f"Edit transaction failed: {e}")
            raise
        else:
            if not self.save(settings):
                raise IOError("Edit transaction failed: could not save settings")
            info("Edit transaction completed")