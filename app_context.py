"""
Application context for managing shared resources and state.
Provides a singleton pattern for accessing camera and other shared resources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from camera.camera_manager import CameraManager
from camera.cameras.base_camera import BaseCamera
from logger import info, error, warning, debug
from forgeConfig import ForgeSettingsManager, ForgeSettings

if TYPE_CHECKING:
    from UI.settings.settings_main import SettingsDialog
    from UI.widgets.toast_widget import ToastManager

# Current Forge version - update this when releasing new versions
FORGE_VERSION = "1.2"


class AppContext:
    """
    Singleton application context managing shared resources.
    """
    _instance: AppContext | None = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._camera_manager: CameraManager | None = None
        self._settings_dialog: SettingsDialog | None = None
        self._settings_manager: ForgeSettingsManager | None = None
        self._settings: ForgeSettings | None = None
        self._toast_manager: ToastManager | None = None
        self._main_window = None
        self._initialized = True
        
        # Load settings
        self._load_settings()
        
        # Initialize camera manager
        self._initialize_camera_manager()
    
    @property
    def camera_manager(self) -> CameraManager:
        """
        Get the camera manager instance.
        Use this to enumerate cameras, switch cameras, etc.
        """
        if self._camera_manager is None:
            self._initialize_camera_manager()
        return self._camera_manager
    
    @property
    def camera(self) -> BaseCamera | None:
        """
        Get the currently active camera instance.
        Returns None if no camera is active.
        
        This is a convenience property that delegates to camera_manager.
        """
        if self._camera_manager is None:
            return None
        return self._camera_manager.active_camera
    
    @property
    def has_camera(self) -> bool:
        """Check if there is an active camera"""
        return self.camera is not None
    
    @property
    def settings(self) -> ForgeSettings | None:
        """Get the Forge settings"""
        return self._settings
    
    @property
    def settings_manager(self) -> ForgeSettingsManager | None:
        """Get the Forge settings manager"""
        return self._settings_manager
    
    @property
    def settings_dialog(self) -> SettingsDialog | None:
        """Get the settings dialog instance"""
        return self._settings_dialog
    
    @property
    def toast(self) -> ToastManager | None:
        """Get the toast manager instance"""
        return self._toast_manager
    
    @property
    def current_version(self) -> str:
        """Get the current Forge version"""
        return FORGE_VERSION
    
    def register_main_window(self, window):
        """Register the main window instance"""
        self._main_window = window
        # Initialize toast manager when main window is registered
        if self._toast_manager is None:
            from UI.widgets.toast_widget import ToastManager
            self._toast_manager = ToastManager(window)
    
    def register_settings_dialog(self, dialog: SettingsDialog):
        """Register the settings dialog instance"""
        self._settings_dialog = dialog
    
    def open_settings(self, category: str):
        """
        Open settings dialog to a specific category.
        
        Args:
            category: Name of the settings category to open to
        """
        if self._settings_dialog:
            self._settings_dialog.open_to(category)
            self._settings_dialog.show()
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()
    
    def _load_settings(self):
        """Load Forge application settings"""
        try:
            self._settings_manager = ForgeSettingsManager()
            self._settings = self._settings_manager.load()
            
            info(f"Forge settings loaded - running v{FORGE_VERSION}")
            
            # Check if we should show patch notes
            if self._settings.show_patchnotes:
                info("New version detected - patch notes should be displayed")
                
        except Exception as e:
            error(f"Failed to load Forge settings: {e}")
            # Create default settings if loading fails
            self._settings = ForgeSettings()
            warning("Using default Forge settings")
    
    def _initialize_camera_manager(self):
        """Initialize the camera manager and open first available camera"""
        if self._camera_manager is not None:
            return
        
        try:
            info("Initializing camera manager...")
            self._camera_manager = CameraManager()
            
            # Enumerate cameras
            info("Enumerating cameras...")
            cameras = self._camera_manager.enumerate_cameras()
            
            if cameras:
                # Auto-open the first camera
                info("Auto-opening first available camera...")
                if self._camera_manager.open_first_available():
                    debug("Camera opened successfully during initialization")
                else:
                    warning("Failed to auto-open first camera")
            else:
                warning("No cameras found during enumeration")
                
        except Exception as e:
            error(f"Failed to initialize camera manager: {e}")
            self._camera_manager = None
    
    def cleanup(self):
        """Cleanup resources"""
        if self._camera_manager:
            self._camera_manager.cleanup()
        
        self._camera_manager = None
        self._settings_dialog = None
        self._settings_manager = None
        self._settings = None
        self._toast_manager = None
        self._main_window = None


# Global instance accessor
def get_app_context() -> AppContext:
    """Get the global application context"""
    return AppContext()

def open_settings(category: str):
    AppContext().open_settings(category)
