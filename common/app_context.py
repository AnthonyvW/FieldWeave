"""
Application context for managing shared resources and state.
Provides a singleton pattern for accessing camera and other shared resources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from camera.camera_manager import CameraManager
from camera.cameras.base_camera import BaseCamera
from common.logger import info, error, warning, debug
from common.fieldweaveConfig import FieldWeaveSettingsManager, FieldWeaveSettings
from motion.motion_controller_manager import MotionControllerManager
from motion.motion_controller_manager import MotionState

if TYPE_CHECKING:
    from UI.settings.settings_main import SettingsDialog
    from UI.widgets.toast_widget import ToastManager

FIELDWEAVE_VERSION = "1.2"


class AppContext:
    """
    Singleton application context managing shared resources.
    """
    _instance: AppContext | None = None

    def __new__(cls) -> AppContext:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._camera_manager: CameraManager | None = None
        self._settings_dialog: SettingsDialog | None = None
        self._settings_manager: FieldWeaveSettingsManager | None = None
        self._settings: FieldWeaveSettings | None = None
        self._toast_manager: ToastManager | None = None
        self._main_window = None
        self._motion_manager: MotionControllerManager | None = None
        self._initialized = True
        self._cleaned_up: bool = False

        # Load settings
        self._load_settings()

        # Initialize camera manager
        self._initialize_camera_manager()

        # Initialize motion controller manager
        self._initialize_motion_manager()

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    @property
    def camera_manager(self) -> CameraManager:
        """
        Get the camera manager instance.
        Use this to enumerate cameras, switch cameras, start/stop streaming, etc.
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
        """Check if there is an active camera."""
        return self.camera is not None

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    @property
    def motion(self) -> MotionControllerManager | None:
        """
        Get the motion controller manager.

        Returns None if the manager failed to initialise.  Callers should
        check is_ready() before issuing moves if they need a homed machine.
        """
        return self._motion_manager

    @property
    def has_motion(self) -> bool:
        """Return True if the motion manager is available and the controller is ready."""
        return self._motion_manager is not None and self._motion_manager.get_state() == MotionState.READY

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    @property
    def settings(self) -> FieldWeaveSettings | None:
        """Get the FieldWeave settings."""
        return self._settings

    @property
    def settings_manager(self) -> FieldWeaveSettingsManager | None:
        """Get the FieldWeave settings manager."""
        return self._settings_manager

    @property
    def settings_dialog(self) -> SettingsDialog | None:
        """Get the settings dialog instance."""
        return self._settings_dialog

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    @property
    def toast(self) -> ToastManager | None:
        """Get the toast manager instance."""
        return self._toast_manager

    @property
    def current_version(self) -> str:
        """Get the current FieldWeave version."""
        return FIELDWEAVE_VERSION

    def register_main_window(self, window) -> None:
        """Register the main window instance."""
        self._main_window = window
        if self._toast_manager is None:
            from UI.widgets.toast_widget import ToastManager
            self._toast_manager = ToastManager(window)

    def register_settings_dialog(self, dialog: SettingsDialog) -> None:
        """Register the settings dialog instance."""
        self._settings_dialog = dialog

    def open_settings(self, category: str) -> None:
        """
        Open settings dialog to a specific category.

        Args:
            category: Name of the settings category to open to.
        """
        if self._settings_dialog:
            self._settings_dialog.open_to(category)
            self._settings_dialog.show()
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()

    # ------------------------------------------------------------------
    # Internal initialisation
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        """Load FieldWeave application settings."""
        try:
            self._settings_manager = FieldWeaveSettingsManager()
            self._settings = self._settings_manager.load()

            info(f"FieldWeave settings loaded - running v{FIELDWEAVE_VERSION}")

            if self._settings.show_patchnotes:
                info("New version detected - patch notes should be displayed")

        except Exception as e:
            error(f"Failed to load FieldWeave settings: {e}")
            self._settings = FieldWeaveSettings()
            warning("Using default FieldWeave settings")

    def _initialize_camera_manager(self) -> None:
        """Initialize the camera manager and open first available camera."""
        if self._camera_manager is not None:
            return

        try:
            info("Initializing camera manager...")
            self._camera_manager = CameraManager()

            info("Enumerating cameras...")
            cameras = self._camera_manager.enumerate_cameras()

            if cameras:
                info("Auto-opening first available camera...")
                if self._camera_manager.open_first_available(start_streaming=True):
                    debug("Camera opened and streaming started successfully")
                else:
                    warning("Failed to auto-open first camera")
            else:
                warning("No cameras found during enumeration")

        except Exception as e:
            error(f"Failed to initialize camera manager: {e}")
            self._camera_manager = None

    def _initialize_motion_manager(self) -> None:
        """Start the motion controller manager (controller connects on its own thread)."""
        try:
            info("Initializing motion controller manager...")
            self._motion_manager = MotionControllerManager()
            # Connection and homing happen on the controller's worker thread.
            # Callers can await readiness via app_context.motion.wait_until_ready().
            info("Motion controller manager started (connecting in background...)")
        except Exception as e:
            error(f"Failed to start motion controller manager: {e}")
            self._motion_manager = None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Cleanup all resources. Safe to call more than once."""
        if self._cleaned_up:
            return
        self._cleaned_up = True


        if self._motion_manager:
            self._motion_manager.shutdown()
            
        if self._camera_manager:
            self._camera_manager.cleanup()

        self._camera_manager = None
        self._motion_manager = None
        self._settings_dialog = None
        self._settings_manager = None
        self._settings = None
        self._toast_manager = None
        self._main_window = None


# Global instance accessors
def get_app_context() -> AppContext:
    """Get the global application context."""
    return AppContext()


def open_settings(category: str) -> None:
    AppContext().open_settings(category)