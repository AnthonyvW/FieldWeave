"""
Application context for managing shared resources and state.
Provides a singleton pattern for accessing camera and other shared resources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from PySide6.QtWidgets import QApplication
from camera.camera_manager import CameraManager
from camera.camera_enumerator import CameraInfo
from camera.cameras.base_camera import BaseCamera
from machine_vision.machine_vision_manager import MachineVisionManager
from post_processing.post_processing_manager import PostProcessingManager
from common.logger import info, error, warning, debug
from common.fieldweaveConfig import (
    FIELDWEAVE_VERSION,
    FieldWeaveSettingsManager,
    FieldWeaveSettings,
)
from common.image_name_formatter import ImageNameFormatter
from motion.motion_controller_manager import MotionControllerManager
from motion.motion_controller_manager import MotionState
from common.updater import Updater

if TYPE_CHECKING:
    from UI.settings.settings_main import SettingsDialog
    from UI.widgets.toast_widget import ToastManager
    from UI.widgets.camera_preview import CameraPreview
    from UI.widgets.update_notifier import UpdateNotifier

class AppContext:
    """
    Singleton application context managing shared resources.
    """
    _instance: AppContext | None = None
    _initialized = False

    def __new__(cls) -> AppContext:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._camera_manager: CameraManager | None = None
        # Opened at startup only because the saved camera was missing; must not overwrite the saved choice.
        self._fallback_camera_info: CameraInfo | None = None
        self._settings_dialog: SettingsDialog | None = None
        self._settings_manager: FieldWeaveSettingsManager | None = None
        self._settings: FieldWeaveSettings | None = None
        self._toast_manager: ToastManager | None = None
        self._main_window = None
        self._motion_manager: MotionControllerManager | None = None
        self._machine_vision_manager: MachineVisionManager | None = None
        self._post_processing_manager: PostProcessingManager | None = None
        self._camera_preview: CameraPreview | None = None
        self._updater: Updater | None = None
        self._update_notifier: UpdateNotifier | None = None
        self._initialized = True
        self._cleaned_up: bool = False

        self.image_name_formatter = ImageNameFormatter(pad_positions=True)

        self._load_settings()
        self._initialize_motion_manager()
        self._initialize_camera_manager()
        self._initialize_machine_vision_manager()
        self._initialize_post_processing_manager()
        self._initialize_updater()

        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.cleanup)
        else:
            warning("AppContext: QApplication not yet created — cleanup will not be wired automatically")

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    @property
    def camera_manager(self) -> CameraManager:
        if self._camera_manager is None:
            if self._cleaned_up:
                raise RuntimeError("AppContext has already been cleaned up")
            self._initialize_camera_manager()
        return self._camera_manager

    @property
    def camera(self) -> BaseCamera | None:
        if self._camera_manager is None:
            return None
        return self._camera_manager.active_camera

    @property
    def has_camera(self) -> bool:
        return self.camera is not None

    # ------------------------------------------------------------------
    # Camera preview
    # ------------------------------------------------------------------

    @property
    def camera_preview(self) -> CameraPreview | None:
        return self._camera_preview

    def register_camera_preview(self, preview: CameraPreview) -> None:
        """Register the application-wide camera preview widget."""
        self._camera_preview = preview

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    @property
    def motion(self) -> MotionControllerManager | None:
        return self._motion_manager

    @property
    def has_motion(self) -> bool:
        return self._motion_manager is not None and self._motion_manager.get_state() == MotionState.READY

    # ------------------------------------------------------------------
    # Machine vision
    # ------------------------------------------------------------------

    @property
    def machine_vision(self) -> MachineVisionManager:
        if self._machine_vision_manager is None:
            if self._cleaned_up:
                raise RuntimeError("AppContext has already been cleaned up")
            self._initialize_machine_vision_manager()
        return self._machine_vision_manager

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    @property
    def post_processing(self) -> PostProcessingManager:
        """
        Get the post-processing manager.

        Start a routine via::

            from post_processing.post_processing_routines import StitchAndMeasureRoutine
            routine = StitchAndMeasureRoutine(ctx.post_processing.settings, input_folder="...")
            ctx.post_processing.start_routine(routine)
        """
        if self._post_processing_manager is None:
            if self._cleaned_up:
                raise RuntimeError("AppContext has already been cleaned up")
            self._initialize_post_processing_manager()
        return self._post_processing_manager

    # ------------------------------------------------------------------
    # Updater
    # ------------------------------------------------------------------

    @property
    def updater(self) -> Updater | None:
        return self._updater

    @property
    def update_notifier(self) -> UpdateNotifier | None:
        return self._update_notifier

    def register_update_notifier(self, notifier: UpdateNotifier) -> None:
        """Register the application-wide update notifier widget."""
        self._update_notifier = notifier

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    @property
    def settings(self) -> FieldWeaveSettings | None:
        return self._settings

    def apply_settings(self, settings: FieldWeaveSettings) -> bool:
        """Replace the shared settings instance without saving to disk.

        Returns True on success, False if *settings* fails validation.
        """
        try:
            settings.validate()
        except ValueError as exc:
            error(f"AppContext: invalid settings — {exc}")
            return False
        self._settings = settings
        return True

    @property
    def settings_manager(self) -> FieldWeaveSettingsManager | None:
        return self._settings_manager

    @property
    def settings_dialog(self) -> SettingsDialog | None:
        return self._settings_dialog

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    @property
    def toast(self) -> ToastManager | None:
        return self._toast_manager

    @property
    def current_version(self) -> str:
        return FIELDWEAVE_VERSION

    def register_main_window(self, window) -> None:
        self._main_window = window
        if self._toast_manager is None:
            from UI.widgets.toast_widget import ToastManager  # pylint: disable=import-outside-toplevel
            self._toast_manager = ToastManager(window)

    def register_settings_dialog(self, dialog: SettingsDialog) -> None:
        self._settings_dialog = dialog

    def open_settings(self, category: str) -> None:
        if self._settings_dialog:
            self._settings_dialog.open_to(category)
            self._settings_dialog.show()
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()

    def open_tab(self, label: str) -> bool:
        """Switch the main window to the tab whose header text matches *label*. Returns whether a matching tab was found."""
        tabs = getattr(self._main_window, "tabs", None)
        if tabs is None:
            return False
        for index in range(tabs.count()):
            if tabs.tabText(index) == label:
                tabs.setCurrentIndex(index)
                return True
        return False

    # ------------------------------------------------------------------
    # Internal initialisation
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        try:
            self._settings_manager = FieldWeaveSettingsManager()
            first_start = not self._settings_manager.active_path().exists()
            self._settings = self._settings_manager.load()
            self._settings.first_start = first_start
            if first_start:
                self._settings_manager.save(self._settings)
            info(f"FieldWeave settings loaded - running v{FIELDWEAVE_VERSION}")
            if self._settings.show_patchnotes:
                info("New version detected - patch notes should be displayed")
        except Exception as e:
            error(f"Failed to load FieldWeave settings: {e}")
            self._settings = FieldWeaveSettings()
            warning("Using default FieldWeave settings")

    def _initialize_camera_manager(self) -> None:
        if self._camera_manager is not None:
            return
        try:
            info("Initializing camera manager...")
            self._camera_manager = CameraManager()
            self._camera_manager.active_camera_changed.connect(self._on_active_camera_changed)
            cameras = self._camera_manager.enumerate_cameras()
            if cameras:
                saved = self._settings.camera_manager.last_camera if self._settings else None
                target = self._camera_manager.find_saved_camera(saved) if saved else None
                if target is not None:
                    info(f"Auto-opening last used camera: {target.display_name}")
                else:
                    target = cameras[0]
                    if saved is not None:
                        warning(
                            f"Last used camera '{saved.display_name}' not found - "
                            f"opening first available camera: {target.display_name}"
                        )
                        self._fallback_camera_info = target
                    else:
                        info("Auto-opening first available camera...")
                if self._camera_manager.switch_camera(target, start_streaming=True):
                    debug("Camera opened and streaming started successfully")
                else:
                    warning(f"Failed to auto-open camera: {target.display_name}")
        except Exception as e:
            error(f"Failed to initialize camera manager: {e}")
            self._camera_manager = None

    def _on_active_camera_changed(self, camera_info: CameraInfo | None) -> None:
        if camera_info is None:
            return

        fallback = self._fallback_camera_info
        self._fallback_camera_info = None
        if camera_info is fallback:
            return

        if self._settings is None or self._settings_manager is None:
            return

        saved = CameraManager.saved_camera_from_info(camera_info)
        if self._settings.camera_manager.last_camera == saved:
            return

        self._settings.camera_manager.last_camera = saved
        if self._settings_manager.save(self._settings):
            info(f"Saved last used camera: {camera_info.display_name}")
        else:
            error("Failed to save last used camera")

    def _initialize_motion_manager(self) -> None:
        try:
            info("Initializing motion controller manager...")
            self._motion_manager = MotionControllerManager()
            info("Motion controller manager started (connecting in background...)")
        except Exception as e:
            error(f"Failed to start motion controller manager: {e}")
            self._motion_manager = None

    def _initialize_machine_vision_manager(self) -> None:
        try:
            info("Initializing machine vision manager...")
            self._machine_vision_manager = MachineVisionManager()
            info("Machine vision manager started")
        except Exception as e:
            error(f"Failed to start machine vision manager: {e}")
            self._machine_vision_manager = None

    def _initialize_post_processing_manager(self) -> None:
        if self._post_processing_manager is not None:
            return
        try:
            info("Initializing post-processing manager...")
            self._post_processing_manager = PostProcessingManager()
            info("Post-processing manager started")
        except Exception as e:
            error(f"Failed to start post-processing manager: {e}")
            self._post_processing_manager = None

    def _initialize_updater(self) -> None:
        if self._updater is not None:
            return
        self._updater = Updater()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Cleanup all resources. Safe to call more than once."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        if self._post_processing_manager:
            self._post_processing_manager.shutdown()

        if self._machine_vision_manager:
            self._machine_vision_manager.shutdown()

        if self._motion_manager:
            self._motion_manager.shutdown()

        if self._camera_manager:
            self._camera_manager.cleanup()

        self._camera_manager = None
        self._motion_manager = None
        self._machine_vision_manager = None
        self._post_processing_manager = None
        self._camera_preview = None
        self._updater = None
        self._update_notifier = None
        self._settings_dialog = None
        self._settings_manager = None
        self._settings = None
        self._toast_manager = None
        self._main_window = None


# Global instance accessors
def get_app_context() -> AppContext:
    return AppContext()


def get_fieldweave_version() -> str:
    return FIELDWEAVE_VERSION


def open_settings(category: str) -> None:
    AppContext().open_settings(category)


def open_tab(label: str) -> bool:
    return AppContext().open_tab(label)