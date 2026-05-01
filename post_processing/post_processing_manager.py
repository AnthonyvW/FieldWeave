"""
post_processing_manager.py

Owns the lifecycle of the active :class:`PostProcessingRoutine` and exposes
a stable public API for the rest of the application.

Mirrors the pattern of :class:`MotionControllerManager`: the manager is the
stable surface callers hold; routines are swappable payloads passed to
:meth:`start_routine`.  Settings are loaded from and saved back to the shared
:class:`FieldWeaveSettingsManager` so post-processing config lives in the
application-wide config file.

Typical usage::

    routine = StitchAndMeasureRoutine(manager.settings, input_folder="/path/to/scan")
    manager.start_routine(routine)
    manager.pause_routine()
    manager.resume_routine()
    manager.stop_routine()
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from common.logger import info, error, warning
from common.fieldweaveConfig import (
    FieldWeaveSettings,
    FieldWeaveSettingsManager,
    PostProcessingSettings,
)

if TYPE_CHECKING:
    from post_processing.routines.post_processing_routine import PostProcessingRoutine, RoutineResult

# Fired when the active routine publishes a status update.
# Signature: (job_name, activity, progress_current, progress_total, eta_seconds) -> None
RoutineStateCallback = Callable[[str, str, int, int, int], None]

# Fired when the active routine finishes (success or failure).
# Signature: (result: RoutineResult) -> None
RoutineCompleteCallback = Callable[["RoutineResult"], None]


class PostProcessingManager:
    """
    Manages the lifecycle of the active :class:`PostProcessingRoutine`.

    Only one routine runs at a time.  The manager wires the routine's
    ``on_state_changed`` callback to its own listener list so UI components
    only need to subscribe once here rather than re-subscribing each time a
    new routine is created.

    Settings are owned here and persisted via :class:`FieldWeaveSettingsManager`.
    Pass the current ``settings`` to each routine at construction time so the
    routine always reads from the live values.

    State change notifications
    --------------------------
    Register a callable with :meth:`add_routine_state_listener` to receive
    live job/activity/progress updates from whatever routine is currently
    active.

    Typical usage
    -------------
    ::

        manager = PostProcessingManager()

        routine = StitchAndMeasureRoutine(manager.settings, input_folder="...")
        manager.start_routine(routine)

        manager.pause_routine()
        manager.resume_routine()
        manager.stop_routine()

        manager.shutdown()
    """

    def __init__(
        self,
        settings_manager: FieldWeaveSettingsManager | None = None,
    ) -> None:
        self._settings_manager = settings_manager or FieldWeaveSettingsManager()
        self._settings: FieldWeaveSettings = self._load_settings()

        self._active_routine: PostProcessingRoutine | None = None
        self._routine_state_listeners: list[RoutineStateCallback] = []
        self._routine_complete_listeners: list[RoutineCompleteCallback] = []

        info("PostProcessingManager: initialised")

    # ------------------------------------------------------------------
    # Settings API
    # ------------------------------------------------------------------

    @property
    def settings(self) -> FieldWeaveSettings:
        """The current application settings (including post-processing config)."""
        return self._settings

    @property
    def post_processing_settings(self) -> PostProcessingSettings:
        """Convenience accessor for the post-processing section of settings."""
        return self._settings.post_processing

    def apply_settings(self, settings: FieldWeaveSettings) -> None:
        """Replace the current settings without saving to disk."""
        try:
            settings.validate()
        except ValueError as exc:
            error(f"PostProcessingManager: invalid settings — {exc}")
            return
        self._settings = settings

    def save_settings(self) -> None:
        """Persist the current settings to the FieldWeave config file."""
        try:
            self._settings_manager.save(self._settings)
            info("PostProcessingManager: settings saved")
        except Exception as exc:
            error(f"PostProcessingManager: failed to save settings — {exc}")

    # ------------------------------------------------------------------
    # Routine state listeners
    # ------------------------------------------------------------------

    def add_routine_state_listener(self, listener: RoutineStateCallback) -> None:
        """Subscribe to routine status updates.

        *listener* is called whenever the active routine reports a change to
        its job name, activity, or progress.
        Signature: ``(job_name, activity, progress_current, progress_total, eta_seconds) -> None``
        """
        self._routine_state_listeners.append(listener)

    def remove_routine_state_listener(self, listener: RoutineStateCallback) -> None:
        try:
            self._routine_state_listeners.remove(listener)
        except ValueError:
            pass

    def _emit_routine_state(
        self,
        job_name: str,
        activity: str,
        progress_current: int,
        progress_total: int,
        eta_seconds: int,
    ) -> None:
        for listener in list(self._routine_state_listeners):
            try:
                listener(job_name, activity, progress_current, progress_total, eta_seconds)
            except Exception as exc:
                warning(f"PostProcessingManager: routine state listener raised: {exc}")

    # ------------------------------------------------------------------
    # Routine complete listeners
    # ------------------------------------------------------------------

    def add_routine_complete_listener(self, listener: RoutineCompleteCallback) -> None:
        """Subscribe to routine completion.

        *listener* is called once when the active routine finishes, whether
        successful or not.
        Signature: ``(result: RoutineResult) -> None``
        """
        self._routine_complete_listeners.append(listener)

    def remove_routine_complete_listener(self, listener: RoutineCompleteCallback) -> None:
        try:
            self._routine_complete_listeners.remove(listener)
        except ValueError:
            pass

    def _emit_routine_complete(self, result: RoutineResult) -> None:
        for listener in list(self._routine_complete_listeners):
            try:
                listener(result)
            except Exception as exc:
                warning(f"PostProcessingManager: routine complete listener raised: {exc}")

    # ------------------------------------------------------------------
    # Routine management
    # ------------------------------------------------------------------

    @property
    def active_routine(self) -> PostProcessingRoutine | None:
        """The currently active routine, or None."""
        return self._active_routine

    @property
    def routine_running(self) -> bool:
        """True if a routine is currently executing (including while paused)."""
        return self._active_routine is not None and self._active_routine.is_running

    @property
    def routine_paused(self) -> bool:
        """True if the active routine is currently paused."""
        return self._active_routine is not None and self._active_routine.is_paused

    def start_routine(self, routine: PostProcessingRoutine) -> None:
        """Start *routine*, replacing any previously finished routine.

        Raises :class:`RuntimeError` if a routine is already running.
        """
        if self.routine_running:
            raise RuntimeError(
                "A post-processing routine is already running. "
                "Call stop_routine() first."
            )
        self._active_routine = routine
        routine.on_state_changed = self._emit_routine_state
        routine.on_complete = self._emit_routine_complete
        info(f"PostProcessingManager: starting routine '{routine.job_name}'")
        routine.start()

    def pause_routine(self) -> None:
        """Pause the active routine (no-op if none is running)."""
        if self._active_routine is not None:
            info("PostProcessingManager: pausing routine")
            self._active_routine.pause()

    def resume_routine(self) -> None:
        """Resume a paused routine (no-op if none is running)."""
        if self._active_routine is not None:
            info("PostProcessingManager: resuming routine")
            self._active_routine.resume()

    def stop_routine(self) -> None:
        """Stop the active routine and block until its thread exits (up to 10 s)."""
        if self._active_routine is not None:
            info("PostProcessingManager: stopping routine")
            self._active_routine.stop()
            self._active_routine.wait(timeout=10)
            self._active_routine = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop any running routine cleanly."""
        self.stop_routine()
        info("PostProcessingManager: shut down")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_settings(self) -> FieldWeaveSettings:
        try:
            settings = self._settings_manager.load()
            info("PostProcessingManager: settings loaded")
            return settings
        except Exception as exc:
            error(f"PostProcessingManager: failed to load settings — {exc}; using defaults")
            return FieldWeaveSettings()