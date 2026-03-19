from __future__ import annotations

import threading
from typing import Callable, TYPE_CHECKING

from .motion_controller import MotionController, MotionState
from .models import Position

if TYPE_CHECKING:
    from motion.routines.automation_routine import AutomationRoutine


class MotionControllerManager:
    """
    Manages the lifecycle of a :class:`MotionController`.

    The controller's internal worker thread is started automatically on
    construction (inside MotionController.__init__).  This manager provides
    a clean interface for the rest of the application to interact with it
    without needing to hold a direct reference to the controller, and makes
    it straightforward to swap or restart the controller if needed.

    It also owns the currently active :class:`AutomationRoutine`, ensuring
    only one routine runs at a time and exposing pause / resume / stop
    controls.

    Typical usage
    -------------
    manager = MotionControllerManager()
    manager.wait_until_ready()           # blocks until homed
    manager.move_axis("x", 1)
    manager.set_speed(80_000)            # 0.08 mm steps

    routine = ZStackScan(manager, ...)
    manager.start_routine(routine)
    manager.pause_routine()
    manager.resume_routine()
    manager.stop_routine()

    manager.shutdown()
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._controller: MotionController | None = None
        self._active_routine: AutomationRoutine | None = None
        self._start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _start(self) -> None:
        """Instantiate and start the controller (non-blocking)."""
        with self._lock:
            self._controller = MotionController()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """
        Block until the controller has connected and homed.

        Returns True if ready, False if *timeout* elapsed.
        Raises the underlying connection exception on failure.
        """
        ctrl = self._get_controller()
        return ctrl.wait_until_ready(timeout)

    def is_ready(self) -> bool:
        """Return True if the controller is connected and homed."""
        ctrl = self._controller
        return ctrl is not None and ctrl.is_ready()

    def shutdown(self) -> None:
        """Cleanly shut down any running routine, the controller, and release the serial port."""
        self.stop_routine()
        with self._lock:
            if self._controller is not None:
                self._controller.shutdown()
                self._controller = None

    def restart(self) -> None:
        """Shut down any existing controller and start a fresh one."""
        self.shutdown()
        self._start()

    # ------------------------------------------------------------------
    # Automation routine management
    # ------------------------------------------------------------------

    @property
    def active_routine(self) -> AutomationRoutine | None:
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

    def start_routine(self, routine: AutomationRoutine) -> None:
        """
        Start *routine*, replacing any previously finished routine.

        Raises :class:`RuntimeError` if a routine is already running.
        """
        if self.routine_running:
            raise RuntimeError(
                "A routine is already running. Call stop_routine() first."
            )
        self._active_routine = routine
        routine.start()

    def pause_routine(self) -> None:
        """Pause the active routine (no-op if none is running)."""
        if self._active_routine is not None:
            self._active_routine.pause()

    def resume_routine(self) -> None:
        """Resume a paused routine (no-op if none is running)."""
        if self._active_routine is not None:
            self._active_routine.resume()

    def stop_routine(self) -> None:
        """
        Stop the active routine and wait for its thread to exit.

        Blocks for up to 10 seconds for a clean shutdown.
        """
        if self._active_routine is not None:
            self._active_routine.stop()
            self._active_routine.wait(timeout=10)
            self._active_routine = None

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    def _get_controller(self) -> MotionController:
        ctrl = self._controller
        if ctrl is None:
            raise RuntimeError("MotionControllerManager has been shut down")
        return ctrl

    # ------------------------------------------------------------------
    # Public motion API
    # ------------------------------------------------------------------

    def move_axis(self, axis: str, direction: int) -> bool:
        """
        Jog *axis* by one speed-step in *direction* (+1 or -1).

        Returns False if the move would exceed axis limits.
        """
        return self._get_controller().move_axis(axis, direction)

    def move(self, axis: str, amount_nm: int, *, is_relative: bool = True) -> bool:
        """
        Move *axis* by *amount_nm* nanometres.

        When *is_relative* is True (the default) *amount_nm* is a delta from
        the current position.  When False it is an absolute target in
        nanometres.

        Returns False if the move would exceed axis limits.
        """
        return self._get_controller().move(axis, amount_nm, is_relative=is_relative)

    def move_to_position(self, position: Position) -> None:
        """Enqueue an absolute move to *position* (coordinates in nanometres)."""
        self._get_controller().move_to_position(position)

    def home(self) -> None:
        """Enqueue a homing sequence."""
        self._get_controller().home()

    def set_speed(self, speed_nm: int) -> None:
        """
        Set the jog step size in nanometres.

        Clamped to a minimum of config.step_size (hardware resolution).
        """
        self._get_controller().set_speed(speed_nm)

    def get_position(self) -> Position:
        """Return the current position (coordinates in nanometres)."""
        return self._get_controller().get_position()

    def get_bed_size(self) -> Position:
        """Return the machine's maximum extents as a Position (nanometres)."""
        return self._get_controller().get_bed_size()

    def reset_fault(self) -> None:
        """Clear a faulted state so the controller can accept commands again."""
        self._get_controller().reset_fault()

    @property
    def is_faulted(self) -> bool:
        return self.get_state() == MotionState.FAULTED

    def get_state(self) -> str:
        """Return the current :class:`MotionState` of the controller.

        Returns ``MotionState.FAILED`` if the manager has been shut down.
        """
        ctrl = self._controller
        if ctrl is None:
            return MotionState.FAILED
        return ctrl.get_state()

    # ------------------------------------------------------------------
    # Message listeners
    # ------------------------------------------------------------------

    def add_message_listener(self, listener: Callable[[str, bool], None]) -> None:
        """Subscribe to controller messages.  Signature: (text: str, log: bool) -> None."""
        self._get_controller().add_message_listener(listener)

    def remove_message_listener(self, listener: Callable[[str, bool], None]) -> None:
        ctrl = self._controller
        if ctrl is not None:
            ctrl.remove_message_listener(listener)