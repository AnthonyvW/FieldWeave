from __future__ import annotations

import threading
from typing import Callable

from .motion_controller import MotionController, MotionState
from .models import Position


class MotionControllerManager:
    """
    Manages the lifecycle of a :class:`MotionController`.

    The controller's internal worker thread is started automatically on
    construction (inside MotionController.__init__).  This manager provides
    a clean interface for the rest of the application to interact with it
    without needing to hold a direct reference to the controller, and makes
    it straightforward to swap or restart the controller if needed.

    Typical usage
    -------------
    manager = MotionControllerManager()
    manager.wait_until_ready()           # blocks until homed
    manager.move_axis("x", 1)
    manager.set_speed(80_000)            # 0.08 mm steps
    manager.shutdown()
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._controller: MotionController | None = None
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
        """Cleanly shut down the controller and release the serial port."""
        with self._lock:
            if self._controller is not None:
                self._controller.shutdown()
                self._controller = None

    def restart(self) -> None:
        """Shut down any existing controller and start a fresh one."""
        self.shutdown()
        self._start()

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