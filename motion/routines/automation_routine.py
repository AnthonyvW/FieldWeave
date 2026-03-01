"""
Base automation routine framework.

Provides :class:`AutomationRoutine`, an abstract base that all automation
routines inherit from.  Each routine runs on its own daemon thread and
communicates pause / resume / stop signals through threading primitives.

Subclasses implement :meth:`steps` as a generator, yielding after each
logical step.  This preserves state across pauses without resorting to
complex state machines.

Example::

    class MyRoutine(AutomationRoutine):
        def steps(self):
            self.motion.move_axis("x", 1)
            yield
            self.motion.move_axis("x", -1)
            yield
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generator

from common.logger import info, error, warning, debug

if TYPE_CHECKING:
    from motion.motion_controller_manager import MotionControllerManager


class AutomationRoutine(ABC):
    """
    Abstract base class for all automation routines.

    Subclasses must implement :meth:`steps`, which is a generator that yields
    between logical steps.  The runner thread advances the generator, honouring
    pause / stop requests between each yield.

    Parameters
    ----------
    motion:
        The :class:`MotionControllerManager` to use for all moves.
    """

    def __init__(self, motion: MotionControllerManager) -> None:
        self.motion = motion

        self._pause_event = threading.Event()
        self._pause_event.set()          # Not paused initially (set = allowed to run)
        self._stop_event = threading.Event()

        self._thread: threading.Thread | None = None
        self._running = False
        self._finished = threading.Event()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def steps(self) -> Generator[None, None, None]:
        """
        Yield-based generator implementing the routine's logic.

        Yield at each point where the routine can be cleanly paused or
        stopped.  The framework will block at each yield until the routine
        is resumed, or raise :class:`_StopRoutine` to abort execution.
        """

    # ------------------------------------------------------------------
    # Control API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the routine on a background thread.

        Raises :class:`RuntimeError` if a routine is already running.
        """
        if self._running:
            raise RuntimeError(
                f"{type(self).__name__} is already running. "
                "Stop it before starting again."
            )
        self._stop_event.clear()
        self._pause_event.set()
        self._finished.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=type(self).__name__)
        self._thread.start()

    def pause(self) -> None:
        """Pause the routine after the current step completes."""
        if not self._running:
            return
        self._pause_event.clear()
        info(f"[{type(self).__name__}] Paused")

    def resume(self) -> None:
        """Resume a paused routine."""
        if not self._running:
            return
        self._pause_event.set()
        info(f"[{type(self).__name__}] Resumed")

    def stop(self) -> None:
        """
        Request the routine to stop.

        This sets the stop event *and* clears the pause event so a paused
        routine is not stuck waiting forever.  The routine will abort after
        its current step.
        """
        self._stop_event.set()
        self._pause_event.set()   # Unblock a paused routine so it can see stop
        info(f"[{type(self).__name__}] Stop requested")

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the routine finishes (or *timeout* seconds pass).

        Returns True if the routine finished, False if timed out.
        """
        return self._finished.wait(timeout)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True while the routine is executing (including while paused)."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """True if the routine is running but currently paused."""
        return self._running and not self._pause_event.is_set()

    # ------------------------------------------------------------------
    # Internal runner
    # ------------------------------------------------------------------

    def _run(self) -> None:
        info(f"[{type(self).__name__}] Starting")
        try:
            gen = self.steps()
            while True:
                # Honour stop before advancing the generator
                if self._stop_event.is_set():
                    info(f"[{type(self).__name__}] Stopped")
                    break

                # Honour pause — block until resumed or stopped
                self._pause_event.wait()
                if self._stop_event.is_set():
                    info(f"[{type(self).__name__}] Stopped while paused")
                    break

                # Advance one step
                try:
                    next(gen)
                except StopIteration:
                    info(f"[{type(self).__name__}] Completed successfully")
                    break

        except Exception as exc:
            error(f"[{type(self).__name__}] Unhandled exception: {exc}")
            import traceback
            error(traceback.format_exc())
        finally:
            self._running = False
            self._finished.set()

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    def _check_stop(self) -> bool:
        """Return True if a stop has been requested.

        Useful for long blocking operations inside a step where the routine
        cannot simply yield (e.g. a loop inside a single step).
        """
        return self._stop_event.is_set()