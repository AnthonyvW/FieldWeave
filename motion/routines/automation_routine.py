"""
Base automation routine framework.

Provides :class:`AutomationRoutine`, an abstract base that all automation
routines inherit from.  Each routine runs on its own daemon thread and
communicates pause / resume / stop signals through threading primitives.

Subclasses implement  `steps` as a generator, yielding after each
logical step.  This preserves state across pauses without resorting to
complex state machines.

Subclasses should set :attr:`job_name` at construction time and call
 `_set_activity` /  `_set_progress` during execution to surface
human-readable status information to the UI.

Subclasses that produce a meaningful result (e.g. autofocus routines) should
call  `_set_result` before returning from  `steps`.  The result is
then available via the :attr:`result` property and is propagated to the
:class:`MotionControllerManager` as :attr:`~MotionControllerManager.last_routine_result`.

Example::

    class MyRoutine(AutomationRoutine):
        job_name = "My Routine"

        def steps(self):
            self._set_activity("Moving right")
            self.motion.move_axis("x", 1)
            yield
            self._set_activity("Moving left")
            self.motion.move_axis("x", -1)
            yield
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, Callable, Generator

from common.logger import info, error, warning, debug

if TYPE_CHECKING:
    from motion.motion_controller_manager import MotionControllerManager

# Signature: (job_name, activity, progress_current, progress_total, eta_seconds) -> None
RoutineStateCallback = Callable[[str, str, int, int, int], None]

# Signature: (result: RoutineResult) -> None
RoutineCompleteCallback = Callable[["RoutineResult"], None]


@dataclass
class RoutineResult:
    """
    Outcome produced by a completed :class:`AutomationRoutine`.

    Attributes
    ----------
    success:
        True if the routine ran to completion and produced a meaningful result.
        False if it was stopped early, aborted due to an error, or produced no
        usable output.
    data:
        Arbitrary key-value payload set by the routine.  Each routine
        documents the keys it populates.  For example, autofocus routines
        store ``z_nm`` and ``focus_score``; the calibration scale routine
        stores ``dpi``, ``image_width``, ``image_height``, and
        ``output_path``.  Use  `get` to retrieve values with a default.
    """

    success: bool
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``data[key]``, or *default* if the key is absent."""
        return self.data.get(key, default)


class AutomationRoutine(ABC):
    """
    Abstract base class for all automation routines.

    Subclasses must implement  `steps`, which is a generator that yields
    between logical steps.  The runner thread advances the generator, honouring
    pause / stop requests between each yield.

    Set the class-level :attr:`job_name` (or override it in ``__init__``) to
    give the routine a human-readable display name.  During execution call
     `_set_activity` and  `_set_progress` to push live status
    information to any registered :attr:`on_state_changed` callback.

    Routines that produce a meaningful result should call  `_set_result`
    before returning from  `steps`.  The result is exposed via the
    :attr:`result` property and forwarded to any registered
    :attr:`on_complete` callback when the routine finishes.

    Parameters
    ----------
    motion:
        The :class:`MotionControllerManager` to use for all moves.
    """

    #: Human-readable name shown in the status bar. Override in subclasses.
    job_name: str = "-"

    def __init__(self, motion: MotionControllerManager) -> None:
        self.motion = motion

        self._pause_event = threading.Event()
        self._pause_event.set()          # Not paused initially (set = allowed to run)
        self._stop_event = threading.Event()

        self._thread: threading.Thread | None = None
        self._running = False
        self._finished = threading.Event()

        # Live status fields — updated by subclasses via helpers below.
        self._activity: str = "-"
        self._progress_current: int = 0
        self._progress_total: int = 0
        self._eta_seconds: int = 0

        # Result produced by the routine (set via _set_result).
        self._result: RoutineResult | None = None

        # Optional callback fired whenever any of the above fields change.
        # Signature: (job_name, activity, progress_current, progress_total, eta_seconds) -> None
        self.on_state_changed: RoutineStateCallback | None = None

        # Optional callback fired once when the routine finishes (success or not).
        # Signature: (result: RoutineResult) -> None
        self.on_complete: RoutineCompleteCallback | None = None

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
    # Result API for subclasses
    # ------------------------------------------------------------------

    def _set_result(self, *, success: bool, **data: Any) -> None:
        """Record the outcome of this routine.

        Should be called by subclasses before returning from  `steps`,
        both on successful completion and on detected failure (e.g. no camera).
        If never called, :attr:`result` will reflect a generic failure.

        Any keyword arguments beyond *success* are stored in
        :attr:`RoutineResult.data` and can be retrieved via
         `RoutineResult.get`.  Each routine should document the keys it
        populates.

        Parameters
        ----------
        success:
            Whether the routine completed successfully and produced usable output.
        **data:
            Arbitrary payload.  For example autofocus routines pass
            ``z_nm=...`` and ``focus_score=...``; the calibration scale
            routine passes ``dpi=...``, ``image_width=...``, etc.
        """
        self._result = RoutineResult(success=success, data=dict(data))

    @property
    def result(self) -> RoutineResult | None:
        """The result produced by this routine, or None if it has not finished."""
        return self._result

    # ------------------------------------------------------------------
    # Status helpers for subclasses
    # ------------------------------------------------------------------

    def _set_activity(self, activity: str) -> None:
        """Update the current activity description and notify listeners."""
        self._activity = activity
        self._notify_state()

    def _set_progress(self, current: int, total: int, eta_seconds: int = 0) -> None:
        """Update progress counters (and optionally ETA) and notify listeners.

        Parameters
        ----------
        current:
            Number of steps completed so far.
        total:
            Total number of steps.
        eta_seconds:
            Estimated seconds remaining.  Pass 0 (the default) when unknown.
        """
        self._progress_current = current
        self._progress_total = total
        self._eta_seconds = eta_seconds
        self._notify_state()

    def _set_status(
        self,
        activity: str,
        current: int,
        total: int,
        eta_seconds: int = 0,
    ) -> None:
        """Update activity and progress atomically in a single notification.

        Prefer this over calling  `_set_activity` and
         `_set_progress` separately to avoid the UI briefly showing a
        mismatched activity/progress pair between the two calls.
        """
        self._activity = activity
        self._progress_current = current
        self._progress_total = total
        self._eta_seconds = eta_seconds
        self._notify_state()

    def _notify_state(self) -> None:
        cb = self.on_state_changed
        if cb is not None:
            try:
                cb(
                    self.job_name,
                    self._activity,
                    self._progress_current,
                    self._progress_total,
                    self._eta_seconds,
                )
            except Exception as exc:
                warning(f"[{type(self).__name__}] on_state_changed raised: {exc}")

    # ------------------------------------------------------------------
    # Read-only state accessors
    # ------------------------------------------------------------------

    @property
    def activity(self) -> str:
        return self._activity

    @property
    def progress_current(self) -> int:
        return self._progress_current

    @property
    def progress_total(self) -> int:
        return self._progress_total

    @property
    def eta_seconds(self) -> int:
        return self._eta_seconds

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
        self._result = None
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

                # Re-check pause immediately after the step completes.
                # Without this, a pause issued during a step is not honoured
                # until after the *next* step has already run.
                self._pause_event.wait()
                if self._stop_event.is_set():
                    info(f"[{type(self).__name__}] Stopped while paused")
                    break

        except Exception as exc:
            error(f"[{type(self).__name__}] Unhandled exception: {exc}")
            import traceback
            error(traceback.format_exc())
        finally:
            self._running = False
            self._finished.set()
            # Clear activity/progress/ETA on exit so the UI resets cleanly.
            self._activity = "-"
            self._progress_current = 0
            self._progress_total = 0
            self._eta_seconds = 0
            self._notify_state()

            # Ensure _result is always populated so callers never see None
            # after the routine has finished.
            if self._result is None:
                self._result = RoutineResult(success=False)

            cb = self.on_complete
            if cb is not None:
                try:
                    cb(self._result)
                except Exception as exc:
                    warning(f"[{type(self).__name__}] on_complete raised: {exc}")

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    def _check_stop(self) -> bool:
        """Return True if a stop has been requested.

        Useful for long blocking operations inside a step where the routine
        cannot simply yield (e.g. a loop inside a single step).
        """
        return self._stop_event.is_set()