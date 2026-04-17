"""
post_processing_routine.py

Abstract base class for all post-processing routines.

Mirrors :class:`AutomationRoutine` from the motion subsystem: each routine
runs on its own daemon thread, communicates pause / resume / stop signals
through threading primitives, and reports live status through a callback.

Subclasses implement :meth:`steps` as a generator, yielding after each
logical step.  This keeps all of a routine's logic in one place without
resorting to complex state machines.

Subclasses should set the class-level :attr:`job_name` and call
:meth:`_set_activity` / :meth:`_set_progress` during execution to push
human-readable status to the UI.

Example::

    class MyRoutine(PostProcessingRoutine):
        job_name = "My Routine"

        def steps(self):
            self._set_activity("Step one")
            do_something()
            yield
            self._set_activity("Step two")
            do_something_else()
            yield
"""

from __future__ import annotations

import threading
import traceback
from abc import ABC, abstractmethod
from typing import Callable, Generator

from common.logger import info, error, warning
from common.fieldweaveConfig import FieldWeaveSettings

# Signature: (job_name, activity, progress_current, progress_total, eta_seconds) -> None
RoutineStateCallback = Callable[[str, str, int, int, int], None]


class PostProcessingRoutine(ABC):
    """
    Abstract base class for all post-processing routines.

    Subclasses must implement :meth:`steps`, a generator that yields between
    logical steps.  The runner thread advances the generator, honouring pause
    and stop requests between each yield.

    Set the class-level :attr:`job_name` (or override in ``__init__``) to give
    the routine a human-readable display name.  During execution call
    :meth:`_set_activity` and :meth:`_set_progress` to push live status to any
    registered :attr:`on_state_changed` callback.

    Parameters
    ----------
    settings:
        The application-wide :class:`FieldWeaveSettings`.  Routines read their
        configuration from ``settings.post_processing`` rather than holding
        their own copies so the manager can apply updated settings between jobs.
    """

    #: Human-readable name shown in the status bar.  Override in subclasses.
    job_name: str = "-"

    def __init__(self, settings: FieldWeaveSettings) -> None:
        self.settings = settings

        self._pause_event = threading.Event()
        self._pause_event.set()          # Not paused initially (set = allowed to run)
        self._stop_event = threading.Event()

        self._thread: threading.Thread | None = None
        self._running = False
        self._finished = threading.Event()

        self._activity: str = "-"
        self._progress_current: int = 0
        self._progress_total: int = 0
        self._eta_seconds: int = 0

        # Wired by the manager after construction.
        self.on_state_changed: RoutineStateCallback | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def steps(self) -> Generator[None, None, None]:
        """
        Yield-based generator implementing the routine's logic.

        Yield at each point where the routine can be cleanly paused or stopped.
        """

    # ------------------------------------------------------------------
    # Status helpers for subclasses
    # ------------------------------------------------------------------

    def _set_activity(self, activity: str) -> None:
        """Update the current activity description and notify listeners."""
        self._activity = activity
        self._notify_state()

    def _set_progress(self, current: int, total: int, eta_seconds: int = 0) -> None:
        """Update progress counters and notify listeners.

        Parameters
        ----------
        current:
            Number of steps completed so far.
        total:
            Total number of steps.
        eta_seconds:
            Estimated seconds remaining.  Pass 0 when unknown.
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

        Prefer this over calling :meth:`_set_activity` and
        :meth:`_set_progress` separately to avoid the UI briefly seeing a
        mismatched pair.
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
        """Start the routine on a background daemon thread.

        Raises :class:`RuntimeError` if the routine is already running.
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
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=type(self).__name__
        )
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
        """Request the routine to stop after its current step.

        Also unblocks a paused routine so it can observe the stop event.
        """
        self._stop_event.set()
        self._pause_event.set()
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
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    def _check_stop(self) -> bool:
        """Return True if a stop has been requested.

        Useful inside long-running steps that cannot simply yield (e.g. a
        per-image loop inside a single step).
        """
        return self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Internal runner
    # ------------------------------------------------------------------

    def _run(self) -> None:
        info(f"[{type(self).__name__}] Starting")
        try:
            gen = self.steps()
            while True:
                if self._stop_event.is_set():
                    info(f"[{type(self).__name__}] Stopped")
                    break

                self._pause_event.wait()
                if self._stop_event.is_set():
                    info(f"[{type(self).__name__}] Stopped while paused")
                    break

                try:
                    next(gen)
                except StopIteration:
                    info(f"[{type(self).__name__}] Completed successfully")
                    break

                # Re-check pause immediately after the step so a pause issued
                # during a step is honoured before the next one starts.
                self._pause_event.wait()
                if self._stop_event.is_set():
                    info(f"[{type(self).__name__}] Stopped while paused")
                    break

        except Exception as exc:
            error(f"[{type(self).__name__}] Unhandled exception: {exc}")
            error(traceback.format_exc())
        finally:
            self._running = False
            self._finished.set()
            self._activity = "-"
            self._progress_current = 0
            self._progress_total = 0
            self._eta_seconds = 0
            self._notify_state()