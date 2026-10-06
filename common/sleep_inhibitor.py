"""
sleep_inhibitor.py

Cross-platform (Windows / Debian-based Linux) system-sleep prevention for the
duration of a long-running automation.

Windows: SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) keeps the
system awake without forcing the display to stay on.

Linux: spawns ``systemd-inhibit --what=sleep:idle ... sleep infinity`` and
holds it open as a subprocess; terminating that subprocess drops the
inhibitor lock held with systemd-logind. Requires systemd, which is standard
on Debian/Ubuntu and other systemd-based distributions.

Usage::

    from common.sleep_inhibitor import get_sleep_inhibitor

    inhibitor = get_sleep_inhibitor()
    inhibitor.acquire("Area scan running")
    ...
    inhibitor.release()

Reference-counted, so nested acquire()/release() pairs (e.g. two automations
that briefly overlap) behave correctly, and safe to call from any thread.
On an unsupported platform, or if the underlying OS call fails, acquire()
logs a warning and otherwise does nothing - it never raises.
"""

from __future__ import annotations

import subprocess
import sys
import threading

from common.logger import debug, warning

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


class SleepInhibitor:
    """Prevents the OS from sleeping while held.

    Not tied to any particular automation type - callers acquire before
    starting work that must not be interrupted by the system sleeping, and
    release once that work is done (usually in a ``finally`` block).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._linux_proc: subprocess.Popen | None = None

    def acquire(self, reason: str = "FieldWeave automation running") -> None:
        """Prevent the system from sleeping. Safe to call repeatedly; only
        the first call (while not already held) takes effect.
        """
        with self._lock:
            self._count += 1
            if self._count > 1:
                return
            self._engage(reason)

    def release(self) -> None:
        """Undo one acquire(). Sleep is allowed again once every acquire()
        has a matching release(). Calling with nothing held is a no-op.
        """
        with self._lock:
            if self._count == 0:
                return
            self._count -= 1
            if self._count == 0:
                self._disengage()

    # ------------------------------------------------------------------
    # Platform dispatch
    # ------------------------------------------------------------------

    def _engage(self, reason: str) -> None:
        if sys.platform == "win32":
            self._engage_windows()
        elif sys.platform.startswith("linux"):
            self._engage_linux(reason)
        else:
            debug(f"SleepInhibitor: unsupported platform {sys.platform!r} - sleep not prevented")

    def _disengage(self) -> None:
        if sys.platform == "win32":
            self._disengage_windows()
        elif sys.platform.startswith("linux"):
            self._disengage_linux()

    # ------------------------------------------------------------------
    # Windows
    # ------------------------------------------------------------------

    def _engage_windows(self) -> None:
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
                _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
            )
            debug("SleepInhibitor: system sleep prevented (Windows)")
        except Exception as exc:
            warning(f"SleepInhibitor: failed to prevent sleep on Windows: {exc}")

    def _disengage_windows(self) -> None:
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)  # type: ignore[attr-defined]
            debug("SleepInhibitor: system sleep allowed again (Windows)")
        except Exception as exc:
            warning(f"SleepInhibitor: failed to restore sleep state on Windows: {exc}")

    # ------------------------------------------------------------------
    # Linux (systemd)
    # ------------------------------------------------------------------

    def _engage_linux(self, reason: str) -> None:
        try:
            self._linux_proc = subprocess.Popen(
                [
                    "systemd-inhibit",
                    "--what=sleep:idle",
                    "--who=FieldWeave",
                    f"--why={reason}",
                    "--mode=block",
                    "sleep", "infinity",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            debug("SleepInhibitor: system sleep prevented (systemd-inhibit)")
        except FileNotFoundError:
            warning("SleepInhibitor: systemd-inhibit not found - sleep not prevented")
        except Exception as exc:
            warning(f"SleepInhibitor: failed to prevent sleep on Linux: {exc}")

    def _disengage_linux(self) -> None:
        proc = self._linux_proc
        self._linux_proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
            debug("SleepInhibitor: system sleep allowed again (systemd-inhibit)")
        except Exception as exc:
            warning(f"SleepInhibitor: failed to release systemd-inhibit: {exc}")


_inhibitor: SleepInhibitor | None = None


def get_sleep_inhibitor() -> SleepInhibitor:
    """Return the process-wide :class:`SleepInhibitor` instance."""
    global _inhibitor
    if _inhibitor is None:
        _inhibitor = SleepInhibitor()
    return _inhibitor
