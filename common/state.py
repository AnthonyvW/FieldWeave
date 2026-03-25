from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MachineState(str, Enum):
    DISCONNECTED: str = "Disconnected"
    CONNECTED: str = "Connected"
    CONNECTING: str = "Connecting"

    def __str__(self) -> str:
        return self.value


class AutomationState(str, Enum):
    IDLE: str = "Idle"
    COMPLETE: str = "Completed"
    RUNNING: str = "Running"
    PAUSED: str = "Paused"

    def __str__(self) -> str:
        return self.value


def _format_eta(seconds: int) -> str:
    """
    Format *seconds* as a compact ETA string for the status bar.

    Rules
    -----
    - 0 (or negative)  → empty string  (caller should hide the field)
    - < 60 s           → "ETA {s}s"
    - < 600 s (10 min) → "ETA {m}m {s}s"
    - < 3600 s (1 h)   → "ETA {m}m"
    - >= 3600 s        → "ETA {h}h {m}m"
    """
    if seconds <= 0:
        return ""
    if seconds < 60:
        return f"ETA {seconds}s"
    minutes, secs = divmod(seconds, 60)
    if seconds < 600:
        return f"ETA {minutes}m {secs}s"
    if seconds < 3600:
        return f"ETA {minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"ETA {hours}h {minutes}m"


@dataclass(frozen=True)
class State:
    machine_state: str = MachineState.DISCONNECTED
    automation_state: str = AutomationState.IDLE
    camera_state: str = ""

    activity: str = "-"
    job_name: str = "-"

    progress_current: int = 0
    progress_total: int = 0

    # Estimated seconds remaining; 0 means "unknown / not applicable".
    eta_seconds: int = 0

    def format_status_text(self) -> str:
        parts: list[str] = [f"{self.machine_state} • {self.automation_state}"]
        if self.job_name != "-" and self.activity != "-":
            parts.append(f"{self.job_name}: {self.activity}")
        elif self.activity != "-":
            parts.append(self.activity)
        if self.progress_total > 0:
            progress = f"{self.progress_current}/{self.progress_total}"
            eta = self.format_eta()
            parts.append(f"{eta}" if eta else progress)
        return "  |  ".join(parts)

    def format_eta(self) -> str:
        """Return a formatted ETA string, or '' if eta_seconds is 0."""
        return _format_eta(self.eta_seconds)