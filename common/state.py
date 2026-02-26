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



@dataclass(frozen=True)
class State:
    machine_state: str = MachineState.DISCONNECTED
    automation_state: str = AutomationState.IDLE
    camera_state: str = ""

    activity: str = "-"
    job_name: str = "-"
    
    progress_current: int = 0
    progress_total: int = 0

    def format_status_text(self) -> str:
        parts: list[str] = [f"{self.machine_state} • {self.automation_state}"]
        if self.job_name != "-" and self.activity != "-":
            parts.append(f"{self.job_name}: {self.activity}")
        elif self.activity != "-":
            parts.append(self.activity)
        if self.progress_total > 0:
            parts.append(f"{self.progress_current}/{self.progress_total}")
        return "  |  ".join(parts)