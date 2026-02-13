from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class State:
    machine_state: str = "Disconnected"
    automation_state: str = "Idle"

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
    
    def status_type(self) -> str:
        a = self.automation_state.strip().lower()
        if a in ("finished", "done", "complete", "completed"):
            return "done"
        if a in ("running", "busy", "capturing", "moving", "scanning", "paused"):
            return "active"
        return "idle"