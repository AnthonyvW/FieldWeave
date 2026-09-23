"""
Motion requirements for automations and calibrations.

Each routine declares which axes it moves and which of those must be homed
first.  :meth:`MotionRequirements.problems` compares that against the current
axis configuration so the UI can explain why something is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from motion.motion_config import AXES

if TYPE_CHECKING:
    from motion.motion_controller_manager import MotionControllerManager


@dataclass(frozen=True)
class MotionRequirements:
    axes: tuple[str, ...] = ()
    homed: tuple[str, ...] = ()

    def problems(self, motion: MotionControllerManager | None) -> list[str]:
        """Human-readable reasons the requirements are not met; empty when they are."""
        settings = motion.settings if motion is not None else None
        if settings is None:
            return []

        enabled = settings.enabled_axes
        needed = [a for a in AXES if a in self.axes or a in self.homed]
        problems = [f"{a.upper()} axis is disabled" for a in needed if a not in enabled]

        # Homed state is only meaningful once the controller has finished connecting.
        check_homed = motion.is_ready()
        for a in self.homed:
            if a not in enabled:
                continue
            if a not in settings.homing_axes:
                problems.append(f"{a.upper()} axis homing is disabled")
            elif check_homed and a not in motion.homed_axes:
                problems.append(f"{a.upper()} axis has not been homed")
        return problems

    def describe_problems(self, motion: MotionControllerManager | None) -> str:
        """One-line summary of :meth:`problems`, or an empty string when met."""
        problems = self.problems(motion)
        if not problems:
            return ""
        return "Unavailable: " + "; ".join(problems) + "."


NO_REQUIREMENTS = MotionRequirements()
XY = MotionRequirements(axes=("x", "y"))
Z_ONLY = MotionRequirements(axes=("z",))
XYZ = MotionRequirements(axes=AXES)
XYZ_HOMED = MotionRequirements(axes=AXES, homed=AXES)
