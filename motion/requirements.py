"""
Motion requirements for automations and calibrations.

Each routine declares which axes it moves, which of those must be homed
before it starts, and which it homes itself (so homing must be enabled).
:meth:`MotionRequirements.problems` compares that against the current axis
configuration so the UI can explain why something is unavailable.
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
    homes: tuple[str, ...] = ()

    def problems(self, motion: MotionControllerManager | None) -> list[str]:
        """Human-readable reasons the requirements are not met; empty when they are."""
        settings = motion.settings if motion is not None else None
        if settings is None:
            return []

        enabled = settings.enabled_axes
        needed = [a for a in AXES if a in self.axes or a in self.homed or a in self.homes]
        problems = [f"{a.upper()} axis is disabled" for a in needed if a not in enabled]

        # Homed state is only meaningful once the controller has finished connecting.
        check_homed = motion.is_ready()
        for a in AXES:
            if a not in enabled or (a not in self.homed and a not in self.homes):
                continue
            if a not in settings.homing_axes:
                problems.append(f"{a.upper()} axis homing is disabled")
            elif a in self.homed and check_homed and a not in motion.homed_axes:
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
X_ONLY = MotionRequirements(axes=("x",))
XYZ_HOMING = MotionRequirements(axes=AXES, homes=AXES)
XYZ_HOMED = MotionRequirements(axes=AXES, homed=AXES)
