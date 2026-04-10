"""
Square move automation routine.

Moves the stage in a square pattern centred on the starting position,
useful for verifying that X/Y motion is calibrated and that the
automation framework is functioning correctly.

The routine traces the four sides of the square in order::

    Start → (+X) → (+Y) → (-X) → (-Y / back to start)

A confirmation callback (``on_confirm``) is invoked before motion begins
so that callers can display the planned travel to the operator and
require explicit approval.  If the callback returns ``False`` the routine
aborts cleanly.

Usage::

    from common.app_context import get_app_context
    from motion.routines.square_move import SquareMove

    ctx = get_app_context()

    def confirm(side_mm: float) -> bool:
        answer = input(f"Move in a {side_mm} mm square? [y/N] ")
        return answer.strip().lower() == "y"

    routine = SquareMove(
        motion=ctx.motion,
        on_confirm=confirm,
        repeats=5,
    )
    routine.start()
    routine.wait()
"""

from __future__ import annotations

import time
from typing import Callable, Generator

from common.logger import info
from motion.motion_controller_manager import MotionControllerManager
from motion.models import Position

from motion.routines.automation_routine import AutomationRoutine

_NM_PER_MM: int = 1_000_000


class SquareMove(AutomationRoutine):
    """
    Move the stage in a square and return to the starting position.

    The four corners are visited in order, each separated by *side_mm*
    along a single axis, so the stage traces::

        (x0, y0)  →  (x0 + side, y0)  →  (x0 + side, y0 + side)
                  →  (x0, y0 + side)  →  (x0, y0)

    Parameters
    ----------
    motion:
        Active :class:`MotionControllerManager`.
    on_confirm:
        Callable that receives the planned side length in millimetres and
        returns ``True`` to proceed or ``False`` to abort.  Defaults to a
        no-op that always confirms.  Called from the background thread, so
        it must be thread-safe (e.g. a simple blocking ``input()`` prompt
        or a threading ``Event`` driven by a UI button).
    side_mm:
        Length of each side of the square in millimetres.  Must be > 0.
        Defaults to ``10.0``.
    repeats:
        Number of times to trace the full square pattern.  Must be >= 1.
        Defaults to ``5``.
    """
    job_name = "Square Move"

    def __init__(
        self,
        motion: MotionControllerManager,
        on_confirm: Callable[[float], bool] | None = None,
        side_mm: float = 10.0,
        repeats: int = 5,
    ) -> None:
        super().__init__(motion)

        if side_mm <= 0:
            raise ValueError(f"side_mm must be positive, got {side_mm!r}")
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats!r}")

        self._side_nm: int = round(side_mm * _NM_PER_MM)
        self._side_mm: float = side_mm
        self._repeats: int = repeats
        self._on_confirm: Callable[[float], bool] = on_confirm or (lambda _: True)

    # ------------------------------------------------------------------
    # AutomationRoutine implementation
    # ------------------------------------------------------------------

    def steps(self) -> Generator[None, None, None]:
        # ------------------------------------------------------------------
        # Confirmation
        # ------------------------------------------------------------------
        self._set_activity("Waiting for confirmation")
        info(f"[SquareMove] Requesting confirmation for {self._side_mm} mm square")
        confirmed = self._on_confirm(self._side_mm)
        if not confirmed:
            info("[SquareMove] Operator did not confirm — aborting")
            return

        if self._check_stop():
            return

        yield  # pause/stop point: after confirmation, before any motion

        # ------------------------------------------------------------------
        # Record origin
        # ------------------------------------------------------------------
        origin = self.motion.get_position()
        origin_mm_x = origin.x / _NM_PER_MM
        origin_mm_y = origin.y / _NM_PER_MM

        info(
            f"[SquareMove] Starting square: side={self._side_mm} mm, "
            f"repeats={self._repeats}, "
            f"origin=({origin_mm_x:.6f}, {origin_mm_y:.6f}) mm"
        )

        # Corners in order; Z and any other axes are kept at their current values.
        corners: list[tuple[str, Position]] = [
            (
                "Corner 1  (+X)",
                Position(x=origin.x + self._side_nm, y=origin.y,                z=origin.z),
            ),
            (
                "Corner 2  (+X, +Y)",
                Position(x=origin.x + self._side_nm, y=origin.y + self._side_nm, z=origin.z),
            ),
            (
                "Corner 3  (+Y)",
                Position(x=origin.x,                 y=origin.y + self._side_nm, z=origin.z),
            ),
            (
                "Origin",
                Position(x=origin.x,                 y=origin.y,                z=origin.z),
            ),
        ]

        # Total legs across all repeats, used for progress reporting.
        total_legs = self._repeats * len(corners)

        # ------------------------------------------------------------------
        # Traverse each corner, repeated self._repeats times
        # ------------------------------------------------------------------
        start_time = time.monotonic()
        legs_done = 0

        for repeat_index in range(self._repeats):
            if self._check_stop():
                break

            info(f"[SquareMove] Repeat {repeat_index + 1}/{self._repeats}")

            for leg_index, (label, target) in enumerate(corners):
                if self._check_stop():
                    break

                yield  # pause/stop point: before each move

                if self._check_stop():
                    break

                target_mm_x = target.x / _NM_PER_MM
                target_mm_y = target.y / _NM_PER_MM

                self._set_status(
                    f"Rep {repeat_index + 1}/{self._repeats}  —  {label}"
                , legs_done, total_legs)

                info(
                    f"[SquareMove] Leg {leg_index + 1}/4: moving to {label} "
                    f"({target_mm_x:.6f}, {target_mm_y:.6f}) mm"
                )

                self.motion.move_to_position(target, wait=True)

                legs_done += 1
                self._set_progress(legs_done, total_legs)

        total_elapsed = time.monotonic() - start_time
        info(f"[SquareMove] Routine complete in {total_elapsed:.3f} s")