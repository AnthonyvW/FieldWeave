"""
Tree core imaging automation routine.

Iterates over a caller-supplied list of slots, centres on each slot's
registration mark, moves to the slot's starting position, and waits
before proceeding to the next slot.

Procedure (per slot)
--------------------
1.  Move to the slot's mark position (perpendicular axis = slot.position_nm,
    main axis = tca.mark_reference_nm, Z = tca.mark_z_nm).
2.  Run the red mark centering routine.
3.  Move to the slot's starting position
    (main axis = tca.starting_offset_nm, Z = tca.starting_height_nm),
    mirroring _on_goto_start_pos_clicked in slot_calibration.py.
4.  Wait 2 seconds.

Usage::

    from common.app_context import get_app_context
    from motion.routines.tree_core_imaging_routine import TreeCoreImagingRoutine

    ctx = get_app_context()
    routine = TreeCoreImagingRoutine(
        motion=ctx.motion,
        output_folder="/path/to/output",
        slots=[(0, "core_A"), (2, "core_B")],
    )
    routine.start()
    routine.wait()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator

from common.app_context import get_app_context
from common.logger import error, info, warning
from motion.motion_controller_manager import MotionControllerManager
from motion.models import Position
from motion.routines.automation_routine import AutomationRoutine
from motion.routines.red_mark_centering_routine import RedMarkCenteringRoutine

_NM_PER_MM = 1_000_000
_SETTLE_S = 0.2
_DWELL_S = 2.0


def _run_centering(motion: MotionControllerManager, capture_timeout_s: float) -> None:
    centering = RedMarkCenteringRoutine(motion=motion, capture_timeout_s=capture_timeout_s)
    centering.start()
    centering.wait()


class TreeCoreImagingRoutine(AutomationRoutine):
    """
    Automated tree core imaging pass over a selected set of slots.

    Parameters
    ----------
    motion:
        Active :class:`MotionControllerManager`.
    output_folder:
        Root directory under which per-slot sub-folders will be created.
    slots:
        Ordered sequence of ``(slot_index, name)`` pairs.  ``slot_index``
        is zero-based and must be within range for the current calibration.
        ``name`` becomes the sub-folder name inside *output_folder*.
    capture_timeout_s:
        Seconds to wait for still-frame captures before giving up.
    """

    job_name = "Tree Core Imaging"

    def __init__(
        self,
        motion: MotionControllerManager,
        *,
        output_folder: str | Path,
        slots: list[tuple[int, str]],
        capture_timeout_s: float = 10.0,
    ) -> None:
        super().__init__(motion)
        self._output_folder = Path(output_folder)
        self._slots = list(slots)
        self._capture_timeout_s = capture_timeout_s

    def steps(self) -> Generator[None, None, None]:
        ctx = get_app_context()
        tca = ctx.motion.settings.tree_core_automation

        if not tca.has_been_calibrated:
            error("[TreeCoreImaging] Slot calibration has not been completed — aborting")
            return

        if not self._slots:
            warning("[TreeCoreImaging] No slots supplied — nothing to do")
            return

        total_steps = len(self._slots) * 3
        step = 0

        def _advance(activity: str) -> None:
            nonlocal step
            step += 1
            self._set_status(activity, step, total_steps)

        axis = tca.axis.lower()
        if axis not in ("x", "y"):
            error(f"[TreeCoreImaging] Unsupported axis '{axis}' — aborting")
            return

        for slot_index, slot_name in self._slots:
            if self._check_stop():
                return

            if slot_index >= tca.num_slots:
                warning(
                    f"[TreeCoreImaging] Slot index {slot_index} out of range"
                    f" ({tca.num_slots} slots) — skipping '{slot_name}'"
                )
                step += 3
                continue

            slot = tca.slots[slot_index]
            slot_label = f"slot {slot_index} ({slot_name!r})"

            # ------------------------------------------------------------------
            # Create output sub-folder
            # ------------------------------------------------------------------

            slot_folder = self._output_folder / slot_name
            slot_folder.mkdir(parents=True, exist_ok=True)
            info(f"[TreeCoreImaging] Output folder: {slot_folder}")

            # ------------------------------------------------------------------
            # Move to the slot's mark position and centre
            # ------------------------------------------------------------------

            _advance(f"Moving to mark — {slot_label}")
            info(f"[TreeCoreImaging] Navigating to mark for {slot_label}")

            perp_nm = slot.position_nm + slot.offset_nm

            if axis == "y":
                mark_target = Position(
                    x=perp_nm,
                    y=tca.mark_reference_nm,
                    z=tca.mark_z_nm,
                )
            else:
                mark_target = Position(
                    x=tca.mark_reference_nm,
                    y=perp_nm,
                    z=tca.mark_z_nm,
                )

            self.motion.move_to_position(mark_target, wait=True)
            time.sleep(_SETTLE_S)

            yield
            if self._check_stop():
                return

            _advance(f"Centering on mark — {slot_label}")
            info(f"[TreeCoreImaging] Running red mark centering for {slot_label}")
            _run_centering(self.motion, self._capture_timeout_s)

            yield
            if self._check_stop():
                return

            # ------------------------------------------------------------------
            # Move to starting position (mirrors _on_goto_start_pos_clicked)
            # ------------------------------------------------------------------

            _advance(f"Moving to start position — {slot_label}")
            info(f"[TreeCoreImaging] Moving to starting position for {slot_label}")

            centered = self.motion.get_position()

            if axis == "y":
                start_target = Position(
                    x=centered.x,
                    y=tca.starting_offset_nm,
                    z=tca.starting_height_nm,
                )
            else:
                start_target = Position(
                    x=tca.starting_offset_nm,
                    y=centered.y,
                    z=tca.starting_height_nm,
                )

            self.motion.move_to_position(start_target, wait=True)
            time.sleep(_SETTLE_S)

            info(f"[TreeCoreImaging] Dwelling {_DWELL_S}s at starting position for {slot_label}")
            time.sleep(_DWELL_S)

            yield

        self._set_status("Complete", total_steps, total_steps)
        info("[TreeCoreImaging] Imaging run complete")

        yield