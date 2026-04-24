"""
Tree core imaging automation routine.

Iterates over a caller-supplied list of slots, centres on each slot's
registration mark, moves to the slot's starting position, then performs
two imaging sweeps:

Procedure (per slot)
--------------------
1.  Move to the slot's mark position (perpendicular axis = slot.position_nm,
    main axis = tca.mark_reference_nm, Z = tca.mark_z_nm).
2.  Run the red mark centering routine.
3.  Move to the slot's starting position
    (main axis = tca.starting_offset_nm, Z = tca.starting_height_nm),
    mirroring _on_goto_start_pos_clicked in slot_calibration.py.
4.  Wait 1 second.
5.  Forward sweep: step towards tca.mark_reference_nm in overlap-derived
    increments, stopping before any step that would overshoot the mark.
6.  Return to the slot's starting position.
7.  Reverse sweep: step away from tca.mark_reference_nm in the same
    increments until background detection reports bare background.

The step size is derived from the live sensor dimensions (one frame captured
per slot after arriving at the start position) combined with the camera
calibration, so that consecutive frames share ``image_overlap`` fractional
overlap (default 0.4).  For a Y-axis stage the FOV height drives the step;
for X the FOV width does.

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
from motion.routines.autofocus.autofocus_utils import capture_still_frame
from machine_vision.algorithms.background_detection import is_background_frame

_NM_PER_MM = 1_000_000
_NM_PER_TICK = 10_000
_SETTLE_S = 0.2
_DWELL_S = 1.0


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
    image_overlap:
        Fractional overlap between consecutive frames in the sweep direction,
        in the range [0, 1).  Defaults to 0.4 (40 % overlap).
    """

    job_name = "Tree Core Imaging"

    def __init__(
        self,
        motion: MotionControllerManager,
        *,
        output_folder: str | Path,
        slots: list[tuple[int, str]],
        capture_timeout_s: float = 10.0,
        image_overlap: float = 0.4,
    ) -> None:
        super().__init__(motion)
        self._output_folder = Path(output_folder)
        self._slots = list(slots)
        self._capture_timeout_s = capture_timeout_s
        self._image_overlap = image_overlap

    def steps(self) -> Generator[None, None, None]:
        ctx = get_app_context()
        tca = ctx.motion.settings.tree_core_automation
        mv = ctx.machine_vision

        if not tca.has_been_calibrated:
            error("[TreeCoreImaging] Slot calibration has not been completed — aborting")
            return

        if not self._slots:
            warning("[TreeCoreImaging] No slots supplied — nothing to do")
            return

        if not mv.is_calibrated:
            error("[TreeCoreImaging] No camera calibration — cannot derive step size, aborting")
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

        motion_settings = ctx.motion.settings
        if axis == "y":
            axis_max_nm = motion_settings.max_y * _NM_PER_MM
        else:
            axis_max_nm = motion_settings.max_x * _NM_PER_MM

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

            # Capture one frame to get live sensor dimensions for FOV derivation.
            # The sensor size won't change during the run so this is done once per slot.
            size_frame = capture_still_frame(ctx.camera_manager, timeout_s=self._capture_timeout_s)
            if size_frame is None:
                error(f"[TreeCoreImaging] Frame capture for step-size derivation failed — skipping {slot_label}")
                step += 3
                continue

            sensor_h, sensor_w = size_frame.shape[:2]
            cal = mv.calibration
            cal_w = float(cal.image_width)
            cal_h = float(cal.image_height)

            # Map the full sensor dimension into calibration pixel space using the
            # same scaling as _to_cal in the centering routine:
            #   cal_x = sensor_x * (cal_w / sensor_w)
            #   cal_y = sensor_y * (cal_h / sensor_h)
            # Substituting the full sensor dimension cancels, giving cal_w / cal_h
            # directly.  pixel_to_world_delta then gives the stage travel in ticks
            # required to shift the full frame width/height to centre — i.e. the FOV.
            if axis == "y":
                cal_offset_x = 0.0
                cal_offset_y = cal_h
            else:
                cal_offset_x = cal_w
                cal_offset_y = 0.0

            fov_delta = cal.pixel_to_world_delta(cal_offset_x, cal_offset_y)
            fov_nm = abs(fov_delta[1] if axis == "y" else fov_delta[0]) * _NM_PER_TICK

            if fov_nm <= 0:
                error(f"[TreeCoreImaging] Derived FOV is zero — skipping {slot_label}")
                step += 3
                continue

            step_nm = int(round(fov_nm * (1.0 - self._image_overlap)))
            info(
                f"[TreeCoreImaging] sensor={sensor_w}x{sensor_h}"
                f"  FOV={fov_nm:.0f} nm"
                f"  overlap={self._image_overlap:.0%}"
                f"  step={step_nm} nm"
            )

            # Record the starting position and fixed perpendicular coordinate for
            # both sweeps.
            start_pos = self.motion.get_position()
            perp_pos = start_pos.x if axis == "y" else start_pos.y
            main_start_nm = start_pos.y if axis == "y" else start_pos.x

            def _in_bounds(main_nm: int) -> bool:
                return 0 <= main_nm <= axis_max_nm

            # ------------------------------------------------------------------
            # Forward sweep: starting position → mark_reference_nm
            # ------------------------------------------------------------------

            info(f"[TreeCoreImaging] Beginning forward sweep for {slot_label}")

            direction = 1 if tca.mark_reference_nm > main_start_nm else -1
            current_main_nm = main_start_nm

            while True:
                if self._check_stop():
                    return

                next_main_nm = current_main_nm + direction * step_nm

                # Don't overshoot the mark.
                if direction > 0 and next_main_nm > tca.mark_reference_nm:
                    break
                if direction < 0 and next_main_nm < tca.mark_reference_nm:
                    break

                if not _in_bounds(next_main_nm):
                    error(
                        f"[TreeCoreImaging] Forward sweep would move to {next_main_nm} nm"
                        f" which is outside [0, {axis_max_nm}] — aborting"
                    )
                    return

                if axis == "y":
                    target = Position(x=perp_pos, y=next_main_nm, z=tca.starting_height_nm)
                else:
                    target = Position(x=next_main_nm, y=perp_pos, z=tca.starting_height_nm)

                self.motion.move_to_position(target, wait=True)
                current_main_nm = next_main_nm

                info(f"[TreeCoreImaging] Forward sweep position: {current_main_nm} nm")

                yield
                if self._check_stop():
                    return

            info(f"[TreeCoreImaging] Forward sweep complete for {slot_label}")

            # ------------------------------------------------------------------
            # Return to starting position
            # ------------------------------------------------------------------

            info(f"[TreeCoreImaging] Returning to start position for {slot_label}")
            self.motion.move_to_position(start_pos, wait=True)
            time.sleep(_SETTLE_S)

            yield
            if self._check_stop():
                return

            # ------------------------------------------------------------------
            # Reverse sweep: starting position → background detected
            # ------------------------------------------------------------------

            info(f"[TreeCoreImaging] Beginning reverse sweep for {slot_label}")

            reverse_direction = -direction
            current_main_nm = main_start_nm

            while True:
                if self._check_stop():
                    return

                next_main_nm = current_main_nm + reverse_direction * step_nm

                if not _in_bounds(next_main_nm):
                    warning(
                        f"[TreeCoreImaging] Reverse sweep reached stage limit at {current_main_nm} nm"
                        f" before background was detected — moving on to next slot"
                    )
                    break

                if axis == "y":
                    target = Position(x=perp_pos, y=next_main_nm, z=tca.starting_height_nm)
                else:
                    target = Position(x=next_main_nm, y=perp_pos, z=tca.starting_height_nm)

                self.motion.move_to_position(target, wait=True)
                current_main_nm = next_main_nm

                info(f"[TreeCoreImaging] Reverse sweep position: {current_main_nm} nm")

                frame = capture_still_frame(ctx.camera_manager, timeout_s=self._capture_timeout_s)
                if frame is None:
                    warning("[TreeCoreImaging] Frame capture failed during reverse sweep — stopping sweep")
                    break

                h, w = frame.shape[:2]
                bg_settings = mv.settings.background
                is_bg, val_median, val_std = is_background_frame(
                    frame,
                    val_median_max=bg_settings.val_median_max,
                    val_std_max=bg_settings.val_std_max,
                    scale=bg_settings.scale,
                )

                info(
                    f"[TreeCoreImaging] Background check:"
                    f" is_bg={is_bg}"
                    f" val_median={val_median:.1f}"
                    f" val_std={val_std:.1f}"
                )

                if is_bg:
                    info(f"[TreeCoreImaging] Background detected — reverse sweep complete for {slot_label}")
                    break

                yield

            yield

        self._set_status("Complete", total_steps, total_steps)
        info("[TreeCoreImaging] Imaging run complete")

        yield