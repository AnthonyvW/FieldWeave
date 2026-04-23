"""
Tree core slot calibration routine.

Calibrates the slot positions for automated tree core imaging by locating
the registration marks on the slide and computing slot separations.

Procedure
---------
1.  Run the red mark centering routine to align to the first mark.
2.  Save slot 1's position (perpendicular axis) along with ``mark_reference_nm``
    and ``mark_z_nm`` from the current position.
3.  Determine which side of the bed the slide is on and set
    ``starting_offset_nm`` accordingly (±30 mm from current position along
    the main axis).
4.  Set ``starting_height_nm`` to current Z + 12 mm.
5.  Move half a frame width perpendicular to the main axis (the same pixel→world
    conversion used by the red mark centering routine).
6.  Check for a red mark.  If none found, repeat step 5.
7.  When a mark is found, run the red mark centering routine and record slot 2.
8.  Compute ``slot_separation_nm`` as the absolute distance between slot 1 and
    slot 2 along the main axis.
9.  Step by ``slot_separation_nm`` along the main axis, run red mark centering,
    and fill each remaining slot until all slots are filled or the axis limit
    is exceeded.
10. Stamp ``last_calibrated_iso`` and save settings.

Usage::

    from common.app_context import get_app_context
    from motion.routines.slot_calibration_routine import SlotCalibrationRoutine

    ctx = get_app_context()
    routine = SlotCalibrationRoutine(motion=ctx.motion)
    routine.start()
    routine.wait()
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Generator

from common.app_context import get_app_context
from common.logger import info, warning, error
from motion.motion_controller_manager import MotionControllerManager
from motion.models import Position
from motion.routines.automation_routine import AutomationRoutine
from motion.routines.autofocus.autofocus_utils import capture_still_frame
from motion.routines.red_mark_centering_routine import RedMarkCenteringRoutine

_NM_PER_MM = 1_000_000
_SETTLE_S = 0.2
_HALF_FRAME_SEARCH_LIMIT = 50


def _run_centering(motion: MotionControllerManager, capture_timeout_s: float) -> bool:
    """Run the red mark centering routine synchronously. Returns True on success."""
    centering = RedMarkCenteringRoutine(motion=motion, capture_timeout_s=capture_timeout_s)
    centering.start()
    centering.wait()
    return True


class SlotCalibrationRoutine(AutomationRoutine):
    """
    Calibrate tree core slot positions by locating registration marks.

    Parameters
    ----------
    motion:
        Active :class:`MotionControllerManager`.
    capture_timeout_s:
        Seconds to wait for still-frame captures before giving up.
    """

    job_name = "Slot Calibration"

    def __init__(
        self,
        motion: MotionControllerManager,
        *,
        capture_timeout_s: float = 10.0,
    ) -> None:
        super().__init__(motion)
        self._capture_timeout_s = capture_timeout_s

    def steps(self) -> Generator[None, None, None]:  # noqa: C901
        ctx = get_app_context()
        camera_manager = ctx.camera_manager
        mv = ctx.machine_vision
        motion_settings = ctx.motion.settings
        tca = motion_settings.tree_core_automation
        ms = motion_settings

        total_steps = 6 + tca.num_slots
        step = 0

        def _advance(activity: str) -> None:
            nonlocal step
            step += 1
            self._set_status(activity, step, total_steps)

        _advance("Initialising")

        if not ctx.has_camera:
            error("[SlotCalibration] No camera available — aborting")
            return

        if motion_settings is None:
            error("[SlotCalibration] Motion settings not available — aborting")
            return

        if not mv.is_calibrated:
            warning("[SlotCalibration] No camera calibration — cannot convert pixels to stage units, aborting")
            return

        main_axis = tca.axis.lower()
        if main_axis not in ("x", "y"):
            error(f"[SlotCalibration] Unsupported main axis '{main_axis}' — aborting")
            return

        perp_axis = "x" if main_axis == "y" else "y"

        # ------------------------------------------------------------------
        # Step 1: Centre on the first red mark
        # ------------------------------------------------------------------

        _advance("Centering on first mark")
        info("[SlotCalibration] Running red mark centering for slot 1")

        _run_centering(self.motion, self._capture_timeout_s)

        yield
        if self._check_stop():
            return

        # ------------------------------------------------------------------
        # Step 2: Record slot 1 position, mark reference, and mark Z
        # ------------------------------------------------------------------

        _advance("Recording slot 1")

        pos1 = self.motion.get_position()

        slot1_position_nm = pos1.x if main_axis == "y" else pos1.y
        tca.slots[0].position_nm = slot1_position_nm
        tca.slots[0].offset_nm = 0

        tca.mark_reference_nm = pos1.y if main_axis == "y" else pos1.x
        tca.mark_z_nm = pos1.z

        info(
            f"[SlotCalibration] Slot 1 position ({perp_axis})={slot1_position_nm} nm"
            f"  mark_reference ({main_axis})={tca.mark_reference_nm} nm"
            f"  mark_z={tca.mark_z_nm} nm"
        )

        # ------------------------------------------------------------------
        # Step 3: Determine starting_offset_nm based on bed side
        # ------------------------------------------------------------------

        _advance("Computing starting offset")

        if main_axis == "y":
            axis_max_nm = ms.max_y * _NM_PER_MM
            current_main_nm = pos1.y
        else:
            axis_max_nm = ms.max_x * _NM_PER_MM
            current_main_nm = pos1.x

        half_axis_nm = axis_max_nm // 2
        offset_30mm = 30 * _NM_PER_MM

        if current_main_nm > half_axis_nm:
            tca.starting_offset_nm = current_main_nm - offset_30mm
        else:
            tca.starting_offset_nm = current_main_nm + offset_30mm

        info(
            f"[SlotCalibration] Bed side: {'far' if current_main_nm > half_axis_nm else 'near'}"
            f"  starting_offset_nm={tca.starting_offset_nm}"
        )

        # ------------------------------------------------------------------
        # Step 4: Set starting_height_nm to current Z + 12 mm
        # ------------------------------------------------------------------

        tca.starting_height_nm = pos1.z + 12 * _NM_PER_MM
        info(f"[SlotCalibration] starting_height_nm={tca.starting_height_nm}")

        yield
        if self._check_stop():
            return

        # ------------------------------------------------------------------
        # Step 5 / 6: Move perpendicular by half-frame increments until a
        # red mark is found, then centre on it and record slot 2.
        # ------------------------------------------------------------------

        _advance("Searching for slot 2 mark")

        cal = mv.calibration
        cal_w = float(cal.image_width)
        cal_h = float(cal.image_height)

        # Compute one full-frame stage distance along the perpendicular axis.
        # pixel_to_world_delta gives the stage delta (ticks) to bring the
        # supplied calibration-space pixel to image centre.  A point at the
        # edge of the frame is half the frame away from centre, so doubling
        # that delta gives the full-frame travel distance.  Moving a full
        # frame ensures the first slot is no longer visible before we begin
        # searching for the next mark.
        if perp_axis == "x":
            edge_delta = cal.pixel_to_world_delta(0.0, cal_h / 2.0)
            full_frame_nm = int(round(abs(edge_delta[0]) * 10_000 * 2))
        else:
            edge_delta = cal.pixel_to_world_delta(cal_w / 2.0, 0.0)
            full_frame_nm = int(round(abs(edge_delta[1]) * 10_000 * 2))

        if full_frame_nm == 0:
            warning("[SlotCalibration] Full-frame distance computed as zero — using 2 mm fallback")
            full_frame_nm = 2 * _NM_PER_MM

        info(f"[SlotCalibration] Full-frame step along {perp_axis}: {full_frame_nm} nm")

        # We always move in the positive perpendicular direction; the
        # registration marks are laid out along the slide so this is correct
        # for the expected slide orientation.
        mark_found = False
        for search_step in range(_HALF_FRAME_SEARCH_LIMIT):
            if self._check_stop():
                return

            cur = self.motion.get_position()
            if perp_axis == "x":
                target = Position(x=cur.x + full_frame_nm, y=cur.y, z=cur.z)
            else:
                target = Position(x=cur.x, y=cur.y + full_frame_nm, z=cur.z)

            info(
                f"[SlotCalibration] Search step {search_step + 1}:"
                f" moving {perp_axis} by {full_frame_nm} nm"
            )
            self.motion.move_to_position(target, wait=True)
            time.sleep(_SETTLE_S)

            frame = capture_still_frame(camera_manager, timeout_s=self._capture_timeout_s)
            if frame is None:
                warning("[SlotCalibration] Frame capture failed during search — skipping")
                yield
                continue

            h, w = frame.shape[:2]
            future = mv.request_red_mark_detection_guaranteed(frame, w, h)
            result = future.result()

            yield
            if self._check_stop():
                return

            if result.valid_centers:
                info(
                    f"[SlotCalibration] Red mark detected after {search_step + 1} half-frame steps"
                )
                mark_found = True
                break

        if not mark_found:
            error(
                f"[SlotCalibration] No red mark found after {_HALF_FRAME_SEARCH_LIMIT}"
                f" half-frame steps — aborting"
            )
            return

        # ------------------------------------------------------------------
        # Centre on slot 2 mark and record its position
        # ------------------------------------------------------------------

        _advance("Centering on slot 2 mark")
        info("[SlotCalibration] Running red mark centering for slot 2")

        _run_centering(self.motion, self._capture_timeout_s)

        yield
        if self._check_stop():
            return

        pos2 = self.motion.get_position()
        slot2_perp_nm = pos2.x if main_axis == "y" else pos2.y

        tca.slots[1].position_nm = slot2_perp_nm
        tca.slots[1].offset_nm = 0

        info(f"[SlotCalibration] Slot 2 perp-axis position ({perp_axis})={slot2_perp_nm} nm")

        # ------------------------------------------------------------------
        # Step 8/9: Compute slot separation
        # ------------------------------------------------------------------

        separation_nm = abs(slot2_perp_nm - slot1_position_nm)

        if separation_nm == 0:
            error("[SlotCalibration] Slot separation is zero — aborting")
            return

        tca.slot_separation_nm = separation_nm
        info(f"[SlotCalibration] slot_separation_nm={separation_nm}")

        # Direction of travel along the perpendicular axis (positive or negative).
        main_direction = 1 if slot2_perp_nm > slot1_position_nm else -1

        # ------------------------------------------------------------------
        # Steps 10 / 11: Fill remaining slots
        # ------------------------------------------------------------------

        _advance("Filling remaining slots")

        for slot_idx in range(2, tca.num_slots):
            if self._check_stop():
                return

            self._set_status(
                f"Calibrating slot {slot_idx + 1} of {tca.num_slots}",
                step + slot_idx - 1,
                total_steps,
            )

            cur = self.motion.get_position()
            cur_perp_nm = cur.x if main_axis == "y" else cur.y

            next_perp_nm = cur_perp_nm + main_direction * separation_nm

            if main_axis == "y":
                if next_perp_nm < 0 or next_perp_nm > ms.max_x * _NM_PER_MM:
                    info(
                        f"[SlotCalibration] Next position {next_perp_nm} nm exceeds X axis"
                        f" limits — stopping at slot {slot_idx}"
                    )
                    break
                target = Position(x=int(next_perp_nm), y=cur.y, z=cur.z)
            else:
                if next_perp_nm < 0 or next_perp_nm > ms.max_y * _NM_PER_MM:
                    info(
                        f"[SlotCalibration] Next position {next_perp_nm} nm exceeds Y axis"
                        f" limits — stopping at slot {slot_idx}"
                    )
                    break
                target = Position(x=cur.x, y=int(next_perp_nm), z=cur.z)

            info(
                f"[SlotCalibration] Moving to slot {slot_idx + 1}:"
                f" {perp_axis}={int(next_perp_nm)} nm"
            )
            self.motion.move_to_position(target, wait=True)
            time.sleep(_SETTLE_S)

            yield
            if self._check_stop():
                return

            info(f"[SlotCalibration] Centering on slot {slot_idx + 1} mark")
            _run_centering(self.motion, self._capture_timeout_s)

            yield
            if self._check_stop():
                return

            final_pos = self.motion.get_position()
            tca.slots[slot_idx].position_nm = final_pos.x if main_axis == "y" else final_pos.y
            tca.slots[slot_idx].offset_nm = 0

            info(
                f"[SlotCalibration] Slot {slot_idx + 1} recorded:"
                f" perp_pos={tca.slots[slot_idx].position_nm} nm"
            )

        # ------------------------------------------------------------------
        # Step 12: Stamp calibration timestamp and persist settings
        # ------------------------------------------------------------------

        tca.last_calibrated_iso = datetime.now(timezone.utc).isoformat()
        info(f"[SlotCalibration] last_calibrated_iso={tca.last_calibrated_iso}")

        ctx.motion._controller._config_manager.save(motion_settings)
        info("[SlotCalibration] Settings saved")

        self._set_status("Complete", total_steps, total_steps)
        info("[SlotCalibration] Slot calibration complete")

        yield