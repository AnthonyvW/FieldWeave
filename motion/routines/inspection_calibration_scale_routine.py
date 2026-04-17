"""
Inspection calibration scale routine.

Autofocuses, then walks the stage along the calibration scale bar until it
has traversed the full length of the bar, saving a raw image at each position
into a ``raw_calibration_scale`` folder inside the supplied output path.

The routine:
  1. Runs an autofocus pass in place.
  2. Captures a snap and runs the inspect-calibration machine-vision pipeline
     to determine the bar's axis and which end of the bar is currently visible.
  3. Moves 0.4 mm toward the far (absent) end of the bar.
  4. Repeats until the last visible tick cluster has crossed the image
     centre along the bar axis.

Images are saved as JPEG with the stage position encoded in the filename:
  x{X_nm}_y{Y_nm}_z{Z_nm}.jpg

Usage::

    from common.app_context import get_app_context
    from motion.routines.inspection_calibration_scale_routine import (
        InspectionCalibrationScaleRoutine,
    )

    ctx = get_app_context()
    routine = InspectionCalibrationScaleRoutine(
        motion=ctx.motion,
        output_path="/data/calibration_run1",
    )
    routine.start()
    routine.wait()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator

import cv2

from common.app_context import get_app_context
from common.logger import info, warning, error
from motion.motion_controller_manager import MotionControllerManager
from motion.models import Position
from motion.routines.automation_routine import AutomationRoutine
from motion.routines.autofocus.autofocus_utils import capture_still_frame, move_z_and_wait
from motion.routines.autofocus.autofocus_routine import Autofocus
from machine_vision.calibration_bar_detection import AxisState, process_frame

_NM_PER_MM = 1_000_000
_STEP_NM = int(0.4 * _NM_PER_MM)


def _axis_move_direction(
    axis: str,
    clusters: list[tuple[int, int]],
    image_width: int,
    image_height: int,
    downsample: int,
) -> int:
    """
    Return +1 or -1 indicating which direction along *axis* to move.

    We count how many tick cluster centres fall above vs below the image
    midpoint (in pixel coordinates, so "above" means a larger pixel index —
    right for horizontal, lower in the frame for vertical).  More ticks on
    the high-pixel side means the dense/start end of the bar is there, so we
    move in the negative stage direction to travel toward the sparse far end.
    Fewer ticks on the high-pixel side means the dense end is on the left/top
    of the frame, so we move in the positive direction instead.

    Returns +1 (positive stage direction) or -1 (negative stage direction).
    """
    if not clusters:
        return 1

    if axis == "horizontal":
        centre = (image_width // 2) // downsample
        cluster_centres = [(s + e) // 2 for s, e in clusters]
        above = sum(1 for c in cluster_centres if c > centre)
        below = len(cluster_centres) - above
        # Dense end on the right side of frame → move stage negative (left)
        return -1 if above >= below else 1
    else:
        centre = (image_height // 2) // downsample
        cluster_centres = [(s + e) // 2 for s, e in clusters]
        above = sum(1 for c in cluster_centres if c > centre)
        below = len(cluster_centres) - above
        # Dense end on the lower side of frame → move stage negative (up)
        return -1 if above >= below else 1


class InspectionCalibrationScaleRoutine(AutomationRoutine):
    """
    Walk the stage along the calibration scale bar, saving images at each step.

    Starting from the current position the routine autofocuses, determines
    the bar axis and travel direction, then steps 0.4 mm at a time toward the
    far end of the bar.  It stops once the last visible tick cluster has
    crossed the midpoint of the image along the bar axis, meaning the final
    tick mark is at or just past centre in the captured frame.

    All images are saved into ``<output_path>/raw_calibration_scale/`` with
    filenames encoding the actual stage position at capture time.

    Parameters
    ----------
    motion:
        Active :class:`MotionControllerManager`.
    output_path:
        Root directory.  The ``raw_calibration_scale`` subfolder is created
        automatically.
    step_mm:
        Distance to move between captures, in mm.  Default is 0.4 mm.
    settle_s:
        Seconds to wait after each move before capturing.
    max_steps:
        Safety cap on the number of steps; the routine stops after this many
        moves even if the bar end has not been detected.
    capture_timeout_s:
        Seconds to wait for each still capture before giving up.
    """

    job_name = "Inspection Calibration Scale"

    def __init__(
        self,
        motion: MotionControllerManager,
        output_path: str | Path,
        *,
        step_mm: float = 0.4,
        settle_s: float = 0.4,
        max_steps: int = 60,
        capture_timeout_s: float = 10.0,
    ) -> None:
        super().__init__(motion)
        self._output_path = Path(output_path)
        self._step_nm = int(round(step_mm * _NM_PER_MM))
        self._settle_s = settle_s
        self._max_steps = max_steps
        self._capture_timeout_s = capture_timeout_s

    # ------------------------------------------------------------------
    # AutomationRoutine implementation
    # ------------------------------------------------------------------

    def steps(self) -> Generator[None, None, None]:  # noqa: C901
        ctx = get_app_context()
        camera_manager = ctx.camera_manager
        mv = ctx.machine_vision

        self._set_activity("Initialising")

        if not ctx.has_camera:
            error("[CalibrationScale] No camera available — aborting")
            return

        save_dir = self._output_path / "raw_calibration_scale"
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            error(f"[CalibrationScale] Could not create output directory: {exc}")
            return

        info(f"[CalibrationScale] Saving images to: {save_dir}")

        # ------------------------------------------------------------------
        # Step 1: Autofocus
        # ------------------------------------------------------------------

        self._set_activity("Autofocusing")
        info("[CalibrationScale] Starting autofocus")

        autofocus = Autofocus(motion=self.motion)
        autofocus.start()
        autofocus.wait()

        if self._check_stop():
            return

        yield

        # ------------------------------------------------------------------
        # Step 2: Initial snap to determine axis and travel direction
        # ------------------------------------------------------------------

        self._set_activity("Detecting scale bar axis")
        info("[CalibrationScale] Capturing initial snap for bar detection")

        if self._settle_s > 0:
            time.sleep(self._settle_s)

        initial_frame = capture_still_frame(camera_manager, timeout_s=self._capture_timeout_s)
        if initial_frame is None:
            error("[CalibrationScale] Initial frame capture failed — aborting")
            return

        mv.reset_inspect_calibration_state()
        axis_state = AxisState()

        bgr_initial = cv2.cvtColor(initial_frame, cv2.COLOR_RGB2BGR)
        initial_result = process_frame(bgr_initial, axis_state, downsample=2)

        info(
            f"[CalibrationScale] Initial detection:"
            f" axis={initial_result.axis}"
            f" ticks={initial_result.tick_count}"
            f" start_present={initial_result.start_present}"
            f" end_present={initial_result.end_present}"
        )

        if initial_result.tick_count == 0:
            error("[CalibrationScale] No ticks detected in initial frame — aborting")
            return

        image_h, image_w = initial_frame.shape[:2]
        direction = _axis_move_direction(
            initial_result.axis,
            initial_result._clusters,
            image_w,
            image_h,
            initial_result._downsample,
        )

        info(
            f"[CalibrationScale] Travel axis={initial_result.axis}"
            f"  direction={'positive' if direction > 0 else 'negative'}"
        )

        yield

        if self._check_stop():
            return

        # ------------------------------------------------------------------
        # Step 3: Save initial image
        # ------------------------------------------------------------------

        pos = self.motion.get_position()
        filename = f"x{pos.x}_y{pos.y}_z{pos.z}.jpg"
        filepath = save_dir / filename

        try:
            cv2.imwrite(str(filepath), bgr_initial)
            info(f"[CalibrationScale] Saved initial image: {filepath}")
        except OSError as exc:
            warning(f"[CalibrationScale] Failed to save initial image: {exc}")

        # ------------------------------------------------------------------
        # Step 4: Walk the bar
        # ------------------------------------------------------------------

        step = 0

        while step < self._max_steps:
            if self._check_stop():
                return

            step += 1
            self._set_status(
                f"Step {step}/{self._max_steps}  ticks={result.tick_count if step > 1 else initial_result.tick_count}",
                step,
                self._max_steps,
            )

            current_pos = self.motion.get_position()
            if initial_result.axis == "horizontal":
                target = Position(
                    x=current_pos.x + direction * self._step_nm,
                    y=current_pos.y,
                    z=current_pos.z,
                )
            else:
                target = Position(
                    x=current_pos.x,
                    y=current_pos.y + direction * self._step_nm,
                    z=current_pos.z,
                )

            info(
                f"[CalibrationScale] Step {step}: moving to"
                f" X={target.x / _NM_PER_MM:.4f} mm"
                f" Y={target.y / _NM_PER_MM:.4f} mm"
            )
            self.motion.move_to_position(target, wait=True)

            yield

            if self._check_stop():
                return

            if self._settle_s > 0:
                time.sleep(self._settle_s)

            frame = capture_still_frame(camera_manager, timeout_s=self._capture_timeout_s)
            if frame is None:
                warning(f"[CalibrationScale] Step {step}: frame capture failed — skipping")
                yield
                continue

            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            result = process_frame(bgr, axis_state, downsample=2)

            info(
                f"[CalibrationScale] Step {step}:"
                f" ticks={result.tick_count}"
                f" start={result.start_present}"
                f" end={result.end_present}"
                f" mode={result.mode}"
            )

            actual_pos = self.motion.get_position()
            fname = f"x{actual_pos.x}_y{actual_pos.y}_z{actual_pos.z}.jpg"
            fpath = save_dir / fname
            try:
                cv2.imwrite(str(fpath), bgr)
                info(f"[CalibrationScale] Saved: {fpath}")
            except OSError as exc:
                warning(f"[CalibrationScale] Failed to save image at step {step}: {exc}")

            # Stop when the last tick mark has passed through the centre of
            # the image.  As we move, ticks peel off the leading edge first.
            # The last tick remaining is always on the trailing side — the
            # high-pixel end when moving negative, the low-pixel end when
            # moving positive.  We stop once that tick's centre has crossed
            # the image midpoint.
            if result._clusters:
                image_mid = (
                    (image_w // 2) // result._downsample
                    if initial_result.axis == "horizontal"
                    else (image_h // 2) // result._downsample
                )
                cluster_centres = [(s + e) // 2 for s, e in result._clusters]
                if direction > 0:
                    # Moving toward higher pixels: ticks peel off the low end,
                    # last tick is the one with the smallest pixel coord.
                    last_tick_centre = min(cluster_centres)
                    past_centre = last_tick_centre >= image_mid
                else:
                    # Moving toward lower pixels: ticks peel off the high end,
                    # last tick is the one with the largest pixel coord.
                    last_tick_centre = max(cluster_centres)
                    past_centre = last_tick_centre <= image_mid

                if past_centre:
                    info(
                        f"[CalibrationScale] Last tick centre at pixel"
                        f" {last_tick_centre} (image mid {image_mid})"
                        f" — last tick passed centre, stopping"
                    )
                    break

            yield

        else:
            warning(
                f"[CalibrationScale] Reached max step limit ({self._max_steps})"
                f" without detecting bar end"
            )

        self._set_activity("Complete")
        info(
            f"[CalibrationScale] Done — {step} steps taken,"
            f" {len(list(save_dir.glob('*.jpg')))} images saved in {save_dir}"
        )