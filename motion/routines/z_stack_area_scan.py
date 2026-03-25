"""
Z-stack area scan automation routine.

Performs a Z-stack scan at each XY position in a grid, capturing images at
every combination of (X, Y, Z).  Images are saved into per-XY subfolders
inside the output directory.

The routine reports elapsed time after each Z-stack and provides a running
estimate of remaining time, incorporating a 1-second-per-remaining-stack
travel-time allowance.

Usage::

    from common.app_context import get_app_context
    from motion.automations.z_stack_area_scan import ZStackAreaScan

    ctx = get_app_context()
    routine = ZStackAreaScan(
        motion=ctx.motion,
        x_start_nm=0,
        x_end_nm=2_000_000,
        x_step_nm=1_000_000,
        y_start_nm=0,
        y_end_nm=2_000_000,
        y_step_nm=1_000_000,
        z_start_nm=0,
        z_end_nm=5_000_000,
        z_step_nm=500_000,
        output_folder="/data/scans/area_run1",
    )
    routine.start()
    routine.wait()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator

from common.app_context import get_app_context
from common.logger import info, warning, error
from motion.motion_controller_manager import MotionControllerManager
from motion.models import Position

from motion.routines.automation_routine import AutomationRoutine

_NM_PER_MM = 1_000_000


def _build_axis_positions(start_nm: int, end_nm: int, step_nm: int) -> list[int]:
    """Return evenly-spaced positions from *start_nm* to *end_nm* inclusive."""
    if start_nm == end_nm:
        return [start_nm]
    direction = 1 if end_nm > start_nm else -1
    positions: list[int] = []
    z = start_nm
    while (direction == 1 and z <= end_nm) or (direction == -1 and z >= end_nm):
        positions.append(z)
        z += direction * step_nm
    return positions


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as ``H:MM:SS`` or ``M:SS``."""
    seconds = max(0.0, seconds)
    total_s = int(seconds)
    h, remainder = divmod(total_s, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class ZStackAreaScan(AutomationRoutine):
    """
    Capture a Z-stack at every XY grid position.

    For each (X, Y) position the stage visits all Z positions between
    *z_start_nm* and *z_end_nm*, capturing one image per Z step and saving
    them into a subfolder named ``x{X_nm}_y{Y_nm}`` inside *output_folder*.

    After each completed Z-stack the routine logs:
    - how long that stack took,
    - how many stacks remain,
    - an estimated time to completion (mean stack duration so far plus
      1 second per remaining stack to account for XY travel).

    Parameters
    ----------
    motion:
        Active :class:`MotionControllerManager`.
    x_start_nm:
        Start of the X range, in nanometres.
    x_end_nm:
        End of the X range, in nanometres.
    x_step_nm:
        Step size along X, in nanometres.  Must be > 0.
    y_start_nm:
        Start of the Y range, in nanometres.
    y_end_nm:
        End of the Y range, in nanometres.
    y_step_nm:
        Step size along Y, in nanometres.  Must be > 0.
    z_start_nm:
        One end of the Z range, in nanometres.
    z_end_nm:
        The other end of the Z range, in nanometres.
    z_step_nm:
        Distance between Z capture positions, in nanometres.  Must be > 0.
    output_folder:
        Root directory for saved images.  Per-XY subfolders are created
        automatically.
    capture_timeout_ms:
        How long (ms) to wait for each image capture to complete.
    """

    job_name = "Z-Stack Area Scan"

    def __init__(
        self,
        motion: MotionControllerManager,
        x_start_nm: int,
        x_end_nm: int,
        x_step_nm: int,
        y_start_nm: int,
        y_end_nm: int,
        y_step_nm: int,
        z_start_nm: int,
        z_end_nm: int,
        z_step_nm: int,
        output_folder: str | Path,
        capture_timeout_ms: int = 5000,
    ) -> None:
        super().__init__(motion)

        for name, value in (
            ("x_step_nm", x_step_nm),
            ("y_step_nm", y_step_nm),
            ("z_step_nm", z_step_nm),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if z_start_nm == z_end_nm:
            raise ValueError("z_start_nm and z_end_nm must be different")

        self._x_start_nm = x_start_nm
        self._x_end_nm = x_end_nm
        self._x_step_nm = x_step_nm
        self._y_start_nm = y_start_nm
        self._y_end_nm = y_end_nm
        self._y_step_nm = y_step_nm
        self._z_start_nm = z_start_nm
        self._z_end_nm = z_end_nm
        self._z_step_nm = z_step_nm
        self._output_folder = Path(output_folder)
        self._capture_timeout_ms = capture_timeout_ms

    # ------------------------------------------------------------------
    # AutomationRoutine implementation
    # ------------------------------------------------------------------

    def steps(self) -> Generator[None, None, None]:  # noqa: C901
        ctx = get_app_context()
        camera = ctx.camera

        self._set_activity("Initialising")

        if camera is None:
            error("[ZStackAreaScan] No camera available — aborting")
            return

        self._output_folder.mkdir(parents=True, exist_ok=True)

        x_positions = _build_axis_positions(self._x_start_nm, self._x_end_nm, self._x_step_nm)
        y_positions = _build_axis_positions(self._y_start_nm, self._y_end_nm, self._y_step_nm)

        # Build the flat list of XY grid points
        xy_grid: list[tuple[int, int]] = [
            (x, y) for y in y_positions for x in x_positions
        ]
        total_stacks = len(xy_grid)

        # Build Z positions once; the direction may flip per-stack depending on
        # the current Z, so we store the canonical (start, end) bounds and
        # resolve direction at each stack.
        z_near_base = self._z_start_nm
        z_far_base = self._z_end_nm

        info(
            f"[ZStackAreaScan] Grid: {len(x_positions)} X × {len(y_positions)} Y"
            f" = {total_stacks} stacks"
        )
        info(
            f"[ZStackAreaScan] X: {self._x_start_nm}–{self._x_end_nm} nm"
            f"  step {self._x_step_nm} nm"
        )
        info(
            f"[ZStackAreaScan] Y: {self._y_start_nm}–{self._y_end_nm} nm"
            f"  step {self._y_step_nm} nm"
        )
        info(
            f"[ZStackAreaScan] Z: {self._z_start_nm}–{self._z_end_nm} nm"
            f"  step {self._z_step_nm} nm"
        )
        info(f"[ZStackAreaScan] Output folder: {self._output_folder}")

        self._set_progress(0, total_stacks)

        routine_start = time.monotonic()
        stack_durations: list[float] = []
        total_images_captured = 0

        for stack_idx, (target_x_nm, target_y_nm) in enumerate(xy_grid):
            if self._check_stop():
                break

            stacks_remaining_before = total_stacks - stack_idx
            stacks_completed = stack_idx

            # ----------------------------------------------------------
            # Move to XY position
            # ----------------------------------------------------------
            current_pos = self.motion.get_position()
            xy_target = Position(x=target_x_nm, y=target_y_nm, z=current_pos.z)
            _stacks_done_so_far = len(stack_durations)
            if _stacks_done_so_far > 0:
                _mean_s = sum(stack_durations) / _stacks_done_so_far
                _stacks_remaining = total_stacks - stack_idx
                _eta = round(_stacks_remaining * (_mean_s + 1.0))
            else:
                _eta = 0
            self._set_status(
                f"Stack {stack_idx + 1}/{total_stacks}  —  moving to XY",
                stack_idx,
                total_stacks,
                _eta,
            )
            info(
                f"[ZStackAreaScan] Stack {stack_idx + 1}/{total_stacks}:"
                f" moving to X={target_x_nm / _NM_PER_MM:.6f} mm"
                f"  Y={target_y_nm / _NM_PER_MM:.6f} mm"
            )
            self.motion.move_to_position(xy_target)

            yield  # pause/stop point: after dispatching XY move

            if self._check_stop():
                break

            # Poll until XY settles (1 µm tolerance)
            _tol = 1_000
            _poll = 0.05
            _timeout = 30.0
            elapsed_poll = 0.0
            while elapsed_poll < _timeout:
                pos = self.motion.get_position()
                if abs(pos.x - target_x_nm) <= _tol and abs(pos.y - target_y_nm) <= _tol:
                    break
                time.sleep(_poll)
                elapsed_poll += _poll
                if self._check_stop():
                    break
            else:
                pos = self.motion.get_position()
                warning(
                    f"[ZStackAreaScan] Timed out waiting for XY:"
                    f" actual X={pos.x} nm  Y={pos.y} nm"
                )

            if self._check_stop():
                break

            # Allow the motion system to settle before starting the Z-stack.
            time.sleep(0.2)

            yield  # pause/stop point: XY settled

            # ----------------------------------------------------------
            # Prepare subfolder for this XY position
            # ----------------------------------------------------------
            subfolder_name = f"x{target_x_nm}_y{target_y_nm}"
            subfolder = self._output_folder / subfolder_name
            subfolder.mkdir(parents=True, exist_ok=True)

            # ----------------------------------------------------------
            # Build Z positions, going to the closest Z end first
            # ----------------------------------------------------------
            current_z = self.motion.get_position().z
            if abs(current_z - z_near_base) <= abs(current_z - z_far_base):
                z_near = z_near_base
                z_far = z_far_base
            else:
                z_near = z_far_base
                z_far = z_near_base

            direction = 1 if z_far > z_near else -1
            z_positions: list[int] = []
            z = z_near
            while (direction == 1 and z <= z_far) or (direction == -1 and z >= z_far):
                z_positions.append(z)
                z += direction * self._z_step_nm

            total_z = len(z_positions)
            info(
                f"[ZStackAreaScan]   Z-stack: {total_z} slices"
                f" from {z_near} nm to {z_far} nm"
            )

            # ----------------------------------------------------------
            # Z-stack loop
            # ----------------------------------------------------------
            stack_start = time.monotonic()
            stack_captures = 0

            for z_idx, target_z_nm in enumerate(z_positions):
                if self._check_stop():
                    break

                z_target_pos = Position(
                    x=target_x_nm,
                    y=target_y_nm,
                    z=target_z_nm,
                )
                self._set_activity(
                    f"Stack {stack_idx + 1}/{total_stacks}"
                    f"  —  Z slice {z_idx + 1}/{total_z}"
                )
                info(
                    f"[ZStackAreaScan]   Z slice {z_idx + 1}/{total_z}:"
                    f" moving to Z={target_z_nm / _NM_PER_MM:.6f} mm"
                )
                self.motion.move_to_position(z_target_pos)

                yield  # pause/stop point: after dispatching Z move

                if self._check_stop():
                    break

                # Poll until Z settles
                elapsed_z_poll = 0.0
                while elapsed_z_poll < _timeout:
                    actual_z = self.motion.get_position().z
                    if abs(actual_z - target_z_nm) <= _tol:
                        break
                    time.sleep(_poll)
                    elapsed_z_poll += _poll
                    if self._check_stop():
                        break
                else:
                    warning(
                        f"[ZStackAreaScan]   Timed out waiting for Z={target_z_nm} nm"
                        f" (actual: {self.motion.get_position().z} nm)"
                    )

                if self._check_stop():
                    break

                yield  # pause/stop point: Z settled

                # Capture
                actual_pos = self.motion.get_position()
                filepath = subfolder / f"{actual_pos.z}.jpg"
                info(f"[ZStackAreaScan]   Capturing: {filepath}")

                capture_success: bool | None = None
                capture_error: Exception | None = None
                capture_done = __import__("threading").Event()

                def _on_complete(
                    success: bool,
                    result: object,
                    _done: object = capture_done,
                ) -> None:
                    nonlocal capture_success, capture_error
                    capture_success = success
                    if not success:
                        capture_error = (
                            result
                            if isinstance(result, Exception)
                            else Exception(str(result))
                        )
                    _done.set()  # type: ignore[union-attr]

                capture_start = time.monotonic()
                camera.capture_and_save_still(
                    filepath=filepath,
                    resolution_index=0,
                    additional_metadata={
                        "x_position_nm": actual_pos.x,
                        "y_position_nm": actual_pos.y,
                        "z_position_nm": actual_pos.z,
                        "x_position_mm": actual_pos.x / _NM_PER_MM,
                        "y_position_mm": actual_pos.y / _NM_PER_MM,
                        "z_position_mm": actual_pos.z / _NM_PER_MM,
                        "source": "z_stack_area_scan",
                        "stack_index": stack_idx,
                        "total_stacks": total_stacks,
                        "z_slice_index": z_idx,
                        "total_z_slices": total_z,
                        "xy_subfolder": subfolder_name,
                    },
                    timeout_ms=self._capture_timeout_ms,
                    on_complete=_on_complete,
                    wait=True,
                )

                file_exists = filepath.exists()

                if not capture_success and not file_exists:
                    warning(
                        f"[ZStackAreaScan]   Capture failed at"
                        f" Z={actual_pos.z} nm: {capture_error}"
                    )
                else:
                    if not capture_success and file_exists:
                        warning(
                            f"[ZStackAreaScan]   Capture callback reported failure"
                            f" but file exists — treating as success: {capture_error}"
                        )
                    stack_captures += 1
                    total_images_captured += 1
                    info(f"[ZStackAreaScan]   Saved {filepath}")

                yield  # pause/stop point: after capture

            # ----------------------------------------------------------
            # Post-stack timing and ETA
            # ----------------------------------------------------------
            stack_elapsed = time.monotonic() - stack_start
            stack_durations.append(stack_elapsed)
            stacks_done = len(stack_durations)
            stacks_left = total_stacks - stacks_done

            mean_stack_s = sum(stack_durations) / stacks_done
            # Add 1 second per remaining stack to account for XY travel
            eta_s = stacks_left * (mean_stack_s + 1.0)
            self._set_progress(stacks_done, total_stacks, round(eta_s) if stacks_left > 0 else 0)

            info(
                f"[ZStackAreaScan] Stack {stacks_done}/{total_stacks} complete"
                f"  ({subfolder_name})"
                f"  captured {stack_captures}/{total_z} images"
            )
            info(
                f"[ZStackAreaScan]   Stack duration:  {_fmt_duration(stack_elapsed)}"
                f"  (mean: {_fmt_duration(mean_stack_s)})"
            )
            if stacks_left > 0:
                info(
                    f"[ZStackAreaScan]   Stacks remaining: {stacks_left}"
                    f"  |  ETA: {_fmt_duration(eta_s)}"
                    f"  (includes ~1 s/stack for XY travel)"
                )
            else:
                info("[ZStackAreaScan]   All stacks complete.")

        self._set_activity("Returning home")
        self.motion.home()
        # ------------------------------------------------------------------
        # Final summary
        # ------------------------------------------------------------------
        total_elapsed = time.monotonic() - routine_start
        stacks_completed_final = len(stack_durations)

        info("[ZStackAreaScan] ===== Scan complete =====")
        info(f"[ZStackAreaScan] Total duration:      {_fmt_duration(total_elapsed)}")
        info(f"[ZStackAreaScan] Stacks completed:    {stacks_completed_final} / {total_stacks}")
        info(f"[ZStackAreaScan] Images captured:     {total_images_captured}")
        info(f"[ZStackAreaScan] Output folder:       {self._output_folder}")

        if stack_durations:
            info(
                f"[ZStackAreaScan] Stack time (s):      "
                f"min={min(stack_durations):.3f}"
                f"  max={max(stack_durations):.3f}"
                f"  avg={sum(stack_durations) / len(stack_durations):.3f}"
            )