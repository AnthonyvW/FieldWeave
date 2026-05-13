"""
Z-stack area scan automation routine.

Performs a Z-stack scan at each XY position in a grid, capturing images at
every combination of (X, Y, Z).  Images are saved into per-XY subfolders
inside the output directory.

The routine reports elapsed time after each Z-stack and provides a running
estimate of remaining time, incorporating a 1-second-per-remaining-stack
travel-time allowance.

If a :class:`FocusStackRoutineConfig` is supplied, a :class:`QueuedFocusStackRoutine`
is launched for each XY subfolder after its Z-stack images have been saved to
disk.  The stacked output is written to ``<subfolder>/stacked.<ext>`` where the
extension comes from ``focus_stack_config.output_extension``.

Usage::

    from common.app_context import get_app_context
    from motion.automations.z_stack_area_scan import AreaScan
    from post_processing.routines.focus_stack_routine import FocusStackRoutineConfig

    ctx = get_app_context()
    routine = AreaScan(
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
        focus_stack_config=FocusStackRoutineConfig(),
    )
    routine.start()
    routine.wait()
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Generator, TYPE_CHECKING

from common.app_context import get_app_context
from common.logger import info, warning, error
from motion.motion_controller_manager import MotionControllerManager
from motion.models import Position

from motion.routines.automation_routine import AutomationRoutine
from post_processing.routines.focus_stack_routine import QueuedFocusStackRoutine

if TYPE_CHECKING:
    from post_processing.post_processing_manager import PostProcessingManager
    from post_processing.routines.focus_stack_routine import FocusStackRoutineConfig

_NM_PER_MM = 1_000_000


def _write_scan_profile(
    output_folder: Path,
    x_start_nm: int,
    y_start_nm: int,
    z_start_nm: int,
    x_step_nm: int,
    y_step_nm: int,
    z_step_nm: int,
    x_positions: list[int],
    y_positions: list[int],
    z_positions_count: int,
    dpi: float | None,
    total_elapsed_s: float,
    stack_profiles: list[dict],
    total_images_captured: int,
    x_settle_ms: int,
    y_settle_ms: int,
    travel_settle_ms: int,
    z_settle_ms: int,
    capture_timeout_ms: int,
    scan_strategy: str,
) -> bool:
    x_count = len(x_positions)
    y_count = len(y_positions)
    total_stacks = x_count * y_count

    durations = [s["duration_s"] for s in stack_profiles]
    mean_s = sum(durations) / len(durations) if durations else 0.0

    all_slices = [sl for s in stack_profiles for sl in s.get("slice_profiles", [])]
    mean_z_move_s = sum(sl["z_move_s"] for sl in all_slices) / len(all_slices) if all_slices else None
    mean_z_settle_s = sum(sl["z_settle_s"] for sl in all_slices) / len(all_slices) if all_slices else None
    mean_capture_s = sum(sl["capture_s"] for sl in all_slices) / len(all_slices) if all_slices else None
    mean_xy_move_s = sum(s["xy_move_s"] for s in stack_profiles) / len(stack_profiles) if stack_profiles else None
    mean_xy_settle_s = sum(s["xy_settle_s"] for s in stack_profiles) / len(stack_profiles) if stack_profiles else None

    dpi_line = f"  DPI: {dpi:.2f}" if dpi is not None else "  DPI: not available"

    if durations:
        timing_lines = (
            f"  Stack duration - min: {min(durations):.3f} s,"
            f" max: {max(durations):.3f} s,"
            f" mean: {mean_s:.3f} s\n"
            f"  Per-slice averages across all stacks:\n"
            f"    XY move:    {mean_xy_move_s:.3f} s  (settle: {mean_xy_settle_s:.3f} s)\n"
            f"    Z move:     {mean_z_move_s:.3f} s  (settle: {mean_z_settle_s:.3f} s)\n"
            f"    Capture:    {mean_capture_s:.3f} s\n"
            f"  Configured settle times: X={x_settle_ms} ms  Y={y_settle_ms} ms  travel={travel_settle_ms} ms  Z={z_settle_ms} ms\n"
            f"  Capture timeout: {capture_timeout_ms} ms"
        )
    else:
        timing_lines = "  No stacks completed."

    description = (
        f"Z-Stack Area Scan - {output_folder.name}\n"
        f"  Grid: {x_count} X x {y_count} Y = {total_stacks} stacks,"
        f" {z_positions_count} Z slices each\n"
        f"  X: start {x_start_nm} nm, step {x_step_nm} nm"
        f" ({x_start_nm / _NM_PER_MM:.6f} mm, step {x_step_nm / _NM_PER_MM:.6f} mm)\n"
        f"  Y: start {y_start_nm} nm, step {y_step_nm} nm"
        f" ({y_start_nm / _NM_PER_MM:.6f} mm, step {y_step_nm / _NM_PER_MM:.6f} mm)\n"
        f"  Z: start {z_start_nm} nm, step {z_step_nm} nm"
        f" ({z_start_nm / _NM_PER_MM:.6f} mm, step {z_step_nm / _NM_PER_MM:.6f} mm)\n"
        f"{dpi_line}\n"
        f"  Total elapsed: {_fmt_duration(total_elapsed_s)}"
        f" ({total_elapsed_s:.3f} s)\n"
        f"  Stacks completed: {len(stack_profiles)} / {total_stacks}\n"
        f"  Images captured: {total_images_captured}\n"
        f"{timing_lines}"
    )

    profile = {
        "description": description,
        "scan_parameters": {
            "output_folder": str(output_folder),
            "dpi": dpi,
            "scan_strategy": scan_strategy,
            "x_start_nm": x_start_nm,
            "x_start_mm": x_start_nm / _NM_PER_MM,
            "x_step_nm": x_step_nm,
            "x_step_mm": x_step_nm / _NM_PER_MM,
            "y_start_nm": y_start_nm,
            "y_start_mm": y_start_nm / _NM_PER_MM,
            "y_step_nm": y_step_nm,
            "y_step_mm": y_step_nm / _NM_PER_MM,
            "z_start_nm": z_start_nm,
            "z_start_mm": z_start_nm / _NM_PER_MM,
            "z_step_nm": z_step_nm,
            "z_step_mm": z_step_nm / _NM_PER_MM,
            "x_positions_count": x_count,
            "y_positions_count": y_count,
            "z_slices_per_stack": z_positions_count,
            "total_stacks": total_stacks,
            "x_settle_ms": x_settle_ms,
            "y_settle_ms": y_settle_ms,
            "travel_settle_ms": travel_settle_ms,
            "z_settle_ms": z_settle_ms,
            "capture_timeout_ms": capture_timeout_ms,
        },
        "summary": {
            "total_elapsed_s": round(total_elapsed_s, 3),
            "total_elapsed_formatted": _fmt_duration(total_elapsed_s),
            "stacks_completed": len(stack_profiles),
            "total_images_captured": total_images_captured,
            "stack_duration_min_s": round(min(durations), 3) if durations else None,
            "stack_duration_max_s": round(max(durations), 3) if durations else None,
            "stack_duration_mean_s": round(mean_s, 3) if durations else None,
            "mean_xy_move_s": round(mean_xy_move_s, 3) if mean_xy_move_s is not None else None,
            "mean_xy_settle_s": round(mean_xy_settle_s, 3) if mean_xy_settle_s is not None else None,
            "mean_z_move_s": round(mean_z_move_s, 3) if mean_z_move_s is not None else None,
            "mean_z_settle_s": round(mean_z_settle_s, 3) if mean_z_settle_s is not None else None,
            "mean_capture_s": round(mean_capture_s, 3) if mean_capture_s is not None else None,
        },
        "stack_profiles": stack_profiles,
    }

    profile_path = output_folder / "scan_profile.json"
    with profile_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    info(f"[AreaScan] Scan profile written to {profile_path}")
    return True


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


class AreaScan(AutomationRoutine):
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

    If *focus_stack_config* is provided, a :class:`QueuedFocusStackRoutine` is
    launched for each XY subfolder immediately after its images have been saved.
    The stacked output is written to ``<subfolder>/stacked.<ext>``.

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
    focus_stack_config:
        When supplied, a :class:`QueuedFocusStackRoutine` is run against each
        XY subfolder after its images are saved.  Pass ``None`` to skip
        focus stacking.

    Settings read at runtime
    ------------------------
    ``ctx.settings.motion.automation.capture_timeout_ms``:
        How long (ms) to wait for each image capture to complete.
    ``ctx.settings.motion.automation.settle_x_ms``:
        Settle time after each X-only move, in milliseconds.
    ``ctx.settings.motion.automation.settle_y_ms``:
        Settle time after each Y-only move, in milliseconds.
    ``ctx.settings.motion.automation.settle_z_ms``:
        Settle time after each Z move before triggering the camera.
    ``ctx.settings.motion.automation.settle_travel_ms``:
        Settle time after the combined XY travel move to each grid position.
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
        focus_stack_config: FocusStackRoutineConfig | None = None,
        scan_strategy: str = "snake",
    ) -> None:
        super().__init__(motion)

        for name, value in (
            ("x_step_nm", x_step_nm),
            ("y_step_nm", y_step_nm),
            ("z_step_nm", z_step_nm),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if scan_strategy not in ("snake", "line"):
            raise ValueError(f"scan_strategy must be 'snake' or 'line', got {scan_strategy!r}")

        ctx = get_app_context()

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
        self._focus_stack_config = focus_stack_config
        self._scan_strategy = scan_strategy

        template = ctx.motion.settings.z_stack_area_scan.image_name_template if ctx.motion.settings else "{d}_{i}"
        ctx.image_name_formatter.set_template(template)
        ctx.image_name_formatter.set_index(1)

    # ------------------------------------------------------------------
    # AutomationRoutine implementation
    # ------------------------------------------------------------------

    def steps(self) -> Generator[None, None, None]:  # noqa: C901
        ctx = get_app_context()
        camera = ctx.camera
        post_processing = ctx.post_processing

        automation = ctx.motion.settings.automation
        capture_timeout_ms = automation.capture_timeout_ms
        x_settle_ms = automation.settle_x_ms
        y_settle_ms = automation.settle_y_ms
        travel_settle_ms = automation.settle_travel_ms
        z_settle_ms = automation.settle_z_ms

        self._set_activity("Initialising")

        if camera is None:
            error("[AreaScan] No camera available - aborting")
            return

        self._output_folder.mkdir(parents=True, exist_ok=True)

        x_positions = _build_axis_positions(self._x_start_nm, self._x_end_nm, self._x_step_nm)
        y_positions = _build_axis_positions(self._y_start_nm, self._y_end_nm, self._y_step_nm)

        # Build the flat list of XY grid points.
        # Snake strategy reverses the X order on every other row so that only a
        # single Y move is needed between rows. Line strategy always traverses X
        # in the same direction.
        xy_grid: list[tuple[int, int]] = []
        for row_idx, y in enumerate(y_positions):
            row_x = x_positions if (self._scan_strategy == "line" or row_idx % 2 == 0) else list(reversed(x_positions))
            xy_grid.extend((x, y) for x in row_x)
        total_stacks = len(xy_grid)

        # Build Z positions once; the direction may flip per-stack depending on
        # the current Z, so we store the canonical (start, end) bounds and
        # resolve direction at each stack.
        z_near_base = self._z_start_nm
        z_far_base = self._z_end_nm

        info(
            f"[AreaScan] Grid: {len(x_positions)} X × {len(y_positions)} Y"
            f" = {total_stacks} stacks"
        )
        info(
            f"[AreaScan] X: {self._x_start_nm}–{self._x_end_nm} nm"
            f"  step {self._x_step_nm} nm"
        )
        info(
            f"[AreaScan] Y: {self._y_start_nm}–{self._y_end_nm} nm"
            f"  step {self._y_step_nm} nm"
        )
        info(
            f"[AreaScan] Z: {self._z_start_nm}–{self._z_end_nm} nm"
            f"  step {self._z_step_nm} nm"
        )
        info(f"[AreaScan] Output folder: {self._output_folder}")
        info(f"[AreaScan] Scan strategy: {self._scan_strategy}")
        info(f"[AreaScan] Settle times: X={x_settle_ms} ms  Y={y_settle_ms} ms  travel={travel_settle_ms} ms  Z={z_settle_ms} ms")
        info(f"[AreaScan] Capture timeout: {capture_timeout_ms} ms")
        if self._focus_stack_config is not None:
            ext = self._focus_stack_config.output_extension
            info(f"[AreaScan] Focus stacking enabled - output extension: {ext}")

        self._set_progress(0, total_stacks)

        routine_start = time.monotonic()
        stack_durations: list[float] = []
        stack_profiles: list[dict] = []
        total_images_captured = 0

        dpi: float | None = None
        mv = getattr(ctx, "machine_vision", None)
        if mv is not None:
            mv_settings = getattr(mv, "settings", None)
            if mv_settings is not None:
                dpi = getattr(mv_settings, "dpi", None)

        for stack_idx, (target_x_nm, target_y_nm) in enumerate(xy_grid):
            if self._check_stop():
                break

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
                f"Stack {stack_idx + 1}/{total_stacks}  -  moving to XY",
                stack_idx,
                total_stacks,
                _eta,
            )
            info(
                f"[AreaScan] Stack {stack_idx + 1}/{total_stacks}:"
                f" moving to X={target_x_nm / _NM_PER_MM:.6f} mm"
                f"  Y={target_y_nm / _NM_PER_MM:.6f} mm"
            )
            stack_start_move = time.monotonic()
            self.motion.move_to_position(xy_target, wait=True)
            xy_move_s = time.monotonic() - stack_start_move

            yield  # pause/stop point: after XY move

            if self._check_stop():
                break

            xy_settle_start = time.monotonic()
            x_changed = target_x_nm != current_pos.x
            y_changed = target_y_nm != current_pos.y
            if x_changed and y_changed:
                settle_ms = travel_settle_ms
            elif x_changed:
                settle_ms = x_settle_ms
            else:
                settle_ms = y_settle_ms
            if settle_ms > 0:
                time.sleep(settle_ms / 1000.0)
            xy_settle_s = time.monotonic() - xy_settle_start

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

            z_positions: list[int] = []
            if z_near == z_far:
                z_positions = [z_near]
            else:
                direction = 1 if z_far > z_near else -1
                z = z_near
                while (direction == 1 and z <= z_far) or (direction == -1 and z >= z_far):
                    z_positions.append(z)
                    z += direction * self._z_step_nm

            total_z = len(z_positions)
            info(
                f"[AreaScan]   Z-stack: {total_z} slices"
                f" from {z_near} nm to {z_far} nm"
            )

            # ----------------------------------------------------------
            # Z-stack loop
            # ----------------------------------------------------------
            stack_start = time.monotonic()

            # Tracks pending background saves for this stack so we can verify
            # all images landed on disk before moving to the next XY position.
            stack_pending_saves: list[tuple[int, Path, threading.Event, list[bool]]] = []
            slice_profiles: list[dict] = []

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
                    f"  -  Z slice {z_idx + 1}/{total_z}"
                )
                info(
                    f"[AreaScan]   Z slice {z_idx + 1}/{total_z}:"
                    f" moving to Z={target_z_nm / _NM_PER_MM:.6f} mm"
                )
                z_move_start = time.monotonic()
                self.motion.move_to_position(z_target_pos, wait=True)
                z_move_s = time.monotonic() - z_move_start

                yield  # pause/stop point: after Z move

                if self._check_stop():
                    break

                z_settle_start = time.monotonic()
                if z_settle_ms > 0:
                    time.sleep(z_settle_ms / 1000.0)
                z_settle_s = time.monotonic() - z_settle_start

                actual_pos = self.motion.get_position()
                filepath = subfolder / f"{actual_pos.z}.jpg"
                info(f"[AreaScan]   Capturing: {str(filepath)}")

                save_done = threading.Event()
                success_cell: list[bool] = [False]

                def _on_complete(success: bool, _done=save_done, _cell=success_cell) -> None:
                    _cell[0] = success
                    _done.set()

                capture_start = time.monotonic()
                # wait=True blocks until snap+pull completes so the stage is free
                # to move to the next Z immediately. The save runs in the background.
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
                    timeout_ms=capture_timeout_ms,
                    on_complete=_on_complete,
                    wait=True,
                )
                capture_s = time.monotonic() - capture_start

                slice_profiles.append({
                    "z_idx": z_idx,
                    "z_nm": actual_pos.z,
                    "z_mm": actual_pos.z / _NM_PER_MM,
                    "z_move_s": round(z_move_s, 3),
                    "z_settle_s": round(z_settle_s, 3),
                    "capture_s": round(capture_s, 3),
                })

                stack_pending_saves.append((actual_pos.z, filepath, save_done, success_cell))

                yield  # pause/stop point: after capture

            # ----------------------------------------------------------
            # Drain saves for this stack before moving to the next XY
            # ----------------------------------------------------------
            save_timeout_s = capture_timeout_ms / 1000.0
            stack_captures = 0
            for z_nm, filepath, save_done, success_cell in stack_pending_saves:
                save_done.wait(timeout=save_timeout_s)
                if success_cell[0]:
                    stack_captures += 1
                    total_images_captured += 1
                    info(f"[AreaScan]   Saved {filepath}")
                else:
                    warning(f"[AreaScan]   Save failed at Z={z_nm} nm")

            # Enqueue focus stack for this XY - the manager's worker thread
            # runs them one at a time while imaging continues freely.
            if self._focus_stack_config is not None and stack_captures > 0:
                if not self._check_stop():
                    self._enqueue_focus_stack(post_processing, subfolder)

            # ----------------------------------------------------------
            # Post-stack timing and ETA
            # ----------------------------------------------------------
            stack_elapsed = time.monotonic() - stack_start
            stack_durations.append(stack_elapsed)
            stacks_done = len(stack_durations)
            stacks_left = total_stacks - stacks_done

            stack_profiles.append({
                "stack_index": stack_idx,
                "x_nm": target_x_nm,
                "y_nm": target_y_nm,
                "x_mm": target_x_nm / _NM_PER_MM,
                "y_mm": target_y_nm / _NM_PER_MM,
                "z_slices": total_z,
                "images_saved": stack_captures,
                "duration_s": round(stack_elapsed, 3),
                "xy_move_s": round(xy_move_s, 3),
                "xy_settle_s": round(xy_settle_s, 3),
                "slice_profiles": slice_profiles,
            })

            mean_stack_s = sum(stack_durations) / stacks_done
            eta_s = stacks_left * (mean_stack_s + 1.0)
            self._set_progress(stacks_done, total_stacks, round(eta_s) if stacks_left > 0 else 0)

            info(
                f"[AreaScan] Stack {stacks_done}/{total_stacks} complete"
                f"  ({subfolder_name})"
                f"  saved {stack_captures}/{total_z} images"
            )
            info(
                f"[AreaScan]   Stack duration:  {_fmt_duration(stack_elapsed)}"
                f"  (mean: {_fmt_duration(mean_stack_s)})"
            )
            if stacks_left > 0:
                info(
                    f"[AreaScan]   Stacks remaining: {stacks_left}"
                    f"  |  ETA: {_fmt_duration(eta_s)}"
                    f"  (includes ~1 s/stack for XY travel)"
                )
            else:
                info("[AreaScan]   All stacks complete.")

        self._set_activity("Returning home")
        self.motion.home()

        if self._focus_stack_config is not None and post_processing is not None:
            self._set_activity("Waiting for focus stacking to finish")
            post_processing.wait_for_queue(check_stop=self._check_stop)

        # ------------------------------------------------------------------
        # Final summary
        # ------------------------------------------------------------------
        total_elapsed = time.monotonic() - routine_start
        stacks_completed_final = len(stack_durations)

        info("[AreaScan] ===== Scan complete =====")
        info(f"[AreaScan] Total duration:      {_fmt_duration(total_elapsed)}")
        info(f"[AreaScan] Stacks completed:    {stacks_completed_final} / {total_stacks}")
        info(f"[AreaScan] Images captured:     {total_images_captured}")
        info(f"[AreaScan] Output folder:       {self._output_folder}")

        if stack_durations:
            info(
                f"[AreaScan] Stack time (s):      "
                f"min={min(stack_durations):.3f}"
                f"  max={max(stack_durations):.3f}"
                f"  avg={sum(stack_durations) / len(stack_durations):.3f}"
            )

        z_canonical_count = len(
            _build_axis_positions(self._z_start_nm, self._z_end_nm, self._z_step_nm)
        )
        _write_scan_profile(
            output_folder=self._output_folder,
            x_start_nm=self._x_start_nm,
            y_start_nm=self._y_start_nm,
            z_start_nm=self._z_start_nm,
            x_step_nm=self._x_step_nm,
            y_step_nm=self._y_step_nm,
            z_step_nm=self._z_step_nm,
            x_positions=x_positions,
            y_positions=y_positions,
            z_positions_count=z_canonical_count,
            dpi=dpi,
            total_elapsed_s=total_elapsed,
            stack_profiles=stack_profiles,
            total_images_captured=total_images_captured,
            x_settle_ms=x_settle_ms,
            y_settle_ms=y_settle_ms,
            travel_settle_ms=travel_settle_ms,
            z_settle_ms=z_settle_ms,
            capture_timeout_ms=capture_timeout_ms,
            scan_strategy=self._scan_strategy,
        )

    # ------------------------------------------------------------------
    # Focus stack helper
    # ------------------------------------------------------------------

    def _enqueue_focus_stack(
        self,
        post_processing: PostProcessingManager | None,
        subfolder: Path,
    ) -> None:
        if post_processing is None:
            error("[AreaScan] No post_processing manager available - skipping focus stack")
            return

        cfg = self._focus_stack_config
        ext = cfg.output_extension
        stacked_folder = self._output_folder / "focus_stacked"
        stacked_folder.mkdir(parents=True, exist_ok=True)
        ctx = get_app_context()
        image_name = ctx.image_name_formatter.get_formatted_string(auto_increment_index=True)
        output_path = str(stacked_folder / f"{image_name}.{ext}")

        routine = QueuedFocusStackRoutine(
            settings=post_processing.settings,
            input_folder=str(subfolder),
            output_path=output_path,
            config=cfg,
        )
        post_processing.queue_routine(routine)
        info(f"[AreaScan]   Queued focus stack - output: {output_path}")