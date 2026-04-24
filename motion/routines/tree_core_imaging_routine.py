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
5.  Run autofocus descent to find the best Z at the starting position.
6.  Capture an image at the starting position (no fine autofocus — descent
    already leaves the stage at best focus).
7.  Forward sweep: step towards tca.mark_reference_nm in overlap-derived
    increments, stopping before any step that would overshoot the mark.
    Fine autofocus is run at each position before capturing.
8.  Return to the slot's starting position at the focused Z.
9.  Reverse sweep: step away from tca.mark_reference_nm in the same
    increments until background detection reports bare background.  The
    starting position is skipped (already imaged in step 6).  At each
    new position a background check is performed first; if background is
    detected the sweep stops without running autofocus or capturing.
    Otherwise fine autofocus runs and the image is captured.

If ``image_calibration_scale`` is True, the calibration scale bar is imaged
after all slots are complete.  The routine moves to the saved scale bar
start position (from ``machine_vision.settings.inspection_calibration_position``)
and runs :class:`~motion.routines.inspection_calibration_scale_routine.InspectionCalibrationScaleRoutine`,
saving its output into ``<output_folder>/calibration_slide/``.

The step size is derived from the live sensor dimensions (one frame captured
per slot after arriving at the start position) combined with the camera
calibration, so that consecutive frames share ``image_overlap`` fractional
overlap (default 0.4).  For a Y-axis stage the FOV height drives the step;
for X the FOV width does.

Images are saved to ``<output_folder>/<sample_name>/Y<y>_X<x>_Z<z>.<ext>``
where the extension is taken from the camera's current file-format setting and
each coordinate is rounded to the nearest machine step size in nanometres.

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
from motion.routines.autofocus.autofocus_descent_routine import AutofocusDescent
from motion.routines.autofocus.autofocus_fine_routine import AutofocusFine
from motion.routines.inspection_calibration_scale_routine import InspectionCalibrationScaleRoutine
from machine_vision.algorithms.background_detection import is_background_frame

_NM_PER_MM = 1_000_000
_NM_PER_TICK = 10_000
_SETTLE_S = 0.2
_DWELL_S = 1.0


def _run_subroutine(subroutine: AutomationRoutine, parent: AutomationRoutine) -> None:
    """Start *subroutine* and wait for it, stopping it immediately if *parent* is stopped."""
    subroutine.start()
    while subroutine.is_running:
        if parent._check_stop():
            subroutine.stop()
            subroutine.wait()
            return
        time.sleep(0.1)
    subroutine.wait()


def _run_centering(motion: MotionControllerManager, capture_timeout_s: float, parent: AutomationRoutine) -> None:
    centering = RedMarkCenteringRoutine(motion=motion, capture_timeout_s=capture_timeout_s)
    _run_subroutine(centering, parent)


def _run_autofocus_descent(motion: MotionControllerManager, parent: AutomationRoutine) -> int:
    """Run an autofocus descent and return the best Z position in nm."""
    routine = AutofocusDescent(motion=motion)
    _run_subroutine(routine, parent)
    return motion.get_position().z


def _run_autofocus_fine(motion: MotionControllerManager, parent: AutomationRoutine) -> int:
    """Run a fine autofocus pass and return the best Z position in nm."""
    routine = AutofocusFine(motion=motion)
    _run_subroutine(routine, parent)
    return motion.get_position().z


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
    image_calibration_scale:
        When True the calibration scale bar is imaged after all slots are
        processed.  Output is saved to ``<output_folder>/calibration_slide/``.
        Requires a saved scale bar position in the machine vision settings.
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
        image_calibration_scale: bool = False,
    ) -> None:
        super().__init__(motion)
        self._output_folder = Path(output_folder)
        self._slots = list(slots)
        self._capture_timeout_s = capture_timeout_s
        self._image_overlap = image_overlap
        self._image_calibration_scale = image_calibration_scale

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

        camera = ctx.camera
        if camera is None:
            error("[TreeCoreImaging] No camera available — aborting")
            return

        fformat = camera.settings.fformat.value
        motion_settings = ctx.motion.settings
        step_size_nm: int = motion_settings.step_size

        def _round_to_step(value_nm: int) -> int:
            return round(value_nm / step_size_nm) * step_size_nm

        n_slots = len(self._slots)

        def _slot_pct(slot_iter: int, fraction: float) -> int:
            slot_share = 1.0 / n_slots
            return int(round((slot_iter + fraction) * slot_share * 100))

        def _advance(activity: str, slot_iter: int, phase: int, n_phases: int) -> None:
            pct = _slot_pct(slot_iter, phase / n_phases)
            self._set_status(activity, pct, 100)

        def _capture_and_save(slot_folder: Path, pos: Position) -> None:
            y_nm = _round_to_step(pos.y)
            x_nm = _round_to_step(pos.x)
            z_nm = _round_to_step(pos.z)
            filename = f"Y{y_nm}_X{x_nm}_Z{z_nm}.{fformat}"
            filepath = slot_folder / filename
            info(f"[TreeCoreImaging] Capturing image: {filepath}")
            camera.capture_and_save_still(
                filepath=filepath,
                resolution_index=0,
                additional_metadata={
                    "x_position_nm": pos.x,
                    "y_position_nm": pos.y,
                    "z_position_nm": pos.z,
                    "x_position_mm": pos.x / _NM_PER_MM,
                    "y_position_mm": pos.y / _NM_PER_MM,
                    "z_position_mm": pos.z / _NM_PER_MM,
                    "source": "tree_core_imaging",
                },
                timeout_ms=int(self._capture_timeout_s * 1000),
                wait=True,
            )
            info(f"[TreeCoreImaging] Saved {filepath}")

        axis = tca.axis.lower()
        if axis not in ("x", "y"):
            error(f"[TreeCoreImaging] Unsupported axis '{axis}' — aborting")
            return

        if axis == "y":
            axis_max_nm = motion_settings.max_y * _NM_PER_MM
        else:
            axis_max_nm = motion_settings.max_x * _NM_PER_MM

        # ------------------------------------------------------------------
        # Optional calibration scale imaging — run first so it is captured
        # even if the slot run is interrupted partway through.
        # ------------------------------------------------------------------

        if self._image_calibration_scale:
            self._set_status("Imaging calibration scale", 0, 100)
            info("[TreeCoreImaging] Starting calibration scale imaging")

            icp = ctx.machine_vision.settings.inspection_calibration_position
            start_position: Position | None = None
            if getattr(icp, "is_set", False):
                start_position = Position(x=icp.x_nm, y=icp.y_nm, z=icp.z_nm)
                info(
                    f"[TreeCoreImaging] Moving to scale bar position"
                    f" X={icp.x_nm / _NM_PER_MM:.3f} mm"
                    f" Y={icp.y_nm / _NM_PER_MM:.3f} mm"
                    f" Z={icp.z_nm / _NM_PER_MM:.3f} mm"
                )
            else:
                warning("[TreeCoreImaging] No saved scale bar position — starting calibration scale routine from current position")

            cal_slide_folder = self._output_folder / "calibration_slide"
            cal_slide_folder.mkdir(parents=True, exist_ok=True)

            cal_routine = InspectionCalibrationScaleRoutine(
                motion=self.motion,
                output_path=str(cal_slide_folder),
                start_position=start_position,
                capture_timeout_s=self._capture_timeout_s,
            )
            cal_routine.start()

            while cal_routine.is_running:
                if self._check_stop():
                    cal_routine.stop()
                    return
                self._set_status(
                    f"Calibration scale — {cal_routine.activity}",
                    0,
                    100,
                )
                time.sleep(0.25)

            cal_routine.wait()
            info("[TreeCoreImaging] Calibration scale imaging complete")

            yield
            if self._check_stop():
                return

        for slot_iter, (slot_index, slot_name) in enumerate(self._slots):
            if self._check_stop():
                return

            if slot_index >= tca.num_slots:
                warning(
                    f"[TreeCoreImaging] Slot index {slot_index} out of range"
                    f" ({tca.num_slots} slots) — skipping '{slot_name}'"
                )
                continue

            slot = tca.slots[slot_index]
            slot_label = f"Slot {slot_index + 1}: {slot_name}"

            # ------------------------------------------------------------------
            # Create output sub-folder
            # ------------------------------------------------------------------

            slot_folder = self._output_folder / slot_name
            slot_folder.mkdir(parents=True, exist_ok=True)
            info(f"[TreeCoreImaging] Output folder: {slot_folder}")

            # ------------------------------------------------------------------
            # Move to the slot's mark position and centre
            # ------------------------------------------------------------------

            _advance(f"Moving to mark — {slot_label}", slot_iter, 0, 6)
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

            _advance(f"Centering on mark — {slot_label}", slot_iter, 1, 6)
            info(f"[TreeCoreImaging] Running red mark centering for {slot_label}")
            _run_centering(self.motion, self._capture_timeout_s, self)

            yield
            if self._check_stop():
                return

            # ------------------------------------------------------------------
            # Move to starting position (mirrors _on_goto_start_pos_clicked)
            # ------------------------------------------------------------------

            _advance(f"Moving to start position — {slot_label}", slot_iter, 2, 6)
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

            # ------------------------------------------------------------------
            # Autofocus descent to find best Z at the starting position
            # ------------------------------------------------------------------

            _advance(f"Autofocus descent — {slot_label}", slot_iter, 3, 6)
            info(f"[TreeCoreImaging] Running autofocus descent for {slot_label}")
            focused_z_nm = _run_autofocus_descent(self.motion, self)
            info(f"[TreeCoreImaging] Autofocus settled at Z={focused_z_nm / _NM_PER_MM:.3f} mm for {slot_label}")

            if self._check_stop():
                return

            # Capture one frame to get live sensor dimensions for FOV derivation.
            # The sensor size won't change during the run so this is done once per slot.
            size_frame = capture_still_frame(ctx.camera_manager, timeout_s=self._capture_timeout_s)
            if size_frame is None:
                error(f"[TreeCoreImaging] Frame capture for step-size derivation failed — skipping {slot_label}")
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
                continue

            step_nm = int(round(fov_nm * (1.0 - self._image_overlap)))
            step_nm = _round_to_step(step_nm)
            info(
                f"[TreeCoreImaging] sensor={sensor_w}x{sensor_h}"
                f"  FOV={fov_nm:.0f} nm"
                f"  overlap={self._image_overlap:.0%}"
                f"  step={step_nm} nm"
            )

            # Record the starting position and fixed perpendicular coordinate for
            # both sweeps. Z uses the autofocus-determined height.
            start_pos = self.motion.get_position()
            perp_pos = start_pos.x if axis == "y" else start_pos.y
            main_start_nm = start_pos.y if axis == "y" else start_pos.x

            def _in_bounds(main_nm: int) -> bool:
                return 0 <= main_nm <= axis_max_nm

            # ------------------------------------------------------------------
            # Capture at starting position (descent already focused here)
            # ------------------------------------------------------------------

            info(f"[TreeCoreImaging] Capturing starting position image for {slot_label}")
            _capture_and_save(slot_folder, start_pos)

            yield
            if self._check_stop():
                return

            # ------------------------------------------------------------------
            # Forward sweep: starting position → mark_reference_nm
            # Phases 4-5 of 6 within the slot share.
            # ------------------------------------------------------------------

            info(f"[TreeCoreImaging] Beginning forward sweep for {slot_label}")

            direction = 1 if tca.mark_reference_nm > main_start_nm else -1
            current_main_nm = main_start_nm
            forward_span_nm = max(1, abs(tca.mark_reference_nm - main_start_nm))

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
                    target = Position(x=perp_pos, y=next_main_nm, z=focused_z_nm)
                else:
                    target = Position(x=next_main_nm, y=perp_pos, z=focused_z_nm)

                self.motion.move_to_position(target, wait=True)
                current_main_nm = next_main_nm

                info(f"[TreeCoreImaging] Forward sweep — fine autofocus at {current_main_nm} nm")
                sweep_frac = abs(current_main_nm - main_start_nm) / forward_span_nm
                # Forward sweep occupies phases 4–5 (2 of 6 sub-phases = 1/3 of slot share)
                slot_frac = 4 / 6 + sweep_frac * (1 / 6)
                self._set_status(
                    f"Forward sweep — {slot_label}  pos={current_main_nm / _NM_PER_MM:.3f} mm",
                    _slot_pct(slot_iter, slot_frac),
                    100,
                )
                focused_z_nm = _run_autofocus_fine(self.motion, self)
                _capture_and_save(slot_folder, actual_pos)

                yield
                if self._check_stop():
                    return

            info(f"[TreeCoreImaging] Forward sweep complete for {slot_label}")

            # ------------------------------------------------------------------
            # Return to starting position at focused Z
            # ------------------------------------------------------------------

            info(f"[TreeCoreImaging] Returning to start position for {slot_label}")
            if axis == "y":
                focused_start_pos = Position(x=start_pos.x, y=main_start_nm, z=focused_z_nm)
            else:
                focused_start_pos = Position(x=main_start_nm, y=start_pos.y, z=focused_z_nm)
            self.motion.move_to_position(focused_start_pos, wait=True)
            time.sleep(_SETTLE_S)

            yield
            if self._check_stop():
                return

            # ------------------------------------------------------------------
            # Reverse sweep: starting position → background detected
            # The starting position itself is skipped — it was already imaged
            # at the top of the forward sweep.
            # Phase 5-6 (last 1/6 of slot share).
            # ------------------------------------------------------------------

            info(f"[TreeCoreImaging] Beginning reverse sweep for {slot_label}")

            reverse_direction = -direction
            current_main_nm = main_start_nm
            reverse_span_nm = max(1, axis_max_nm)

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
                    target = Position(x=perp_pos, y=next_main_nm, z=focused_z_nm)
                else:
                    target = Position(x=next_main_nm, y=perp_pos, z=focused_z_nm)

                self.motion.move_to_position(target, wait=True)
                current_main_nm = next_main_nm

                frame = capture_still_frame(ctx.camera_manager, timeout_s=self._capture_timeout_s)
                if frame is None:
                    warning("[TreeCoreImaging] Frame capture failed during reverse sweep — stopping sweep")
                    break

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

                info(f"[TreeCoreImaging] Reverse sweep — fine autofocus at {current_main_nm} nm")
                sweep_frac = abs(current_main_nm - main_start_nm) / reverse_span_nm
                # Reverse sweep occupies phase 5–6 (last 1/6 of slot share)
                slot_frac = 5 / 6 + sweep_frac * (1 / 6)
                self._set_status(
                    f"Reverse sweep — {slot_label}  pos={current_main_nm / _NM_PER_MM:.3f} mm",
                    _slot_pct(slot_iter, slot_frac),
                    100,
                )
                focused_z_nm = _run_autofocus_fine(self.motion, self)

                if self._check_stop():
                    return

                actual_pos = self.motion.get_position()
                info(f"[TreeCoreImaging] Reverse sweep position: {current_main_nm} nm  Z={focused_z_nm / _NM_PER_MM:.3f} mm")
                _capture_and_save(slot_folder, actual_pos)

                yield

            yield

        self._set_status("Complete", 100, 100)
        info("[TreeCoreImaging] Imaging run complete")

        yield