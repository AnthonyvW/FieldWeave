"""
Inspection calibration scale routine.

Autofocuses, then walks the stage along the calibration scale bar until it
has traversed the full length of the bar, saving a raw image at each position
into a ``raw_calibration_scale`` folder inside the supplied output path.

Once all images are captured the routine hands the folder to a
:class:`~post_processing.post_processing_routines.StitchAndMeasureRoutine`
via the global :class:`~post_processing.post_processing_manager.PostProcessingManager`
and blocks until stitching and DPI measurement are complete.  The routine only
reports completion after the post-processing step finishes.

The routine:
  1. Runs an autofocus pass in place.
  2. Captures a snap and runs the inspect-calibration machine-vision pipeline
     to determine the bar's axis and which end of the bar is currently visible.
  3. Moves 0.4 mm toward the far (absent) end of the bar.
  4. Repeats until the last visible tick cluster has crossed the image
     centre along the bar axis.
  5. Stitches the captured images and measures DPI.

Images are saved with the stage position encoded in the filename:
  x{X_nm}_y{Y_nm}_z{Z_nm}.jpg

The result is available via :attr:`~AutomationRoutine.result` after the routine
finishes.  On success, ``result.data`` contains all keys produced by
:class:`~post_processing.post_processing_routines.StitchAndMeasureRoutine`:

- ``dpi`` (:class:`float` | None): measured dots per inch.
- ``image_width`` (:class:`int`): stitched image width in pixels.
- ``image_height`` (:class:`int`): stitched image height in pixels.
- ``output_path`` (:class:`str`): path to the stitched output image.
- ``debug_path`` (:class:`str` | None): path to the debug overlay image, if saved.
- ``tick_count`` (:class:`int` | None): number of ticks detected on the scale bar.
- ``tick_count_valid`` (:class:`bool` | None): whether the tick count matches the expected value.
- ``tick_spacing_valid`` (:class:`bool`): whether all inter-tick gaps were within tolerance.
- ``tick_outlier_gaps`` (list): gaps flagged as outliers, each a ``(i, j, gap_px)`` tuple.
- ``stitched_rgb`` (:class:`numpy.ndarray`): the stitched image as an RGB array.

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
    result = routine.result
    if result and result.success:
        print(f"DPI: {result.get('dpi')}")
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import cv2

from common.app_context import get_app_context
from common.logger import info, warning, error
from motion.motion_controller_manager import MotionControllerManager
from motion.models import Position, fraction_to_stage_delta
from motion.routines.automation_routine import AutomationRoutine
from motion.routines.autofocus.autofocus_utils import capture_still_frame
from motion.routines.autofocus.autofocus_routine import Autofocus
from machine_vision.algorithms.calibration_bar_detection import AxisState, process_frame
from post_processing.routines.stitch_and_measure import StitchAndMeasureRoutine

_NM_PER_MM = 1_000_000


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
        return -1 if above >= below else 1
    else:
        centre = (image_height // 2) // downsample
        cluster_centres = [(s + e) // 2 for s, e in clusters]
        above = sum(1 for c in cluster_centres if c > centre)
        below = len(cluster_centres) - above
        return -1 if above >= below else 1


class InspectionCalibrationScaleRoutine(AutomationRoutine):
    """
    Walk the stage along the calibration scale bar, saving images at each step,
    then stitch the images and measure DPI.

    Starting from the current position the routine autofocuses, determines
    the bar axis and travel direction, then steps 0.4 mm at a time toward the
    far end of the bar.  It stops once the last visible tick mark has passed
    through the image centre.

    All images are saved into ``<output_path>/raw_calibration_scale/`` with
    filenames encoding the actual stage position at capture time.

    After all images are captured a
    :class:`~post_processing.post_processing_routines.StitchAndMeasureRoutine`
    is started via the global post-processing manager and the routine blocks
    until it completes.

    On success, :attr:`~AutomationRoutine.result` is populated with
    ``success=True`` and all data keys from
    :class:`~post_processing.post_processing_routines.StitchAndMeasureRoutine`:
    ``dpi``, ``image_width``, ``image_height``, ``output_path``, ``debug_path``,
    ``tick_count``, ``tick_count_valid``, ``tick_spacing_valid``,
    ``tick_outlier_gaps``, and ``stitched_rgb``.

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

    # Phase weight boundaries (percent, 0–100).
    # Weights shift slightly depending on whether a start position move is needed.
    _PHASE_MOVE_END   = 5
    _PHASE_AF_END     = 20
    _PHASE_IMAGING_END = 75
    # Stitching fills the remainder up to 100.

    def __init__(
        self,
        motion: MotionControllerManager,
        output_path: str | Path,
        *,
        start_position: Position | None = None,
        settle_s: float = 0.4,
        max_steps: int = 30,
        capture_timeout_s: float = 10.0,
    ) -> None:
        super().__init__(motion)
        self._output_path = Path(output_path)
        self._start_position = start_position
        self._settle_s = settle_s
        self._max_steps = max_steps
        self._capture_timeout_s = capture_timeout_s

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_percent(self, activity: str, percent: float) -> None:
        """Report progress as a 0–100 percentage rather than a step count."""
        clamped = max(0, min(100, int(round(percent))))
        self._set_status(activity, clamped, 100)

    def _resolve_step_nm(self, mv: object, reference_frame: object, bar_axis: str) -> int:
        """Return the step size in nanometres derived from overlap_y_pct and the frame size.

        The stage moves along X for a horizontal bar and Y for a vertical bar,
        so the relevant frame dimension is selected accordingly.
        """
        h, w = reference_frame.shape[:2]
        overlap_pct = self.motion.settings.automation.overlap_y_pct
        if bar_axis == "horizontal":
            frame_nm, _ = fraction_to_stage_delta(mv.calibration, 1.0, "x", w, h)
        else:
            _, frame_nm = fraction_to_stage_delta(mv.calibration, 1.0, "y", w, h)
        return int(round(frame_nm * (1.0 - overlap_pct / 100.0)))

    # ------------------------------------------------------------------
    # AutomationRoutine implementation
    # ------------------------------------------------------------------

    def steps(self) -> Generator[None, None, None]:  # noqa: C901
        ctx = get_app_context()
        camera_manager = ctx.camera_manager
        mv = ctx.machine_vision
        post_processing = ctx.post_processing

        self._set_percent("Initialising", 0)

        if not ctx.has_camera:
            error("[CalibrationScale] No camera available — aborting")
            self._set_result(success=False)
            return

        if not mv.is_calibrated:
            error("[CalibrationScale] Camera not calibrated — aborting")
            self._set_result(success=False)
            return

        save_dir = self._output_path / "raw_calibration_scale"
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            error(f"[CalibrationScale] Could not create output directory: {exc}")
            self._set_result(success=False)
            return

        info(f"[CalibrationScale] Saving images to: {save_dir}")

        # ------------------------------------------------------------------
        # Step 0: Move to start position (if one was saved)
        # ------------------------------------------------------------------

        if self._start_position is not None:
            self._set_percent("Moving to start position", 0)
            info(
                f"[CalibrationScale] Moving to start position:"
                f" X={self._start_position.x / _NM_PER_MM:.3f} mm"
                f" Y={self._start_position.y / _NM_PER_MM:.3f} mm"
                f" Z={self._start_position.z / _NM_PER_MM:.3f} mm"
            )
            self.motion.move_to_position(self._start_position, wait=True)

            if self._check_stop():
                return

            yield

        # ------------------------------------------------------------------
        # Step 1: Autofocus
        # ------------------------------------------------------------------

        af_start = self._PHASE_MOVE_END if self._start_position is not None else 0
        af_end = self._PHASE_AF_END
        imaging_end = self._PHASE_IMAGING_END

        self._set_percent("Autofocusing", af_start)
        info("[CalibrationScale] Starting autofocus")

        autofocus = Autofocus(motion=self.motion)
        autofocus.start()
        while autofocus.is_running:
            if self._check_stop():
                autofocus.stop()
                autofocus.wait()
                return
            time.sleep(0.1)
        autofocus.wait()

        if self._check_stop():
            return

        yield

        # ------------------------------------------------------------------
        # Step 2: Initial snap to determine axis and travel direction
        # ------------------------------------------------------------------

        self._set_percent("Detecting scale bar axis", af_end)
        info("[CalibrationScale] Capturing initial snap for bar detection")

        if self._settle_s > 0:
            time.sleep(self._settle_s)

        initial_frame = capture_still_frame(camera_manager, timeout_s=self._capture_timeout_s)
        if initial_frame is None:
            error("[CalibrationScale] Initial frame capture failed — aborting")
            self._set_result(success=False)
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
            self._set_result(success=False)
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

        step_nm = self._resolve_step_nm(mv, initial_frame, initial_result.axis)
        info(f"[CalibrationScale] Step size: {step_nm / _NM_PER_MM:.4f} mm"
             f" (overlap_y_pct={self.motion.settings.automation.overlap_y_pct:.1f}%)")

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
        result = initial_result

        while step < self._max_steps:
            if self._check_stop():
                return

            step += 1
            imaging_fraction = min(step / max(1, self._max_steps), 1.0)
            imaging_percent = af_end + imaging_fraction * (imaging_end - af_end)
            self._set_percent(
                f"Imaging step {step}  ticks={result.tick_count}",
                imaging_percent,
            )

            current_pos = self.motion.get_position()
            if initial_result.axis == "horizontal":
                target = Position(
                    x=current_pos.x + direction * step_nm,
                    y=current_pos.y,
                    z=current_pos.z,
                )
            else:
                target = Position(
                    x=current_pos.x,
                    y=current_pos.y + direction * step_nm,
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

            if result._clusters:
                image_mid = (
                    (image_w // 2) // result._downsample
                    if initial_result.axis == "horizontal"
                    else (image_h // 2) // result._downsample
                )
                cluster_centres = [(s + e) // 2 for s, e in result._clusters]
                if direction > 0:
                    last_tick_centre = min(cluster_centres)
                    past_centre = last_tick_centre >= image_mid
                else:
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

        # ------------------------------------------------------------------
        # Step 5: Stitch images and measure DPI
        # ------------------------------------------------------------------

        self._set_percent("Stitching and measuring", imaging_end)
        info(f"[CalibrationScale] Starting stitch-and-measure on {self._output_path}")

        stitch_routine = StitchAndMeasureRoutine(
            settings=post_processing.settings,
            input_folder=str(self._output_path),
        )
        post_processing.start_routine(stitch_routine)

        while stitch_routine.is_running:
            if self._check_stop():
                post_processing.stop_routine()
                return
            self._set_percent(f"Stitching — {stitch_routine.activity}", imaging_end)
            time.sleep(0.25)

        stitch_routine.wait()

        stitch_result = stitch_routine.result

        if stitch_result is None or not stitch_result.success:
            warning("[CalibrationScale] Stitch-and-measure completed but produced no result")
            self._set_result(success=False)
        else:
            info(
                f"[CalibrationScale] Stitch complete —"
                f" DPI={stitch_result.get('dpi')}"
                f" size={stitch_result.get('image_width')}x{stitch_result.get('image_height')}"
                f" output={stitch_result.get('output_path')}"
            )

            mv.settings.inspect_calibration.last_calibrated = (
                datetime.now(timezone.utc).isoformat()
            )
            mv.save_settings()
            mv.notify_settings_changed()
            self._set_result(success=True, **stitch_result.data)

        yield

        self._set_percent("Complete", 100)
        info(
            f"[CalibrationScale] Done — {step} steps taken,"
            f" {len(list(save_dir.glob('*.jpg')))} images saved in {save_dir}"
        )