"""
Camera calibration automation routine.

Performs the three-capture phase-correlation calibration that maps image
pixel coordinates to physical stage coordinates.  The stage moves a known
distance in +X, returns to the reference position, then moves a known
distance in +Y.  A still frame is captured at each position.  The three
frames are submitted to :class:`MachineVisionManager` for processing; on
success the result is persisted automatically.

Usage::

    from common.app_context import get_app_context
    from motion.routines.camera_calibration_routine import CameraCalibrationRoutine

    ctx = get_app_context()
    routine = CameraCalibrationRoutine(motion=ctx.motion)
    routine.start()
    routine.wait()
    if ctx.machine_vision.is_calibrated:
        print("Calibration succeeded")
"""

from __future__ import annotations

import time
from typing import Generator

from common.app_context import get_app_context
from common.logger import info, error
from motion.motion_controller_manager import MotionControllerManager
from motion.models import Position
from motion.routines.automation_routine import AutomationRoutine
from motion.routines.autofocus.autofocus_utils import capture_still_frame

# 1 tick = 0.01 mm = 10 000 nm
_NM_PER_TICK = 10_000
_NM_PER_MM = 1_000_000

# Progress step indices (out of _TOTAL_STEPS)
_STEP_BASE = 1
_STEP_MOVE_X = 2
_STEP_RETURN_X = 3
_STEP_MOVE_Y = 4
_STEP_RETURN_Y = 5
_STEP_SUBMIT = 6
_TOTAL_STEPS = 6


class CameraCalibrationRoutine(AutomationRoutine):
    """
    Automation routine that calibrates the camera-to-stage spatial mapping.

    Captures three still frames — at the reference position, after a +X
    move, and after a +Y move — then submits them to
    :class:`~machine_vision.machine_vision_manager.MachineVisionManager`
    for calibration.  On success the calibration is stored in the manager
    and persisted to disk automatically.

    The calibration move distances are read from
    ``machine_vision.settings.camera_calibration`` at runtime so any UI
    edits are reflected without reconstructing the routine.  They can also
    be overridden directly via the constructor parameters.

    Parameters
    ----------
    motion:
        Active :class:`MotionControllerManager`.
    move_x_ticks:
        Distance to move in +X during calibration, in 0.01 mm tick units.
        ``None`` (default) reads the value from
        ``machine_vision.settings.camera_calibration.move_x_ticks`` at
        runtime.
    move_y_ticks:
        Distance to move in +Y during calibration, in 0.01 mm tick units.
        ``None`` (default) reads the value from
        ``machine_vision.settings.camera_calibration.move_y_ticks`` at
        runtime.
    settle_s:
        Seconds to wait after each move before capturing a still frame.
        Increase this if the stage vibrates noticeably after stopping.
    capture_timeout_s:
        Maximum seconds to wait for each still frame to arrive from the
        camera.
    """

    job_name = "Camera Calibration"

    def __init__(
        self,
        motion: MotionControllerManager,
        *,
        move_x_ticks: int | None = None,
        move_y_ticks: int | None = None,
        settle_s: float = 0.6,
        capture_timeout_s: float = 10.0,
    ) -> None:
        super().__init__(motion)

        # None means "read from settings at runtime".
        self._move_x_ticks_override: int | None = move_x_ticks
        self._move_y_ticks_override: int | None = move_y_ticks
        self._settle_s = settle_s
        self._capture_timeout_s = capture_timeout_s

    # ------------------------------------------------------------------
    # AutomationRoutine implementation
    # ------------------------------------------------------------------

    def steps(self) -> Generator[None, None, None]:
        ctx = get_app_context()
        camera_manager = ctx.camera_manager
        mv = ctx.machine_vision

        self._set_activity("Initialising")

        if not ctx.has_camera:
            error("[CameraCalibration] No camera available — aborting")
            return

        # Read move distances from settings unless the caller overrode them.
        cal_settings = mv.settings.camera_calibration
        move_x_ticks: int = (
            self._move_x_ticks_override
            if self._move_x_ticks_override is not None
            else cal_settings.move_x_ticks
        )
        move_y_ticks: int = (
            self._move_y_ticks_override
            if self._move_y_ticks_override is not None
            else cal_settings.move_y_ticks
        )
        move_x_nm = move_x_ticks * _NM_PER_TICK
        move_y_nm = move_y_ticks * _NM_PER_TICK

        info(
            f"[CameraCalibration] move_x={move_x_ticks} ticks "
            f"({move_x_nm / _NM_PER_MM:.3f} mm)  "
            f"move_y={move_y_ticks} ticks "
            f"({move_y_nm / _NM_PER_MM:.3f} mm)"
        )

        # ----------------------------------------------------------------
        # Capture helper
        # ----------------------------------------------------------------

        import numpy as np

        def capture(label: str) -> np.ndarray | None:
            if self._settle_s > 0:
                time.sleep(self._settle_s)
            frame = capture_still_frame(camera_manager, timeout_s=self._capture_timeout_s)
            if frame is None:
                error(f"[CameraCalibration] Frame capture failed at {label}")
            return frame

        # ----------------------------------------------------------------
        # Move helper — absolute XY, Z unchanged
        # ----------------------------------------------------------------

        def move_xy(x_nm: int, y_nm: int) -> None:
            current = self.motion.get_position()
            self.motion.move_to_position(
                Position(x=x_nm, y=y_nm, z=current.z),
                wait=True,
            )

        # ----------------------------------------------------------------
        # Step 1: capture base frame at reference position
        # ----------------------------------------------------------------

        ref = self.motion.get_position()
        ref_x_mm = ref.x / _NM_PER_MM
        ref_y_mm = ref.y / _NM_PER_MM

        self._set_status(
            f"Capturing base frame  XY=({ref_x_mm:.3f}, {ref_y_mm:.3f}) mm",
            _STEP_BASE - 1, _TOTAL_STEPS,
        )
        info(
            f"[CameraCalibration] Reference position: "
            f"X={ref_x_mm:.3f} mm  Y={ref_y_mm:.3f} mm  Z={ref.z / _NM_PER_MM:.3f} mm"
        )

        frame_base = capture("base")
        if frame_base is None:
            return

        self._set_progress(_STEP_BASE, _TOTAL_STEPS)
        info("[CameraCalibration] Base frame captured")

        yield  # pause/stop point: after base capture

        if self._check_stop():
            return

        # ----------------------------------------------------------------
        # Step 2: move +X, capture X frame
        # ----------------------------------------------------------------

        target_x_nm = ref.x + move_x_nm
        self._set_status(
            f"Moving +X  Δ={move_x_nm / _NM_PER_MM:.3f} mm",
            _STEP_MOVE_X - 1, _TOTAL_STEPS,
        )
        info(
            f"[CameraCalibration] Moving +X to {target_x_nm / _NM_PER_MM:.3f} mm"
        )

        move_xy(target_x_nm, ref.y)

        frame_x = capture("+X")
        if frame_x is None:
            # Return home before aborting so the stage is not left displaced.
            move_xy(ref.x, ref.y)
            return

        self._set_progress(_STEP_MOVE_X, _TOTAL_STEPS)
        info("[CameraCalibration] +X frame captured")

        yield  # pause/stop point: after +X capture

        if self._check_stop():
            move_xy(ref.x, ref.y)
            return

        # ----------------------------------------------------------------
        # Step 3: return to reference
        # ----------------------------------------------------------------

        self._set_status(
            f"Returning to reference  XY=({ref_x_mm:.3f}, {ref_y_mm:.3f}) mm",
            _STEP_RETURN_X - 1, _TOTAL_STEPS,
        )
        move_xy(ref.x, ref.y)
        self._set_progress(_STEP_RETURN_X, _TOTAL_STEPS)
        info("[CameraCalibration] Returned to reference after +X move")

        yield  # pause/stop point: after return

        if self._check_stop():
            return

        # ----------------------------------------------------------------
        # Step 4: move +Y, capture Y frame
        # ----------------------------------------------------------------

        target_y_nm = ref.y + move_y_nm
        self._set_status(
            f"Moving +Y  Δ={move_y_nm / _NM_PER_MM:.3f} mm",
            _STEP_MOVE_Y - 1, _TOTAL_STEPS,
        )
        info(
            f"[CameraCalibration] Moving +Y to {target_y_nm / _NM_PER_MM:.3f} mm"
        )

        move_xy(ref.x, target_y_nm)

        frame_y = capture("+Y")
        if frame_y is None:
            move_xy(ref.x, ref.y)
            return

        self._set_progress(_STEP_MOVE_Y, _TOTAL_STEPS)
        info("[CameraCalibration] +Y frame captured")

        yield  # pause/stop point: after +Y capture

        if self._check_stop():
            move_xy(ref.x, ref.y)
            return

        # ----------------------------------------------------------------
        # Step 5: return to reference
        # ----------------------------------------------------------------

        self._set_status(
            f"Returning to reference  XY=({ref_x_mm:.3f}, {ref_y_mm:.3f}) mm",
            _STEP_RETURN_Y - 1, _TOTAL_STEPS,
        )
        move_xy(ref.x, ref.y)
        self._set_progress(_STEP_RETURN_Y, _TOTAL_STEPS)
        info("[CameraCalibration] Returned to reference after +Y move")

        yield  # pause/stop point: after final return

        if self._check_stop():
            return

        # ----------------------------------------------------------------
        # Step 6: submit frames and wait for the calibration result
        # ----------------------------------------------------------------

        self._set_status("Computing calibration…", _STEP_SUBMIT - 1, _TOTAL_STEPS)
        info("[CameraCalibration] Submitting frames to MachineVisionManager")

        future = mv.submit_calibration_frames_async(
            base_frame=frame_base,
            base_width=frame_base.shape[1],
            base_height=frame_base.shape[0],
            x_frame=frame_x,
            x_width=frame_x.shape[1],
            x_height=frame_x.shape[0],
            y_frame=frame_y,
            y_width=frame_y.shape[1],
            y_height=frame_y.shape[0],
            ref_x=ref.x // _NM_PER_TICK,
            ref_y=ref.y // _NM_PER_TICK,
            ref_z=ref.z // _NM_PER_TICK,
            move_x_ticks=move_x_ticks,
            move_y_ticks=move_y_ticks,
        )

        try:
            calibration = future.result(timeout=30.0)
        except Exception as exc:
            error(f"[CameraCalibration] Calibration computation failed: {exc!r}")
            self._set_status("Failed — see log for details", _TOTAL_STEPS, _TOTAL_STEPS)
            return

        dpi_str = f"  DPI={calibration.dpi:.1f}" if calibration.dpi is not None else ""
        self._set_status(
            f"Done{dpi_str}",
            _TOTAL_STEPS, _TOTAL_STEPS,
        )
        info(
            f"[CameraCalibration] Complete: "
            f"ref=({ref.x // _NM_PER_TICK}, {ref.y // _NM_PER_TICK}) ticks"
            f"  move_x={move_x_ticks} ticks  move_y={move_y_ticks} ticks"
            f"{dpi_str}"
        )