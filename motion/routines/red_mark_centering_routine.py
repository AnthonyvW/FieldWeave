"""
Red mark centering routine.

Captures a single frame, runs the red-mark detection pipeline, then moves
the stage so the mark ends up at the centre of the relevant axis.

Behaviour
---------
- No marks found → do nothing and exit.
- Multiple marks found → use the one closest to the image centre.
- Mark found:
    1. Move to the far edge of the 15 % band centred on the mark (past
       the mark, away from the image centre).
    2. Wait 0.2 s.
    3. Move to place the mark at the image centre.
  The away-move ensures the final approach always arrives from the same
  direction, giving consistent results regardless of starting position.

The "axis" is determined by ``RedMarkDetectionResult.line_orientation``:
  ``'vertical'``   → marks are a vertical reference line → operate on X.
  ``'horizontal'`` → marks are a horizontal reference line → operate on Y.

All stage moves are expressed in nanometres via ``pixel_to_world_delta`` from
the camera calibration.  If no calibration is present the routine aborts with
a warning.

Usage::

    from common.app_context import get_app_context
    from motion.routines.red_mark_centering_routine import RedMarkCenteringRoutine

    ctx = get_app_context()
    routine = RedMarkCenteringRoutine(motion=ctx.motion)
    routine.start()
    routine.wait()
"""

from __future__ import annotations

import time
from typing import Generator

from common.app_context import get_app_context
from common.logger import info, warning, error
from motion.motion_controller_manager import MotionControllerManager
from motion.models import Position, pixels_to_stage_delta
from motion.routines.automation_routine import AutomationRoutine
from motion.routines.autofocus.autofocus_utils import capture_still_frame
from machine_vision.algorithms.red_mark_detection import RedMarkDetectionResult

_CENTRE_BAND = 0.05         # Middle 15 % of the image considered "centred"
_SETTLE_S    = 0.2          # Seconds to wait between the away-move and return-move


def _closest_to_centre(
    centers: list[tuple[float, float]],
    cx: float,
    cy: float,
) -> tuple[float, float]:
    """Return the centroid in *centers* nearest to ``(cx, cy)``."""
    return min(centers, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)


class RedMarkCenteringRoutine(AutomationRoutine):
    """
    Move the stage so that the detected red registration mark sits at the
    centre of the camera frame along its reference axis.

    Parameters
    ----------
    motion:
        Active :class:`MotionControllerManager`.
    capture_timeout_s:
        Seconds to wait for the still-frame capture before giving up.
    """

    job_name = "Red Mark Centering"

    def __init__(
        self,
        motion: MotionControllerManager,
        *,
        capture_timeout_s: float = 10.0,
    ) -> None:
        super().__init__(motion)
        self._capture_timeout_s = capture_timeout_s

    def steps(self) -> Generator[None, None, None]:
        ctx = get_app_context()
        camera_manager = ctx.camera_manager
        mv = ctx.machine_vision

        self._set_status("Initialising", 0, 5)

        if not ctx.has_camera:
            error("[RedMarkCentering] No camera available — aborting")
            return

        if not mv.is_calibrated:
            warning("[RedMarkCentering] No camera calibration — cannot convert pixels to stage units, aborting")
            return

        # ------------------------------------------------------------------
        # Step 1: Capture frame and detect red mark
        # ------------------------------------------------------------------

        self._set_status("Capturing frame", 1, 5)
        info("[RedMarkCentering] Capturing still frame")

        frame = capture_still_frame(camera_manager, timeout_s=self._capture_timeout_s)
        if frame is None:
            error("[RedMarkCentering] Frame capture failed — aborting")
            return

        yield

        if self._check_stop():
            return

        # ------------------------------------------------------------------
        # Step 2: Run detection (guaranteed — we need this specific frame)
        # ------------------------------------------------------------------

        self._set_status("Detecting red mark", 2, 5)
        info("[RedMarkCentering] Submitting frame for red-mark detection")

        h, w = frame.shape[:2]
        future = mv.request_red_mark_detection_guaranteed(frame, w, h)

        result: RedMarkDetectionResult = future.result()

        if not result.valid_centers:
            info("[RedMarkCentering] No red marks detected — nothing to do")
            return

        yield

        if self._check_stop():
            return

        # ------------------------------------------------------------------
        # Step 3: Select target mark and compute required move
        # ------------------------------------------------------------------

        self._set_status("Computing move", 3, 5)

        img_cx = float(w) / 2.0
        img_cy = float(h) / 2.0

        mark_x, mark_y = _closest_to_centre(result.valid_centers, img_cx, img_cy)

        info(
            f"[RedMarkCentering] Selected mark at ({mark_x:.1f}, {mark_y:.1f})"
            f"  orientation={result.line_orientation}"
            f"  image_size={w}x{h}"
        )

        # Determine which pixel axis and image dimension to work with.
        # orientation='vertical'   → vertical reference line → X axis drives centering
        # orientation='horizontal' → horizontal reference line → Y axis drives centering
        if result.line_orientation == "vertical":
            mark_pos   = mark_x
            centre_pos = img_cx
            image_dim  = float(w)
        else:
            mark_pos   = mark_y
            centre_pos = img_cy
            image_dim  = float(h)

        # Band is centred on the mark.
        band_half = (_CENTRE_BAND / 2.0) * image_dim
        band_lo_mark = mark_pos - band_half
        band_hi_mark = mark_pos + band_half
        info(
            f"[RedMarkCentering] mark_pos={mark_pos:.1f}"
            f"  band=[{band_lo_mark:.1f}, {band_hi_mark:.1f}]"
        )

        # Offset from image centre to the mark (used for sign of the away-move).
        pixel_offset_to_mark = mark_pos - centre_pos

        # The away-move always goes to the far edge of the band (the edge
        # further from the image centre), putting the stage on the far side
        # of the mark.  The final move then arrives at centre from that same
        # side consistently, regardless of whether we started inside or
        # outside the band.
        if pixel_offset_to_mark >= 0:
            away_pixel = band_hi_mark - centre_pos
        else:
            away_pixel = band_lo_mark - centre_pos

        # pixel_to_world_delta returns the stage move (in ticks) to bring the
        # given pixel coordinate to the image centre.  We only need it once,
        # for the mark itself.  The away-move is a proportional fraction of
        # that same delta — scaling by away_pixel / pixel_offset_to_mark gives
        # the correct stage distance without any additional calibration call
        # (which would wrongly compute "bring this other pixel to centre").
        cal = mv.calibration

        def _to_stage(sensor_x: float, sensor_y: float) -> tuple[int, int]:
            return pixels_to_stage_delta(cal, sensor_x, sensor_y, w, h)

        if result.line_orientation == "vertical":
            final_dx, final_dy = _to_stage(mark_x, img_cy)
        else:
            final_dx, final_dy = _to_stage(img_cx, mark_y)

        # Scale final delta by the fraction of pixel_offset_to_mark that the
        # away-move covers.  This is exact regardless of axis coupling in M_inv.
        away_ratio = away_pixel / pixel_offset_to_mark if pixel_offset_to_mark != 0 else 0.0
        away_dx = int(round(final_dx * away_ratio))
        away_dy = int(round(final_dy * away_ratio))

        info(
            f"[RedMarkCentering] pixel_offset={pixel_offset_to_mark:.1f}"
            f"  away_pixel={away_pixel:.1f}  away_ratio={away_ratio:.3f}"
            f"  final_delta=({final_dx}, {final_dy}) nm"
        )

        # ------------------------------------------------------------------
        # Step 4: Away-move
        # ------------------------------------------------------------------

        self._set_status("Moving", 4, 5)

        current = self.motion.get_position()
        away_target = Position(
            x=current.x + away_dx,
            y=current.y + away_dy,
            z=current.z,
        )

        info(
            f"[RedMarkCentering] Away-move to"
            f" X={away_target.x} Y={away_target.y}"
        )
        self.motion.move_to_position(away_target, wait=True)

        yield

        if self._check_stop():
            return

        time.sleep(_SETTLE_S)

        # ------------------------------------------------------------------
        # Step 5: Recapture, redetect, and move to centre
        # ------------------------------------------------------------------
        # final_delta was computed from the original frame.  Lens distortion
        # means the calibration is only accurate near the image centre, so
        # after the away-move we recapture and recompute from the new position
        # where the mark is closer to centre and distortion error is smaller.

        self._set_status("Redetecting", 5, 5)
        info("[RedMarkCentering] Recapturing frame after away-move")

        frame2 = capture_still_frame(camera_manager, timeout_s=self._capture_timeout_s)
        if frame2 is None:
            error("[RedMarkCentering] Second frame capture failed — aborting")
            return

        h2, w2 = frame2.shape[:2]
        future2 = mv.request_red_mark_detection_guaranteed(frame2, w2, h2)
        result2: RedMarkDetectionResult = future2.result()

        if not result2.valid_centers:
            warning("[RedMarkCentering] No mark detected after away-move — aborting")
            return

        img_cx2 = float(w2) / 2.0
        img_cy2 = float(h2) / 2.0
        mark_x2, mark_y2 = _closest_to_centre(result2.valid_centers, img_cx2, img_cy2)

        info(
            f"[RedMarkCentering] Post-away mark at ({mark_x2:.1f}, {mark_y2:.1f})"
        )

        def _to_stage2(sensor_x: float, sensor_y: float) -> tuple[int, int]:
            return pixels_to_stage_delta(cal, sensor_x, sensor_y, w2, h2)

        if result2.line_orientation == "vertical":
            final_dx2, final_dy2 = _to_stage2(mark_x2, img_cy2)
        else:
            final_dx2, final_dy2 = _to_stage2(img_cx2, mark_y2)

        info(
            f"[RedMarkCentering] Recomputed final_delta=({final_dx2}, {final_dy2}) nm"
        )

        current2 = self.motion.get_position()
        final_target = Position(
            x=current2.x + final_dx2,
            y=current2.y + final_dy2,
            z=current2.z,
        )

        info(
            f"[RedMarkCentering] Final-move to"
            f" X={final_target.x} Y={final_target.y}"
        )
        self.motion.move_to_position(final_target, wait=True)

        yield

        info("[RedMarkCentering] Centering complete")