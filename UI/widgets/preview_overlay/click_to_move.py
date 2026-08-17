from __future__ import annotations

from PySide6.QtCore import QRect, QPoint
from PySide6.QtGui import QPainter

from common.logger import info, warning
from common.app_context import get_app_context

from UI.widgets.preview_overlay.overlay_base import Overlay
from motion.models import Position
from motion.motion_controller import MotionState


class ClickToMoveOverlay(Overlay):
    """
    Handles pixel-to-world coordinate conversion for click-to-move.

    Click-to-move is active whenever the machine-vision manager reports a
    valid calibration — no external enable/disable management is required.
    Click notifications arrive from ``OverlayLabel`` via
    ``handle_click(widget_x, widget_y, image_rect, full_width, full_height)``.
    """

    def __init__(self) -> None:
        super().__init__()
        # Always report as enabled so OverlayLabel forwards clicks here;
        # handle_click gates on live calibration state instead.
        self.set_enabled(True)

    # ------------------------------------------------------------------
    # Click handling
    # ------------------------------------------------------------------

    def handle_click(
        self,
        widget_x: int,
        widget_y: int,
        image_rect: QRect,
        full_width: int,
        full_height: int,
    ) -> None:
        """
        Process a mouse click from ``OverlayLabel`` at the normal (un-zoomed)
        display scale.

        Parameters
        ----------
        widget_x, widget_y:
            Click position in ``OverlayLabel`` widget coordinates.
        image_rect:
            The rectangle (in widget coordinates) that the camera image
            currently occupies, as returned by ``OverlayLabel._image_rect()``.
        full_width, full_height:
            Full camera-sensor resolution of the current frame.  Used to
            scale the display-space delta to full-resolution pixel space.
        """
        if not image_rect.contains(QPoint(widget_x, widget_y)):
            return

        disp_w = image_rect.width()
        disp_h = image_rect.height()
        if disp_w <= 0 or disp_h <= 0:
            return

        rel_x = widget_x - image_rect.x()
        rel_y = widget_y - image_rect.y()
        scale_x = full_width / disp_w
        scale_y = full_height / disp_h

        self.handle_full_pixel_click(rel_x * scale_x, rel_y * scale_y, full_width, full_height)

    def handle_full_pixel_click(
        self,
        full_px: float,
        full_py: float,
        full_width: int,
        full_height: int,
        reference_x: float | None = None,
        reference_y: float | None = None,
    ) -> None:
        """
        Move to the machine-vision position for a full camera-resolution
        pixel coordinate.

        This is the shared entry point ``handle_click`` uses once it has
        resolved a widget click to full-resolution pixel space. Callers
        that already have that mapping — such as ``ZoomPreviewOverlay``,
        which maps through its own pan/zoom crop instead of the plain
        display-to-full-resolution scale — can call it directly.

        reference_x, reference_y:
            The full-resolution pixel currently shown at the centre of the
            display. Defaults to the sensor's true centre
            (full_width / 2, full_height / 2), which is correct for the
            normal, un-zoomed view. When the display is panned/zoomed, the
            centre of the visible crop is passed instead, so the move is
            relative to what the operator is actually looking at rather
            than the sensor's absolute centre — otherwise the stage would
            recentre on the sensor's optical axis instead of on the
            operator's current view, landing the click somewhere else.
        """
        ctx = get_app_context()
        mv = ctx.machine_vision

        if not mv.is_calibrated:
            warning("ClickToMoveOverlay: click ignored — no calibration")
            return

        if ctx.motion is None or not ctx.motion.is_ready():
            warning("ClickToMoveOverlay: click ignored — motion controller not ready")
            return

        if ctx.motion.routine_running:
            warning("ClickToMoveOverlay: click ignored — automation routine is running")
            return

        state = ctx.motion.get_state()
        if state in (MotionState.HOMING, MotionState.CONNECTING):
            warning(f"ClickToMoveOverlay: click ignored — motion state is {state}")
            return

        if full_width <= 0 or full_height <= 0:
            return

        cal = mv.calibration  # CameraCalibration

        if reference_x is None:
            reference_x = full_width / 2
        if reference_y is None:
            reference_y = full_height / 2

        # ----------------------------------------------------------------
        # Offset from what's currently centred on screen, then re-expressed
        # as an absolute calibration-image pixel around the calibration
        # centre. pixel_to_world_delta subtracts that centre back out, so
        # this nets out to a delta from the on-screen centre, not the
        # sensor's absolute centre.
        # ----------------------------------------------------------------
        offset_x = full_px - reference_x
        offset_y = full_py - reference_y
        cal_px = cal.image_width / 2 + offset_x * (cal.image_width / full_width)
        cal_py = cal.image_height / 2 + offset_y * (cal.image_height / full_height)

        dx_ticks, dy_ticks = cal.pixel_to_world_delta(cal_px, cal_py)

        # pixel_to_world_delta returns ticks (0.01 mm = 10 000 nm).
        _NM_PER_TICK = 10_000
        current = ctx.motion.get_position()
        target = Position(
            x=current.x + int(round(dx_ticks * _NM_PER_TICK)),
            y=current.y + int(round(dy_ticks * _NM_PER_TICK)),
            z=current.z,
        )

        info(
            f"ClickToMove: full=({full_px:.1f}, {full_py:.1f})  "
            f"cal=({cal_px:.1f}, {cal_py:.1f})  "
            f"delta=({dx_ticks:.2f}, {dy_ticks:.2f}) ticks  "
            f"target=({target.x}, {target.y}) nm"
        )

        ctx.motion.move_to_position(target, wait=False)

    # ------------------------------------------------------------------
    # Overlay.draw — nothing to draw
    # ------------------------------------------------------------------

    def draw(self, painter: QPainter, rect: QRect) -> None:
        pass