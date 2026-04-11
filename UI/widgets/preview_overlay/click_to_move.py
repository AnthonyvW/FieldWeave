from __future__ import annotations

from PySide6.QtCore import QRect, QPoint
from PySide6.QtGui import QPainter

from common.logger import info, warning
from common.app_context import get_app_context

from UI.widgets.preview_overlay.overlay_base import Overlay
from motion.models import Position


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
        Process a mouse click from ``OverlayLabel``.

        Parameters
        ----------
        widget_x, widget_y:
            Click position in ``OverlayLabel`` widget coordinates.
        image_rect:
            The rectangle (in widget coordinates) that the camera image
            currently occupies, as returned by ``OverlayLabel._image_rect()``.
        full_width, full_height:
            Full camera-sensor resolution of the current frame.  Used to
            scale the display-space delta to calibration-image space before
            passing it to ``CameraCalibration.pixel_to_world_delta()``.
        """
        # Ignore clicks outside the image area.
        if not image_rect.contains(QPoint(widget_x, widget_y)):
            return

        ctx = get_app_context()
        mv = ctx.machine_vision

        if not mv.is_calibrated:
            warning("ClickToMoveOverlay: click ignored — no calibration")
            return

        if ctx.motion is None or not ctx.motion.is_ready():
            warning("ClickToMoveOverlay: click ignored — motion controller not ready")
            return

        cal = mv.calibration  # CameraCalibration

        # ----------------------------------------------------------------
        # Convert widget click → full-resolution pixel coordinate
        # ----------------------------------------------------------------
        disp_w = image_rect.width()
        disp_h = image_rect.height()

        if disp_w <= 0 or disp_h <= 0:
            return

        # Click position relative to the top-left of the displayed image.
        rel_x = widget_x - image_rect.x()
        rel_y = widget_y - image_rect.y()

        # Scale to full camera resolution.
        scale_x = full_width / disp_w
        scale_y = full_height / disp_h
        full_px = rel_x * scale_x
        full_py = rel_y * scale_y

        # ----------------------------------------------------------------
        # Convert to world delta and move
        # ----------------------------------------------------------------
        # Remap the click into calibration-image pixel space so that the
        # centre subtraction and M_inv multiply in pixel_to_world_delta
        # operate in a consistent space.
        cal_px = full_px * (cal.image_width / full_width)
        cal_py = full_py * (cal.image_height / full_height)

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
            f"ClickToMove: display=({rel_x:.1f}, {rel_y:.1f})  "
            f"full=({full_px:.1f}, {full_py:.1f})  "
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