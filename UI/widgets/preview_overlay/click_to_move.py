from __future__ import annotations

from PySide6.QtCore import Qt, QRect, QPoint, QTimer, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtWidgets import QPushButton, QWidget


from common.logger import info, warning
from UI.widgets.preview_overlay.overlay_base import Overlay

# Fade duration and tick interval for the click indicator.
_FADE_DURATION_MS: int = 800
_FADE_TICK_MS: int = 30

# Visual constants for the click target marker.
_RING_RADIUS: int = 14      # outer ring radius (px in display space)
_DOT_RADIUS: int = 3        # filled centre dot radius
_PEN_WIDTH: int = 2


class ClickToMoveOverlay(Overlay):
    """
    Draws a fading target ring at the last clicked position and handles the
    pixel-to-world coordinate conversion for click-to-move.

    The overlay is enabled/disabled by the toggle button.  When enabled it
    accepts click notifications from ``OverlayLabel`` via
    ``handle_click(widget_x, widget_y, image_rect, full_width, full_height)``.

    If the machine-vision manager has no calibration the click is silently
    ignored (the button should already be hidden/disabled in that case, but
    this is a belt-and-suspenders guard).

    The fading marker is drawn in the overlay's ``draw()`` method, which is
    called from ``OverlayLabel.paintEvent``.  A ``QTimer`` drives repaints
    during the fade so the label refreshes without any external wiring.
    """

    def __init__(self, repaint_target: QWidget) -> None:
        """
        Parameters
        ----------
        repaint_target:
            The widget to call ``update()`` on during the fade animation
            (typically the ``OverlayLabel``).
        """
        super().__init__()
        self._repaint_target = repaint_target

        # Last click position in *display* image coordinates (relative to the
        # top-left of the image rect, not the widget origin).
        self._click_display_x: float = 0.0
        self._click_display_y: float = 0.0
        self._has_click: bool = False

        # Fade state: 1.0 = fully opaque, 0.0 = invisible.
        self._alpha: float = 0.0

        self._fade_timer = QTimer()
        self._fade_timer.setInterval(_FADE_TICK_MS)
        self._fade_timer.timeout.connect(self._tick_fade)

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
        if not self._enabled:
            return

        # Ignore clicks outside the image area.
        if not image_rect.contains(QPoint(widget_x, widget_y)):
            return

        from common.app_context import get_app_context
        from motion.models import Position

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
        # image_rect gives us where the (possibly letterboxed) image sits
        # inside the label widget.
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
        # M_inv and the stored image centre in CameraCalibration were built at
        # calibration resolution (cal.image_width × cal.image_height).  The
        # live stream may be running at a different resolution (full_width ×
        # full_height).  Remap the click into calibration-image pixel space
        # before calling pixel_to_world_delta so that both the centre
        # subtraction and the M_inv multiply operate in a consistent space.
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

        # ----------------------------------------------------------------
        # Show the fading click indicator at the display position
        # ----------------------------------------------------------------
        self._click_display_x = float(rel_x)
        self._click_display_y = float(rel_y)
        self._has_click = True
        self._alpha = 1.0
        self._fade_timer.start()
        self._repaint_target.update()

    # ------------------------------------------------------------------
    # Fade animation
    # ------------------------------------------------------------------

    def _tick_fade(self) -> None:
        step = _FADE_TICK_MS / _FADE_DURATION_MS
        self._alpha = max(0.0, self._alpha - step)
        self._repaint_target.update()
        if self._alpha <= 0.0:
            self._fade_timer.stop()
            self._has_click = False

    # ------------------------------------------------------------------
    # Overlay.draw
    # ------------------------------------------------------------------

    def draw(self, painter: QPainter, rect: QRect) -> None:
        if not self._has_click or self._alpha <= 0.0:
            return

        # Translate display-relative coords back to widget coords.
        cx = int(rect.x() + self._click_display_x)
        cy = int(rect.y() + self._click_display_y)

        opacity = int(self._alpha * 220)

        # Outer ring
        pen = QPen(QColor(255, 140, 0, opacity))
        pen.setWidth(_PEN_WIDTH)
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawEllipse(
            QPoint(cx, cy),
            _RING_RADIUS,
            _RING_RADIUS,
        )

        # Centre dot
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.setBrush(QBrush(QColor(255, 140, 0, opacity)))
        painter.drawEllipse(QPoint(cx, cy), _DOT_RADIUS, _DOT_RADIUS)


# ---------------------------------------------------------------------------
# Button
# ---------------------------------------------------------------------------

class ClickToMoveButton(QPushButton):
    """
    Checkable overlay button that enables/disables click-to-move.

    Only shown (and checkable) when the machine-vision manager reports
    ``is_calibrated``.  Call ``refresh_calibration_state()`` whenever
    calibration status may have changed.
    """

    toggled_click_to_move = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("⊕", parent)
        self.setObjectName("OverlayButton")
        self.setCheckable(True)
        self.setFixedSize(30, 30)
        self.setToolTip("Click-to-Move (requires calibration)")
        self.clicked.connect(self._on_clicked)
        self.refresh_calibration_state()

    def refresh_calibration_state(self) -> None:
        """
        Show/hide and enable/disable the button based on current calibration.

        Call this after a calibration completes or is cleared so the button
        reflects reality without requiring a restart.
        """
        from common.app_context import get_app_context
        try:
            calibrated = get_app_context().machine_vision.is_calibrated
        except Exception:
            calibrated = False

        self.setEnabled(calibrated)
        self.setToolTip(
            "Click-to-Move" if calibrated
            else "Click-to-Move (calibration required)"
        )
        if not calibrated:
            # Force off if calibration was just cleared.
            self.setChecked(False)
            self.toggled_click_to_move.emit(False)

    def _on_clicked(self, checked: bool) -> None:
        info(f"Preview: Click-to-Move {'enabled' if checked else 'disabled'}")
        self.toggled_click_to_move.emit(checked)