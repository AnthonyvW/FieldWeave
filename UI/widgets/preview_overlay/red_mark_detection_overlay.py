"""
red_mark_detection_overlay.py

Red-mark detection overlay and its painting logic.

RedMarkDetectionOverlay receives RedMarkDetectionResult objects produced by
MachineVisionManager and draws the detection geometry directly using QPainter.
No image rendering or OpenCV work happens in this file.

The overlay detects a red registration mark on the sample tray and reports
the centroid of valid marks, a stabilized reference line (vertical or
horizontal depending on mark distribution), and the offset from image center.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QRect, Qt, QObject, Signal, Slot
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QImage

from common.app_context import get_app_context
from UI.widgets.preview_overlay.overlay_base import Overlay

if TYPE_CHECKING:
    from machine_vision.algorithms.red_mark_detection import RedMarkDetectionResult

_COL_VALID   = QColor(0, 255, 0)
_COL_INVALID = QColor(255, 80, 80)
_COL_LINE    = QColor(255, 220, 0)
_COL_REF     = QColor(255, 220, 0, 80)
_COL_TEXT    = QColor(255, 255, 255, 220)
_COL_SHADOW  = QColor(0, 0, 0, 200)


def _paint_mask(
    painter: QPainter,
    mask: np.ndarray,
    rect: QRect,
    color: QColor,
    alpha: int = 160,
) -> None:
    """
    Paint a uint8 binary mask as a semi-transparent colour overlay.

    Converts the mask to an ARGB QImage (set pixels get ``color`` at
    ``alpha`` opacity, clear pixels are fully transparent), scales it to
    the display rect, and draws it with QPainter.
    """
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    where = mask > 0
    rgba[where, 0] = color.red()
    rgba[where, 1] = color.green()
    rgba[where, 2] = color.blue()
    rgba[where, 3] = alpha
    img = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
    scaled = img.scaled(
        rect.width(), rect.height(),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )
    painter.drawImage(rect.topLeft(), scaled)


def _draw_text_shadowed(
    painter: QPainter,
    x: int,
    y: int,
    text: str,
    color: QColor = _COL_TEXT,
) -> None:
    painter.setPen(QPen(_COL_SHADOW))
    painter.drawText(x + 1, y + 1, text)
    painter.setPen(QPen(color))
    painter.drawText(x, y, text)


class _ResultRelay(QObject):
    result_ready = Signal(object)


class RedMarkDetectionOverlay(Overlay):
    """
    Overlay that draws red-mark detection results onto the preview.

    Detection is performed by MachineVisionWorker off-thread.  This class
    stores the latest RedMarkDetectionResult and source dimensions, then
    reprojects all coordinates into the display rect at paint time.

    Detected mark centroids are drawn as small circles (green = accepted,
    red = filtered out).  A stabilized reference line is drawn across the
    full display width or height depending on the detected mark orientation.
    A dimmer line shows the true image centre for comparison.

    Status text in the top-left shows mark counts, the active line axis,
    the stabilized position, offset from centre, and elapsed processing time.
    """

    def __init__(self) -> None:
        super().__init__()
        self._result: RedMarkDetectionResult | None = None
        self._source_width: int = 1
        self._source_height: int = 1
        self._relay = _ResultRelay()
        self._relay.result_ready.connect(self._on_result)

    # ------------------------------------------------------------------
    # Called by OverlayLabel on every rendered frame (GUI thread)
    # ------------------------------------------------------------------

    def update_full(self, frame: np.ndarray) -> None:
        """
        Called by OverlayLabel.notify_full() with each full-resolution frame.

        Submits a detection job to the vision manager and attaches a
        done-callback to deliver the result back to the GUI thread.
        """
        h, w = frame.shape[:2]
        future = get_app_context().machine_vision.request_red_mark_detection(frame, w, h)
        future.add_done_callback(self._on_future_done)

    # ------------------------------------------------------------------
    # Future done-callback (called on worker thread)
    # ------------------------------------------------------------------

    def _on_future_done(self, future: Future) -> None:
        if future.cancelled() or future.exception() is not None:
            return
        self._relay.result_ready.emit(future.result())

    # ------------------------------------------------------------------
    # Slot — receives result on GUI thread via queued signal
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_result(self, result: RedMarkDetectionResult) -> None:
        self._result = result
        self._source_width = result.source_width
        self._source_height = result.source_height

    # ------------------------------------------------------------------
    # Painting (GUI thread, called by OverlayLabel.paintEvent)
    # ------------------------------------------------------------------

    def draw(self, painter: QPainter, rect: QRect) -> None:
        """
        Paint the detection geometry and status text into *rect*.

        *rect* is the pixel-accurate bounding box of the camera image within
        the label (letterboxed), as computed by OverlayLabel._image_rect().
        All source-space coordinates are scaled to display-space here.

        Blob footprints are painted as semi-transparent colour fills by
        converting the valid/filtered pixel masks to scaled QImages, matching
        the main.py standalone tool behaviour.  Centroid crosshairs are drawn
        on top.
        """
        if not self.enabled or self._result is None:
            return

        r = self._result
        dw = rect.width()
        dh = rect.height()
        rx = rect.left()
        ry = rect.top()
        sx = dw / self._source_width
        sy = dh / self._source_height

        def tx(src_x: float) -> int:
            return rx + int(src_x * sx)

        def ty(src_y: float) -> int:
            return ry + int(src_y * sy)

        painter.save()

        font = QFont(painter.font())
        font.setPointSize(8)
        painter.setFont(font)

        if r.filtered_mask.any():
            _paint_mask(painter, r.filtered_mask, rect, _COL_INVALID)
        if r.valid_mask.any():
            _paint_mask(painter, r.valid_mask, rect, _COL_VALID)

        painter.setPen(QPen(_COL_VALID, 1))
        for cx, cy in r.valid_centers:
            painter.drawLine(tx(cx) - 6, ty(cy), tx(cx) + 6, ty(cy))
            painter.drawLine(tx(cx), ty(cy) - 6, tx(cx), ty(cy) + 6)

        if r.line_orientation == "horizontal":
            if r.image_center_y is not None:
                painter.setPen(QPen(_COL_REF, 1))
                painter.drawLine(rx, ty(r.image_center_y), rx + dw, ty(r.image_center_y))
            if r.stabilized_y is not None:
                painter.setPen(QPen(_COL_LINE, 2))
                painter.drawLine(rx, ty(r.stabilized_y), rx + dw, ty(r.stabilized_y))
        else:
            if r.image_center_x is not None:
                painter.setPen(QPen(_COL_REF, 1))
                painter.drawLine(tx(r.image_center_x), ry, tx(r.image_center_x), ry + dh)
            if r.stabilized_x is not None:
                painter.setPen(QPen(_COL_LINE, 2))
                painter.drawLine(tx(r.stabilized_x), ry, tx(r.stabilized_x), ry + dh)

        y_cursor = ry + 28
        _draw_text_shadowed(painter, rx + 8, y_cursor, f"marks: {len(r.valid_centers)} valid  {len(r.filtered_centers)} filtered")
        y_cursor += 16
        _draw_text_shadowed(painter, rx + 8, y_cursor, f"axis: {r.line_orientation}")
        y_cursor += 16

        if r.line_orientation == "horizontal" and r.stabilized_y is not None and r.image_center_y is not None:
            offset = r.stabilized_y - r.image_center_y
            direction = "down" if offset >= 0 else "up"
            pct = abs(offset) / r.image_center_y * 100.0 if r.image_center_y > 0 else 0.0
            _draw_text_shadowed(painter, rx + 8, y_cursor, f"Y stable: {r.stabilized_y:.1f} px")
            y_cursor += 16
            _draw_text_shadowed(painter, rx + 8, y_cursor, f"offset: {abs(offset):.1f} px ({pct:.1f}% {direction})")
            y_cursor += 16
        elif r.line_orientation == "vertical" and r.stabilized_x is not None and r.image_center_x is not None:
            offset = r.stabilized_x - r.image_center_x
            direction = "right" if offset >= 0 else "left"
            pct = abs(offset) / r.image_center_x * 100.0 if r.image_center_x > 0 else 0.0
            _draw_text_shadowed(painter, rx + 8, y_cursor, f"X stable: {r.stabilized_x:.1f} px")
            y_cursor += 16
            _draw_text_shadowed(painter, rx + 8, y_cursor, f"offset: {abs(offset):.1f} px ({pct:.1f}% {direction})")
            y_cursor += 16

        _draw_text_shadowed(painter, rx + 8, y_cursor, f"vision: {r.elapsed_ms:.1f} ms")

        painter.restore()


"""
red_mark.py

Toolbar button for the red-mark detection overlay.
"""

from PySide6.QtCore import Signal
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QPushButton, QWidget


class RedMarkButton(QPushButton):
    """
    Checkable toolbar button that toggles the red-mark detection overlay.

    Draws a small red filled circle with a thin crosshair to indicate the
    mark-detection function.  Checked state is indicated by a highlighted
    background, consistent with the other overlay buttons.
    """

    toggled_red_mark = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(30, 30)
        self.setObjectName("RedMarkButton")
        self.toggled.connect(self.toggled_red_mark)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0

        dot_r = h * 0.18
        arm = h * 0.36
        pen = QPen(QColor(180, 180, 180) if not self.isChecked() else QColor(60, 60, 60))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(int(cx - arm), int(cy), int(cx + arm), int(cy))
        painter.drawLine(int(cx), int(cy - arm), int(cx), int(cy + arm))

        painter.setPen(QPen(QColor(180, 40, 40), 1))
        painter.setBrush(QColor(220, 50, 50))
        painter.drawEllipse(
            int(cx - dot_r), int(cy - dot_r),
            int(dot_r * 2), int(dot_r * 2),
        )

        painter.end()