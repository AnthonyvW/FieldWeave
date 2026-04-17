"""
inspect_calibration.py

Inspect-calibration overlay and its toolbar button.

InspectCalibrationOverlay receives InspectCalibrationResult objects produced
by MachineVisionManager and draws the detection geometry directly using
QPainter.  No image rendering or OpenCV work happens in this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QRect, Slot
from PySide6.QtGui import QPainter, QColor, QPen, QFont

from common.app_context import get_app_context
from UI.widgets.preview_overlay.overlay_base import Overlay

if TYPE_CHECKING:
    from machine_vision.calibration_bar_detection import BarDetectionResult
    from machine_vision.machine_vision_worker import InspectCalibrationResult

_COL_BASELINE = QColor(255, 200, 0)
_COL_TICK     = QColor(0, 255, 0)
_COL_PRESENT  = QColor(0, 255, 0)
_COL_ABSENT   = QColor(255, 0, 0)
_COL_TEXT     = QColor(255, 255, 255, 220)
_COL_SHADOW   = QColor(0, 0, 0, 200)


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


class InspectCalibrationOverlay(Overlay):
    """
    Overlay that draws calibration-bar detection geometry onto the preview.

    Detection is performed by MachineVisionWorker off-thread.  This class
    stores the latest BarDetectionResult and source dimensions, then
    reprojects all coordinates into the display rect at paint time.

    A status line is drawn in the bottom-left corner showing the detected
    axis, tick count, end-cap presence, axis-state mode, and elapsed time.
    """

    def __init__(self) -> None:
        super().__init__()
        self._detection: BarDetectionResult | None = None
        self._source_width: int = 1
        self._source_height: int = 1
        self._status_text: str = ""
        get_app_context().machine_vision.inspect_calibration_result_ready.connect(
            self.receive_result
        )

    # ------------------------------------------------------------------
    # Called by OverlayLabel on every rendered frame (GUI thread)
    # ------------------------------------------------------------------

    def update_full(self, frame: np.ndarray) -> None:
        """
        Called by OverlayLabel.notify_full() with each full-resolution frame.

        Submits an inspection job to the vision manager.  The manager
        silently drops the request if it is still processing the previous frame.
        """
        h, w = frame.shape[:2]
        get_app_context().machine_vision.request_inspect_calibration(frame, w, h)

    # ------------------------------------------------------------------
    # Receive results from MachineVisionManager (GUI thread, queued signal)
    # ------------------------------------------------------------------

    @Slot(object)
    def receive_result(self, result: InspectCalibrationResult) -> None:
        """
        Store the latest detection result for painting.

        Always called on the GUI thread via Qt queued connection.
        """
        self._detection = result.detection
        self._source_width = result.source_width
        self._source_height = result.source_height

        d = result.detection
        start = "S+" if d.start_present else "S-"
        end = "E+" if d.end_present else "E-"
        self._status_text = (
            f"Inspect ({d.mode})  "
            f"axis={d.axis}  ticks={d.tick_count}  "
            f"{start}  {end}  "
            f"{d.elapsed_ms:.1f} ms"
        )

    # ------------------------------------------------------------------
    # Painting (GUI thread, called by OverlayLabel.paintEvent)
    # ------------------------------------------------------------------

    def draw(self, painter: QPainter, rect: QRect) -> None:
        """
        Paint the detection geometry and status text into *rect*.

        *rect* is the pixel-accurate bounding box of the camera image within
        the label (letterboxed), as computed by OverlayLabel._image_rect().
        All source-space coordinates are scaled to display-space here.
        """
        if not self.enabled or self._detection is None:
            return

        d = self._detection
        s = d._downsample
        sx = rect.width()  / self._source_width
        sy = rect.height() / self._source_height
        rx = rect.left()
        ry = rect.top()

        def tx(src_downsampled: int) -> int:
            return rx + int(src_downsampled * s * sx)

        def ty(src_downsampled: int) -> int:
            return ry + int(src_downsampled * s * sy)

        painter.save()

        font = QFont(painter.font())
        font.setPointSize(8)
        painter.setFont(font)

        bl   = d._baseline_index
        bst  = d._bar_start
        ben  = d._bar_end

        if d.axis == "horizontal":
            painter.setPen(QPen(_COL_BASELINE, 2))
            painter.drawLine(tx(bst), ty(bl), tx(ben), ty(bl))

            painter.setPen(QPen(_COL_TICK, 1))
            for ts, te in d._clusters:
                cx = tx((ts + te) // 2)
                painter.drawLine(cx, ry, cx, ry + rect.height())

            start_col = _COL_PRESENT if d.start_present else _COL_ABSENT
            end_col   = _COL_PRESENT if d.end_present   else _COL_ABSENT
            _draw_text_shadowed(painter, tx(bst) + 4,  ty(bl) - 14, "END" if d.start_present else "OPEN", start_col)
            _draw_text_shadowed(painter, tx(ben) - 36, ty(bl) - 14, "END" if d.end_present   else "OPEN", end_col)
            _draw_text_shadowed(painter, rx + 8, ry + 28, "HORIZONTAL")
        else:
            painter.setPen(QPen(_COL_BASELINE, 2))
            painter.drawLine(tx(bl), ty(bst), tx(bl), ty(ben))

            painter.setPen(QPen(_COL_TICK, 1))
            for ts, te in d._clusters:
                cy = ty((ts + te) // 2)
                painter.drawLine(rx, cy, rx + rect.width(), cy)

            start_col = _COL_PRESENT if d.start_present else _COL_ABSENT
            end_col   = _COL_PRESENT if d.end_present   else _COL_ABSENT
            _draw_text_shadowed(painter, tx(bl) + 8, ty(bst) + 20, "END" if d.start_present else "OPEN", start_col)
            _draw_text_shadowed(painter, tx(bl) + 8, ty(ben) - 8,  "END" if d.end_present   else "OPEN", end_col)
            _draw_text_shadowed(painter, rx + 8, ry + 28, "VERTICAL")

        _draw_text_shadowed(painter, rx + 8, ry + 48, f"ticks: {d.tick_count}")
        _draw_text_shadowed(painter, rx + 8, ry + 64, f"vision: {d.elapsed_ms:.1f}ms")
        _draw_text_shadowed(painter, rx + 8, ry + 80, f"mode: {d.mode}")
        _draw_text_shadowed(painter, rx + 8, ry + 96, f"ds: {s}x")

        if self._status_text:
            painter.setPen(QPen(_COL_SHADOW))
            painter.drawText(rect.left() + 6, rect.bottom() - 7, self._status_text)
            painter.setPen(QPen(_COL_TEXT))
            painter.drawText(rect.left() + 5, rect.bottom() - 8, self._status_text)

        painter.restore()
