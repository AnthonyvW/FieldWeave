"""
background_detection_overlay.py

Background detection overlay.

BackgroundDetectionOverlay receives BackgroundDetectionResult objects produced
by MachineVisionManager and draws a status indicator directly using QPainter.
No image rendering or OpenCV work happens in this file.

The overlay classifies each frame as background (bare black-plastic tray
visible) or foreground (something is on the tray).  Because the result applies
to the whole frame there is nothing to localise spatially: the draw method
renders a coloured border around the preview and a small status readout.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QRect, Qt, QObject, Signal, Slot
from PySide6.QtGui import QPainter, QColor, QPen, QFont

from common.app_context import get_app_context
from UI.widgets.preview_overlay.overlay_base import Overlay

if TYPE_CHECKING:
    from machine_vision.algorithms.background_detection import BackgroundDetectionResult

_COL_BACKGROUND = QColor(0, 200, 100)
_COL_FOREGROUND = QColor(220, 80, 40)
_COL_TEXT       = QColor(255, 255, 255, 220)
_COL_SHADOW     = QColor(0, 0, 0, 200)

_BORDER_WIDTH = 4


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


class BackgroundDetectionOverlay(Overlay):
    """
    Overlay that indicates whether the full frame is classified as background.

    Detection is performed by MachineVisionWorker off-thread.  This class
    stores the latest BackgroundDetectionResult and redraws on the GUI thread.

    A coloured border is drawn around the image rect — green when background
    is detected, red-orange when foreground is present.  A status readout in
    the top-left shows the classification, mean brightness, mean saturation,
    and elapsed processing time.
    """

    def __init__(self) -> None:
        super().__init__()
        self._result: BackgroundDetectionResult | None = None
        self._relay = _ResultRelay()
        self._relay.result_ready.connect(self._on_result)

    def update_full(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        future = get_app_context().machine_vision.request_background_detection(frame, w, h)
        future.add_done_callback(self._on_future_done)

    def _on_future_done(self, future: Future) -> None:
        if future.cancelled() or future.exception() is not None:
            return
        self._relay.result_ready.emit(future.result())

    @Slot(object)
    def _on_result(self, result: BackgroundDetectionResult) -> None:
        self._result = result

    def draw(self, painter: QPainter, rect: QRect) -> None:
        if not self.enabled or self._result is None:
            return

        r = self._result
        color = _COL_BACKGROUND if r.is_background else _COL_FOREGROUND

        painter.save()

        font = QFont(painter.font())
        font.setPointSize(8)
        painter.setFont(font)

        border_pen = QPen(color, _BORDER_WIDTH)
        border_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect.adjusted(
            _BORDER_WIDTH // 2,
            _BORDER_WIDTH // 2,
            -(_BORDER_WIDTH // 2),
            -(_BORDER_WIDTH // 2),
        ))

        rx, ry = rect.left(), rect.top()
        label = "background" if r.is_background else "foreground"
        y_cursor = ry + 28
        _draw_text_shadowed(painter, rx + 8, y_cursor, f"frame: {label}", color)
        y_cursor += 16
        _draw_text_shadowed(painter, rx + 8, y_cursor, f"val median: {r.val_median:.1f}")
        y_cursor += 16
        _draw_text_shadowed(painter, rx + 8, y_cursor, f"val std: {r.val_std:.1f}")
        y_cursor += 16
        _draw_text_shadowed(painter, rx + 8, y_cursor, f"vision: {r.elapsed_ms:.1f} ms")

        painter.restore()
