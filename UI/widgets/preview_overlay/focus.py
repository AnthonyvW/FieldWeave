"""
focus.py

Focus overlay and its toolbar button.

FocusOverlay receives FocusResult objects produced by MachineVisionManager
and composites the pre-rendered heatmap into the camera preview.  All heavy
OpenCV work happens off the GUI thread; this file only handles display.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, QRect, QObject, Signal, Slot
from PySide6.QtGui import QPainter, QImage, QPixmap, QColor, QPen
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from common.app_context import get_app_context
from common.logger import info
from UI.widgets.preview_overlay.overlay_base import Overlay

if TYPE_CHECKING:
    from machine_vision.algorithms.focus_detection import FocusResult


class _ResultRelay(QObject):
    result_ready = Signal(object)


class FocusOverlay(Overlay):
    """
    Overlay that displays a Laplacian focus heatmap.

    The heatmap is rendered by MachineVisionWorker off-thread.  This class
    stores the latest result pixmap and paints it when Qt asks.

    When a focus region is active a dashed rectangle is drawn on top of the
    heatmap showing exactly which area of the frame is being analysed.
    """

    def __init__(self) -> None:
        super().__init__()
        self._result_pixmap: QPixmap | None = None
        self._scores_text: str = ""
        self._relay = _ResultRelay()
        self._relay.result_ready.connect(self._on_result)

    # ------------------------------------------------------------------
    # Called by OverlayLabel on every rendered frame (GUI thread)
    # ------------------------------------------------------------------

    def update_full(self, frame: np.ndarray) -> None:
        """
        Called by OverlayLabel.notify_full() with each full-resolution frame.

        Submits an analysis job to the vision manager and attaches a
        done-callback to deliver the result back to the GUI thread.
        """
        h, w = frame.shape[:2]
        future = get_app_context().machine_vision.request_focus_analysis(frame, w, h)
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
    def _on_result(self, result: FocusResult) -> None:
        arr = result.heatmap_rgb
        image = QImage(
            arr,
            result.source_width,
            result.source_height,
            result.source_width * 3,
            QImage.Format.Format_RGB888,
        )
        self._result_pixmap = QPixmap.fromImage(image.copy())

        s = result.scores
        ceiling_str = "auto"
        active = get_app_context().machine_vision.settings.focus.active
        if not active.auto_ceiling:
            ceiling_str = f"{active.score_ceiling:.1f}"
        self._scores_text = (
            f"Focus ({result.method})  "
            f"whole={s.whole:.2f}  center={s.center:.2f}  peak={s.peak:.2f}"
            f"    raw max={result.raw_score_max:.1f}  ceiling={ceiling_str}"
        )

    # ------------------------------------------------------------------
    # Painting (GUI thread, called by OverlayLabel.paintEvent)
    # ------------------------------------------------------------------

    def draw(self, painter: QPainter, rect: QRect) -> None:
        """
        Paint the heatmap and score text into *rect* on the preview label.

        *rect* is the pixel-accurate bounding box of the camera image within
        the label (letterboxed), as computed by OverlayLabel._image_rect().
        """
        if not self.enabled or self._result_pixmap is None:
            return

        scaled = self._result_pixmap.scaled(
            rect.width(),
            rect.height(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )

        fr = get_app_context().machine_vision.settings.focus.focus_region
        painter.setOpacity(1.0)
        if fr.enabled:
            rw = rect.width()
            rh = rect.height()
            x0 = rect.left() + int(round(rw * fr.left / 100.0))
            x1 = rect.left() + int(round(rw * (1.0 - fr.right / 100.0)))
            y0 = rect.top() + int(round(rh * fr.top / 100.0))
            y1 = rect.top() + int(round(rh * (1.0 - fr.bottom / 100.0)))
            painter.save()
            painter.setClipRect(QRect(x0, y0, x1 - x0, y1 - y0))
            painter.drawPixmap(rect.topLeft(), scaled)
            painter.restore()
        else:
            painter.drawPixmap(rect.topLeft(), scaled)

        self._draw_region_rect(painter, rect)

        if self._scores_text:
            painter.save()
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            painter.setPen(QPen(QColor(0, 0, 0, 200)))
            painter.drawText(rect.left() + 6, rect.bottom() - 7, self._scores_text)
            painter.setPen(QPen(QColor(255, 255, 255, 220)))
            painter.drawText(rect.left() + 5, rect.bottom() - 8, self._scores_text)
            painter.restore()

    def _draw_region_rect(self, painter: QPainter, rect: QRect) -> None:
        """
        Draw a dashed rectangle indicating the active focus region.

        The rectangle is computed from the current ``FocusRegionSettings``
        margins, scaled to the display rect so it always matches what the
        worker actually analyses regardless of the preview resolution.

        Does nothing when the focus region is not enabled.
        """
        fr = get_app_context().machine_vision.settings.focus.focus_region
        if not fr.enabled:
            return

        rw = rect.width()
        rh = rect.height()

        x0 = rect.left() + int(round(rw * fr.left / 100.0))
        x1 = rect.left() + int(round(rw * (1.0 - fr.right / 100.0)))
        y0 = rect.top() + int(round(rh * fr.top / 100.0))
        y1 = rect.top() + int(round(rh * (1.0 - fr.bottom / 100.0)))

        region_rect = QRect(x0, y0, x1 - x0, y1 - y0)

        painter.save()

        painter.setOpacity(0.25)
        painter.setBrush(QColor(0, 0, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRect(rect.left(), rect.top(), rw, y0 - rect.top()))
        painter.drawRect(QRect(rect.left(), y1, rw, rect.bottom() - y1 + 1))
        painter.drawRect(QRect(rect.left(), y0, x0 - rect.left(), y1 - y0))
        painter.drawRect(QRect(x1, y0, rect.right() - x1 + 1, y1 - y0))

        painter.setOpacity(1.0)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(QColor(255, 255, 255, 220))
        pen.setWidth(1)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(region_rect)

        pen_shadow = QPen(QColor(0, 0, 0, 160))
        pen_shadow.setWidth(1)
        pen_shadow.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen_shadow)
        painter.drawRect(region_rect.adjusted(1, 1, 1, 1))

        painter.restore()
