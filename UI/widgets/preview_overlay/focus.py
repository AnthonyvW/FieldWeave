"""
focus.py

Focus overlay and its toolbar button.

FocusOverlay receives FocusResult objects produced by MachineVisionManager
and composites the pre-rendered heatmap into the camera preview.  All heavy
OpenCV work happens off the GUI thread; this file only handles display.

Wiring
------
CameraPreview calls:

  1. focus_overlay.set_vision_manager(ctx.machine_vision) once after
     construction, giving the overlay a direct reference to the manager.
  2. update_full() on every frame (via OverlayLabel.notify_full()), where
     the overlay submits analysis jobs to the manager itself.
  3. mv.focus_result_ready.connect(focus_overlay.receive_result) so results
     are delivered back on the GUI thread via Qt's queued connection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from PySide6.QtCore import Qt, QRect, Slot, Signal
from PySide6.QtGui import QPainter, QImage, QPixmap
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from common.logger import info
from .overlay_base import Overlay

if TYPE_CHECKING:
    from machine_vision.machine_vision_manager import FocusResult, MachineVisionManager


class FocusOverlay(Overlay):
    """
    Overlay that displays a Laplacian focus heatmap.

    The heatmap is rendered by MachineVisionWorker off-thread.  This class
    only stores the latest result pixmap and paints it when Qt asks.

    Call set_vision_manager() once after construction to connect the overlay
    to the shared MachineVisionManager.
    """

    def __init__(self) -> None:
        super().__init__()
        self._result_pixmap: QPixmap | None = None
        self._scores_text: str = ""
        self._vision_manager: MachineVisionManager | None = None

    def set_vision_manager(self, manager: MachineVisionManager) -> None:
        """Connect this overlay to the shared MachineVisionManager."""
        self._vision_manager = manager
        manager.focus_result_ready.connect(self.receive_result)

    # ------------------------------------------------------------------
    # Called by OverlayLabel on every rendered frame (GUI thread)
    # ------------------------------------------------------------------

    def update_full(self, frame: np.ndarray) -> None:
        """
        Called by OverlayLabel.notify_full() with each full-resolution frame.

        Submits an analysis job to the vision manager when the overlay is
        enabled.  The manager silently drops the request if it is still
        processing the previous frame.
        """
        if self._vision_manager is None:
            return
        h, w = frame.shape[:2]
        self._vision_manager.request_focus_analysis(frame, w, h)

    # ------------------------------------------------------------------
    # Receive results from MachineVisionManager (GUI thread, queued signal)
    # ------------------------------------------------------------------

    @Slot(object)
    def receive_result(self, result: FocusResult) -> None:
        """
        Store and display the latest focus analysis result.

        This slot is connected to MachineVisionManager.focus_result_ready.
        It is always called on the GUI thread (Qt queued connection across
        the worker thread boundary).
        """
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
        if self._vision_manager is not None:
            active = self._vision_manager.settings.focus.active
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

        # Scale the heatmap to exactly fill the image rect.
        scaled = self._result_pixmap.scaled(
            rect.width(),
            rect.height(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        painter.setOpacity(1.0)
        painter.drawPixmap(rect.topLeft(), scaled)

        # Score legend — bottom-left corner of the image rect.
        if self._scores_text:
            painter.save()
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)

            from PySide6.QtGui import QColor, QPen
            # Shadow pass for readability.
            painter.setPen(QPen(QColor(0, 0, 0, 200)))
            painter.drawText(rect.left() + 6, rect.bottom() - 7, self._scores_text)
            # White foreground pass.
            painter.setPen(QPen(QColor(255, 255, 255, 220)))
            painter.drawText(rect.left() + 5, rect.bottom() - 8, self._scores_text)

            painter.restore()


class FocusButton(QPushButton):
    """
    Checkable overlay button for the focus overlay.

    The icon is composed of three transparent child labels (top corners,
    bottom corners, and a centre crosshair) layered over the button so that
    the checked/unchecked background is still rendered by the button itself.
    """

    toggled_focus = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("FocusButton")
        self.setCheckable(True)
        self.setFixedSize(30, 30)
        self.setToolTip("Toggle Focus Overlay")
        self.clicked.connect(self._on_clicked)
        self._build_icon_labels()

    def _build_icon_labels(self) -> None:
        top_corners = QLabel("⌜⌝", self)
        top_corners.setObjectName("FocusOverlayLabel")
        top_corners.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_corners.setGeometry(0, -2, 30, 30)
        top_corners.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        bottom_corners = QLabel("⌞⌟", self)
        bottom_corners.setObjectName("FocusOverlayLabel")
        bottom_corners.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom_corners.setGeometry(0, 2, 30, 30)
        bottom_corners.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        center = QLabel("⌖", self)
        center.setObjectName("FocusOverlayLabel")
        center.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.setGeometry(0, 0, 30, 30)
        center.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def _on_clicked(self, checked: bool) -> None:
        info(f"Focus Overlay Toggled {'on' if checked else 'off'}")
        self.toggled_focus.emit(checked)