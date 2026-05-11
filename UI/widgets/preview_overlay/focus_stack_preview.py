from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QPainter, QImage, QPixmap

from common.app_context import get_app_context
from UI.widgets.preview_overlay.overlay_base import Overlay


class FocusStackPreviewOverlay(Overlay):
    """
    Displays partial focus stack preview frames on top of the camera feed.

    Activated by calling :meth:`show_frame` with each new preview array as it
    arrives from :class:`~post_processing.routines.focus_stack_routine.StreamingFocusStackRoutine`.
    Calling :meth:`clear` removes the overlay and resumes the normal camera display.

    The frame is scaled to fill the image rect (preserving aspect ratio) and
    drawn over the camera feed, so no changes to the camera preview pipeline
    are needed.

    Position tracking
    -----------------
    When :meth:`show_frame` is called with ``track_position=True`` the current
    stage position is recorded. On each subsequent camera frame :meth:`update_full`
    checks whether the stage has moved; if it has the overlay clears itself
    automatically, matching the behaviour of the original ``show_static_image``
    mechanism. Position tracking is only active for the final stacked result —
    during live stacking the overlay stays up regardless of stage movement so
    the partial preview is not interrupted.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self._pinned_position: tuple[int, int, int] | None = None

    def show_frame(self, frame: np.ndarray, track_position: bool = False) -> None:
        """Update the displayed frame.

        *frame* must be a uint8 RGB ndarray, shape (H, W, 3).  Safe to call
        from any thread provided the caller marshals to the main thread first
        (e.g. via ``QMetaObject.invokeMethod``).

        Parameters
        ----------
        track_position:
            When True, record the current stage position and automatically
            clear the overlay if the stage moves.  Pass True for the final
            stacked result; leave False during live incremental previews.
        """
        h, w = frame.shape[:2]
        q_image = QImage(frame.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(q_image)

        if track_position:
            ctx = get_app_context()
            if ctx.motion is not None:
                pos = ctx.motion.get_position()
                self._pinned_position = (pos.x, pos.y, pos.z)
            else:
                self._pinned_position = None
        else:
            self._pinned_position = None

    def clear(self) -> None:
        """Remove the overlay frame and stop drawing over the camera feed."""
        self._pixmap = None
        self._pinned_position = None
        self.set_enabled(False)

    def update_full(self, frame: np.ndarray) -> None:
        if self._pinned_position is None or not self._enabled:
            return
        ctx = get_app_context()
        if ctx.motion is None:
            return
        pos = ctx.motion.get_position()
        if (pos.x, pos.y, pos.z) != self._pinned_position:
            self.clear()

    def draw(self, painter: QPainter, rect: QRect) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            rect.width(),
            rect.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = rect.x() + (rect.width() - scaled.width()) // 2
        y = rect.y() + (rect.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)