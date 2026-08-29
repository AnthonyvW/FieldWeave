from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap

from UI.widgets.preview_overlay.overlay_base import Overlay


class LoadedImageOverlay(Overlay):
    """
    Displays a static loaded image in place of the live camera feed.

    Added first among the video label's overlays (see CameraPreview), so
    every other overlay — crosshair, grid, a future measurement-marker
    overlay — layers on top of it exactly as it would over the live feed.
    Drawing an opaque background before the image means it fully occludes
    the live feed underneath even while inactive frames keep arriving.

    Enabled state is intentionally not self-managed: it's driven by
    CaptureControlWidget via CameraPreview.overlays, which only turns this
    on while the measurement tab is the one currently showing the shared
    preview. Every other tab must never see it.

    ``full_array`` exposes the loaded image as an RGB array so
    ``ZoomPreviewOverlay`` can crop/zoom into it in place of the live
    camera frame — see ``ZoomPreviewOverlay.set_loaded_image_overlay``.
    """

    _PLACEHOLDER_COLOR = QColor(200, 200, 200)

    def __init__(self) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self._full_array: np.ndarray | None = None

    def set_image(self, pixmap: QPixmap | None) -> None:
        self._pixmap = pixmap
        self._full_array = self._to_array(pixmap)

    @property
    def has_image(self) -> bool:
        return self._pixmap is not None and not self._pixmap.isNull()

    @property
    def full_array(self) -> np.ndarray | None:
        """The loaded image as a full-resolution RGB array (H×W×3, uint8), or None if no image is loaded."""
        return self._full_array

    @staticmethod
    def _to_array(pixmap: QPixmap | None) -> np.ndarray | None:
        if pixmap is None or pixmap.isNull():
            return None

        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        ptr = image.bits()
        array = (
            np.frombuffer(ptr, dtype=np.uint8)
            .reshape((image.height(), image.bytesPerLine()))
            [:, : image.width() * 3]
            .reshape((image.height(), image.width(), 3))
            .copy()
        )
        del ptr
        return array

    def draw(self, painter: QPainter, rect: QRect) -> None:
        painter.fillRect(rect, QColor(0, 0, 0))

        if not self.has_image:
            painter.setPen(self._PLACEHOLDER_COLOR)
            font = painter.font()
            font.setPointSize(12)
            font.setItalic(True)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No image loaded")
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