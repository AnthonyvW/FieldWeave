from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap

from UI.widgets.preview_overlay.large_image_source import LargeImageSource
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

    This overlay only ever draws the fit-to-widget whole image, so it
    only ever needs ``source.preview`` — the small resident thumbnail
    LargeImageSource always keeps around — never a tile. ``source`` is
    exposed so ``ZoomPreviewOverlay`` can crop/zoom into the full-
    resolution tiles in place of the live camera frame — see
    ``ZoomPreviewOverlay.set_loaded_image_overlay``.
    """

    _PLACEHOLDER_COLOR = QColor(200, 200, 200)

    def __init__(self) -> None:
        super().__init__()
        self._source: LargeImageSource | None = None
        # Cache of the preview pre-scaled to the last-drawn rect, so a
        # full SmoothTransformation scale doesn't run on every paint.
        self._scaled_cache: QPixmap | None = None
        self._scaled_cache_size: tuple[int, int] | None = None

    def set_source(self, source: LargeImageSource | None) -> None:
        if self._source is not None:
            self._source.close()
        self._source = source
        self._scaled_cache = None
        self._scaled_cache_size = None

    @property
    def source(self) -> LargeImageSource | None:
        return self._source

    @property
    def has_image(self) -> bool:
        return self._source is not None and self._source.preview is not None

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

        size = (rect.width(), rect.height())
        if self._scaled_cache is None or self._scaled_cache_size != size:
            self._scaled_cache = self._preview_pixmap().scaled(
                rect.width(),
                rect.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._scaled_cache_size = size

        scaled = self._scaled_cache
        x = rect.x() + (rect.width() - scaled.width()) // 2
        y = rect.y() + (rect.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

    def _preview_pixmap(self) -> QPixmap:
        array = np.ascontiguousarray(self._source.preview)
        h, w = array.shape[:2]
        image = QImage(array.data, w, h, w * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image)