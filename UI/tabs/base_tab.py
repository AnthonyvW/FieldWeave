from __future__ import annotations

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QHBoxLayout, QWidget

from UI.style import OUTER_MARGIN
from common.app_context import get_app_context


class CameraWithSidebarPage(QWidget):
    def __init__(self, sidebar_widget: QWidget, parent: QWidget | None = None):
        super().__init__(parent)

        self._root_layout = QHBoxLayout(self)
        self._root_layout.setContentsMargins(OUTER_MARGIN, OUTER_MARGIN, OUTER_MARGIN, OUTER_MARGIN)
        self._root_layout.setSpacing(OUTER_MARGIN)
        # Index 0 is reserved for the camera preview; insert a placeholder so
        # the sidebar stays at index 1 regardless of whether the preview is
        # currently parented here.
        self._preview_placeholder = QWidget()
        self._preview_placeholder.hide()
        self._root_layout.addWidget(self._preview_placeholder, 1)
        self._root_layout.addWidget(sidebar_widget, 0)

    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        preview = get_app_context().camera_preview
        if preview is not None:
            self._root_layout.replaceWidget(self._preview_placeholder, preview)
            self._preview_placeholder.hide()
            preview.show()

    def hideEvent(self, event: QEvent) -> None:
        super().hideEvent(event)
        preview = get_app_context().camera_preview
        if preview is not None and preview.parent() is self:
            self._root_layout.replaceWidget(preview, self._preview_placeholder)
            preview.hide()

    def set_sidebar_flush_right(self, flush: bool) -> None:
        l, t, r, b = self._root_layout.getContentsMargins()
        new_r = 0 if flush else OUTER_MARGIN
        if r != new_r:
            self._root_layout.setContentsMargins(l, t, new_r, b)