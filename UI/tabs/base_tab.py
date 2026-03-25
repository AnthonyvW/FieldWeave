from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QWidget

from ..style import OUTER_MARGIN

class CameraWithSidebarPage(QWidget):
    def __init__(self, camera_widget: QWidget, sidebar_widget: QWidget, parent: QWidget | None = None):
        super().__init__(parent)

        self._root_layout = QHBoxLayout(self)
        self._root_layout.setContentsMargins(OUTER_MARGIN, OUTER_MARGIN, OUTER_MARGIN, OUTER_MARGIN)
        self._root_layout.setSpacing(OUTER_MARGIN)
        self._root_layout.addWidget(camera_widget, 1)
        self._root_layout.addWidget(sidebar_widget, 0)

    def set_sidebar_flush_right(self, flush: bool) -> None:
        l, t, r, b = self._root_layout.getContentsMargins()
        new_r = 0 if flush else OUTER_MARGIN
        if r != new_r:
            self._root_layout.setContentsMargins(l, t, new_r, b)