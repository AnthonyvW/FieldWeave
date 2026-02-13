from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QWidget

from ..style import OUTER_MARGIN

class CameraWithSidebarPage(QWidget):
    def __init__(self, camera_widget: QWidget, sidebar_widget: QWidget, parent: QWidget | None = None):
        super().__init__(parent)

        root = QHBoxLayout(self)
        root.setContentsMargins(OUTER_MARGIN, OUTER_MARGIN, OUTER_MARGIN, OUTER_MARGIN)
        root.setSpacing(OUTER_MARGIN)
        root.addWidget(camera_widget, 1)
        root.addWidget(sidebar_widget, 0)