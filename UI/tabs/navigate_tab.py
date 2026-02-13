from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QFrame
)
from UI.style import RIGHT_SIDEBAR_WIDTH
from UI.tabs.base_tab import CameraWithSidebarPage

from UI.widgets.camera_preview import CameraPreview
from UI.widgets.collapsible_section import CollapsibleSection
from UI.widgets.camera_controls_widget import CameraControlsWidget

from app_context import open_settings

class NavigateTab(CameraWithSidebarPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(CameraPreview(), self._make_sidebar(), parent)

    def _make_sidebar(self) -> QWidget:
        sidebar_container = QWidget()
        sidebar_container.setFixedWidth(RIGHT_SIDEBAR_WIDTH)

        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(10)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        # Start Widgets

        camera_controls = CollapsibleSection("Camera Controls", on_settings=lambda: open_settings("Camera"))
        camera_controls.layout_for_content().addWidget(CameraControlsWidget())
        content_layout.addWidget(camera_controls)

        # End Widgets

        content_layout.addStretch(1)
        sidebar_layout.addWidget(self._wrap_scroll(content), 1)
        return sidebar_container
    
    def _wrap_scroll(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(widget)
        return scroll