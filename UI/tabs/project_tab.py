from __future__ import annotations

from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QFrame
)
from UI.style import RIGHT_SIDEBAR_WIDTH
from UI.tabs.base_tab import CameraWithSidebarPage

from UI.widgets.camera_preview import CameraPreview
from UI.widgets.collapsible_section import CollapsibleSection
from UI.widgets.automation_control_widget import AutomationWidget
from UI.widgets.navigation_widget import NavigationWidget

from common.app_context import open_settings

class ProjectTab(CameraWithSidebarPage):
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

        navigation = CollapsibleSection("Navigation", on_settings=lambda: open_settings("Navigation"))
        navigation.layout_for_content().addWidget(NavigationWidget())
        content_layout.addWidget(navigation)

        automation = CollapsibleSection("Automation", on_settings=lambda: open_settings("Automation"))
        automation.layout_for_content().addWidget(AutomationWidget())
        content_layout.addWidget(automation)

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