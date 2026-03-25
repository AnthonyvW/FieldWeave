from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QFrame,
)
from UI.style import RIGHT_SIDEBAR_WIDTH, OUTER_MARGIN
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

        scroll = self._wrap_scroll(content, sidebar_container)
        sidebar_layout.addWidget(scroll, 1)
        return sidebar_container

    def _wrap_scroll(self, widget: QWidget, sidebar: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(widget)

        scrollbar_width = scroll.style().pixelMetric(
            scroll.style().PixelMetric.PM_ScrollBarExtent
        )

        def _on_range_changed(min_val: int, max_val: int) -> None:
            needed = max_val > min_val
            sidebar.setFixedWidth(RIGHT_SIDEBAR_WIDTH + (scrollbar_width if needed else 0))
            self.set_sidebar_flush_right(needed)

        scroll.verticalScrollBar().rangeChanged.connect(_on_range_changed)
        return scroll