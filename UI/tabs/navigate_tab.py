from __future__ import annotations

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QFrame,
)
from UI.style import RIGHT_SIDEBAR_WIDTH
from UI.tabs.base_tab import CameraWithSidebarPage

from UI.widgets.collapsible_section import CollapsibleSection
from UI.widgets.camera_controls_widget import CameraControlsWidget
from UI.widgets.navigation_widget import NavigationWidget
from UI.widgets.preview_overlay.interaction_mode import NAVIGATE_MODE, ModeToken

from common.app_context import get_app_context, open_settings


class NavigateTab(CameraWithSidebarPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        self._mode_token: ModeToken | None = None
        super().__init__(self._make_sidebar(), parent)

    # Pushes the default interaction mode explicitly rather than relying
    # on the preview's baseline (no mode pushed) happening to match —
    # see UI.widgets.preview_overlay.interaction_mode.NAVIGATE_MODE.
    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        preview = get_app_context().camera_preview
        if preview is not None:
            self._mode_token = preview.modes.push(NAVIGATE_MODE)

    def hideEvent(self, event: QEvent) -> None:
        super().hideEvent(event)
        if self._mode_token is not None:
            self._mode_token.pop()
            self._mode_token = None

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

        navigation = CollapsibleSection("Navigation", on_settings=lambda: open_settings("Navigation"))
        navigation.layout_for_content().addWidget(NavigationWidget())
        content_layout.addWidget(navigation)

        camera_controls = CollapsibleSection("Camera Controls", on_settings=lambda: open_settings("Camera"))
        camera_controls.layout_for_content().addWidget(CameraControlsWidget())
        content_layout.addWidget(camera_controls)

        content_layout.addStretch(1)
        sidebar_layout.addWidget(self._wrap_scroll(content), 1)
        return sidebar_container

    def _wrap_scroll(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(widget)
        return scroll
