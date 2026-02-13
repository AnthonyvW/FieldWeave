from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from ..settings.settings_main import SettingsButton

class CollapsibleSection(QFrame):
    """
    Collapsible boxed section:
    - full-width header strip
    - collapses entire widget height when collapsed
    - callback on collapse/expand (useful to adjust parent stretch)
    """

    def __init__(
        self,
        title: str,
        *,
        on_settings=None,
        start_collapsed: bool = False,
        on_collapsed_changed=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("CollapsibleSection")
        self._on_collapsed_changed = on_collapsed_changed

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = QFrame()
        self.header.setObjectName("SectionHeader")
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.setProperty("collapsed", False)

        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(10, 7, 8, 7)
        header_layout.setSpacing(8)

        self.caret = QLabel("▾")
        self.caret.setFixedWidth(16)
        self.caret.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("SectionHeaderTitle")

        header_layout.addWidget(self.caret)
        header_layout.addWidget(self.title_lbl)
        header_layout.addStretch(1)

        if on_settings is not None:
            gear = SettingsButton("Section settings")
            gear.clicked.connect(on_settings)
            header_layout.addWidget(gear)

        root.addWidget(self.header)

        self.content = QWidget()
        self.content.setObjectName("SectionContent")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(10, 10, 10, 12)
        self.content_layout.setSpacing(10)
        root.addWidget(self.content)

        # natural height unless parent gives stretch
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self._collapsed = False
        self.header.mousePressEvent = self._on_header_click  # type: ignore
        self.set_collapsed(start_collapsed)

    def _on_header_click(self, event) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        if self._collapsed == collapsed:
            return
        self._collapsed = collapsed

        self.content.setVisible(not collapsed)
        self.caret.setText("▸" if collapsed else "▾")

        # inform stylesheet for corner rounding on collapse
        self.header.setProperty("collapsed", collapsed)
        self.header.style().unpolish(self.header)
        self.header.style().polish(self.header)

        # collapse entire widget height
        header_h = self.header.sizeHint().height()
        if collapsed:
            self.setMaximumHeight(header_h + 2)
            self.setMinimumHeight(header_h + 2)
        else:
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)

        if self._on_collapsed_changed is not None:
            self._on_collapsed_changed(collapsed)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def layout_for_content(self) -> QVBoxLayout:
        return self.content_layout
