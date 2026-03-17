from __future__ import annotations

from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from common.logger import info
from .overlay_base import Overlay


class FocusOverlay(Overlay):
    """Focus overlay — rendering to be implemented."""

    def draw(self, painter: QPainter, rect: QRect) -> None:
        pass


class FocusButton(QPushButton):
    """
    Checkable overlay button for the focus overlay.

    The icon is composed of three transparent child labels (top corners,
    bottom corners, and a centre crosshair) layered over the button so that
    the checked/unchecked background is still rendered by the button itself.
    """

    toggled_focus = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("FocusButton")
        self.setCheckable(True)
        self.setFixedSize(30, 30)
        self.setToolTip("Toggle Focus Overlay")
        self.clicked.connect(self._on_clicked)
        self._build_icon_labels()

    def _build_icon_labels(self) -> None:
        top_corners = QLabel("⌜⌝", self)
        top_corners.setObjectName("FocusOverlayLabel")
        top_corners.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_corners.setGeometry(0, -2, 30, 30)
        top_corners.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        bottom_corners = QLabel("⌞⌟", self)
        bottom_corners.setObjectName("FocusOverlayLabel")
        bottom_corners.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom_corners.setGeometry(0, 2, 30, 30)
        bottom_corners.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        center = QLabel("⌖", self)
        center.setObjectName("FocusOverlayLabel")
        center.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.setGeometry(0, 0, 30, 30)
        center.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def _on_clicked(self, checked: bool) -> None:
        info(f"Focus Overlay Toggled {'on' if checked else 'off'}")
        self.toggled_focus.emit(checked)