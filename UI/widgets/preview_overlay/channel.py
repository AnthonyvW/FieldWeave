from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QRect, Signal, Slot
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from common.logger import info
from .overlay_base import Overlay


class ChannelOverlay(Overlay):
    """
    Applies RGB channel masking and optional grayscale conversion to each frame.

    Unlike paint overlays this one operates on the raw pixel array in ``update``
    and stores the result for ``draw`` to display. Because it modifies the image
    data rather than painting on top, ``draw`` is a no-op — the filtered image
    is returned via ``filtered_image`` and applied before the pixmap is set.
    """

    def __init__(self) -> None:
        super().__init__()
        self.show_red = True
        self.show_green = True
        self.show_blue = True
        self.show_grayscale = False
        self._filtered: QImage | None = None

    @property
    def needs_filter(self) -> bool:
        return (
            self.show_grayscale
            or not self.show_red
            or not self.show_green
            or not self.show_blue
        )

    def apply(self, image: QImage) -> QImage:
        """Return a new QImage with the current channel settings applied."""
        width = image.width()
        height = image.height()

        ptr = image.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
            (height, image.bytesPerLine())
        )[:, : width * 3].reshape((height, width, 3)).copy()

        if not self.show_red:
            arr[:, :, 0] = 0
        if not self.show_green:
            arr[:, :, 1] = 0
        if not self.show_blue:
            arr[:, :, 2] = 0

        if self.show_grayscale:
            r_w = 0.299 if self.show_red else 0.0
            g_w = 0.587 if self.show_green else 0.0
            b_w = 0.114 if self.show_blue else 0.0
            total = r_w + g_w + b_w
            if total > 0:
                r_w, g_w, b_w = r_w / total, g_w / total, b_w / total
            gray = (
                r_w * arr[:, :, 0].astype(np.float32)
                + g_w * arr[:, :, 1].astype(np.float32)
                + b_w * arr[:, :, 2].astype(np.float32)
            ).astype(np.uint8)
            arr[:, :, 0] = gray
            arr[:, :, 1] = gray
            arr[:, :, 2] = gray

        filtered = QImage(arr.tobytes(), width, height, width * 3, QImage.Format.Format_RGB888)
        return filtered.copy()

    def draw(self, painter: QPainter, rect: QRect) -> None:
        """No-op — channel filtering is applied to the image data, not painted on top."""


class ChannelButton(QPushButton):
    """
    Checkable overlay button that opens a flyout menu for RGB / grayscale
    channel filtering.

    The flyout (``self.menu``) is a ``QFrame`` that must be added to the
    *same parent* as this button so it can float freely over the preview.
    Call ``place_menu()`` after the button has been positioned so the menu
    appears to the right.

    Signals
    -------
    channel_changed(show_red, show_green, show_blue, show_grayscale)
        Emitted whenever any checkbox in the flyout is toggled.
    """

    channel_changed = Signal(bool, bool, bool, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ChannelButton")
        self.setCheckable(True)
        self.setFixedSize(30, 30)
        self.setToolTip("Channel Filters")
        self.setProperty("ChannelFiltered", False)
        self.clicked.connect(self._on_clicked)

        self._build_icon_labels()
        self.menu = self._build_menu(parent)

    def place_menu(self) -> None:
        """Position the flyout to the right of this button."""
        btn_pos = self.pos()
        self.menu.move(btn_pos.x() + 35, btn_pos.y())

    def _build_icon_labels(self) -> None:
        for x_offset, y_offset in [(-5, -4), (5, -4), (0, 4)]:
            label = QLabel("○", self)
            label.setObjectName("VennOverlayLabel")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setGeometry(x_offset, y_offset, 30, 30)
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def _build_menu(self, parent: QWidget | None) -> QFrame:
        menu = QFrame(parent)
        menu.setObjectName("ChannelMenu")
        menu.setFixedWidth(110)
        menu.setAutoFillBackground(True)
        menu.setFrameShape(QFrame.Shape.StyledPanel)
        menu.setFrameShadow(QFrame.Shadow.Raised)
        menu.hide()

        layout = QVBoxLayout(menu)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._cb_red = QCheckBox("Red", menu)
        self._cb_red.setObjectName("ChannelCheckRed")
        self._cb_red.setChecked(True)
        self._cb_red.toggled.connect(self._on_channel_changed)

        self._cb_green = QCheckBox("Green", menu)
        self._cb_green.setObjectName("ChannelCheckGreen")
        self._cb_green.setChecked(True)
        self._cb_green.toggled.connect(self._on_channel_changed)

        self._cb_blue = QCheckBox("Blue", menu)
        self._cb_blue.setObjectName("ChannelCheckBlue")
        self._cb_blue.setChecked(True)
        self._cb_blue.toggled.connect(self._on_channel_changed)

        self._cb_gray = QCheckBox("Grayscale", menu)
        self._cb_gray.setObjectName("ChannelCheckGray")
        self._cb_gray.setChecked(False)
        self._cb_gray.toggled.connect(self._on_channel_changed)

        layout.addWidget(self._cb_red)
        layout.addWidget(self._cb_green)
        layout.addWidget(self._cb_blue)
        layout.addWidget(self._cb_gray)
        menu.adjustSize()
        return menu

    def _filters_are_default(self) -> bool:
        return (
            self._cb_red.isChecked()
            and self._cb_green.isChecked()
            and self._cb_blue.isChecked()
            and not self._cb_gray.isChecked()
        )

    def _update_highlight(self) -> None:
        active = not self._filters_are_default()
        self.setProperty("ChannelFiltered", active)
        self.style().unpolish(self)
        self.style().polish(self)

    @Slot(bool)
    def _on_clicked(self, checked: bool) -> None:
        if checked:
            self.place_menu()
            self.menu.show()
            self.menu.raise_()
        else:
            self.menu.hide()
            self._update_highlight()

    @Slot()
    def _on_channel_changed(self) -> None:
        show_red = self._cb_red.isChecked()
        show_green = self._cb_green.isChecked()
        show_blue = self._cb_blue.isChecked()
        show_grayscale = self._cb_gray.isChecked()
        info(
            f"Preview: Channels R={show_red} G={show_green} "
            f"B={show_blue} Gray={show_grayscale}"
        )
        if not self.menu.isVisible():
            self._update_highlight()
        self.channel_changed.emit(show_red, show_green, show_blue, show_grayscale)