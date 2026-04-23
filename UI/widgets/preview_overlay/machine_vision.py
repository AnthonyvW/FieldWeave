from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MachineVisionButton(QPushButton):
    """
    Checkable overlay button that opens a flyout menu for machine vision mode
    selection. Exactly one mode may be active at a time; the default is none.

    The flyout (``self.menu``) must be added to the *same parent* as this
    button so it can float freely over the preview. Call ``place_menu()``
    after the button has been positioned.

    The button uses the concentric-circle target symbol to indicate its
    machine-vision purpose.

    Signals
    -------
    vision_mode_changed(focus, red_mark, scale)
        Emitted whenever the active mode changes. Exactly one value will be
        True, or all will be False when no mode is selected.

        Order: focus, red_mark, scale, background.
    """

    vision_mode_changed = Signal(bool, bool, bool, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("FocusButton")
        self.setCheckable(True)
        self.setFixedSize(30, 30)
        self.setToolTip("Machine Vision")
        self.setProperty("MachineVisionActive", False)
        self.clicked.connect(self._on_clicked)

        self._build_icon_labels()
        self.menu = self._build_menu(parent)

    def place_menu(self) -> None:
        """Position the flyout to the right of this button."""
        btn_pos = self.pos()
        self.menu.move(btn_pos.x() + 35, btn_pos.y())

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

    def _build_menu(self, parent: QWidget | None) -> QFrame:
        menu = QFrame(parent)
        menu.setObjectName("MachineVisionMenu")
        menu.setFixedWidth(160)
        menu.setAutoFillBackground(True)
        menu.setFrameShape(QFrame.Shape.StyledPanel)
        menu.setFrameShadow(QFrame.Shadow.Raised)
        menu.hide()

        layout = QVBoxLayout(menu)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._cb_none = QCheckBox("None", menu)
        self._cb_none.setObjectName("VisionCheckNone")
        self._cb_none.setChecked(True)

        self._cb_focus = QCheckBox("Focus Detection", menu)
        self._cb_focus.setObjectName("VisionCheckFocus")
        self._cb_focus.setChecked(False)

        self._cb_red_mark = QCheckBox("Red Mark Detection", menu)
        self._cb_red_mark.setObjectName("VisionCheckRedMark")
        self._cb_red_mark.setChecked(False)

        self._cb_scale = QCheckBox("Scale Detection", menu)
        self._cb_scale.setObjectName("VisionCheckScale")
        self._cb_scale.setChecked(False)

        self._cb_background = QCheckBox("Background Detection", menu)
        self._cb_background.setObjectName("VisionCheckBackground")
        self._cb_background.setChecked(False)

        self._button_group = QButtonGroup(menu)
        self._button_group.setExclusive(True)
        self._button_group.addButton(self._cb_none)
        self._button_group.addButton(self._cb_focus)
        self._button_group.addButton(self._cb_red_mark)
        self._button_group.addButton(self._cb_scale)
        self._button_group.addButton(self._cb_background)

        self._button_group.buttonToggled.connect(self._on_mode_changed)

        layout.addWidget(self._cb_none)
        layout.addWidget(self._cb_focus)
        layout.addWidget(self._cb_red_mark)
        layout.addWidget(self._cb_scale)
        layout.addWidget(self._cb_background)
        menu.adjustSize()
        return menu

    def _mode_is_default(self) -> bool:
        return self._cb_none.isChecked()

    def _update_highlight(self) -> None:
        active = not self._mode_is_default()
        self.setProperty("MachineVisionActive", active)
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
    def _on_mode_changed(self) -> None:
        focus = self._cb_focus.isChecked()
        red_mark = self._cb_red_mark.isChecked()
        scale = self._cb_scale.isChecked()
        background = self._cb_background.isChecked()
        if not self.menu.isVisible():
            self._update_highlight()
        self.vision_mode_changed.emit(focus, red_mark, scale, background)
