from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from common.logger import info


class ImageAppearance:
    """
    Display-only image adjustments for the preview.

    Applied to the already display-resolution pixels right before they are
    handed to Qt, never to the full frame that overlays, exports and the
    camera see. Anything that draws image pixels into the viewport calls
    ``apply`` / ``apply_pixmap`` and keys its own cache on ``version``.

    Every setting is an integer percentage; zero (or the field's default)
    is a no-op, so ``active`` stays False and the whole path is skipped
    until the operator moves a slider.
    """

    FIELDS: tuple[tuple[str, str, int, int], ...] = (
        ("brightness", "Brightness", -100, 100),
        ("contrast", "Contrast", -100, 100),
        ("saturation", "Saturation", -100, 100),
        ("sharpness", "Sharpness", 0, 100),
        ("emboss", "Emboss", 0, 100),
        ("edge_detect", "Edge Detect", 0, 100),
    )

    _SHARPEN_SIGMA: float = 1.5
    _EMBOSS_KERNEL = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    _EDGE_GAIN: float = 3.0

    def __init__(self) -> None:
        self.brightness = 0
        self.contrast = 0
        self.saturation = 0
        self.sharpness = 0
        self.emboss = 0
        self.edge_detect = 0
        self.version = 0
        self._tone_lut: np.ndarray | None = None

    @property
    def active(self) -> bool:
        return any(getattr(self, name) != 0 for name, *_ in self.FIELDS)

    def set_value(self, name: str, value: int) -> None:
        if getattr(self, name) == value:
            return
        setattr(self, name, value)
        self._tone_lut = None
        self.version += 1

    def reset(self) -> None:
        for name, *_ in self.FIELDS:
            setattr(self, name, 0)
        self._tone_lut = None
        self.version += 1

    def apply(self, arr: np.ndarray) -> np.ndarray:
        """Return *arr* (H x W x 3 uint8 RGB) with the current settings applied."""
        out = arr

        if self.saturation:
            gray = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            factor = 1.0 + self.saturation / 100.0
            out = cv2.addWeighted(out, factor, gray, 1.0 - factor, 0)

        if self.brightness or self.contrast:
            out = cv2.LUT(out, self._tone_table())

        if self.sharpness:
            amount = self.sharpness / 100.0 * 2.0
            blurred = cv2.GaussianBlur(out, (0, 0), self._SHARPEN_SIGMA)
            out = cv2.addWeighted(out, 1.0 + amount, blurred, -amount, 0)

        if self.emboss:
            gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
            relief = cv2.filter2D(gray, -1, self._EMBOSS_KERNEL, delta=128)
            out = self._blend(out, cv2.cvtColor(relief, cv2.COLOR_GRAY2RGB), self.emboss)

        if self.edge_detect:
            gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
            grad_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3))
            grad_y = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3))
            edges = cv2.convertScaleAbs(cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0), alpha=self._EDGE_GAIN)
            out = self._blend(out, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), self.edge_detect)

        return np.ascontiguousarray(out)

    def apply_pixmap(self, pixmap: QPixmap) -> QPixmap:
        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        w, h = image.width(), image.height()
        arr = (
            np.frombuffer(image.bits(), dtype=np.uint8)
            .reshape((h, image.bytesPerLine()))[:, : w * 3]
            .reshape((h, w, 3))
        )
        out = self.apply(arr)
        return QPixmap.fromImage(QImage(out.data, w, h, w * 3, QImage.Format.Format_RGB888))

    def _tone_table(self) -> np.ndarray:
        if self._tone_lut is None:
            contrast = self.contrast / 100.0
            gain = 1.0 + contrast if contrast < 0 else 1.0 + contrast * 2.0
            levels = np.arange(256, dtype=np.float32)
            table = (levels - 128.0) * gain + 128.0 + self.brightness * 2.55
            self._tone_lut = np.clip(table, 0, 255).astype(np.uint8)
        return self._tone_lut

    @staticmethod
    def _blend(base: np.ndarray, effect: np.ndarray, percent: int) -> np.ndarray:
        weight = percent / 100.0
        return cv2.addWeighted(base, 1.0 - weight, effect, weight, 0)


class AppearanceButton(QPushButton):
    """
    Checkable overlay button that opens a flyout of sliders for the
    display-only image adjustments in ``ImageAppearance``.

    The flyout (``self.menu``) must share this button's parent so it can
    float over the preview; ``place_menu()`` puts it to the right.

    Signals
    -------
    appearance_changed(name, value)
        A slider moved. ``name`` is the ``ImageAppearance`` field.
    reset_requested()
        The reset button was pressed; sliders have already returned to zero.
    """

    appearance_changed = Signal(str, int)
    reset_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("◐", parent)
        self.setObjectName("AppearanceButton")
        self.setCheckable(True)
        self.setFixedSize(30, 30)
        self.setToolTip("Image Appearance")
        self.clicked.connect(self._on_clicked)

        self._sliders: dict[str, QSlider] = {}
        self._value_labels: dict[str, QLabel] = {}
        self.menu = self._build_menu(parent)

    def place_menu(self) -> None:
        btn_pos = self.pos()
        self.menu.move(btn_pos.x() + 35, btn_pos.y())

    def _build_menu(self, parent: QWidget | None) -> QFrame:
        menu = QFrame(parent)
        menu.setObjectName("AppearanceMenu")
        menu.setFixedWidth(230)
        menu.setAutoFillBackground(True)
        menu.setFrameShape(QFrame.Shape.StyledPanel)
        menu.setFrameShadow(QFrame.Shadow.Raised)
        menu.hide()

        layout = QVBoxLayout(menu)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 1)

        for row, (name, title, low, high) in enumerate(ImageAppearance.FIELDS):
            caption = QLabel(title, menu)
            value_label = QLabel("0", menu)
            value_label.setObjectName("AppearanceValueLabel")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            slider = QSlider(Qt.Orientation.Horizontal, menu)
            slider.setRange(low, high)
            slider.setValue(0)
            slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            slider.valueChanged.connect(lambda value, n=name: self._on_slider_changed(n, value))
            # Double-click snaps back to neutral — quicker than dragging to exactly 0.
            slider.mouseDoubleClickEvent = lambda _event, s=slider: s.setValue(0)

            self._sliders[name] = slider
            self._value_labels[name] = value_label

            grid.addWidget(caption, row * 2, 0)
            grid.addWidget(value_label, row * 2, 1)
            grid.addWidget(slider, row * 2 + 1, 0, 1, 2)

        layout.addLayout(grid)

        reset = QPushButton("Reset", menu)
        reset.setObjectName("AppearanceResetButton")
        reset.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        reset.clicked.connect(self._on_reset_clicked)
        layout.addWidget(reset)

        menu.adjustSize()
        return menu

    def _on_clicked(self, checked: bool) -> None:
        if checked:
            self.place_menu()
            self.menu.show()
            self.menu.raise_()
        else:
            self.menu.hide()

    def _on_slider_changed(self, name: str, value: int) -> None:
        self._value_labels[name].setText(str(value))
        self.appearance_changed.emit(name, value)

    def _on_reset_clicked(self) -> None:
        info("Preview: Image appearance reset")
        for slider in self._sliders.values():
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
        for label in self._value_labels.values():
            label.setText("0")
        self.reset_requested.emit()
