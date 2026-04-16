from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from UI.tabs.calibration_pages.camera_space_calibration import (
    CameraSpaceCalibrationPage,
    CameraSpaceCalibrationWidget,
)
from UI.tabs.calibration_pages.dpi_calibration import DpiCalibrationWidget
from UI.tabs.calibration_pages.slot_calibration import SlotCalibrationWidget

_SIDEBAR_TITLES: list[str] = [
    "Camera Space Calibration",
    "DPI Calibration",
    "Sample Slot Position Calibration",
]

_CAMERA_SPACE_INDEX = 2


class CalibrationTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._build_ui()
        self._calibration_list.setCurrentRow(0)
        self._on_calibration_selected(0)

    def _build_ui(self) -> None:
        # Outer stack: index 0 = normal calibration layout, index 1 = live page
        self._outer_stack = QStackedWidget()
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        outer_layout.addWidget(self._outer_stack)

        # --- Normal calibration layout ---
        calibration_widget = QWidget()
        main_layout = QHBoxLayout(calibration_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        self._calibration_list = QListWidget()
        self._calibration_list.setMaximumWidth(250)
        self._calibration_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._calibration_list.addItems(_SIDEBAR_TITLES)
        self._calibration_list.setStyleSheet(
            "QListWidget {"
            "  font-size: 13px; padding: 5px;"
            "  border: none; border-right: 2px solid #b3b4b6;"
            "  background: #f8f8f8;"
            "}"
            "QListWidget::item { padding: 12px; border-bottom: 1px solid #e0e0e0; color: #000000; }"
            "QListWidget::item:selected { background: #dbdbdb; color: #000000; border: none; }"
            "QListWidget::item:hover { background: #e8e8e8; color: #000000; }"
        )
        self._calibration_list.currentRowChanged.connect(self._on_calibration_selected)
        sidebar_layout.addWidget(self._calibration_list)
        main_layout.addWidget(sidebar)

        self._content_stack = QStackedWidget()

        self._camera_space_widget = CameraSpaceCalibrationWidget()
        self._camera_space_widget.calibration_started.connect(self._on_calibration_started)
        self._content_stack.addWidget(self._camera_space_widget)

        self._dpi_widget = DpiCalibrationWidget()
        self._content_stack.addWidget(self._dpi_widget)

        self._slot_widget = SlotCalibrationWidget()
        self._content_stack.addWidget(self._slot_widget)

        main_layout.addWidget(self._content_stack, 1)
        self._outer_stack.addWidget(calibration_widget)

        # --- Live calibration page ---
        self._live_page = CameraSpaceCalibrationPage()
        self._live_page.finished.connect(self._on_calibration_finished)
        self._outer_stack.addWidget(self._live_page)

    @Slot(int)
    def _on_calibration_selected(self, index: int) -> None:
        if index < 0:
            return
        widget = self._content_stack.widget(index)
        if hasattr(widget, "reset"):
            widget.reset()
        self._content_stack.setCurrentIndex(index)

    @Slot()
    def _on_calibration_started(self) -> None:
        self._live_page.start()
        self._outer_stack.setCurrentIndex(1)

    @Slot()
    def _on_calibration_finished(self) -> None:
        self._outer_stack.setCurrentIndex(0)
        self._calibration_list.setCurrentRow(_CAMERA_SPACE_INDEX)
        self._camera_space_widget.reset()