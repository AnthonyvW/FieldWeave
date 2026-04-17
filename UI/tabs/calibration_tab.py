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
from UI.tabs.calibration_pages.dpi_calibration import DpiCalibrationPage, DpiCalibrationWidget
from UI.tabs.calibration_pages.slot_calibration import SlotCalibrationWidget

_SIDEBAR_TITLES: list[str] = [
    "Camera Space Calibration",
    "DPI Calibration",
    "Sample Slot Position Calibration",
]

_CAMERA_SPACE_INDEX = 0
_DPI_INDEX = 1


class CalibrationTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._build_ui()
        self._calibration_list.setCurrentRow(0)
        self._on_calibration_selected(0)

    def _build_ui(self) -> None:
        self._outer_stack = QStackedWidget()
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        outer_layout.addWidget(self._outer_stack)

        # --- Normal calibration layout (outer stack index 0) ---
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
        self._calibration_list.setObjectName("CalibrationSidebar")
        self._calibration_list.currentRowChanged.connect(self._on_calibration_selected)
        sidebar_layout.addWidget(self._calibration_list)
        main_layout.addWidget(sidebar)

        self._content_stack = QStackedWidget()

        self._camera_space_widget = CameraSpaceCalibrationWidget()
        self._camera_space_widget.calibration_started.connect(
            lambda: self._on_calibration_started(_CAMERA_SPACE_INDEX)
        )
        self._content_stack.addWidget(self._camera_space_widget)

        self._dpi_widget = DpiCalibrationWidget()
        self._dpi_widget.calibration_started.connect(
            lambda: self._on_calibration_started(_DPI_INDEX)
        )
        self._content_stack.addWidget(self._dpi_widget)

        self._slot_widget = SlotCalibrationWidget()
        self._content_stack.addWidget(self._slot_widget)

        main_layout.addWidget(self._content_stack, 1)
        self._outer_stack.addWidget(calibration_widget)

        # --- Live calibration pages (outer stack indices 1, 2) ---
        self._camera_space_page = CameraSpaceCalibrationPage()
        self._camera_space_page.finished.connect(self._on_calibration_finished)
        self._outer_stack.addWidget(self._camera_space_page)

        self._dpi_page = DpiCalibrationPage()
        self._dpi_page.finished.connect(self._on_calibration_finished)
        self._outer_stack.addWidget(self._dpi_page)

        self._live_page_indices: dict[int, int] = {
            _CAMERA_SPACE_INDEX: 1,
            _DPI_INDEX: 2,
        }
        self._active_list_index: int = 0

    @Slot(int)
    def _on_calibration_selected(self, index: int) -> None:
        if index < 0:
            return
        widget = self._content_stack.widget(index)
        if hasattr(widget, "reset"):
            widget.reset()
        self._content_stack.setCurrentIndex(index)

    @Slot()
    def _on_calibration_started(self, list_index: int) -> None:
        self._active_list_index = list_index
        if list_index == _CAMERA_SPACE_INDEX:
            self._camera_space_page.start()
            self._outer_stack.setCurrentIndex(self._live_page_indices[_CAMERA_SPACE_INDEX])
        elif list_index == _DPI_INDEX:
            self._dpi_page.start()
            self._outer_stack.setCurrentIndex(self._live_page_indices[_DPI_INDEX])

    @Slot()
    def _on_calibration_finished(self) -> None:
        self._outer_stack.setCurrentIndex(0)
        self._calibration_list.setCurrentRow(self._active_list_index)
        widget = self._content_stack.widget(self._active_list_index)
        if hasattr(widget, "reset"):
            widget.reset()