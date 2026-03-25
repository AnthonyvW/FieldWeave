from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from UI.tabs.calibration_pages.dpi_calibration import DpiCalibrationWidget
from UI.tabs.calibration_pages.slot_calibration import SlotCalibrationWidget

# ---------------------------------------------------------------------------
# Per-calibration metadata
# ---------------------------------------------------------------------------

_CALIBRATIONS: list[dict[str, str]] = [
    {
        "title": "DPI Calibration",
        "description": (
            "<b>Purpose:</b><br>"
            "Fine-tunes the camera's pixels-per-millimetre ratio for accurate "
            "image to real world movement.<br><br>"
            "<b>What it does:</b><br>"
            "• Measures pixels per millimetre<br>"
            "• Enables click to move<br><br>"
            "<b>What you need:</b><br>"
            "• A calibration target with known dimensions<br>"
            "• Stable lighting conditions<br>"
            "• Approximately 3 minutes<br><br>"
            "<b>Process:</b><br>"
            "The calibration will capture images of the target and calculate "
            "scaling factors for your specific camera setup."
        ),
    },
    {
        "title": "Sample Slot Position Calibration",
        "description": (
            "<b>Purpose:</b><br>"
            "Maps the position of every sample slot so the system can navigate "
            "to each one accurately and repeatably.<br><br>"
            "<b>What it does:</b><br>"
            "• Records reference positions for the slots<br>"
            "• Calculates slot spacing and geometry<br>"
            "• Enables automatic slot navigation<br><br>"
            "<b>What you need:</b><br>"
            "• An empty sample tray mounted on the machine<br>"
            "• Clear visibility of the start of each slot<br>"
            "• Approximately 5 minutes<br><br>"
            "<b>Process:</b><br>"
            "The calibration will guide you through the process of calibrating the sample slot positions."
        ),
    },
]


# ---------------------------------------------------------------------------
# Info / start page
# ---------------------------------------------------------------------------

class _InfoPage(QWidget):
    """Displays a description for the selected calibration and a start button."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self._title = QLabel()
        self._title.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #5a5a5a;"
        )
        layout.addWidget(self._title)

        self._description = QLabel()
        self._description.setWordWrap(True)
        self._description.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._description.setStyleSheet(
            "QLabel {"
            "  font-size: 14px; color: #000000;"
            "  background: #f8f8f8; padding: 20px;"
            "  border: 1px solid #e0e0e0;"
            "}"
        )
        layout.addWidget(self._description)

        layout.addStretch()

        start_btn = QPushButton("Start Calibration")
        start_btn.setStyleSheet(
            "QPushButton {"
            "  font-size: 16px; font-weight: bold; padding: 12px 30px;"
            "  background: #dbdbdb; border: 2px solid #b3b4b6;"
            "}"
            "QPushButton:hover { background: #b3b4b6; }"
        )
        start_btn.setMinimumHeight(45)
        layout.addWidget(start_btn)

    # ------------------------------------------------------------------

    def load(self, index: int) -> None:
        entry = _CALIBRATIONS[index]
        self._title.setText(entry["title"])
        self._description.setText(entry["description"])


# ---------------------------------------------------------------------------
# Main tab
# ---------------------------------------------------------------------------

class CalibrationTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._build_ui()
        # Select first item and show its info page
        self._calibration_list.setCurrentRow(0)
        self._on_calibration_selected(0)

    # ---------------------------------------------------------- UI construction

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Sidebar ---
        self._sidebar = QWidget()
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        self._calibration_list = QListWidget()
        self._calibration_list.setMaximumWidth(250)
        self._calibration_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._calibration_list.addItems(
            [entry["title"] for entry in _CALIBRATIONS]
        )
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
        self._calibration_list.currentRowChanged.connect(
            self._on_calibration_selected
        )
        sidebar_layout.addWidget(self._calibration_list)
        main_layout.addWidget(self._sidebar)

        # --- Content stack ---
        self._content_stack = QStackedWidget()

        # Page 0: info/start
        self._info_page = _InfoPage()
        self._content_stack.addWidget(self._info_page)

        # Page 1: DPI calibration
        self._dpi_widget = DpiCalibrationWidget()
        self._content_stack.addWidget(self._dpi_widget)

        # Page 2: Slot calibration
        self._slot_widget = SlotCalibrationWidget()
        self._content_stack.addWidget(self._slot_widget)

        main_layout.addWidget(self._content_stack, 1)

        # Track which calibration is active
        self._active_calibration_index: int = 0

    # ---------------------------------------------------------- slots

    @Slot(int)
    def _on_calibration_selected(self, index: int) -> None:
        if index < 0:
            return
        self._active_calibration_index = index
        self._info_page.load(index)
        self._sidebar.setVisible(True)
        self._content_stack.setCurrentIndex(0)