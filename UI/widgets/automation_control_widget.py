from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QStackedWidget,
)
from PySide6.QtGui import QPainter, QColor, QFont
from PySide6.QtCore import Qt, QSize

from common.app_context import get_app_context

from UI.widgets.automation.focus_stack_widget import FocusStackWidget
from UI.widgets.automation.focus_stack_area_scan_widget import ZStackAreaScanWidget
from UI.widgets.automation.square_move_widget import SquareMoveWidget
from UI.widgets.automation.tree_core_widget import TreeCoreWidget
from UI.widgets.automation.camera_calibration_widget import CameraCalibrationWidget

class _ArrowComboBox(QComboBox):
    """QComboBox that draws a ▼ character in the drop-down button area."""

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw the grey drop-down panel on the right
        arrow_w = 24
        panel_x = self.width() - arrow_w
        painter.fillRect(panel_x, 0, arrow_w, self.height(), QColor(215, 217, 220))

        # Left border of the panel
        painter.setPen(QColor(140, 140, 140))
        painter.drawLine(panel_x, 0, panel_x, self.height())

        # Draw the ▼ character centred in the panel
        font = QFont(self.font())
        font.setPixelSize(11)
        painter.setFont(font)
        painter.setPen(QColor(60, 60, 60))
        painter.drawText(panel_x, 0, arrow_w, self.height(), Qt.AlignmentFlag.AlignCenter, "▼")

class _CollapsibleStack(QStackedWidget):
    def sizeHint(self) -> QSize:
        w = self.currentWidget()
        return w.sizeHint() if w else super().sizeHint()

    def minimumSizeHint(self) -> QSize:
        w = self.currentWidget()
        return w.minimumSizeHint() if w else super().minimumSizeHint()

# ---------------------------------------------------------------------------
# Automation widget
# ---------------------------------------------------------------------------

_MODES: list[str] = ["Tree Core Imaging", "Focus Stacking", "Z-Stack Area Scan", "(DEV) Square Move", "Camera Calibration"]


class AutomationWidget(QWidget):
    """
    Top-level automation widget.

    Contains a mode selector drop-down, pause/stop controls, and a stacked
    content area that swaps between automation-specific sub-widgets.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._running: bool = False
        self._paused: bool = False
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setStyleSheet("AutomationWidget { background: white; }")

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(8)

        # ---- Top control bar ----
        control_bar = QWidget()
        control_layout = QHBoxLayout(control_bar)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(8)

        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("font-size: 13px;")
        control_layout.addWidget(mode_label)

        self._mode_combo = _ArrowComboBox()
        self._mode_combo.addItems(_MODES)
        self._mode_combo.setFixedHeight(30)
        self._mode_combo.setFixedWidth(155)
        self._mode_combo.setStyleSheet("""
            QComboBox {
                font-size: 13px;
                padding: 2px 28px 2px 6px;
                border: 1px solid rgb(140, 140, 140);
                border-radius: 0px;
                background: white;
            }
            QComboBox:hover {
                border: 1px solid rgb(80, 80, 80);
                background: rgb(245, 245, 245);
            }
            QComboBox::drop-down {
                width: 0px;
                border: none;
                background: transparent;
            }
            QComboBox QAbstractItemView {
                font-size: 13px;
                border: 1px solid rgb(140, 140, 140);
                selection-background-color: #f28c28;
                selection-color: white;
                outline: none;
            }
        """)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        control_layout.addWidget(self._mode_combo)

        control_layout.addStretch(1)

        # Pause button
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setFixedSize(70, 30)
        self._pause_btn.setCheckable(True)
        self._pause_btn.setStyleSheet(self._action_button_style(checked_color="rgb(230, 180, 60)"))
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        control_layout.addWidget(self._pause_btn)

        # Stop button
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedSize(70, 30)
        self._stop_btn.setStyleSheet(self._action_button_style(checked_color="rgb(210, 80, 80)"))
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        control_layout.addWidget(self._stop_btn)

        outer_layout.addWidget(control_bar)

        # Divider line
        divider = QWidget()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background: rgb(210, 210, 210);")
        outer_layout.addWidget(divider)

        # ---- Stacked content area ----
        self._stack = _CollapsibleStack()
        self._focus_stack_widget = FocusStackWidget()
        self._area_scan_widget = ZStackAreaScanWidget()
        self._tree_core_widget = TreeCoreWidget()
        self._square_move_widget = SquareMoveWidget()
        self._camera_calibration_widget = CameraCalibrationWidget()

        self._stack.addWidget(self._tree_core_widget)     # Tree Core Imaging
        self._stack.addWidget(self._focus_stack_widget)   # Focus Stacking
        self._stack.addWidget(self._area_scan_widget)     # Z-Stack Area Scan
        self._stack.addWidget(self._square_move_widget)   # Square Move
        self._stack.addWidget(self._camera_calibration_widget)   # Square Move

        outer_layout.addWidget(self._stack)

        # Reflect default selection
        self._stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _action_button_style(*, checked_color: str) -> str:
        return f"""
            QPushButton {{
                background-color: rgb(208, 211, 214);
                border: 1px solid rgb(150, 150, 150);
                border-radius: 0px;
                font-size: 13px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: rgb(187, 190, 193);
            }}
            QPushButton:pressed {{
                background-color: rgb(170, 173, 175);
            }}
            QPushButton:checked {{
                background-color: {checked_color};
                border: 1px solid rgb(120, 120, 120);
                color: white;
                font-weight: bold;
            }}
        """

    def _set_running(self, running: bool) -> None:
        self._running = running
        if not running:
            self._paused = False
            self._pause_btn.setChecked(False)
            self._pause_btn.setText("Pause")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_mode_changed(self, index: int) -> None:
        self._stack.setCurrentIndex(index)

    def _on_pause_clicked(self) -> None:
        self._paused = self._pause_btn.isChecked()
        self._pause_btn.setText("Resume" if self._paused else "Pause")
        manager = get_app_context().motion
        if manager is None:
            return
        if self._paused:
            manager.pause_routine()
        else:
            manager.resume_routine()

    def _on_stop_clicked(self) -> None:
        self._set_running(False)
        manager = get_app_context().motion
        if manager is not None:
            manager.stop_routine()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def current_mode(self) -> str:
        return self._mode_combo.currentText()

    @property
    def focus_stack_widget(self) -> FocusStackWidget:
        return self._focus_stack_widget

    @property
    def area_scan_widget(self) -> ZStackAreaScanWidget:
        return self._area_scan_widget

    @property
    def tree_core_widget(self) -> TreeCoreWidget:
        return self._tree_core_widget
