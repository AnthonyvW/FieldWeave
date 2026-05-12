from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QSizePolicy,
    QStackedWidget,
)
from PySide6.QtGui import QPainter, QColor, QFont
from PySide6.QtCore import Qt, QSize, QTimer

from common.app_context import get_app_context

from UI.widgets.automation.focus_stack_widget import FocusStackWidget
from UI.widgets.automation.area_scan_widget import AreaScanWidget
from UI.widgets.automation.square_move_widget import SquareMoveWidget
from UI.widgets.automation.tree_core_widget import TreeCoreWidget
from UI.widgets.automation.inspection_calibration_scale_widget import InspectionCalibrationScaleWidget


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
    """QStackedWidget whose size tracks only the current page, not the largest page."""

    def addWidget(self, widget: QWidget) -> int:
        index = super().addWidget(widget)
        if widget is not self.currentWidget():
            widget.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Ignored,
            )
        return index

    def setCurrentIndex(self, index: int) -> None:
        prev = self.currentWidget()
        super().setCurrentIndex(index)
        curr = self.currentWidget()
        if prev is not None and prev is not curr:
            prev.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Ignored,
            )
        if curr is not None:
            curr.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Preferred,
            )
        self.adjustSize()

    def sizeHint(self) -> QSize:
        w = self.currentWidget()
        return w.sizeHint() if w else super().sizeHint()

    def minimumSizeHint(self) -> QSize:
        w = self.currentWidget()
        return w.minimumSizeHint() if w else super().minimumSizeHint()


# ---------------------------------------------------------------------------
# Automation widget
# ---------------------------------------------------------------------------

# How often (ms) to poll the manager for routine state changes.
_POLL_INTERVAL_MS: int = 250


class AutomationWidget(QWidget):
    """
    Top-level automation widget.

    Contains a mode selector drop-down, pause/stop controls, and a stacked
    content area that swaps between automation-specific sub-widgets.

    Pause and Stop are disabled whenever no routine is running.  A QTimer
    polls the MotionControllerManager every 250 ms so the buttons stay in
    sync without coupling the manager to Qt signals.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._paused: bool = False
        self._setup_ui()
        self._setup_poll_timer()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
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
        self._mode_combo.setFixedHeight(30)
        self._mode_combo.setFixedWidth(155)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        control_layout.addWidget(self._mode_combo)

        control_layout.addStretch(1)

        # Pause button — disabled until a routine is running
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setObjectName("AutomationPause")
        self._pause_btn.setFixedSize(70, 30)
        self._pause_btn.setCheckable(True)
        self._pause_btn.setEnabled(False)
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        control_layout.addWidget(self._pause_btn)

        # Stop button — red, disabled until a routine is running
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("AutomationStop")
        self._stop_btn.setFixedSize(70, 30)
        self._stop_btn.setEnabled(False)
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
        self._sub_widgets: list[QWidget] = [
            TreeCoreWidget(),
            FocusStackWidget(),
            AreaScanWidget(),
            SquareMoveWidget(),
            InspectionCalibrationScaleWidget(),
        ]

        for widget in self._sub_widgets:
            self._stack.addWidget(widget)
            self._mode_combo.addItem(widget.mode_name)

        outer_layout.addWidget(self._stack)
        self._stack.setCurrentIndex(0)

    def _setup_poll_timer(self) -> None:
        """Start a timer that syncs button state with the manager's routine state."""
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(_POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._sync_button_state)
        self._poll_timer.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_button_state(self) -> None:
        """Enable/disable controls based on whether a routine is running."""
        manager = get_app_context().motion
        running = manager is not None and manager.routine_running

        self._pause_btn.setEnabled(running)
        self._stop_btn.setEnabled(running)

        if not running:
            # Reset pause visual state when no routine is active.
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
        manager = get_app_context().motion
        if manager is not None:
            manager.stop_routine()
        # _sync_button_state will clean up button visuals on the next tick.
