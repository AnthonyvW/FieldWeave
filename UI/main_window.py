from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QSizePolicy,
    QTabWidget,
    QWidget,
)

from .tabs.navigate_tab import NavigateTab
from .tabs.project_tab import ProjectTab
from .tabs.calibration_tab import CalibrationTab
from .tabs.logs_tab import LogsTab

from common.state import State, MachineState, AutomationState
from .settings.settings_main import SettingsButton, SettingsDialog

from common.app_context import get_app_context
from motion.motion_controller import MotionState


# Map MotionState strings to the MachineState enum values shown in the status bar.
_MOTION_TO_MACHINE_STATE: dict[str, str] = {
    MotionState.CONNECTING: MachineState.CONNECTING,
    MotionState.READY:      MachineState.CONNECTED,
    MotionState.FAULTED:    MachineState.CONNECTED,   # still physically connected
    MotionState.FAILED:     MachineState.DISCONNECTED,
}

# Map AutomationState enum values to the "kind" attribute strings used by the
# stylesheet to colour the status bar (see style.py).
_AUTOMATION_STATE_KIND: dict[str, str] = {
    AutomationState.IDLE:     "idle",
    AutomationState.RUNNING:  "active",
    AutomationState.PAUSED:   "active",
    AutomationState.COMPLETE: "done",
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        # Get app context
        self.app_context = get_app_context()

        # Register this main window with app context (initializes toast manager)
        self.app_context.register_main_window(self)

        # Set window title with version
        self.setWindowTitle(f"FieldWeave - v{self.app_context.current_version}")
        self.resize(1920, 1080)
        self.move(500, 200)
        self._state = State()

        # Pending state fields updated from background threads; applied on the
        # main thread via a QTimer to keep Qt widget access thread-safe.
        self._pending_machine_state: str = MachineState.DISCONNECTED
        self._pending_job_name: str = "-"
        self._pending_activity: str = "-"
        self._pending_progress_current: int = 0
        self._pending_progress_total: int = 0

        # When a routine finishes this latches True so the COMPLETE state is
        # held in the status bar.  It is cleared when:
        #   - a new routine starts (detected via _on_routine_state_changed), or
        #   - the user interacts with the motion system directly (jog, home, etc.)
        #     which is detected via the interaction listener registered below.
        self._completed_latch: bool = False
        # The job name preserved while the latch is held so the status bar can
        # show "Completed  |  <job>" even after the routine has cleaned up.
        self._completed_job_name: str = "-"

        # Create and register settings dialog
        self.settings_dialog = SettingsDialog(self)
        self.app_context.register_settings_dialog(self.settings_dialog)

        # Header Bar
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # Create tabs
        self.navigate_tab = NavigateTab()
        self.tabs.addTab(self.navigate_tab, "Navigate")
        self.tabs.addTab(ProjectTab(), "Project")
        self.tabs.addTab(CalibrationTab(), "Calibration")
        self.tabs.addTab(LogsTab(), "Logs")

        self._setup_header_right()
        self.setCentralWidget(self.tabs)

        # Wire up motion state callbacks and start the flush timer.
        self._wire_motion_state()
        self._start_state_flush_timer()

    # ------------------------------------------------------------------
    # Motion state wiring
    # ------------------------------------------------------------------

    def _wire_motion_state(self) -> None:
        """Subscribe to motion controller state and routine state changes."""
        motion = self.app_context.motion
        if motion is None:
            return

        motion.add_state_listener(self._on_motion_state_changed)
        motion.add_routine_state_listener(self._on_routine_state_changed)
        motion.add_interaction_listener(self._on_motion_interaction)

        # Seed the pending state with whatever the controller reports right now
        # so the status bar is correct even before the first callback fires.
        self._pending_machine_state = _MOTION_TO_MACHINE_STATE.get(
            motion.get_state(), MachineState.DISCONNECTED
        )

    def _on_motion_state_changed(self, new_state: str) -> None:
        """Called on the poll thread when MotionState transitions."""
        self._pending_machine_state = _MOTION_TO_MACHINE_STATE.get(
            new_state, MachineState.DISCONNECTED
        )

    def _on_routine_state_changed(
        self, job_name: str, activity: str, progress_current: int, progress_total: int
    ) -> None:
        """Called on the routine thread when job/activity/progress updates."""
        if job_name != "-" or activity != "-":
            # A routine is actively reporting — clear any stale latch so the
            # live job info is shown instead of the previous completion.
            self._completed_latch = False

        self._pending_job_name = job_name
        self._pending_activity = activity
        self._pending_progress_current = progress_current
        self._pending_progress_total = progress_total

    def _on_motion_interaction(self) -> None:
        """Called when the user issues a direct motion command (jog, home, etc.).

        Any manual interaction signals the operator has moved on, so the
        COMPLETE latch is cleared and the status bar returns to IDLE.
        """
        self._completed_latch = False

    # ------------------------------------------------------------------
    # State flush timer
    # ------------------------------------------------------------------

    def _start_state_flush_timer(self) -> None:
        """
        Poll pending state fields every 250 ms on the main thread and rebuild
        the status bar if anything has changed.

        Using a timer (rather than Qt signals emitted from callbacks) keeps the
        motion module free of any Qt dependency.
        """
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(250)
        self._flush_timer.timeout.connect(self._flush_state)
        self._flush_timer.start()

    def _flush_state(self) -> None:
        """Rebuild State from pending fields and repaint the status bar if changed."""
        motion = self.app_context.motion

        if motion is not None and motion.routine_running:
            # A routine is actively executing.
            automation_state = (
                AutomationState.PAUSED if motion.routine_paused else AutomationState.RUNNING
            )
            job_name = self._pending_job_name
            activity = self._pending_activity
            progress_current = self._pending_progress_current
            progress_total = self._pending_progress_total

        elif self._state.automation_state in (AutomationState.RUNNING, AutomationState.PAUSED):
            # Routine just finished on this flush tick — set the latch and
            # capture the job name before the pending fields are cleared.
            self._completed_latch = True
            self._completed_job_name = self._state.job_name
            automation_state = AutomationState.COMPLETE
            job_name = self._completed_job_name
            activity = "-"
            progress_current = 0
            progress_total = 0

        elif self._completed_latch:
            # Latch is held from a previous completion — keep showing COMPLETE.
            automation_state = AutomationState.COMPLETE
            job_name = self._completed_job_name
            activity = "-"
            progress_current = 0
            progress_total = 0

        else:
            automation_state = AutomationState.IDLE
            job_name = "-"
            activity = "-"
            progress_current = 0
            progress_total = 0

        new_state = State(
            machine_state=self._pending_machine_state,
            automation_state=automation_state,
            activity=activity,
            job_name=job_name,
            progress_current=progress_current,
            progress_total=progress_total,
        )

        if new_state != self._state:
            self._state = new_state
            self._apply_status()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

    def _setup_header_right(self) -> None:
        header_edge = QWidget()
        header_edge.setObjectName("TabCorner")

        layout = QHBoxLayout(header_edge)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Status
        self.status_bar = self._build_status_bar()

        # Settings Button
        self.settingsButton = SettingsButton("Settings")
        self.settingsButton.clicked.connect(lambda: self._open_settings("Camera"))

        layout.addWidget(self.status_bar)
        layout.addWidget(self.settingsButton)

        self.tabs.setCornerWidget(header_edge, Qt.Corner.TopRightCorner)

        # Pin every element to the exact tab bar height so the coloured status
        # frame fills the full header strip rather than only wrapping its text.
        h = self.tabs.tabBar().sizeHint().height()

        header_edge.setFixedHeight(h)
        self.status_bar.setFixedHeight(h)
        self.settingsButton.setFixedHeight(h)
        self.settingsButton.setFixedWidth(max(34, int(h * 0.95)))

        self._apply_status()

    def _build_status_bar(self) -> QWidget:
        status_bar = QFrame()
        status_bar.setObjectName("StatusBar")
        status_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        row = QHBoxLayout(status_bar)
        row.setContentsMargins(10, 0, 10, 0)
        row.setSpacing(10)

        # Status Text
        self.status_line = QLabel("-")
        self.status_line.setObjectName("StatusLine")
        self.status_line.setWordWrap(False)
        self.status_line.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding,
            QSizePolicy.Policy.Fixed,
        )

        # Progress Bar | Optional
        self.progress = QProgressBar()
        self.progress.setObjectName("CornerStatusProgress")
        self.progress.setRange(0, 100)
        self.progress.setFixedWidth(120)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed
        )

        row.addWidget(self.status_line, stretch=1)
        row.addWidget(self.progress, stretch=0)

        return status_bar

    def _open_settings(self, category: str) -> None:
        self.app_context.open_settings(category)

    def _apply_status(self) -> None:
        self.status_line.setText(self._state.format_status_text())

        show_progress = self._state.progress_total > 0
        self.progress.setVisible(show_progress)

        if show_progress:
            percent = int(round(100.0 * self._state.progress_current / max(1, self._state.progress_total)))
            self.progress.setValue(max(0, min(100, percent)))

        kind = _AUTOMATION_STATE_KIND.get(self._state.automation_state, "idle")
        self.status_bar.setProperty("kind", kind)
        self.status_bar.style().unpolish(self.status_bar)
        self.status_bar.style().polish(self.status_bar)

    def closeEvent(self, event) -> None:
        """Handle application close - cleanup resources."""
        self._flush_timer.stop()

        # Cleanup camera preview
        if hasattr(self.navigate_tab, 'camera_preview'):
            self.navigate_tab.camera_preview.cleanup()

        # Cleanup app context
        ctx = get_app_context()
        ctx.cleanup()

        super().closeEvent(event)