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

from UI.tabs.logs_tab import LogsTab
from UI.tabs.project_tab import ProjectTab
from UI.tabs.navigate_tab import NavigateTab
from UI.tabs.calibration_tab import CalibrationTab
from UI.widgets.camera_preview import CameraPreview
from UI.widgets.update_notifier import UpdateNotifier
from UI.widgets.changelog_dialog import ChangelogDialog
from UI.settings.settings_main import SettingsButton, SettingsDialog

from common.app_context import get_app_context
from common.state import State, MachineState, AutomationState

from motion.motion_controller import MotionState


_MOTION_TO_MACHINE_STATE: dict[str, str] = {
    MotionState.CONNECTING: MachineState.CONNECTING,
    MotionState.HOMING:     MachineState.HOMING,
    MotionState.READY:      MachineState.CONNECTED,
    MotionState.FAULTED:    MachineState.CONNECTED,
    MotionState.FAILED:     MachineState.DISCONNECTED,
}

_AUTOMATION_STATE_KIND: dict[str, str] = {
    AutomationState.IDLE:     "idle",
    AutomationState.RUNNING:  "active",
    AutomationState.PAUSED:   "active",
    AutomationState.COMPLETE: "done",
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.app_context = get_app_context()
        self.app_context.register_main_window(self)

        self.setWindowTitle(f"FieldWeave - v{self.app_context.current_version}")
        self.resize(1920, 1080)
        self.move(500, 200)
        self._state = State()

        self._pending_machine_state: str = MachineState.DISCONNECTED
        self._pending_job_name: str = "-"
        self._pending_activity: str = "-"
        self._pending_progress_current: int = 0
        self._pending_progress_total: int = 0
        self._pending_eta_seconds: int = 0

        self._completed_latch: bool = False
        self._completed_job_name: str = "-"

        self.settings_dialog = SettingsDialog(self)
        self.app_context.register_settings_dialog(self.settings_dialog)

        # Create and register the single shared camera preview before any tab
        # is constructed, so tabs can reference it during their own __init__.
        self._camera_preview = CameraPreview(self)
        self.app_context.register_camera_preview(self._camera_preview)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.navigate_tab = NavigateTab()
        self.project_tab = ProjectTab()
        self.tabs.addTab(self.navigate_tab, "Navigate")
        self.tabs.addTab(self.project_tab, "Project")
        self.tabs.addTab(CalibrationTab(), "Calibration")
        self.tabs.addTab(LogsTab(), "Logs")

        self._setup_header_right()
        self.setCentralWidget(self.tabs)

        self._wire_motion_state()
        self._start_state_flush_timer()

        if self.app_context.updater is not None:
            self._update_notifier = UpdateNotifier(self.app_context.updater, self)
            self.app_context.register_update_notifier(self._update_notifier)

        self._show_changelog_if_updated()

    # ------------------------------------------------------------------
    # Changelog on update
    # ------------------------------------------------------------------

    def _show_changelog_if_updated(self) -> None:
        """Pop the changelog once on a version change or a fresh install.

        FieldWeaveSettingsManager sets show_patchnotes when the saved
        config's version doesn't match the running version, and
        AppContext sets first_start when no config file existed yet.
        Deferred with singleShot so the main window is shown first and
        the dialog opens on top of it rather than blocking construction.
        """
        settings = self.app_context.settings
        if settings is None or not (settings.show_patchnotes or settings.first_start):
            return
        settings.show_patchnotes = False
        settings.first_start = False
        QTimer.singleShot(0, lambda: ChangelogDialog(self).exec())

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

        self._pending_machine_state = _MOTION_TO_MACHINE_STATE.get(
            motion.get_state(), MachineState.DISCONNECTED
        )

    def _on_motion_state_changed(self, new_state: str) -> None:
        self._pending_machine_state = _MOTION_TO_MACHINE_STATE.get(
            new_state, MachineState.DISCONNECTED
        )

    def _on_routine_state_changed(
        self, job_name: str, activity: str, progress_current: int, progress_total: int, eta_seconds: int
    ) -> None:
        if job_name != "-" or activity != "-":
            self._completed_latch = False

        self._pending_job_name = job_name
        self._pending_activity = activity
        self._pending_progress_current = progress_current
        self._pending_progress_total = progress_total
        self._pending_eta_seconds = eta_seconds

    def _on_motion_interaction(self) -> None:
        self._completed_latch = False

    # ------------------------------------------------------------------
    # State flush timer
    # ------------------------------------------------------------------

    def _start_state_flush_timer(self) -> None:
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(250)
        self._flush_timer.timeout.connect(self._flush_state)
        self._flush_timer.start()

    def _flush_state(self) -> None:
        """Rebuild State from pending fields and repaint the status bar if changed."""
        motion = self.app_context.motion

        if motion is not None and motion.routine_running:
            automation_state = (
                AutomationState.PAUSED if motion.routine_paused else AutomationState.RUNNING
            )
            job_name = self._pending_job_name
            activity = self._pending_activity
            progress_current = self._pending_progress_current
            progress_total = self._pending_progress_total
            eta_seconds = self._pending_eta_seconds

        elif self._state.automation_state in (AutomationState.RUNNING, AutomationState.PAUSED):
            self._completed_latch = True
            self._completed_job_name = self._state.job_name
            automation_state = AutomationState.COMPLETE
            job_name = self._completed_job_name
            activity = "-"
            progress_current = 0
            progress_total = 0
            eta_seconds = 0

        elif self._completed_latch:
            automation_state = AutomationState.COMPLETE
            job_name = self._completed_job_name
            activity = "-"
            progress_current = 0
            progress_total = 0
            eta_seconds = 0

        else:
            automation_state = AutomationState.IDLE
            job_name = "-"
            activity = "-"
            progress_current = 0
            progress_total = 0
            eta_seconds = 0

        new_state = State(
            machine_state=self._pending_machine_state,
            automation_state=automation_state,
            activity=activity,
            job_name=job_name,
            progress_current=progress_current,
            progress_total=progress_total,
            eta_seconds=eta_seconds,
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

        self.status_bar = self._build_status_bar()

        self.settingsButton = SettingsButton("Settings")
        self.settingsButton.clicked.connect(lambda: self._open_settings("Camera"))

        layout.addWidget(self.status_bar)
        layout.addWidget(self.settingsButton)

        self.tabs.setCornerWidget(header_edge, Qt.Corner.TopRightCorner)

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

        self.status_line = QLabel("-")
        self.status_line.setObjectName("StatusLine")
        self.status_line.setWordWrap(False)
        self.status_line.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding,
            QSizePolicy.Policy.Fixed,
        )

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
            self.progress.setFormat("%p%")

        kind = _AUTOMATION_STATE_KIND.get(self._state.automation_state, "idle")
        self.status_bar.setProperty("kind", kind)
        self.status_bar.style().unpolish(self.status_bar)
        self.status_bar.style().polish(self.status_bar)

    def closeEvent(self, event) -> None:
        self._flush_timer.stop()
        self._camera_preview.cleanup()
        super().closeEvent(event)
