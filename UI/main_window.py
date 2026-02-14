from __future__ import annotations

from PySide6.QtCore import Qt
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

from .state import State
from .settings.settings_main import SettingsButton, SettingsDialog

from app_context import get_app_context


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
        self.move(500,200) # Move window to a more convenient position. 
        self._state = State()
        
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

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Toast manager now tracks moves/resizes via event filter
        
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

        # Get the width and height of the settings button match the height of the header bar
        h = self.tabs.tabBar().sizeHint().height()

        self.settingsButton.setFixedHeight(h)
        self.settingsButton.setFixedWidth(max(34, int(h * 0.95)))

        self._apply_status()

    def _build_status_bar(self) -> QWidget:
        status_bar = QFrame()
        status_bar.setObjectName("StatusBar")
        status_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

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
        self.progress.setObjectName("StatusProgress")
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

        self.status_bar.setProperty("kind", self._state.status_type())
        self.status_bar.style().unpolish(self.status_bar)
        self.status_bar.style().polish(self.status_bar)

    def closeEvent(self, event):
        """Handle application close - cleanup resources"""
        # Cleanup camera preview
        if hasattr(self.navigate_tab, 'camera_preview'):
            self.navigate_tab.camera_preview.cleanup()
        
        # Cleanup app context
        ctx = get_app_context()
        ctx.cleanup()
        
        super().closeEvent(event)