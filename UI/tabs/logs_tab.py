from __future__ import annotations

import re
import subprocess
import sys
from collections import deque
from datetime import datetime

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from common.logger import get_logger

MAX_LOG_LINES = 5000
FLUSH_INTERVAL_MS = 100
# Pixels from the bottom that still count as "at the latest logs"
BOTTOM_TOLERANCE = 4

LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
LEVEL_COLORS = {
    'DEBUG': '#666666',
    'INFO': '#0066cc',
    'WARNING': '#cc6600',
    'ERROR': '#cc0000',
    'CRITICAL': '#990000',
}
LOG_LINE_PATTERN = re.compile(r'\[([^\]]+)\]\s+(\w+):\s+(.*)')


def _escape_html(text: str) -> str:
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))


def _format_entry(timestamp: str, level: str, message: str) -> str:
    color = LEVEL_COLORS.get(level, '#000000')
    return (
        f'<span style="color: #666666;">[{timestamp}]</span> '
        f'<span style="color: {color};">[{level}]</span> '
        f'{_escape_html(message)}'
    )


class LogsTab(QWidget):
    """Logs tab showing application logs with controls"""

    _log_received = Signal(str, str)  # level, message

    def __init__(self) -> None:
        super().__init__()

        self._level_filters = {
            'DEBUG': False,
            'INFO': True,
            'WARNING': True,
            'ERROR': True,
            'CRITICAL': True,
        }

        # Entries are (level, preformatted html) so filter changes don't re-escape everything
        self._log_entries: deque[tuple[str, str]] = deque(maxlen=MAX_LOG_LINES)
        self._pending: deque[tuple[str, str]] = deque(maxlen=MAX_LOG_LINES)

        self._log_display = QPlainTextEdit()
        self._log_display.setReadOnly(True)
        self._log_display.setMaximumBlockCount(MAX_LOG_LINES)
        self._log_display.setUndoRedoEnabled(False)
        self._log_display.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._log_display.setStyleSheet("""
            QPlainTextEdit {
                background-color: #ffffff;
                color: #000000;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                border: 1px solid #cccccc;
            }
        """)

        self._clear_btn = QPushButton("Clear Display")
        self._clear_btn.clicked.connect(self._clear_display)

        self._open_folder_btn = QPushButton("Open Log Folder")
        self._open_folder_btn.clicked.connect(self._open_log_folder)

        self._auto_scroll_check = QCheckBox("Auto-scroll")
        self._auto_scroll_check.setChecked(True)

        self._level_checkboxes: dict[str, QCheckBox] = {}

        control_layout = QHBoxLayout()
        control_layout.addWidget(self._clear_btn)
        control_layout.addWidget(self._open_folder_btn)
        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Show levels:"))

        for level in LEVELS:
            checkbox = QCheckBox(level)
            checkbox.setChecked(self._level_filters[level])
            checkbox.stateChanged.connect(
                lambda state, lvl=level: self._on_filter_changed(lvl, state))
            self._level_checkboxes[level] = checkbox
            control_layout.addWidget(checkbox)

        control_layout.addStretch()
        control_layout.addWidget(self._auto_scroll_check)

        layout = QVBoxLayout(self)
        layout.addWidget(self._log_display, 1)
        layout.addLayout(control_layout)

        # Batch incoming messages so bursts of logging cause one repaint instead of hundreds
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(FLUSH_INTERVAL_MS)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.timeout.connect(self._flush_pending)

        self._log_received.connect(
            self._handle_log_on_main_thread, Qt.ConnectionType.QueuedConnection)

        self._logger = get_logger()
        self._logger.register_callback(self._on_log_message)

        # A child tab never receives closeEvent, so unregister on quit and on destruction instead.
        # The destroyed handler must not reference self's C++ side, which is already gone by then.
        logger = self._logger
        callback = self._on_log_message
        self.destroyed.connect(lambda: logger.unregister_callback(callback))
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._unregister_from_logger)

        self._load_existing_logs()

    def _load_existing_logs(self) -> None:
        """Load the tail of the current log file (one time only at startup)"""
        try:
            current_log_file = self._logger.get_current_log_file()
            if not current_log_file or not current_log_file.exists():
                return

            with open(current_log_file, 'r', encoding='utf-8') as f:
                tail = deque(f, maxlen=MAX_LOG_LINES)
        except Exception as e:
            self._log_display.appendPlainText(f"Error loading existing logs: {e}")
            return

        # Continuation lines (e.g. tracebacks) inherit the level of the line they follow
        last_level = 'INFO'
        for line in tail:
            line = line.rstrip()
            if not line:
                continue
            match = LOG_LINE_PATTERN.match(line)
            if match:
                timestamp, level, message = match.groups()
                last_level = level
                self._log_entries.append((level, _format_entry(timestamp, level, message)))
            else:
                self._log_entries.append((last_level, _escape_html(line)))

        self._redisplay_logs()

    def _on_filter_changed(self, level: str, state: int) -> None:
        self._level_filters[level] = bool(state)
        self._redisplay_logs()

    def _redisplay_logs(self) -> None:
        """Rebuild the display from memory using the current filters"""
        self._pending.clear()
        self._log_display.setUpdatesEnabled(False)
        try:
            self._log_display.clear()
            for level, html in self._log_entries:
                if self._level_filters.get(level, True):
                    self._log_display.appendHtml(html)
        finally:
            self._log_display.setUpdatesEnabled(True)
        self._scroll_to_bottom()

    def _on_log_message(self, level: str, message: str) -> None:
        """Called from the logger, possibly on a worker thread"""
        self._log_received.emit(level, message)

    def _handle_log_on_main_thread(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        entry = (level, _format_entry(timestamp, level, message))
        self._log_entries.append(entry)

        if not self._level_filters.get(level, True):
            return

        self._pending.append(entry)
        if self.isVisible() and not self._flush_timer.isActive():
            self._flush_timer.start()

    def _flush_pending(self) -> None:
        if not self._pending:
            return

        scrollbar = self._log_display.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= scrollbar.maximum() - BOTTOM_TOLERANCE
        previous_value = scrollbar.value()

        self._log_display.setUpdatesEnabled(False)
        try:
            while self._pending:
                self._log_display.appendHtml(self._pending.popleft()[1])
        finally:
            self._log_display.setUpdatesEnabled(True)

        if self._auto_scroll_check.isChecked() and was_at_bottom:
            self._scroll_to_bottom()
        else:
            # appendHtml can move the viewport; keep the user where they were reading
            scrollbar.setValue(min(previous_value, scrollbar.maximum()))

    def _scroll_to_bottom(self) -> None:
        scrollbar = self._log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def showEvent(self, event: QShowEvent) -> None:
        # Messages that arrived while hidden are only rendered once the tab is shown
        super().showEvent(event)
        if self._pending:
            self._flush_timer.start()

    def _clear_display(self) -> None:
        self._pending.clear()
        self._log_display.clear()
        self._log_entries.clear()

    def _open_log_folder(self) -> None:
        log_dir = self._logger.get_log_directory()

        try:
            if sys.platform == 'win32':
                subprocess.Popen(['explorer', str(log_dir)])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', str(log_dir)])
            else:
                subprocess.Popen(['xdg-open', str(log_dir)])

            self._logger.info(f"Opened log folder: {log_dir}")
        except Exception as e:
            self._logger.error(f"Failed to open log folder: {e}")

    def _unregister_from_logger(self) -> None:
        self._flush_timer.stop()
        self._logger.unregister_callback(self._on_log_message)
