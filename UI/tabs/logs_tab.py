from __future__ import annotations

import subprocess
import sys
import re
from datetime import datetime

from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QLabel,
)

from common.logger import get_logger


class LogsTab(QWidget):
    """Logs tab showing application logs with controls"""
    
    def __init__(self) -> None:
        super().__init__()
        
        # Log level filters (DEBUG disabled by default)
        self._level_filters = {
            'DEBUG': False,
            'INFO': True,
            'WARNING': True,
            'ERROR': True,
            'CRITICAL': True,
        }
        
        # Store all log entries that have been received
        self._log_entries = []
        
        # Track if we've done initial load
        self._initial_load_done = False
        
        # Log display
        self._log_display = QTextEdit()
        self._log_display.setReadOnly(True)
        self._log_display.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #000000;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                border: 1px solid #cccccc;
            }
        """)
        
        # Buttons
        self._clear_btn = QPushButton("Clear Display")
        self._clear_btn.clicked.connect(self._clear_display)
        
        self._open_folder_btn = QPushButton("Open Log Folder")
        self._open_folder_btn.clicked.connect(self._open_log_folder)
        
        # Auto-scroll checkbox
        self._auto_scroll_check = QCheckBox("Auto-scroll")
        self._auto_scroll_check.setChecked(True)
        
        # Log level filter checkboxes
        self._level_checkboxes = {}
        
        # Control layout - all on one line
        control_layout = QHBoxLayout()
        control_layout.addWidget(self._clear_btn)
        control_layout.addWidget(self._open_folder_btn)
        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Show levels:"))
        
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            checkbox = QCheckBox(level)
            checkbox.setChecked(self._level_filters[level])
            checkbox.stateChanged.connect(lambda state, lvl=level: self._on_filter_changed(lvl, state))
            self._level_checkboxes[level] = checkbox
            control_layout.addWidget(checkbox)
        
        control_layout.addStretch()
        control_layout.addWidget(self._auto_scroll_check)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.addWidget(self._log_display, 1)
        layout.addLayout(control_layout)
        
        # Register with logger
        self._logger = get_logger()
        self._logger.register_callback(self._on_log_message)
        
        # Load existing logs from current log file (one time only)
        self._load_existing_logs()
        self._initial_load_done = True
    
    def _load_existing_logs(self):
        """Load existing logs from the current log file (one time only at startup)"""
        try:
            current_log_file = self._logger.get_current_log_file()
            if not current_log_file or not current_log_file.exists():
                return
            
            # Read and parse existing log file
            # Format: [2025-01-26 14:30:45] INFO: Message
            log_pattern = re.compile(r'\[([^\]]+)\]\s+(\w+):\s+(.*)')
            
            with open(current_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip()
                    if not line:
                        continue
                    
                    match = log_pattern.match(line)
                    if match:
                        timestamp, level, message = match.groups()
                        
                        # Store the log entry
                        self._log_entries.append({
                            'timestamp': timestamp,
                            'level': level,
                            'message': message
                        })
                        
                        # Apply level filter and display
                        if self._level_filters.get(level, True):
                            color = self._get_level_color(level)
                            formatted = f'<span style="color: #666666;">[{timestamp}]</span> <span style="color: {color};">[{level}]</span> {self._escape_html(message)}'
                            self._log_display.append(formatted)
                    else:
                        # Line doesn't match pattern, show as-is (might be multiline continuation)
                        self._log_display.append(self._escape_html(line))
            
            # Auto-scroll to bottom
            scrollbar = self._log_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
        except Exception as e:
            self._log_display.append(f"Error loading existing logs: {e}")
    
    def _on_filter_changed(self, level: str, state: int):
        """Handle log level filter checkbox change"""
        self._level_filters[level] = bool(state)
        # Redisplay logs from memory with new filter
        self._redisplay_logs()
    
    def _redisplay_logs(self):
        """Redisplay all logs from memory with current filters"""
        self._log_display.clear()
        
        for entry in self._log_entries:
            level = entry['level']
            
            # Apply level filter
            if not self._level_filters.get(level, True):
                continue
            
            # Format with color
            color = self._get_level_color(level)
            formatted = f'<span style="color: #666666;">[{entry["timestamp"]}]</span> <span style="color: {color};">[{level}]</span> {self._escape_html(entry["message"])}'
            self._log_display.append(formatted)
        
        # Auto-scroll to bottom
        scrollbar = self._log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _on_log_message(self, level: str, message: str):
        """
        Handle incoming log message.
        This is called from the logger for each message.
        """
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Store the log entry in memory
        self._log_entries.append({
            'timestamp': timestamp,
            'level': level,
            'message': message
        })
        
        # Apply level filter
        if not self._level_filters.get(level, True):
            return
        
        # Format with color based on level
        color = self._get_level_color(level)
        formatted = f'<span style="color: #666666;">[{timestamp}]</span> <span style="color: {color};">[{level}]</span> {self._escape_html(message)}'
        
        self._log_display.append(formatted)
        
        # Auto-scroll to bottom if enabled
        if self._auto_scroll_check.isChecked():
            scrollbar = self._log_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def _get_level_color(self, level: str) -> str:
        """Get color for log level"""
        colors = {
            'DEBUG': '#666666',
            'INFO': '#0066cc',
            'WARNING': '#cc6600',
            'ERROR': '#cc0000',
            'CRITICAL': '#990000',
        }
        return colors.get(level, '#000000')
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))
    
    def _clear_display(self):
        """Clear the log display and memory"""
        self._log_display.clear()
        self._log_entries.clear()
    
    def _open_log_folder(self):
        """Open the log folder in file explorer"""
        log_dir = self._logger.get_log_directory()
        
        try:
            if sys.platform == 'win32':
                # Windows
                subprocess.Popen(['explorer', str(log_dir)])
            elif sys.platform == 'darwin':
                # macOS
                subprocess.Popen(['open', str(log_dir)])
            else:
                # Linux
                subprocess.Popen(['xdg-open', str(log_dir)])
            
            self._logger.info(f"Opened log folder: {log_dir}")
        except Exception as e:
            self._logger.error(f"Failed to open log folder: {e}")
    
    def closeEvent(self, event):
        """Unregister from logger when widget closes"""
        self._logger.unregister_callback(self._on_log_message)
        super().closeEvent(event)
