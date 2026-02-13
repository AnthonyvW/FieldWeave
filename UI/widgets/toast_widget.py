"""
Toast notification widget for Forge microscope application.

Provides temporary, color-coded notifications that stack and auto-dismiss.
Integrates with the logging system for consistent message handling.
"""

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QPushButton, QProgressBar
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Property, QElapsedTimer
from PySide6.QtGui import QFont
from enum import Enum
from typing import Optional


class ToastType(Enum):
    """Toast notification types with associated colors."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class Toast(QFrame):
    """Individual toast notification widget."""
    
    # Color schemes for each toast type (background, border, progress bar)
    COLORS = {
        ToastType.INFO: ("#E3F2FD", "#1976D2", "#1976D2"),
        ToastType.SUCCESS: ("#E8F5E9", "#388E3C", "#388E3C"),
        ToastType.WARNING: ("#FFF3E0", "#F57C00", "#F57C00"),
        ToastType.ERROR: ("#FFEBEE", "#D32F2F", "#D32F2F"),
    }
    
    # Titles for each toast type
    TITLES = {
        ToastType.INFO: "Information",
        ToastType.SUCCESS: "Success",
        ToastType.WARNING: "Warning",
        ToastType.ERROR: "Error",
    }
    
    def __init__(self, message: str, toast_type: ToastType = ToastType.INFO, 
                 duration: int = 3000, title: str = None, parent: Optional[QWidget] = None):
        """
        Initialize a toast notification.
        
        Args:
            message: Description text to display
            toast_type: Type of toast (INFO, SUCCESS, WARNING, ERROR)
            duration: Duration in milliseconds before auto-dismiss (0 = no auto-dismiss)
            title: Optional custom title (defaults to toast type name)
            parent: Parent widget
        """
        super().__init__(parent)
        self.message = message
        self.toast_type = toast_type
        self.duration = duration
        self.title = title if title is not None else self.TITLES[toast_type]
        self._opacity = 1.0
        self._progress_value = 100
        self._start_time = None
        
        self._setup_ui()
        self._setup_animations()
        self._setup_progress_timer()
        
        # Auto-dismiss timer
        if duration > 0:
            QTimer.singleShot(duration, self.dismiss)
    
    def _setup_ui(self):
        """Setup the toast UI with appropriate styling."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        
        # Get colors for this toast type
        bg_color, border_color, progress_color = self.COLORS[self.toast_type]
        
        # Apply styling - no padding so progress bar can span full width
        self.setStyleSheet(f"""
            Toast {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 0px;
                padding: 0px;
            }}
        """)
        
        # Main layout - no margins so progress bar spans full width
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Content container (with padding for text)
        content_widget = QWidget()
        content_widget.setStyleSheet("background: transparent; border: none;")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(12, 10, 12, 10)  # Increased from 8,6,8,6
        content_layout.setSpacing(6)  # Increased from 4
        
        # Header row: Title and Close button
        header_layout = QHBoxLayout()
        header_layout.setSpacing(6)
        
        # Title label - no color styling
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("background: transparent; border: none; font-weight: bold;")
        self.title_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header_layout.addWidget(self.title_label, 1)
        
        # Close button - no hover effect
        self.close_button = QPushButton("×")
        self.close_button.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                font-size: 18px;
                font-weight: bold;
                padding: 0px;
                margin: 0px;
            }}
        """)
        self.close_button.setFixedSize(18, 18)
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_button.clicked.connect(self.dismiss)
        header_layout.addWidget(self.close_button)
        
        content_layout.addLayout(header_layout)
        
        # Message/Description label - no color styling
        self.message_label = QLabel(self.message)
        self.message_label.setStyleSheet("background: transparent; border: none;")
        self.message_label.setWordWrap(True)
        self.message_label.setFont(QFont("Segoe UI", 9))
        content_layout.addWidget(self.message_label)
        
        # Add content widget to main layout
        layout.addWidget(content_widget)
        
        # Progress bar at the very bottom edge - spans full width
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: rgba(0, 0, 0, 0.1);
                border: none;
                border-radius: 0px;
                margin: 0px;
            }}
            QProgressBar::chunk {{
                background-color: {progress_color};
                border-radius: 0px;
            }}
        """)
        layout.addWidget(self.progress_bar)
        
        # Set size constraints
        self.setMinimumWidth(260)
        self.setMaximumWidth(350)
        self.adjustSize()
    
    def _setup_progress_timer(self):
        """Setup timer to update progress bar."""
        if self.duration > 0:
            # Use QElapsedTimer for precise timing
            self.elapsed_timer = QElapsedTimer()
            self.elapsed_timer.start()
            
            # Update progress every 16ms for smooth 60fps animation
            self.progress_timer = QTimer(self)
            self.progress_timer.timeout.connect(self._update_progress)
            self.progress_timer.start(16)
        else:
            # No duration, hide progress bar
            self.progress_bar.hide()
    
    def _update_progress(self):
        """Update the progress bar based on elapsed time."""
        elapsed_ms = self.elapsed_timer.elapsed()
        
        if elapsed_ms >= self.duration:
            # Ensure we end at exactly 0%
            self.progress_bar.setValue(0)
            self.progress_timer.stop()
        else:
            # Calculate remaining percentage
            remaining_percent = int(((self.duration - elapsed_ms) / self.duration) * 100)
            self.progress_bar.setValue(remaining_percent)
    
    def _setup_animations(self):
        """Setup fade in/out animations."""
        # Fade in animation
        self.fade_in_animation = QPropertyAnimation(self, b"opacity")
        self.fade_in_animation.setDuration(200)
        self.fade_in_animation.setStartValue(0.0)
        self.fade_in_animation.setEndValue(1.0)
        self.fade_in_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        # Fade out animation
        self.fade_out_animation = QPropertyAnimation(self, b"opacity")
        self.fade_out_animation.setDuration(200)
        self.fade_out_animation.setStartValue(1.0)
        self.fade_out_animation.setEndValue(0.0)
        self.fade_out_animation.setEasingCurve(QEasingCurve.Type.InCubic)
        self.fade_out_animation.finished.connect(self._on_fade_out_finished)
    
    def show_animated(self):
        """Show the toast with fade-in animation."""
        self.show()
        self.fade_in_animation.start()
    
    def dismiss(self):
        """Dismiss the toast with fade-out animation."""
        # Stop progress timer if it exists
        if hasattr(self, 'progress_timer') and self.progress_timer.isActive():
            self.progress_timer.stop()
        self.fade_out_animation.start()
    
    def _on_fade_out_finished(self):
        """Called when fade-out animation completes."""
        self.hide()
        self.deleteLater()
    
    def _get_opacity(self):
        """Get current opacity value."""
        return self._opacity
    
    def _set_opacity(self, value):
        """Set opacity value and update window opacity."""
        self._opacity = value
        self.setWindowOpacity(value)
    
    opacity = Property(float, _get_opacity, _set_opacity)


class ToastManager(QWidget):
    """
    Manages multiple toast notifications in a stack.
    
    Toasts appear in the bottom-right corner and stack vertically upward.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the toast manager.
        
        Args:
            parent: Parent widget (typically the main window)
        """
        super().__init__(parent)
        self.parent_widget = parent
        self.toasts = []
        
        # Setup container
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | 
                           Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)  # Allow mouse events
        
        # Layout for stacking toasts (bottom to top) - reduced spacing
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(6)  # Reduced from 10 to 6
        self.layout.setAlignment(Qt.AlignmentFlag.AlignBottom)  # Align to bottom
        
        # Install event filter on parent to track moves
        if self.parent_widget:
            self.parent_widget.installEventFilter(self)
        
        # Position and show
        self._update_position()
        self.show()
    
    def eventFilter(self, obj, event):
        """Track parent window moves to reposition toasts."""
        if obj == self.parent_widget:
            # Update position on move or resize
            if event.type() in (event.Type.Move, event.Type.Resize):
                self._update_position()
        return super().eventFilter(obj, event)
    
    def _update_position(self):
        """Update position to bottom-right corner of parent."""
        if self.parent_widget:
            # Get the parent widget's geometry in global screen coordinates
            parent_global_rect = self.parent_widget.geometry()
            parent_pos = self.parent_widget.pos()
            
            # For QMainWindow, we need to get the actual screen position
            if hasattr(self.parent_widget, 'frameGeometry'):
                parent_global_rect = self.parent_widget.frameGeometry()
                parent_pos = parent_global_rect.topLeft()
            else:
                # Map parent position to global coordinates
                parent_pos = self.parent_widget.mapToGlobal(parent_pos)
            
            margin = 10  # Reduced from 20 to 10
            toast_width = 350  # Reduced from 420 to 350
            toast_container_height = 600  # Max height for toast container
            
            # Calculate position for bottom-right corner
            x = parent_pos.x() + parent_global_rect.width() - toast_width - margin
            y = parent_pos.y() + parent_global_rect.height() - toast_container_height - margin
            
            self.setGeometry(
                x,
                y,
                toast_width,
                toast_container_height
            )
    
    def show_toast(self, message: str, toast_type: ToastType = ToastType.INFO, 
                   duration: int = 3000, title: str = None):
        """
        Show a new toast notification.
        
        Args:
            message: Description text to display
            toast_type: Type of toast (INFO, SUCCESS, WARNING, ERROR)
            duration: Duration in milliseconds before auto-dismiss
            title: Optional custom title (defaults to toast type name)
        """
        # Create new toast
        toast = Toast(message, toast_type, duration, title, self)
        
        # Add to layout and list
        self.layout.addWidget(toast)
        self.toasts.append(toast)
        
        # Show with animation
        toast.show_animated()
        
        # Connect deletion signal to cleanup
        toast.destroyed.connect(lambda: self._remove_toast(toast))
        
        # Update position
        self._update_position()
    
    def _remove_toast(self, toast: Toast):
        """Remove toast from tracking list."""
        if toast in self.toasts:
            self.toasts.remove(toast)
    
    def info(self, message: str, duration: int = 3000, title: str = None):
        """Show an info toast."""
        self.show_toast(message, ToastType.INFO, duration, title)
    
    def success(self, message: str, duration: int = 3000, title: str = None):
        """Show a success toast."""
        self.show_toast(message, ToastType.SUCCESS, duration, title)
    
    def warning(self, message: str, duration: int = 4000, title: str = None):
        """Show a warning toast (slightly longer duration)."""
        self.show_toast(message, ToastType.WARNING, duration, title)
    
    def error(self, message: str, duration: int = 5000, title: str = None):
        """Show an error toast (longer duration)."""
        self.show_toast(message, ToastType.ERROR, duration, title)
    
    def clear_all(self):
        """Dismiss all active toasts."""
        for toast in self.toasts[:]:  # Copy list to avoid modification during iteration
            toast.dismiss()


# Convenience function for standalone usage
def show_toast(parent: QWidget, message: str, toast_type: ToastType = ToastType.INFO,
               duration: int = 3000, title: str = None):
    """
    Show a toast notification (convenience function).
    
    Args:
        parent: Parent widget
        message: Description text to display
        toast_type: Type of toast
        duration: Duration in milliseconds
        title: Optional custom title
    """
    if not hasattr(parent, '_toast_manager'):
        parent._toast_manager = ToastManager(parent)
    
    parent._toast_manager.show_toast(message, toast_type, duration, title)