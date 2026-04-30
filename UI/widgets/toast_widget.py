from __future__ import annotations

from enum import Enum

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Property, QElapsedTimer, QRect
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ToastType(Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class Toast(QFrame):
    COLORS = {
        ToastType.INFO:    ("#E3F2FD", "#1976D2", "#1976D2"),
        ToastType.SUCCESS: ("#E8F5E9", "#388E3C", "#388E3C"),
        ToastType.WARNING: ("#FFF3E0", "#F57C00", "#F57C00"),
        ToastType.ERROR:   ("#FFEBEE", "#D32F2F", "#D32F2F"),
    }

    TITLES = {
        ToastType.INFO:    "Information",
        ToastType.SUCCESS: "Success",
        ToastType.WARNING: "Warning",
        ToastType.ERROR:   "Error",
    }

    def __init__(
        self,
        message: str,
        toast_type: ToastType = ToastType.INFO,
        duration: int = 3000,
        title: str | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent, Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.message = message
        self.toast_type = toast_type
        self.duration = duration
        self.title = title if title is not None else self.TITLES[toast_type]
        self._opacity = 1.0

        self._setup_ui()
        self._setup_animations()
        self._setup_progress_timer()

        if duration > 0:
            QTimer.singleShot(duration, self.dismiss)

    def _setup_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        bg_color, border_color, progress_color = self.COLORS[self.toast_type]

        self.setStyleSheet(f"""
            Toast {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 0px;
                padding: 0px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        content_widget = QWidget()
        content_widget.setStyleSheet("background: transparent; border: none;")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(12, 10, 12, 10)
        content_layout.setSpacing(6)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(6)

        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("background: transparent; border: none; font-weight: bold;")
        self.title_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header_layout.addWidget(self.title_label, 1)

        self.close_button = QPushButton("×")
        self.close_button.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                font-size: 18px;
                font-weight: bold;
                padding: 0px;
                margin: 0px;
            }
        """)
        self.close_button.setFixedSize(18, 18)
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_button.clicked.connect(self.dismiss)
        header_layout.addWidget(self.close_button)

        content_layout.addLayout(header_layout)

        self.message_label = QLabel(self.message)
        self.message_label.setStyleSheet("background: transparent; border: none;")
        self.message_label.setWordWrap(True)
        self.message_label.setFont(QFont("Segoe UI", 9))
        self.message_label.setMaximumWidth(326)
        content_layout.addWidget(self.message_label)

        layout.addWidget(content_widget)

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

        self.setFixedWidth(350)
        self.adjustSize()

    def _setup_progress_timer(self) -> None:
        if self.duration > 0:
            self.elapsed_timer = QElapsedTimer()
            self.elapsed_timer.start()
            self.progress_timer = QTimer(self)
            self.progress_timer.timeout.connect(self._update_progress)
            self.progress_timer.start(16)
        else:
            self.progress_bar.hide()

    def _update_progress(self) -> None:
        elapsed_ms = self.elapsed_timer.elapsed()
        if elapsed_ms >= self.duration:
            self.progress_bar.setValue(0)
            self.progress_timer.stop()
        else:
            remaining_percent = int(((self.duration - elapsed_ms) / self.duration) * 100)
            self.progress_bar.setValue(remaining_percent)

    def _setup_animations(self) -> None:
        self.fade_in_animation = QPropertyAnimation(self, b"opacity")
        self.fade_in_animation.setDuration(200)
        self.fade_in_animation.setStartValue(0.0)
        self.fade_in_animation.setEndValue(1.0)
        self.fade_in_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.fade_out_animation = QPropertyAnimation(self, b"opacity")
        self.fade_out_animation.setDuration(200)
        self.fade_out_animation.setStartValue(1.0)
        self.fade_out_animation.setEndValue(0.0)
        self.fade_out_animation.setEasingCurve(QEasingCurve.Type.InCubic)
        self.fade_out_animation.finished.connect(self._on_fade_out_finished)

    def show_animated(self) -> None:
        self.show()
        self.fade_in_animation.start()

    def dismiss(self) -> None:
        if hasattr(self, 'progress_timer') and self.progress_timer.isActive():
            self.progress_timer.stop()
        self.fade_out_animation.start()

    def _on_fade_out_finished(self) -> None:
        self.hide()
        self.deleteLater()

    def _get_opacity(self) -> float:
        return self._opacity

    def _set_opacity(self, value: float) -> None:
        self._opacity = value
        self.setWindowOpacity(value)

    opacity = Property(float, _get_opacity, _set_opacity)


class ToastManager:
    """
    Manages multiple toast notifications stacked in the bottom-right corner
    of the parent window. Each Toast is an independent top-level window so
    no invisible overlay can block mouse events on the parent.
    """

    MARGIN = 10
    SPACING = 6

    def __init__(self, parent: QWidget | None = None) -> None:
        self.parent_widget = parent
        self.toasts: list[Toast] = []

        if self.parent_widget:
            self.parent_widget.installEventFilter(self._make_event_filter())

    def _make_event_filter(self):
        manager = self

        from PySide6.QtCore import QObject

        class _Filter(QObject):
            def eventFilter(self, obj: object, event: object) -> bool:
                if event.type() in (event.Type.Move, event.Type.Resize):
                    manager._reposition_all()
                return False

        self._filter = _Filter(self.parent_widget)
        return self._filter

    def _parent_bottom_right(self) -> tuple[int, int]:
        if not self.parent_widget:
            return (0, 0)
        r = self.parent_widget.frameGeometry()
        return (r.x() + r.width(), r.y() + r.height())

    def _reposition_all(self) -> None:
        right_x, bottom_y = self._parent_bottom_right()
        y = bottom_y - self.MARGIN
        for toast in reversed(self.toasts):
            toast.adjustSize()
            h = toast.height()
            x = right_x - toast.width() - self.MARGIN
            toast.setGeometry(QRect(x, y - h, toast.width(), h))
            y -= h + self.SPACING

    def show_toast(
        self,
        message: str,
        toast_type: ToastType = ToastType.INFO,
        duration: int = 3000,
        title: str | None = None,
    ) -> None:
        toast = Toast(message, toast_type, duration, title, self.parent_widget)
        self.toasts.append(toast)
        toast.destroyed.connect(lambda: self._remove_toast(toast))
        self._reposition_all()
        toast.show_animated()

    def _remove_toast(self, toast: Toast) -> None:
        if toast in self.toasts:
            self.toasts.remove(toast)
        self._reposition_all()

    def info(self, message: str, duration: int = 3000, title: str | None = None) -> None:
        self.show_toast(message, ToastType.INFO, duration, title)

    def success(self, message: str, duration: int = 3000, title: str | None = None) -> None:
        self.show_toast(message, ToastType.SUCCESS, duration, title)

    def warning(self, message: str, duration: int = 4000, title: str | None = None) -> None:
        self.show_toast(message, ToastType.WARNING, duration, title)

    def error(self, message: str, duration: int = 5000, title: str | None = None) -> None:
        self.show_toast(message, ToastType.ERROR, duration, title)

    def clear_all(self) -> None:
        for toast in self.toasts[:]:
            toast.dismiss()