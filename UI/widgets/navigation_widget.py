from __future__ import annotations

import math
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QGroupBox
)
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtCore import Qt, QRectF, QTimer, QEvent

from common.app_context import get_app_context
from common.logger import warning
from motion.motion_controller_manager import MotionState


def clamp(v: int, lo: int = 0, hi: int = 255) -> int:
    return max(lo, min(hi, v))


def adjust_color(c: QColor, factor: float) -> QColor:
    return QColor(
        clamp(int(c.red() * factor)),
        clamp(int(c.green() * factor)),
        clamp(int(c.blue() * factor)),
    )


class DiamondButton(QPushButton):
    def __init__(
        self,
        label: str = "",
        parent: QWidget | None = None,
        base_color: QColor = QColor(208, 211, 214),
        font_px: int = 28,
        size: int = 90,
        text_offset_y: int = 0
    ):
        super().__init__("", parent)
        self.setFixedSize(size, size)
        self.setStyleSheet("border: none; background: transparent;")
        self.setMouseTracking(True)

        # Enable mouse tracking on parent to handle pass-through
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        self._base = QColor(base_color)
        self._hover: bool = False
        self._label = label
        self._font_px = font_px
        self._text_offset_y = text_offset_y

    @property
    def hover(self) -> bool:
        return self._hover

    @hover.setter
    def hover(self, value: bool) -> None:
        if self._hover != value:
            self._hover = value
            self.update()

    def enterEvent(self, event):
        # Don't automatically set hover - check in mouseMoveEvent
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hover = False
        self.unsetCursor()
        super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        # Update hover state based on whether mouse is over diamond
        is_over_diamond = self.hitButton(event.position().toPoint())
        self.hover = is_over_diamond

        # Set or unset cursor based on position
        if is_over_diamond:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()

        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        # Check if click is inside diamond
        if self.hitButton(event.position().toPoint()):
            super().mousePressEvent(event)
        else:
            # Pass the event to the parent by ignoring it
            event.ignore()

    def mouseReleaseEvent(self, event):
        if self.hitButton(event.position().toPoint()):
            super().mouseReleaseEvent(event)
        else:
            event.ignore()

    def hitButton(self, pos):
        w = self.width()
        h = self.height()

        cx = w / 2
        cy = h / 2

        # Translate point to origin
        x = pos.x() - cx
        y = pos.y() - cy

        # Rotate point by -45 degrees (inverse of the 45 degree rotation in paintEvent)
        angle = -45 * math.pi / 180
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        rotated_x = x * cos_a - y * sin_a
        rotated_y = x * sin_a + y * cos_a

        # Check if the rotated point is inside the square
        side = min(w, h) / math.sqrt(2)
        half_side = side / 2

        return (abs(rotated_x) <= half_side and abs(rotated_y) <= half_side)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = self.width()
        h = self.height()

        color = QColor(self._base)
        if self._hover:
            color = adjust_color(color, 0.90)  # Darken on hover
        if self.isDown():
            color = adjust_color(color, 0.85)

        # Move origin to center
        painter.translate(w / 2, h / 2)

        # Rotate 45 degrees
        painter.rotate(45)

        # Define square centered at origin
        side = min(w, h) / math.sqrt(2)  # scale factor controls diamond size
        rect = QRectF(-side / 2, -side / 2, side, side)

        # Fill
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(rect)

        # Border
        pen = QPen(QColor(120, 120, 120))
        pen.setWidth(2)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        # Reset transform for text
        painter.resetTransform()

        # Draw label normally with offset
        font = self.font()
        font.setPixelSize(self._font_px)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.black)

        # Apply vertical offset to text rect
        text_rect = self.rect()
        if self._text_offset_y != 0:
            text_rect = text_rect.adjusted(
                0, self._text_offset_y, 0, self._text_offset_y)

        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self._label)


class NavigationWidget(QWidget):
    # Shared step size across all instances
    _shared_step_size: float = 0.4
    _instances: list[NavigationWidget] = []

    @property
    def _step_size(self) -> float:
        return NavigationWidget._shared_step_size

    @_step_size.setter
    def _step_size(self, value: float) -> None:
        NavigationWidget._shared_step_size = value

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        NavigationWidget._instances.append(self)
        self.destroyed.connect(
            lambda: NavigationWidget._instances.remove(self))

        # Declare all widget attributes here so they are always defined in __init__
        self.top_btn: DiamondButton
        self.bot_btn: DiamondButton
        self.left_btn: DiamondButton
        self.right_btn: DiamondButton
        self.center_btn: DiamondButton
        self.z_up_btn: QPushButton
        self.z_down_btn: QPushButton
        self.position_label: QLabel
        self.step_buttons: list[tuple[QPushButton, float]]

        self._setup_ui()

        # Overlay is created lazily in showEvent once the widget has a parent,
        # so we can reparent it to the grandparent and cover its layout margins.
        self._overlay: QWidget | None = None
        self._overlay_label: QLabel | None = None
        self._motion_available: bool = False

        # Poll until the controller is ready, then switch to a position-refresh timer
        self._ready_timer = QTimer(self)
        self._ready_timer.setInterval(500)
        self._ready_timer.timeout.connect(self._check_motion_ready)
        self._ready_timer.start()

        self._position_timer = QTimer(self)
        self._position_timer.setInterval(200)
        self._position_timer.timeout.connect(self._update_position_display)

        # Start in the unavailable state
        self._set_motion_available(False)

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Set white background
        self.setStyleSheet("""
            NavigationWidget {
                background: white;
            }
        """)

        # Step size controls
        step_size_controls = self._create_step_size_controls()
        main_layout.addWidget(step_size_controls)

        # Combined jog controls
        jog_controls = self._create_jog_controls()
        main_layout.addWidget(jog_controls)

        main_layout.addStretch(1)

    def _ensure_overlay(self) -> None:
        """
        Create the overlay the first time we are shown with a real parent.
        Parenting to self.parent() (the CollapsibleSection content widget) means
        Qt will not clip the overlay to NavigationWidget's own bounds, so we can
        expand it to cover the layout margins that surround us.
        """
        if self._overlay is not None:
            return
        overlay_parent = self.parent()
        if overlay_parent is None:
            return  # Not yet placed in a hierarchy — try again on next showEvent

        self._overlay = QWidget(overlay_parent)
        self._overlay.setStyleSheet("background: rgba(0, 0, 0, 0);")
        self._overlay.show()

        self._overlay_label = QLabel(
            "Motion System Not Connected", self._overlay)
        self._overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18px;
                font-style: italic;
                background: transparent;
            }
        """)

        # Apply whatever state we already decided on
        self._set_motion_available(self._motion_available)

    def _reposition_overlay(self) -> None:
        """
        Position the overlay in its parent's coordinate space so it covers
        NavigationWidget plus the layout margins the parent applies around it.
        """
        if self._overlay is None:
            return
        overlay_parent = self._overlay.parent()
        if overlay_parent is None:
            return

        # Where does NavigationWidget sit inside overlay_parent?
        origin = self.mapTo(overlay_parent, self.rect().topLeft())
        x, y = origin.x(), origin.y()
        w, h = self.width(), self.height()

        # Expand outward by the margins the parent layout adds around us
        parent = self.parent()
        if parent is not None:
            layout = parent.layout()
            if layout is not None:
                left, top, right, bottom = layout.getContentsMargins()
                x -= left
                y -= top
                w += left + right
                h += top + bottom

        self._overlay.setGeometry(x, y, w, h)
        self._overlay_label.setGeometry(
            0, 0, self._overlay.width(), self._overlay.height())
        self._overlay.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_overlay()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._ensure_overlay()
        self._reposition_overlay()

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        self._reposition_overlay()

    def _check_motion_ready(self) -> None:
        """Poll every 500 ms until the controller is ready or has reached a terminal state."""
        ctx = get_app_context()
        if ctx.motion is None:
            return
        state = ctx.motion.get_state()
        if state == MotionState.READY:
            self._ready_timer.stop()
            self._set_motion_available(True)
            self._position_timer.start()
        elif state in (MotionState.FAILED, MotionState.FAULTED):
            self._ready_timer.stop()  # Terminal failure — stay overlaid

    def _set_motion_available(self, available: bool) -> None:
        """Show or hide the tinted overlay without touching button enabled state.

        Interaction is blocked by raising an opaque overlay that absorbs all
        mouse events, so the buttons retain their normal appearance at all times.
        """
        self._motion_available = available
        if self._overlay is None:
            return
        if available:
            # Let mouse events fall through to the buttons beneath
            self._overlay.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self._overlay.setStyleSheet("background: rgba(0, 0, 0, 0);")
            self._overlay_label.hide()
            self._overlay.lower()
        else:
            # Overlay sits on top and absorbs all input — buttons stay visually unchanged
            self._overlay.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
            self._overlay.setStyleSheet("background: rgba(0, 0, 0, 100);")
            self._overlay_label.show()
            self._overlay.raise_()
            self.position_label.setText("X: --  Y: --  Z: -- mm")

    def _create_step_size_controls(self) -> QWidget:
        """Create step size selection buttons with position display"""

        group = QGroupBox("Step Size")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(10, 5, 10, 5)
        group_layout.setSpacing(10)

        # Buttons row
        buttons_row = QWidget()
        buttons_layout = QHBoxLayout(buttons_row)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(8)

        # Step size buttons
        step_sizes = [0.04, 0.4, 2.0, 10.0]
        self.step_buttons = []

        for size in step_sizes:
            btn = QPushButton(f"{size}mm")
            btn.setFixedHeight(30)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    padding: 0px;
                }
                QPushButton:checked {
                    background-color: rgb(140, 143, 146);
                    color: white;
                    border: 1px solid rgb(100, 103, 106);
                }
            """)
            btn.clicked.connect(lambda checked, s=size: self._set_step_size(s))
            buttons_layout.addWidget(btn)
            self.step_buttons.append((btn, size))

        # Reflect the current shared step size
        self.sync_step_size_buttons()

        group_layout.addWidget(buttons_row)

        # Position display
        self.position_label = QLabel()
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.position_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                padding: 2px;
            }
        """)
        self._update_position_display()
        group_layout.addWidget(self.position_label)

        return group

    def _set_step_size(self, size: float) -> None:
        """Set the step size on all instances and update button states"""
        NavigationWidget._shared_step_size = size

        for instance in NavigationWidget._instances:
            instance.sync_step_size_buttons()

        ctx = get_app_context()
        if ctx.motion is not None:
            ctx.motion.set_speed(round(size * 1_000_000))

    def sync_step_size_buttons(self) -> None:
        """Update button checked states to reflect the current shared step size."""
        size = NavigationWidget._shared_step_size
        for btn, btn_size in self.step_buttons:
            btn.setChecked(btn_size == size)

    def _create_jog_controls(self) -> QWidget:
        """Create combined jog controls with diamond navigation and Z-axis"""

        group = QGroupBox("Jog")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(0)

        # Top row: diamond and z-axis controls
        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(15)

        # Diamond panel
        diamond_container = self._create_diamond_panel()
        top_layout.addWidget(diamond_container, 0,
                             Qt.AlignmentFlag.AlignCenter)

        # Z-axis controls
        z_container = self._create_z_controls()
        top_layout.addWidget(z_container, 0, Qt.AlignmentFlag.AlignCenter)

        top_layout.addStretch(1)

        group_layout.addWidget(top_row)

        return group

    def _create_diamond_panel(self) -> QWidget:
        """Create the diamond navigation with home button in center"""
        container = QWidget()
        container.setFixedSize(240, 200)  # Slightly larger for better spacing

        # Outer arrows (Unicode) - larger size
        self.top_btn = DiamondButton(
            "↑", parent=container, font_px=32, size=90)
        self.left_btn = DiamondButton(
            "←", parent=container, font_px=32, size=90, text_offset_y=-3)
        self.right_btn = DiamondButton(
            "→", parent=container, font_px=32, size=90, text_offset_y=-3)
        self.bot_btn = DiamondButton(
            "↓", parent=container, font_px=32, size=90)

        # Center home icon - smaller and orange
        self.center_btn = DiamondButton(
            "H",
            parent=container,
            font_px=20,
            size=60  # Smaller than outer buttons
        )

        # Install event filters on all buttons for click pass-through
        for btn in [self.top_btn, self.left_btn, self.right_btn, self.bot_btn, self.center_btn]:
            btn.installEventFilter(self)

        # Connect buttons to placeholder functions
        self.top_btn.clicked.connect(self._move_up)
        self.left_btn.clicked.connect(self._move_left)
        self.right_btn.clicked.connect(self._move_right)
        self.bot_btn.clicked.connect(self._move_down)
        self.center_btn.clicked.connect(self._go_home)

        self.center_btn.raise_()

        # Position buttons when container is shown
        container.resizeEvent = lambda event: self._layout_diamond_buttons(
            container)

        return container

    def eventFilter(self, obj, event):
        """Intercept button events and pass through if in corner regions"""

        # Only filter events on DiamondButtons
        if not isinstance(obj, DiamondButton):
            return super().eventFilter(obj, event)

        # Handle mouse move events for hover
        if event.type() == QEvent.Type.MouseMove:
            btn_local_pos = event.position().toPoint()
            is_over_obj_diamond = obj.hitButton(btn_local_pos)
            global_pos = obj.mapToGlobal(btn_local_pos)

            # Define buttons in z-order (home button is on top)
            buttons = [self.center_btn, self.top_btn, self.left_btn,
                       self.right_btn, self.bot_btn]

            # Find which button should be hovered
            hovered_btn = None

            if is_over_obj_diamond:
                # Mouse is over this button's diamond - it should be hovered
                hovered_btn = obj
            else:
                # Mouse is in corner region - check buttons beneath
                for btn in buttons:
                    if btn is obj:
                        continue

                    btn_local = btn.mapFromGlobal(global_pos)
                    if btn.geometry().contains(btn.mapToParent(btn_local)):
                        if btn.hitButton(btn_local):
                            hovered_btn = btn
                            break  # Stop at first match (top-most button)

            # Update hover state on all buttons
            for btn in buttons:
                if btn is hovered_btn:
                    # Set hover on this button
                    if not btn.hover:
                        btn.hover = True
                        btn.setCursor(Qt.CursorShape.PointingHandCursor)
                else:
                    # Clear hover on all other buttons
                    if btn.hover:
                        btn.hover = False
                        btn.unsetCursor()

            return False  # Don't consume move events

        # Handle mouse button press events
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                # Check if click is actually inside the diamond shape
                btn_local_pos = event.position().toPoint()
                if not obj.hitButton(btn_local_pos):
                    # Click is in corner region - manually pass to buttons beneath
                    global_pos = obj.mapToGlobal(btn_local_pos)

                    # Define buttons in z-order (home button is on top)
                    buttons = [self.center_btn, self.top_btn, self.left_btn,
                               self.right_btn, self.bot_btn]

                    for btn in buttons:
                        if btn is obj:
                            continue  # Skip the button we're filtering

                        # Check if this button is beneath the click
                        btn_local = btn.mapFromGlobal(global_pos)
                        if btn.geometry().contains(btn.mapToParent(btn_local)):
                            if btn.hitButton(btn_local):
                                # Manually trigger this button (first match due to z-order)
                                btn.clicked.emit()
                                return True  # Consume the event

                    # No button beneath, just consume the event (don't trigger anything)
                    return True

        return super().eventFilter(obj, event)

    def _layout_diamond_buttons(self, container: QWidget) -> None:
        """Layout the diamond buttons in proper positions"""
        # Container center
        cx = container.width() // 2
        cy = container.height() // 2

        # Simple positioning: place outer buttons at fixed distance from center
        # Distance should be enough to have visible gap between buttons
        distance = 50  # Distance from center to outer button centers

        def place(btn: QPushButton, x: int, y: int) -> None:
            """Place button centered at x, y"""
            btn.move(x - btn.width() // 2, y - btn.height() // 2)

        # Place outer buttons in cardinal directions
        place(self.top_btn, cx, cy - distance)
        place(self.bot_btn, cx, cy + distance)
        place(self.left_btn, cx - distance, cy)
        place(self.right_btn, cx + distance, cy)

        # Place home button at center
        place(self.center_btn, cx, cy)

        self.center_btn.raise_()

    def _create_z_controls(self) -> QWidget:
        """Create Z-axis increase/decrease buttons"""
        container = QWidget()
        container.setFixedHeight(200)  # Match diamond container height
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Add stretch to center vertically
        layout.addStretch(1)

        # Increase button - smaller with border
        self.z_up_btn = QPushButton("▲")
        self.z_up_btn.setFixedSize(55, 55)  # Smaller square
        self.z_up_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(208, 211, 214);
                border: 2px solid rgb(120, 120, 120);
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: rgb(187, 190, 193);
            }
            QPushButton:pressed {
                background-color: rgb(177, 180, 182);
            }
        """)
        self.z_up_btn.clicked.connect(self._z_increase)
        layout.addWidget(self.z_up_btn, 0, Qt.AlignmentFlag.AlignCenter)

        # Decrease button - smaller with border
        self.z_down_btn = QPushButton("▼")
        self.z_down_btn.setFixedSize(55, 55)  # Smaller square
        self.z_down_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(208, 211, 214);
                border: 2px solid rgb(120, 120, 120);
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: rgb(187, 190, 193);
            }
            QPushButton:pressed {
                background-color: rgb(177, 180, 182);
            }
        """)
        self.z_down_btn.clicked.connect(self._z_decrease)
        layout.addWidget(self.z_down_btn, 0, Qt.AlignmentFlag.AlignCenter)

        # Add stretch to center vertically
        layout.addStretch(1)

        return container

    def _update_position_display(self) -> None:
        """Update the position display label from the live controller position."""
        ctx = get_app_context()
        if ctx.motion is None or not ctx.motion.is_ready():
            return
        x_mm, y_mm, z_mm = ctx.motion.get_position().to_mm()
        self.position_label.setText(
            f"X: {x_mm:.2f}  Y: {y_mm:.2f}  Z: {z_mm:.2f} mm"
        )

    # Placeholder movement functions
    def _move_up(self) -> None:
        """Move stage up (positive Y)"""
        ctx = get_app_context()
        if ctx.motion is None:
            warning("NavigationWidget: motion command ignored — controller not ready")
            return
        ctx.motion.move_axis("y", 1)
        self._update_position_display()

    def _move_down(self) -> None:
        """Move stage down (negative Y)"""
        ctx = get_app_context()
        if ctx.motion is None:
            warning("NavigationWidget: motion command ignored — controller not ready")
            return
        ctx.motion.move_axis("y", -1)
        self._update_position_display()

    def _move_left(self) -> None:
        """Move stage left (negative X)"""
        ctx = get_app_context()
        if ctx.motion is None:
            warning("NavigationWidget: motion command ignored — controller not ready")
            return
        ctx.motion.move_axis("x", -1)
        self._update_position_display()

    def _move_right(self) -> None:
        """Move stage right (positive X)"""
        ctx = get_app_context()
        if ctx.motion is None:
            warning("NavigationWidget: motion command ignored — controller not ready")
            return
        ctx.motion.move_axis("x", 1)
        self._update_position_display()

    def _go_home(self) -> None:
        """Return stage to home position"""
        ctx = get_app_context()
        if ctx.motion is None:
            warning("NavigationWidget: motion command ignored — controller not ready")
            return

        dialog = QMessageBox(self)
        dialog.setWindowTitle("Confirm Homing")
        dialog.setText("Are you sure you want to home the motion system?")
        dialog.setInformativeText(
            "Ensure the path is clear before continuing."
        )
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
        dialog.button(QMessageBox.StandardButton.Ok).setText("Yes")
        dialog.button(QMessageBox.StandardButton.Cancel).setText("No")

        if dialog.exec() != QMessageBox.StandardButton.Ok:
            return

        ctx.motion.home()
        self._update_position_display()

    def _z_increase(self) -> None:
        """Increase Z height"""
        ctx = get_app_context()
        if ctx.motion is None:
            warning("NavigationWidget: motion command ignored — controller not ready")
            return
        ctx.motion.move_axis("z", 1)
        self._update_position_display()

    def _z_decrease(self) -> None:
        """Decrease Z height"""
        ctx = get_app_context()
        if ctx.motion is None:
            warning("NavigationWidget: motion command ignored — controller not ready")
            return
        ctx.motion.move_axis("z", -1)
        self._update_position_display()
