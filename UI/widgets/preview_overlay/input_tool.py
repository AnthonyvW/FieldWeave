from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QKeyEvent, QMouseEvent


@dataclass
class InputContext:
    """Geometry an InputTool needs to interpret an event — built fresh by OverlayLabel for each dispatch."""

    widget_rect: QRect
    display_rect: QRect
    has_pixmap: bool
    has_content: bool
    full_width: int
    full_height: int


class InputTool:
    """
    One coherent mouse/keyboard gesture handler for the camera preview —
    measurement placement, endpoint drag, tag hover/click, zoom-pan,
    click-to-move, etc. OverlayLabel forwards raw Qt events to a
    ToolDispatcher, which tries each registered tool in registration
    order; the first one whose handler returns True has claimed the
    event and no later tool sees it. Adding a new interactive tool means
    writing a subclass of this and registering it with the dispatcher —
    no edits to OverlayLabel itself.

    Override only the handlers a tool actually cares about; the rest
    default to "didn't claim this event."
    """

    def handle_press(self, event: QMouseEvent, ctx: InputContext) -> bool:
        return False

    def handle_move(self, event: QMouseEvent, ctx: InputContext) -> bool:
        return False

    def handle_release(self, event: QMouseEvent, ctx: InputContext) -> bool:
        return False

    def handle_key(self, event: QKeyEvent, ctx: InputContext) -> bool:
        return False


class DragGestureRecognizer:
    """
    Tracks a candidate click-or-drag from press to release using one
    shared threshold, instead of every gesture that needs click-vs-drag
    disambiguation (measurement placement, zoom pan) re-implementing it.
    """

    THRESHOLD_PX = 4

    def __init__(self) -> None:
        self._press_pos: QPoint | None = None
        self._dragging: bool = False

    @property
    def pending(self) -> bool:
        """True from begin() until end() — a press has happened but not yet resolved to a click or an in-progress drag."""
        return self._press_pos is not None

    @property
    def dragging(self) -> bool:
        return self._dragging

    @property
    def press_pos(self) -> QPoint | None:
        return self._press_pos

    def begin(self, pos: QPoint) -> None:
        self._press_pos = pos
        self._dragging = False

    def crossed_threshold(self, pos: QPoint) -> bool:
        """Call on move while pending. Returns True exactly once — the move that first crosses the drag threshold."""
        if self._press_pos is None or self._dragging:
            return False
        delta = pos - self._press_pos
        if abs(delta.x()) > self.THRESHOLD_PX or abs(delta.y()) > self.THRESHOLD_PX:
            self._dragging = True
            return True
        return False

    def end(self) -> None:
        """Call on release, after reading .dragging/.press_pos for one last time. Resets unconditionally."""
        self._press_pos = None
        self._dragging = False


class ToolDispatcher:
    """
    Owned by OverlayLabel. Holds an ordered list of InputTool instances;
    for each event, tries them in registration order and stops at the
    first one that claims it. Registration order is the gesture-priority
    ordering — declared once as a list instead of derived from reading a
    long if/elif chain.
    """

    def __init__(self) -> None:
        self._tools: list[InputTool] = []

    def register(self, tool: InputTool) -> None:
        self._tools.append(tool)

    def mouse_press(self, event: QMouseEvent, ctx: InputContext) -> bool:
        return any(tool.handle_press(event, ctx) for tool in self._tools)

    def mouse_move(self, event: QMouseEvent, ctx: InputContext) -> bool:
        return any(tool.handle_move(event, ctx) for tool in self._tools)

    def mouse_release(self, event: QMouseEvent, ctx: InputContext) -> bool:
        return any(tool.handle_release(event, ctx) for tool in self._tools)

    def key_press(self, event: QKeyEvent, ctx: InputContext) -> bool:
        return any(tool.handle_key(event, ctx) for tool in self._tools)
