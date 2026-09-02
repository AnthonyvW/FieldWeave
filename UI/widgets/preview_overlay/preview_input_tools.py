from __future__ import annotations

from typing import Callable, Protocol

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from UI.widgets.preview_overlay.click_to_move import ClickToMoveOverlay
from UI.widgets.preview_overlay.input_tool import DragGestureRecognizer, InputContext, InputTool
from UI.widgets.preview_overlay.measurement_interaction import MeasurementInteraction
from UI.widgets.preview_overlay.measurement_overlay import MeasurementOverlay
from UI.widgets.preview_overlay.zoom_preview import ZoomPreviewOverlay


class _Repaintable(Protocol):
    def update(self) -> None: ...


class MeasurementTagInteractionTool(InputTool):
    """Tag hover/click/delete for already-placed measurements — a thin adapter around MeasurementInteraction, which already owns this UI; see its own docstring."""

    def __init__(self, interaction: MeasurementInteraction, placement_pending: Callable[[], bool]) -> None:
        self._interaction = interaction
        self._placement_pending = placement_pending

    def handle_press(self, event: QMouseEvent, ctx: InputContext) -> bool:
        return self._interaction.handle_mouse_press(event)

    def handle_move(self, event: QMouseEvent, ctx: InputContext) -> bool:
        # A passive observer alongside whatever else the move turns out
        # to be — paused only while a measurement click-vs-drag is still
        # being disambiguated (see MeasurementPlacementTool.pending).
        if not self._placement_pending():
            self._interaction.handle_mouse_move(event)
        return False

    def handle_key(self, event: QKeyEvent, ctx: InputContext) -> bool:
        return self._interaction.handle_key_press(event)


class MeasurementEndpointDragTool(InputTool):
    """Dragging an already-placed measurement's endpoint to a new position."""

    def __init__(self, measurement: MeasurementOverlay, video_label: _Repaintable, active: Callable[[], bool]) -> None:
        self._measurement = measurement
        self._video_label = video_label
        self._active = active

    def handle_press(self, event: QMouseEvent, ctx: InputContext) -> bool:
        if not self._active() or self._measurement.in_progress or event.button() != Qt.MouseButton.LeftButton:
            return False
        return self._measurement.begin_endpoint_drag(event.position().toPoint(), ctx.display_rect, ctx.widget_rect)

    def handle_move(self, event: QMouseEvent, ctx: InputContext) -> bool:
        if not self._measurement.dragging_endpoint:
            return False
        self._measurement.update_endpoint_drag(event.position().toPoint(), ctx.widget_rect)
        self._video_label.update()
        return True

    def handle_release(self, event: QMouseEvent, ctx: InputContext) -> bool:
        if not self._measurement.dragging_endpoint:
            return False
        self._measurement.end_endpoint_drag()
        self._video_label.update()
        return True


class MeasurementPlacementTool(InputTool):
    """
    Placing a new measurement (or a manual-calibration line): the first
    click starts a draft, moves in between show a dashed preview, and
    further clicks add points until the kind's required count is
    reached. Right-click cancels an in-progress draft. A press that
    turns into a drag pans the zoomed view instead of adding a point,
    since both gestures start with the same mouse-down — see
    DragGestureRecognizer.
    """

    def __init__(
        self,
        measurement: MeasurementOverlay,
        zoom: ZoomPreviewOverlay,
        video_label: _Repaintable,
        active: Callable[[], bool],
    ) -> None:
        self._measurement = measurement
        self._zoom = zoom
        self._video_label = video_label
        self._active = active
        self._drag = DragGestureRecognizer()

    @property
    def pending(self) -> bool:
        """Whether a press is currently being disambiguated between a click and a pan-drag — read by MeasurementTagInteractionTool to pause hover while this is unresolved."""
        return self._drag.pending

    def reset(self) -> None:
        """Clear any in-progress click-vs-drag disambiguation — called when measurement mode is toggled off mid-gesture."""
        self._drag.end()

    def handle_press(self, event: QMouseEvent, ctx: InputContext) -> bool:
        if not self._active():
            return False
        if self._measurement.in_progress and event.button() == Qt.MouseButton.RightButton:
            self._measurement.cancel_placement()
            self._video_label.update()
            return True
        if self._measurement.drawing_enabled and event.button() == Qt.MouseButton.LeftButton:
            self._drag.begin(event.position().toPoint())
            return True
        return False

    def handle_move(self, event: QMouseEvent, ctx: InputContext) -> bool:
        if self._drag.pending:
            pos = event.position().toPoint()
            if self._drag.crossed_threshold(pos) and self._zoom.active:
                self._zoom.begin_drag(self._drag.press_pos)
            if self._drag.dragging and self._zoom.active and ctx.has_content:
                self._zoom.drag_to(pos, ctx.widget_rect)
                self._video_label.update()
            return True
        if self._measurement.in_progress:
            self._measurement.update_preview(event.position().toPoint(), ctx.widget_rect)
            self._video_label.update()
            return True
        return False

    def handle_release(self, event: QMouseEvent, ctx: InputContext) -> bool:
        if not self._drag.pending or event.button() != Qt.MouseButton.LeftButton:
            return False
        pos = event.position().toPoint()
        was_dragging = self._drag.dragging
        self._drag.end()
        if was_dragging:
            if self._zoom.active:
                self._zoom.end_drag()
        else:
            # A plain click (no drag): place_point itself knows whether
            # this starts a new draft, adds a point to one already in
            # progress, or (for "Point") finalizes immediately.
            self._measurement.place_point(pos, ctx.widget_rect)
            self._video_label.update()
        return True


class ZoomPanTool(InputTool):
    """
    Click-and-drag panning while zoomed in — and, since a plain click
    while zoomed has to be told apart from the start of a pan, this is
    also where a resulting non-drag click resolves to click-to-move
    (through the zoom viewport's own pixel mapping) once released.
    """

    def __init__(
        self,
        zoom: ZoomPreviewOverlay,
        click: ClickToMoveOverlay,
        video_label: _Repaintable,
        click_to_move_allowed: Callable[[], bool],
    ) -> None:
        self._zoom = zoom
        self._click = click
        self._video_label = video_label
        self._click_to_move_allowed = click_to_move_allowed
        self._drag = DragGestureRecognizer()

    def handle_press(self, event: QMouseEvent, ctx: InputContext) -> bool:
        if not self._zoom.active or event.button() != Qt.MouseButton.LeftButton:
            return False
        self._drag.begin(event.position().toPoint())
        return True

    def handle_move(self, event: QMouseEvent, ctx: InputContext) -> bool:
        if not (
            self._zoom.active
            and bool(event.buttons() & Qt.MouseButton.LeftButton)
            and self._drag.pending
            and ctx.has_content
        ):
            return False
        pos = event.position().toPoint()
        if self._drag.crossed_threshold(pos):
            self._zoom.begin_drag(self._drag.press_pos)
        if self._drag.dragging:
            self._zoom.drag_to(pos, ctx.widget_rect)
            self._video_label.update()
        return True

    def handle_release(self, event: QMouseEvent, ctx: InputContext) -> bool:
        if not self._zoom.active or event.button() != Qt.MouseButton.LeftButton:
            return False
        was_dragging = self._drag.dragging
        press_pos = self._drag.press_pos
        self._drag.end()
        if was_dragging:
            self._zoom.end_drag()
        elif (
            self._click.enabled
            and self._click_to_move_allowed()
            and press_pos is not None
            and ctx.has_pixmap
            and ctx.display_rect.contains(press_pos)
        ):
            full_pixel = self._zoom.widget_pos_to_full_pixel(press_pos, ctx.widget_rect)
            if full_pixel is not None:
                full_px, full_py, full_w, full_h = full_pixel
                ref = self._zoom.current_view_center_full_pixel(ctx.widget_rect)
                ref_x, ref_y = ref if ref is not None else (None, None)
                self._click.handle_full_pixel_click(full_px, full_py, full_w, full_h, ref_x, ref_y)
        return True


class ClickToMoveTool(InputTool):
    """Immediate click-to-move on the unzoomed live feed — while zoomed, ZoomPanTool claims the press instead and resolves click-to-move on release."""

    def __init__(self, click: ClickToMoveOverlay, allowed: Callable[[], bool]) -> None:
        self._click = click
        self._allowed = allowed

    def handle_press(self, event: QMouseEvent, ctx: InputContext) -> bool:
        if not (
            self._click.enabled
            and self._allowed()
            and event.button() == Qt.MouseButton.LeftButton
            and ctx.has_pixmap
        ):
            return False
        if ctx.full_width <= 0 or ctx.full_height <= 0:
            return False
        pos = event.position().toPoint()
        self._click.handle_click(pos.x(), pos.y(), ctx.display_rect, ctx.full_width, ctx.full_height)
        return True
