from __future__ import annotations

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QToolTip, QWidget

from UI.widgets.measurements.measurement_meta import MeasurementMeta
from UI.widgets.preview_overlay.measurement_customize_menu import MeasurementCustomizeMenu
from UI.widgets.preview_overlay.measurement_overlay import MeasurementOverlay


class MeasurementInteraction:
    """
    Everything about turning raw preview input into this one overlay's
    own UI: tag hover/click/delete, anchor-point proximity, and the
    customize-menu popup's lifecycle, pan/zoom tracking, and live-
    preview/apply/cancel wiring back into *overlay*.

    OverlayLabel's event handlers just forward their raw Qt events here
    and act on the bool they get back; CameraPreview constructs this
    once and never touches MeasurementCustomizeMenu or MeasurementOverlay's
    hover/proximity/tag state directly. That keeps both of those about
    driving the preview widget and its overlays in general, rather than
    accumulating one overlay's own UI as an afterthought.

    *video_label* needs only a small public surface: ``display_rect()``,
    ``rect()``, ``mapTo()``, and ``update()`` — see the ``_geometry``/
    ``_tag_anchor_point`` helpers below.
    """

    def __init__(self, overlay: MeasurementOverlay, video_label: QWidget, popup_parent: QWidget) -> None:
        self._overlay = overlay
        self._video_label = video_label
        self._popup_parent = popup_parent
        self._active = False
        self._popup_tracking_index: int | None = None

        self._menu = MeasurementCustomizeMenu(popup_parent)
        self._menu.applied.connect(self._on_meta_changed)
        self._menu.applied.connect(self._on_menu_closed)
        self._menu.preview_changed.connect(self._on_meta_changed)
        self._menu.cancelled.connect(self._on_menu_closed)

    def set_active(self, active: bool) -> None:
        """
        Called from OverlayLabel.set_measurement_mode_active — tag
        hover/tooltip state and an open popup shouldn't survive
        switching away from the measurement tab.

        Checks the menu's own tracked index rather than ``isVisible()``:
        by the time this runs on a tab switch, CameraPreview (the
        menu's parent) has typically already been hidden by
        CameraWithSidebarPage's own hideEvent, and a widget with a
        hidden ancestor reports ``isVisible() == False`` regardless of
        its own explicit shown/hidden state — so that check would
        silently skip closing it, leaving it still marked shown
        underneath. It would then reappear on its own the moment some
        other tab re-parents and re-shows the (shared) preview widget.
        """
        self._active = active
        if active:
            return
        QToolTip.hideText()
        if self._menu.current_index() is not None:
            self._menu.close_immediately()
        self._popup_tracking_index = None
        self._overlay.clear_hover()
        self._overlay.clear_proximity()

    def wants_focus_on_hover(self) -> bool:
        """Whether OverlayLabel should grab keyboard focus on enterEvent — purely so Delete reaches it from hovering a tag with no click needed first, so pointless whenever this isn't even active."""
        return self._active

    def handle_mouse_press(self, event: QMouseEvent) -> bool:
        """True if this fully handled the click (opened the menu, or deleted the hovered tag's measurement) — OverlayLabel should accept the event and stop there rather than fall through to placement/panning."""
        if not self._active or self._overlay.in_progress or event.button() != Qt.MouseButton.LeftButton:
            return False
        index = self._overlay.hovered_index
        if index is None:
            return False
        if self._overlay.hover_delete:
            self._overlay.remove_measurement(index)
            self._close_for_removed_measurement()
        else:
            anchor = self._tag_anchor_point(index) or event.position().toPoint()
            self._open_menu(index, anchor)
        self._video_label.update()
        return True

    def handle_mouse_move(self, event: QMouseEvent) -> None:
        """Tag hover and anchor-point proximity — a passive check alongside whatever else the move turns out to be, never a reason by itself to accept the event."""
        if not self._active or self._overlay.dragging_endpoint or self._overlay.in_progress:
            return
        pos = event.position().toPoint()
        display_rect, widget_rect = self._geometry()
        changed = self._overlay.update_hover(pos, display_rect, widget_rect)
        changed = self._overlay.update_proximity(pos, display_rect, widget_rect) or changed
        if changed:
            self._video_label.update()
        self._update_hover_tooltip(event.globalPosition().toPoint())

    def handle_key_press(self, event: QKeyEvent) -> bool:
        """True if Delete/Backspace removed the hovered tag's measurement — OverlayLabel should accept the event rather than let it fall through to any other shortcut handling."""
        if not self._active or self._overlay.hovered_index is None:
            return False
        if event.key() not in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            return False
        self._overlay.remove_measurement(self._overlay.hovered_index)
        self._close_for_removed_measurement()
        self._video_label.update()
        return True

    def handle_leave(self) -> None:
        repaint = self._overlay.clear_hover()
        repaint = self._overlay.clear_proximity() or repaint
        if repaint:
            self._video_label.update()
        QToolTip.hideText()

    def handle_paint_finished(self) -> None:
        """Called after every OverlayLabel repaint — keeps an open customize menu following its tag's current on-screen position through pans/zooms instead of drifting away once positioned."""
        if self._popup_tracking_index is None:
            return
        anchor = self._tag_anchor_point(self._popup_tracking_index)
        if anchor is None:
            return
        self._menu.reposition(self._video_label.mapTo(self._popup_parent, anchor))

    # ------------------------------------------------------------------

    def _geometry(self) -> tuple:
        return self._video_label.display_rect(), self._video_label.rect()

    def _tag_anchor_point(self, index: int) -> QPoint | None:
        """Bottom-center of measurement *index*'s own tag box, for the customize menu to center under — None if the tag's box wasn't recorded for some reason (e.g. it has no tag at all right now)."""
        display_rect, widget_rect = self._geometry()
        box = self._overlay.label_screen_rect(index, display_rect, widget_rect)
        if box is None:
            return None
        return QPoint(round(box.center().x()), round(box.bottom()))

    def _update_hover_tooltip(self, global_pos: QPoint) -> None:
        """Show the hovered measurement's description as a tooltip, if it has one — hides any tooltip otherwise, since a tag can be hovered without a description or not hovered at all."""
        index = self._overlay.hovered_index
        meta = self._overlay.measurement_meta(index) if index is not None else None
        if meta is None or not meta.description:
            QToolTip.hideText()
            return
        QToolTip.showText(global_pos, meta.description, self._video_label)

    def _open_menu(self, index: int, anchor: QPoint) -> None:
        meta = self._overlay.measurement_meta(index)
        kind = self._overlay.measurement_kind(index)
        if meta is None or kind is None:
            return
        self._menu.open_for(index, kind, meta, self._video_label.mapTo(self._popup_parent, anchor))
        self._popup_tracking_index = index

    def _on_meta_changed(self, index: int, meta: MeasurementMeta) -> None:
        """Shared by MeasurementCustomizeMenu.applied (final) and .preview_changed (live, including its own revert-on-cancel) — both just mean "this is this measurement's meta now"."""
        self._overlay.set_measurement_meta(index, meta)
        self._video_label.update()

    def _on_menu_closed(self, *_args: object) -> None:
        self._popup_tracking_index = None

    def _close_for_removed_measurement(self) -> None:
        """
        Deleting any measurement shifts every later one's index in the
        underlying list, so a currently-open menu's own tracked index
        may no longer even point at the measurement it was opened for —
        simplest correct response is to just close it unconditionally
        rather than try to confirm it was this exact one.
        """
        if self._menu.current_index() is not None:
            self._menu.close_immediately()
        self._popup_tracking_index = None