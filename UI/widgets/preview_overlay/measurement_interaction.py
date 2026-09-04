from __future__ import annotations

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QWidget

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
        self._menu.delete_requested.connect(self._on_delete_requested)
        self._menu.reset_requested.connect(self._on_reset_requested)

    def set_active(self, active: bool) -> None:
        """
        Called from OverlayLabel.set_measurement_mode_active — tag hover
        state and an open popup shouldn't survive switching away from
        the measurement tab.

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
        if self._menu.current_index() is not None:
            self._menu.close_immediately()
        self._popup_tracking_index = None
        self._overlay.clear_hover()
        self._overlay.clear_proximity()

    def wants_focus_on_hover(self) -> bool:
        """Whether OverlayLabel should grab keyboard focus on enterEvent — purely so Delete reaches it from hovering a tag with no click needed first, so pointless whenever this isn't even active."""
        return self._active

    def handle_delete_press(self, event: QMouseEvent) -> bool:
        """
        True if the click landed on a hovered tag's delete glyph and
        removed that measurement — OverlayLabel should accept the event
        and stop. A plain tag/point press (opening the customize menu or
        starting a tag/endpoint drag) is handled by OverlayLabel's own
        click-vs-drag logic, not here, so those cases return False.
        """
        if not self._active or self._overlay.in_progress or event.button() != Qt.MouseButton.LeftButton:
            return False
        index = self._overlay.hovered_index
        if index is None or not self._overlay.hover_delete:
            return False
        self._overlay.remove_measurement(index)
        self._close_for_removed_measurement()
        self._video_label.update()
        return True

    def open_menu_for(self, index: int, fallback_pos: QPoint) -> None:
        """Open the customize menu for measurement *index*, anchored under its tag if it has one, or at *fallback_pos* (a click straight on a point with no tag — feature 3) otherwise."""
        anchor = self._tag_anchor_point(index) or fallback_pos
        self._open_menu(index, anchor)
        self._video_label.update()

    def handle_mouse_move(self, event: QMouseEvent) -> None:
        """
        Tag hover and anchor-point proximity — a passive check alongside
        whatever else the move turns out to be, never a reason by itself
        to accept the event. Hovering a tag reveals its description
        in-box, the same rendering "always show description" uses (see
        MeasurementOverlay._draw_measurement_label) rather than a
        separate floating tooltip, so a hover change needs a repaint the
        same as any other hover-driven redraw.
        """
        if not self._active or self._overlay.dragging_endpoint or self._overlay.in_progress:
            return
        pos = event.position().toPoint()
        display_rect, widget_rect = self._geometry()
        changed = self._overlay.update_hover(pos, display_rect, widget_rect)
        changed = self._overlay.update_proximity(pos, display_rect, widget_rect) or changed
        if changed:
            self._video_label.update()

    def handle_key_press(self, event: QKeyEvent) -> bool:
        """
        True if Delete/Backspace acted on a hovered tag — dismissing a
        hovered secondary tag if there is one, otherwise removing the
        hovered measurement outright. OverlayLabel should then accept the
        event rather than let it fall through to other shortcut handling.
        """
        if not self._active:
            return False
        if event.key() not in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            return False
        if self._overlay.hovered_extra is not None:
            if self._overlay.delete_hovered_extra():
                self._video_label.update()
                return True
            return False
        if self._overlay.hovered_index is None:
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

    def _open_menu(self, index: int, anchor: QPoint) -> None:
        meta = self._overlay.measurement_meta(index)
        kind = self._overlay.measurement_kind(index)
        if meta is None or kind is None:
            return
        self._menu.open_for(index, kind, meta, self._video_label.mapTo(self._popup_parent, anchor))
        self._popup_tracking_index = index

    def _on_meta_changed(self, index: int, meta: MeasurementMeta) -> None:
        """
        Shared by MeasurementCustomizeMenu.applied (final) and
        .preview_changed (live, including its own revert-on-cancel) —
        both just mean "this is this measurement's meta now".

        Positional/interaction state the menu never edits — a tag's
        dragged offset, dismissed or dragged secondary tags — is carried
        over from the measurement's live meta rather than the menu's
        open-time snapshot, so dragging a tag (e.g. a text annotation)
        while the menu is open isn't reverted when it applies or cancels.
        """
        current = self._overlay.measurement_meta(index)
        if current is not None:
            meta = meta._replace(
                tag_offset_x=current.tag_offset_x,
                tag_offset_y=current.tag_offset_y,
                extra_offsets=current.extra_offsets,
                hidden_extra=current.hidden_extra,
            )
        self._overlay.set_measurement_meta(index, meta)
        self._video_label.update()

    def _on_menu_closed(self, *_args: object) -> None:
        self._popup_tracking_index = None

    def _on_delete_requested(self, index: int) -> None:
        """The customize menu's own "Delete measurement" button (already confirmed) — remove it and drop the now-stale popup."""
        self._overlay.remove_measurement(index)
        self._close_for_removed_measurement()
        self._video_label.update()

    def _on_reset_requested(self, index: int) -> None:
        """
        Reset a measurement's style to the current default (keeping its
        own title/description and geometry) and bring back any tags it
        had hidden — the default meta carries empty hidden_extra/offset
        fields, so those clear for free. The still-open menu is reloaded
        to reflect the reset.
        """
        current = self._overlay.measurement_meta(index)
        kind = self._overlay.measurement_kind(index)
        if current is None or kind is None:
            return
        reset = self._overlay.default_meta._replace(title=current.title, description=current.description)
        self._overlay.set_measurement_meta(index, reset)
        self._video_label.update()
        anchor = self._tag_anchor_point(index)
        if anchor is not None:
            self._menu.open_for(index, kind, reset, self._video_label.mapTo(self._popup_parent, anchor))

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