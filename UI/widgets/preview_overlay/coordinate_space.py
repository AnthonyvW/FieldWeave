from __future__ import annotations

from typing import Protocol

from PySide6.QtCore import QPoint, QPointF, QRect
from PySide6.QtGui import QTransform


class CoordinateSpace(Protocol):
    """
    The coordinate contract every overlay draws and hit-tests against.

    ``ZoomPreviewOverlay`` is the live preview's implementation; a
    consumer that only needs coordinate math (``MeasurementOverlay``)
    should depend on this protocol rather than that concrete class, so
    anything satisfying it — like ``IdentityCoordinateSpace`` below, for
    rendering onto a static export image — works as a drop-in
    replacement. Something that also needs pan control (the zoom-pan
    input tool) still needs the concrete ``ZoomPreviewOverlay``.

    Contract every ``draw()``/hit-test implicitly relies on: coordinates
    passed to and returned by these methods are frame fractions/pixels
    against *rect*/*widget_rect* as if the view were never zoomed or
    panned — ``OverlayLabel._paint_overlays`` applies the ambient
    ``QTransform`` from ``paint_transform()`` around each overlay's
    ``draw()`` call, so drawing code should never need to account for
    zoom/pan itself. Anything needing an on-screen-constant size
    (endpoint markers, stroke width) must counter-scale by
    ``current_scale_xy()``.
    """

    def paint_transform(self, rect: QRect) -> QTransform | None: ...

    def widget_pos_to_full_pixel(
        self, pos: QPoint, widget_rect: QRect
    ) -> tuple[float, float, int, int] | None: ...

    def widget_pos_for_rect_point(self, point: QPointF, rect: QRect, widget_rect: QRect) -> QPointF: ...

    def current_scale_xy(self) -> tuple[float, float]: ...

    def current_frame_dims(self) -> tuple[int, int] | None: ...

    def current_view_center_full_pixel(self, widget_rect: QRect) -> tuple[float, float] | None: ...


class IdentityCoordinateSpace:
    """
    A ``CoordinateSpace`` representing "no pan/zoom, 1:1 against a
    target image of a known size" — for rendering measurements onto a
    static full-resolution export image through the exact same
    draw/hit-test code the live interactive preview uses, rather than a
    separate export-specific renderer. *rect*/*widget_rect* passed to
    its methods are expected to already equal the full frame's own rect.
    """

    def __init__(self, frame_dims: tuple[int, int]) -> None:
        self._frame_dims = frame_dims

    def paint_transform(self, rect: QRect) -> QTransform | None:
        return None

    def widget_pos_to_full_pixel(
        self, pos: QPoint, widget_rect: QRect
    ) -> tuple[float, float, int, int] | None:
        full_w, full_h = self._frame_dims
        return float(pos.x()), float(pos.y()), full_w, full_h

    def widget_pos_for_rect_point(self, point: QPointF, rect: QRect, widget_rect: QRect) -> QPointF:
        return QPointF(point)

    def current_scale_xy(self) -> tuple[float, float]:
        return 1.0, 1.0

    def current_frame_dims(self) -> tuple[int, int] | None:
        return self._frame_dims

    def current_view_center_full_pixel(self, widget_rect: QRect) -> tuple[float, float] | None:
        full_w, full_h = self._frame_dims
        return full_w / 2.0, full_h / 2.0
