from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QEnterEvent, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QToolButton, QWidget

from UI.widgets.measurements.measurement_style import (
    ICON_PEN_WIDTH,
    ICON_SIZE,
    POINT_ACTIVE_COLOR,
    POINT_COLOR,
    TILE_HEIGHT,
    TILE_WIDTH,
)


class MeasurementButton(QToolButton):
    """
    Base tile for a single measurement type: icon above name, checkable.

    Visual styling (transparent idle background, sharp corners, orange
    hover/checked fill) lives in UI/style.py under QToolButton#MeasurementTile.
    Points are drawn in that same orange, so they'd vanish against the tile's
    own hover/checked fill — to avoid that, two icon variants are baked up
    front: idle (orange points) and active (blue points), and `_sync_icon`
    swaps between them as hover/checked state changes.

    Subclasses set `name` (and optionally a shorter `display_name`, shown
    on the tile while `name` becomes the tooltip) and implement `_paint_icon`,
    using `_set_pen` for line/outline strokes and `_draw_point` for points.
    """

    name: str = ""
    display_name: str | None = None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setObjectName("MeasurementTile")
        self.setCheckable(True)
        self.setFixedSize(TILE_WIDTH, TILE_HEIGHT)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.setIconSize(QSize(ICON_SIZE, ICON_SIZE))

        self.setText(self.display_name or self.name)
        if self.display_name and self.display_name != self.name:
            self.setToolTip(self.name)

        self._idle_icon = self._render_icon(active=False)
        self._active_icon = self._render_icon(active=True)
        self.setIcon(self._idle_icon)

        self.toggled.connect(self._sync_icon)

    def enterEvent(self, event: QEnterEvent) -> None:
        super().enterEvent(event)
        self._sync_icon()

    def leaveEvent(self, event: QEvent) -> None:
        super().leaveEvent(event)
        self._sync_icon()

    def _sync_icon(self, *_args: object) -> None:
        active = self.isChecked() or self.underMouse()
        self.setIcon(self._active_icon if active else self._idle_icon)

    def _render_icon(self, *, active: bool) -> QIcon:
        pixmap = QPixmap(ICON_SIZE, ICON_SIZE)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_icon(painter, pixmap.rect(), active)
        painter.end()

        return QIcon(pixmap)

    def _set_pen(self, painter: QPainter, color: QColor, width: int = ICON_PEN_WIDTH) -> None:
        pen = painter.pen()
        pen.setColor(color)
        pen.setWidth(width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        # Reset to a solid line so a dashed pen set earlier in the same
        # icon (e.g. a dimension guide) doesn't bleed into a solid stroke.
        pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(pen)

    def _draw_point(self, painter: QPainter, point: QPoint, radius: int, active: bool) -> None:
        color = POINT_ACTIVE_COLOR if active else POINT_COLOR
        self._set_pen(painter, color)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(point, radius, radius)

    def _paint_icon(self, painter: QPainter, rect: QRect, active: bool) -> None:
        raise NotImplementedError