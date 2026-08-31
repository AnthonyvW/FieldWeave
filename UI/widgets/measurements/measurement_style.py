from __future__ import annotations

from PySide6.QtGui import QColor

# Tile geometry
TILE_WIDTH = 84
TILE_HEIGHT = 72
ICON_SIZE = 32
ICON_PEN_WIDTH = 2
POINT_RADIUS = 5
ENDPOINT_RADIUS = 2
LINE_MARGIN = 6
CIRCLE_MARGIN = 4

# Two-tone icon colors: POINT_COLOR marks every placed point (the same
# orange QToolButton#MeasurementTile turns on hover/checked — see
# UI/style.py); LINE_COLOR strokes whatever connects those points (a
# line, or the radius circle's outline).
#
# An orange point disappears against that same orange hover/checked
# background, so points swap to POINT_ACTIVE_COLOR (blue) whenever the
# tile is hovered or checked — orange and blue read as high-contrast
# opposites, so this keeps points legible without breaking from the
# app's palette. See MeasurementButton._draw_point.
#
# Each tile's name is printed directly below its icon, so WCAG contrast
# only has to hold for that text (handled by QToolButton#MeasurementTile
# in UI/style.py) — the icon itself just needs its colors to read as
# distinct against the tile background and against each other.
POINT_COLOR = QColor("#f28c28")
POINT_ACTIVE_COLOR = QColor("#1a73e8")
LINE_COLOR = QColor("#000000")

# Full-viewport measurement overlay drawing — a different scale from the
# tile icons above, which are rendered into a fixed ICON_SIZE pixmap.
# White fill with a thin black outline reads against both light and dark
# backgrounds; sizes here are screen-pixel targets that MeasurementOverlay
# counter-scales against the current zoom level (see
# ZoomPreviewOverlay.current_scale_xy) so they stay this size on screen
# rather than growing with zoom.
OVERLAY_LINE_WIDTH = 2
OVERLAY_LINE_COLOR = QColor("#ffffff")
OVERLAY_OUTLINE_COLOR = QColor("#000000")
OVERLAY_OUTLINE_WIDTH = 1
OVERLAY_ENDPOINT_RADIUS = 4

# On/off lengths for the in-progress preview line's dashes, in the same
# screen-pixel-target terms as the sizes above. MeasurementOverlay derives
# each stroke's actual dash pattern from these divided by that stroke's
# own width, rather than letting Qt's default (pattern is in multiples of
# pen width) apply — otherwise the wider black outline and the thinner
# white fill drawn over it end up with differently-spaced dashes.
OVERLAY_DASH_LENGTH = 6
OVERLAY_DASH_GAP = 10

# Screen-pixel radius within which a click grabs an existing endpoint for
# dragging — deliberately larger than OVERLAY_ENDPOINT_RADIUS so the
# circle is easy to grab without needing pixel-perfect precision.
OVERLAY_ENDPOINT_HIT_RADIUS = 10

# Length/diameter label drawn next to a finalized measurement. Same
# fixed-on-screen-size treatment as the rest of this block — see
# MeasurementOverlay._draw_label.
OVERLAY_LABEL_FONT_SIZE = 13
OVERLAY_LABEL_OFFSET = 8
OVERLAY_LABEL_PADDING_X = 6
OVERLAY_LABEL_PADDING_Y = 3
OVERLAY_LABEL_CORNER_RADIUS = 4