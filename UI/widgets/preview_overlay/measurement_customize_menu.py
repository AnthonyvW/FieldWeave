from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QObject, Qt, Signal, QPoint, QPointF, QSize
from PySide6.QtGui import QColor, QFont, QIcon, QMouseEvent, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractScrollArea, QApplication, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFontComboBox, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSlider, QSpinBox,
    QStyle, QVBoxLayout, QWidget,
)

from UI.widgets.measurements.measurement_style import OVERLAY_LABEL_FONT_SIZE


class _WheelBlocker(QObject):
    """
    Event filter that stops the scroll wheel from cycling a combo box,
    spin box, or slider's value (an easy way to change a setting by
    accident while scrolling past it). The wheel is instead forwarded to
    the nearest ancestor scroll area so the surrounding panel still
    scrolls normally.
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() != QEvent.Type.Wheel:
            return False
        widget = obj
        while widget is not None:
            if isinstance(widget, QAbstractScrollArea):
                QApplication.sendEvent(widget.viewport(), event)
                break
            widget = widget.parentWidget()
        return True


_WHEEL_BLOCKER = _WheelBlocker()


def block_wheel(widget: QWidget) -> None:
    """Stop *widget* (a combo/spin/slider) from changing value on scroll — see _WheelBlocker."""
    widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    widget.installEventFilter(_WHEEL_BLOCKER)

from UI.widgets.measurements.lines import MEASUREMENT_LINE_CAPS, MEASUREMENT_MIDPOINT_STYLES
from UI.widgets.measurements.points import MEASUREMENT_POINT_STYLES
from UI.widgets.measurements.measurement_kind import DEFAULT_REGISTRY
from UI.widgets.measurements.measurement_meta import DEFAULT_META, MeasurementMeta
from UI.widgets.measurements.measurement_style import (
    OVERLAY_LINE_COLOR, OVERLAY_LINE_WIDTH, OVERLAY_OUTLINE_COLOR, OVERLAY_OUTLINE_WIDTH,
)
from UI.widgets.measurements.units import MeasurementUnit
from UI.widgets.preview_overlay.measurement_overlay import MEASUREMENT_DASH_PATTERNS

# MeasurementCustomizeMenu and its supporting widgets used to live in
# camera_preview.py. They're pulled out here because they're a
# self-contained editor for one measurement's title/description/unit/
# appearance — CameraPreview only needs to open it, read its signals,
# and host it as a child widget (see CameraPreview._on_measurement_tag_clicked
# and ._on_measurement_meta_applied), not know how it's built inside.


def _field_label(text: str) -> QLabel:
    """Section/field label styled via #MeasurementFieldLabel in style.py — uppercased here since Qt stylesheets have no text-transform of their own."""
    label = QLabel(text.upper())
    label.setObjectName("MeasurementFieldLabel")
    return label


_ICON_SIZE = QSize(48, 16)


def _dash_style_icon(dash_style: str) -> QIcon:
    """A short preview stroke in *dash_style* — used as a combo item's whole content (see _StylePicker) so the option shows itself instead of naming itself."""
    pixmap = QPixmap(_ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor("#333333"))
    pen.setWidth(2)
    pattern = MEASUREMENT_DASH_PATTERNS.get(dash_style)
    if pattern:
        # Round dash caps read as the intended dots/rounded dashes.
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setStyle(Qt.PenStyle.CustomDashLine)
        pen.setDashPattern(pattern)
    else:
        # A solid line shows squared-off ends rather than the rounded
        # ones a round cap would add.
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    painter.setPen(pen)
    y = _ICON_SIZE.height() / 2
    painter.drawLine(QPointF(4, y), QPointF(_ICON_SIZE.width() - 4, y))
    painter.end()
    return QIcon(pixmap)


def _line_cap_icon(cap: str) -> QIcon:
    """A short stroke ending in *cap*'s decoration — mirrors MeasurementOverlay._draw_stroke's cap shapes closely enough to read as the same style, without needing the overlay's zoom/stroke-scale machinery for a fixed-size icon."""
    pixmap = QPixmap(_ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    color = QColor("#333333")
    y = _ICON_SIZE.height() / 2
    x1, x2 = 4.0, _ICON_SIZE.width() - 14.0
    tip = QPointF(_ICON_SIZE.width() - 4.0, y)

    # "arrow_open"'s and "bracket"'s shafts run the full way to the tip
    # (flat-ended, so "arrow_open" stays sharp rather than blunted by a
    # round cap, matching the overlay where the barbs alone form the
    # point; "bracket" sits its tick directly on the tip).
    pen = QPen(color)
    pen.setWidth(2)
    if cap == "square":
        pen.setCapStyle(Qt.PenCapStyle.SquareCap)
    elif cap in ("arrow_open", "bracket"):
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    else:
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.drawLine(QPointF(x1, y), tip if cap in ("arrow_open", "bracket") else QPointF(x2, y))

    if cap == "arrow":
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([tip, QPointF(x2 - 4, y - 5), QPointF(x2 - 4, y + 5)]))
    elif cap == "arrow_diamond":
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        mid = QPointF((tip.x() + x2 - 4) / 2, y)
        painter.drawPolygon(QPolygonF([tip, QPointF(mid.x(), y - 5), QPointF(x2 - 4, y), QPointF(mid.x(), y + 5)]))
    elif cap == "arrow_circle":
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(tip.x() - 4, y), 4.0, 4.0)
    elif cap == "arrow_open":
        open_pen = QPen(color)
        open_pen.setWidth(2)
        open_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(open_pen)
        painter.drawLine(tip, QPointF(x2 - 4, y - 5))
        painter.drawLine(tip, QPointF(x2 - 4, y + 5))
    elif cap == "bracket":
        tick_pen = QPen(color)
        tick_pen.setWidth(2)
        tick_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(tick_pen)
        painter.drawLine(QPointF(tip.x(), y - 5), QPointF(tip.x(), y + 5))

    painter.end()
    return QIcon(pixmap)


def _midpoint_style_icon(style: str) -> QIcon:
    """A short horizontal stroke with *style*'s midpoint marker on it — nothing for "none", a perpendicular tick, or an x — mirroring MeasurementOverlay._draw_midpoint_marker closely enough to read as the same choice."""
    pixmap = QPixmap(_ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor("#333333"))
    pen.setWidth(2)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    cx = _ICON_SIZE.width() / 2
    cy = _ICON_SIZE.height() / 2
    painter.drawLine(QPointF(4, cy), QPointF(_ICON_SIZE.width() - 4, cy))
    half = 5.0
    if style == "tick":
        painter.drawLine(QPointF(cx, cy - half), QPointF(cx, cy + half))
    elif style == "x":
        painter.drawLine(QPointF(cx - half, cy - half), QPointF(cx + half, cy + half))
        painter.drawLine(QPointF(cx + half, cy - half), QPointF(cx - half, cy + half))
    painter.end()
    return QIcon(pixmap)


_POINT_STYLE_POLYGON_OFFSETS = {
    "square": ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),
    "diamond": ((0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)),
    "triangle": ((0.0, -1.0), (0.866, 0.5), (-0.866, 0.5)),
}


def _point_style_icon(style: str) -> QIcon:
    """A filled/stroked preview of *style*'s marker shape, centered in the icon box — mirrors MeasurementOverlay._draw_point_marker closely enough to read as the same choice."""
    pixmap = QPixmap(_ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    color = QColor("#333333")
    cx, cy = _ICON_SIZE.width() / 2, _ICON_SIZE.height() / 2
    r = _ICON_SIZE.height() / 2 - 3.0

    if style == "circle":
        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), r, r)
    elif style in ("cross", "x"):
        pen = QPen(color)
        pen.setWidth(2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        axes = ((1.0, 0.0), (0.0, 1.0)) if style == "cross" else ((1.0, 1.0), (1.0, -1.0))
        for ax, ay in axes:
            norm = math.hypot(ax, ay) or 1.0
            ux, uy = ax / norm * r, ay / norm * r
            painter.drawLine(QPointF(cx - ux, cy - uy), QPointF(cx + ux, cy + uy))
    elif style in _POINT_STYLE_POLYGON_OFFSETS:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([
            QPointF(cx + dx * r, cy + dy * r) for dx, dy in _POINT_STYLE_POLYGON_OFFSETS[style]
        ]))
    else:  # "dot"
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(cx, cy), r, r)
    painter.end()
    return QIcon(pixmap)


class _StylePicker(QWidget):
    """One line: a name label plus a combo box whose items show themselves — a rendered preview icon and no text — rather than naming the style, since seeing a line style is more useful than reading its name."""

    value_changed = Signal(str)

    def __init__(self, label: str, options: list[tuple[str, QIcon]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(_field_label(label))

        self._combo = QComboBox()
        self._combo.setIconSize(_ICON_SIZE)
        for value, icon in options:
            self._combo.addItem(icon, "", value)
        self._combo.currentIndexChanged.connect(self._on_index_changed)
        block_wheel(self._combo)
        row.addWidget(self._combo, 1)

    def set_value(self, value: str) -> None:
        index = self._combo.findData(value)
        if index >= 0:
            self._combo.setCurrentIndex(index)

    def value(self) -> str:
        return self._combo.currentData()

    def _on_index_changed(self, _index: int) -> None:
        self.value_changed.emit(self._combo.currentData())


class _ColorPicker(QWidget):
    """
    One line: a name label, a clickable color swatch (opens
    QColorDialog), and an editable hex code field, kept in sync in both
    directions. The field always shows a real hex value — *default_hex*
    when nothing's been overridden — rather than a blank "Default"
    placeholder; overriding with that same color is indistinguishable
    from not overriding at all, so it's treated as clearing the override.
    """

    color_changed = Signal(str)

    def __init__(self, label: str, default_hex: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color: str = ""
        self._default_hex = default_hex

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        row.addWidget(_field_label(label))

        self._swatch = QPushButton()
        self._swatch.setFixedSize(20, 20)
        self._swatch.clicked.connect(self._pick_color)
        row.addWidget(self._swatch)

        self._hex_edit = QLineEdit()
        self._hex_edit.setMaximumWidth(80)
        self._hex_edit.editingFinished.connect(self._on_hex_edited)
        row.addWidget(self._hex_edit)
        row.addStretch(1)

        self._refresh()

    def set_color(self, hex_color: str) -> None:
        self._color = hex_color
        self._refresh()

    def color(self) -> str:
        return self._color

    def effective_color(self) -> str:
        """The color actually shown — the override, or the default when none is set. Used where an empty override still needs a concrete color (e.g. an enabled fill)."""
        return self._color or self._default_hex

    def _refresh(self) -> None:
        display = self._color or self._default_hex
        self._swatch.setStyleSheet(f"background-color: {display}; border: 1px solid #888888; border-radius: 3px;")
        self._hex_edit.setText(display)

    def _pick_color(self) -> None:
        initial = QColor(self._color or self._default_hex)
        chosen = QColorDialog.getColor(initial, self, "Choose Color")
        if not chosen.isValid():
            return
        self._apply(chosen.name())

    def _on_hex_edited(self) -> None:
        text = self._hex_edit.text().strip()
        color = QColor(text if text.startswith("#") else f"#{text}")
        if not color.isValid():
            self._refresh()  # invalid entry — revert the field to the last valid value
            return
        self._apply(color.name())

    def _apply(self, hex_color: str) -> None:
        self._color = "" if hex_color == QColor(self._default_hex).name() else hex_color
        self._refresh()
        self.color_changed.emit(self._color)


class _ThicknessControl(QWidget):
    """Slider plus a numeric spinbox for the same value, kept in sync in both directions — a slider alone can't be set precisely, a spinbox alone can't be dragged."""

    value_changed = Signal(float)

    _STEPS_PER_UNIT = 10  # QSlider is integer-only; scaled for 0.1 precision

    def __init__(self, minimum: float, maximum: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._syncing = False

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(round(minimum * self._STEPS_PER_UNIT), round(maximum * self._STEPS_PER_UNIT))
        self._slider.valueChanged.connect(self._on_slider_changed)
        block_wheel(self._slider)
        row.addWidget(self._slider, 1)

        self._spin = QDoubleSpinBox()
        self._spin.setRange(minimum, maximum)
        self._spin.setSingleStep(1.0 / self._STEPS_PER_UNIT * 5)
        self._spin.setDecimals(1)
        self._spin.valueChanged.connect(self._on_spin_changed)
        block_wheel(self._spin)
        row.addWidget(self._spin)

    def set_value(self, value: float) -> None:
        self._syncing = True
        self._slider.setValue(round(value * self._STEPS_PER_UNIT))
        self._spin.setValue(value)
        self._syncing = False

    def value(self) -> float:
        return self._spin.value()

    def _on_slider_changed(self, raw: int) -> None:
        if self._syncing:
            return
        value = raw / self._STEPS_PER_UNIT
        self._syncing = True
        self._spin.setValue(value)
        self._syncing = False
        self.value_changed.emit(value)

    def _on_spin_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        self._slider.setValue(round(value * self._STEPS_PER_UNIT))
        self._syncing = False
        self.value_changed.emit(value)


class _VerticalResizeGrip(QWidget):
    """Diagonal-hatch handle in a text box's bottom-right corner — drag to grow/shrink its height, since QPlainTextEdit has no resize handle of its own."""

    dragged = Signal(int)

    _SIZE = 12

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self._SIZE, self._SIZE)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self._drag_y: int | None = None

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#9a9a9a"))
        pen.setWidth(1)
        painter.setPen(pen)
        for offset in (2, 5, 8):
            painter.drawLine(self._SIZE - offset, self._SIZE - 1, self._SIZE - 1, self._SIZE - offset)
        painter.end()

    def mousePressEvent(self, event) -> None:
        self._drag_y = event.globalPosition().toPoint().y()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_y is None:
            return
        current_y = event.globalPosition().toPoint().y()
        delta = current_y - self._drag_y
        self._drag_y = current_y
        if delta:
            self.dragged.emit(delta)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_y = None
        event.accept()


class _ResizableDescriptionEdit(QPlainTextEdit):
    """QPlainTextEdit with a drag handle (see _VerticalResizeGrip) pinned to its bottom-right corner for resizing its height."""

    resized = Signal()

    _MIN_HEIGHT = 44
    _MAX_HEIGHT = 220

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._grip = _VerticalResizeGrip(self)
        self._grip.dragged.connect(self._on_grip_dragged)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._grip.move(self.width() - self._grip.width() - 2, self.height() - self._grip.height() - 2)

    def _on_grip_dragged(self, delta: int) -> None:
        target = max(self._MIN_HEIGHT, min(self._MAX_HEIGHT, self.height() + delta))
        if target != self.height():
            self.setFixedHeight(target)
            self.resized.emit()


class MeasurementCustomizeMenu(QFrame):
    """
    Floating panel for editing one placed measurement's title,
    description, unit, and appearance.

    Opened by ``OverlayLabel`` when a measurement's tag is clicked
    (anywhere but its delete glyph — see ``OverlayLabel.mousePressEvent``)
    and positioned right there over the preview, since it's editing
    something drawn on the preview rather than acting as a general
    settings dialog. A plain child widget of ``CameraPreview`` rather
    than a QDialog, the same way the crosshair/grid/zoom buttons sit
    directly on the preview.
    """

    _ANCHOR_GAP_PX: int = 6

    applied = Signal(int, object)  # measurement index, MeasurementMeta
    preview_changed = Signal(int, object)  # measurement index, MeasurementMeta — live, while still open
    cancelled = Signal()
    delete_requested = Signal(int)  # measurement index — after the user confirms deletion
    reset_requested = Signal(int)  # measurement index — reset this measurement's style to the default
    defaults_changed = Signal(str, object)  # kind, MeasurementMeta — embedded defaults-template mode only

    def __init__(self, parent: QWidget | None = None, *, embedded: bool = False) -> None:
        """
        *embedded* makes this a plain child widget with no footer, driving
        a kind's own default template live (see open_defaults_for) instead
        of a floating popup that edits one placed measurement (open_for) —
        used by MeasurementsWidget's "Customize Default <kind>" panel so
        it shares the exact same fields/visibility rules as the popup
        rather than a hand-maintained subset that drifts out of sync.
        """
        super().__init__(parent)
        self._embedded = embedded
        self.setObjectName("MeasurementCustomizeMenu")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        if not embedded:
            # A real top-level window, so it can be dragged outside the
            # preview frame; closing it (its X button) acts like Cancel —
            # see closeEvent.
            self.setWindowFlags(Qt.WindowType.Window)
            self.setWindowTitle("Customize Measurement")
        # QFrame subclasses don't pick up a stylesheet background on
        # their own unless told to paint one — without this the panel
        # stayed transparent regardless of #MeasurementCustomizeMenu's
        # background rule in style.py.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._index: int | None = None
        self._original_meta: MeasurementMeta | None = None
        # Embedded-mode state: which kind's template is currently shown,
        # and every kind's own stored template (so switching back to a
        # previously-edited kind shows its edits, not fresh defaults) —
        # see open_defaults_for/default_meta_for.
        self._current_defaults_kind: str | None = None
        self._default_meta_by_kind: dict[str, MeasurementMeta] = {}
        # Suppresses live preview_changed emissions while open_for is
        # populating the fields.
        self._loading: bool = False
        # Whether the measurement currently open is a text annotation, whose
        # content comes from the multi-line editor rather than the title.
        self._is_text: bool = False
        # Whether the measurement currently open is a scale bar, whose Unit
        # combo lives on its own Bar Length row rather than the generic one.
        self._is_scalebar: bool = False
        # Once the user drags the panel out of the way (feature 2), it
        # stops auto-following its tag until reopened.
        self._manually_moved: bool = False
        self._drag_origin: QPoint | None = None

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # The many fields live in a scroll area so a tall set of options
        # never pushes the Apply/Cancel/Delete footer off-screen — see the
        # footer built at the end of __init__.
        content = QWidget()
        layout = QVBoxLayout(content)
        # Extra right margin beyond the other three sides' 14px — the
        # scroll area's vertical scrollbar (when it appears) otherwise eats
        # directly into a plain 14px margin, leaving fields looking
        # cramped against it rather than evenly padded.
        scrollbar_width = QApplication.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        layout.setContentsMargins(14, 14, 14 + scrollbar_width, 14)
        layout.setSpacing(8)
        self._scroll_content = content
        self._outer_layout = outer_layout

        self._title_label = _field_label("Title")
        layout.addWidget(self._title_label)
        self._title_edit = QLineEdit()
        self._title_edit.setObjectName("MeasurementCustomizeTitle")
        self._title_edit.setPlaceholderText("Label this measurement...")
        self._title_edit.textChanged.connect(self._on_live_field_changed)
        layout.addWidget(self._title_edit)

        # A text annotation's content is multi-line and can be bold — shown
        # in place of the single-line title/description for the text kind.
        self._text_contents_edit = _ResizableDescriptionEdit()
        self._text_contents_edit.setObjectName("MeasurementCustomizeDescription")
        self._text_contents_edit.setFixedHeight(60)
        self._text_contents_edit.setPlaceholderText("Text")
        self._text_contents_edit.textChanged.connect(self._on_live_field_changed)
        self._text_contents_edit.resized.connect(self.adjustSize)
        layout.addWidget(self._text_contents_edit)
        self._text_bold_check = QCheckBox("Bold")
        self._text_bold_check.toggled.connect(self._on_live_field_changed)
        layout.addWidget(self._text_bold_check)

        self._description_label = _field_label("Description")
        layout.addWidget(self._description_label)
        self._description_edit = _ResizableDescriptionEdit()
        self._description_edit.setObjectName("MeasurementCustomizeDescription")
        self._description_edit.setFixedHeight(60)
        self._description_edit.resized.connect(self.adjustSize)
        self._description_edit.textChanged.connect(self._on_live_field_changed)
        layout.addWidget(self._description_edit)

        unit_row = QHBoxLayout()
        unit_row.setContentsMargins(0, 0, 0, 0)
        unit_row.addWidget(_field_label("Unit"))
        self._unit_combo = QComboBox()
        for unit in MeasurementUnit:
            self._unit_combo.addItem(unit.value, unit)
        block_wheel(self._unit_combo)
        self._unit_combo.currentIndexChanged.connect(self._on_live_field_changed)
        unit_row.addWidget(self._unit_combo, 1)
        unit_row.addWidget(_field_label("Decimals"))
        self._decimals_spin = QSpinBox()
        self._decimals_spin.setRange(0, 6)
        block_wheel(self._decimals_spin)
        self._decimals_spin.valueChanged.connect(self._on_live_field_changed)
        unit_row.addWidget(self._decimals_spin)
        self._unit_container = QWidget()
        self._unit_container.setLayout(unit_row)
        layout.addWidget(self._unit_container)

        # Scale-bar-only fields sit right after Unit/Decimals — effectively
        # the top of the popup for a scale bar, since Title/Description are
        # hidden for it — with Bar Length first (feature: length at the top).
        self._build_scalebar_controls(layout)

        self._build_font_width_controls(layout)

        self._area_check = QCheckBox("Show area")
        self._area_check.toggled.connect(self._on_area_toggled)
        area_row = QHBoxLayout()
        area_row.addWidget(self._area_check)
        self._area_unit_combo = QComboBox()
        for unit in MeasurementUnit:
            self._area_unit_combo.addItem(unit.value, unit)
        block_wheel(self._area_unit_combo)
        self._area_unit_combo.currentIndexChanged.connect(self._on_live_field_changed)
        area_row.addWidget(self._area_unit_combo, 1)
        layout.addLayout(area_row)

        self._always_show_description_check = QCheckBox("Always show description")
        self._always_show_description_check.toggled.connect(self._on_live_field_changed)
        layout.addWidget(self._always_show_description_check)

        self._hidden_check = QCheckBox("Hide tag")
        self._hidden_check.toggled.connect(self._on_live_field_changed)
        layout.addWidget(self._hidden_check)

        self._always_show_center_check = QCheckBox("Always show center point")
        self._always_show_center_check.toggled.connect(self._on_live_field_changed)
        layout.addWidget(self._always_show_center_check)

        self._show_leg_lengths_check = QCheckBox("Show leg lengths")
        self._show_leg_lengths_check.toggled.connect(self._on_live_field_changed)
        layout.addWidget(self._show_leg_lengths_check)

        self._indicator_enabled_check = QCheckBox("Show indicator line")
        self._indicator_enabled_check.toggled.connect(self._on_indicator_toggled)
        layout.addWidget(self._indicator_enabled_check)

        self._indicator_color_picker = _ColorPicker("Indicator Color", OVERLAY_LINE_COLOR.name())
        self._indicator_color_picker.color_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._indicator_color_picker)

        self._indicator_style_picker = _StylePicker(
            "Indicator Style", [(style, _dash_style_icon(style)) for style in MEASUREMENT_DASH_PATTERNS]
        )
        self._indicator_style_picker.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._indicator_style_picker)

        self._indicator_opacity_label = _field_label("Indicator Opacity")
        layout.addWidget(self._indicator_opacity_label)
        self._indicator_opacity_control = _ThicknessControl(0.0, 1.0)
        self._indicator_opacity_control.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._indicator_opacity_control)

        self._opacity_label = _field_label("Opacity")
        layout.addWidget(self._opacity_label)
        self._opacity_control = _ThicknessControl(0.0, 1.0)
        self._opacity_control.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._opacity_control)

        self._build_tag_style_controls(layout)
        self._build_line_style_controls(layout)

        if not embedded:
            # Reset Style lives at the bottom of the scrollable field list
            # itself, not the fixed footer below it — it acts on the fields
            # above it, not as a persistent action like Apply/Cancel/Delete.
            # Meaningless in embedded (defaults-template) mode: there's no
            # separate "current default" to reset to, and no per-instance
            # hidden tags to restore.
            reset_button = QPushButton("Reset Style")
            reset_button.setToolTip("Reset this measurement's style to the current default and restore hidden tags")
            reset_button.clicked.connect(self._on_reset_clicked)
            layout.addWidget(reset_button)

        if embedded:
            # No footer, and no scroll area of its own — embedded directly
            # into MeasurementsWidget's sidebar, which is already one big
            # scroll area (see MeasurementTab._wrap_scroll); a second,
            # nested one here would make its own scrolling get stuck
            # rather than handing off to the sidebar's.
            outer_layout.addWidget(content, 1)
            return

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Otherwise the scroll area's own Base-colored viewport shows
        # through as a lighter rectangle against #MeasurementCustomizeMenu's
        # background rather than one continuous panel color.
        scroll.setStyleSheet("background: transparent;")
        scroll.viewport().setStyleSheet("background: transparent;")
        content.setStyleSheet("background: transparent;")
        scroll.setWidget(content)
        outer_layout.addWidget(scroll, 1)

        # Footer stays fixed below the scroll area so its buttons are
        # always reachable, however tall the field list gets.
        footer = QVBoxLayout()
        footer.setContentsMargins(14, 8, 14, 14)
        footer.setSpacing(8)
        self._delete_button = QPushButton("Delete measurement")
        self._delete_button.setObjectName("MeasurementDeleteButton")
        self._delete_button.clicked.connect(self._on_delete_clicked)
        footer.addWidget(self._delete_button)
        button_row = QHBoxLayout()
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self._on_apply_clicked)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self._on_cancel_clicked)
        button_row.addWidget(apply_button)
        button_row.addWidget(cancel_button)
        footer.addLayout(button_row)
        outer_layout.addLayout(footer)

        self.setMaximumHeight(560)
        self.hide()

    def _build_font_width_controls(self, layout: QVBoxLayout) -> None:
        # Font and its size share one row the same way Unit/Decimals do,
        # so the size field lines up with the rest rather than sitting
        # alone as a full-width box.
        font_row = QHBoxLayout()
        font_row.addWidget(_field_label("Font"))
        self._font_combo = QFontComboBox()
        # Non-scalable (bitmap) fonts like "Fixedsys" make DirectWrite log a
        # warning to the console whenever they're scrolled past on Windows —
        # excluding them from the list avoids hitting one at all.
        self._font_combo.setFontFilters(QFontComboBox.FontFilter.ScalableFonts)
        # Pick-only: a QFontComboBox is editable by default, which invites
        # typing a family name into it; the list is the whole interface.
        self._font_combo.setEditable(False)
        block_wheel(self._font_combo)
        self._font_combo.currentFontChanged.connect(self._on_live_field_changed)
        font_row.addWidget(self._font_combo, 1)
        font_row.addWidget(_field_label("Size"))
        self._font_size_spin = QSpinBox()
        self._font_size_spin.setRange(6, 96)
        block_wheel(self._font_size_spin)
        self._font_size_spin.valueChanged.connect(self._on_live_field_changed)
        font_row.addWidget(self._font_size_spin)
        layout.addLayout(font_row)

        self._tag_width_label = _field_label("Tag Width (0 = auto)")
        layout.addWidget(self._tag_width_label)
        self._tag_width_control = _ThicknessControl(0.0, 400.0)
        self._tag_width_control.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._tag_width_control)

    def _build_tag_style_controls(self, layout: QVBoxLayout) -> None:
        self._tag_transparent_check = QCheckBox("Transparent tag background")
        self._tag_transparent_check.toggled.connect(self._on_tag_transparent_toggled)
        layout.addWidget(self._tag_transparent_check)

        self._tag_bg_picker = _ColorPicker("Background Color", OVERLAY_LINE_COLOR.name())
        self._tag_bg_picker.color_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._tag_bg_picker)

        self._tag_text_picker = _ColorPicker("Tag Text Color", OVERLAY_OUTLINE_COLOR.name())
        self._tag_text_picker.color_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._tag_text_picker)

        # Text-annotation only: padding around the text. (Its overall
        # opacity is the shared "Opacity" control, which text now honors —
        # see MeasurementOverlay._draw_text_annotation.)
        self._text_margin_label = _field_label("Text Margin")
        layout.addWidget(self._text_margin_label)
        self._text_margin_control = _ThicknessControl(0.0, 40.0)
        self._text_margin_control.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._text_margin_control)

    def _build_fill_controls(self, layout: QVBoxLayout) -> None:
        self._fill_enabled_check = QCheckBox("Fill interior")
        self._fill_enabled_check.toggled.connect(self._on_fill_toggled)
        layout.addWidget(self._fill_enabled_check)

        self._fill_color_picker = _ColorPicker("Fill Color", "#1a73e8")
        self._fill_color_picker.color_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._fill_color_picker)

        self._fill_opacity_label = _field_label("Fill Opacity")
        layout.addWidget(self._fill_opacity_label)
        self._fill_opacity_control = _ThicknessControl(0.0, 1.0)
        self._fill_opacity_control.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._fill_opacity_control)

    def _build_scalebar_controls(self, layout: QVBoxLayout) -> None:
        """
        Scale-bar-only fields, ordered: what the bar is (length,
        thickness) — length first and, since this whole method is now
        called right after Unit/Decimals (see __init__), effectively at
        the top of the popup for a scale bar — then where it goes
        (anchor, position), then how its background panel looks (padding,
        background toggle).
        """
        self._scalebar_widgets: list[QWidget] = []

        # Unit sits on the same row as Bar Length — the bar has no separate
        # per-measurement decimal-places setting the way ordinary tags do
        # (its label always shows the length as entered), so there's no
        # Decimals field to pair it with.
        length_row = QHBoxLayout()
        self._scalebar_length_label = _field_label("Bar Length")
        length_row.addWidget(self._scalebar_length_label)
        self._scalebar_length_spin = QDoubleSpinBox()
        self._scalebar_length_spin.setRange(0.001, 1_000_000.0)
        self._scalebar_length_spin.setDecimals(3)
        block_wheel(self._scalebar_length_spin)
        self._scalebar_length_spin.valueChanged.connect(self._on_live_field_changed)
        length_row.addWidget(self._scalebar_length_spin, 1)
        length_row.addWidget(_field_label("Unit"))
        self._scalebar_unit_combo = QComboBox()
        for unit in MeasurementUnit:
            self._scalebar_unit_combo.addItem(unit.value, unit)
        block_wheel(self._scalebar_unit_combo)
        self._scalebar_unit_combo.currentIndexChanged.connect(self._on_live_field_changed)
        length_row.addWidget(self._scalebar_unit_combo)
        length_row.addWidget(_field_label("Decimals"))
        self._scalebar_decimals_spin = QSpinBox()
        self._scalebar_decimals_spin.setRange(0, 6)
        block_wheel(self._scalebar_decimals_spin)
        self._scalebar_decimals_spin.valueChanged.connect(self._on_live_field_changed)
        length_row.addWidget(self._scalebar_decimals_spin)
        length_container = QWidget()
        length_container.setLayout(length_row)
        length_row.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(length_container)
        self._scalebar_widgets.append(length_container)

        self._scalebar_thickness_label = _field_label("Bar Thickness")
        layout.addWidget(self._scalebar_thickness_label)
        self._scalebar_thickness_control = _ThicknessControl(1.0, 40.0)
        self._scalebar_thickness_control.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._scalebar_thickness_control)
        self._scalebar_widgets += [self._scalebar_thickness_label, self._scalebar_thickness_control]

        anchor_row = QHBoxLayout()
        anchor_row.addWidget(_field_label("Anchor"))
        self._scalebar_anchor_combo = QComboBox()
        self._scalebar_anchor_combo.addItem("Preview", True)
        self._scalebar_anchor_combo.addItem("Image", False)
        block_wheel(self._scalebar_anchor_combo)
        self._scalebar_anchor_combo.currentIndexChanged.connect(self._on_live_field_changed)
        anchor_row.addWidget(self._scalebar_anchor_combo, 1)
        anchor_container = QWidget()
        anchor_container.setLayout(anchor_row)
        anchor_row.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(anchor_container)
        self._scalebar_widgets.append(anchor_container)

        position_row = QHBoxLayout()
        position_row.addWidget(_field_label("Position"))
        self._scalebar_position_combo = QComboBox()
        for label, value in (
            ("Lower Left", "lower_left"), ("Lower Right", "lower_right"),
            ("Upper Left", "upper_left"), ("Upper Right", "upper_right"), ("Custom", "custom"),
        ):
            self._scalebar_position_combo.addItem(label, value)
        block_wheel(self._scalebar_position_combo)
        self._scalebar_position_combo.currentIndexChanged.connect(self._on_live_field_changed)
        position_row.addWidget(self._scalebar_position_combo, 1)
        position_container = QWidget()
        position_container.setLayout(position_row)
        position_row.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(position_container)
        self._scalebar_widgets.append(position_container)

        # Padding is the gap between the bar/label and the background
        # panel's edge, separate from the corner margin (Bar Margin, shared
        # with text via _text_margin_control) — grouped here with the
        # background toggle since both are about the panel's own look.
        self._scalebar_padding_label = _field_label("Bar Padding")
        layout.addWidget(self._scalebar_padding_label)
        self._scalebar_padding_control = _ThicknessControl(0.0, 40.0)
        self._scalebar_padding_control.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._scalebar_padding_control)
        self._scalebar_widgets += [self._scalebar_padding_label, self._scalebar_padding_control]

        # Own dedicated control (edits the same shared text_margin field a
        # text annotation's own "Text Margin" control does — see
        # _current_meta) rather than reusing that widget directly, so its
        # position here (just below Bar Padding) doesn't move Text Margin
        # around in a text annotation's own field order.
        self._scalebar_margin_label = _field_label("Bar Margin")
        layout.addWidget(self._scalebar_margin_label)
        self._scalebar_margin_control = _ThicknessControl(0.0, 40.0)
        self._scalebar_margin_control.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._scalebar_margin_control)
        self._scalebar_widgets += [self._scalebar_margin_label, self._scalebar_margin_control]

        self._scalebar_bg_check = QCheckBox("Scale bar background")
        self._scalebar_bg_check.toggled.connect(self._on_live_field_changed)
        layout.addWidget(self._scalebar_bg_check)
        self._scalebar_widgets.append(self._scalebar_bg_check)

        # Bar Color edits the same line_color field the generic Line Color
        # picker does (see _current_meta) — kept here so Font/Size, built
        # right after this method returns, sit just below the pair of them.
        self._scalebar_color_picker = _ColorPicker("Bar Color", OVERLAY_LINE_COLOR.name())
        self._scalebar_color_picker.color_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._scalebar_color_picker)
        self._scalebar_widgets.append(self._scalebar_color_picker)

        # Own dedicated picker (edits the same shared tag_background_color
        # field the generic "Background Color" picker below does) so a
        # scale bar's panel color sits right under Bar Color instead of
        # down with the other tags' fields.
        self._scalebar_bg_picker = _ColorPicker("Background Color", OVERLAY_LINE_COLOR.name())
        self._scalebar_bg_picker.color_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._scalebar_bg_picker)
        self._scalebar_widgets.append(self._scalebar_bg_picker)

    def _build_line_style_controls(self, layout: QVBoxLayout) -> None:
        self._build_fill_controls(layout)
        self._line_color_picker = _ColorPicker("Line Color", OVERLAY_LINE_COLOR.name())
        self._line_color_picker.color_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._line_color_picker)

        self._line_thickness_label = _field_label("Line Thickness")
        layout.addWidget(self._line_thickness_label)
        self._line_thickness_control = _ThicknessControl(0.5, 12.0)
        self._line_thickness_control.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._line_thickness_control)

        self._line_style_picker = _StylePicker(
            "Line Style", [(style, _dash_style_icon(style)) for style in MEASUREMENT_DASH_PATTERNS]
        )
        self._line_style_picker.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._line_style_picker)

        self._midpoint_picker = _StylePicker(
            "Midpoint", [(style, _midpoint_style_icon(style)) for style in MEASUREMENT_MIDPOINT_STYLES]
        )
        self._midpoint_picker.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._midpoint_picker)

        # "Point" and "Count" — the placed marker's own shape, in place of
        # a line's caps.
        self._point_style_picker = _StylePicker(
            "Point Style", [(style, _point_style_icon(style)) for style in MEASUREMENT_POINT_STYLES]
        )
        self._point_style_picker.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._point_style_picker)

        # "Count" only — hides every point's number at once; a specific
        # point is instead removed by hovering it and pressing Delete.
        self._count_hide_numbers_check = QCheckBox("Hide Numbers")
        self._count_hide_numbers_check.toggled.connect(self._on_live_field_changed)
        layout.addWidget(self._count_hide_numbers_check)

        # Caps are a line-only decoration (MeasurementOverlay never
        # passes start_cap/end_cap when drawing a circle) — open_for
        # hides this pair for any kind whose category isn't "line".
        self._start_cap_picker = _StylePicker(
            "Start", [(cap, _line_cap_icon(cap)) for cap in MEASUREMENT_LINE_CAPS]
        )
        self._start_cap_picker.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._start_cap_picker)

        self._end_cap_picker = _StylePicker(
            "End", [(cap, _line_cap_icon(cap)) for cap in MEASUREMENT_LINE_CAPS]
        )
        self._end_cap_picker.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._end_cap_picker)

        # One shared size for both the "arrow"/"arrow_open" and
        # "bracket" caps, rather than a separate control per style —
        # they're all sized off arrow_dims (see lines.py), so one slider
        # scales whichever cap is actually chosen.
        self._cap_size_label = _field_label("Arrow/Bracket Size")
        layout.addWidget(self._cap_size_label)
        # A high ceiling is impractical for arrowheads but genuinely
        # useful for brackets, which read fine even quite large (e.g. a
        # dimension-line-style bracket spanning most of a short line).
        self._cap_size_control = _ThicknessControl(0.25, 20.0)
        self._cap_size_control.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._cap_size_control)

        self._outline_enabled_check = QCheckBox("Enable Outline")
        self._outline_enabled_check.toggled.connect(self._on_outline_enabled_toggled)
        layout.addWidget(self._outline_enabled_check)

        self._outline_color_picker = _ColorPicker("Outline Color", OVERLAY_OUTLINE_COLOR.name())
        self._outline_color_picker.color_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._outline_color_picker)

        self._outline_thickness_label = _field_label("Outline Thickness")
        layout.addWidget(self._outline_thickness_label)
        self._outline_thickness_control = _ThicknessControl(0.0, 8.0)
        self._outline_thickness_control.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._outline_thickness_control)

    def wheelEvent(self, event) -> None:
        # Otherwise an ignored wheel event at a child's scroll limit
        # (e.g. the description box) bubbles up to CameraPreview, whose
        # own wheelEvent treats it as a request to zoom the preview.
        event.accept()

    def current_index(self) -> int | None:
        return self._index

    def open_for(self, index: int, kind: str, meta: MeasurementMeta, anchor: QPoint) -> None:
        """
        *anchor* is the tag's own on-screen box (see
        MeasurementOverlay.label_screen_rect), already mapped into this
        widget's parent's coordinates by the caller — not the raw click
        position, so the panel centers under the tag itself regardless
        of where within it the opening click landed.

        *kind* decides whether the start/end cap pickers show at all —
        caps only ever apply to line-category measurements (see
        MeasurementKind.category in measurement_kind.py) — not just
        whether they're editable.
        """
        self._loading = True
        self._manually_moved = False
        self._index = index
        self._original_meta = meta
        self._populate_fields(kind, meta)
        self._loading = False
        self.adjustSize()
        self.reposition(anchor)
        # As its own window the panel stays where opened (and wherever the
        # user drags it) rather than chasing its tag through pans/zooms.
        self._manually_moved = True
        self.show()
        self.raise_()
        self._title_edit.setFocus()

    def _populate_fields(self, kind: str, meta: MeasurementMeta) -> None:
        """
        Set every field's value and visibility from (*kind*, *meta*) —
        shared by open_for (editing one placed measurement) and
        open_defaults_for (editing a kind's default template), so the two
        never drift out of sync on which fields a kind actually shows.
        Caller is responsible for self._loading/adjustSize/etc. around it.
        """
        entry = DEFAULT_REGISTRY.get(kind)
        show_caps = entry is not None and entry.category in ("line", "angle", "arc", "line_pair", "curve")
        show_area = entry is not None and entry.category in ("circle", "ellipse")
        show_center = (entry is not None and entry.category in ("circle", "ellipse")) or kind == "Radius Arc"
        show_angle_extras = entry is not None and entry.category == "angle"
        show_fill = entry is not None and entry.category in ("circle", "ellipse", "polygon", "annulus")
        self._is_text = entry is not None and entry.category == "text"
        self._title_edit.setText(meta.title)
        self._text_contents_edit.setPlainText(meta.title)
        self._text_bold_check.setChecked(meta.font_bold)
        self._description_edit.setPlainText(meta.description)
        unit = meta.unit if meta.unit is not None else MeasurementUnit.MM
        self._unit_combo.setCurrentIndex(self._unit_combo.findData(unit))
        self._decimals_spin.setValue(meta.decimal_places)
        self._area_check.setChecked(meta.show_area)
        self._area_check.setVisible(show_area)
        area_unit = meta.area_unit if meta.area_unit is not None else unit
        self._area_unit_combo.setCurrentIndex(self._area_unit_combo.findData(area_unit))
        self._area_unit_combo.setVisible(show_area and meta.show_area)
        self._always_show_description_check.setChecked(meta.always_show_description)
        self._hidden_check.setChecked(meta.hidden)
        self._always_show_center_check.setChecked(meta.always_show_center)
        self._always_show_center_check.setVisible(show_center)
        self._show_leg_lengths_check.setChecked(meta.show_leg_lengths)
        self._show_leg_lengths_check.setVisible(show_angle_extras)
        # The indicator line exists for angles (the dashed guide + curve)
        # and for any line-pair kind with dimension connectors.
        show_indicator = entry is not None and (entry.category == "angle" or entry.connector_segments is not None)
        indicator_on = meta.indicator_enabled
        self._indicator_enabled_check.setChecked(indicator_on)
        self._indicator_enabled_check.setVisible(show_indicator)
        self._indicator_color_picker.set_color(meta.indicator_color)
        self._indicator_color_picker.setVisible(show_indicator and indicator_on)
        self._indicator_style_picker.set_value(meta.indicator_dash_style)
        self._indicator_style_picker.setVisible(show_indicator and indicator_on)
        self._indicator_opacity_control.set_value(meta.indicator_opacity)
        self._indicator_opacity_label.setVisible(show_indicator and indicator_on)
        self._indicator_opacity_control.setVisible(show_indicator and indicator_on)
        self._opacity_control.set_value(meta.opacity)
        self._tag_transparent_check.setChecked(meta.tag_background_transparent)
        self._tag_bg_picker.setVisible(not meta.tag_background_transparent)
        self._tag_bg_picker.set_color(meta.tag_background_color)
        self._tag_text_picker.set_color(meta.tag_text_color)
        if meta.font_family:
            self._font_combo.setCurrentFont(QFont(meta.font_family))
        self._font_size_spin.setValue(round(meta.font_size) if meta.font_size > 0 else round(OVERLAY_LABEL_FONT_SIZE))
        self._tag_width_control.set_value(meta.tag_width)
        show_scalebar = entry is not None and entry.category == "scalebar"
        self._is_scalebar = show_scalebar
        self._scalebar_length_spin.setValue(meta.scalebar_length if meta.scalebar_length > 0 else 1.0)
        self._scalebar_unit_combo.setCurrentIndex(self._scalebar_unit_combo.findData(unit))
        self._scalebar_decimals_spin.setValue(meta.decimal_places)
        self._scalebar_thickness_control.set_value(meta.scalebar_thickness)
        self._scalebar_anchor_combo.setCurrentIndex(0 if meta.scalebar_anchor_preview else 1)
        position_index = self._scalebar_position_combo.findData(meta.scalebar_position)
        if position_index >= 0:
            self._scalebar_position_combo.setCurrentIndex(position_index)
        self._scalebar_bg_check.setChecked(meta.scalebar_show_bg)
        self._scalebar_padding_control.set_value(meta.scalebar_padding)
        self._scalebar_margin_control.set_value(meta.text_margin)
        self._scalebar_bg_picker.set_color(meta.tag_background_color)
        for widget in self._scalebar_widgets:
            widget.setVisible(show_scalebar)

        show_text = entry is not None and entry.category == "text"
        is_count = entry is not None and entry.category == "count"
        is_annotation = show_text or show_scalebar
        # A "count" group has no tag at all (its numbers are drawn
        # directly — see _draw_count_numbers) — folded in with the two
        # annotation kinds for the tag-only fields that don't apply to it
        # either (title/description/unit/tag width), but NOT for line
        # color/thickness, which it still uses for its point markers.
        no_tag = is_annotation or is_count
        # A text annotation's title IS its drawn content — a multi-line,
        # optionally-bold editor replaces the single-line title/description;
        # a scale bar has neither a title nor a free-form description tag.
        self._title_label.setText("Text Contents" if show_text else "Title")
        self._title_label.setVisible(not show_scalebar and not is_count)
        self._title_edit.setVisible(not no_tag)
        self._text_contents_edit.setVisible(show_text)
        self._text_bold_check.setVisible(show_text)
        self._description_label.setVisible(not no_tag)
        self._description_edit.setVisible(not no_tag)
        # Neither annotation (nor a tagless "count" group) carries a
        # length tag, so its show/hide-tag and always-show-description
        # toggles don't apply.
        self._always_show_description_check.setVisible(not no_tag)
        self._hidden_check.setVisible(not no_tag)
        # A text annotation has no measured value at all; a scale bar has
        # its own Unit combo on the Bar Length row instead (and no
        # separate decimal-places setting); a "count" group's points have
        # nothing to measure either — so this generic Unit/Decimals row is
        # only for ordinary measurements.
        self._unit_container.setVisible(not no_tag)
        # An auto-sizing tag width is meaningless for a free-drawn text box,
        # a scale bar (each sizes to its own content), or a tagless "count"
        # group.
        self._tag_width_label.setVisible(not no_tag)
        self._tag_width_control.setVisible(not no_tag)
        # Text's own margin control — a scale bar has its own dedicated
        # "Bar Margin" control up in the scale-bar section instead (see
        # _build_scalebar_controls), so this one is text-only.
        self._text_margin_control.set_value(meta.text_margin)
        self._text_margin_label.setVisible(show_text)
        self._text_margin_control.setVisible(show_text)
        # Opacity fades an ordinary measurement or a text annotation; a
        # scale bar is always fully opaque.
        self._opacity_label.setVisible(not show_scalebar)
        self._opacity_control.setVisible(not show_scalebar)
        self._line_color_picker.set_color(meta.line_color)
        self._line_thickness_control.set_value(meta.line_thickness or OVERLAY_LINE_WIDTH)
        self._line_style_picker.set_value(meta.line_dash_style)
        # A scale bar edits the same line_color field through its own "Bar
        # Color" picker up in the scale-bar section instead; neither
        # annotation strokes an ordinary line at all.
        self._line_color_picker.setVisible(not is_annotation)
        self._scalebar_color_picker.set_color(meta.line_color)
        # A point (or a "count" group of them) has no line to dash, so its
        # dash-style picker is hidden; circles and lines both keep it.
        # Neither annotation strokes a line the ordinary way, so both hide
        # line thickness/style too.
        is_point = entry is not None and entry.category in ("point", "count")
        self._line_thickness_label.setVisible(not is_annotation)
        self._line_thickness_control.setVisible(not is_annotation)
        self._line_style_picker.setVisible(not is_point and not is_annotation)
        # A text annotation keeps its background color and text color
        # (its box and glyphs) but not the separate transparency toggle —
        # Opacity at 0 already makes the background invisible, so the
        # toggle would just be a second way to do the same thing; a scale
        # bar's panel uses the tag background as its own and colors its
        # label from the bar color.
        self._tag_transparent_check.setVisible(not no_tag)
        # A "count" group has no box to color at all (its number color
        # comes from Tag Text Color below instead), so its background
        # picker stays hidden regardless of the transparency toggle's
        # state; a scale bar edits the same field through its own
        # "Background Color" picker up in the scale-bar section instead.
        self._tag_bg_picker.setVisible(
            not is_count and not show_scalebar and (is_annotation or not meta.tag_background_transparent)
        )
        self._tag_text_picker.setVisible(not show_scalebar)
        self._midpoint_picker.set_value(meta.midpoint_style)
        self._midpoint_picker.setVisible(show_caps)
        self._point_style_picker.set_value(meta.point_style)
        self._point_style_picker.setVisible(is_point)
        self._count_hide_numbers_check.setChecked(meta.count_hide_numbers)
        self._count_hide_numbers_check.setVisible(is_count)
        self._start_cap_picker.set_value(meta.line_start_cap)
        self._start_cap_picker.setVisible(show_caps)
        self._end_cap_picker.set_value(meta.line_end_cap)
        self._end_cap_picker.setVisible(show_caps)
        self._cap_size_control.set_value(meta.cap_size_scale)
        self._cap_size_control.setVisible(show_caps)
        self._cap_size_label.setVisible(show_caps)
        fill_on = bool(meta.fill_color)
        self._fill_enabled_check.setChecked(fill_on)
        self._fill_enabled_check.setVisible(show_fill)
        self._fill_color_picker.set_color(meta.fill_color)
        self._fill_color_picker.setVisible(show_fill and fill_on)
        self._fill_opacity_control.set_value(meta.fill_opacity)
        self._fill_opacity_label.setVisible(show_fill and fill_on)
        self._fill_opacity_control.setVisible(show_fill and fill_on)
        # Neither annotation strokes an outline pass, so its whole outline
        # group is hidden.
        self._outline_enabled_check.setChecked(meta.outline_enabled)
        self._outline_enabled_check.setVisible(not is_annotation)
        self._set_outline_controls_visible(meta.outline_enabled and not is_annotation)
        self._outline_color_picker.set_color(meta.outline_color)
        self._outline_thickness_control.set_value(meta.outline_thickness or OVERLAY_OUTLINE_WIDTH)

    def reposition(self, anchor: QPoint) -> None:
        """Position the window near its tag on open, clamped so the whole panel stays on the screen it opens on (feature 9). *anchor* is in the parent's coordinate space; a top-level window is moved in global coordinates, so it's mapped through the parent. A no-op once positioned/dragged (the panel is its own window and stays put)."""
        if self._manually_moved:
            return
        parent = self.parentWidget()
        global_anchor = parent.mapToGlobal(anchor) if parent is not None else anchor
        x = global_anchor.x() - self.width() // 2
        y = global_anchor.y() + self._ANCHOR_GAP_PX
        screen = QApplication.screenAt(global_anchor) or QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            x = max(avail.left(), min(x, avail.right() - self.width() + 1))
            y = max(avail.top(), min(y, avail.bottom() - self.height() + 1))
        self.move(x, y)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Press on the panel's own background (not a child control) begins dragging it out of the way — feature 2."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.globalPosition().toPoint() - self.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin is not None:
            # A top-level window: move freely in global coordinates so it
            # can be dragged outside the preview frame.
            self.move(event.globalPosition().toPoint() - self._drag_origin)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def close_immediately(self) -> None:
        """Used when the measurement being edited is deleted out from under this menu (e.g. via the tag's own delete glyph, or the Delete key while hovering it) — closes without emitting applied/cancelled, since there's nothing left to apply or revert."""
        self._index = None
        self._original_meta = None
        self.hide()

    def _set_outline_controls_visible(self, visible: bool) -> None:
        self._outline_color_picker.setVisible(visible)
        self._outline_thickness_label.setVisible(visible)
        self._outline_thickness_control.setVisible(visible)

    def _on_outline_enabled_toggled(self, enabled: bool) -> None:
        self._set_outline_controls_visible(enabled)
        self.adjustSize()
        self._on_live_field_changed()

    def _on_area_toggled(self, enabled: bool) -> None:
        self._area_unit_combo.setVisible(enabled)
        self.adjustSize()
        self._on_live_field_changed()

    def _on_tag_transparent_toggled(self, transparent: bool) -> None:
        self._tag_bg_picker.setVisible(not transparent)
        self.adjustSize()
        self._on_live_field_changed()

    def _on_indicator_toggled(self, enabled: bool) -> None:
        visible = enabled and self._indicator_enabled_check.isVisible()
        self._indicator_color_picker.setVisible(visible)
        self._indicator_style_picker.setVisible(visible)
        self._indicator_opacity_label.setVisible(visible)
        self._indicator_opacity_control.setVisible(visible)
        self.adjustSize()
        self._on_live_field_changed()

    def _on_fill_toggled(self, enabled: bool) -> None:
        self._fill_color_picker.setVisible(enabled and self._fill_enabled_check.isVisible())
        self._fill_opacity_label.setVisible(enabled and self._fill_enabled_check.isVisible())
        self._fill_opacity_control.setVisible(enabled and self._fill_enabled_check.isVisible())
        self.adjustSize()
        self._on_live_field_changed()

    def _current_meta(self) -> MeasurementMeta:
        # Built from the original so per-measurement state the menu
        # doesn't edit — style_id and the tag's dragged offset — carries
        # through an edit rather than resetting to the default.
        base = self._original_meta if self._original_meta is not None else DEFAULT_META
        # A text annotation's content comes from its own multi-line editor
        # (newlines preserved); every other kind uses the single-line title.
        title = self._text_contents_edit.toPlainText() if self._is_text else self._title_edit.text().strip()
        return base._replace(
            title=title,
            description=self._description_edit.toPlainText().strip(),
            font_bold=self._text_bold_check.isChecked(),
            unit=self._scalebar_unit_combo.currentData() if self._is_scalebar else self._unit_combo.currentData(),
            tag_background_color=self._scalebar_bg_picker.color() if self._is_scalebar else self._tag_bg_picker.color(),
            tag_text_color=self._tag_text_picker.color(),
            always_show_description=self._always_show_description_check.isChecked(),
            line_color=self._scalebar_color_picker.color() if self._is_scalebar else self._line_color_picker.color(),
            line_thickness=self._line_thickness_control.value(),
            outline_enabled=self._outline_enabled_check.isChecked(),
            outline_color=self._outline_color_picker.color(),
            outline_thickness=self._outline_thickness_control.value(),
            line_dash_style=self._line_style_picker.value(),
            line_start_cap=self._start_cap_picker.value(),
            line_end_cap=self._end_cap_picker.value(),
            cap_size_scale=self._cap_size_control.value(),
            decimal_places=self._scalebar_decimals_spin.value() if self._is_scalebar else self._decimals_spin.value(),
            hidden=self._hidden_check.isChecked(),
            always_show_center=self._always_show_center_check.isChecked(),
            show_leg_lengths=self._show_leg_lengths_check.isChecked(),
            indicator_enabled=self._indicator_enabled_check.isChecked(),
            indicator_color=self._indicator_color_picker.color(),
            indicator_opacity=self._indicator_opacity_control.value(),
            indicator_dash_style=self._indicator_style_picker.value(),
            opacity=self._opacity_control.value(),
            # Text has no separate transparency toggle (Opacity at 0
            # already hides its background) — always store it opaque so a
            # stale checkbox state from another kind never sticks.
            tag_background_transparent=False if self._is_text else self._tag_transparent_check.isChecked(),
            midpoint_style=self._midpoint_picker.value(),
            point_style=self._point_style_picker.value(),
            count_hide_numbers=self._count_hide_numbers_check.isChecked(),
            show_area=self._area_check.isChecked(),
            area_unit=self._area_unit_combo.currentData(),
            fill_color=self._fill_color_picker.effective_color() if self._fill_enabled_check.isChecked() else "",
            fill_opacity=self._fill_opacity_control.value(),
            font_family=self._font_combo.currentFont().family(),
            font_size=float(self._font_size_spin.value()),
            tag_width=self._tag_width_control.value(),
            text_margin=self._scalebar_margin_control.value() if self._is_scalebar else self._text_margin_control.value(),
            scalebar_length=self._scalebar_length_spin.value(),
            scalebar_thickness=self._scalebar_thickness_control.value(),
            scalebar_anchor_preview=bool(self._scalebar_anchor_combo.currentData()),
            scalebar_position=self._scalebar_position_combo.currentData(),
            scalebar_show_bg=self._scalebar_bg_check.isChecked(),
            scalebar_padding=self._scalebar_padding_control.value(),
        )

    def _on_live_field_changed(self, *_args: object) -> None:
        if self._loading:
            return
        if self._index is not None:
            self.preview_changed.emit(self._index, self._current_meta())
        elif self._current_defaults_kind is not None:
            meta = self._current_meta()
            self._default_meta_by_kind[self._current_defaults_kind] = meta
            self.defaults_changed.emit(self._current_defaults_kind, meta)

    def open_defaults_for(self, kind: str) -> None:
        """
        Switch this embedded panel to *kind*'s own default template —
        see MeasurementsWidget's "Customize Default <kind>" panel. Each
        kind's edits are kept (in _default_meta_by_kind, updated live by
        _on_live_field_changed as they're made) so switching back to a
        previously-edited kind shows its own edits again, not fresh
        defaults.
        """
        if kind == self._current_defaults_kind:
            return
        self._loading = True
        self._current_defaults_kind = kind
        self._populate_fields(kind, self._default_meta_by_kind.get(kind, DEFAULT_META))
        self._loading = False
        self.adjustSize()

    def default_meta_for(self, kind: str) -> MeasurementMeta:
        """kind's own stored default template, or DEFAULT_META if it's never been edited."""
        return self._default_meta_by_kind.get(kind, DEFAULT_META)

    def _on_apply_clicked(self) -> None:
        if self._index is None:
            return
        index = self._index
        self._index = None
        self._original_meta = None
        self.hide()
        self.applied.emit(index, self._current_meta())

    def _on_reset_clicked(self) -> None:
        if self._index is not None:
            self.reset_requested.emit(self._index)

    def closeEvent(self, event) -> None:
        # The window's own close button behaves like Cancel: revert any
        # live preview and drop the tracked measurement.
        if self._index is not None and self._original_meta is not None:
            self.preview_changed.emit(self._index, self._original_meta)
        had_index = self._index is not None
        self._index = None
        self._original_meta = None
        if had_index:
            self.cancelled.emit()
        super().closeEvent(event)

    def _on_delete_clicked(self) -> None:
        if self._index is None:
            return
        # Holding Shift while clicking skips the confirmation entirely.
        skip_confirm = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)
        if not skip_confirm:
            confirm = QMessageBox.question(
                self, "Delete measurement", "Delete this measurement?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
        index = self._index
        self._index = None
        self._original_meta = None
        self.hide()
        self.delete_requested.emit(index)

    def _on_cancel_clicked(self) -> None:
        if self._index is not None and self._original_meta is not None:
            self.preview_changed.emit(self._index, self._original_meta)
        self._index = None
        self._original_meta = None
        self.hide()
        self.cancelled.emit()