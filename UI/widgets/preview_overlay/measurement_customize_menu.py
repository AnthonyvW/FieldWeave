from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QPoint, QPointF, QSize
from PySide6.QtGui import QColor, QIcon, QMouseEvent, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from UI.widgets.measurements.lines import MEASUREMENT_LINE_CAPS, MEASUREMENT_MIDPOINT_STYLES
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
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pattern = MEASUREMENT_DASH_PATTERNS.get(dash_style)
    if pattern:
        pen.setStyle(Qt.PenStyle.CustomDashLine)
        pen.setDashPattern(pattern)
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

    # "arrow_open"'s shaft runs the full way to the tip, flat-ended so
    # it stays sharp rather than blunted by a round cap — matching the
    # overlay, where the barbs alone form the point.
    pen = QPen(color)
    pen.setWidth(2)
    if cap == "square":
        pen.setCapStyle(Qt.PenCapStyle.SquareCap)
    elif cap == "arrow_open":
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    else:
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.drawLine(QPointF(x1, y), tip if cap == "arrow_open" else QPointF(x2, y))

    if cap == "arrow":
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([tip, QPointF(x2 - 4, y - 5), QPointF(x2 - 4, y + 5)]))
    elif cap == "arrow_open":
        open_pen = QPen(color)
        open_pen.setWidth(2)
        open_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(open_pen)
        painter.drawLine(tip, QPointF(x2 - 4, y - 5))
        painter.drawLine(tip, QPointF(x2 - 4, y + 5))

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
        row.addWidget(self._slider, 1)

        self._spin = QDoubleSpinBox()
        self._spin.setRange(minimum, maximum)
        self._spin.setSingleStep(1.0 / self._STEPS_PER_UNIT * 5)
        self._spin.setDecimals(1)
        self._spin.valueChanged.connect(self._on_spin_changed)
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MeasurementCustomizeMenu")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        # QFrame subclasses don't pick up a stylesheet background on
        # their own unless told to paint one — without this the panel
        # stayed transparent regardless of #MeasurementCustomizeMenu's
        # background rule in style.py.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._index: int | None = None
        self._original_meta: MeasurementMeta | None = None
        # Suppresses live preview_changed emissions while open_for is
        # populating the fields.
        self._loading: bool = False
        # Once the user drags the panel out of the way (feature 2), it
        # stops auto-following its tag until reopened.
        self._manually_moved: bool = False
        self._drag_origin: QPoint | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        layout.addWidget(_field_label("Title"))
        self._title_edit = QLineEdit()
        self._title_edit.setObjectName("MeasurementCustomizeTitle")
        self._title_edit.setPlaceholderText("Label this measurement...")
        layout.addWidget(self._title_edit)

        layout.addWidget(_field_label("Description"))
        self._description_edit = _ResizableDescriptionEdit()
        self._description_edit.setObjectName("MeasurementCustomizeDescription")
        self._description_edit.setFixedHeight(60)
        self._description_edit.resized.connect(self.adjustSize)
        layout.addWidget(self._description_edit)

        unit_row = QHBoxLayout()
        unit_row.addWidget(_field_label("Unit"))
        self._unit_combo = QComboBox()
        for unit in MeasurementUnit:
            self._unit_combo.addItem(unit.value, unit)
        self._unit_combo.currentIndexChanged.connect(self._on_live_field_changed)
        unit_row.addWidget(self._unit_combo, 1)
        unit_row.addWidget(_field_label("Decimals"))
        self._decimals_spin = QSpinBox()
        self._decimals_spin.setRange(0, 6)
        self._decimals_spin.valueChanged.connect(self._on_live_field_changed)
        unit_row.addWidget(self._decimals_spin)
        layout.addLayout(unit_row)

        self._area_check = QCheckBox("Show area")
        self._area_check.toggled.connect(self._on_area_toggled)
        area_row = QHBoxLayout()
        area_row.addWidget(self._area_check)
        self._area_unit_combo = QComboBox()
        for unit in MeasurementUnit:
            self._area_unit_combo.addItem(unit.value, unit)
        self._area_unit_combo.currentIndexChanged.connect(self._on_live_field_changed)
        area_row.addWidget(self._area_unit_combo, 1)
        layout.addLayout(area_row)

        self._always_show_description_check = QCheckBox("Always show description")
        self._always_show_description_check.toggled.connect(self._on_live_field_changed)
        layout.addWidget(self._always_show_description_check)

        self._hidden_check = QCheckBox("Hide tag")
        self._hidden_check.toggled.connect(self._on_live_field_changed)
        layout.addWidget(self._hidden_check)

        layout.addWidget(_field_label("Opacity"))
        self._opacity_control = _ThicknessControl(0.0, 1.0)
        self._opacity_control.value_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._opacity_control)

        self._build_tag_style_controls(layout)
        self._build_line_style_controls(layout)

        button_row = QHBoxLayout()
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self._on_apply_clicked)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self._on_cancel_clicked)
        button_row.addWidget(apply_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

        self.hide()

    def _build_tag_style_controls(self, layout: QVBoxLayout) -> None:
        self._tag_transparent_check = QCheckBox("Transparent tag background")
        self._tag_transparent_check.toggled.connect(self._on_tag_transparent_toggled)
        layout.addWidget(self._tag_transparent_check)

        self._tag_bg_picker = _ColorPicker("Tag Background Color", OVERLAY_LINE_COLOR.name())
        self._tag_bg_picker.color_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._tag_bg_picker)

        self._tag_text_picker = _ColorPicker("Tag Text Color", OVERLAY_OUTLINE_COLOR.name())
        self._tag_text_picker.color_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._tag_text_picker)

    def _build_line_style_controls(self, layout: QVBoxLayout) -> None:
        self._line_color_picker = _ColorPicker("Line Color", OVERLAY_LINE_COLOR.name())
        self._line_color_picker.color_changed.connect(self._on_live_field_changed)
        layout.addWidget(self._line_color_picker)

        layout.addWidget(_field_label("Line Thickness"))
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
        entry = DEFAULT_REGISTRY.get(kind)
        show_caps = entry is not None and entry.category == "line"
        show_area = entry is not None and entry.category == "circle"
        self._title_edit.setText(meta.title)
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
        self._opacity_control.set_value(meta.opacity)
        self._tag_transparent_check.setChecked(meta.tag_background_transparent)
        self._tag_bg_picker.setVisible(not meta.tag_background_transparent)
        self._tag_bg_picker.set_color(meta.tag_background_color)
        self._tag_text_picker.set_color(meta.tag_text_color)
        self._line_color_picker.set_color(meta.line_color)
        self._line_thickness_control.set_value(meta.line_thickness or OVERLAY_LINE_WIDTH)
        self._line_style_picker.set_value(meta.line_dash_style)
        # A point has no line to dash, so its dash-style picker is hidden;
        # circles and lines both keep it.
        is_point = entry is not None and entry.category == "point"
        self._line_style_picker.setVisible(not is_point)
        self._midpoint_picker.set_value(meta.midpoint_style)
        self._midpoint_picker.setVisible(show_caps)
        self._start_cap_picker.set_value(meta.line_start_cap)
        self._start_cap_picker.setVisible(show_caps)
        self._end_cap_picker.set_value(meta.line_end_cap)
        self._end_cap_picker.setVisible(show_caps)
        self._outline_enabled_check.setChecked(meta.outline_enabled)
        self._set_outline_controls_visible(meta.outline_enabled)
        self._outline_color_picker.set_color(meta.outline_color)
        self._outline_thickness_control.set_value(meta.outline_thickness or OVERLAY_OUTLINE_WIDTH)
        self._loading = False
        self.adjustSize()
        self.reposition(anchor)
        self.show()
        self.raise_()
        self._title_edit.setFocus()

    def reposition(self, anchor: QPoint) -> None:
        """Move without touching any field's value — used both by open_for and by CameraPreview to keep the panel following its tag while the user pans/zooms with it still open. A no-op once the user has dragged the panel themselves (feature 2)."""
        if self._manually_moved:
            return
        target = QPoint(anchor.x() - self.width() // 2, anchor.y() + self._ANCHOR_GAP_PX)
        self.move(self._clamped(target))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Press on the panel's own background (not a child control) begins dragging it out of the way — feature 2."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.globalPosition().toPoint() - self.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin is not None:
            self._manually_moved = True
            self.move(self._clamped(event.globalPosition().toPoint() - self._drag_origin))
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

    def _clamped(self, pos: QPoint) -> QPoint:
        """Keep the panel fully within its parent regardless of where the tag that opened it sits, since a tag near an edge would otherwise push it partly off screen."""
        parent = self.parentWidget()
        if parent is None:
            return pos
        max_x = max(0, parent.width() - self.width())
        max_y = max(0, parent.height() - self.height())
        return QPoint(min(max(pos.x(), 0), max_x), min(max(pos.y(), 0), max_y))

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

    def _current_meta(self) -> MeasurementMeta:
        # Built from the original so per-measurement state the menu
        # doesn't edit — style_id and the tag's dragged offset — carries
        # through an edit rather than resetting to the default.
        base = self._original_meta if self._original_meta is not None else DEFAULT_META
        return base._replace(
            title=self._title_edit.text().strip(),
            description=self._description_edit.toPlainText().strip(),
            unit=self._unit_combo.currentData(),
            tag_background_color=self._tag_bg_picker.color(),
            tag_text_color=self._tag_text_picker.color(),
            always_show_description=self._always_show_description_check.isChecked(),
            line_color=self._line_color_picker.color(),
            line_thickness=self._line_thickness_control.value(),
            outline_enabled=self._outline_enabled_check.isChecked(),
            outline_color=self._outline_color_picker.color(),
            outline_thickness=self._outline_thickness_control.value(),
            line_dash_style=self._line_style_picker.value(),
            line_start_cap=self._start_cap_picker.value(),
            line_end_cap=self._end_cap_picker.value(),
            decimal_places=self._decimals_spin.value(),
            hidden=self._hidden_check.isChecked(),
            opacity=self._opacity_control.value(),
            tag_background_transparent=self._tag_transparent_check.isChecked(),
            midpoint_style=self._midpoint_picker.value(),
            show_area=self._area_check.isChecked(),
            area_unit=self._area_unit_combo.currentData(),
        )

    def _on_live_field_changed(self, *_args: object) -> None:
        if self._index is not None and not self._loading:
            self.preview_changed.emit(self._index, self._current_meta())

    def _on_apply_clicked(self) -> None:
        if self._index is None:
            return
        index = self._index
        self._index = None
        self._original_meta = None
        self.hide()
        self.applied.emit(index, self._current_meta())

    def _on_cancel_clicked(self) -> None:
        if self._index is not None and self._original_meta is not None:
            self.preview_changed.emit(self._index, self._original_meta)
        self._index = None
        self._original_meta = None
        self.hide()
        self.cancelled.emit()