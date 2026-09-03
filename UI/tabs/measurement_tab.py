from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QFrame,
)
from UI.tabs.base_tab import CameraWithSidebarPage

# Narrower than the shared RIGHT_SIDEBAR_WIDTH (380px) — this tab's tile
# grid only needs GRID_COLUMNS(3) * TILE_WIDTH(84) + margins wide, so the
# full shared width leaves the tab looking sparse; see _wrap_scroll for
# the scrollbar-width compensation that keeps this figure accurate even
# once a vertical scrollbar appears.
_SIDEBAR_WIDTH = 340

from UI.widgets.collapsible_section import CollapsibleSection
from UI.widgets.capture_control_widget import CaptureControlWidget
from UI.widgets.measurements_widget import MeasurementsWidget
from UI.widgets.measurements.measurement_meta import MeasurementMeta
from UI.widgets.preview_overlay.interaction_mode import MEASUREMENT_MODE, ModeToken

from common.app_context import get_app_context, open_settings


class MeasurementTab(CameraWithSidebarPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        self._capture_control = CaptureControlWidget()
        self._measurements = MeasurementsWidget()
        self._mode_token: ModeToken | None = None

        self._measurements.selection_changed.connect(self._on_measurement_selected)
        self._measurements.default_meta_changed.connect(self._on_default_meta_changed)
        self._on_default_meta_changed(self._measurements.current_default_meta())

        self._measurements.dpi_value_submitted.connect(self._capture_control.submit_dpi_value)
        self._measurements.manual_calibration_started.connect(self._capture_control.request_manual_calibration)
        self._measurements.calibration_dpi_submitted.connect(self._capture_control.submit_calibration_dpi)
        self._measurements.calibration_cancelled.connect(self._capture_control.cancel_calibration)

        self._capture_control.dpi_changed.connect(self._measurements.set_dpi_display)
        self._capture_control.calibration_line_ready.connect(self._measurements.set_calibration_line_ready)
        self._capture_control.loaded_dpi_missing.connect(self._measurements.expand_calibration_panel)
        self._measurements.set_dpi_display(None, True)

        super().__init__(self._make_sidebar(), parent)

    def _make_sidebar(self) -> QWidget:
        sidebar_container = QWidget()
        sidebar_container.setFixedWidth(_SIDEBAR_WIDTH)

        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(10)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        capture_control = CollapsibleSection("Capture Control", on_settings=lambda: open_settings("Camera"))
        capture_control.layout_for_content().addWidget(self._capture_control)
        content_layout.addWidget(capture_control)

        measurements = CollapsibleSection("Measurements")
        measurements.layout_for_content().addWidget(self._measurements)
        content_layout.addWidget(measurements)

        content_layout.addStretch(1)
        sidebar_layout.addWidget(self._wrap_scroll(content, sidebar_container), 1)
        return sidebar_container

    def _wrap_scroll(self, widget: QWidget, sidebar: QWidget) -> QScrollArea:
        """
        Disables the horizontal scrollbar outright and grows *sidebar*
        by exactly the vertical scrollbar's own width whenever one
        appears — same pattern as every other tab's sidebar (see
        ProjectTab._wrap_scroll) — rather than letting a vertical
        scrollbar eat into the tile grid's already-tight width budget
        and force a second, horizontal scrollbar.
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(widget)

        scrollbar_width = scroll.style().pixelMetric(
            scroll.style().PixelMetric.PM_ScrollBarExtent
        )

        def _on_range_changed(min_val: int, max_val: int) -> None:
            needed = max_val > min_val
            sidebar.setFixedWidth(_SIDEBAR_WIDTH + (scrollbar_width if needed else 0))
            self.set_sidebar_flush_right(needed)

        scroll.verticalScrollBar().rangeChanged.connect(_on_range_changed)
        return scroll

    # ------------------------------------------------------------------
    # The loaded-image overlay lives on the shared CameraPreview, so it
    # must only be switched on while this tab is the one actually showing
    # that preview — tied to show/hide rather than CollapsibleSection's
    # collapse state, which shouldn't affect it. Click-to-move and
    # scroll-to-move-Z are suppressed the same way, live feed or loaded
    # image alike — see UI.widgets.preview_overlay.interaction_mode.MEASUREMENT_MODE.
    # ------------------------------------------------------------------

    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        self._capture_control.set_tab_active(True)
        preview = get_app_context().camera_preview
        if preview is not None:
            self._mode_token = preview.modes.push(MEASUREMENT_MODE)

    def hideEvent(self, event: QEvent) -> None:
        super().hideEvent(event)
        self._capture_control.set_tab_active(False)
        if self._mode_token is not None:
            self._mode_token.pop()
            self._mode_token = None

    def _on_measurement_selected(self, kind: str | None) -> None:
        preview = get_app_context().camera_preview
        if preview is not None:
            preview.overlays.measurement.type = kind
        if kind is not None:
            self._capture_control.cancel_calibration()

    def _on_default_meta_changed(self, meta: MeasurementMeta) -> None:
        preview = get_app_context().camera_preview
        if preview is not None:
            preview.overlays.measurement.set_default_meta(meta)
            # meta.unit is always concrete now that the Customize panel's
            # own combo is the only source of a default unit (see
            # MeasurementsWidget.current_default_meta) — keeps
            # MeasurementOverlay._unit's fallback (used only when an
            # imported measurement has no unit override) in sync with
            # whatever the panel currently has selected.
            preview.overlays.measurement.set_unit(meta.unit)