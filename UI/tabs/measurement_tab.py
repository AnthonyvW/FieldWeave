from __future__ import annotations

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QFrame,
)
from UI.style import RIGHT_SIDEBAR_WIDTH
from UI.tabs.base_tab import CameraWithSidebarPage

from UI.widgets.collapsible_section import CollapsibleSection
from UI.widgets.capture_control_widget import CaptureControlWidget
from UI.widgets.measurements_widget import MeasurementsWidget
from UI.widgets.measurements.units import MeasurementUnit

from common.app_context import get_app_context, open_settings


class MeasurementTab(CameraWithSidebarPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        self._capture_control = CaptureControlWidget()
        self._measurements = MeasurementsWidget()

        self._measurements.selection_changed.connect(self._on_measurement_selected)
        self._measurements.unit_changed.connect(self._on_unit_changed)
        self._on_unit_changed(self._measurements.current_unit())

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
        sidebar_container.setFixedWidth(RIGHT_SIDEBAR_WIDTH)

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
        sidebar_layout.addWidget(self._wrap_scroll(content), 1)
        return sidebar_container

    def _wrap_scroll(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(widget)
        return scroll

    # ------------------------------------------------------------------
    # The loaded-image overlay lives on the shared CameraPreview, so it
    # must only be switched on while this tab is the one actually showing
    # that preview — tied to show/hide rather than CollapsibleSection's
    # collapse state, which shouldn't affect it. Click-to-move and
    # scroll-to-move-Z are suppressed the same way, live feed or loaded
    # image alike — see CameraPreview.OverlayController.measurement_tab_active.
    # ------------------------------------------------------------------

    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        self._capture_control.set_tab_active(True)
        preview = get_app_context().camera_preview
        if preview is not None:
            preview.overlays.measurement_tab_active = True

    def hideEvent(self, event: QEvent) -> None:
        super().hideEvent(event)
        self._capture_control.set_tab_active(False)
        preview = get_app_context().camera_preview
        if preview is not None:
            preview.overlays.measurement_tab_active = False

    def _on_measurement_selected(self, kind: str | None) -> None:
        preview = get_app_context().camera_preview
        if preview is not None:
            preview.overlays.measurement.type = kind
        if kind is not None:
            self._capture_control.cancel_calibration()

    def _on_unit_changed(self, unit: MeasurementUnit) -> None:
        preview = get_app_context().camera_preview
        if preview is not None:
            preview.overlays.measurement.set_unit(unit)