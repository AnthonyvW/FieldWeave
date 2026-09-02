from __future__ import annotations

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import (
    QFileDialog,
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
from UI.widgets.measurements.measurement_meta import MeasurementMeta
from UI.widgets.measurements.units import MeasurementUnit
from UI.widgets.preview_overlay.interaction_mode import MEASUREMENT_MODE, ModeToken

from common.app_context import get_app_context, open_settings
from common.logger import warning


class MeasurementTab(CameraWithSidebarPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        self._capture_control = CaptureControlWidget()
        self._measurements = MeasurementsWidget()
        self._mode_token: ModeToken | None = None

        self._measurements.selection_changed.connect(self._on_measurement_selected)
        self._measurements.unit_changed.connect(self._on_unit_changed)
        self._measurements.default_meta_changed.connect(self._on_default_meta_changed)
        self._on_unit_changed(self._measurements.current_unit())

        self._measurements.dpi_value_submitted.connect(self._capture_control.submit_dpi_value)
        self._measurements.manual_calibration_started.connect(self._capture_control.request_manual_calibration)
        self._measurements.calibration_dpi_submitted.connect(self._capture_control.submit_calibration_dpi)
        self._measurements.calibration_cancelled.connect(self._capture_control.cancel_calibration)
        self._measurements.export_measurements_requested.connect(self._on_export_measurements_clicked)
        self._measurements.import_measurements_requested.connect(self._on_import_measurements_clicked)
        self._measurements.export_image_requested.connect(self._on_export_image_clicked)

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

    def _on_unit_changed(self, unit: MeasurementUnit) -> None:
        preview = get_app_context().camera_preview
        if preview is not None:
            preview.overlays.measurement.set_unit(unit)

    def _on_default_meta_changed(self, meta: MeasurementMeta) -> None:
        preview = get_app_context().camera_preview
        if preview is not None:
            preview.overlays.measurement.set_default_meta(meta)

    # ------------------------------------------------------------------
    # Import/export — see MeasurementOverlayController and measurement_io.py
    # for the actual serialization/validation; this just supplies the
    # file dialogs and toast feedback.
    # ------------------------------------------------------------------

    def _on_export_measurements_clicked(self) -> None:
        preview = get_app_context().camera_preview
        if preview is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Measurements", "", "Measurement files (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        preview.overlays.measurement.export_measurements_to_file(path)
        ctx = get_app_context()
        if ctx.toast is not None:
            ctx.toast.success(path, title="Measurements Exported")

    def _on_import_measurements_clicked(self) -> None:
        preview = get_app_context().camera_preview
        if preview is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import Measurements", "", "Measurement files (*.json)")
        if not path:
            return
        result = preview.overlays.measurement.import_measurements_from_file(path)
        for issue in result.warnings:
            warning(f"[MeasurementImport] {issue}")
        ctx = get_app_context()
        if ctx.toast is None:
            return
        if result.warnings:
            ctx.toast.warning(f"{len(result.entries)} loaded, {len(result.warnings)} skipped — see log", title="Import Completed With Issues")
        else:
            ctx.toast.success(f"{len(result.entries)} measurement(s) loaded", title="Measurements Imported")

    def _on_export_image_clicked(self) -> None:
        preview = get_app_context().camera_preview
        ctx = get_app_context()
        if preview is None:
            return
        image = preview.export_measurement_image()
        if image is None:
            if ctx.toast is not None:
                ctx.toast.warning("Nothing to export yet", title="Export Image")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Image with Measurements", "", "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        image.save(path)
        if ctx.toast is not None:
            ctx.toast.success(path, title="Image Exported")