from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from UI.style import RIGHT_SIDEBAR_WIDTH
from UI.tabs.base_tab import CameraWithSidebarPage
from UI.widgets.collapsible_section import CollapsibleSection
from UI.widgets.measurements.units import MeasurementUnit, dpi_from_measurement
from UI.widgets.navigation_widget import NavigationWidget
from UI.widgets.preview_overlay.interaction_mode import CALIBRATION_LINE_MODE, PreviewModeSpec, ModeToken
from UI.widgets.preview_overlay.measurement_customize_menu import block_wheel
from common.app_context import get_app_context, open_settings
from common.logger import error, info
from motion.models import Position
from motion.routines.inspection_calibration_scale_routine import InspectionCalibrationScaleRoutine

_NM_PER_MM = 1_000_000

_STEP_INFO: dict[str, tuple[str, str]] = {
    "position": (
        "Position the calibration target",
        "Place the calibration slide in the calibration slide holder, then place it at the beginning of one of the slots. It is recommended to use the rightmost slot.",
    ),
    "align": (
        "Align the camera",
        "Use the movement controls to center the calibration slide in the camera preview. Ensure that the center of the image is either at the first, or the last tick mark in the calibration pattern.",
    ),
    "choose": (
        "Choose Calibration Method",
        "Automatic runs the calibration slide routine below and measures DPI from the resulting image mosaic — more precise, but takes a few minutes. Manual instead derives DPI from a single reference line you place on the preview yourself — faster, and lets you skip the automatic routine entirely.",
    ),
    "tick": (
        "Calibrate tick detection",
        "Adjust the slider until all tick marks on the calibration slide are highlighted in the camera preview. Reduce the value if marks are missing; increase it if false detections appear.",
    ),
    "capture": (
        "Calculate DPI",
        "The system will capture a sequence of images and measure the DPI from the resulting image mosaic.",
    ),
    "qc": (
        "Quality Control",
        "Inspect the resulting mosaic and ensure all tick marks are present. Press View Image to open it. The next step lets you optionally refine or verify the measured DPI with a manually-placed reference line before finishing.",
    ),
    "manual": (
        "Manual DPI Calibration",
        "This step is optional — if the DPI value below already looks correct, you can press Finish Calibration right away. Otherwise, click two points on the preview to place a reference line, enter the real-world length it represents, then press Calculate DPI. You can use the zoom tools on the left side of the camera view, or hold Ctrl and scroll the mouse wheel, to zoom in and out for a more precise placement.",
    ),
}

_DESCRIPTION = (
    "<b>Purpose:</b><br>"
    "Calculates the camera's DPI at the current magnification.<br><br>"
    "<b>What it does:</b><br>"
    "• Measures pixels per millimetre<br><br>"
    "<b>What you need:</b><br>"
    "• A 10mm micrometer calibration slide with 0.1mm tick marks.<br>"
    "• Approximately 3 minutes<br><br>"
)


class _ResultsDialog(QDialog):
    """Modal dialog showing the outcome of a completed DPI calibration routine."""

    def __init__(self, result: object, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("DPI Calibration Results")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        success: bool = result.success
        dpi: float | None = result.get("dpi")
        qa_pass: bool = result.get("qa_pass", False)
        qa_warnings: list[str] = result.get("qa_warnings") or []
        tick_count: int | None = result.get("tick_count")
        image_width: int | None = result.get("image_width")
        image_height: int | None = result.get("image_height")
        output_path: str | None = result.get("output_path")

        if success:
            status_text = "PASS" if qa_pass else "FAIL — QA checks failed"
            status_name = "CalScaleStatusPass" if qa_pass else "CalScaleStatusFail"
        else:
            status_text = "Routine did not complete successfully"
            status_name = "CalScaleStatusFail"

        title = QLabel(status_text)
        title.setObjectName(status_name)
        layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setObjectName("SampleDivider")
        layout.addWidget(line)

        form = QFormLayout()
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        def _row(label: str, value: str) -> None:
            lbl = QLabel(label + ":")
            lbl.setObjectName("CalScaleRowLabel")
            val = QLabel(value)
            val.setObjectName("CalScaleRowValue")
            val.setWordWrap(True)
            form.addRow(lbl, val)

        if dpi is not None:
            _row("DPI", f"{dpi:.2f}")
        if tick_count is not None:
            from post_processing.routines.stitch_and_measure import EXPECTED_TICK_COUNT
            _row("Ticks", f"{tick_count} / {EXPECTED_TICK_COUNT}")
        if image_width is not None and image_height is not None:
            _row("Image size", f"{image_width} \u00d7 {image_height} px")
        if output_path is not None:
            _row("Output", output_path)

        layout.addLayout(form)

        if qa_warnings:
            warn_line = QFrame()
            warn_line.setFrameShape(QFrame.Shape.HLine)
            warn_line.setObjectName("SampleDivider")
            layout.addWidget(warn_line)

            # More than a few warnings would otherwise grow this dialog
            # past the screen — scroll them instead, capped to a
            # reasonable height, once there's enough to actually need it.
            warnings_host: QVBoxLayout
            if len(qa_warnings) > 3:
                warnings_container = QWidget()
                warnings_host = QVBoxLayout(warnings_container)
                warnings_host.setContentsMargins(0, 0, 0, 0)
                warnings_host.setSpacing(4)
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setFrameShape(QFrame.Shape.NoFrame)
                scroll.setMaximumHeight(150)
                scroll.setWidget(warnings_container)
                layout.addWidget(scroll)
            else:
                warnings_host = layout

            for msg in qa_warnings:
                warn_label = QLabel(f"Warning: {msg}")
                warn_label.setObjectName("CalErrorLabel")
                warn_label.setWordWrap(True)
                warnings_host.addWidget(warn_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class DpiCalibrationWidget(QWidget):
    """Info / launch widget shown in the calibration tab list."""

    calibration_started: Signal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._idle_poll_timer = QTimer(self)
        self._idle_poll_timer.setInterval(1000)
        self._idle_poll_timer.timeout.connect(self.refresh)
        self._idle_poll_timer.start()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Precise DPI Calibration")
        title.setObjectName("CalPageTitle")
        layout.addWidget(title)

        self._description_label = QLabel(_DESCRIPTION)
        self._description_label.setWordWrap(True)
        self._description_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._description_label.setObjectName("CalDescriptionBox")
        layout.addWidget(self._description_label)
        layout.addStretch()

        start_btn = QPushButton("Start Calibration")
        start_btn.setObjectName("CalStartCalibration")
        start_btn.setMinimumHeight(45)
        start_btn.clicked.connect(self.calibration_started)
        layout.addWidget(start_btn)

    def refresh(self) -> None:
        last_done = self._read_last_calibrated()
        if last_done:
            full_text = _DESCRIPTION + f"<br><br><b>Last calibrated:</b><br>{last_done}"
        else:
            full_text = _DESCRIPTION + "<br><br><b>Last calibrated:</b><br>Never"
        self._description_label.setText(full_text)

    def _read_last_calibrated(self) -> str | None:
        ctx = get_app_context()
        ts = ctx.machine_vision.settings.inspect_calibration.last_calibrated
        if not ts:
            return None
        dt = datetime.fromisoformat(ts)
        return dt.astimezone().strftime("%Y-%m-%d  %H:%M:%S")

    def reset(self) -> None:
        self.refresh()


class DpiCalibrationStepsWidget(QWidget):
    """Step-through widget embedded in the sidebar of the calibration page."""

    finished: Signal = Signal()

    def __init__(
        self,
        *,
        on_title_changed=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._current_step: int = 0
        # Which tail of steps follows "choose" — None until chosen there,
        # "automatic" for the existing slide-routine flow (tick, capture,
        # qc), "manual" for a single reference line instead (skipping the
        # automatic routine entirely) — see _active_steps.
        self._method: str | None = None
        self._capture_complete: bool = False
        self._output_folder: str | None = None
        self._routine = None
        self._last_result = None
        self._on_title_changed = on_title_changed
        self._mode_token: ModeToken | None = None
        self._calibration_mode_token: ModeToken | None = None
        self._build_ui()
        self._update_step_display()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._step_title = QLabel()
        self._step_title.setObjectName("CalStepTitle")
        layout.addWidget(self._step_title)

        self._step_body = QLabel()
        self._step_body.setWordWrap(True)
        self._step_body.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._step_body.setObjectName("CalStepBody")
        layout.addWidget(self._step_body)

        self._position_widget = self._build_position_widget()
        layout.addWidget(self._position_widget)

        self._choose_widget = self._build_choose_widget()
        layout.addWidget(self._choose_widget)

        self._tick_calibration_widget = self._build_tick_calibration_widget()
        layout.addWidget(self._tick_calibration_widget)

        self._capture_widget = self._build_capture_widget()
        layout.addWidget(self._capture_widget)

        self._qc_widget = self._build_qc_widget()
        layout.addWidget(self._qc_widget)

        self._manual_widget = self._build_manual_widget()
        layout.addWidget(self._manual_widget)

        nav_layout = QHBoxLayout()
        # Shares this same slot with _prev_btn (mutually exclusive via
        # visibility, the same way _next_btn/_finish_btn already share
        # theirs) — the first step has nothing to go "Previous" from, so
        # a disabled Previous button sat there uselessly; this replaces
        # it with a real action instead.
        self._cancel_calibration_btn = QPushButton("Cancel Calibration")
        self._cancel_calibration_btn.clicked.connect(self._on_cancel_calibration_clicked)
        nav_layout.addWidget(self._cancel_calibration_btn)

        self._prev_btn = QPushButton("Previous")
        self._prev_btn.clicked.connect(self._previous_step)
        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(self._next_step)
        self._finish_btn = QPushButton("Finish Calibration")
        self._finish_btn.clicked.connect(self._on_finish_clicked)

        nav_layout.addWidget(self._prev_btn)
        nav_layout.addWidget(self._next_btn)
        nav_layout.addWidget(self._finish_btn)
        layout.addLayout(nav_layout)

        self._status_label = QLabel("")
        self._status_label.setObjectName("CalStatusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._status_label.setWordWrap(True)
        self._status_label.hide()
        layout.addWidget(self._status_label)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_capture_state)

    def _build_position_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._saved_pos_label = QLabel("Saved position: Not set")
        self._saved_pos_label.setObjectName("CalSavedPosLabel")
        layout.addWidget(self._saved_pos_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._set_pos_btn = QPushButton("Set Position")
        self._set_pos_btn.setObjectName("CalSecondaryButton")
        self._set_pos_btn.setMinimumHeight(34)
        self._set_pos_btn.setToolTip("Save the current stage XYZ as the calibration start position")
        self._set_pos_btn.clicked.connect(self._on_set_position_clicked)
        btn_row.addWidget(self._set_pos_btn)

        self._goto_pos_btn = QPushButton("Go to Position")
        self._goto_pos_btn.setObjectName("CalSecondaryButton")
        self._goto_pos_btn.setMinimumHeight(34)
        self._goto_pos_btn.setToolTip("Move the stage to the saved calibration position")
        self._goto_pos_btn.setEnabled(False)
        self._goto_pos_btn.clicked.connect(self._on_goto_position_clicked)
        btn_row.addWidget(self._goto_pos_btn)

        self._clear_pos_btn = QPushButton("Clear Position")
        self._clear_pos_btn.setObjectName("CalSecondaryButton")
        self._clear_pos_btn.setMinimumHeight(34)
        self._clear_pos_btn.setToolTip("Remove the saved calibration position")
        self._clear_pos_btn.setEnabled(False)
        self._clear_pos_btn.clicked.connect(self._on_clear_position_clicked)
        btn_row.addWidget(self._clear_pos_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)
        widget.hide()
        return widget

    def _build_choose_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        automatic_btn = QPushButton("Automatic DPI Calibration")
        automatic_btn.setObjectName("CalSecondaryButton")
        automatic_btn.setMinimumHeight(34)
        automatic_btn.clicked.connect(self._on_choose_automatic_clicked)
        layout.addWidget(automatic_btn)

        manual_btn = QPushButton("Manual DPI Calibration")
        manual_btn.setObjectName("CalSecondaryButton")
        manual_btn.setMinimumHeight(34)
        manual_btn.clicked.connect(self._on_choose_manual_clicked)
        layout.addWidget(manual_btn)

        widget.hide()
        return widget

    def _build_tick_calibration_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._tick_min_length_label = QLabel()
        self._tick_min_length_label.setObjectName("CalSavedPosLabel")
        layout.addWidget(self._tick_min_length_label)

        self._tick_slider = QSlider(Qt.Orientation.Horizontal)
        self._tick_slider.setMinimum(10)
        self._tick_slider.setMaximum(500)
        self._tick_slider.setSingleStep(5)
        self._tick_slider.setPageStep(20)
        self._tick_slider.valueChanged.connect(self._on_tick_slider_changed)
        layout.addWidget(self._tick_slider)

        widget.hide()
        return widget

    def _build_capture_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._start_capture_btn = QPushButton("Start DPI Calculation")
        self._start_capture_btn.setObjectName("CalStartCapture")
        self._start_capture_btn.setMinimumHeight(34)
        self._start_capture_btn.clicked.connect(self._on_start_capture_clicked)
        layout.addWidget(self._start_capture_btn)

        self._stop_capture_btn = QPushButton("Stop")
        self._stop_capture_btn.setObjectName("CalStopCapture")
        self._stop_capture_btn.setMinimumHeight(32)
        self._stop_capture_btn.clicked.connect(self._on_stop_capture_clicked)
        self._stop_capture_btn.setVisible(False)
        layout.addWidget(self._stop_capture_btn)

        widget.hide()
        return widget

    def _build_qc_widget(self) -> QWidget:
        """
        Purely informational now — the measured DPI, and buttons to
        inspect the capture itself. Whether to accept it or refine it
        manually happens on the dedicated "manual" step right after this
        one (see _build_manual_widget), rather than a calibration-line
        row bolted onto this one alongside Finish.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._qc_dpi_label = QLabel()
        self._qc_dpi_label.setObjectName("CalSavedPosLabel")
        layout.addWidget(self._qc_dpi_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._view_image_btn = QPushButton("View Image")
        self._view_image_btn.setObjectName("CalSecondaryButton")
        self._view_image_btn.setMinimumHeight(34)
        self._view_image_btn.setEnabled(False)
        self._view_image_btn.clicked.connect(self._on_view_image_clicked)
        btn_row.addWidget(self._view_image_btn)

        self._open_folder_btn = QPushButton("Open Folder")
        self._open_folder_btn.setObjectName("CalSecondaryButton")
        self._open_folder_btn.setMinimumHeight(34)
        self._open_folder_btn.setEnabled(False)
        self._open_folder_btn.clicked.connect(self._on_open_folder_clicked)
        btn_row.addWidget(self._open_folder_btn)

        layout.addLayout(btn_row)
        widget.hide()
        return widget

    def _build_manual_widget(self) -> QWidget:
        """
        The very last step of either path — for the manual-only path it
        follows straight from "Choose Calibration Method", skipping the
        automatic routine (tick detection, capture, mosaic QC) entirely;
        for the automatic path it follows "qc", as an optional place to
        refine or verify the just-measured DPI with a manually-placed
        line instead of accepting it as-is. Its own dpi spin
        (_manual_dpi_spin) is seeded from whatever's already on record
        each time this step is shown — see _refresh_manual_dpi_display —
        and is the one value Finish Calibration actually saves (see
        _on_finish_clicked), regardless of which path got here.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._manual_dpi_spin = QDoubleSpinBox()
        self._manual_dpi_spin.setMinimum(1.0)
        self._manual_dpi_spin.setMaximum(100_000.0)
        self._manual_dpi_spin.setDecimals(2)
        self._manual_dpi_spin.setSingleStep(10.0)
        self._manual_dpi_spin.setFixedWidth(110)
        self._manual_dpi_spin.setToolTip("DPI value that will be saved when you press Finish Calibration.")
        block_wheel(self._manual_dpi_spin)

        # No inline Apply here — "Calculate DPI" sits on its own line
        # below, next to the DPI field it fills in. No visible "Place
        # Calibration Line" toggle either — placement is armed by default
        # for this whole step (see _update_step_display), so there's
        # nothing left for the user to press.
        manual_row, self._manual_calibration_line_btn, value_spin, unit_combo = self._build_manual_calibration_row()
        layout.addWidget(manual_row)

        calc_row = QHBoxLayout()
        calc_row.setSpacing(6)
        calculate_btn = QPushButton("Calculate DPI")
        calculate_btn.setObjectName("CalSecondaryButton")
        calculate_btn.clicked.connect(
            lambda: self._on_calibration_apply_clicked(value_spin, unit_combo, self._manual_dpi_spin)
        )
        calc_row.addWidget(calculate_btn)
        calc_row.addWidget(QLabel("DPI:"))
        calc_row.addWidget(self._manual_dpi_spin)
        calc_row.addStretch()
        layout.addLayout(calc_row)

        widget.hide()
        return widget

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------

    def _restore_overlay_state(self) -> None:
        """Pop this wizard's mode, restoring whatever was active before it — safe to call more than once."""
        if self._mode_token is not None:
            self._mode_token.pop()
            self._mode_token = None

    def _apply_overlays_for_step(self, key: str) -> None:
        if key == "align":
            visible = frozenset({"crosshair"})
        elif key in ("tick", "capture"):
            visible = frozenset({"inspect_calibration"})
        else:
            visible = frozenset()

        self._restore_overlay_state()
        preview = get_app_context().camera_preview
        if preview is not None:
            self._mode_token = preview.modes.push(
                PreviewModeSpec(name=f"dpi-calibration-step-{key}", visible_overlays=visible)
            )

    # ------------------------------------------------------------------
    # Step 3 — tick calibration slots
    # ------------------------------------------------------------------

    def _refresh_tick_calibration_display(self) -> None:
        mv = get_app_context().machine_vision
        current = mv.settings.inspect_calibration.preview.tick_min_length
        self._tick_slider.blockSignals(True)
        self._tick_slider.setValue(current)
        self._tick_slider.blockSignals(False)
        self._tick_min_length_label.setText(f"Minimum tick length: {current} px")

    def _on_tick_slider_changed(self, value: int) -> None:
        self._tick_min_length_label.setText(f"Minimum tick length: {value} px")
        mv = get_app_context().machine_vision
        mv.settings.inspect_calibration.preview.tick_min_length = value

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _active_steps(self) -> list[str]:
        """
        The ordered step keys for whichever path is active. "choose" is
        always the last of the base three until a method is actually
        picked there (see _on_choose_automatic_clicked/_on_choose_manual_clicked),
        at which point its own tail is appended. "manual" is always the
        very last step of either path — the automatic path's own tail
        ends with it too, as an optional refinement/verification step
        after the measured DPI is already in hand (see
        _refresh_manual_dpi_display), rather than a calibration-line row
        bolted onto the QC step itself; skipping the whole automatic
        routine is just the case where "manual" is the tail on its own.
        """
        steps = ["position", "align", "choose"]
        if self._method == "manual":
            steps.append("manual")
        elif self._method == "automatic":
            steps += ["tick", "capture", "qc", "manual"]
        return steps

    def _update_step_display(self) -> None:
        steps = self._active_steps()
        key = steps[self._current_step]
        if key not in ("qc", "manual"):
            self._cancel_calibration_line_if_active()
        step_title, step_body = _STEP_INFO[key]
        self._step_title.setText(step_title)
        self._step_body.setText(step_body)
        # A word-wrapped QLabel's height doesn't always reliably
        # repropagate up through the nested CollapsibleSection/QScrollArea
        # chain when its text changes after the widget's already shown —
        # long text (e.g. "choose"'s) could end up vertically clipped
        # rather than wrapping into the extra lines it actually needs.
        # Forcing an explicit minimum height from heightForWidth at the
        # label's own current (already-laid-out) width sidesteps that.
        width = self._step_body.width()
        if width > 0:
            self._step_body.setMinimumHeight(self._step_body.heightForWidth(width))
        self._prev_btn.setVisible(self._current_step > 0)
        self._cancel_calibration_btn.setVisible(self._current_step == 0)

        if self._on_title_changed is not None:
            self._on_title_changed(
                f"DPI Calibration  {self._current_step + 1} / {len(steps)}"
            )

        is_choose = key == "choose"
        is_last = self._current_step == len(steps) - 1
        self._next_btn.setVisible(not is_choose and not is_last)
        self._finish_btn.setVisible(not is_choose and is_last)

        self._position_widget.setVisible(key == "align")
        if key == "align":
            self._refresh_position_display()

        self._choose_widget.setVisible(is_choose)

        self._tick_calibration_widget.setVisible(key == "tick")
        if key == "tick":
            self._refresh_tick_calibration_display()

        self._capture_widget.setVisible(key == "capture")
        if key == "capture":
            self._next_btn.setEnabled(self._capture_complete)
        else:
            self._next_btn.setEnabled(True)

        self._qc_widget.setVisible(key == "qc")
        if key == "qc":
            has_output = self._output_folder is not None
            self._view_image_btn.setEnabled(has_output)
            self._open_folder_btn.setEnabled(has_output)
            self._refresh_qc_dpi_display()

        self._manual_widget.setVisible(key == "manual")
        if key == "manual":
            # Seeds from whatever DPI is already on record — after the
            # automatic routine, that's the value it just measured (the
            # routine itself writes settings.dpi), so this step starts
            # from "does this already look right?" rather than blank;
            # for the manual-only path it's just whatever was last saved.
            self._refresh_manual_dpi_display()

        self._relayout_sidebar()
        self._apply_overlays_for_step(key)
        self._set_status("")
        if key == "manual" and not self._manual_calibration_line_btn.isChecked():
            # This step's whole point is placing (or re-placing) a line —
            # arm placement (and disable click-to-move) the moment it
            # appears rather than making the user press "Place
            # Calibration Line" first. Done last so its own status
            # message ("Click two points...") isn't immediately wiped by
            # the blank _set_status("") reset above.
            self._manual_calibration_line_btn.setChecked(True)

    def _next_step(self) -> None:
        steps = self._active_steps()
        if self._current_step < len(steps) - 1:
            if steps[self._current_step] == "tick":
                get_app_context().machine_vision.save_settings()
            self._current_step += 1
            self._update_step_display()

    def _previous_step(self) -> None:
        if self._current_step > 0:
            self._current_step -= 1
            if self._active_steps()[self._current_step] == "choose":
                # Stepping back onto "choose" means reconsidering it —
                # drop the tail so _active_steps reflects "not yet chosen"
                # again rather than silently keeping the old one.
                self._method = None
            self._update_step_display()

    def _on_choose_automatic_clicked(self) -> None:
        self._method = "automatic"
        self._current_step += 1
        self._update_step_display()

    def _on_choose_manual_clicked(self) -> None:
        self._method = "manual"
        self._current_step += 1
        self._update_step_display()

    def _relayout_sidebar(self) -> None:
        """
        Force the sidebar's QScrollArea to recompute its scrollable range
        — showing/hiding a whole step's worth of controls (e.g. switching
        onto "manual", which is taller than most other steps) doesn't
        always reliably repropagate a resize request up through the
        nested CollapsibleSection/QScrollArea chain on its own, which
        could otherwise leave the bottom of the newly-shown content
        clipped until some unrelated resize nudges it.
        """
        widget: QWidget | None = self
        while widget is not None:
            widget.updateGeometry()
            if isinstance(widget, QScrollArea):
                break
            widget = widget.parentWidget()

    def _on_cancel_calibration_clicked(self) -> None:
        """Bail out to the calibration selection list without saving anything — only offered on the very first step, so there's nothing in progress to lose."""
        self._cancel_calibration_line_if_active()
        self._restore_overlay_state()
        self.finished.emit()

    def teardown_active_placement(self) -> None:
        """Drop any in-progress manual-calibration placement and restore click-to-move — called when the whole page is hidden (switching to another main tab), since leaving the wizard mid-step otherwise wouldn't fire any of the navigation methods that normally do this."""
        self._cancel_calibration_line_if_active()

    def reset(self) -> None:
        self._restore_overlay_state()
        self._current_step = 0
        self._method = None
        self._capture_complete = False
        self._output_folder = None
        self._last_result = None
        self._update_step_display()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)
        self._status_label.setVisible(bool(text))

    # ------------------------------------------------------------------
    # Step 2 — position slots
    # ------------------------------------------------------------------

    def _get_saved_position(self) -> tuple[int, int, int] | None:
        ctx = get_app_context()
        if ctx is None or ctx.machine_vision is None:
            return None
        icp = ctx.machine_vision.settings.inspection_calibration_position
        if not icp.is_set:
            return None
        return icp.x_nm, icp.y_nm, icp.z_nm

    def _refresh_position_display(self) -> None:
        saved = self._get_saved_position()
        if saved is not None:
            x_mm = saved[0] / _NM_PER_MM
            y_mm = saved[1] / _NM_PER_MM
            z_mm = saved[2] / _NM_PER_MM
            self._saved_pos_label.setText(
                f"Saved X: {x_mm:.3f}  Y: {y_mm:.3f}  Z: {z_mm:.3f} mm"
            )
            self._goto_pos_btn.setEnabled(True)
            self._clear_pos_btn.setEnabled(True)
        else:
            self._saved_pos_label.setText("Saved position: Not set")
            self._goto_pos_btn.setEnabled(False)
            self._clear_pos_btn.setEnabled(False)

    def _on_set_position_clicked(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.is_ready():
            self._set_status("Motion controller not ready.")
            return

        pos = motion.get_position()
        
        mv = ctx.machine_vision
        mv.settings.inspection_calibration_position.x_nm = pos.x
        mv.settings.inspection_calibration_position.y_nm = pos.y
        mv.settings.inspection_calibration_position.z_nm = pos.z
        mv.settings.inspection_calibration_position.is_set = True
        mv.save_settings()
        info(
            f"[DpiCalibration] Position saved:"
            f" X={pos.x / _NM_PER_MM:.3f} mm"
            f" Y={pos.y / _NM_PER_MM:.3f} mm"
            f" Z={pos.z / _NM_PER_MM:.3f} mm"
        )
        self._refresh_position_display()
        self._set_status(
            f"Position saved:"
            f" ({pos.x / _NM_PER_MM:.3f},"
            f" {pos.y / _NM_PER_MM:.3f},"
            f" {pos.z / _NM_PER_MM:.3f}) mm"
        )

    def _on_goto_position_clicked(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.is_ready():
            self._set_status("Motion controller not ready.")
            return
        saved = self._get_saved_position()
        if saved is None:
            self._set_status("No calibration position saved.")
            return
        x_nm, y_nm, z_nm = saved
        try:
            motion.move_to_position(Position(x=x_nm, y=y_nm, z=z_nm), wait=False)
        except Exception as exc:
            error(f"DpiCalibration: move_to_position failed — {exc}")
            self._set_status("Move failed — see log.")
            return
        self._set_status(
            f"Moving to ({x_nm / _NM_PER_MM:.3f},"
            f" {y_nm / _NM_PER_MM:.3f},"
            f" {z_nm / _NM_PER_MM:.3f}) mm…"
        )

    def _on_clear_position_clicked(self) -> None:
        mv = get_app_context().machine_vision
        mv.settings.inspection_calibration_position.x_nm = 0
        mv.settings.inspection_calibration_position.y_nm = 0
        mv.settings.inspection_calibration_position.z_nm = 0
        mv.settings.inspection_calibration_position.is_set = False
        mv.save_settings()
        info("[DpiCalibration] Calibration position cleared")
        self._refresh_position_display()
        self._set_status("Calibration position cleared.")

    # ------------------------------------------------------------------
    # Step 4 — capture slots
    # ------------------------------------------------------------------

    def _on_start_capture_clicked(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.is_ready():
            self._set_status("Motion controller not ready.")
            return
        if not ctx.has_camera:
            self._set_status("No camera available.")
            return
        if self._get_saved_position() is None:
            self._on_set_position_clicked()
        saved = self._get_saved_position()
        start_position = Position(x=saved[0], y=saved[1], z=saved[2]) if saved is not None else None
        output_path = str(Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S"))
        self._routine = InspectionCalibrationScaleRoutine(
            motion=motion,
            output_path=output_path,
            start_position=start_position,
        )
        motion.start_routine(self._routine)
        self._output_folder = output_path
        self._start_capture_btn.setEnabled(False)
        self._stop_capture_btn.setVisible(True)
        self._prev_btn.setEnabled(False)
        self._next_btn.setEnabled(False)
        self._set_status("Running…")
        self._poll_timer.start()

    def _on_stop_capture_clicked(self) -> None:
        if self._routine is not None:
            self._routine.stop()
        self._set_status("Stopping…")

    def _poll_capture_state(self) -> None:
        if self._routine is None or not self._routine.is_running:
            self._poll_timer.stop()
            routine = self._routine
            self._routine = None
            self._start_capture_btn.setEnabled(True)
            self._stop_capture_btn.setVisible(False)
            self._prev_btn.setEnabled(self._current_step > 0)
            result = routine.result if routine is not None else None
            if result is not None and result.success:
                self._capture_complete = True
                self._last_result = result
                _ResultsDialog(result, parent=self).exec()
                self._next_step()
            else:
                self._last_result = result
                activity = routine.activity if routine is not None else ""
                if activity:
                    self._set_status(activity)
                if result is not None:
                    _ResultsDialog(result, parent=self).exec()
            return

        activity = self._routine.activity
        if activity:
            prog = self._routine.progress_current
            total = self._routine.progress_total
            if total > 0:
                self._set_status(f"[{prog}/{total}]  {activity}")
            else:
                self._set_status(activity)

    # ------------------------------------------------------------------
    # Step 5 — DPI display / finish
    # ------------------------------------------------------------------

    def _refresh_qc_dpi_display(self) -> None:
        dpi = get_app_context().machine_vision.settings.dpi
        self._qc_dpi_label.setText(f"Measured DPI: {dpi:.2f}" if dpi is not None else "Measured DPI: not set")

    def _refresh_manual_dpi_display(self) -> None:
        dpi = get_app_context().machine_vision.settings.dpi
        self._manual_dpi_spin.blockSignals(True)
        self._manual_dpi_spin.setValue(dpi if dpi is not None else self._manual_dpi_spin.minimum())
        self._manual_dpi_spin.blockSignals(False)

    def _on_finish_clicked(self) -> None:
        # "manual" is always the last step of either path now (see
        # _active_steps), so its own DPI field is the one and only value
        # actually saved here — whether that's a manually-placed line's
        # result or just the automatic routine's own measurement carried
        # over unedited (see _refresh_manual_dpi_display).
        self._cancel_calibration_line_if_active()
        mv = get_app_context().machine_vision
        mv.settings.dpi = self._manual_dpi_spin.value()
        mv.save_settings()
        info(f"[DpiCalibration] DPI saved as {mv.settings.dpi:.2f}")
        self._restore_overlay_state()
        self.finished.emit()

    # ------------------------------------------------------------------
    # Manual calibration — an alternative (or a refinement/verification
    # step after the automatic routine — see _build_manual_widget) to the
    # automated routine above: place a reference line of known length
    # directly on the preview and derive DPI from its pixel length, the
    # same way MeasurementTab's own "Calibrate DPI" panel does (see
    # CaptureControlWidget).
    # ------------------------------------------------------------------

    def _build_manual_calibration_row(self) -> tuple[QWidget, QPushButton, QDoubleSpinBox, QComboBox]:
        """
        A hidden "Place Calibration Line" toggle (this step's placement is
        armed automatically as soon as it appears — see
        _update_step_display — so there's nothing for the user to press,
        but its checked state is still how placement is tracked/armed/
        cancelled internally — see _cancel_calibration_line_if_active)
        plus a labeled real-world length/unit. "Calculate DPI" lives on
        its own line below instead of inline here — see
        _build_manual_widget, which wires it using the value_spin/
        unit_combo returned here.
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        line_btn = QPushButton("Place Calibration Line")
        line_btn.setObjectName("CalSecondaryButton")
        line_btn.setCheckable(True)
        line_btn.setVisible(False)
        line_btn.toggled.connect(self._on_calibration_line_toggled)
        layout.addWidget(line_btn)

        layout.addWidget(QLabel("Distance:"))

        value_spin = QDoubleSpinBox()
        value_spin.setRange(0.001, 100_000.0)
        value_spin.setDecimals(3)
        value_spin.setValue(1.0)
        value_spin.setFixedWidth(80)
        layout.addWidget(value_spin)

        unit_combo = QComboBox()
        for unit in MeasurementUnit:
            unit_combo.addItem(unit.value, unit)
        unit_combo.setCurrentIndex(unit_combo.findData(MeasurementUnit.MM))
        layout.addWidget(unit_combo)

        layout.addStretch()
        return widget, line_btn, value_spin, unit_combo

    def _cancel_calibration_line_if_active(self) -> None:
        """Tear down an in-progress manual-calibration placement — called whenever the wizard leaves the "manual" step (Next/Previous/Finish) so the calibration-line mode token never outlives that step."""
        if self._manual_calibration_line_btn.isChecked():
            self._manual_calibration_line_btn.setChecked(False)

    def _on_calibration_line_toggled(self, checked: bool) -> None:
        ctx = get_app_context()
        preview = ctx.camera_preview
        if preview is None:
            return
        if checked:
            # DPI is calibrated against the still-capture resolution, not
            # the (lower-resolution) live preview stream — MeasurementTab
            # keeps this refreshed while it's visible (CaptureControlWidget),
            # but this wizard may be the first place DPI is ever set in a
            # session, so refresh it here too.
            camera = ctx.camera
            if camera is not None and camera.underlying_camera.is_open:
                _, width, height = camera.settings.get_current_still_resolution()
                preview.overlays.measurement.set_live_reference_dims((width, height))
            self._calibration_mode_token = preview.modes.push(CALIBRATION_LINE_MODE)
            preview.overlays.measurement.start_calibration_placement()
            self._set_status("Click two points on the preview to place a reference line.")
        else:
            preview.overlays.measurement.cancel_calibration_placement()
            if self._calibration_mode_token is not None:
                self._calibration_mode_token.pop()
                self._calibration_mode_token = None
            self._set_status("")

    def _on_calibration_apply_clicked(
        self, value_spin: QDoubleSpinBox, unit_combo: QComboBox, target_dpi_spin: QDoubleSpinBox
    ) -> None:
        """
        Doesn't disarm placement afterward — placement is armed for this
        whole step (see _update_step_display), so a single calculation
        shouldn't stop the user from re-placing a more precise line and
        calculating again. It's only torn down when the step is actually
        left (Finish, Previous, or hiding the page) — see
        _cancel_calibration_line_if_active.
        """
        preview = get_app_context().camera_preview
        if preview is None:
            return
        pixel_length = preview.overlays.measurement.calibration_line_length_px()
        if pixel_length is None:
            self._set_status("Place the calibration line first.")
            return
        unit = unit_combo.currentData()
        dpi = dpi_from_measurement(pixel_length, value_spin.value(), unit)
        if dpi is None:
            self._set_status("Enter a positive measurement.")
            return
        target_dpi_spin.setValue(dpi)
        self._set_status(f"Calibration line applied — DPI set to {dpi:.2f}.")

    # ------------------------------------------------------------------
    # Step 5 — quality control slots
    # ------------------------------------------------------------------

    def _on_view_image_clicked(self) -> None:
        if self._output_folder is None:
            return
        folder = Path(self._output_folder)
        ctx = get_app_context()
        ext = ctx.camera.underlying_camera.settings.fformat.value if ctx.camera is not None else "jpg"
        image_path = folder / f"{folder.name}.{ext}"
        if not image_path.exists():
            self._set_status(f"Image not found: {image_path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(image_path)))

    def _on_open_folder_clicked(self) -> None:
        if self._output_folder is None:
            return
        folder = Path(self._output_folder)
        if not folder.exists():
            self._set_status(f"Folder not found: {folder}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


class DpiCalibrationPage(CameraWithSidebarPage):
    finished: Signal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(self._make_sidebar(), parent)
        self._steps_widget.finished.connect(self.finished)

    def start(self) -> None:
        self._steps_widget.reset()
        self.set_sidebar_flush_right(False)

    def hideEvent(self, event: QEvent) -> None:
        """
        Beyond CameraWithSidebarPage's own preview-detach: dropping this
        whole page (switching to the Navigate or Project tab mid-wizard)
        doesn't go through any of the steps widget's own navigation
        methods (Next/Previous/Finish), so an in-progress manual
        calibration-line placement — and the click-to-move suppression
        that comes with it — would otherwise be left armed indefinitely.
        """
        super().hideEvent(event)
        self._steps_widget.teardown_active_placement()

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

        navigation = CollapsibleSection("Navigation", on_settings=lambda: open_settings("Navigation"))
        navigation.layout_for_content().addWidget(NavigationWidget())
        content_layout.addWidget(navigation)

        calibration = CollapsibleSection("DPI Calibration Step 1 / 5")
        self._steps_widget = DpiCalibrationStepsWidget(
            on_title_changed=calibration.set_title
        )
        calibration.layout_for_content().addWidget(self._steps_widget)
        content_layout.addWidget(calibration)

        content_layout.addStretch(1)

        scroll = self._wrap_scroll(content, sidebar_container)
        sidebar_layout.addWidget(scroll, 1)
        return sidebar_container

    def _wrap_scroll(self, widget: QWidget, sidebar: QWidget) -> QScrollArea:
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
            sidebar.setFixedWidth(RIGHT_SIDEBAR_WIDTH + (scrollbar_width if needed else 0))
            self.set_sidebar_flush_right(needed)

        scroll.verticalScrollBar().rangeChanged.connect(_on_range_changed)
        return scroll