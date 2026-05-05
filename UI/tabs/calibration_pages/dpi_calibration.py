from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
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
from UI.widgets.navigation_widget import NavigationWidget
from common.app_context import get_app_context, open_settings
from common.logger import error, info
from motion.models import Position
from motion.routines.inspection_calibration_scale_routine import InspectionCalibrationScaleRoutine

_NM_PER_MM = 1_000_000

_STEPS: list[tuple[str, str]] = [
    (
        "Position the calibration target",
        "Place the calibration slide in the calibration slide holder, then place it at the beginning of one of the slots. It is recommended to use the rightmost slot.",
    ),
    (
        "Align the camera",
        "Use the movement controls to center the calibration slide in the camera preview. Ensure that the center of the image is either at the first, or the last tick mark in the calibration pattern.",
    ),
    (
        "Calibrate tick detection",
        "Adjust the slider until all tick marks on the calibration slide are highlighted in the camera preview. Reduce the value if marks are missing; increase it if false detections appear.",
    ),
    (
        "Calculate DPI",
        "The system will capture a sequence of images and measure the DPI from the resulting image mosaic.",
    ),
    (
        "Quality Control",
        "Inspect the resulting mosaic and ensure all tick marks are present. Press view image to open the image, and finish once you are done.",
    ),
]

_DESCRIPTION = (
    "<b>Purpose:</b><br>"
    "Calculates the camera's DPI at the current magnification.<br><br>"
    "<b>What it does:</b><br>"
    "• Measures pixels per millimetre<br><br>"
    "<b>What you need:</b><br>"
    "• A 10mm micrometer calibration slide with 0.1mm tick marks.<br>"
    "• Approximately 3 minutes<br><br>"
)


class DpiCalibrationWidget(QWidget):
    """Info / launch widget shown in the calibration tab list."""

    calibration_started: Signal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

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
        self._total_steps: int = len(_STEPS)
        self._capture_complete: bool = False
        self._output_folder: str | None = None
        self._routine = None
        self._on_title_changed = on_title_changed
        self._crosshair_state_before: bool | None = None
        self._inspect_calibration_state_before: bool | None = None
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

        self._tick_calibration_widget = self._build_tick_calibration_widget()
        layout.addWidget(self._tick_calibration_widget)

        self._capture_widget = self._build_capture_widget()
        layout.addWidget(self._capture_widget)

        self._qc_widget = self._build_qc_widget()
        layout.addWidget(self._qc_widget)

        nav_layout = QHBoxLayout()
        self._prev_btn = QPushButton("Previous")
        self._prev_btn.clicked.connect(self._previous_step)
        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(self._next_step)
        self._finish_btn = QPushButton("Finish Calibration")
        self._finish_btn.clicked.connect(self._restore_overlay_state)
        self._finish_btn.clicked.connect(self.finished)

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
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._dpi_label = QLabel("")
        self._dpi_label.setObjectName("CalSavedPosLabel")
        layout.addWidget(self._dpi_label)

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

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------

    def _get_overlays(self):
        ctx = get_app_context()
        preview = ctx.camera_preview
        return preview.overlays if preview is not None else None

    def _save_overlay_state(self) -> None:
        overlays = self._get_overlays()
        if overlays is not None:
            self._crosshair_state_before = overlays.crosshair
            self._inspect_calibration_state_before = overlays.inspect_calibration
        else:
            self._crosshair_state_before = None
            self._inspect_calibration_state_before = None

    def _restore_overlay_state(self) -> None:
        overlays = self._get_overlays()
        if overlays is None:
            return
        if self._crosshair_state_before is not None:
            overlays.crosshair = self._crosshair_state_before
            self._crosshair_state_before = None
        if self._inspect_calibration_state_before is not None:
            overlays.inspect_calibration = self._inspect_calibration_state_before
            self._inspect_calibration_state_before = None

    def _apply_overlays_for_step(self, step: int) -> None:
        overlays = self._get_overlays()
        if overlays is None:
            return
        if step == 1:
            overlays.crosshair = True
            overlays.inspect_calibration = False
        elif step == 2 or step == 3:
            overlays.crosshair = False
            overlays.inspect_calibration = True
        else:
            overlays.crosshair = False
            overlays.inspect_calibration = False

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

    def _update_step_display(self) -> None:
        step_title, step_body = _STEPS[self._current_step]
        self._step_title.setText(step_title)
        self._step_body.setText(step_body)
        self._prev_btn.setEnabled(self._current_step > 0)

        if self._on_title_changed is not None:
            self._on_title_changed(
                f"DPI Calibration  {self._current_step + 1} / {self._total_steps}"
            )

        is_last = self._current_step == self._total_steps - 1
        self._next_btn.setVisible(not is_last)
        self._finish_btn.setVisible(is_last)

        self._position_widget.setVisible(self._current_step == 1)
        if self._current_step == 1:
            self._refresh_position_display()

        self._tick_calibration_widget.setVisible(self._current_step == 2)
        if self._current_step == 2:
            self._refresh_tick_calibration_display()

        self._capture_widget.setVisible(self._current_step == 3)
        if self._current_step == 3:
            self._next_btn.setEnabled(self._capture_complete)
        else:
            self._next_btn.setEnabled(True)

        self._qc_widget.setVisible(self._current_step == 4)
        if self._current_step == 4:
            has_output = self._output_folder is not None
            self._view_image_btn.setEnabled(has_output)
            self._open_folder_btn.setEnabled(has_output)
            dpi = get_app_context().machine_vision.settings.dpi
            self._dpi_label.setText(f"DPI: {dpi:.1f}" if dpi is not None else "DPI: Unknown")

        self._apply_overlays_for_step(self._current_step)
        self._set_status("")

    def _next_step(self) -> None:
        if self._current_step < self._total_steps - 1:
            if self._current_step == 2:
                get_app_context().machine_vision.save_settings()
            self._current_step += 1
            self._update_step_display()

    def _previous_step(self) -> None:
        if self._current_step > 0:
            self._current_step -= 1
            self._update_step_display()

    def reset(self) -> None:
        self._restore_overlay_state()
        self._save_overlay_state()
        self._current_step = 0
        self._capture_complete = False
        self._output_folder = None
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
        try:
            self._routine = InspectionCalibrationScaleRoutine(
                motion=motion,
                output_path=output_path,
                start_position=start_position,
            )
            self._routine.on_state_changed = self._on_routine_state_changed
            motion.start_routine(self._routine)
        except Exception as exc:
            error(f"DpiCalibration: failed to start routine — {exc}")
            self._set_status(f"Failed to start: {exc}")
            return

        self._output_folder = output_path

        self._latest_activity: str = ""
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

    def _on_routine_state_changed(
        self,
        job_name: str,
        activity: str,
        progress_current: int,
        progress_total: int,
        eta_seconds: int,
    ) -> None:
        self._latest_activity = activity

    def _poll_capture_state(self) -> None:
        if self._routine is None or not self._routine.is_running:
            final_activity = getattr(self, "_latest_activity", "")
            self._poll_timer.stop()
            self._routine = None

            self._start_capture_btn.setEnabled(True)
            self._stop_capture_btn.setVisible(False)
            self._prev_btn.setEnabled(self._current_step > 0)

            succeeded = get_app_context().machine_vision.settings.dpi is not None

            if succeeded:
                self._capture_complete = True
                mv = get_app_context().machine_vision
                mv.settings.inspect_calibration.last_calibrated = datetime.now(tz=timezone.utc).isoformat()
                mv.save_settings()
                self._next_step()
            else:
                if final_activity:
                    self._set_status(final_activity)
            return

        activity = getattr(self, "_latest_activity", "")
        if activity:
            prog = self._routine.progress_current
            total = self._routine.progress_total
            if total > 0:
                self._set_status(f"[{prog}/{total}]  {activity}")
            else:
                self._set_status(activity)

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