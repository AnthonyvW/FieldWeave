from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from UI.style import RIGHT_SIDEBAR_WIDTH
from UI.tabs.base_tab import CameraWithSidebarPage
from UI.widgets.collapsible_section import CollapsibleSection
from UI.widgets.navigation_widget import NavigationWidget
from common.app_context import get_app_context, open_settings
from common.logger import error, info

_NM_PER_MM = 1_000_000

_STEPS: list[tuple[str, str]] = [
    (
        "Mount Calibration Slide",
        "Place the calibration slide into the calibration slide holder.\n\n"
        "Place calibration slide holder at the beginning of a slot. It is recommended that you use the one on the rightmost edge.",
    ),
    (
        "Centre the Slide in the Camera View",
        "Use the movement controls to position the calibration pattern at the centre of the camera preview.\n\n"
        "If you have done this before, you can use the \"Go to Position\" to quickly return to the old slide location. Adjust focus if necessary before continuing.",
    ),
    (
        "Start Calibration Capture",
        "Press \"Start Capture\" to begin the automated calibration sequence.\n\n",
    ),
    (
        "Verify the coordinate mapping",
        "Click a point in the camera preview to confirm that the stage moves to that location\n\n"
        "If the movement is inaccurate, repeat the capture step after readjusting the focus. Once satisfied, click \"Finish Calibration\".",
    ),
]

_DESCRIPTION = (
    "<b>Purpose:</b><br>"
    "Maps camera pixel coordinates to physical stage coordinates for accurate machine vision and click-to-move operation.<br><br>"
    "<b>What it does:</b><br>"
    "• Sets calibration slide position<br>"
    "• Enables click-to-move in the camera preview<br>"
    "• Enables additional machine vision features<br><br>"
    "<b>What you need:</b><br>"
    "• The calibration slide and slide holder<br>"
    "• Approximately 5 minutes<br><br>"
)


class _BodyText(QTextEdit):
    """Read-only text area that sizes itself to its content height.

    QLabel with word-wrap does not correctly report heightForWidth inside a
    scroll area. QTextEdit's document layout engine handles this natively —
    connecting to contentsChanged and adjusting minimumHeight is enough to
    make the widget grow with its content without any manual measurement.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(
            self.sizePolicy().horizontalPolicy(),
            self.sizePolicy().verticalPolicy(),
        )
        self.document().contentsChanged.connect(self._adjust_height)

    def _adjust_height(self) -> None:
        doc_h = int(self.document().size().height())
        margins = self.contentsMargins()
        h = doc_h + margins.top() + margins.bottom()
        self.setMinimumHeight(h)
        self.setMaximumHeight(h)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._adjust_height()


class CameraSpaceCalibrationWidget(QWidget):
    calibration_started: Signal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        title = QLabel("Camera Space Calibration")
        title.setObjectName("CalPageTitle")
        main_layout.addWidget(title)

        self._description_label = QLabel()
        self._description_label.setWordWrap(True)
        self._description_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._description_label.setObjectName("CalDescriptionBox")
        main_layout.addWidget(self._description_label)
        main_layout.addStretch()

        start_btn = QPushButton("Start Calibration")
        start_btn.setObjectName("CalStartCalibration")
        start_btn.setMinimumHeight(45)
        start_btn.clicked.connect(self.calibration_started)
        main_layout.addWidget(start_btn)

    def refresh(self) -> None:
        last_done = self._read_last_calibrated()
        if last_done:
            full_text = (
                _DESCRIPTION
                + f"<br><br><b>Last calibrated:</b><br>{last_done}"
            )
        else:
            full_text = _DESCRIPTION + "<br><br><b>Last calibrated:</b><br>Never"
        self._description_label.setText(full_text)

    def _read_last_calibrated(self) -> str | None:
        try:
            ctx = get_app_context()
            cal_pos = ctx.motion.settings.camera_calibration_position
            if not cal_pos.has_been_calibrated:
                return None
            dt = datetime.fromisoformat(cal_pos.last_calibrated_iso)
            return dt.astimezone().strftime("%Y-%m-%d  %H:%M:%S")
        except Exception:
            return None

    def reset(self) -> None:
        self.refresh()

    @staticmethod
    def steps() -> list[tuple[str, str]]:
        return _STEPS


class CameraSpaceStepsWidget(QWidget):
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
        self._on_title_changed = on_title_changed
        self._crosshair_state_before: bool | None = None
        self._build_ui()
        self._update_step_display()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._step_title = QLabel()
        self._step_title.setObjectName("CalStepTitle")
        layout.addWidget(self._step_title)

        self._step_body = _BodyText()
        self._step_body.setObjectName("CalStepBody")
        layout.addWidget(self._step_body)

        # Step 2 — position controls (hidden on other steps)
        self._position_widget = self._build_position_widget()
        layout.addWidget(self._position_widget)

        # Step 3 — capture controls (hidden on other steps)
        self._capture_widget = self._build_capture_widget()
        layout.addWidget(self._capture_widget)

        nav_layout = QHBoxLayout()
        self._prev_btn = QPushButton("Previous")
        self._prev_btn.clicked.connect(self._previous_step)
        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(self._next_step)
        self._finish_btn = QPushButton("Finish Calibration")
        self._finish_btn.clicked.connect(self._restore_crosshair_state)
        self._finish_btn.clicked.connect(self.finished)

        nav_layout.addWidget(self._prev_btn)
        nav_layout.addWidget(self._next_btn)
        nav_layout.addWidget(self._finish_btn)
        layout.addLayout(nav_layout)

        self._status_label = QLabel("")
        self._status_label.setObjectName("CalStatusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._status_label.setWordWrap(False)
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

        self._saved_pos_label = QLabel("Not set")
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

        layout.addLayout(btn_row)
        widget.hide()
        return widget

    def _build_capture_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._start_capture_btn = QPushButton("Start Capture")
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

    def _set_crosshair(self, enabled: bool) -> None:
        ctx = get_app_context()
        preview = ctx.camera_preview
        if preview is not None:
            preview.overlays.crosshair = enabled

    def _save_crosshair_state(self) -> None:
        ctx = get_app_context()
        preview = ctx.camera_preview
        self._crosshair_state_before: bool | None = (
            preview.overlays.crosshair if preview is not None else None
        )

    def _restore_crosshair_state(self) -> None:
        if self._crosshair_state_before is not None:
            self._set_crosshair(self._crosshair_state_before)
            self._crosshair_state_before = None

    def _update_step_display(self) -> None:
        step_title, step_body = _STEPS[self._current_step]
        self._step_title.setText(step_title)
        self._step_body.setPlainText(step_body)
        self._prev_btn.setEnabled(self._current_step > 0)

        if self._on_title_changed is not None:
            self._on_title_changed(
                f"Camera Calibration  {self._current_step + 1} / {self._total_steps}"
            )

        is_last = self._current_step == self._total_steps - 1
        self._next_btn.setVisible(not is_last)
        self._finish_btn.setVisible(is_last)

        # Step 2 (index 1) shows position controls
        self._position_widget.setVisible(self._current_step == 1)
        if self._current_step == 1:
            self._refresh_position_display()

        # Crosshair active during steps 2, 3, and 4 (indices 1–3)
        self._set_crosshair(self._current_step >= 1)

        # Step 3 (index 2) shows capture controls; Next is gated on completion
        self._capture_widget.setVisible(self._current_step == 2)
        if self._current_step == 2:
            self._next_btn.setEnabled(self._capture_complete)
        else:
            self._next_btn.setEnabled(True)

        self._set_status("")

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)
        self._status_label.setVisible(bool(text))

    def _refresh_position_display(self) -> None:
        try:
            ctx = get_app_context()
            motion_settings = ctx.motion.settings
            if motion_settings is None:
                return
            cal_pos = motion_settings.camera_calibration_position
            if cal_pos.is_set:
                x_mm = cal_pos.x_nm / _NM_PER_MM
                y_mm = cal_pos.y_nm / _NM_PER_MM
                z_mm = cal_pos.z_nm / _NM_PER_MM
                self._saved_pos_label.setText(
                    f"Saved X: {x_mm:.3f} Y: {y_mm:.3f} Z: {z_mm:.3f} mm"
                )
                self._goto_pos_btn.setEnabled(True)
                self._clear_pos_btn.setEnabled(True)
            else:
                self._saved_pos_label.setText("Saved position: Not set")
                self._goto_pos_btn.setEnabled(False)
                self._clear_pos_btn.setEnabled(False)
        except Exception:
            pass

    def _next_step(self) -> None:
        if self._current_step < self._total_steps - 1:
            self._current_step += 1
            self._update_step_display()

    def _previous_step(self) -> None:
        if self._current_step > 0:
            self._current_step -= 1
            self._update_step_display()

    def reset(self) -> None:
        self._restore_crosshair_state()
        self._save_crosshair_state()
        self._current_step = 0
        self._capture_complete = False
        self._update_step_display()

    # ------------------------------------------------------------------
    # Step 2 — position slots
    # ------------------------------------------------------------------

    def _on_set_position_clicked(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.is_ready():
            self._set_status("Motion controller not ready.")
            return
        motion_settings = motion.settings
        if motion_settings is None:
            return
        try:
            pos = motion.get_position()
        except Exception as exc:
            error(f"CameraSpaceCalibration: get_position failed — {exc}")
            self._set_status("Could not read stage position.")
            return
        cal_pos = motion_settings.camera_calibration_position
        cal_pos.x_nm = pos.x
        cal_pos.y_nm = pos.y
        cal_pos.z_nm = pos.z
        cal_pos.is_set = True
        motion._controller._config_manager.save(motion_settings)
        info(
            f"[CameraSpaceCalibration] Saved position: "
            f"X={pos.x / _NM_PER_MM:.3f} mm  Y={pos.y / _NM_PER_MM:.3f} mm  Z={pos.z / _NM_PER_MM:.3f} mm"
        )
        self._refresh_position_display()
        self._set_status(
            f"Position saved: ({pos.x / _NM_PER_MM:.3f}, {pos.y / _NM_PER_MM:.3f}, {pos.z / _NM_PER_MM:.3f}) mm"
        )

    def _on_goto_position_clicked(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.is_ready():
            self._set_status("Motion controller not ready.")
            return
        motion_settings = motion.settings
        if motion_settings is None:
            return
        cal_pos = motion_settings.camera_calibration_position
        if not cal_pos.is_set:
            self._set_status("No calibration position saved.")
            return
        from motion.models import Position
        try:
            motion.move_to_position(
                Position(x=cal_pos.x_nm, y=cal_pos.y_nm, z=cal_pos.z_nm),
                wait=False,
            )
        except Exception as exc:
            error(f"CameraSpaceCalibration: move_to_position failed — {exc}")
            self._set_status("Move failed — see log.")
            return
        self._set_status(
            f"Moving to ({cal_pos.x_nm / _NM_PER_MM:.3f}, "
            f"{cal_pos.y_nm / _NM_PER_MM:.3f}, "
            f"{cal_pos.z_nm / _NM_PER_MM:.3f}) mm…"
        )

    def _on_clear_position_clicked(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None:
            return
        motion_settings = motion.settings
        if motion_settings is None:
            return
        cal_pos = motion_settings.camera_calibration_position
        cal_pos.x_nm = 0
        cal_pos.y_nm = 0
        cal_pos.z_nm = 0
        cal_pos.is_set = False
        motion._controller._config_manager.save(motion_settings)
        info("[CameraSpaceCalibration] Calibration position cleared")
        self._refresh_position_display()
        self._set_status("Calibration position cleared.")

    # ------------------------------------------------------------------
    # Step 3 — capture slots
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

        try:
            from motion.routines.camera_calibration_routine import CameraCalibrationRoutine
            self._routine = CameraCalibrationRoutine(motion=motion)
            self._routine.on_state_changed = self._on_routine_state_changed
            motion.start_routine(self._routine)
        except Exception as exc:
            error(f"CameraSpaceCalibration: failed to start routine — {exc}")
            self._set_status(f"Failed to start: {exc}")
            return

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

            try:
                succeeded = get_app_context().machine_vision.is_calibrated
            except Exception:
                succeeded = False

            if succeeded:
                self._capture_complete = True
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


class CameraSpaceCalibrationPage(CameraWithSidebarPage):
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

        calibration = CollapsibleSection("Camera Calibration Step 1 / 4")
        self._steps_widget = CameraSpaceStepsWidget(
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