from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
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

from motion.routines.slot_calibration_routine import SlotCalibrationRoutine  # noqa: PLC0415

_NM_PER_MM = 1_000_000

_STEPS: list[tuple[str, str]] = [
    (
        "Place a sample in slot 1",
        "Load a sample into slot 1. Ensure it is pushed all the way to the end of the sample slot.",
    ),
    (
        "Set the number of slots",
        "Enter the number of sample slots. This determines how many usable sample slots there are.",
    ),
    (
        "Move to the slot reference mark",
        "Navigate to the beginning of the sample slots so that the first red "
        "reference mark is visible in the camera preview. Use the Go to Slot 1 "
        "button below if a calibration has been done before.",
    ),
    (
        "Bring the red mark into focus",
        "Adjust the height until the red reference mark is sharp and clearly "
        "visible in the camera preview.",
    ),
    (
        "Run automatic calibration",
        "The system will perform an automatic calibration pass across all "
        "slots. Press the button below to begin.",
    ),
    (
        "Confirm positions and offsets",
        "Review the detected slot positions. Adjust any per-slot offsets as "
        "needed, set the starting height and offset, then finish.",
    ),
]

_DESCRIPTION = (
    "<b>Purpose:</b><br>"
    "Maps the position of every sample slot so the system can navigate "
    "to each one accurately and repeatably.<br><br>"
    "<b>What it does:</b><br>"
    "• Records reference positions for the slots<br>"
    "• Enables automatic slot navigation<br>"
    "• Enables Tree Core Scanning<br><br>"
    "<b>What you need:</b><br>"
    "• A tree core sample mounted on the machine<br>"
    "• Clear visibility of the start of each slot<br>"
    "• Approximately 3 minutes<br><br>"
    "<b>Process:</b><br>"
    "The calibration will guide you through each step of the procedure."
)


def _get_tca():
    """Return the TreeCoreAutomationSettings from the motion context, or None."""
    ctx = get_app_context()
    if ctx is None or ctx.motion is None:
        return None
    return getattr(ctx.motion.settings, "tree_core_automation", None)


class SlotCalibrationWidget(QWidget):
    """Info / launch widget shown in the calibration tab list."""

    calibration_started: Signal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Sample Slot Position Calibration")
        title.setObjectName("CalPageTitle")
        layout.addWidget(title)

        self._description_label = QLabel(_DESCRIPTION)
        self._description_label.setWordWrap(True)
        self._description_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._description_label.setObjectName("CalDescriptionBox")
        layout.addWidget(self._description_label)
        layout.addStretch()

        self._warning_label = QLabel(
            "Camera Space Calibration must be completed before running Slot Calibration."
        )
        self._warning_label.setWordWrap(True)
        self._warning_label.setObjectName("CalWarningLabel")
        self._warning_label.hide()
        layout.addWidget(self._warning_label)

        self._start_btn = QPushButton("Start Calibration")
        self._start_btn.setObjectName("CalStartCalibration")
        self._start_btn.setMinimumHeight(45)
        self._start_btn.clicked.connect(self.calibration_started)
        layout.addWidget(self._start_btn)

    def _is_camera_space_calibrated(self) -> bool:
        ctx = get_app_context()
        if ctx is None or ctx.machine_vision is None:
            return False
        return ctx.machine_vision.is_calibrated

    def reset(self) -> None:
        calibrated = self._is_camera_space_calibrated()
        self._start_btn.setEnabled(calibrated)
        self._warning_label.setVisible(not calibrated)


class SlotCalibrationStepsWidget(QWidget):
    """Step-through widget embedded in the sidebar of the calibration page."""

    finished: Signal = Signal()
    cancelled: Signal = Signal()

    def __init__(
        self,
        *,
        on_title_changed=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._current_step: int = 0
        self._total_steps: int = len(_STEPS)
        self._on_title_changed = on_title_changed
        self._routine = None
        self._latest_activity: str = ""
        self._original_red_mark: bool | None = None
        # Pending in-memory edits for step 7; written to tca on finish.
        self._pending_start_height_nm: int | None = None
        self._pending_start_offset_nm: int | None = None
        self._pending_slot_positions: dict[int, int] = {}
        self._pending_slot_offsets: dict[int, int] = {}
        self._build_ui()
        self._update_step_display()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

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

        self._mark_widget = self._build_mark_widget()
        layout.addWidget(self._mark_widget)

        self._num_slots_widget = self._build_num_slots_widget()
        layout.addWidget(self._num_slots_widget)

        self._auto_cal_widget = self._build_automation_widget(
            "Begin Auto Calibration", "_auto_cal_btn"
        )
        layout.addWidget(self._auto_cal_widget)

        self._confirm_widget = self._build_confirm_widget()
        layout.addWidget(self._confirm_widget)

        nav_layout = QHBoxLayout()
        self._prev_btn = QPushButton("Previous")
        self._prev_btn.clicked.connect(self._previous_step)
        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(self._next_step)
        self._finish_btn = QPushButton("Finish Calibration")
        self._finish_btn.clicked.connect(self._on_finish)

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

    def _build_mark_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._mark_pos_label = QLabel("Mark reference: Not set")
        self._mark_pos_label.setObjectName("CalSavedPosLabel")
        layout.addWidget(self._mark_pos_label)

        self._slot1_pos_label = QLabel("Slot 1 position: Not set")
        self._slot1_pos_label.setObjectName("CalSavedPosLabel")
        layout.addWidget(self._slot1_pos_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._goto_mark_slot1_btn = QPushButton("Go to Slot 1")
        self._goto_mark_slot1_btn.setObjectName("CalSecondaryButton")
        self._goto_mark_slot1_btn.setMinimumHeight(34)
        self._goto_mark_slot1_btn.setToolTip(
            "Move to the mark reference position on the main axis / Z, "
            "using the slot 1 position for the cross-axis if available"
        )
        self._goto_mark_slot1_btn.setEnabled(False)
        self._goto_mark_slot1_btn.clicked.connect(self._on_goto_mark_slot1_clicked)
        btn_row.addWidget(self._goto_mark_slot1_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)
        widget.hide()
        return widget

    def _build_num_slots_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("Number of slots:"))

        self._num_slots_spin = QSpinBox()
        self._num_slots_spin.setRange(1, 100)
        self._num_slots_spin.setValue(20)
        self._num_slots_spin.setFixedWidth(70)
        row.addWidget(self._num_slots_spin)
        row.addStretch()
        layout.addLayout(row)

        widget.hide()
        return widget

    def _build_automation_widget(self, btn_label: str, btn_attr: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        btn = QPushButton(btn_label)
        btn.setObjectName("CalStartCapture")
        btn.setMinimumHeight(34)
        btn.clicked.connect(self._on_start_auto_cal_clicked)
        setattr(self, btn_attr, btn)
        layout.addWidget(btn)

        self._stop_auto_cal_btn = QPushButton("Stop")
        self._stop_auto_cal_btn.setObjectName("CalStopCapture")
        self._stop_auto_cal_btn.setMinimumHeight(32)
        self._stop_auto_cal_btn.clicked.connect(self._on_stop_auto_cal_clicked)
        self._stop_auto_cal_btn.setVisible(False)
        layout.addWidget(self._stop_auto_cal_btn)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_auto_cal_state)

        widget.hide()
        return widget

    def _on_start_auto_cal_clicked(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.is_ready():
            self._set_status("Motion controller not ready.")
            return
        if not ctx.has_camera:
            self._set_status("No camera available.")
            return

        self._routine = SlotCalibrationRoutine(motion=motion)
        self._routine.on_state_changed = self._on_auto_cal_state_changed
        motion.start_routine(self._routine)

        self._latest_activity = ""
        self._auto_cal_btn.setEnabled(False)
        self._stop_auto_cal_btn.setVisible(True)
        self._prev_btn.setEnabled(False)
        self._next_btn.setEnabled(False)
        self._set_status("Running…")
        self._poll_timer.start()

    def _on_stop_auto_cal_clicked(self) -> None:
        if self._routine is not None:
            self._routine.stop()
        self._set_status("Stopping…")

    def _on_auto_cal_state_changed(
        self,
        job_name: str,
        activity: str,
        progress_current: int,
        progress_total: int,
        eta_seconds: int,
    ) -> None:
        self._latest_activity = activity

    def _poll_auto_cal_state(self) -> None:
        if self._routine is None or not self._routine.is_running:
            self._poll_timer.stop()
            self._routine = None
            self._auto_cal_btn.setEnabled(True)
            self._stop_auto_cal_btn.setVisible(False)
            self._prev_btn.setEnabled(self._current_step > 0)
            self._next_btn.setEnabled(True)
            self._next_step()
            return

        if self._latest_activity:
            prog = self._routine.progress_current
            total = self._routine.progress_total
            if total > 0:
                self._set_status(f"[{prog}/{total}]  {self._latest_activity}")
            else:
                self._set_status(self._latest_activity)

    def _build_confirm_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        globals_group = QGroupBox("Starting Parameters")
        globals_layout = QVBoxLayout(globals_group)
        globals_layout.setSpacing(6)

        note_label = QLabel(
            "The starting position should be above and out of focus of a sample on the sample tray."
        )
        note_label.setWordWrap(True)
        note_label.setObjectName("CalStepBody")
        globals_layout.addWidget(note_label)

        height_row = QHBoxLayout()
        height_row.addWidget(QLabel("Starting Height (mm):"))
        self._start_height_spin = QDoubleSpinBox()
        self._start_height_spin.wheelEvent = lambda e: e.ignore()
        self._start_height_spin.setDecimals(3)
        self._start_height_spin.setRange(-9999.0, 9999.0)
        self._start_height_spin.setSingleStep(0.001)
        height_row.addWidget(self._start_height_spin)
        globals_layout.addLayout(height_row)

        offset_row = QHBoxLayout()
        offset_row.addWidget(QLabel("Starting Offset (mm):"))
        self._start_offset_spin = QDoubleSpinBox()
        self._start_offset_spin.wheelEvent = lambda e: e.ignore()
        self._start_offset_spin.setDecimals(3)
        self._start_offset_spin.setRange(-9999.0, 9999.0)
        self._start_offset_spin.setSingleStep(0.001)
        offset_row.addWidget(self._start_offset_spin)
        globals_layout.addLayout(offset_row)

        self._goto_start_pos_btn = QPushButton("Go to Starting Position")
        self._goto_start_pos_btn.setObjectName("CalSecondaryButton")
        self._goto_start_pos_btn.setMinimumHeight(34)
        self._goto_start_pos_btn.clicked.connect(self._on_goto_start_pos_clicked)
        globals_layout.addWidget(self._goto_start_pos_btn)

        layout.addWidget(globals_group)

        # Slot rows are built dynamically in _rebuild_slot_rows() because the
        # count comes from tca.num_slots which isn't known at widget-build time.
        self._slots_group = QGroupBox("Slot Positions")
        self._slots_layout = QVBoxLayout(self._slots_group)
        self._slots_layout.setSpacing(6)
        layout.addWidget(self._slots_group)

        self._slot_position_labels: list[QLabel] = []
        self._slot_position_spins: list[QDoubleSpinBox] = []
        self._slot_offset_spins: list[QDoubleSpinBox] = []

        widget.hide()
        return widget

    def _rebuild_slot_rows(self, num_slots: int) -> None:
        """Clear and repopulate the slot rows to match the current slot count."""
        while self._slots_layout.count():
            item = self._slots_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._slot_position_labels = []
        self._slot_position_spins: list[QDoubleSpinBox] = []
        self._slot_offset_spins = []

        # Set=40, Go=36, spacing=4 between them — offset button matches that total.
        _BTN_PAIR_WIDTH = 40 + 4 + 36

        for i in range(num_slots):
            if i > 0:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setFrameShadow(QFrame.Shadow.Sunken)
                self._slots_layout.addWidget(sep)

            slot_container = QWidget()
            slot_vbox = QVBoxLayout(slot_container)
            slot_vbox.setContentsMargins(0, 0, 0, 0)
            slot_vbox.setSpacing(2)

            # Row 1: label, position display, Set, Go
            top_row = QHBoxLayout()
            top_row.setSpacing(4)

            slot_label = QLabel(f"Slot {i + 1}:")
            slot_label.setFixedWidth(46)
            top_row.addWidget(slot_label)

            pos_label = QLabel("Not set")
            pos_label.setObjectName("CalSavedPosLabel")
            top_row.addWidget(pos_label, 1)
            self._slot_position_labels.append(pos_label)

            pos_spin = QDoubleSpinBox()
            pos_spin.setDecimals(3)
            pos_spin.setRange(-9999.0, 9999.0)
            pos_spin.setSingleStep(0.001)
            pos_spin.setReadOnly(True)
            pos_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            pos_spin.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            pos_spin.wheelEvent = lambda e: e.ignore()
            pos_spin.setToolTip(f"Current position for slot {i + 1} (mm)")
            self._slot_position_spins.append(pos_spin)
            top_row.addWidget(pos_spin, 1)

            set_btn = QPushButton("Set")
            set_btn.setObjectName("CalSecondaryButton")
            set_btn.setMinimumHeight(26)
            set_btn.setFixedWidth(40)
            set_btn.clicked.connect(self._make_set_slot_handler(i))
            top_row.addWidget(set_btn)

            goto_btn = QPushButton("Go")
            goto_btn.setObjectName("CalSecondaryButton")
            goto_btn.setMinimumHeight(26)
            goto_btn.setFixedWidth(36)
            goto_btn.clicked.connect(self._make_goto_slot_handler(i))
            top_row.addWidget(goto_btn)

            slot_vbox.addLayout(top_row)

            # Row 2: offset label, spinbox, Set Offset button aligned under Set+Go
            offset_row = QHBoxLayout()
            offset_row.setSpacing(4)
            offset_row.addSpacing(46)

            offset_label = QLabel("Offset (mm):")
            offset_row.addWidget(offset_label)

            offset_spin = QDoubleSpinBox()
            offset_spin.setDecimals(3)
            offset_spin.setRange(-9999.0, 9999.0)
            offset_spin.setSingleStep(0.001)
            offset_spin.setToolTip(f"Offset for slot {i + 1} (mm)")
            offset_spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            offset_spin.wheelEvent = lambda e: e.ignore()
            self._slot_offset_spins.append(offset_spin)
            offset_row.addWidget(offset_spin, 1)

            set_off_btn = QPushButton("Set Offset")
            set_off_btn.setObjectName("CalSecondaryButton")
            set_off_btn.setMinimumHeight(26)
            set_off_btn.setFixedWidth(_BTN_PAIR_WIDTH)
            set_off_btn.clicked.connect(self._make_set_offset_handler(i))
            offset_row.addWidget(set_off_btn)

            slot_vbox.addLayout(offset_row)
            self._slots_layout.addWidget(slot_container)

    # ------------------------------------------------------------------
    # Handler factories
    # ------------------------------------------------------------------

    def _make_set_slot_handler(self, index: int):
        def _handler() -> None:
            self._on_set_slot_clicked(index)
        return _handler

    def _make_goto_slot_handler(self, index: int):
        def _handler() -> None:
            self._goto_slot(index, f"slot {index + 1}")
        return _handler

    def _make_set_offset_handler(self, index: int):
        def _handler() -> None:
            self._on_set_slot_offset_clicked(index)
        return _handler

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _update_step_display(self) -> None:
        step_title, step_body = _STEPS[self._current_step]
        self._step_title.setText(step_title)
        self._step_body.setText(step_body)
        self._prev_btn.setEnabled(True)
        if self._current_step == 0:
            self._prev_btn.setText("Cancel Calibration")
        else:
            self._prev_btn.setText("Previous")

        if self._on_title_changed is not None:
            self._on_title_changed(
                f"Slot Calibration  {self._current_step + 1} / {self._total_steps}"
            )

        is_last = self._current_step == self._total_steps - 1
        self._next_btn.setVisible(not is_last)
        self._finish_btn.setVisible(is_last)

        self._num_slots_widget.setVisible(self._current_step == 1)
        if self._current_step == 1:
            self._refresh_num_slots_display()

        self._mark_widget.setVisible(self._current_step == 2)
        if self._current_step == 2:
            self._refresh_mark_display()

        self._auto_cal_widget.setVisible(self._current_step == 4)

        self._confirm_widget.setVisible(self._current_step == 5)
        if self._current_step == 5:
            self._refresh_confirm_display()

        self._set_red_mark_overlay(self._current_step in (4, 5))

        self._set_status("")

    def _set_red_mark_overlay(self, enabled: bool) -> None:
        ctx = get_app_context()
        if ctx is None:
            return
        preview = getattr(ctx, "camera_preview", None)
        if preview is None:
            return
        preview.overlays.red_mark = enabled

    def _next_step(self) -> None:
        if self._current_step < self._total_steps - 1:
            self._current_step += 1
            self._update_step_display()

    def _previous_step(self) -> None:
        if self._current_step == 0:
            self._on_cancel()
        elif self._current_step > 0:
            self._current_step -= 1
            self._update_step_display()

    def _on_cancel(self) -> None:
        if self._original_red_mark is not None:
            self._set_red_mark_overlay(self._original_red_mark)
            self._original_red_mark = None
        self.cancelled.emit()

    def reset(self) -> None:
        ctx = get_app_context()
        preview = getattr(ctx, "camera_preview", None) if ctx is not None else None
        self._original_red_mark = preview.overlays.red_mark if preview is not None else None
        self._current_step = 0
        self._pending_start_height_nm = None
        self._pending_start_offset_nm = None
        self._pending_slot_positions.clear()
        self._pending_slot_offsets.clear()
        self._update_step_display()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)
        self._status_label.setVisible(bool(text))

    # ------------------------------------------------------------------
    # Settings accessors
    # ------------------------------------------------------------------

    def _get_mark_position(self) -> tuple[int, int] | None:
        """Return (mark_reference_nm, mark_z_nm) or None if not saved."""
        tca = _get_tca()
        if tca is None:
            return None
        if tca.mark_reference_nm > 0 or tca.mark_z_nm > 0:
            return tca.mark_reference_nm, tca.mark_z_nm
        return None

    def _get_slot_position_nm(self, index: int) -> int | None:
        """Return the saved position_nm for slot ``index``, or None if unset."""
        if index in self._pending_slot_positions:
            return self._pending_slot_positions[index]
        tca = _get_tca()
        if tca is None or index >= len(tca.slots):
            return None
        pos = tca.slots[index].position_nm
        return pos if pos > 0 else None

    # ------------------------------------------------------------------
    # Step 2 — mark reference helpers
    # ------------------------------------------------------------------

    def _refresh_mark_display(self) -> None:
        mark = self._get_mark_position()
        if mark is not None:
            ref_mm = mark[0] / _NM_PER_MM
            z_mm = mark[1] / _NM_PER_MM
            self._mark_pos_label.setText(f"Mark ref: {ref_mm:.3f} mm  Z: {z_mm:.3f} mm")
        else:
            self._mark_pos_label.setText("Mark reference: Not set")

        slot1_nm = self._get_slot_position_nm(0)
        if slot1_nm is not None:
            self._slot1_pos_label.setText(f"Slot 1: {slot1_nm / _NM_PER_MM:.3f} mm")
        else:
            self._slot1_pos_label.setText("Slot 1 position: Not set")

        self._goto_mark_slot1_btn.setEnabled(mark is not None or slot1_nm is not None)

    def _on_goto_mark_slot1_clicked(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.is_ready():
            self._set_status("Motion controller not ready.")
            return
        mark = self._get_mark_position()
        tca = _get_tca()
        axis = tca.axis if tca is not None else "y"
        slot1_nm = self._get_slot_position_nm(0)
        try:
            current = motion.get_position()
            # Main axis and Z come from the mark if available, else stay current.
            main_nm = mark[0] if mark is not None else (current.y if axis == "y" else current.x)
            z_nm = mark[1] if mark is not None else current.z
            # Cross-axis comes from slot 1 position if available, else stay current.
            cross_nm = slot1_nm if slot1_nm is not None else (current.x if axis == "y" else current.y)
            if axis == "y":
                target = Position(x=cross_nm, y=main_nm, z=z_nm)
            else:
                target = Position(x=main_nm, y=cross_nm, z=z_nm)
            motion.move_to_position(target, wait=False)
        except Exception as exc:
            error(f"SlotCalibration: move to mark/slot 1 failed — {exc}")
            self._set_status("Move failed — see log.")
            return
        opp = "x" if axis == "y" else "y"
        self._set_status(
            f"Moving to mark ({axis.upper()}={main_nm / _NM_PER_MM:.3f} mm,"
            f" {opp.upper()}={cross_nm / _NM_PER_MM:.3f} mm,"
            f" Z={z_nm / _NM_PER_MM:.3f} mm)…"
        )

    # ------------------------------------------------------------------
    # Step 2 — number of slots helpers
    # ------------------------------------------------------------------

    def _refresh_num_slots_display(self) -> None:
        tca = _get_tca()
        if tca is not None:
            self._num_slots_spin.setValue(tca.num_slots)
        else:
            self._num_slots_spin.setValue(20)

    # ------------------------------------------------------------------
    # Step 5 — confirm helpers
    # ------------------------------------------------------------------

    def _refresh_confirm_display(self) -> None:
        tca = _get_tca()
        if tca is None:
            return

        num_slots = tca.num_slots
        self._rebuild_slot_rows(num_slots)

        height_nm = (
            self._pending_start_height_nm
            if self._pending_start_height_nm is not None
            else tca.starting_height_nm
        )
        self._start_height_spin.setValue(height_nm / _NM_PER_MM)

        offset_nm = (
            self._pending_start_offset_nm
            if self._pending_start_offset_nm is not None
            else tca.starting_offset_nm
        )
        self._start_offset_spin.setValue(offset_nm / _NM_PER_MM)

        for i in range(num_slots):
            pos_nm = self._get_slot_position_nm(i)
            if pos_nm is not None:
                self._slot_position_labels[i].setText(f"{pos_nm / _NM_PER_MM:.3f} mm")
                self._slot_position_spins[i].setValue(pos_nm / _NM_PER_MM)
            else:
                self._slot_position_labels[i].setText("Not set")
                self._slot_position_spins[i].setValue(0.0)
            off_nm = (
                self._pending_slot_offsets[i]
                if i in self._pending_slot_offsets
                else tca.slots[i].offset_nm
            )
            self._slot_offset_spins[i].setValue(off_nm / _NM_PER_MM)

    def _on_set_start_height_clicked(self) -> None:
        self._pending_start_height_nm = int(self._start_height_spin.value() * _NM_PER_MM)
        self._set_status(
            f"Starting height staged to {self._pending_start_height_nm / _NM_PER_MM:.3f} mm."
        )

    def _on_set_start_offset_clicked(self) -> None:
        self._pending_start_offset_nm = int(self._start_offset_spin.value() * _NM_PER_MM)
        self._set_status(
            f"Starting offset staged to {self._pending_start_offset_nm / _NM_PER_MM:.3f} mm."
        )

    def _on_goto_start_pos_clicked(self) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.is_ready():
            self._set_status("Motion controller not ready.")
            return
        tca = _get_tca()
        if tca is None:
            self._set_status("No calibration settings available.")
            return
        height_nm = (
            self._pending_start_height_nm
            if self._pending_start_height_nm is not None
            else tca.starting_height_nm
        )
        offset_nm = (
            self._pending_start_offset_nm
            if self._pending_start_offset_nm is not None
            else tca.starting_offset_nm
        )
        axis = tca.axis
        try:
            current = motion.get_position()
            if axis == "y":
                target = Position(x=current.x, y=offset_nm, z=height_nm)
            else:
                target = Position(x=offset_nm, y=current.y, z=height_nm)
            motion.move_to_position(target, wait=False)
        except Exception as exc:
            error(f"SlotCalibration: move to starting position failed — {exc}")
            self._set_status("Move failed — see log.")
            return
        self._set_status(
            f"Moving to starting position ({axis.upper()}={offset_nm / _NM_PER_MM:.3f} mm,"
            f" Z={height_nm / _NM_PER_MM:.3f} mm)…"
        )

    def _on_set_slot_clicked(self, index: int) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.is_ready():
            self._set_status("Motion controller not ready.")
            return
        tca = _get_tca()
        if tca is None or index >= tca.num_slots:
            return
        try:
            pos = motion.get_position()
        except Exception as exc:
            error(f"SlotCalibration: get_position failed — {exc}")
            self._set_status("Could not read stage position.")
            return
        axis_val = pos.y if (tca.axis == "y") else pos.x
        self._pending_slot_positions[index] = axis_val
        self._slot_position_labels[index].setText(f"{axis_val / _NM_PER_MM:.3f} mm")
        self._slot_position_spins[index].setValue(axis_val / _NM_PER_MM)
        self._set_status(f"Slot {index + 1} position staged.")

    def _on_set_slot_offset_clicked(self, index: int) -> None:
        val_nm = int(self._slot_offset_spins[index].value() * _NM_PER_MM)
        self._pending_slot_offsets[index] = val_nm
        self._set_status(f"Slot {index + 1} offset staged to {val_nm / _NM_PER_MM:.3f} mm.")

    def _on_finish(self) -> None:
        tca = _get_tca()
        if tca is not None:
            desired = self._num_slots_spin.value()
            from motion.motion_config import TreeCoreSlot  # noqa: PLC0415
            while len(tca.slots) < desired:
                tca.slots.append(TreeCoreSlot())
            if len(tca.slots) > desired:
                tca.slots = tca.slots[:desired]
            if self._pending_start_height_nm is not None:
                tca.starting_height_nm = self._pending_start_height_nm
            if self._pending_start_offset_nm is not None:
                tca.starting_offset_nm = self._pending_start_offset_nm
            for idx, pos_nm in self._pending_slot_positions.items():
                if idx < len(tca.slots):
                    tca.slots[idx].position_nm = pos_nm
            for idx, off_nm in self._pending_slot_offsets.items():
                if idx < len(tca.slots):
                    tca.slots[idx].offset_nm = off_nm
            get_app_context().motion.save_settings()
            info("[SlotCalibration] Settings saved on finish.")
        if self._original_red_mark is not None:
            self._set_red_mark_overlay(self._original_red_mark)
            self._original_red_mark = None
        self.finished.emit()

    # ------------------------------------------------------------------
    # Shared motion helper
    # ------------------------------------------------------------------

    def _goto_slot(self, index: int, label: str) -> None:
        ctx = get_app_context()
        motion = ctx.motion
        if motion is None or not motion.is_ready():
            self._set_status("Motion controller not ready.")
            return
        pos_nm = self._get_slot_position_nm(index)
        if pos_nm is None:
            self._set_status(f"No position saved for {label}.")
            return
        tca = _get_tca()
        axis = tca.axis if tca is not None else "y"
        opp = "x" if axis == "y" else "y"
        mark = self._get_mark_position()
        try:
            current = motion.get_position()
            main_nm = mark[0] if mark is not None else (current.y if axis == "y" else current.x)
            z_nm = mark[1] if mark is not None else current.z
            if axis == "y":
                target = Position(x=pos_nm, y=main_nm, z=z_nm)
            else:
                target = Position(x=main_nm, y=pos_nm, z=z_nm)
            motion.move_to_position(target, wait=False)
        except Exception as exc:
            error(f"SlotCalibration: move to {label} failed — {exc}")
            self._set_status("Move failed — see log.")
            return
        self._set_status(
            f"Moving to {label} ({opp.upper()}={pos_nm / _NM_PER_MM:.3f} mm,"
            f" {axis.upper()}={main_nm / _NM_PER_MM:.3f} mm,"
            f" Z={z_nm / _NM_PER_MM:.3f} mm)…"
        )


class SlotCalibrationPage(CameraWithSidebarPage):
    finished: Signal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(self._make_sidebar(), parent)
        self._steps_widget.finished.connect(self.finished)
        self._steps_widget.cancelled.connect(self.finished)

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

        calibration = CollapsibleSection("Slot Calibration Step 1 / 7")
        self._steps_widget = SlotCalibrationStepsWidget(
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