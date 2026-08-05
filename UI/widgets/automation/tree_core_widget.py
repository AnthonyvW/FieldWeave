from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from PySide6.QtCore import QEvent, QObject, Qt, QMimeData, QTimer
from PySide6.QtGui import QColor, QDragEnterEvent, QDragLeaveEvent, QDropEvent, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from common.app_context import get_app_context
from common.logger import error, warning
from motion.models import Position
from motion.routines.tree_core_imaging_routine import TreeCoreImagingRoutine
from post_processing.routines.focus_stack_routine import FocusStackRoutineConfig
from UI.widgets.automation.output_folder_widget import OutputFolderWidget

_NM_PER_MM = 1_000_000
_CSV_MAX_ROWS = 20


def _get_tca():
    """Return the TreeCoreAutomationSettings from the motion context, or None."""
    ctx = get_app_context()
    if ctx is None or ctx.motion is None:
        return None
    return getattr(ctx.motion.settings, "tree_core_automation", None)


# ---------------------------------------------------------------------------
# Application-level drag watcher
# ---------------------------------------------------------------------------

class _AppDragWatcher(QObject):
    """Event filter installed on QApplication to detect when any CSV drag
    starts or ends anywhere in the application window, regardless of which
    widget the cursor is over.

    Calls show_overlay() when a CSV drag enters any top-level window and
    hide_overlay() when the drag is released or leaves all windows.
    """

    def __init__(
        self,
        show_overlay: Callable[[], None],
        hide_overlay: Callable[[], None],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._show_overlay = show_overlay
        self._hide_overlay = hide_overlay
        self._active = False

    @staticmethod
    def _is_csv_drag(event: QEvent) -> bool:
        mime = getattr(event, "mimeData", None)
        if mime is None:
            return False
        return mime().hasUrls() and any(
            u.toLocalFile().lower().endswith(".csv") for u in mime().urls()
        )

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        t = event.type()
        if t == QEvent.Type.DragEnter and not self._active and self._is_csv_drag(event):
            self._active = True
            self._show_overlay()
        elif t in (QEvent.Type.Drop, QEvent.Type.DragLeave) and self._active:
            self._active = False
            self._hide_overlay()
        return False


# ---------------------------------------------------------------------------
# Drop overlay
# ---------------------------------------------------------------------------

class _DropOverlay(QWidget):
    """Semi-transparent overlay shown over the parent widget during a CSV drag."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.hide()

    def show_over(self) -> None:
        self.setGeometry(self.parent().rect())  # type: ignore[union-attr]
        self.raise_()
        self.show()

    def set_hovering(self, hovering: bool) -> None:
        pass

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QColor(255, 165, 0, 60))

        pen = QPen(QColor(255, 165, 0, 220), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(4, 4, -4, -4))

        font = painter.font()
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(0, 0, 0, 210))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Drop CSV here")


# ---------------------------------------------------------------------------
# Confirmation dialog
# ---------------------------------------------------------------------------

class _ConfirmDialog(QDialog):
    """Modal dialog summarising the planned tree core imaging run."""

    def __init__(
        self,
        output_path: str,
        slot_count: int,
        image_calibration_scale: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm Tree Core Imaging")
        self.setModal(True)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Ready to start tree core imaging?")
        title.setObjectName("CalScaleDialogTitle")
        layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setObjectName("SampleDivider")
        layout.addWidget(line)

        form = QFormLayout()
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        rows: list[tuple[str, str]] = [
            ("Slots to image", str(slot_count)),
            ("Image calibration scale", "Yes" if image_calibration_scale else "No"),
            ("Output path", output_path),
        ]
        for label_text, value_text in rows:
            lbl = QLabel(label_text + ":")
            lbl.setObjectName("CalScaleRowLabel")
            val = QLabel(value_text)
            val.setObjectName("CalScaleRowValue")
            val.setWordWrap(True)
            form.addRow(lbl, val)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Start")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


# ---------------------------------------------------------------------------
# QLineEdit subclass that distributes multi-line pastes across slots
# ---------------------------------------------------------------------------

class _MultilinePasteEdit(QLineEdit):
    """QLineEdit that intercepts multi-line pastes.

    When the pasted text contains newlines the first line is inserted normally
    and the remaining lines are handed to an overflow callback so the parent
    can distribute them to subsequent sample slots.
    """

    def __init__(
        self,
        overflow_callback: Callable[[list[str]], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._overflow_callback = overflow_callback

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.matches(QKeySequence.StandardKey.Paste):
            clipboard = QApplication.clipboard()
            raw = clipboard.text()

            if "\n" not in raw and "\r" not in raw:
                super().keyPressEvent(event)
                return

            lines = [ln.strip() for ln in raw.splitlines()]
            lines = [ln for ln in lines if ln]

            if not lines:
                return

            self.insert(lines[0])

            if len(lines) > 1:
                self._overflow_callback(lines[1:])
        else:
            super().keyPressEvent(event)

    def insertFromMimeData(self, source: QMimeData) -> None:
        if not source.hasText():
            super().insertFromMimeData(source)
            return

        raw = source.text()

        if "\n" not in raw and "\r" not in raw:
            super().insertFromMimeData(source)
            return

        lines = [ln.strip() for ln in raw.splitlines()]
        lines = [ln for ln in lines if ln]

        if not lines:
            return

        plain = QMimeData()
        plain.setText(lines[0])
        super().insertFromMimeData(plain)

        if len(lines) > 1:
            self._overflow_callback(lines[1:])


class _SampleRowWidget(QWidget):
    """One row in the sample list: toggle, sample ID label, and name text box.

    The row starts disabled (toggle unchecked).  The name field is always
    editable.  Typing into a blank field for the first time auto-enables the
    row.  When enabled, the entire row is highlighted orange.  When disabled
    but containing text the name field turns white so it stands out.
    """

    def __init__(
        self,
        sample_number: int,
        slot_index: int,
        overflow_callback: Callable[[int, list[str]], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._sample_number = sample_number
        self._ever_typed = False

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        self._toggle = QCheckBox()
        self._toggle.setObjectName("SampleToggleInactive")
        self._toggle.setChecked(False)
        self._toggle.setFixedWidth(20)
        layout.addWidget(self._toggle)

        self._id_label = QLabel(f"{sample_number:02d}")
        self._id_label.setObjectName("SampleIdInactive")
        self._id_label.setFixedWidth(28)
        self._id_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._id_label)

        bound_overflow: Callable[[list[str]], None] = lambda lines: overflow_callback(slot_index, lines)
        self._name_edit = _MultilinePasteEdit(bound_overflow)
        self._name_edit.setObjectName("SampleEditInactive")
        self._name_edit.setFixedHeight(26)
        self._name_edit.setPlaceholderText(f"Sample {sample_number} name...")
        layout.addWidget(self._name_edit, 1)

        self._clear_btn = QToolButton()
        self._clear_btn.setText("🗑")
        self._clear_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._clear_btn.setAutoRaise(True)
        self._clear_btn.setObjectName("SampleClearButton")
        self._clear_btn.setFixedSize(22, 22)
        self._clear_btn.setToolTip("Clear slot")
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        layout.addWidget(self._clear_btn)

        self._toggle.toggled.connect(self._on_toggle_changed)
        self._name_edit.textChanged.connect(self._on_text_changed)

        self._apply_style()

    def _apply_style(self) -> None:
        active = self._toggle.isChecked()
        self.setProperty("active", active)
        self.style().unpolish(self)
        self.style().polish(self)
        toggle_name = "SampleToggleActive" if active else "SampleToggleInactive"
        self._toggle.setObjectName(toggle_name)
        self._toggle.style().unpolish(self._toggle)
        self._toggle.style().polish(self._toggle)
        self._id_label.setObjectName("SampleIdActive" if active else "SampleIdInactive")
        self._id_label.style().unpolish(self._id_label)
        self._id_label.style().polish(self._id_label)
        if active:
            edit_name = "SampleEditActive"
        elif self._name_edit.text().strip():
            edit_name = "SampleEditInactiveFilled"
        else:
            edit_name = "SampleEditInactive"
        self._name_edit.setObjectName(edit_name)
        self._name_edit.style().unpolish(self._name_edit)
        self._name_edit.style().polish(self._name_edit)

    def _on_toggle_changed(self, _checked: bool) -> None:
        self._apply_style()

    def _on_text_changed(self, text: str) -> None:
        if not self._ever_typed and text.strip():
            self._ever_typed = True
            if not self._toggle.isChecked():
                self._toggle.setChecked(True)
                return
        self._apply_style()

    def _on_clear_clicked(self) -> None:
        self._ever_typed = False
        self._name_edit.clear()
        self._toggle.setChecked(False)

    def set_text(self, text: str) -> None:
        """Programmatically set the sample name (used for multi-line paste)."""
        self._name_edit.setText(text)

    def focus_name_edit(self) -> None:
        """Focus the name field and select its contents."""
        self._name_edit.setFocus()
        self._name_edit.selectAll()

    def connect_return_to_next(self, next_row: "_SampleRowWidget | None") -> None:
        """Wire Enter in the name field to move focus to the next row's name field."""
        if next_row is not None:
            self._name_edit.returnPressed.connect(next_row.focus_name_edit)

    def set_interactive(self, enabled: bool) -> None:
        """Enable or disable all interactive elements in this row."""
        self._toggle.setEnabled(enabled)
        self._name_edit.setEnabled(enabled)
        self._clear_btn.setEnabled(enabled)

    @property
    def sample_number(self) -> int:
        return self._sample_number

    @property
    def enabled(self) -> bool:
        return self._toggle.isChecked()

    @property
    def name(self) -> str:
        return self._name_edit.text().strip()


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class TreeCoreWidget(QWidget):
    """Widget for configuring and running the Tree Core Imaging automation."""
    mode_name: str = "Tree Core Imaging"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._routine: TreeCoreImagingRoutine | None = None
        self._sample_rows: list[_SampleRowWidget] = []
        self._z_near_mm: float | None = None
        self._z_far_mm: float | None = None

        # Widgets assigned during _setup_ui and sub-builders
        self._slot_cal_warning: QLabel
        self._output_folder: OutputFolderWidget
        self._controls_widget: QWidget
        self._pause_resume_btn: QPushButton
        self._stop_btn: QPushButton
        self._poll_timer: QTimer
        self._slot_spin: QSpinBox
        self._start_btn: QPushButton

        # Focus mode group
        self._optimal_focus_radio: QRadioButton
        self._focus_stack_radio: QRadioButton
        self._focus_stack_settings: QWidget
        self._z_near_label: QLabel
        self._z_far_label: QLabel
        self._z_near_mm: float | None
        self._z_far_mm: float | None
        self._z_step_spin: QDoubleSpinBox

        # Focus stack algorithm settings
        self._fs_keep_size_check: QCheckBox
        self._fs_advanced_toggle: QToolButton
        self._fs_advanced_widget: QWidget
        self._fs_no_align_check: QCheckBox
        self._fs_crop_check: QCheckBox
        self._fs_sharpness_spin: QDoubleSpinBox
        self._fs_cull_check: QCheckBox
        self._fs_cull_threshold_spin: QDoubleSpinBox
        self._fs_slab_check: QCheckBox
        self._fs_slab_params_widget: QWidget
        self._fs_slab_size_spin: QSpinBox
        self._fs_slab_overlap_spin: QSpinBox
        self._fs_workers_spin: QSpinBox

        # Calibration scale group
        self._inspect_cal_warning: QLabel
        self._cal_scale_toggle: QCheckBox
        self._cal_scale_details: QWidget
        self._cal_dpi_label: QLabel
        self._cal_last_label: QLabel
        self._cal_goto_btn: QPushButton

        # Sample list group
        self._sample_list_layout: QVBoxLayout

        self.setAcceptDrops(True)
        self._setup_ui()
        self._drop_overlay = _DropOverlay(self)
        self._drag_watcher = _AppDragWatcher(
            show_overlay=self._on_global_drag_enter,
            hide_overlay=self._on_global_drag_end,
        )
        QApplication.instance().installEventFilter(self._drag_watcher)  # type: ignore[union-attr]
        self._idle_poll_timer = QTimer(self)
        self._idle_poll_timer.setInterval(1000)
        self._idle_poll_timer.timeout.connect(self._poll_idle_state)
        self._idle_poll_timer.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self._slot_cal_warning = QLabel(
            "Slot calibration has not been completed. "
            "Please run Slot Calibration before using Tree Core Imaging."
        )
        self._slot_cal_warning.setObjectName("CalErrorLabel")
        self._slot_cal_warning.setWordWrap(True)
        self._slot_cal_warning.setVisible(False)
        main_layout.addWidget(self._slot_cal_warning)

        main_layout.addWidget(self._build_controls_group())

        self._output_folder = OutputFolderWidget()
        main_layout.addWidget(self._output_folder)

        main_layout.addWidget(self._build_focus_mode_group())
        main_layout.addWidget(self._build_calibration_scale_group())

        main_layout.addWidget(self._build_sample_list_group())

        self._controls_widget = QWidget()
        controls_layout = QHBoxLayout(self._controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self._pause_resume_btn = QPushButton("Pause")
        self._pause_resume_btn.setFixedHeight(32)
        self._pause_resume_btn.setObjectName("CalSecondaryButton")
        self._pause_resume_btn.clicked.connect(self._on_pause_resume_clicked)
        controls_layout.addWidget(self._pause_resume_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedHeight(32)
        self._stop_btn.setObjectName("CalScaleStop")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        controls_layout.addWidget(self._stop_btn)

        self._controls_widget.setVisible(False)
        main_layout.addWidget(self._controls_widget)

        main_layout.addStretch(1)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_routine_state)

    def _build_controls_group(self) -> QGroupBox:
        group = QGroupBox("Controls")

        layout = QHBoxLayout(group)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        slot_label = QLabel("Slot:")
        layout.addWidget(slot_label)

        self._slot_spin = QSpinBox()
        self._slot_spin.setFixedHeight(30)
        self._slot_spin.setMinimum(1)
        self._slot_spin.setMaximum(1)
        self._slot_spin.setValue(1)
        self._slot_spin.setFixedWidth(60)
        layout.addWidget(self._slot_spin)

        go_btn = QPushButton("Go to Slot")
        go_btn.setObjectName("GoToSlot")
        go_btn.setFixedHeight(30)
        go_btn.clicked.connect(self._on_go_to_slot_clicked)
        layout.addWidget(go_btn)

        layout.addStretch(1)

        self._start_btn = QPushButton("Start Automation")
        self._start_btn.setObjectName("AreaScanStart")
        self._start_btn.setFixedHeight(30)
        self._start_btn.clicked.connect(self._on_start_clicked)
        layout.addWidget(self._start_btn)

        return group

    def _build_focus_mode_group(self) -> QGroupBox:
        group = QGroupBox("Imaging Mode")

        outer_layout = QVBoxLayout(group)
        outer_layout.setContentsMargins(10, 8, 10, 8)
        outer_layout.setSpacing(6)

        self._optimal_focus_radio = QRadioButton("Optimal Focus")
        self._optimal_focus_radio.setChecked(True)
        self._optimal_focus_radio.setToolTip(
            "Run autofocus at each position during the sweep. "
            "Captures a single image per position at the best focus found."
        )
        outer_layout.addWidget(self._optimal_focus_radio)

        self._focus_stack_radio = QRadioButton("Focus Stacking")
        self._focus_stack_radio.setToolTip(
            "Capture a Z-stack at each position and combine them into a "
            "fully-focused composite image."
        )
        self._focus_stack_radio.toggled.connect(self._on_focus_mode_changed)
        outer_layout.addWidget(self._focus_stack_radio)

        self._focus_stack_settings = QWidget()
        fs_layout = QVBoxLayout(self._focus_stack_settings)
        fs_layout.setContentsMargins(16, 4, 0, 0)
        fs_layout.setSpacing(6)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("SampleDivider")
        fs_layout.addWidget(divider)

        # Z planes and step
        near_row = QWidget()
        near_layout = QHBoxLayout(near_row)
        near_layout.setContentsMargins(0, 0, 0, 0)
        near_layout.setSpacing(8)
        set_near_btn = QPushButton("Set Near Plane")
        set_near_btn.setFixedHeight(28)
        set_near_btn.clicked.connect(self._on_set_z_near_clicked)
        near_layout.addWidget(set_near_btn)
        self._z_near_label = QLabel("Not set")
        self._z_near_label.setObjectName("AreaScanAxisReadout")
        self._z_near_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        near_layout.addWidget(self._z_near_label, 1)
        fs_layout.addWidget(near_row)

        far_row = QWidget()
        far_layout = QHBoxLayout(far_row)
        far_layout.setContentsMargins(0, 0, 0, 0)
        far_layout.setSpacing(8)
        set_far_btn = QPushButton("Set Far Plane")
        set_far_btn.setFixedHeight(28)
        set_far_btn.clicked.connect(self._on_set_z_far_clicked)
        far_layout.addWidget(set_far_btn)
        self._z_far_label = QLabel("Not set")
        self._z_far_label.setObjectName("AreaScanAxisReadout")
        self._z_far_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        far_layout.addWidget(self._z_far_label, 1)
        fs_layout.addWidget(far_row)

        step_row = QWidget()
        step_layout = QHBoxLayout(step_row)
        step_layout.setContentsMargins(0, 0, 0, 0)
        step_layout.setSpacing(8)
        step_layout.addWidget(QLabel("Z Step (mm):"))
        self._z_step_spin = QDoubleSpinBox()
        self._z_step_spin.setFixedHeight(28)
        self._z_step_spin.setDecimals(4)
        self._z_step_spin.setSuffix(" mm")
        printer_step_mm = self._get_printer_step_mm()
        self._z_step_spin.setMinimum(printer_step_mm)
        self._z_step_spin.setMaximum(10.0)
        self._z_step_spin.setSingleStep(printer_step_mm)
        self._z_step_spin.setValue(0.2)
        self._z_step_spin.valueChanged.connect(self._on_z_step_changed)
        step_layout.addWidget(self._z_step_spin)
        step_layout.addStretch(1)
        fs_layout.addWidget(step_row)

        # Stacking settings divider
        stack_divider = QFrame()
        stack_divider.setFrameShape(QFrame.Shape.HLine)
        stack_divider.setObjectName("SampleDivider")
        fs_layout.addWidget(stack_divider)

        # Advanced toggle
        self._fs_advanced_toggle = QToolButton()
        self._fs_advanced_toggle.setText("Advanced settings")
        self._fs_advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._fs_advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._fs_advanced_toggle.setCheckable(True)
        self._fs_advanced_toggle.setChecked(False)
        self._fs_advanced_toggle.setObjectName("AdvancedSettingsToggle")
        self._fs_advanced_toggle.toggled.connect(self._on_fs_advanced_toggled)
        fs_layout.addWidget(self._fs_advanced_toggle)

        self._fs_advanced_widget = QWidget()
        self._fs_advanced_widget.setVisible(False)
        advanced_layout = QVBoxLayout(self._fs_advanced_widget)
        advanced_layout.setContentsMargins(12, 0, 0, 0)
        advanced_layout.setSpacing(6)

        # Keep original size (first item in advanced)
        self._fs_keep_size_check = QCheckBox("Keep original size")
        self._fs_keep_size_check.setChecked(True)
        self._fs_keep_size_check.setToolTip(
            "Keep the output image the same size as the input images. "
            "Warps are applied in-place rather than expanding the canvas."
        )
        self._fs_keep_size_check.stateChanged.connect(
            lambda v: self._write_tca_check("keep_size", v)
        )
        advanced_layout.addWidget(self._fs_keep_size_check)

        self._fs_no_align_check = QCheckBox("Skip alignment")
        self._fs_no_align_check.setChecked(False)
        self._fs_no_align_check.setToolTip("Skip ECC alignment. Use when images are already registered.")
        self._fs_no_align_check.stateChanged.connect(
            lambda v: self._write_tca_check("no_align", v)
        )
        advanced_layout.addWidget(self._fs_no_align_check)

        self._fs_crop_check = QCheckBox("Crop to intersection")
        self._fs_crop_check.setChecked(False)
        self._fs_crop_check.setToolTip(
            "Crop the output to the largest rectangle covered by every frame after "
            "alignment. Removes border regions but shrinks the output image."
        )
        self._fs_crop_check.stateChanged.connect(
            lambda v: self._write_tca_check("crop", v)
        )
        advanced_layout.addWidget(self._fs_crop_check)

        sharpness_row = QWidget()
        sharpness_layout = QHBoxLayout(sharpness_row)
        sharpness_layout.setContentsMargins(0, 0, 0, 0)
        sharpness_layout.setSpacing(8)
        sharpness_layout.addWidget(QLabel("Sharpness:"))
        self._fs_sharpness_spin = QDoubleSpinBox()
        self._fs_sharpness_spin.setFixedHeight(28)
        self._fs_sharpness_spin.setDecimals(1)
        self._fs_sharpness_spin.setMinimum(1.0)
        self._fs_sharpness_spin.setMaximum(8.0)
        self._fs_sharpness_spin.setSingleStep(0.5)
        self._fs_sharpness_spin.setValue(4.0)
        self._fs_sharpness_spin.setToolTip(
            "Weight sharpness exponent. Higher values favour the sharpest pixel "
            "more aggressively (approaching hard selection). Lower values blend "
            "more smoothly. Useful range: 1.0 (soft) to 8.0 (near-hard)."
        )
        self._fs_sharpness_spin.valueChanged.connect(
            lambda v: self._write_tca_float("sharpness", v)
        )
        sharpness_layout.addWidget(self._fs_sharpness_spin)
        sharpness_layout.addStretch(1)
        advanced_layout.addWidget(sharpness_row)

        cull_row = QWidget()
        cull_layout = QHBoxLayout(cull_row)
        cull_layout.setContentsMargins(0, 0, 0, 0)
        cull_layout.setSpacing(8)
        self._fs_cull_check = QCheckBox("Cull out-of-focus frames")
        self._fs_cull_check.setChecked(False)
        self._fs_cull_check.setToolTip(
            "Discard frames whose focus score falls below the threshold fraction "
            "of the sharpest frame. At least the two sharpest frames are always kept."
        )
        self._fs_cull_check.stateChanged.connect(
            lambda v: self._write_tca_check("cull_enabled", v)
        )
        cull_layout.addWidget(self._fs_cull_check)
        self._fs_cull_threshold_spin = QDoubleSpinBox()
        self._fs_cull_threshold_spin.setFixedHeight(28)
        self._fs_cull_threshold_spin.setDecimals(2)
        self._fs_cull_threshold_spin.setMinimum(0.0)
        self._fs_cull_threshold_spin.setMaximum(1.0)
        self._fs_cull_threshold_spin.setSingleStep(0.05)
        self._fs_cull_threshold_spin.setValue(0.6)
        self._fs_cull_threshold_spin.setToolTip(
            "Frames scoring below this fraction of the peak score are culled. "
            "Raise toward 1.0 to cull more aggressively."
        )
        self._fs_cull_threshold_spin.valueChanged.connect(
            lambda v: self._write_tca_float("cull_threshold", v)
        )
        cull_layout.addWidget(self._fs_cull_threshold_spin)
        cull_layout.addStretch(1)
        advanced_layout.addWidget(cull_row)

        self._fs_slab_check = QCheckBox("Enable slabbing")
        self._fs_slab_check.setChecked(False)
        self._fs_slab_check.setToolTip(
            "Split the image set into overlapping sub-stacks, stack each "
            "independently, then fuse the results. Reduces peak RAM for large stacks."
        )
        self._fs_slab_check.stateChanged.connect(self._on_fs_slab_enabled_changed)
        self._fs_slab_check.stateChanged.connect(
            lambda v: self._write_tca_check("slab_enabled", v)
        )
        advanced_layout.addWidget(self._fs_slab_check)

        self._fs_slab_params_widget = QWidget()
        self._fs_slab_params_widget.setVisible(False)
        slab_params_layout = QHBoxLayout(self._fs_slab_params_widget)
        slab_params_layout.setContentsMargins(20, 0, 0, 0)
        slab_params_layout.setSpacing(8)
        slab_params_layout.addWidget(QLabel("Size:"))
        self._fs_slab_size_spin = QSpinBox()
        self._fs_slab_size_spin.setFixedHeight(28)
        self._fs_slab_size_spin.setMinimum(2)
        self._fs_slab_size_spin.setMaximum(500)
        self._fs_slab_size_spin.setValue(20)
        self._fs_slab_size_spin.setToolTip("Number of images per sub-stack.")
        self._fs_slab_size_spin.valueChanged.connect(
            lambda v: self._write_tca_int("slab_size", v)
        )
        slab_params_layout.addWidget(self._fs_slab_size_spin)
        slab_params_layout.addWidget(QLabel("Overlap:"))
        self._fs_slab_overlap_spin = QSpinBox()
        self._fs_slab_overlap_spin.setFixedHeight(28)
        self._fs_slab_overlap_spin.setMinimum(0)
        self._fs_slab_overlap_spin.setMaximum(499)
        self._fs_slab_overlap_spin.setValue(5)
        self._fs_slab_overlap_spin.setToolTip(
            "Number of images shared between adjacent slabs. Must be less than size."
        )
        self._fs_slab_overlap_spin.valueChanged.connect(
            lambda v: self._write_tca_int("slab_overlap", v)
        )
        slab_params_layout.addWidget(self._fs_slab_overlap_spin)
        slab_params_layout.addStretch(1)
        advanced_layout.addWidget(self._fs_slab_params_widget)

        workers_row = QWidget()
        workers_layout = QHBoxLayout(workers_row)
        workers_layout.setContentsMargins(0, 0, 0, 0)
        workers_layout.setSpacing(8)
        workers_layout.addWidget(QLabel("Workers:"))
        self._fs_workers_spin = QSpinBox()
        self._fs_workers_spin.setFixedHeight(28)
        self._fs_workers_spin.setMinimum(0)
        self._fs_workers_spin.setMaximum(16)
        self._fs_workers_spin.setValue(3)
        self._fs_workers_spin.setToolTip(
            "Number of parallel workers for stacking. 0 = no limit (use all available). "
            "Higher values are faster but increase peak RAM by ~100 MiB per additional worker."
        )
        self._fs_workers_spin.valueChanged.connect(
            lambda v: self._write_tca_int("workers", v)
        )
        workers_layout.addWidget(self._fs_workers_spin)
        workers_layout.addStretch(1)
        advanced_layout.addWidget(workers_row)

        fs_layout.addWidget(self._fs_advanced_widget)

        self._focus_stack_settings.setVisible(False)
        outer_layout.addWidget(self._focus_stack_settings)

        self._populate_focus_mode_from_settings()

        return group

    def _get_printer_step_mm(self) -> float:
        ctx = get_app_context()
        motion = ctx.motion if ctx is not None else None
        if motion is not None and motion.settings is not None:
            return motion.settings.step_size / _NM_PER_MM
        return 0.04

    def _populate_focus_mode_from_settings(self) -> None:
        tca = _get_tca()
        if tca is None:
            return
        if tca.focus_mode == "focus_stack":
            self._focus_stack_radio.setChecked(True)
        else:
            self._optimal_focus_radio.setChecked(True)
        if tca.z_step_nm > 0:
            self._z_step_spin.blockSignals(True)
            self._z_step_spin.setValue(tca.z_step_nm / _NM_PER_MM)
            self._z_step_spin.blockSignals(False)
        if tca.z_near_plane_nm != 0:
            self._z_near_mm = tca.z_near_plane_nm / _NM_PER_MM
            self._z_near_label.setText(f"Z = {self._z_near_mm:.4f} mm")
        if tca.z_far_plane_nm != 0:
            self._z_far_mm = tca.z_far_plane_nm / _NM_PER_MM
            self._z_far_label.setText(f"Z = {self._z_far_mm:.4f} mm")

        for widget, value in (
            (self._fs_keep_size_check,  tca.keep_size),
            (self._fs_no_align_check,   tca.no_align),
            (self._fs_crop_check,       tca.crop),
            (self._fs_cull_check,       tca.cull_enabled),
            (self._fs_slab_check,       tca.slab_enabled),
        ):
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)

        for spin, value in (
            (self._fs_sharpness_spin,      tca.sharpness),
            (self._fs_cull_threshold_spin, tca.cull_threshold),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

        for spin, value in (
            (self._fs_slab_size_spin,    tca.slab_size),
            (self._fs_slab_overlap_spin, tca.slab_overlap),
            (self._fs_workers_spin,      tca.workers),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

        self._on_fs_slab_enabled_changed()

    def _build_calibration_scale_group(self) -> QGroupBox:
        group = QGroupBox("Calibration Scale")

        outer_layout = QVBoxLayout(group)
        outer_layout.setContentsMargins(10, 8, 10, 8)
        outer_layout.setSpacing(6)

        self._inspect_cal_warning = QLabel(
            "Inspection calibration has not been completed. "
            "Please run Inspection Calibration before using the Calibration Scale routine."
        )
        self._inspect_cal_warning.setObjectName("CalWarningLabel")
        self._inspect_cal_warning.setWordWrap(True)
        self._inspect_cal_warning.setVisible(False)
        outer_layout.addWidget(self._inspect_cal_warning)

        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(8)

        self._cal_scale_toggle = QCheckBox("Image calibration scale during run")
        self._cal_scale_toggle.setChecked(False)
        self._cal_scale_toggle.toggled.connect(self._on_cal_scale_toggled)
        toggle_row.addWidget(self._cal_scale_toggle)
        toggle_row.addStretch(1)

        outer_layout.addLayout(toggle_row)

        self._cal_scale_details = QWidget()
        details_layout = QVBoxLayout(self._cal_scale_details)
        details_layout.setContentsMargins(0, 4, 0, 0)
        details_layout.setSpacing(4)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("SampleDivider")
        details_layout.addWidget(divider)

        self._cal_dpi_label = QLabel("DPI: —")
        self._cal_dpi_label.setObjectName("CalScalePosLabel")
        details_layout.addWidget(self._cal_dpi_label)

        self._cal_last_label = QLabel("Last calibrated: —")
        self._cal_last_label.setObjectName("CalScalePosLabel")
        details_layout.addWidget(self._cal_last_label)

        self._cal_goto_btn = QPushButton("Go to Scale Bar Position")
        self._cal_goto_btn.setFixedHeight(28)
        self._cal_goto_btn.setObjectName("CalSecondaryButton")
        self._cal_goto_btn.clicked.connect(self._on_goto_scale_position_clicked)
        details_layout.addWidget(self._cal_goto_btn, 0, Qt.AlignmentFlag.AlignLeft)

        self._cal_scale_details.setVisible(False)
        outer_layout.addWidget(self._cal_scale_details)

        return group

    def _build_sample_list_group(self) -> QGroupBox:
        group = QGroupBox("Samples")

        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(6, 6, 6, 6)
        group_layout.setSpacing(0)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(6, 2, 6, 4)
        header_layout.setSpacing(8)

        enabled_hdr = QLabel("On")
        enabled_hdr.setObjectName("SampleListHeader")
        enabled_hdr.setFixedWidth(20)
        enabled_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(enabled_hdr)

        id_hdr = QLabel("ID")
        id_hdr.setObjectName("SampleListHeader")
        id_hdr.setFixedWidth(28)
        id_hdr.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(id_hdr)

        name_hdr = QLabel("Sample Name")
        name_hdr.setObjectName("SampleListHeader")
        header_layout.addWidget(name_hdr, 1)

        group_layout.addWidget(header)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("SampleDivider")
        group_layout.addWidget(divider)

        self._sample_list_layout = group_layout

        return group

    def _rebuild_sample_rows(self, num_slots: int) -> None:
        """Rebuild the sample row widgets to match the current slot count."""
        for row in self._sample_rows:
            row.deleteLater()
        self._sample_rows.clear()

        # Header is index 0, divider is index 1; clear everything after.
        while self._sample_list_layout.count() > 2:
            item = self._sample_list_layout.takeAt(2)
            if item.widget():
                item.widget().deleteLater()

        for i in range(1, num_slots + 1):
            row = _SampleRowWidget(i, i - 1, self._on_paste_overflow)
            self._sample_rows.append(row)
            self._sample_list_layout.addWidget(row)

            if i < num_slots:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setObjectName("SampleSeparator")
                self._sample_list_layout.addWidget(sep)

        for i, row in enumerate(self._sample_rows):
            next_row = self._sample_rows[i + 1] if i + 1 < len(self._sample_rows) else None
            row.connect_return_to_next(next_row)

        self._slot_spin.setMaximum(num_slots)

    def hideEvent(self, event) -> None:  # type: ignore[override]
        super().hideEvent(event)
        self._drop_overlay.hide()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        QApplication.instance().removeEventFilter(self._drag_watcher)  # type: ignore[union-attr]
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._drop_overlay.isVisible():
            self._drop_overlay.setGeometry(self.rect())

    # ------------------------------------------------------------------
    # showEvent — refresh slot count from calibration
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._refresh_slot_calibration_state()
        self._refresh_inspection_calibration_state()
        tca = _get_tca()
        num_slots = tca.num_slots if tca is not None else 20
        if len(self._sample_rows) != num_slots:
            self._rebuild_sample_rows(num_slots)

    def _is_slot_calibrated(self) -> bool:
        tca = _get_tca()
        return tca is not None and tca.has_been_calibrated

    def _refresh_slot_calibration_state(self) -> None:
        calibrated = self._is_slot_calibrated()
        self._slot_cal_warning.setVisible(not calibrated)
        self._start_btn.setEnabled(calibrated)

    def _is_inspection_calibrated(self) -> bool:
        ctx = get_app_context()
        if ctx is None or ctx.machine_vision is None:
            return False
        return bool(ctx.machine_vision.settings.inspect_calibration.last_calibrated)

    def _refresh_inspection_calibration_state(self) -> None:
        calibrated = self._is_inspection_calibrated()
        self._inspect_cal_warning.setVisible(not calibrated)
        self._cal_scale_toggle.setEnabled(calibrated)
        self._cal_scale_details.setEnabled(calibrated)

    def _poll_idle_state(self) -> None:
        self._refresh_inspection_calibration_state()
        if self._cal_scale_toggle.isChecked():
            self._refresh_calibration_scale_info()

    # ------------------------------------------------------------------
    # Calibration scale helpers
    # ------------------------------------------------------------------

    def _refresh_calibration_scale_info(self) -> None:
        ctx = get_app_context()
        if ctx is None or ctx.machine_vision is None:
            self._cal_dpi_label.setText("DPI: —")
            self._cal_last_label.setText("Last calibrated: —")
            self._cal_goto_btn.setEnabled(False)
            return

        s = ctx.machine_vision.settings
        if s.dpi is not None:
            self._cal_dpi_label.setText(f"DPI: {s.dpi:.1f}")
        else:
            self._cal_dpi_label.setText("DPI: —")

        last_cal = s.inspect_calibration.last_calibrated
        if last_cal:
            try:
                dt = datetime.fromisoformat(last_cal)
                self._cal_last_label.setText(
                    f"Last calibrated: {dt.strftime('%Y-%m-%d %H:%M')}"
                )
            except ValueError:
                self._cal_last_label.setText(f"Last calibrated: {last_cal}")
        else:
            self._cal_last_label.setText("Last calibrated: —")

        icp = s.inspection_calibration_position
        self._cal_goto_btn.setEnabled(getattr(icp, "is_set", False))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _write_tca_check(self, key: str, value: int) -> None:
        tca = _get_tca()
        if tca is not None:
            setattr(tca, key, value != 0)

    def _write_tca_float(self, key: str, value: float) -> None:
        tca = _get_tca()
        if tca is not None:
            setattr(tca, key, value)

    def _write_tca_int(self, key: str, value: int) -> None:
        tca = _get_tca()
        if tca is not None:
            setattr(tca, key, value)

    def _on_fs_advanced_toggled(self, checked: bool) -> None:
        self._fs_advanced_widget.setVisible(checked)
        self._fs_advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def _on_fs_slab_enabled_changed(self) -> None:
        self._fs_slab_params_widget.setVisible(self._fs_slab_check.isChecked())

    def _build_focus_stack_config(self) -> FocusStackRoutineConfig:
        slab: tuple[int, int] | None = None
        if self._fs_slab_check.isChecked():
            slab = (self._fs_slab_size_spin.value(), self._fs_slab_overlap_spin.value())
        return FocusStackRoutineConfig(
            no_align=self._fs_no_align_check.isChecked(),
            keep_size=self._fs_keep_size_check.isChecked(),
            crop=self._fs_crop_check.isChecked(),
            sharpness=self._fs_sharpness_spin.value(),
            cull=self._fs_cull_threshold_spin.value() if self._fs_cull_check.isChecked() else None,
            workers=self._fs_workers_spin.value(),
            slab=slab,
        )

    def _on_focus_mode_changed(self, stack_checked: bool) -> None:
        self._focus_stack_settings.setVisible(stack_checked)
        tca = _get_tca()
        if tca is not None:
            tca.focus_mode = "focus_stack" if stack_checked else "optimal_focus"

    def _on_set_z_near_clicked(self) -> None:
        ctx = get_app_context()
        if ctx is None or ctx.motion is None or not ctx.motion.is_ready():
            warning("TreeCoreWidget: motion controller not ready for Z near plane")
            return
        z_mm = ctx.motion.get_position().z / _NM_PER_MM
        self._z_near_mm = z_mm
        self._z_near_label.setText(f"Z = {z_mm:.4f} mm")
        tca = _get_tca()
        if tca is not None:
            tca.z_near_plane_nm = round(z_mm * _NM_PER_MM)

    def _on_set_z_far_clicked(self) -> None:
        ctx = get_app_context()
        if ctx is None or ctx.motion is None or not ctx.motion.is_ready():
            warning("TreeCoreWidget: motion controller not ready for Z far plane")
            return
        z_mm = ctx.motion.get_position().z / _NM_PER_MM
        self._z_far_mm = z_mm
        self._z_far_label.setText(f"Z = {z_mm:.4f} mm")
        tca = _get_tca()
        if tca is not None:
            tca.z_far_plane_nm = round(z_mm * _NM_PER_MM)

    def _on_z_step_changed(self, value_mm: float) -> None:
        tca = _get_tca()
        if tca is not None:
            tca.z_step_nm = round(value_mm * _NM_PER_MM)

    def _on_cal_scale_toggled(self, checked: bool) -> None:
        self._cal_scale_details.setVisible(checked)
        if checked:
            self._refresh_calibration_scale_info()

    def _on_goto_scale_position_clicked(self) -> None:
        ctx = get_app_context()
        if ctx is None or ctx.motion is None or not ctx.motion.is_ready():
            warning("TreeCoreWidget: motion controller not ready for scale bar move")
            return
        if ctx.machine_vision is None:
            return
        icp = ctx.machine_vision.settings.inspection_calibration_position
        if not getattr(icp, "is_set", False):
            warning("TreeCoreWidget: no scale bar position saved")
            return

        ctx.motion.move_to_position(
            Position(x=icp.x_nm, y=icp.y_nm, z=icp.z_nm),
            wait=False,
        )

    def _on_go_to_slot_clicked(self) -> None:
        slot_number = self._slot_spin.value()
        slot_index = slot_number - 1

        ctx = get_app_context()
        if ctx.motion is None or not ctx.motion.is_ready():
            warning("TreeCoreWidget: motion controller not ready")
            return

        tca = _get_tca()
        if tca is None:
            warning("TreeCoreWidget: no calibration settings available")
            return

        if slot_index >= tca.num_slots:
            warning(f"TreeCoreWidget: slot {slot_number} out of range ({tca.num_slots} slots)")
            return

        slot = tca.slots[slot_index]
        pos_nm = slot.position_nm if slot.position_nm > 0 else None
        if pos_nm is None:
            warning(f"TreeCoreWidget: no position saved for slot {slot_number}")
            return

        axis = tca.axis
        mark_set = tca.mark_reference_nm > 0 or tca.mark_z_nm > 0

        current = ctx.motion.get_position()
        main_nm = tca.mark_reference_nm if mark_set else (current.y if axis == "y" else current.x)
        z_nm = tca.mark_z_nm if mark_set else current.z
        if axis == "y":
            target = Position(x=pos_nm, y=main_nm, z=z_nm)
        else:
            target = Position(x=main_nm, y=pos_nm, z=z_nm)
        ctx.motion.move_to_position(target, wait=False)


    def _on_start_clicked(self) -> None:
        ctx = get_app_context()
        if ctx.motion is None or not ctx.motion.is_ready():
            error("TreeCoreWidget: motion controller not ready")
            ctx.toast.error("Motion controller not ready.")
            return

        active_samples = [r for r in self._sample_rows if r.enabled]
        if not active_samples:
            warning("TreeCoreWidget: no samples enabled")
            ctx.toast.warning("No samples enabled.")
            return

        tca = _get_tca()
        if tca is None or not tca.has_been_calibrated:
            error("TreeCoreWidget: slot calibration has not been completed")
            ctx.toast.error("Slot calibration has not been completed.")
            return

        focus_stack_config: FocusStackRoutineConfig | None = None
        if self._focus_stack_radio.isChecked():
            if self._z_near_mm is None or self._z_far_mm is None:
                warning("TreeCoreWidget: near and far Z planes must be set for focus stacking")
                ctx.toast.warning("Set the near and far Z planes before starting.")
                return
            if self._z_near_mm == self._z_far_mm:
                warning("TreeCoreWidget: near and far Z planes must be different")
                ctx.toast.warning("Near and far Z planes must be different.")
                return
            focus_stack_config = self._build_focus_stack_config()

        output_path = self._output_folder.resolved_path
        if not OutputFolderWidget.confirm_if_exists(output_path, self):
            return

        slots = [(r.sample_number - 1, r.name or f"slot_{r.sample_number:02d}") for r in active_samples]
        image_calibration_scale = self._cal_scale_toggle.isChecked()

        dlg = _ConfirmDialog(
            output_path=output_path,
            slot_count=len(slots),
            image_calibration_scale=image_calibration_scale,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        if ctx.motion.routine_running:
            error("TreeCoreWidget: a routine is already running")
            ctx.toast.error("A routine is already running.")
            return

        self._routine = TreeCoreImagingRoutine(
            motion=ctx.motion,
            output_folder=output_path,
            slots=slots,
            image_calibration_scale=image_calibration_scale,
            focus_stack_config=focus_stack_config,
        )
        ctx.motion.start_routine(self._routine)
        self._enter_running_state()

    def _on_pause_resume_clicked(self) -> None:
        if self._routine is None:
            return
        if self._routine.is_paused:
            self._routine.resume()
            self._pause_resume_btn.setText("Pause")
        else:
            self._routine.pause()
            self._pause_resume_btn.setText("Resume")

    def _on_stop_clicked(self) -> None:
        if self._routine is not None:
            self._routine.stop()

    # ------------------------------------------------------------------
    # Drag-and-drop CSV loading
    # ------------------------------------------------------------------

    def _on_global_drag_enter(self) -> None:
        self._drop_overlay.show_over()

    def _on_global_drag_end(self) -> None:
        self._drop_overlay.hide()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        mime = event.mimeData()
        if mime.hasUrls() and any(
            u.toLocalFile().lower().endswith(".csv") for u in mime.urls()
        ):
            self._drop_overlay.set_hovering(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self._drop_overlay.set_hovering(False)

    def dropEvent(self, event: QDropEvent) -> None:
        self._drop_overlay.hide()
        urls = [u for u in event.mimeData().urls() if u.toLocalFile().lower().endswith(".csv")]
        if not urls:
            event.ignore()
            return
        event.acceptProposedAction()
        self._load_csv_file(urls[0].toLocalFile())

    def _load_csv_file(self, path: str) -> None:
        import csv

        try:
            fh = open(path, newline="", encoding="utf-8-sig")
        except OSError:
            warning(f"TreeCoreWidget: could not open CSV file: {path}")
            return

        with fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                return

            col_index = 0
            for i, cell in enumerate(header):
                if cell.strip().lower() == "sampleid":
                    col_index = i
                    break

            # If no SampleID header was found, the header row is data — include it.
            values: list[str] = []
            if col_index == 0 and (not header or header[0].strip().lower() != "sampleid"):
                first_val = header[0].strip() if header else ""
                if first_val:
                    values.append(first_val)

            for row in reader:
                if len(row) <= col_index:
                    continue
                value = row[col_index].strip()
                if value:
                    values.append(value)
                if len(values) >= _CSV_MAX_ROWS:
                    break

        empty_rows = [r for r in self._sample_rows if not r.name]
        for i, value in enumerate(values):
            if i >= len(empty_rows):
                break
            empty_rows[i].set_text(value)

    def _on_paste_overflow(self, source_index: int, lines: list[str]) -> None:
        """Distribute overflow lines from a multi-line paste into subsequent slots."""
        for offset, line in enumerate(lines, start=1):
            target_index = source_index + offset
            if target_index >= len(self._sample_rows):
                break
            self._sample_rows[target_index].set_text(line)

    # ------------------------------------------------------------------
    # Routine state helpers
    # ------------------------------------------------------------------

    def _enter_running_state(self) -> None:
        self._start_btn.setEnabled(False)
        self._output_folder.setEnabled(False)
        self._cal_scale_toggle.setEnabled(False)
        self._cal_goto_btn.setEnabled(False)
        self._optimal_focus_radio.setEnabled(False)
        self._focus_stack_radio.setEnabled(False)
        self._focus_stack_settings.setEnabled(False)
        self._pause_resume_btn.setText("Pause")
        self._controls_widget.setVisible(True)
        self._poll_timer.start()
        for row in self._sample_rows:
            row.set_interactive(False)

    def _exit_running_state(self) -> None:
        self._poll_timer.stop()
        self._start_btn.setEnabled(self._is_slot_calibrated())
        self._output_folder.setEnabled(True)
        self._cal_scale_toggle.setEnabled(True)
        self._optimal_focus_radio.setEnabled(True)
        self._focus_stack_radio.setEnabled(True)
        self._focus_stack_settings.setEnabled(True)
        self._controls_widget.setVisible(False)
        self._routine = None
        for row in self._sample_rows:
            row.set_interactive(True)
        if self._cal_scale_toggle.isChecked():
            self._refresh_calibration_scale_info()

    def _poll_routine_state(self) -> None:
        if self._routine is None or not self._routine.is_running:
            self._exit_running_state()
            return
        self._pause_resume_btn.setText(
            "Resume" if self._routine.is_paused else "Pause"
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def active_samples(self) -> list[_SampleRowWidget]:
        """Returns only the enabled sample rows."""
        return [r for r in self._sample_rows if r.enabled]