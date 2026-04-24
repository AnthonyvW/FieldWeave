from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, QMimeData, QTimer
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from common.app_context import get_app_context
from common.logger import error, warning
from motion.models import Position
from motion.routines.tree_core_imaging_routine import TreeCoreImagingRoutine
from UI.widgets.automation.output_folder_widget import OutputFolderWidget

_NM_PER_MM = 1_000_000


def _get_tca():
    """Return the TreeCoreAutomationSettings from the motion context, or None."""
    ctx = get_app_context()
    if ctx is None or ctx.motion is None:
        return None
    return getattr(ctx.motion.settings, "tree_core_automation", None)


# ---------------------------------------------------------------------------
# Confirmation dialog
# ---------------------------------------------------------------------------

class _ConfirmDialog(QDialog):
    """Modal dialog summarising the planned tree core imaging run."""

    def __init__(
        self,
        output_path: str,
        slot_count: int,
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

        for label_text, value_text in [
            ("Slots to image", str(slot_count)),
            ("Output path", output_path),
        ]:
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

    def set_text(self, text: str) -> None:
        """Programmatically set the sample name (used for multi-line paste)."""
        self._name_edit.setText(text)

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
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        main_layout.addWidget(self._build_controls_group())

        self._output_folder = OutputFolderWidget()
        main_layout.addWidget(self._output_folder)

        main_layout.addWidget(self._build_sample_list_group(), 1)

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

        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._status_label.setObjectName("CalScaleStatusLabel")
        main_layout.addWidget(self._status_label)

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

        self._slot_spin.setMaximum(num_slots)

    # ------------------------------------------------------------------
    # Size hint
    # ------------------------------------------------------------------

    def sizeHint(self):  # type: ignore[override]
        hint = super().sizeHint()
        hint.setHeight(max(hint.height(), 900))
        return hint

    # ------------------------------------------------------------------
    # showEvent — refresh slot count from calibration
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        tca = _get_tca()
        num_slots = tca.num_slots if tca is not None else 20
        if len(self._sample_rows) != num_slots:
            self._rebuild_sample_rows(num_slots)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

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

        try:
            current = ctx.motion.get_position()
            main_nm = tca.mark_reference_nm if mark_set else (current.y if axis == "y" else current.x)
            z_nm = tca.mark_z_nm if mark_set else current.z
            if axis == "y":
                target = Position(x=pos_nm, y=main_nm, z=z_nm)
            else:
                target = Position(x=main_nm, y=pos_nm, z=z_nm)
            ctx.motion.move_to_position(target, wait=False)
        except Exception as exc:
            error(f"TreeCoreWidget: failed to go to slot {slot_number} — {exc}")

    def _on_start_clicked(self) -> None:
        ctx = get_app_context()
        if ctx.motion is None or not ctx.motion.is_ready():
            error("TreeCoreWidget: motion controller not ready")
            self._status_label.setText("Motion controller not ready.")
            return

        active_samples = [r for r in self._sample_rows if r.enabled]
        if not active_samples:
            warning("TreeCoreWidget: no samples enabled")
            self._status_label.setText("No samples enabled.")
            return

        tca = _get_tca()
        if tca is None or not tca.has_been_calibrated:
            error("TreeCoreWidget: slot calibration has not been completed")
            self._status_label.setText("Slot calibration has not been completed.")
            return

        output_path = self._output_folder.resolved_path
        if not OutputFolderWidget.confirm_if_exists(output_path, self):
            return

        slots = [(r.sample_number - 1, r.name or f"slot_{r.sample_number:02d}") for r in active_samples]

        dlg = _ConfirmDialog(
            output_path=output_path,
            slot_count=len(slots),
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            self._routine = TreeCoreImagingRoutine(
                motion=ctx.motion,
                output_folder=output_path,
                slots=slots,
            )
            ctx.motion.start_routine(self._routine)
        except Exception as exc:
            error(f"TreeCoreWidget: failed to start routine — {exc}")
            self._status_label.setText(f"Failed to start: {exc}")
            return

        self._enter_running_state()

    def _on_pause_resume_clicked(self) -> None:
        if self._routine is None:
            return
        if self._routine.is_paused:
            self._routine.resume()
            self._pause_resume_btn.setText("Pause")
            self._status_label.setText("Running...")
        else:
            self._routine.pause()
            self._pause_resume_btn.setText("Resume")
            self._status_label.setText("Paused.")

    def _on_stop_clicked(self) -> None:
        if self._routine is not None:
            self._routine.stop()
        self._status_label.setText("Stopping...")

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
        self._pause_resume_btn.setText("Pause")
        self._controls_widget.setVisible(True)
        self._status_label.setText("Running...")
        self._poll_timer.start()

    def _exit_running_state(self) -> None:
        self._poll_timer.stop()
        self._start_btn.setEnabled(True)
        self._output_folder.setEnabled(True)
        self._controls_widget.setVisible(False)
        self._status_label.setText("Finished.")
        self._routine = None

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