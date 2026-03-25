from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, QMimeData, QTimer
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
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
from common.logger import error, warning, info


NUM_SLOTS = 20


# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

def _button_style() -> str:
    return """
        QPushButton {
            background-color: rgb(208, 211, 214);
            border: 1px solid rgb(150, 150, 150);
            border-radius: 0px;
            font-size: 13px;
            padding: 0 8px;
        }
        QPushButton:hover {
            background-color: rgb(187, 190, 193);
        }
        QPushButton:pressed {
            background-color: rgb(170, 173, 175);
        }
        QPushButton:disabled {
            background-color: rgb(225, 227, 229);
            color: rgb(160, 163, 166);
            border: 1px solid rgb(190, 193, 196);
        }
    """


# ---------------------------------------------------------------------------
# Single sample row
# ---------------------------------------------------------------------------

_ROW_BG_INACTIVE = "rgb(245, 246, 247)"
_ROW_BG_ACTIVE   = "rgb(242, 140, 40)"   # orange

_EDIT_INACTIVE = """
    QLineEdit {
        font-size: 13px;
        padding: 2px 4px;
        border: 1px solid rgb(180, 180, 180);
        border-radius: 0px;
        background-color: rgb(235, 237, 239);
        color: rgb(100, 102, 104);
    }
"""
# Inactive row but the field has content — white background so it stands out.
_EDIT_INACTIVE_FILLED = """
    QLineEdit {
        font-size: 13px;
        padding: 2px 4px;
        border: 1px solid rgb(160, 160, 160);
        border-radius: 0px;
        background-color: rgb(255, 255, 255);
        color: rgb(40, 40, 40);
    }
"""
_EDIT_ACTIVE = """
    QLineEdit {
        font-size: 13px;
        padding: 2px 4px;
        border: 1px solid rgb(200, 100, 0);
        border-radius: 0px;
        background-color: rgb(255, 210, 160);
        color: rgb(60, 30, 0);
    }
"""


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

            # Split, strip each line, drop empties produced by trailing newlines
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

        # Split, strip each line, drop empties produced by trailing newlines
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

        # Required so that setStyleSheet background-color actually paints
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        self._toggle = QCheckBox()
        self._toggle.setChecked(False)
        self._toggle.setFixedWidth(20)
        layout.addWidget(self._toggle)

        self._id_label = QLabel(f"{sample_number:02d}")
        self._id_label.setFixedWidth(28)
        self._id_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._id_label)

        # Bind slot_index into the callback so the parent always knows the source
        bound_overflow: Callable[[list[str]], None] = lambda lines: overflow_callback(slot_index, lines)
        self._name_edit = _MultilinePasteEdit(bound_overflow)
        self._name_edit.setFixedHeight(26)
        self._name_edit.setPlaceholderText(f"Sample {sample_number} name...")
        layout.addWidget(self._name_edit, 1)

        self._toggle.toggled.connect(self._on_toggle_changed)
        self._name_edit.textChanged.connect(self._on_text_changed)

        self._apply_style()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_style(self) -> None:
        active = self._toggle.isChecked()
        self.setStyleSheet(
            f"_SampleRowWidget {{ background-color: {_ROW_BG_ACTIVE}; }}"
            if active else
            f"_SampleRowWidget {{ background-color: {_ROW_BG_INACTIVE}; }}"
        )
        self._id_label.setStyleSheet(
            "font-size: 13px; color: rgb(60, 30, 0);"
            if active else
            "font-size: 13px; color: #666;"
        )
        if active:
            self._name_edit.setStyleSheet(_EDIT_ACTIVE)
        elif self._name_edit.text().strip():
            self._name_edit.setStyleSheet(_EDIT_INACTIVE_FILLED)
        else:
            self._name_edit.setStyleSheet(_EDIT_INACTIVE)

    def _on_toggle_changed(self, _checked: bool) -> None:
        self._apply_style()

    def _on_text_changed(self, text: str) -> None:
        if not self._ever_typed and text.strip():
            self._ever_typed = True
            if not self._toggle.isChecked():
                self._toggle.setChecked(True)  # triggers _apply_style via signal
                return  # _apply_style already called via signal
        self._apply_style()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
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
        main_layout.addWidget(self._build_sample_list_group(), 1)

        # Timer for polling routine state on the UI thread
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_routine_state)

    def _build_controls_group(self) -> QGroupBox:
        group = QGroupBox("Controls")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: normal;
                border: 1px solid rgb(180, 180, 180);
                border-radius: 0px;
                margin-top: 6px;
                padding-top: 4px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 3px;
            }
        """)

        layout = QHBoxLayout(group)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        slot_label = QLabel("Slot:")
        slot_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(slot_label)

        self._slot_spin = QSpinBox()
        self._slot_spin.setFixedHeight(30)
        self._slot_spin.setMinimum(1)
        self._slot_spin.setMaximum(NUM_SLOTS)
        self._slot_spin.setValue(1)
        self._slot_spin.setFixedWidth(60)
        self._slot_spin.setStyleSheet("""
            QSpinBox {
                font-size: 13px;
                padding: 2px 4px;
                border: 1px solid rgb(180, 180, 180);
                border-radius: 0px;
            }
        """)
        layout.addWidget(self._slot_spin)

        go_btn = QPushButton("Go to Slot")
        go_btn.setFixedHeight(30)
        go_btn.setStyleSheet(_button_style())
        go_btn.clicked.connect(self._on_go_to_slot_clicked)
        layout.addWidget(go_btn)

        layout.addStretch(1)

        self._start_btn = QPushButton("Start Automation")
        self._start_btn.setFixedHeight(30)
        self._start_btn.setStyleSheet("""
            QPushButton {
                background-color: #f28c28;
                color: white;
                border: 1px solid #c97020;
                border-radius: 0px;
                font-size: 13px;
                font-weight: bold;
                padding: 0 12px;
            }
            QPushButton:hover {
                background-color: #d97a20;
            }
            QPushButton:pressed {
                background-color: #bf6a18;
            }
            QPushButton:disabled {
                background-color: rgb(208, 211, 214);
                color: rgb(150, 153, 156);
                border: 1px solid rgb(170, 173, 176);
            }
        """)
        self._start_btn.clicked.connect(self._on_start_clicked)
        layout.addWidget(self._start_btn)

        return group

    def _build_sample_list_group(self) -> QGroupBox:
        group = QGroupBox("Samples")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: normal;
                border: 1px solid rgb(180, 180, 180);
                border-radius: 0px;
                margin-top: 6px;
                padding-top: 4px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 3px;
            }
        """)

        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(6, 6, 6, 6)
        group_layout.setSpacing(0)

        # Header row
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(6, 2, 6, 4)
        header_layout.setSpacing(8)

        enabled_hdr = QLabel("On")
        enabled_hdr.setFixedWidth(20)
        enabled_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        enabled_hdr.setStyleSheet("font-size: 11px; color: #666; font-weight: bold;")
        header_layout.addWidget(enabled_hdr)

        id_hdr = QLabel("ID")
        id_hdr.setFixedWidth(28)
        id_hdr.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        id_hdr.setStyleSheet("font-size: 11px; color: #666; font-weight: bold;")
        header_layout.addWidget(id_hdr)

        name_hdr = QLabel("Sample Name")
        name_hdr.setStyleSheet("font-size: 11px; color: #666; font-weight: bold;")
        header_layout.addWidget(name_hdr, 1)

        group_layout.addWidget(header)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: rgb(200, 200, 200);")
        group_layout.addWidget(divider)

        for i in range(1, NUM_SLOTS + 1):
            row = _SampleRowWidget(i, i - 1, self._on_paste_overflow)
            self._sample_rows.append(row)
            group_layout.addWidget(row)

            if i < NUM_SLOTS:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setStyleSheet("color: rgb(220, 222, 224);")
                group_layout.addWidget(sep)

        return group

    # ------------------------------------------------------------------
    # Size hint — tall enough to show all 20 rows without a scrollbar
    # ------------------------------------------------------------------

    def sizeHint(self):  # type: ignore[override]
        hint = super().sizeHint()
        # Each row is ~32 px tall; 20 rows + header + separators + group chrome
        hint.setHeight(max(hint.height(), 900))
        return hint

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_go_to_slot_clicked(self) -> None:
        slot = self._slot_spin.value()
        ctx = get_app_context()
        if ctx.motion is None or not ctx.motion.is_ready():
            warning("TreeCoreWidget: motion controller not ready")
            return
        try:
            ctx.motion.go_to_slot(slot)
        except Exception as exc:
            error(f"TreeCoreWidget: failed to go to slot {slot} — {exc}")

    def _on_start_clicked(self) -> None:
        active_samples = [r for r in self._sample_rows if r.enabled]
        if not active_samples:
            warning("TreeCoreWidget: no samples enabled")
            return

        ctx = get_app_context()
        if ctx.motion is None or not ctx.motion.is_ready():
            error("TreeCoreWidget: motion controller not ready — cannot start automation")
            return

        self._enter_running_state()

    def _on_paste_overflow(self, source_index: int, lines: list[str]) -> None:
        """Distribute overflow lines from a multi-line paste into subsequent slots.

        ``source_index`` is the zero-based index of the row that received the
        paste; ``lines`` contains everything after the first line.
        """
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
        self._poll_timer.start()

    def _exit_running_state(self) -> None:
        self._poll_timer.stop()
        self._start_btn.setEnabled(True)

    def _poll_routine_state(self) -> None:
        ctx = get_app_context()
        if ctx.motion is None or not ctx.motion.routine_running:
            self._exit_running_state()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def active_samples(self) -> list[_SampleRowWidget]:
        """Returns only the enabled sample rows."""
        return [r for r in self._sample_rows if r.enabled]