from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from common.updater import ReleaseInfo


class VersionDialog(QDialog):
    """Lets the user pick any published release to switch to; the caller performs the install."""

    prereleases_toggled = Signal(bool)

    def __init__(
        self,
        releases: list[ReleaseInfo],
        current_version: str,
        include_prereleases: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Change FieldWeave Version")
        self.resize(720, 540)

        self._all_releases = releases
        self._current_version = current_version
        self._visible: list[ReleaseInfo] = []
        self.selected_tag: str | None = None

        layout = QVBoxLayout(self)
        header = QLabel(f"Currently running version {current_version}. Select a version to switch to:")
        header.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._list = QListWidget(splitter)
        self._notes = QTextBrowser(splitter)
        self._notes.setOpenExternalLinks(True)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)

        self._beta_checkbox = QCheckBox("Show beta versions (use at your own risk)", self)
        self._beta_checkbox.setChecked(include_prereleases)
        layout.addWidget(self._beta_checkbox)

        buttons = QHBoxLayout()
        buttons.addStretch()

        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(cancel_button)

        self._install_button = QPushButton("Switch to Version", self)
        self._install_button.clicked.connect(self._confirm)
        buttons.addWidget(self._install_button)

        layout.addLayout(buttons)

        self._list.currentRowChanged.connect(self._on_selection_changed)
        self._beta_checkbox.toggled.connect(self._on_beta_toggled)
        self._populate()

    def _populate(self) -> None:
        show_betas = self._beta_checkbox.isChecked()
        self._visible = [r for r in self._all_releases if show_betas or not r.prerelease]

        self._list.clear()
        current_row = 0
        for row, release in enumerate(self._visible):
            label = release.version
            if release.prerelease:
                label += " (beta)"
            if release.version == self._current_version:
                label += " (current)"
                current_row = row
            self._list.addItem(QListWidgetItem(label))

        self._list.setCurrentRow(current_row if self._visible else -1)
        self._on_selection_changed(self._list.currentRow())

    def _on_beta_toggled(self, checked: bool) -> None:
        self.prereleases_toggled.emit(checked)
        self._populate()

    def _on_selection_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._visible):
            self._install_button.setEnabled(False)
            self._notes.clear()
            return

        release = self._visible[row]
        self._install_button.setEnabled(release.version != self._current_version)

        beta_note = "**This is a beta release and may be unstable.**\n\n" if release.prerelease else ""
        notes = release.notes.replace("\r\n", "\n").strip() or "No release notes provided."
        self._notes.setMarkdown(f"# FieldWeave v{release.version}\n\n{beta_note}{notes}")

    def _confirm(self) -> None:
        row = self._list.currentRow()
        if row < 0 or row >= len(self._visible):
            return

        release = self._visible[row]
        warning = f"Switch to FieldWeave {release.version}? A restart is required."
        if release.prerelease:
            warning += "\n\nThis is a beta release and may be unstable. Use it at your own risk."
        warning += "\n\nSettings saved by a different version may not be compatible."

        reply = QMessageBox.warning(
            self,
            "Confirm Version Change",
            warning,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.selected_tag = release.tag
            self.accept()
