from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QPushButton, QTextBrowser, QVBoxLayout, QWidget

from UI.changelog import load_changelog


class ChangelogDialog(QDialog):
    """Renders the combined UI/changelog markdown files as formatted release notes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Changelog")
        self.resize(560, 640)

        layout = QVBoxLayout(self)

        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(True)

        markdown = load_changelog()
        if markdown is not None:
            browser.setMarkdown(markdown)
        else:
            browser.setPlainText("Changelog is unavailable.")

        layout.addWidget(browser)

        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)