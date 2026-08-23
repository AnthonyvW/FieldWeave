from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QGroupBox,
    QLabel,
    QPushButton,
)

from common.app_context import get_fieldweave_version, get_app_context
from UI.widgets.changelog_dialog import ChangelogDialog

WEBSITE_URL = "https://www.fieldweave.com/"
CONTACT_URL = "https://www.fieldweave.com/contact"
SURVEY_URL = "https://forms.gle/kPGoiTHzmh6irCft7"
ISSUES_URL = "https://github.com/AnthonyvW/FieldWeave/issues/new"
DISCORD_URL = "https://discord.gg/nZh4uWUV4b"
GITHUB_SPONSORS_URL = "https://github.com/sponsors/AnthonyvW"
KOFI_URL = "https://ko-fi.com/procerand"


def _link_label(html: str) -> QLabel:
    label = QLabel(html)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setOpenExternalLinks(True)
    return label


def _check_updates_row(parent: QWidget) -> QWidget:
    """Button that triggers a manual update check, reusing the app-wide Updater/UpdateNotifier."""
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)

    button = QPushButton("Check for Updates")
    row_layout.addWidget(button)
    row_layout.addStretch()

    # Polls just to re-enable the button once the check finishes; the
    # UpdateNotifier itself (on the main thread) is what reads Updater
    # state and shows the resulting popup or toast.
    poll_timer = QTimer(parent)
    poll_timer.setInterval(300)

    def on_poll() -> None:
        updater = get_app_context().updater
        if updater is None or not updater.is_busy():
            poll_timer.stop()
            button.setEnabled(True)

    def on_click() -> None:
        notifier = get_app_context().update_notifier
        if notifier is None:
            return
        if notifier.check_for_updates(manual=True):
            button.setEnabled(False)
            poll_timer.start()

    poll_timer.timeout.connect(on_poll)
    button.clicked.connect(on_click)

    return row


def _changelog_row(parent: QWidget) -> QWidget:
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)

    button = QPushButton("View Changelog")
    row_layout.addWidget(button)
    row_layout.addStretch()

    def on_click() -> None:
        ChangelogDialog(parent).exec()

    button.clicked.connect(on_click)

    return row


def about_page() -> QWidget:
    w = QWidget()
    layout = QVBoxLayout(w)

    top = QGroupBox("About FieldWeave")
    top_layout = QVBoxLayout(top)
    version = get_fieldweave_version()
    top_layout.addWidget(QLabel(f"Version {version}"))
    top_layout.addWidget(QLabel(
        "Created by Anthony van Weel."
    ))
    top_layout.addWidget(_link_label(f'Visit FieldWeave\'s Website at <a href="{WEBSITE_URL}">{WEBSITE_URL}</a>'))
    top_layout.addWidget(_check_updates_row(w))
    top_layout.addWidget(_changelog_row(w))
    layout.addWidget(top)

    support = QGroupBox("Support FieldWeave")
    support_layout = QVBoxLayout(support)
    support_layout.addWidget(QLabel(
        "FieldWeave is free and open source. If it's useful to your lab, "
        "workflow, or project, please let us know below and consider "
        "supporting its development."
    ))
    support_layout.setSpacing(support_layout.spacing() + 4)
    support_layout.addWidget(_link_label(f'<a href="{SURVEY_URL}">Take the FieldWeave usage survey</a>'))
    support_layout.addWidget(_link_label(f'<a href="{GITHUB_SPONSORS_URL}">GitHub Sponsors</a>'))
    support_layout.addWidget(_link_label(f'<a href="{KOFI_URL}">Ko-fi</a>'))
    layout.addWidget(support)

    feedback = QGroupBox("Feedback && Community")
    feedback_layout = QVBoxLayout(feedback)
    feedback_layout.addWidget(_link_label(f'<a href="{CONTACT_URL}">Contact us with feature suggestions or feedback</a>'))
    feedback_layout.addWidget(_link_label(f'<a href="{ISSUES_URL}">Report a bug or request a feature on GitHub</a>'))
    feedback_layout.addWidget(_link_label(f'<a href="{DISCORD_URL}">Join the Discord server</a>'))
    layout.addWidget(feedback)

    layout.addStretch()

    return w