from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QGroupBox,
    QLabel,
)

from common.app_context import FIELDWEAVE_VERSION

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


def about_page() -> QWidget:
    w = QWidget()
    layout = QVBoxLayout(w)

    top = QGroupBox("About FieldWeave")
    top_layout = QVBoxLayout(top)
    version = FIELDWEAVE_VERSION
    top_layout.addWidget(QLabel(f"Version {version}"))
    top_layout.addWidget(QLabel(
        "Created by Anthony van Weel."
    ))
    top_layout.addWidget(_link_label(f'Visit FieldWeave\'s Website at <a href="{WEBSITE_URL}">{WEBSITE_URL}</a>'))
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