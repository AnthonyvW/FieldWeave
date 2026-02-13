from __future__ import annotations

from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QFormLayout,
    QGroupBox,
)
    
def navigation_page() ->QWidget:
    w = QWidget()
    layout = QVBoxLayout(w)

    top = QGroupBox("Navigation Settings")
    form = QFormLayout(top)
    layout.addWidget(top)

    return w