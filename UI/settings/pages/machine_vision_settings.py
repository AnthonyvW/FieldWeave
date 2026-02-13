from __future__ import annotations

from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QFormLayout,
    QGroupBox,
)
    
def machine_vision_page() ->QWidget:
    w = QWidget()
    layout = QVBoxLayout(w)

    top = QGroupBox("Machine Vision Settings")
    form = QFormLayout(top)
    layout.addWidget(top)

    return w