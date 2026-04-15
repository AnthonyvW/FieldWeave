from __future__ import annotations

from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication

RIGHT_SIDEBAR_WIDTH = 380
OUTER_MARGIN = 10
CAL_LEFT_WIDTH = 260

def apply_style(app: QApplication) -> None:
    palette = app.palette()

    window_bg = QColor(215, 218, 222)
    panel_bg = QColor(245, 246, 248)
    text = QColor(35, 35, 35)

    palette.setColor(QPalette.ColorRole.Window, window_bg)
    palette.setColor(QPalette.ColorRole.Base, panel_bg)
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(235, 237, 240))

    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Button, QColor(238, 240, 243))
    palette.setColor(QPalette.ColorRole.ButtonText, text)

    app.setPalette(palette)

    header_bar_color = "#5f6368" # Dark Gray
    header_bar_text_color = "#ffffff"
    header_bar_selected_color = "#f28c28" # Orange
    header_bar_selected_text_color = "#ffffff"
    tab_corner_button = "#ffffff"

    header_bar_idle = "#5f6368" # Dark Gray
    header_bar_active = "#f28c28" # Orange
    header_bar_finished = "#2e9b51" # Green

    corner_status_line_color = "#ffffff"

    app.setStyleSheet(
        f"""
        QTabWidget::pane {{ border: none; }}
        
        /* Header Bar */
        QTabBar {{
            background : {header_bar_color};
            color: {header_bar_text_color};
        }}
        QTabBar::Tab {{
            padding: 8px 12px;
            margin: 0px;
            border-radius: 0px;
            background: transparent;
        }}
        QTabBar::tab:selected {{
            background: {header_bar_selected_color};
            color: {header_bar_selected_text_color};
        }}


        /* Corner Widget */
        QWidget#TabCorner {{
            background : {header_bar_color};
            padding: 0px;
            margin: 0px;
        }}
        QWidget#TabCorner QToolButton {{
            color: {tab_corner_button};
            background : transparent;
        }}

        /* Push Buttons - Grey styling */
        QPushButton {{
            background-color: #d0d3d6;
            border: 1px solid #b0b3b6;
            border-radius: 0px;
            padding: 2px 8px;
            font-size: 13px;
            color: #2c2c2c;
        }}
        QPushButton:hover {{
            background-color: #c0c3c6;
            border-color: #a0a3a6;
        }}
        QPushButton:pressed {{
            background-color: #b0b3b6;
            border-color: #909396;
        }}
        QPushButton:disabled {{
            background-color: #e0e3e6;
            border-color: #d0d3d6;
            color: #a0a3a6;
        }}

        
        /* Status panel in tab corner */
        QFrame#StatusBar {{
            padding: 0px 10px;
            border-radius: 0px;
            margin: 0px;
        }}
        QLabel#StatusLine {{
            color: {corner_status_line_color};
            font-weight: 800;
        }}

        /* Status State */
        QFrame#StatusBar[kind="idle"] {{
            background: {header_bar_idle};
        }}
        QFrame#StatusBar[kind="active"] {{
            background: {header_bar_active};
        }}
        QFrame#StatusBar[kind="done"] {{
            background: {header_bar_finished};
        }}

        /* Status Progress Bar */
        QProgressBar#CornerStatusProgress {{
            border: none;
            background: rgba(255,255,255,0.22);
            border-radius: 4px;
            height: 8px;

            color: {header_bar_color};
            font-weight: 800;
        }}
        QProgressBar#CornerStatusProgress::chunk {{
            background: rgba(255,255,255,0.95);
            border-radius: 4px;
        }}


        /* Collapsible section box */
        QFrame#CollapsibleSection {{
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(0,0,0,0.10);
        }}

        /* Full-width header strip: dark grey */
        QFrame#SectionHeader {{
            background: #5f6368;
            border-bottom: 1px solid rgba(0,0,0,0.10);
        }}
        QLabel#SectionHeaderTitle, QFrame#SectionHeader QLabel {{
            color: white;
            font-weight: 800;
        }}

        /* When collapsed: header rounds bottom corners too (prevents "sticking out" corners) */
        QFrame#SectionHeader[collapsed="true"] {{
            border-bottom: none;
        }}

        QListWidget#SampleList {{
            background: rgba(255,255,255,0.95);
            border: 1px solid rgba(0,0,0,0.10);
            border-radius: 10px;
        }}

        QFrame#StepCard {{
            background: rgba(0,0,0,0.03);
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 10px;
        }}

        /* Calibration selection panels: flat */
        QFrame#CalLeft, QFrame#CalMid {{
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(0,0,0,0.10);
            border-radius: 12px;
        }}

        /* Selected calibration title bar */
        QFrame#CalTitleBar {{
            background: rgba(0,0,0,0.10);
            border-top-left-radius: 12px;
            border-top-right-radius: 12px;
            border-bottom: 1px solid rgba(0,0,0,0.08);
        }}
        QLabel#CalTitleText {{
            font-size: 18px;
            font-weight: 900;
            color: rgba(0,0,0,0.80);
        }}
        QLabel#CalNotesText {{
            color: rgba(0,0,0,0.62);
        }}

        /* Camera Preview */
        QFrame#CameraPreview {{
            background: #000000;
        }}
        
        QLabel#VideoLabel {{
            color: #888888;
            font-size: 16px;
        }}

        /* Camera Preview Overlay Buttons */
        QPushButton#OverlayButton, QPushButton#CrosshairButton, QPushButton#FocusButton, QPushButton#ChannelButton {{
            background-color: rgba(240, 240, 240, 180);
            color: #000;
            border: 1px solid rgba(200, 200, 200, 255);
            border-radius: 4px;
            font-size: 18px;
            font-weight: bold;
        }}
        QPushButton#OverlayButton:hover, QPushButton#CrosshairButton:hover, QPushButton#FocusButton:hover, QPushButton#ChannelButton:hover {{
            background-color: rgba(255, 255, 255, 200);
        }}
        QPushButton#OverlayButton:checked, QPushButton#CrosshairButton:checked, QPushButton#FocusButton:checked, QPushButton#ChannelButton:checked {{
            background-color: rgba(100, 150, 200, 200);
            color: white;
            border: 2px solid rgba(150, 200, 255, 255);
        }}
        
        QPushButton#CrosshairButton {{
            padding-bottom: 4px;
        }}
        
        /* Hide-preview eye toggle button — matches overlay button style */
        QPushButton#HidePreviewButton {{
            background-color: rgba(240, 240, 240, 180);
            color: #000000;
            border: 1px solid rgba(200, 200, 200, 255);
            border-radius: 4px;
            font-size: 18px;
            font-weight: bold;
            padding: 0px;
        }}
        QPushButton#HidePreviewButton:hover {{
            background-color: rgba(255, 255, 255, 200);
        }}

        /* Preview-disabled overlay */
        QLabel#PreviewHiddenLabel {{
            color: #ffffff;
            font-size: 15px;
            font-weight: 600;
            background-color: rgba(0, 0, 0, 160);
            border-radius: 6px;
            padding: 6px 14px;
        }}
        QPushButton#PreviewReenableButton {{
            background-color: rgba(240, 240, 240, 210);
            border: 1px solid rgba(200, 200, 200, 255);
            border-radius: 0px;
            font-size: 13px;
            font-weight: bold;
            color: #2c2c2c;
            padding: 6px 18px;
            min-width: 130px;
            min-height: 28px;
        }}
        QPushButton#PreviewReenableButton:hover {{
            background-color: rgba(255, 255, 255, 230);
        }}
        QPushButton#PreviewReenableButton:pressed {{
            background-color: rgba(210, 213, 216, 230);
        }}
        
        QLabel#FocusOverlayLabel, QLabel#VennOverlayLabel {{
            font-size: 18px;
            font-weight: normal;
        }}

        /* Automation mode combo box */
        QComboBox {{
            background-color: #ffffff;
            border: 1px solid #b0b3b6;
            border-radius: 0px;
            padding: 2px 28px 2px 6px;
            color: #2c2c2c;
        }}
        QComboBox QAbstractItemView {{
            background-color: #ffffff;
            border: 1px solid #b0b3b6;
            selection-background-color: #d0d3d6;
            color: #2c2c2c;
        }}

        /* Automation control bar — Pause button */
        QPushButton#AutomationPause {{
            background-color: rgb(208, 211, 214);
            border: 1px solid rgb(150, 150, 150);
            border-radius: 0px;
            font-size: 13px;
            font-weight: normal;
            color: #2c2c2c;
        }}
        QPushButton#AutomationPause:hover:enabled {{
            background-color: rgb(187, 190, 193);
        }}
        QPushButton#AutomationPause:pressed:enabled {{
            background-color: rgb(170, 173, 175);
        }}
        QPushButton#AutomationPause:checked {{
            background-color: #f28c28;
            border: 1px solid #c97220;
            color: white;
            font-weight: bold;
        }}
        QPushButton#AutomationPause:disabled {{
            background-color: rgb(225, 227, 229);
            border: 1px solid rgb(190, 190, 190);
            color: rgb(170, 170, 170);
        }}

        /* Automation control bar — Stop button */
        QPushButton#AutomationStop {{
            background-color: rgb(200, 60, 60);
            border: 1px solid rgb(160, 40, 40);
            border-radius: 0px;
            font-size: 13px;
            font-weight: bold;
            color: white;
        }}
        QPushButton#AutomationStop:hover:enabled {{
            background-color: rgb(220, 70, 70);
        }}
        QPushButton#AutomationStop:pressed:enabled {{
            background-color: rgb(170, 40, 40);
        }}
        QPushButton#AutomationStop:disabled {{
            background-color: rgb(210, 150, 150);
            border: 1px solid rgb(190, 130, 130);
            color: rgb(240, 210, 210);
        }}

        /* Group boxes */
        QGroupBox {{
            font-size: 13px;
            font-weight: normal;
            border: 1px solid rgb(180, 180, 180);
            border-radius: 0px;
            margin-top: 6px;
            padding-top: 4px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 3px;
        }}

        /* Spin boxes */
        QDoubleSpinBox {{
            font-size: 13px;
            padding: 2px 4px;
            border: 1px solid rgb(180, 180, 180);
            border-radius: 0px;
        }}

        /* Line edits */
        QLineEdit {{
            font-size: 13px;
            padding: 2px 4px;
            border: 1px solid rgb(180, 180, 180);
            border-radius: 0px;
        }}

        /* Area scan — start automation button */
        QPushButton#AreaScanStart {{
            background-color: #f28c28;
            color: white;
            border: 1px solid #c97020;
            font-weight: bold;
        }}
        QPushButton#AreaScanStart:hover {{
            background-color: #d97a20;
        }}
        QPushButton#AreaScanStart:pressed {{
            background-color: #bf6a18;
        }}
        QPushButton#AreaScanStart:disabled {{
            background-color: rgb(208, 211, 214);
            color: rgb(150, 153, 156);
            border: 1px solid rgb(170, 173, 176);
        }}

        /* Area scan — label roles */
        QLabel#AreaScanDialogTitle {{
            font-size: 14px;
            font-weight: bold;
        }}
        QLabel#AreaScanRowLabel {{
            font-size: 13px;
            color: #555;
        }}
        QLabel#AreaScanRowValue {{
            font-size: 13px;
            font-weight: bold;
        }}
        QLabel#AreaScanAxisReadout {{
            font-size: 13px;
            color: #555;
        }}
        QLabel#AreaScanMinLabel {{
            font-size: 11px;
            color: #777;
        }}
        QLabel#AreaScanSummary {{
            font-size: 12px;
            color: #444;
            padding: 2px 0;
        }}

        """
    )