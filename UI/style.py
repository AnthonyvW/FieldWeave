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

            color: white;
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

        /* When collapsed: header rounds bottom corners too (prevents “sticking out” corners) */
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

        
        """
    )