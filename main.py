from __future__ import annotations
import sys
import traceback
import multiprocessing as mp

import faulthandler
import sys
import traceback

import faulthandler
faulthandler.enable()

from PySide6.QtCore import QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import QApplication

# GUI
from UI.main_window import MainWindow
from UI.style import apply_style

# Initialize app context early
from common.app_context import get_app_context
from common.logger import info

def excepthook(exc_type, exc_value, exc_tb):
    traceback.print_exception(exc_type, exc_value, exc_tb)
    # Optionally re-raise or call sys.exit here
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = excepthook

def qt_message_handler(mode: QtMsgType, context: object, message: str) -> None:
    prefix = {
        QtMsgType.QtFatalMsg: "[Qt FATAL]",
        QtMsgType.QtCriticalMsg: "[Qt CRITICAL]",
        QtMsgType.QtWarningMsg: "[Qt WARNING]",
    }.get(mode, "[Qt]")
    print(f"{prefix} {message}", file=sys.stderr)
    if mode == QtMsgType.QtFatalMsg:
        sys.exit(1)

qInstallMessageHandler(qt_message_handler)

if __name__ == "__main__":

    mp.freeze_support()   
    mp.set_start_method("spawn", force=True)

    app = QApplication(sys.argv)
    apply_style(app)
    
    ctx = get_app_context()
    info("FieldWeave application starting")
    
    win = MainWindow()
    win.show()
    
    exit_code = app.exec()
    
    # Cleanup
    info("FieldWeave application shutting down")
    ctx.cleanup()
    
    sys.exit(exit_code)
