import sys
import multiprocessing as mp

from PySide6.QtWidgets import QApplication

# GUI
from UI.main_window import MainWindow
from UI.style import apply_style

# Initialize app context early
from app_context import get_app_context
from logger import info


if __name__ == "__main__":
    mp.freeze_support()   
    mp.set_start_method("spawn", force=True)

    app = QApplication(sys.argv)
    apply_style(app)
    
    # Initialize app context (this will load camera SDK and config)
    ctx = get_app_context()
    info("Forge application starting")
    
    win = MainWindow()
    win.show()
    
    exit_code = app.exec()
    
    # Cleanup
    info("Forge application shutting down")
    ctx.cleanup()
    
    sys.exit(exit_code)
