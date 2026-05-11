from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QPushButton, QWidget

from common.app_context import get_app_context
from common.logger import warning


class OpenFolderButton(QPushButton):
    """
    A button that opens a folder in the system file explorer.

    Hidden by default.  Call :meth:`set_folder` to assign a path and make
    it visible.  If the folder no longer exists when clicked, a warning is
    logged and an error toast is shown instead of attempting to open it.

    Typical usage::

        self._open_folder_btn = OpenFolderButton()
        layout.addWidget(self._open_folder_btn)

        # Once a scan starts:
        self._open_folder_btn.set_folder(output_folder)
    """

    def __init__(self, label: str = "Open Output Folder", parent: QWidget | None = None) -> None:
        super().__init__(label, parent)
        self._folder: str | None = None
        self.setFixedHeight(32)
        self.setVisible(False)
        self.clicked.connect(self._on_clicked)

    def set_folder(self, folder: str) -> None:
        """Assign *folder* and make the button visible."""
        self._folder = folder
        self.setVisible(True)

    def clear_folder(self) -> None:
        """Remove the assigned folder and hide the button."""
        self._folder = None
        self.setVisible(False)

    def _on_clicked(self) -> None:
        if self._folder is None:
            return
        if not Path(self._folder).is_dir():
            warning(f"OpenFolderButton: folder not found: {self._folder}")
            ctx = get_app_context()
            if ctx.toast is not None:
                ctx.toast.error("Output folder not found.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._folder))


class OpenFileButton(QPushButton):
    """
    A button that opens a file in its default application.

    Hidden by default.  Call :meth:`set_file` to assign a path and make it
    visible; the button is automatically hidden again if the file does not
    exist at that point.  If the file has been deleted by the time the user
    clicks, a warning is logged and an error toast is shown.

    Typical usage::

        self._view_btn = OpenFileButton("View Stacked Image")
        layout.addWidget(self._view_btn)

        # After a routine completes:
        self._view_btn.set_file(stacked_image_path)
    """

    def __init__(self, label: str = "Open File", parent: QWidget | None = None) -> None:
        super().__init__(label, parent)
        self._file: str | None = None
        self.setFixedHeight(30)
        self.setVisible(False)
        self.clicked.connect(self._on_clicked)

    def set_file(self, file_path: str) -> None:
        """Assign *file_path* and show the button only if the file currently exists."""
        self._file = file_path
        self.setVisible(Path(file_path).exists())

    def clear_file(self) -> None:
        """Remove the assigned file and hide the button."""
        self._file = None
        self.setVisible(False)

    def _on_clicked(self) -> None:
        if self._file is None:
            return
        if not Path(self._file).exists():
            warning(f"OpenFileButton: file not found: {self._file}")
            ctx = get_app_context()
            if ctx.toast is not None:
                ctx.toast.error("File not found.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._file))