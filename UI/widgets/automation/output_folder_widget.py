from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from UI.widgets.utilities.open_filesystem_object_button import OpenFileButton, OpenFolderButton


class OutputFolderWidget(QWidget):
    """
    A reusable group box containing a text field and Browse button for
    selecting an output folder.

    If the user leaves the field empty, :py `resolved_path` returns
    ``./output/<timestamp>``.  A relative path is rooted under ``./output/``.
    An absolute path is returned as-is.
    """

    _DEFAULT_PLACEHOLDER: str = "Default: ./output/<timestamp>"

    def __init__(
        self,
        label: str = "Output Folder",
        initial_path: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        outer = QGroupBox(label)
        layout = QHBoxLayout(outer)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        self._edit = QLineEdit()
        self._edit.setFixedHeight(30)
        self._edit.setPlaceholderText(self._DEFAULT_PLACEHOLDER)
        if initial_path:
            self._edit.setText(initial_path)
        layout.addWidget(self._edit, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedHeight(30)
        browse_btn.clicked.connect(self._browse)
        layout.addWidget(browse_btn)

        wrapper = QHBoxLayout(self)
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addWidget(outer)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def text(self) -> str:
        """Raw text currently in the input field (may be empty)."""
        return self._edit.text().strip()

    @text.setter
    def text(self, value: str) -> None:
        self._edit.setText(value)

    @property
    def resolved_path(self) -> str:
        """
        The effective output path, always as a string.

        - Empty  -> ``./output/<timestamp>``
        - Relative -> rooted under ``./output/``
        - Absolute -> returned unchanged
        """
        text = self._edit.text().strip()
        if not text:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return str(Path("output") / timestamp)
        p = Path(text)
        if p.is_absolute():
            return text
        return str(Path("output") / p)

    @staticmethod
    def confirm_if_exists(path: str, parent: QWidget | None = None) -> bool:
        """
        Return True if it is safe to proceed with *path* as the output folder.

        If the folder does not yet exist, returns True immediately.  If it
        exists and already contains files, shows a warning dialog giving the
        user the choice to continue anyway or cancel.  An empty existing
        directory is treated as safe and returns True without prompting.
        """
        p = Path(path)
        if not p.exists():
            return True
        existing_files = list(p.iterdir())
        if not existing_files:
            return True
        msg = QMessageBox(parent)
        msg.setWindowTitle("Folder Already Exists")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(f"The output folder already contains {len(existing_files)} item(s):")
        msg.setInformativeText(
            f"<b>{path}</b><br><br>"
            "Saving here may mix new images with existing ones. "
            "Continue anyway?"
        )
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        msg.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return msg.exec() == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            self._edit.text().strip() or "./output/",
        )
        if folder:
            self._edit.setText(folder)


class ViewImageWidget(QWidget):
    """
    A reusable row with an "Open Folder" button and a configurable
    "View Image" button. Hidden by default; call :py `show_result` to
    reveal it after a routine completes.

    Parameters
    ----------
    view_label:
        Text shown on the view-image button (e.g. ``"View Stacked Image"``).
    """

    def __init__(
        self,
        view_label: str = "View Image",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._open_folder_btn = OpenFolderButton("Open Folder")
        self._open_folder_btn.setFixedHeight(30)
        layout.addWidget(self._open_folder_btn, 1)

        self._view_image_btn = OpenFileButton(view_label)
        layout.addWidget(self._view_image_btn, 1)

        self.setVisible(False)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def show_result(self, output_folder: str, image_path: str | None = None) -> None:
        """
        Reveal the widget after a routine finishes.

        Parameters
        ----------
        output_folder:
            Path to open when "Open Folder" is clicked.
        image_path:
            Path to the image file for the view button.  When ``None`` or the
            file does not exist, the view button is hidden.
        """
        self._open_folder_btn.set_folder(output_folder)
        if image_path is not None:
            self._view_image_btn.set_file(image_path)
        else:
            self._view_image_btn.clear_file()
        self.setVisible(True)

    def hide_result(self) -> None:
        """Hide the widget and clear stored paths."""
        self._open_folder_btn.clear_folder()
        self._view_image_btn.clear_file()
        self.setVisible(False)