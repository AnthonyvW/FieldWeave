from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path

from PIL import ExifTags
from PIL import Image as PILImage
from PySide6.QtCore import Slot, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from common.app_context import get_app_context
from common.logger import info, error, warning
from common.setting_types import FileFormat


class CaptureMode(Enum):
    LIVE = "live"
    LOADED_IMAGE = "loaded_image"


class MetadataDialog(QDialog):
    """Read-only key/value table shown for both live and loaded-image metadata."""

    def __init__(self, title: str, metadata: dict[str, str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(460, 380)

        layout = QVBoxLayout(self)

        table = QTableWidget(len(metadata), 2, self)
        table.setHorizontalHeaderLabels(["Field", "Value"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setWordWrap(True)

        for row, (key, value) in enumerate(metadata.items()):
            table.setItem(row, 0, QTableWidgetItem(key))
            table.setItem(row, 1, QTableWidgetItem(value))

        layout.addWidget(table)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)


class CaptureControlWidget(QWidget):
    """
    Measurement tab's capture widget: live camera vs. loaded-image source,
    three capture actions, output/format controls, a metadata viewer, and
    a stub for measurement export.

    Kept single-column throughout (one control per row) except where
    buttons are short enough to sit side by side, so it stays within
    RIGHT_SIDEBAR_WIDTH without triggering the sidebar's horizontal
    scrollbar, matching CameraControlsWidget.

    The loaded-image overlay this widget drives lives on the shared
    CameraPreview, so it must only be enabled while the measurement tab is
    actually the one showing that preview — see ``set_tab_active``, called
    by MeasurementTab's showEvent/hideEvent.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._default_folder = Path("./output/measurements")
        self._current_folder = self._default_folder
        self._ensure_output_folder()

        self._mode = CaptureMode.LIVE
        self._tab_active = False
        self._camera_available = False

        self._loaded_pixmap: QPixmap | None = None
        self._loaded_image_path: Path | None = None

        # Capture state — mutated only by the on_complete callback on the
        # camera thread; read only by _poll_capture_state on the main thread.
        self._capture_pending = False
        self._capture_success: bool | None = None
        self._capture_filepath = ""
        self._capture_toast_id: int | None = None

        self._setup_ui()

        self._capture_poll_timer = QTimer(self)
        self._capture_poll_timer.setInterval(100)
        self._capture_poll_timer.timeout.connect(self._poll_capture_state)

        self._camera_poll_timer = QTimer(self)
        self._camera_poll_timer.setInterval(500)
        self._camera_poll_timer.timeout.connect(self._refresh_capture_availability)
        self._camera_poll_timer.start()

        self._refresh_capture_availability()

    # ------------------------------------------------------------------
    # Tab-visibility wiring
    # ------------------------------------------------------------------

    def set_tab_active(self, active: bool) -> None:
        """Called by MeasurementTab's showEvent/hideEvent to gate the loaded-image overlay."""
        self._tab_active = active
        self._sync_loaded_image_overlay()

    def _sync_loaded_image_overlay(self) -> None:
        preview = get_app_context().camera_preview
        if preview is None:
            return
        preview.overlays.loaded_image_enabled = self._tab_active and self._mode == CaptureMode.LOADED_IMAGE

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        layout.addLayout(self._create_mode_section())
        layout.addWidget(self._create_capture_group())
        layout.addWidget(self._create_measurements_group())

        self._metadata_button = QPushButton("View Image Metadata")
        self._metadata_button.clicked.connect(self._view_metadata)
        layout.addWidget(self._metadata_button)

        layout.addStretch()

    def _create_mode_section(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(4)

        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(0)

        self._live_mode_btn = QPushButton("Live View")
        self._live_mode_btn.setObjectName("LiveViewButton")
        self._live_mode_btn.setCheckable(True)
        self._live_mode_btn.setChecked(True)
        self._live_mode_btn.setMinimumHeight(32)
        self._live_mode_btn.clicked.connect(self._on_live_mode_clicked)

        # No separate "Load Image..." button: clicking this one opens the
        # file picker, except when switching in from live view with an
        # image already loaded, in which case that image is reused as-is.
        # Its color comes entirely from style.py's QPushButton#LoadImageButton
        # rule rather than inline styling here.
        self._loaded_mode_btn = QPushButton("Load Image")
        self._loaded_mode_btn.setObjectName("LoadImageButton")
        self._loaded_mode_btn.setCheckable(True)
        self._loaded_mode_btn.setMinimumHeight(32)
        self._loaded_mode_btn.setToolTip("Click again to choose a different image")
        self._loaded_mode_btn.clicked.connect(self._on_loaded_mode_clicked)

        self._mode_button_group = QButtonGroup(self)
        self._mode_button_group.setExclusive(True)
        self._mode_button_group.addButton(self._live_mode_btn)
        self._mode_button_group.addButton(self._loaded_mode_btn)

        toggle_row.addWidget(self._live_mode_btn, 1)
        toggle_row.addWidget(self._loaded_mode_btn, 1)
        layout.addLayout(toggle_row)

        self._loaded_image_label = QLabel("No image loaded")
        self._loaded_image_label.setWordWrap(True)
        self._loaded_image_label.setStyleSheet("color: #888; font-size: 11px;")
        self._loaded_image_label.setVisible(False)
        layout.addWidget(self._loaded_image_label)

        return layout

    def _create_capture_group(self) -> QGroupBox:
        group = QGroupBox("Photo Capture")
        layout = QVBoxLayout(group)

        folder_layout = QHBoxLayout()
        folder_label = QLabel("Output Folder:")
        folder_label.setMinimumWidth(100)

        self._folder_edit = QLineEdit()
        self._folder_edit.setText(str(self._current_folder))
        self._folder_edit.setPlaceholderText("Select output folder...")

        self._browse_button = QPushButton("Browse...")
        self._browse_button.clicked.connect(self._browse_folder)

        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self._folder_edit, 1)
        folder_layout.addWidget(self._browse_button)

        filename_layout = QHBoxLayout()
        filename_label = QLabel("Filename:")
        filename_label.setMinimumWidth(100)

        self._filename_edit = QLineEdit()
        self._filename_edit.setPlaceholderText("Leave empty for auto-generated name")

        filename_layout.addWidget(filename_label)
        filename_layout.addWidget(self._filename_edit, 1)

        format_layout = QHBoxLayout()
        format_label = QLabel("Image Format:")
        format_label.setMinimumWidth(100)

        self._format_combo = QComboBox()
        self._format_combo.addItems(f.value for f in FileFormat)

        self._open_folder_button = QPushButton("Browse Output")
        self._open_folder_button.clicked.connect(self._open_folder)

        format_layout.addWidget(format_label)
        format_layout.addWidget(self._format_combo)
        format_layout.addWidget(self._open_folder_button)
        format_layout.addStretch()

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(6)

        self._capture_button = QPushButton("Take Photo")
        self._capture_button.setMinimumHeight(36)
        self._capture_button.clicked.connect(self._take_photo)

        self._capture_ui_button = QPushButton("Take Photo with UI")
        self._capture_ui_button.setMinimumHeight(36)
        self._capture_ui_button.clicked.connect(self._take_photo_with_ui)

        buttons_layout.addWidget(self._capture_button)
        buttons_layout.addWidget(self._capture_ui_button)

        layout.addLayout(folder_layout)
        layout.addLayout(filename_layout)
        layout.addLayout(format_layout)
        layout.addLayout(buttons_layout)
        return group

    def _create_measurements_group(self) -> QGroupBox:
        group = QGroupBox("Measurements")
        layout = QHBoxLayout(group)
        layout.setSpacing(6)

        self._capture_measurements_button = QPushButton("Capture Measurements")
        self._capture_measurements_button.setMinimumHeight(36)
        self._capture_measurements_button.setToolTip("Not yet implemented")
        self._capture_measurements_button.clicked.connect(self._capture_measurements)

        self._export_measurements_button = QPushButton("Export Measurements")
        self._export_measurements_button.setMinimumHeight(36)
        self._export_measurements_button.setToolTip("Not yet implemented")
        self._export_measurements_button.clicked.connect(self._export_measurements)

        layout.addWidget(self._capture_measurements_button)
        layout.addWidget(self._export_measurements_button)
        return group

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    @Slot(bool)
    def _on_live_mode_clicked(self, checked: bool) -> None:
        self._set_mode(CaptureMode.LIVE)

    @Slot(bool)
    def _on_loaded_mode_clicked(self, checked: bool) -> None:
        """
        Fires on every click, including while this button is already the
        selected one — that repeat click is how the user swaps in a
        different image, since there's no separate "Load Image..." button.
        """
        switching_in = self._mode != CaptureMode.LOADED_IMAGE

        loaded = switching_in and self._loaded_pixmap is not None
        if not loaded:
            loaded = self._browse_image()

        if loaded:
            self._set_mode(CaptureMode.LOADED_IMAGE)
            self._loaded_mode_btn.setChecked(True)
        elif switching_in:
            # Cancelled or failed on the first selection — fall back to
            # live rather than leaving this selected with nothing loaded.
            self._live_mode_btn.setChecked(True)

    def _set_mode(self, mode: CaptureMode) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        self._loaded_image_label.setVisible(mode == CaptureMode.LOADED_IMAGE)
        self._refresh_capture_availability()
        self._sync_loaded_image_overlay()

    # ------------------------------------------------------------------
    # Loaded image
    # ------------------------------------------------------------------

    def _browse_image(self) -> bool:
        """Prompt for and load an image. Returns whether one was loaded."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            str(self._current_folder),
            "Images (*.png *.jpg *.jpeg *.tif *.tiff)",
        )
        if not filepath:
            return False

        path = Path(filepath)
        image = QImage(str(path))
        if image.isNull():
            warning(f"CaptureControlWidget: failed to load image {path}")
            ctx = get_app_context()
            if ctx.toast:
                ctx.toast.error(f"Could not read image: {path.name}", title="Load Image Failed")
            return False

        self._loaded_pixmap = QPixmap.fromImage(image)
        self._loaded_image_path = path
        self._loaded_image_label.setText(path.name)

        ctx = get_app_context()
        if ctx.camera_preview is not None:
            ctx.camera_preview.overlays.set_loaded_image(self._loaded_pixmap)

        info(f"CaptureControlWidget: loaded image {path}")
        return True

    # ------------------------------------------------------------------
    # Output folder / format
    # ------------------------------------------------------------------

    def _ensure_output_folder(self) -> bool:
        try:
            self._current_folder.mkdir(parents=True, exist_ok=True)
            return True
        except OSError as e:
            error(f"Failed to create output folder: {e}")
            ctx = get_app_context()
            if ctx.toast:
                ctx.toast.error(str(e), title="Folder Creation Failed")
            return False

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            str(self._current_folder),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not folder:
            return

        self._current_folder = Path(folder)
        self._folder_edit.setText(str(self._current_folder))
        self._ensure_output_folder()

        ctx = get_app_context()
        if ctx.toast:
            ctx.toast.success(self._current_folder.name, title="Output Folder Changed")

    def _open_folder(self) -> None:
        ctx = get_app_context()
        toast = ctx.toast
        folder_path = str(self._current_folder.resolve())

        try:
            if sys.platform == "win32":
                subprocess.run(["explorer", folder_path])
            elif sys.platform == "darwin":
                subprocess.run(["open", folder_path])
            else:
                subprocess.run(["xdg-open", folder_path])
        except OSError as e:
            error(f"Failed to open folder: {e}")
            if toast:
                toast.error(str(e), title="Failed to Open Folder")
            return

        if toast:
            toast.info("Opening in file explorer...", title="Opening Folder", duration=10000)

    def _generate_filename(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = self._format_combo.currentText()
        return f"measurement_{timestamp}.{extension}"

    def _get_filepath(self) -> Path:
        extension = self._format_combo.currentText()

        filename = self._filename_edit.text().strip()
        if not filename:
            filename = self._generate_filename()
        else:
            filename = f"{Path(filename).stem}.{extension}"

        return self._current_folder / filename

    # ------------------------------------------------------------------
    # Capture availability
    # ------------------------------------------------------------------

    def _refresh_capture_availability(self) -> None:
        ctx = get_app_context()
        available = ctx.camera is not None and ctx.camera.underlying_camera.is_open

        if available and not self._camera_available:
            self._format_combo.blockSignals(True)
            self._format_combo.setCurrentText(ctx.camera.settings.fformat.value)
            self._format_combo.blockSignals(False)

        self._camera_available = available
        self._capture_button.setEnabled(available and self._mode == CaptureMode.LIVE)

    # ------------------------------------------------------------------
    # Capture actions
    # ------------------------------------------------------------------

    @Slot()
    def _take_photo(self) -> None:
        ctx = get_app_context()
        camera = ctx.camera

        if camera is None or not camera.underlying_camera.is_open:
            warning("Attempted to capture photo but camera is not available")
            if ctx.toast:
                ctx.toast.warning("Please open the camera first", title="Camera Not Open")
            return

        filepath = self._get_filepath()
        self._ensure_output_folder()

        info(f"Capturing measurement still to: {filepath}")
        self._capture_toast_id = ctx.toast.info("Capturing high-resolution image...", title="Capturing Image")

        self._capture_button.setEnabled(False)
        self._capture_pending = False
        self._capture_success = None
        self._capture_filepath = str(filepath)

        def on_capture_complete(success: bool) -> None:
            self._capture_success = success
            self._capture_pending = True

        camera.capture_and_save_still(
            filepath=filepath,
            additional_metadata={"source": "measurement_capture"},
            timeout_ms=5000,
            on_complete=on_capture_complete,
        )
        self._capture_poll_timer.start()

    def _poll_capture_state(self) -> None:
        if not self._capture_pending:
            return

        self._capture_poll_timer.stop()
        self._capture_pending = False
        success = self._capture_success
        filepath = self._capture_filepath

        self._refresh_capture_availability()

        ctx = get_app_context()
        if success:
            ctx.toast.success(f"Saved to: {Path(filepath).name}", title="Image Captured", duration=10000, dismiss_id=self._capture_toast_id)
            self._filename_edit.clear()
        else:
            ctx.toast.error("Unable to capture image from camera", title="Capture Failed", dismiss_id=self._capture_toast_id)
        self._capture_toast_id = None

    @Slot()
    def _take_photo_with_ui(self) -> None:
        """Screenshot the display as shown — active overlays only, no letterbox bars, no toolbar buttons — rather than a raw sensor capture."""
        ctx = get_app_context()
        preview = ctx.camera_preview
        if preview is None:
            warning("Attempted UI capture but no camera preview is registered")
            if ctx.toast:
                ctx.toast.warning("Camera preview not available", title="Capture Failed")
            return

        pixmap = preview.grab_display()
        if pixmap.isNull():
            warning("Attempted UI capture but there was nothing to render yet")
            if ctx.toast:
                ctx.toast.warning("Nothing to capture", title="Capture Failed")
            return

        filepath = self._get_filepath()
        if not self._ensure_output_folder():
            return

        saved = pixmap.save(str(filepath))
        if saved:
            info(f"Captured UI screenshot to: {filepath}")
            if ctx.toast:
                ctx.toast.success(f"Saved to: {filepath.name}", title="Image Captured")
            self._filename_edit.clear()
        else:
            error(f"Failed to save UI screenshot to: {filepath}")
            if ctx.toast:
                ctx.toast.error("Unable to save captured image", title="Capture Failed")

    @Slot()
    def _capture_measurements(self) -> None:
        ctx = get_app_context()
        if ctx.toast:
            ctx.toast.info("Measurement capture is not yet implemented", title="Coming Soon")

    @Slot()
    def _export_measurements(self) -> None:
        ctx = get_app_context()
        if ctx.toast:
            ctx.toast.info("Measurement export is not yet implemented", title="Coming Soon")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @Slot()
    def _view_metadata(self) -> None:
        if self._mode == CaptureMode.LOADED_IMAGE:
            metadata = self._build_loaded_image_metadata()
            title = "Loaded Image Metadata"
        else:
            metadata = self._build_live_metadata()
            title = "Projected Capture Metadata"

        dialog = MetadataDialog(title, metadata, self)
        dialog.exec()

    def _build_loaded_image_metadata(self) -> dict[str, str]:
        if self._loaded_pixmap is None or self._loaded_image_path is None:
            return {"Status": "No image loaded"}

        path = self._loaded_image_path
        try:
            return self._read_image_file_metadata(path)
        except (OSError, ValueError) as e:
            warning(f"CaptureControlWidget: failed to read metadata for {path} — {e}")
            return {
                "File": path.name,
                "Folder": str(path.parent),
                "Status": "Could not read embedded metadata",
            }

    def _read_image_file_metadata(self, path: Path) -> dict[str, str]:
        """
        Read the image's own embedded metadata directly, rather than a
        curated preset field list. Covers both the top-level EXIF IFD
        (Software, Model, ...) and the Exif sub-IFD nested underneath it,
        where the standard camera fields — ExposureTime, ISOSpeedRatings,
        and UserComment (this app's own full capture-metadata JSON,
        see base_camera.py's save_image) — actually live; getexif() alone
        only surfaces the top-level IFD and silently misses those.
        """
        metadata: dict[str, str] = {}

        with PILImage.open(path) as img:
            metadata["Format"] = img.format or path.suffix.lstrip(".").upper()
            metadata["Dimensions"] = f"{img.width} x {img.height}"
            metadata["Mode"] = img.mode

            exif = img.getexif()
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if tag_name in ("ExifOffset", "GPSInfo"):
                    continue  # just pointers to the sub-IFDs expanded below
                self._add_metadata_field(metadata, tag_name, value)

            for ifd_id in (ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo):
                for tag_id, value in exif.get_ifd(ifd_id).items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    self._add_metadata_field(metadata, tag_name, value)

            for key, value in getattr(img, "text", {}).items():
                self._add_metadata_field(metadata, key, value)

        stat = path.stat()
        metadata["File Size"] = f"{stat.st_size / 1024:.1f} KB"
        metadata["File Modified"] = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        return metadata

    def _add_metadata_field(self, metadata: dict[str, str], name: str, value: object) -> None:
        """
        Add one raw EXIF/PNG-text field. UserComment / Metadata carry this
        app's own capture metadata as a JSON blob (see base_camera.py) —
        flatten those into individual rows instead of one unreadable string.
        """
        text = self._decode_metadata_bytes(value) if isinstance(value, bytes) else str(value)

        if name in ("UserComment", "Metadata"):
            parsed = self._try_parse_json(text)
            if isinstance(parsed, dict):
                for flat_key, flat_value in self._flatten(parsed).items():
                    metadata[flat_key] = str(flat_value)
                return

        metadata[name] = text

    @staticmethod
    def _decode_metadata_bytes(value: bytes) -> str:
        for encoding in ("utf-16", "utf-8", "ascii"):
            decoded = value.decode(encoding, errors="replace")
            if "\ufffd" not in decoded:
                return decoded.strip("\x00")
        return value.decode("utf-8", errors="replace").strip("\x00")

    @staticmethod
    def _try_parse_json(text: str) -> object | None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _flatten(data: dict, prefix: str = "") -> dict[str, object]:
        flat: dict[str, object] = {}
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                flat.update(CaptureControlWidget._flatten(value, full_key))
            else:
                flat[full_key] = value
        return flat

    def _build_live_metadata(self) -> dict[str, str]:
        ctx = get_app_context()
        camera = ctx.camera
        camera_manager = ctx.camera_manager

        metadata: dict[str, str] = {"Status": "Preview — reflects a capture taken right now"}

        if camera is not None and camera.underlying_camera.is_open:
            metadata["Format"] = camera.settings.fformat.value
        else:
            metadata["Format"] = self._format_combo.currentText()

        if camera_manager is not None and camera_manager.has_active_camera:
            width, height = camera_manager.frame_dimensions
            metadata["Resolution"] = f"{width} x {height}"

        metadata["Output Folder"] = str(self._current_folder)
        metadata["Filename"] = self._get_filepath().name
        metadata["Captured At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return metadata