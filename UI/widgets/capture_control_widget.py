from __future__ import annotations

import subprocess
import sys
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path

from PySide6.QtCore import Signal, Slot, QTimer, Qt, QMimeData, QEvent, QObject
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent, QResizeEvent, QMoveEvent, QShowEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from common.app_context import AppContext, get_app_context
from common.logger import info, error, warning
from common.read_metadata import extract_dpi, read_metadata
from common.setting_types import FileFormat
from UI.widgets.measurements.units import MeasurementUnit, dpi_from_measurement
from UI.widgets.preview_overlay.large_image_source import LargeImageSource


_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})

# Pyramidal/tiled TIFF writers (libvips and others) commonly stamp
# XResolution/YResolution as exactly 1 when no real value was supplied,
# rather than omitting the tag — a real photo or scan is never actually
# ~1 pixel per inch, so treat that (and anything <=) as no DPI at all,
# not a real value. See _load_image_routine.
_MIN_PLAUSIBLE_METADATA_DPI = 1.0


class CaptureMode(Enum):
    LIVE = "live"
    LOADED_IMAGE = "loaded_image"


class CaptureControlWidget(QWidget):
    """
    Measurement tab's capture widget: live camera vs. loaded-image source
    and output/format controls.

    Kept single-column throughout (one control per row) except where
    buttons are short enough to sit side by side, so it stays within
    RIGHT_SIDEBAR_WIDTH without triggering the sidebar's horizontal
    scrollbar, matching CameraControlsWidget.

    The loaded-image overlay this widget drives lives on the shared
    CameraPreview, so it must only be enabled while the measurement tab is
    actually the one showing that preview — see ``set_tab_active``, called
    by MeasurementTab's showEvent/hideEvent.
    """

    # Emitted with (dpi, is_live) for whichever mode is currently active,
    # any time the DPI value changes — mode switch, a newly loaded
    # image, a live-view settings/calibration change, or the user
    # entering one by hand via "Calibrate DPI" -> Enter DPI (loaded
    # image only — see MeasurementsWidget.set_dpi_display).
    dpi_changed = Signal(object, bool)

    # Whether a manual-calibration reference line is currently placed on
    # the preview, so MeasurementsWidget can enable/disable its "Finish
    # Calibration" button.
    calibration_line_ready = Signal(bool)

    # A newly loaded image had no DPI in its metadata — tells
    # MeasurementsWidget to expand its DPI Calibration panel so the fix
    # is right there rather than requiring the user to hunt for it.
    loaded_dpi_missing = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._default_folder = Path("./output/measurements")
        self._current_folder = self._default_folder
        self._ensure_output_folder()

        self._mode = CaptureMode.LIVE
        self._tab_active = False
        self._camera_available = False

        self._loaded_source: LargeImageSource | None = None
        self._loaded_image_path: Path | None = None

        # DPI, tracked separately per source the same way measurements
        # are — a loaded image keeps the DPI read from its own metadata
        # even while the live feed is showing, and vice versa. Live DPI
        # is always mirrored from machine_vision.settings.dpi — the only
        # way to change it is through calibration (automatic, via the
        # Calibration tab, or manual, via submit_calibration_dpi below),
        # both of which write to settings directly.
        self._live_dpi: float | None = None
        self._loaded_dpi: float | None = None
        self._live_reference_dims: tuple[int, int] | None = None
        self._calibration_line_ready = False

        # Capture state — mutated only by the on_complete callback on the
        # camera thread; read only by _poll_capture_state on the main thread.
        self._capture_pending = False
        self._capture_success: bool | None = None
        self._capture_filepath = ""
        self._capture_toast_id: int | None = None

        # Image-load state — mutated only by _load_image_routine on a
        # background thread; read only by _poll_load_state on the main
        # thread. Opening a large image can take a while (format
        # detection plus building the initial preview), so this runs off
        # the UI thread the same way capture does.
        self._load_pending = False
        self._load_success: bool | None = None
        self._load_source: LargeImageSource | None = None
        self._load_path: Path | None = None
        self._load_dpi: float | None = None
        self._load_switching_in = False
        self._load_toast_id: int | None = None

        self._drag_active = False
        self._window_drag_filter_installed = False
        self._overlay: QWidget | None = None
        self._overlay_frame: QWidget | None = None
        self._overlay_label: QLabel | None = None

        self.setAcceptDrops(True)
        self._setup_ui()

        self._capture_poll_timer = QTimer(self)
        self._capture_poll_timer.setInterval(100)
        self._capture_poll_timer.timeout.connect(self._poll_capture_state)

        self._load_poll_timer = QTimer(self)
        self._load_poll_timer.setInterval(100)
        self._load_poll_timer.timeout.connect(self._poll_load_state)

        self._camera_poll_timer = QTimer(self)
        self._camera_poll_timer.setInterval(500)
        self._camera_poll_timer.timeout.connect(self._refresh_capture_availability)
        self._camera_poll_timer.timeout.connect(self._refresh_live_dpi)
        self._camera_poll_timer.timeout.connect(self._refresh_calibration_line_ready)
        self._camera_poll_timer.start()

        self._refresh_capture_availability()

    # ------------------------------------------------------------------
    # Tab-visibility wiring
    # ------------------------------------------------------------------

    def set_tab_active(self, active: bool) -> None:
        """Called by MeasurementTab's showEvent/hideEvent to gate the loaded-image overlay."""
        self._tab_active = active
        self._sync_loaded_image_overlay()
        if active and self._mode == CaptureMode.LIVE:
            self._ensure_live_dpi()

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

        layout.addStretch()

    def _ensure_overlay(self) -> None:
        """
        Create the overlay the first time we're shown with a real parent.

        Parenting to self.parent() rather than self means Qt won't clip the
        overlay to this widget's own bounds, so _reposition_overlay can
        expand it to cover the layout margins the parent places around us.
        """
        if self._overlay is not None:
            return
        overlay_parent = self.parent()
        if overlay_parent is None:
            return

        self._overlay = QWidget(overlay_parent)
        self._overlay.setObjectName("CaptureImageOverlay")
        self._overlay.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._overlay.hide()

        overlay_layout = QVBoxLayout(self._overlay)
        overlay_layout.setContentsMargins(14, 14, 14, 14)

        # Inset frame keeps the border a little in from the widget's true
        # edge rather than flush against it. Expanding policy makes it fill
        # that inset area so the border traces the widget's edge rather
        # than shrink-wrapping to the label.
        self._overlay_frame = QWidget(self._overlay)
        self._overlay_frame.setObjectName("CaptureImageOverlayFrame")
        self._overlay_frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._overlay_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        frame_layout = QVBoxLayout(self._overlay_frame)
        frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._overlay_label = QLabel(self._overlay_frame)
        self._overlay_label.setObjectName("CaptureImageOverlayLabel")
        self._overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay_label.setWordWrap(True)
        frame_layout.addWidget(self._overlay_label)

        overlay_layout.addWidget(self._overlay_frame)

    def _reposition_overlay(self) -> None:
        """Position the overlay in its parent's coordinate space so it covers this widget plus the layout margins the parent places around it."""
        if self._overlay is None:
            return
        overlay_parent = self._overlay.parent()
        if overlay_parent is None:
            return

        origin = self.mapTo(overlay_parent, self.rect().topLeft())
        x, y = origin.x(), origin.y()
        w, h = self.width(), self.height()

        parent = self.parent()
        if parent is not None:
            layout = parent.layout()
            if layout is not None:
                left, top, right, bottom = layout.getContentsMargins()
                x -= left
                y -= top
                w += left + right
                h += top + bottom

        self._overlay.setGeometry(x, y, w, h)
        self._overlay.raise_()

    def _show_overlay(self, text: str, *, drag_hint: bool = False) -> None:
        if self._overlay is None or self._overlay_frame is None or self._overlay_label is None:
            return
        self._overlay_label.setText(text)
        for w in (self._overlay_frame, self._overlay_label):
            w.setProperty("dragHint", drag_hint)
            w.style().unpolish(w)
            w.style().polish(w)
        self._reposition_overlay()
        self._overlay.show()

    def _hide_overlay(self) -> None:
        if self._overlay is None:
            return
        self._overlay.hide()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._ensure_overlay()
        self._reposition_overlay()
        self._install_window_drag_filter()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._reposition_overlay()

    def moveEvent(self, event: QMoveEvent) -> None:
        super().moveEvent(event)
        self._reposition_overlay()

    # ------------------------------------------------------------------
    # Drag and drop
    # ------------------------------------------------------------------

    def _install_window_drag_filter(self) -> None:
        """
        Watch the top-level window for drag events, not just this widget.

        Qt only delivers drag events to a widget under the cursor that
        accepts drops (or the nearest ancestor that does), so without this
        the drag hint would only appear once the cursor is already over us.
        Watching the window catches the drag as soon as it enters the app.
        """
        if self._window_drag_filter_installed:
            return
        window = self.window()
        if window is None:
            return
        window.setAcceptDrops(True)
        window.installEventFilter(self)
        self._window_drag_filter_installed = True

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self.window():
            event_type = event.type()
            if event_type in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
                if not self._load_pending and self._image_path_from_mime(event.mimeData()) is not None:
                    event.acceptProposedAction()
                    if not self._drag_active:
                        self._drag_active = True
                        self._show_overlay("Drop image here", drag_hint=True)
            elif event_type == QEvent.Type.DragLeave:
                if self._drag_active and not self._load_pending:
                    self._drag_active = False
                    self._hide_overlay()
            elif event_type == QEvent.Type.Drop:
                if self._drag_active:
                    self._drag_active = False
                    if not self._load_pending:
                        self._hide_overlay()
        return super().eventFilter(obj, event)

    def _image_path_from_mime(self, mime: QMimeData) -> Path | None:
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.suffix.lower() in _IMAGE_SUFFIXES:
                return path
        return None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._load_pending:
            event.ignore()
            return
        if self._image_path_from_mime(event.mimeData()) is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self._drag_active = True
        self._show_overlay("Drop image here", drag_hint=True)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if self._load_pending or self._image_path_from_mime(event.mimeData()) is None:
            event.ignore()
            return
        event.acceptProposedAction()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self._drag_active = False
        if not self._load_pending:
            self._hide_overlay()

    def dropEvent(self, event: QDropEvent) -> None:
        self._drag_active = False
        path = self._image_path_from_mime(event.mimeData())
        if path is None or self._load_pending:
            event.ignore()
            return
        event.acceptProposedAction()
        self._start_loading(path, switching_in=self._mode != CaptureMode.LOADED_IMAGE)

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

        self._open_folder_button = QPushButton("Open Folder")
        self._open_folder_button.clicked.connect(self._open_folder)

        format_layout.addWidget(format_label)
        format_layout.addWidget(self._format_combo)
        format_layout.addWidget(self._open_folder_button)
        format_layout.addStretch()

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(6)

        self._capture_button = QPushButton("Take Photo")
        self._capture_button.setMinimumHeight(36)
        self._capture_button.clicked.connect(self._on_capture_clicked)
        buttons_layout.addWidget(self._capture_button)

        # What "Take Photo" actually does — see _on_capture_clicked. Only
        # "Full-res image (no measurements)" keeps doing a real camera
        # sensor capture (Live mode only, unchanged from before); every
        # other kind instead renders+saves whatever is currently
        # displayed (works in Live or Loaded-Image mode) via
        # _export_current_frame, the same operation the old standalone
        # "Export..." button performed. Sole widget in its own row (no
        # label competing for space) so it's stretched to the group's
        # full resolved width the same way _capture_button is — wide
        # enough that every option's text is actually legible.
        capture_kind_layout = QHBoxLayout()
        self._export_kind_combo = QComboBox()
        self._export_kind_combo.addItem("Preview image (measurements baked in)", "preview")
        self._export_kind_combo.addItem("Preview image (no measurements)", "preview_no_measurements")
        self._export_kind_combo.addItem("Full-res image (measurements baked in)", "full_res")
        self._export_kind_combo.addItem("Full-res image (no measurements)", "full_res_no_measurements")
        self._export_kind_combo.addItem("Full-res image + measurements file", "full_res_sidecar")
        self._export_kind_combo.currentIndexChanged.connect(self._refresh_capture_availability)
        capture_kind_layout.addWidget(self._export_kind_combo)

        layout.addLayout(folder_layout)
        layout.addLayout(filename_layout)
        layout.addLayout(format_layout)
        layout.addLayout(buttons_layout)
        layout.addLayout(capture_kind_layout)
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

        Loading itself happens on a background thread (see _browse_image),
        so the mode switch is finished asynchronously by _poll_load_state
        once the image has actually decoded.
        """
        if self._load_pending:
            return

        switching_in = self._mode != CaptureMode.LOADED_IMAGE

        if switching_in and self._loaded_source is not None:
            self._set_mode(CaptureMode.LOADED_IMAGE)
            self._loaded_mode_btn.setChecked(True)
            return

        started = self._browse_image(switching_in)
        if not started and switching_in:
            # Cancelled on the first selection — fall back to live rather
            # than leaving this selected with nothing loaded.
            self._live_mode_btn.setChecked(True)

    def _set_mode(self, mode: CaptureMode) -> None:
        if mode == self._mode:
            return
        self.cancel_calibration()
        self._discard_draft_measurement()
        self._mode = mode
        self._loaded_image_label.setVisible(mode == CaptureMode.LOADED_IMAGE)
        self._refresh_capture_availability()
        self._sync_loaded_image_overlay()
        self._update_dpi_display()
        if mode == CaptureMode.LIVE:
            self._ensure_live_dpi()

    def _discard_draft_measurement(self) -> None:
        preview = get_app_context().camera_preview
        if preview is not None:
            preview.overlays.measurement.discard_draft()

    # ------------------------------------------------------------------
    # Loaded image
    # ------------------------------------------------------------------

    def _browse_image(self, switching_in: bool) -> bool:
        """Prompt for an image and start decoding it in the background. Returns whether a load was started."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            str(self._current_folder),
            "Images (*.png *.jpg *.jpeg *.tif *.tiff)",
        )
        if not filepath:
            return False

        self._start_loading(Path(filepath), switching_in)
        return True

    def _start_loading(self, path: Path, switching_in: bool) -> None:
        self._load_switching_in = switching_in
        self._load_pending = False
        self._load_success = None

        self._live_mode_btn.setEnabled(False)
        self._loaded_mode_btn.setEnabled(False)
        self._show_overlay(f"Loading {path.name}...")

        ctx = get_app_context()
        if ctx.toast:
            self._load_toast_id = ctx.toast.info(f"Loading {path.name}...", title="Loading Image")

        threading.Thread(target=self._load_image_routine, args=(path,), daemon=True).start()
        self._load_poll_timer.start()

    def _load_image_routine(self, path: Path) -> None:
        """
        Runs on a background thread — mutates only plain data, never
        touches widgets, signals, or app-context notifications directly.
        _poll_load_state picks the result up on the main thread.

        This only opens the file and builds LargeImageSource's small
        preview — it never decodes the whole image, so this stays fast
        even for a file far too large to hold fully in memory. Reading
        the file's own metadata for a DPI value is cheap enough to do
        here too, alongside the other file I/O.
        """
        source = LargeImageSource(str(path))
        success = source.open()
        if not success:
            source.close()

        metadata = read_metadata(path) if success else None
        dpi = extract_dpi(metadata) if metadata is not None else None
        if dpi is not None and dpi <= _MIN_PLAUSIBLE_METADATA_DPI:
            dpi = None

        self._load_source = source if success else None
        self._load_success = success
        self._load_path = path
        self._load_dpi = dpi
        self._load_pending = True

    def _poll_load_state(self) -> None:
        if not self._load_pending:
            return

        self._load_poll_timer.stop()
        self._load_pending = False
        success = self._load_success
        source = self._load_source
        path = self._load_path
        dpi = self._load_dpi
        self._load_source = None

        self._hide_overlay()
        self._live_mode_btn.setEnabled(True)
        self._loaded_mode_btn.setEnabled(True)

        ctx = get_app_context()

        if not success:
            warning(f"CaptureControlWidget: failed to load image {path}")
            if ctx.toast:
                ctx.toast.error(f"Could not read image: {path.name}", title="Load Image Failed", dismiss_id=self._load_toast_id)
            self._load_toast_id = None
            if self._load_switching_in:
                self._live_mode_btn.setChecked(True)
                self._loaded_image_label.setVisible(False)
            return

        self._loaded_source = source
        self._loaded_image_path = path
        self._loaded_image_label.setText(path.name)

        no_dpi_found = dpi is None
        if no_dpi_found:
            QMessageBox.information(
                self,
                "No DPI Found",
                "This image has no DPI in its metadata. Set it manually using the DPI Calibration panel.",
            )

        if ctx.camera_preview is not None:
            ctx.camera_preview.overlays.set_loaded_image(self._loaded_source)

        info(f"CaptureControlWidget: loaded image {path}")
        if ctx.toast:
            ctx.toast.success(path.name, title="Image Loaded", dismiss_id=self._load_toast_id)
        self._load_toast_id = None

        self._set_mode(CaptureMode.LOADED_IMAGE)
        self._loaded_mode_btn.setChecked(True)
        self._apply_dpi(dpi)

        # Emitted last, after the mode switch and its own dpi_changed —
        # otherwise the mode-change-triggered panel auto-hide (see
        # MeasurementsWidget.set_dpi_display) would immediately undo this.
        if no_dpi_found:
            self.loaded_dpi_missing.emit()

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
        extension = self._format_combo.currentText()
        return f"{self.default_export_stem()}.{extension}"

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

        # Only the raw-sensor-capture kind needs an actual open camera in
        # Live mode — every other kind renders+saves whatever is already
        # displayed (see _on_capture_clicked) and works regardless.
        if self._export_kind_combo.currentData() == "full_res_no_measurements":
            self._capture_button.setEnabled(available and self._mode == CaptureMode.LIVE)
        else:
            self._capture_button.setEnabled(True)

    # ------------------------------------------------------------------
    # DPI
    #
    # Live view mirrors machine_vision.settings.dpi — the only way to
    # change it is calibration, automatic (the Calibration tab) or
    # manual (submit_calibration_dpi below), both of which write to
    # settings directly, so there is exactly one source of truth and
    # nothing here can fight it on the next poll tick.
    #
    # A loaded image instead carries its own DPI in its file metadata —
    # see _load_image_routine — with a manual entry dialog as fallback.
    #
    # Either way, once DPI is available placement produces a labeled
    # real-world length; without one the tab still works as a plain
    # image/live viewer, just with unlabeled shapes — see
    # MeasurementOverlay._draw_measurement_label.
    # ------------------------------------------------------------------

    def _refresh_live_dpi(self) -> None:
        """Silently pick up settings.dpi / still-resolution changes (e.g. a calibration run on another tab)."""
        if self._mode != CaptureMode.LIVE:
            return
        ctx = get_app_context()
        dpi = ctx.machine_vision.settings.dpi
        if dpi != self._live_dpi:
            self._apply_dpi(dpi)
        self._refresh_live_reference_dims(ctx)

    def _refresh_live_reference_dims(self, ctx: AppContext) -> None:
        """
        Push the camera's still-capture resolution to the overlay — DPI
        is calibrated against that, not the (lower-resolution) live
        preview stream, so real-world length math must scale against it
        too. See MeasurementOverlay._draw_measurement_label.
        """
        camera = ctx.camera
        dims = None
        if camera is not None and camera.underlying_camera.is_open:
            _, width, height = camera.settings.get_current_still_resolution()
            dims = (width, height)
        if dims != self._live_reference_dims:
            self._live_reference_dims = dims
            if ctx.camera_preview is not None:
                ctx.camera_preview.overlays.measurement.set_live_reference_dims(dims)

    def _ensure_live_dpi(self) -> None:
        """Resolve live DPI from settings. Never prompts — calibration is initiated explicitly via MeasurementsWidget's "Calibrate DPI" panel."""
        if not self._tab_active:
            return
        ctx = get_app_context()
        self._refresh_live_reference_dims(ctx)
        self._apply_dpi(ctx.machine_vision.settings.dpi)

    def submit_dpi_value(self, value: float) -> None:
        """Directly apply a DPI value entered in MeasurementsWidget's own panel — live writes it to machine_vision.settings like calibration does; loaded just updates this widget's own _loaded_dpi."""
        if value <= 0:
            ctx = get_app_context()
            if ctx.toast:
                ctx.toast.error("Enter a positive DPI", title="Invalid DPI")
            return
        ctx = get_app_context()
        if self._mode == CaptureMode.LIVE:
            ctx.machine_vision.settings.dpi = value
            ctx.machine_vision.save_settings()
        self._apply_dpi(value)

    def _apply_dpi(self, dpi: float | None) -> None:
        """Store *dpi* for whichever source is currently active and push it to the preview overlay. Non-positive values are treated as unset."""
        if dpi is not None and dpi <= 0:
            dpi = None
        ctx = get_app_context()
        if self._mode == CaptureMode.LIVE:
            self._live_dpi = dpi
            if ctx.camera_preview is not None:
                ctx.camera_preview.overlays.measurement.set_live_dpi(dpi)
        else:
            self._loaded_dpi = dpi
            if ctx.camera_preview is not None:
                ctx.camera_preview.overlays.measurement.set_loaded_dpi(dpi)
        self._update_dpi_display()

    def _update_dpi_display(self) -> None:
        dpi = self._live_dpi if self._mode == CaptureMode.LIVE else self._loaded_dpi
        self.dpi_changed.emit(dpi, self._mode == CaptureMode.LIVE)

    # ------------------------------------------------------------------
    # DPI calibration — manual works for either source (live or loaded
    # image). There's no automatic option surfaced here at all; that
    # lives entirely on the Calibration tab (see dpi_calibration.py),
    # which writes machine_vision.settings.dpi itself.
    #
    # Manual places a single reference line on the preview (see
    # MeasurementOverlay.start_calibration_placement) and derives DPI
    # from its pixel length plus a user-entered real-world length. For
    # live view that's a global machine_vision.settings write, same as
    # the Calibration tab's own routine; for a loaded image it's
    # per-image, so it only ever updates this widget's own _loaded_dpi.
    # ------------------------------------------------------------------

    def request_manual_calibration(self) -> None:
        preview = get_app_context().camera_preview
        if preview is not None:
            preview.overlays.measurement.start_calibration_placement()

    def cancel_calibration(self) -> None:
        preview = get_app_context().camera_preview
        if preview is not None:
            preview.overlays.measurement.cancel_calibration_placement()

    def submit_calibration_dpi(self, value: float, unit: MeasurementUnit) -> None:
        ctx = get_app_context()
        preview = ctx.camera_preview
        if preview is None:
            return

        pixel_length = preview.overlays.measurement.calibration_line_length_px()
        if pixel_length is None:
            if ctx.toast:
                ctx.toast.warning("Place the calibration line first", title="No Calibration Line")
            return

        dpi = dpi_from_measurement(pixel_length, value, unit)
        if dpi is None:
            if ctx.toast:
                ctx.toast.error("Enter a positive measurement", title="Invalid Calibration")
            return

        if self._mode == CaptureMode.LIVE:
            ctx.machine_vision.settings.dpi = dpi
            ctx.machine_vision.save_settings()

        preview.overlays.measurement.clear_calibration_line()
        self._apply_dpi(dpi)
        if ctx.toast:
            ctx.toast.success(f"DPI set to {dpi:.2f}", title="Calibration Complete")

    def _refresh_calibration_line_ready(self) -> None:
        preview = get_app_context().camera_preview
        ready = preview is not None and preview.overlays.measurement.has_calibration_line
        if ready != self._calibration_line_ready:
            self._calibration_line_ready = ready
            self.calibration_line_ready.emit(ready)

    # ------------------------------------------------------------------
    # Capture actions — "Take Photo" is now a single button whose actual
    # behavior is chosen by _export_kind_combo (see _on_capture_clicked):
    # a real camera sensor capture for "Full-res image (no measurements)"
    # (the one kind that still needs one — Live mode only), or a
    # render-and-save of whatever is currently displayed for every other
    # kind (works in Live or Loaded-Image mode) — see CameraPreview's own
    # export_plain_image/export_measurement_image/export_preview_image/
    # export_preview_measurement_image for the actual rendering, and
    # MeasurementOverlayController/measurement_io.py for the measurements
    # JSON sidecar. Neither path prompts with a file dialog — both save
    # straight to _get_filepath(), same as a raw capture always has.
    # ------------------------------------------------------------------

    def default_export_stem(self) -> str:
        """Default filename stem for an export — the loaded image's own name while one is active, else a fresh timestamped stem matching _generate_filename's own pattern. Public: also used by MeasurementTab's relocated Export/Import Measurements defaults."""
        if self._mode == CaptureMode.LOADED_IMAGE and self._loaded_image_path is not None:
            return self._loaded_image_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"measurement_{timestamp}"

    @property
    def current_folder(self) -> Path:
        """Current output folder — read by MeasurementTab for its relocated Export/Import Measurements dialogs' default directory."""
        return self._current_folder

    @Slot()
    def _on_capture_clicked(self) -> None:
        if self._export_kind_combo.currentData() == "full_res_no_measurements":
            self._capture_raw_photo()
        else:
            self._export_current_frame(self._export_kind_combo.currentData())

    def _capture_raw_photo(self) -> None:
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
            additional_metadata={
                "DPI": ctx.machine_vision.settings.dpi,
                "source": "measurement_capture"
                },
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

    def _export_current_frame(self, kind: str) -> None:
        ctx = get_app_context()
        preview = ctx.camera_preview
        if preview is None:
            return

        if kind == "preview":
            image = preview.export_preview_measurement_image()
        elif kind == "preview_no_measurements":
            image = preview.export_preview_image()
        elif kind == "full_res":
            image = preview.export_measurement_image()
        else:  # "full_res_sidecar"
            image = preview.export_plain_image()
        if image is None:
            if ctx.toast:
                ctx.toast.warning("Nothing to capture yet", title="Capture Failed")
            return

        filepath = self._get_filepath()
        if not self._ensure_output_folder():
            return

        if not image.save(str(filepath)):
            error(f"Failed to save exported image to: {filepath}")
            if ctx.toast:
                ctx.toast.error("Unable to save exported image", title="Capture Failed")
            return
        info(f"Exported image to: {filepath}")
        if ctx.toast:
            ctx.toast.success(f"Saved to: {filepath.name}", title="Image Captured")
        self._filename_edit.clear()

        if kind == "full_res_sidecar":
            sidecar = str(filepath.with_suffix(".json"))
            preview.overlays.measurement.export_measurements_to_file(sidecar)
            if ctx.toast:
                ctx.toast.success(Path(sidecar).name, title="Measurements Exported")