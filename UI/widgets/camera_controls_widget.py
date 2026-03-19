from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLineEdit, QLabel, QFileDialog, QMessageBox, QComboBox
)
from PySide6.QtCore import Slot, Signal, QTimer
from PySide6.QtGui import QPainter, QPen, QColor, QPainterPath, QBrush
from common.logger import info, error, warning, debug
from common.app_context import get_app_context


class HistogramWidget(QWidget):
    """
    Paints a live RGB or mono histogram sourced from the camera.

    Displays preview-frame data continuously while enabled, and overlays the
    most recent still-capture histogram as a lighter trace after each capture.
    All rendering is done with QPainter — no third-party plotting libraries.
    """

    # (fill colour, line colour) per RGB channel
    _CHANNEL_COLORS: list[tuple[QColor, QColor]] = [
        (QColor(220, 60,  60,  50),  QColor(220, 60,  60,  210)),  # R
        (QColor(60,  180, 60,  50),  QColor(60,  180, 60,  210)),  # G
        (QColor(60,  120, 220, 50),  QColor(60,  120, 220, 210)),  # B
    ]
    _MONO_COLORS: tuple[QColor, QColor] = (
        QColor(80, 80, 80, 50), QColor(80, 80, 80, 210)
    )
    _STILL_ALPHA_FILL = 15
    _STILL_ALPHA_LINE = 100
    _BACKGROUND  = QColor(255, 255, 255)
    _GRID_COLOR  = QColor(220, 220, 220)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._preview: np.ndarray | None = None  # shape (channels, bins)
        self._still:   np.ndarray | None = None
        # Which channels are visible; populated on first data arrival
        self._channel_visible: list[bool] = [True, True, True]
        self.setMinimumHeight(140)

    def set_channel_visible(self, channel: int, visible: bool) -> None:
        """Show or hide a single channel (0=R, 1=G, 2=B)."""
        if 0 <= channel < len(self._channel_visible):
            self._channel_visible[channel] = visible
            self.update()

    def update_preview(self, histogram: np.ndarray | None) -> None:
        """Set the current preview histogram and schedule a repaint."""
        self._preview = histogram
        self.update()

    def update_still(self, histogram: np.ndarray | None) -> None:
        """Set the still-capture histogram overlay and schedule a repaint."""
        self._still = histogram
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        pad     = 6
        label_h = 16

        # White background
        painter.fillRect(0, 0, w, h, self._BACKGROUND)

        # Light vertical grid lines (stop above the label area)
        painter.setPen(QPen(self._GRID_COLOR, 1))
        for i in range(1, 4):
            x = pad + (w - 2 * pad) * i // 4
            painter.drawLine(x, pad, x, h - pad - label_h)

        if self._preview is None:
            painter.end()
            return

        self._paint_histogram(painter, self._preview, w, h, pad, still=False)
        if self._still is not None:
            self._paint_histogram(painter, self._still, w, h, pad, still=True)

        painter.end()

    @staticmethod
    def _bucket(values: np.ndarray, target_pixels: int) -> np.ndarray:
        """
        Reduce bin count to target_pixels by summing within each bucket.
        Summing preserves the histogram's total area and avoids the
        alternating-peak artefact that max-pooling produces on sparse data.
        """
        n = len(values)
        if n <= target_pixels:
            return values
        trim = (n // target_pixels) * target_pixels
        return values[:trim].reshape(target_pixels, n // target_pixels).sum(axis=1)

    def _paint_histogram(
        self,
        painter: QPainter,
        data: np.ndarray,
        w: int,
        h: int,
        pad: int,
        still: bool,
    ) -> None:
        channels, bins = data.shape
        label_h = 16           # pixels reserved at the bottom for axis labels
        draw_w  = w - 2 * pad
        draw_h  = h - 2 * pad - label_h

        for ch in range(channels):
            if not self._channel_visible[ch % len(self._channel_visible)]:
                continue

            # Bucket bins down to the drawable width
            values = self._bucket(data[ch], draw_w)
            n      = len(values)
            peak   = float(values.max())
            if peak < 1e-9:
                continue
            # Build a step-function path: each bucket is a rectangle whose top
            # edge is at the bucket height.  We walk left-to-right along the
            # top of the histogram so the filled shape has no internal spikes.
            path = QPainterPath()
            baseline_y = h - pad - label_h
            path.moveTo(pad, baseline_y)          # bottom-left

            info(str(values) + ", "+  str(peak) + ", " + str(n))
            prev_y = baseline_y
            for i in range(n):
                x_left  = pad + draw_w * i       / n
                x_right = pad + draw_w * (i + 1) / n
                y = baseline_y - draw_h * float(values[i]) / peak
                path.lineTo(x_left,  prev_y)  # vertical drop/rise to this bucket's level
                path.lineTo(x_left,  y)
                path.lineTo(x_right, y)
                prev_y = y

            path.lineTo(pad + draw_w, baseline_y)  # bottom-right
            path.closeSubpath()

            if channels == 1:
                fill_color, line_color = self._MONO_COLORS
            else:
                fill_color, line_color = self._CHANNEL_COLORS[ch % len(self._CHANNEL_COLORS)]

            if still:
                fill_color = QColor(fill_color.red(), fill_color.green(),
                                    fill_color.blue(), self._STILL_ALPHA_FILL)
                line_color = QColor(line_color.red(), line_color.green(),
                                    line_color.blue(), self._STILL_ALPHA_LINE)

            painter.fillPath(path, QBrush(fill_color))
            painter.setPen(QPen(line_color, 1))
            painter.drawPath(path)

        # Axis tick marks — drawn once, outside the per-channel loop
        if not still:
            self._paint_axis(painter, bins, w, h, pad, draw_w, label_h)

    def _paint_axis(
        self,
        painter: QPainter,
        bins: int,
        w: int,
        h: int,
        pad: int,
        draw_w: int,
        label_h: int,
    ) -> None:
        """Draw tick marks and labels along the bottom of the histogram."""
        tick_color = QColor(120, 120, 120)
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        painter.setPen(QPen(tick_color, 1))

        # Choose sensible tick positions based on bin count
        if bins <= 256:
            ticks = [0, 64, 128, 192, 255]
        else:
            step = bins // 4
            ticks = list(range(0, bins, step)) + [bins - 1]

        baseline_y = h - pad - label_h + 2  # just below the chart area

        for tick_bin in ticks:
            x = int(pad + draw_w * tick_bin / (bins - 1))
            # Short tick line
            painter.drawLine(x, baseline_y, x, baseline_y + 3)
            # Label
            label = str(tick_bin)
            fm = painter.fontMetrics()
            lw = fm.horizontalAdvance(label)
            # Clamp label so it doesn't overflow left/right edges
            lx = max(0, min(x - lw // 2, w - lw))
            painter.drawText(lx, baseline_y + 3 + fm.ascent(), label)


class CameraControlsWidget(QWidget):
    """
    Camera-agnostic widget for camera controls including photo capture and file management.
    """
    
    # Signal emitted when photo capture completes
    photo_captured = Signal(bool, str)  # success, filepath
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        
        # Default values
        self._default_folder = Path("./output")
        self._current_folder = self._default_folder
        
        # Supported image formats
        self._image_formats = {
            "TIFF": ".tiff",
            "JPEG": ".jpg",
            "PNG": ".png"
        }
        
        # Ensure output folder exists
        self._ensure_output_folder()
        
        # Setup UI
        self._setup_ui()
        
        # Connect signal to handler
        self.photo_captured.connect(self._on_photo_captured)

        # Histogram polling timer — only started when histogram is enabled
        self._histogram_timer = QTimer(self)
        self._histogram_timer.setInterval(33)  # ~30 fps
        self._histogram_timer.timeout.connect(self._poll_histogram)

        # Histogram is temporarily removed until rendering issues are fixed
        # Show histogram group only if the camera supports it
        #self._refresh_histogram_visibility()
        
    def _setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Photo capture group
        capture_group = self._create_capture_group()
        layout.addWidget(capture_group)

        """
        # Temporarily removing histogram until rendering issues are fixed
        # Histogram group (hidden if unsupported)
        self._histogram_group = self._create_histogram_group()
        layout.addWidget(self._histogram_group)
        """
        
        layout.addStretch()
        
    def _create_histogram_group(self) -> QGroupBox:
        """Create the histogram display group."""
        group = QGroupBox("Histogram")
        layout = QVBoxLayout(group)

        # Toggle button row
        toggle_layout = QHBoxLayout()

        self._histogram_toggle = QPushButton("Enable Histogram")
        self._histogram_toggle.setCheckable(True)
        self._histogram_toggle.clicked.connect(self._toggle_histogram)
        toggle_layout.addWidget(self._histogram_toggle)
        toggle_layout.addStretch()

        # Channel toggle buttons (hidden for mono cameras)
        self._legend_widget = QWidget()
        legend_layout = QHBoxLayout(self._legend_widget)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        legend_layout.setSpacing(4)
        self._channel_buttons: list[QPushButton] = []
        for i, (text, fg, bg_on, bg_off) in enumerate((
            ("R", "#fff", "#c03030", "#e8b0b0"),
            ("G", "#fff", "#228b22", "#a8d8a8"),
            ("B", "#fff", "#2060c0", "#a0b8e8"),
        )):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setFixedSize(24, 20)
            btn.setStyleSheet(
                f"QPushButton {{ background: {bg_on}; color: {fg}; font-weight: bold; "
                f"font-size: 11px; border: none; border-radius: 3px; }}"
                f"QPushButton:!checked {{ background: {bg_off}; color: #666; }}"
            )
            channel_index = i
            btn.toggled.connect(
                lambda checked, ch=channel_index: self._histogram_canvas.set_channel_visible(ch, checked)
            )
            legend_layout.addWidget(btn)
            self._channel_buttons.append(btn)
        toggle_layout.addWidget(self._legend_widget)
        self._legend_widget.hide()

        layout.addLayout(toggle_layout)

        """
        # Temporarily removing Histogram until rendering issues are fixed
        # Histogram canvas
        self._histogram_canvas = HistogramWidget()
        self._histogram_canvas.hide()
        layout.addWidget(self._histogram_canvas)
        """
        # Still overlay label
        self._still_label = QLabel("Still overlay shown after capture")
        self._still_label.setStyleSheet("color: #888; font-size: 10px;")
        self._still_label.hide()
        layout.addWidget(self._still_label)

        return group

    def _refresh_histogram_visibility(self) -> None:
        """Show or hide the histogram group based on camera support."""
        ctx = get_app_context()
        camera = getattr(ctx, "camera", None)
        underlying = getattr(camera, "underlying_camera", None)
        supported = underlying is not None and underlying.supports_histogram()
        self._histogram_group.setVisible(supported)

    @Slot()
    def _toggle_histogram(self) -> None:
        """Enable or disable live histogram capture on the camera."""
        ctx = get_app_context()
        camera = getattr(ctx, "camera", None)
        underlying = getattr(camera, "underlying_camera", None)
        if underlying is None:
            return

        enabled = self._histogram_toggle.isChecked()
        success = underlying.set_histogram_enabled(enabled)

        if not success:
            # Revert toggle state on failure
            self._histogram_toggle.setChecked(not enabled)
            error("Failed to toggle histogram on camera")
            return

        self._histogram_toggle.setText(
            "Disable Histogram" if enabled else "Enable Histogram"
        )
        self._histogram_canvas.setVisible(enabled)
        self._still_label.setVisible(enabled)

        if enabled:
            self._histogram_timer.start()
            debug("Histogram display started")
        else:
            self._histogram_timer.stop()
            self._histogram_canvas.update_preview(None)
            self._histogram_canvas.update_still(None)
            self._legend_widget.hide()
            for btn in self._channel_buttons:
                btn.setChecked(True)
            debug("Histogram display stopped")

    @Slot()
    def _poll_histogram(self) -> None:
        """
        Fetch the latest preview histogram from the camera and push it to the
        canvas.  Called by the timer at ~30 fps — never blocks.
        """
        ctx = get_app_context()
        camera = getattr(ctx, "camera", None)
        underlying = getattr(camera, "underlying_camera", None)
        if underlying is None or not underlying.histogram_enabled:
            return

        histogram = underlying.get_preview_histogram()
        if histogram is None:
            return

        # Show RGB legend only for colour histograms
        is_rgb = histogram.shape[0] == 3
        self._legend_widget.setVisible(is_rgb)

        self._histogram_canvas.update_preview(histogram)

    def _create_capture_group(self) -> QGroupBox:
        """Create the photo capture control group"""
        group = QGroupBox("Photo Capture")
        layout = QVBoxLayout(group)
        
        # Folder selection row
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
        
        # Filename row
        filename_layout = QHBoxLayout()
        filename_label = QLabel("Filename:")
        filename_label.setMinimumWidth(100)
        
        self._filename_edit = QLineEdit()
        self._filename_edit.setPlaceholderText("Leave empty for auto-generated name")
        
        filename_layout.addWidget(filename_label)
        filename_layout.addWidget(self._filename_edit, 1)
        
        # Image format row
        format_layout = QHBoxLayout()
        format_label = QLabel("Image Format:")
        format_label.setMinimumWidth(100)
        
        self._format_combo = QComboBox()
        self._format_combo.addItems(self._image_formats.keys())
        
        # Set default format from camera settings
        self._format_combo.setCurrentText(get_app_context().camera.settings.fformat.upper())
        
        self._open_folder_button = QPushButton("Browse Output")
        self._open_folder_button.clicked.connect(self._open_folder)
        
        format_layout.addWidget(format_label)
        format_layout.addWidget(self._format_combo)
        format_layout.addWidget(self._open_folder_button)
        format_layout.addStretch()
        
        # Capture button row
        buttons_layout = QHBoxLayout()
        
        self._capture_button = QPushButton("Take Photo")
        self._capture_button.setMinimumHeight(40)
        self._capture_button.clicked.connect(self._take_photo)
        
        buttons_layout.addWidget(self._capture_button)
        
        # Add all to group layout
        layout.addLayout(folder_layout)
        layout.addLayout(filename_layout)
        layout.addLayout(format_layout)
        layout.addLayout(buttons_layout)
        
        return group
    
    def _ensure_output_folder(self):
        """Ensure the output folder exists"""
        try:
            self._current_folder.mkdir(parents=True, exist_ok=True)
            debug(f"Output folder ready: {self._current_folder}")
        except Exception as e:
            error(f"Failed to create output folder: {e}")
            # Show toast for error
            ctx = get_app_context()
            if ctx.toast:
                ctx.toast.error(f"{str(e)}", title="Folder Creation Failed")
    
    def _browse_folder(self):
        """Open folder selection dialog"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            str(self._current_folder),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            self._current_folder = Path(folder)
            self._folder_edit.setText(str(self._current_folder))
            self._ensure_output_folder()
            info(f"Output folder changed to: {self._current_folder}")
            
            # Show toast notification
            ctx = get_app_context()
            if ctx.toast:
                ctx.toast.success(f"{self._current_folder.name}", title="Output Folder Changed")
    
    def _open_folder(self):
        """Open the output folder in the system file manager"""
        import subprocess
        import sys
        
        ctx = get_app_context()
        toast = ctx.toast
        
        try:
            folder_path = str(self._current_folder.resolve())
            
            if sys.platform == 'win32':
                # Windows
                subprocess.run(['explorer', folder_path])
            elif sys.platform == 'darwin':
                # macOS
                subprocess.run(['open', folder_path])
            else:
                # Linux
                subprocess.run(['xdg-open', folder_path])
            
            info(f"Opened folder: {folder_path}")
            toast.info("Opening in file explorer...", title="Opening Folder", duration=10000)
        except Exception as e:
            error(f"Failed to open folder: {e}")
            toast.error(f"{str(e)}", title="Failed to Open Folder")
            QMessageBox.warning(
                self,
                "Error",
                f"Could not open folder: {e}"
            )
    
    def _generate_filename(self) -> str:
        """Generate a filename based on current timestamp and selected format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        format_name = self._format_combo.currentText()
        extension = self._image_formats[format_name]
        return f"image_{timestamp}{extension}"
    
    def _get_filepath(self) -> Path:
        """Get the complete filepath for saving"""
        # Get selected format
        format_name = self._format_combo.currentText()
        extension = self._image_formats[format_name]
        
        # Use custom filename if provided, otherwise generate one
        filename = self._filename_edit.text().strip()
        if not filename:
            filename = self._generate_filename()
        else:
            # Remove any existing extension
            filename_path = Path(filename)
            filename_base = filename_path.stem
            
            # Add the selected extension
            filename = f"{filename_base}{extension}"
        
        return self._current_folder / filename
    
    @Slot()
    def _take_photo(self):
        """
        Capture a still photo from the camera.
        Works with any camera implementation that supports capture_and_save_still.
        """
        ctx = get_app_context()
        toast = ctx.toast
        
        try:
            camera = ctx.camera
            
            if camera is None:
                warning("Attempted to capture photo but camera is not available")
                toast.warning("Camera not available", title="Camera Error")
                return
            
            if not camera.underlying_camera.is_open:
                warning("Attempted to capture photo but camera is not open")
                toast.warning("Please open the camera first", title="Camera Not Open")
                return
            
            # Get filepath
            filepath = self._get_filepath()
            
            # Ensure folder exists
            self._ensure_output_folder()
            
            info(f"Capturing still image to: {filepath}")
            toast.info("Capturing high-resolution image...", title="Capturing Image")
            
            # Disable button while capturing
            self._capture_button.setEnabled(False)
            
            # Define completion callback - runs on camera thread!
            def on_capture_complete(success: bool, result):
                """Called when capture completes (on camera thread)"""
                # Emit signal to handle UI updates on main thread
                self.photo_captured.emit(success, str(filepath))
            
            # Capture and save still image asynchronously at highest resolution
            # This returns immediately - UI stays responsive!
            camera.capture_and_save_still(
                filepath=filepath,
                additional_metadata={
                    "source": "still_capture"
                },
                timeout_ms=5000,
                on_complete=on_capture_complete
            )
                
        except Exception as e:
            error(f"Error capturing photo: {e}")
            toast.error(f"{str(e)}", title="Capture Error")
            import traceback
            error(traceback.format_exc())
            # Re-enable button on error
            self._capture_button.setEnabled(True)
    
    @Slot(bool, str)
    def _on_photo_captured(self, success: bool, filepath: str):
        """
        Handle photo capture completion on UI thread.
        This slot is called via signal from the camera thread.
        """
        ctx = get_app_context()
        toast = ctx.toast
        
        # Re-enable button
        self._capture_button.setEnabled(True)
        
        if success:
            toast.success(f"Saved to: {Path(filepath).name}", 
                        title="Image Captured", 
                        duration=10000)
            # Clear custom filename after successful capture
            self._filename_edit.clear()

            # Update still histogram overlay if histogram is active
            ctx2 = get_app_context()
            camera = getattr(ctx2, "camera", None)
            underlying = getattr(camera, "underlying_camera", None)
            if underlying is not None and underlying.settings.histogram_enabled:
                self._histogram_canvas.update_still(underlying.settings.get_still_histogram())
        else:
            toast.error("Unable to capture image from camera", title="Capture Failed")