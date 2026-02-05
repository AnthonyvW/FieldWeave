from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QFormLayout,
    QGroupBox,
    QComboBox,
    QSlider,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QScrollArea,
    QFrame,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal, Slot

from app_context import get_app_context
from logger import info, error, warning, debug
from camera.cameras.base_camera import CameraResolution

class CameraSettingsWidget(QWidget):
    """Widget for displaying and editing camera settings"""
    
    settings_loaded = Signal(bool, object)  # success, result
    modifications_changed = Signal(bool)  # has_modifications
    
    def __init__(self, parent_dialog=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        self.parent_dialog = parent_dialog
        self.ctx = get_app_context()
        self._settings_widgets: dict[str, QWidget] = {}
        self._updating_from_camera = False
        self._modified_settings: set[str] = set()  # Track which settings have been modified
        self._saved_values: dict[str, any] = {}  # Store saved values for comparison
        self._group_names: list[str] = []  # Track group names in order
        self._group_widgets: dict[str, QGroupBox] = {}  # Map group names to widgets
        
        self._setup_ui()
        self._connect_signals()
        self._populate_camera_list()
    
    def _setup_ui(self) -> None:
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        # Content widget inside scroll area with white background
        content = QWidget()
        content.setStyleSheet("background: white;")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)
        
        # Camera title with larger font
        camera_title = QLabel("Camera")
        camera_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #5f6368;")
        content_layout.addWidget(camera_title)
        
        # Camera selection group
        camera_group = QGroupBox("Camera Device")
        camera_layout = QFormLayout(camera_group)
        
        # Camera combo with refresh button on same line
        camera_select_layout = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(300)
        camera_select_layout.addWidget(self.camera_combo)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setMaximumWidth(80)
        camera_select_layout.addWidget(self.refresh_btn)
        
        camera_layout.addRow("Select Camera:", camera_select_layout)
        
        content_layout.addWidget(camera_group)
        
        # Camera settings groups (will be populated dynamically)
        self.settings_container = QWidget()
        self.settings_layout = QVBoxLayout(self.settings_container)
        self.settings_layout.setContentsMargins(0, 0, 0, 0)
        self.settings_layout.setSpacing(10)
        
        content_layout.addWidget(self.settings_container)
        content_layout.addStretch()
        
        # Reset and Load buttons at bottom
        button_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.setEnabled(False)
        self.load_btn = QPushButton("Load Settings")
        self.load_btn.setEnabled(False)
        
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.load_btn)
        button_layout.addStretch()
        
        content_layout.addLayout(button_layout)
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
    
    def _connect_signals(self) -> None:
        """Connect signals and slots"""
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        self.refresh_btn.clicked.connect(lambda: self._populate_camera_list(force_enumerate=True))
        self.reset_btn.clicked.connect(self._reset_settings)
        self.load_btn.clicked.connect(self._load_settings)
        self.settings_loaded.connect(self._on_settings_loaded)
        
        # Connect to parent dialog's save button if available
        if self.parent_dialog:
            if hasattr(self.parent_dialog, 'save_btn'):
                self.parent_dialog.save_btn.clicked.connect(self._save_settings)
                # Connect modifications_changed signal to enable/disable save button
                self.modifications_changed.connect(self.parent_dialog.save_btn.setEnabled)
            if hasattr(self.parent_dialog, 'save_camera_settings'):
                self.parent_dialog.save_camera_settings.connect(self._save_settings)
        
        # Connect to camera manager signals
        if self.ctx.camera_manager:
            self.ctx.camera_manager.camera_list_changed.connect(
                self._on_camera_list_changed
            )
            self.ctx.camera_manager.active_camera_changed.connect(
                self._on_active_camera_changed
            )
    
    def _populate_camera_list(self, force_enumerate: bool = False) -> None:
        """Populate the camera dropdown with available cameras
        
        Args:
            force_enumerate: If True, force re-enumeration. Otherwise use cached list.
        """
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        
        if not self.ctx.camera_manager:
            self.camera_combo.addItem("No camera manager available")
            self.camera_combo.setEnabled(False)
            self.camera_combo.blockSignals(False)
            return
        
        # Use cached list unless forced to enumerate
        if force_enumerate:
            cameras = self.ctx.camera_manager.enumerate_cameras()
        else:
            cameras = self.ctx.camera_manager.available_cameras
            # If no cached cameras, enumerate once
            if not cameras:
                cameras = self.ctx.camera_manager.enumerate_cameras()
        
        if not cameras:
            self.camera_combo.addItem("No cameras detected")
            self.camera_combo.setEnabled(False)
        else:
            self.camera_combo.setEnabled(True)
            
            # Add cameras to dropdown
            for camera_info in cameras:
                display_text = f"{camera_info.display_name} ({camera_info.model})"
                self.camera_combo.addItem(display_text, camera_info)
            
            # Select the active camera if any
            active_info = self.ctx.camera_manager.active_camera_info
            if active_info:
                for i in range(self.camera_combo.count()):
                    info_at_index = self.camera_combo.itemData(i)
                    if info_at_index and info_at_index.device_id == active_info.device_id:
                        self.camera_combo.setCurrentIndex(i)
                        break
        
        self.camera_combo.blockSignals(False)
        
        # Only refresh settings if we have an active camera
        if self.ctx.camera and self.ctx.camera.underlying_camera.is_open:
            self._refresh_settings_display()
    
    @Slot(int)
    def _on_camera_changed(self, index: int) -> None:
        """Handle camera selection change"""
        if index < 0:
            return
        
        camera_info = self.camera_combo.itemData(index)
        if not camera_info:
            return
        
        # Switch to the selected camera
        info(f"Switching to camera: {camera_info.display_name}")
        success = self.ctx.camera_manager.switch_camera(camera_info)
        
        if success:
            self._refresh_settings_display()
        else:
            error(f"Failed to switch to camera: {camera_info.display_name}")
    
    @Slot()
    def _on_camera_list_changed(self) -> None:
        """Handle camera list changes from camera manager"""
        # Use cached list since camera_list_changed is emitted after enumeration
        self._populate_camera_list(force_enumerate=False)
    
    @Slot(object)
    def _on_active_camera_changed(self, camera_info) -> None:
        """Handle active camera changes from camera manager"""
        self._refresh_settings_display()
        
        # Update combo box selection
        if camera_info:
            self.camera_combo.blockSignals(True)
            for i in range(self.camera_combo.count()):
                info_at_index = self.camera_combo.itemData(i)
                if info_at_index and info_at_index.device_id == camera_info.device_id:
                    self.camera_combo.setCurrentIndex(i)
                    break
            self.camera_combo.blockSignals(False)
    
    def _refresh_settings_display(self) -> None:
        """Refresh the settings display based on current camera"""
        # Clear existing settings widgets
        self._clear_settings_display()
        
        camera = self.ctx.camera
        if not camera:
            self._show_no_camera_message()
            return
        
        # Check if camera is open
        if not camera.underlying_camera.is_open:
            self._show_camera_not_open_message()
            return
        
        # Get settings metadata
        try:
            settings = camera.settings
            metadata_list = settings.get_metadata()
            
            # Store current values as "saved" baseline
            self._saved_values.clear()
            self._modified_settings.clear()
            for meta in metadata_list:
                current_value = getattr(settings, meta.name, None)
                self._saved_values[meta.name] = current_value
            
            # Group settings by category
            grouped_settings = self._group_settings(metadata_list)
            
            # Clear and rebuild group tracking
            self._group_names.clear()
            self._group_widgets.clear()
            
            # Create UI for each group
            for group_name, settings_in_group in grouped_settings.items():
                group_box = self._create_settings_group(group_name, settings_in_group)
                self.settings_layout.addWidget(group_box)
                
                # Track the group
                self._group_names.append(group_name)
                self._group_widgets[group_name] = group_box
                
                # Register with parent dialog for scrolling
                if self.parent_dialog and hasattr(self.parent_dialog, 'register_group_box'):
                    self.parent_dialog.register_group_box("Camera", group_name, group_box)
            
            # Update tree items in parent dialog
            if self.parent_dialog and hasattr(self.parent_dialog, '_update_camera_groups'):
                self.parent_dialog._update_camera_groups(self._group_names)
            
            # Enable buttons
            if self.parent_dialog and hasattr(self.parent_dialog, 'save_btn'):
                self.parent_dialog.save_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            
        except Exception as e:
            error(f"Error loading camera settings: {e}")
            self._show_error_message(str(e))
    
    def _clear_settings_display(self) -> None:
        """Clear all settings widgets"""
        while self.settings_layout.count():
            item = self.settings_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self._settings_widgets.clear()
        
        # Disable buttons
        if self.parent_dialog and hasattr(self.parent_dialog, 'save_btn'):
            self.parent_dialog.save_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
    
    def _show_no_camera_message(self) -> None:
        """Show message when no camera is available"""
        label = QLabel("No camera selected. Please select a camera from the dropdown above.")
        label.setWordWrap(True)
        label.setStyleSheet("color: gray; padding: 20px;")
        self.settings_layout.addWidget(label)
    
    def _show_camera_not_open_message(self) -> None:
        """Show message when camera is not open"""
        label = QLabel("Camera is not open. Please open the camera first.")
        label.setWordWrap(True)
        label.setStyleSheet("color: orange; padding: 20px;")
        self.settings_layout.addWidget(label)
    
    def _show_error_message(self, error_msg: str) -> None:
        """Show error message"""
        label = QLabel(f"Error loading settings: {error_msg}")
        label.setWordWrap(True)
        label.setStyleSheet("color: red; padding: 20px;")
        self.settings_layout.addWidget(label)
    
    def _group_settings(self, metadata_list: list) -> dict[str, list]:
        """Group settings by their group property"""
        grouped = {}
        
        for meta in metadata_list:
            group = meta.group if hasattr(meta, 'group') and meta.group else "General"
            
            if group not in grouped:
                grouped[group] = []
            
            grouped[group].append(meta)
        
        return grouped
    
    def _create_settings_group(self, group_name: str, settings_list: list) -> QGroupBox:
        """Create a group box for a category of settings"""
        group_box = QGroupBox(group_name)
        layout = QFormLayout(group_box)
        
        for setting_meta in settings_list:
            widget = self._create_setting_widget(setting_meta)
            if widget:
                # Create label with tooltip
                label = QLabel(setting_meta.display_name + ":")
                if hasattr(setting_meta, 'description') and setting_meta.description:
                    label.setToolTip(setting_meta.description)
                
                layout.addRow(label, widget)
                
                # Store both widget and label for styling
                # Create a container that holds both for easier styling
                widget_container = QWidget()
                widget_container.setProperty("label", label)
                widget_container.setProperty("control", widget)
                self._settings_widgets[setting_meta.name] = widget_container
        
        return group_box
    
    def _create_setting_widget(self, meta) -> QWidget | None:
        """Create appropriate widget for a setting based on its metadata"""
        camera = self.ctx.camera
        if not camera:
            return None
        
        settings = camera.settings
        setting_type = meta.setting_type if hasattr(meta, 'setting_type') else None
        
        # Handle both enum and string values
        if setting_type is None:
            warning(f"No setting type for {meta.name}")
            return None
        
        # Convert enum to string value if needed
        type_str = setting_type.value if hasattr(setting_type, 'value') else str(setting_type)
        
        # Create widget based on type
        if type_str == "bool":
            return self._create_bool_widget(meta, settings)
        elif type_str == "range":
            return self._create_range_widget(meta, settings)
        elif type_str == "dropdown":
            return self._create_dropdown_widget(meta, settings)
        elif type_str == "rgba_level":
            return self._create_rgba_level_widget(meta, settings)
        elif type_str == "rgb_gain":
            # TODO: Implement custom RGB gain widget
            warning(f"RGB_GAIN widget not yet implemented for {meta.name}")
            return None
        else:
            warning(f"Unknown setting type: {type_str} for {meta.name}")
            return None
    
    def _create_bool_widget(self, meta, settings) -> QCheckBox:
        """Create checkbox for boolean settings"""
        checkbox = QCheckBox()
        
        # Get current value
        current_value = getattr(settings, meta.name, False)
        checkbox.setChecked(current_value)
        
        # Set tooltip
        if hasattr(meta, 'description') and meta.description:
            checkbox.setToolTip(meta.description)
        
        # Connect to setter
        setter_name = f"set_{meta.name}"
        if hasattr(settings, setter_name):
            checkbox.stateChanged.connect(
                lambda state: self._on_bool_changed(setter_name, state == Qt.CheckState.Checked)
            )
        else:
            warning(f"No setter found: {setter_name}")
        
        return checkbox
    
    def _create_range_widget(self, meta, settings) -> QWidget:
        """Create slider with value display for range settings"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Determine if we need float or int
        is_float = hasattr(meta, 'min_value') and isinstance(meta.min_value, float)
        
        if is_float:
            # Use double spin box for float values
            spinbox = QDoubleSpinBox()
            spinbox.setDecimals(2)
        else:
            # Use regular spin box for int values
            spinbox = QSpinBox()
        
        # Set fixed width to accommodate 6 digits plus decimals/sign
        spinbox.setFixedWidth(90)
        
        # Set range
        if hasattr(meta, 'min_value') and hasattr(meta, 'max_value'):
            spinbox.setMinimum(meta.min_value)
            spinbox.setMaximum(meta.max_value)
        
        # Get current value
        current_value = getattr(settings, meta.name, 0)
        spinbox.setValue(current_value)
        
        # Set tooltip
        if hasattr(meta, 'description') and meta.description:
            spinbox.setToolTip(meta.description)
        
        # Create slider
        slider = QSlider(Qt.Orientation.Horizontal)
        
        if is_float:
            # For float values, scale to int range for slider
            slider.setMinimum(0)
            slider.setMaximum(1000)
            slider.setValue(
                int((current_value - meta.min_value) / (meta.max_value - meta.min_value) * 1000)
            )
        else:
            slider.setMinimum(int(meta.min_value) if hasattr(meta, 'min_value') else 0)
            slider.setMaximum(int(meta.max_value) if hasattr(meta, 'max_value') else 100)
            slider.setValue(int(current_value))
        
        # Connect signals
        setter_name = f"set_{meta.name}"
        if hasattr(settings, setter_name):
            if is_float:
                spinbox.valueChanged.connect(
                    lambda val: self._on_float_changed(setter_name, val, slider, meta)
                )
                slider.valueChanged.connect(
                    lambda val: self._on_slider_changed_float(setter_name, val, spinbox, meta)
                )
            else:
                spinbox.valueChanged.connect(
                    lambda val: self._on_int_changed(setter_name, val, slider)
                )
                slider.valueChanged.connect(
                    lambda val: self._on_slider_changed_int(setter_name, val, spinbox)
                )
        
        layout.addWidget(slider)
        layout.addWidget(spinbox)
        
        return container
    
    def _create_dropdown_widget(self, meta, settings) -> QComboBox:
        """Create dropdown for choice settings"""
        combo = QComboBox()
        
        # Add choices
        if hasattr(meta, 'choices') and meta.choices:
            for choice in meta.choices:
                combo.addItem(str(choice), choice)
        
        # Set current value
        current_value = getattr(settings, meta.name, None)
        if current_value is not None:
            index = combo.findData(current_value)
            if index >= 0:
                combo.setCurrentIndex(index)
        
        # Set tooltip
        if hasattr(meta, 'description') and meta.description:
            combo.setToolTip(meta.description)
        
        # Connect to setter
        setter_name = f"set_{meta.name}"
        if hasattr(settings, setter_name):
            combo.currentIndexChanged.connect(
                lambda idx: self._on_dropdown_changed(setter_name, idx, combo.itemData(idx))
            )
        else:
            warning(f"No setter found: {setter_name}")
        
        return combo
    
    def _create_rgba_level_widget(self, meta, settings) -> QWidget:
        """Create RGBA level widget with four spinboxes for R, G, B, A"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Get current value (should be an RGBALevel object)
        current_value = getattr(settings, meta.name, None)
        
        # Create spinboxes for each channel
        spinboxes = {}
        for channel in ['r', 'g', 'b', 'a']:
            channel_layout = QVBoxLayout()
            channel_layout.setSpacing(2)
            
            label = QLabel(channel.upper())
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            spinbox = QSpinBox()
            spinbox.setMinimum(0)
            spinbox.setMaximum(255)
            spinbox.setFixedWidth(60)
            
            # Set current value if available
            if current_value and hasattr(current_value, channel):
                spinbox.setValue(getattr(current_value, channel))
            
            channel_layout.addWidget(label)
            channel_layout.addWidget(spinbox)
            
            layout.addLayout(channel_layout)
            spinboxes[channel] = spinbox
        
        # Connect to setter
        setter_name = f"set_{meta.name}"
        if hasattr(settings, setter_name):
            # Create a function that updates all values when any spinbox changes
            def on_rgba_changed():
                if self._updating_from_camera:
                    return
                
                # Import RGBALevel here to avoid circular imports
                try:
                    from camera.settings.camera_settings import RGBALevel
                    
                    new_value = RGBALevel(
                        r=spinboxes['r'].value(),
                        g=spinboxes['g'].value(),
                        b=spinboxes['b'].value(),
                        a=spinboxes['a'].value()
                    )
                    
                    setter = getattr(settings, setter_name)
                    setter(new_value)
                    
                    # Mark as modified
                    setting_name = setter_name.replace("set_", "")
                    self._mark_setting_modified(setting_name, new_value)
                    
                    debug(f"Set {setter_name} to {new_value}")
                except Exception as e:
                    error(f"Error setting {setter_name}: {e}")
            
            # Connect all spinboxes to the same handler
            for spinbox in spinboxes.values():
                spinbox.valueChanged.connect(on_rgba_changed)
        
        layout.addStretch()
        return container
    
    def _mark_setting_modified(self, setting_name: str, current_value) -> None:
        """Mark a setting as modified and update its widget styling"""
        # Check if value actually changed from saved value
        saved_value = self._saved_values.get(setting_name)
        
        # Handle different value types for comparison
        is_modified = False
        if saved_value is None:
            is_modified = current_value is not None
        elif hasattr(saved_value, '__dict__'):
            # For objects like RGBALevel, compare attributes
            is_modified = str(saved_value) != str(current_value)
        else:
            is_modified = saved_value != current_value
        
        # Update modified tracking
        if is_modified:
            self._modified_settings.add(setting_name)
        else:
            self._modified_settings.discard(setting_name)
        
        # Update widget styling
        self._update_widget_styling(setting_name, is_modified)
        
        # Update category color in parent dialog
        if self.parent_dialog and hasattr(self.parent_dialog, 'set_category_modified'):
            self.parent_dialog.set_category_modified("Camera", len(self._modified_settings) > 0)
        
        # Emit signal about modification state change
        self.modifications_changed.emit(len(self._modified_settings) > 0)
    
    def _update_widget_styling(self, setting_name: str, is_modified: bool) -> None:
        """Update the visual styling of a widget to indicate modification"""
        widget = self._settings_widgets.get(setting_name)
        if not widget:
            return
        
        if is_modified:
            # Orange text and slider for modified settings
            # Apply styling to all child widgets
            self._apply_orange_styling(widget, True)
        else:
            # Clear custom styling to revert to default
            self._apply_orange_styling(widget, False)
    
    def _apply_orange_styling(self, widget: QWidget, orange: bool) -> None:
        """Apply or remove orange styling to a widget and its children"""
        # Get the actual control widget and label from the container
        label = widget.property("label")
        control = widget.property("control")
        
        if not control:
            return
        
        if orange:
            # Color the label text
            if label:
                label.setStyleSheet("QLabel { color: #FFA500; }")
            
            # For different widget types, apply orange
            if isinstance(control, QCheckBox):
                control.setStyleSheet("QCheckBox { color: #FFA500; }")
            elif isinstance(control, QComboBox):
                control.setStyleSheet("QComboBox { color: #FFA500; }")
            elif isinstance(control, QWidget):
                for child in control.findChildren(QSlider):
                    child.setStyleSheet("""
                        QSlider::handle:horizontal {
                            background: #FFA500;
                            border: 1px solid #FFA500;
                            width: 18px;
                            margin: -2px 0;
                            border-radius: 3px;
                        }
                    """)
        else:
            # Clear styling
            if label:
                label.setStyleSheet("")
            control.setStyleSheet("")
            for child in control.findChildren(QWidget):
                child.setStyleSheet("")
    
    def _clear_all_modifications(self) -> None:
        """Clear all modification markers and update saved values"""
        camera = self.ctx.camera
        if not camera:
            return
        
        settings = camera.settings
        
        # Update saved values to current values
        for setting_name in list(self._modified_settings):
            current_value = getattr(settings, setting_name, None)
            self._saved_values[setting_name] = current_value
            self._update_widget_styling(setting_name, False)
        
        self._modified_settings.clear()
        
        # Update category color in parent dialog
        if self.parent_dialog and hasattr(self.parent_dialog, 'set_category_modified'):
            self.parent_dialog.set_category_modified("Camera", False)
        
        # Emit signal about modification state change
        self.modifications_changed.emit(False)
    
    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes"""
        return len(self._modified_settings) > 0
    
    def get_group_names(self) -> list[str]:
        """Get list of group names in the settings"""
        return self._group_names.copy()
    
    def _on_bool_changed(self, setter_name: str, value: bool) -> None:
        """Handle boolean setting change"""
        if self._updating_from_camera:
            return
        
        camera = self.ctx.camera
        if not camera:
            return
        
        try:
            setter = getattr(camera.settings, setter_name)
            setter(value)
            
            # Extract setting name from setter name (remove "set_" prefix)
            setting_name = setter_name.replace("set_", "")
            self._mark_setting_modified(setting_name, value)
            
            debug(f"Set {setter_name} to {value}")
        except Exception as e:
            error(f"Error setting {setter_name}: {e}")
    
    def _on_int_changed(self, setter_name: str, value: int, slider: QSlider) -> None:
        """Handle integer setting change from spinbox"""
        if self._updating_from_camera:
            return
        
        # Update slider
        slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(False)
        
        camera = self.ctx.camera
        if not camera:
            return
        
        try:
            setter = getattr(camera.settings, setter_name)
            setter(value)
            
            # Extract setting name from setter name (remove "set_" prefix)
            setting_name = setter_name.replace("set_", "")
            self._mark_setting_modified(setting_name, value)
            
            debug(f"Set {setter_name} to {value}")
        except Exception as e:
            error(f"Error setting {setter_name}: {e}")
    
    def _on_float_changed(self, setter_name: str, value: float, slider: QSlider, meta) -> None:
        """Handle float setting change from spinbox"""
        if self._updating_from_camera:
            return
        
        # Update slider
        slider.blockSignals(True)
        slider_val = int((value - meta.min_value) / (meta.max_value - meta.min_value) * 1000)
        slider.setValue(slider_val)
        slider.blockSignals(False)
        
        camera = self.ctx.camera
        if not camera:
            return
        
        try:
            setter = getattr(camera.settings, setter_name)
            setter(value)
            
            # Extract setting name from setter name (remove "set_" prefix)
            setting_name = setter_name.replace("set_", "")
            self._mark_setting_modified(setting_name, value)
            
            debug(f"Set {setter_name} to {value}")
        except Exception as e:
            error(f"Error setting {setter_name}: {e}")
    
    def _on_slider_changed_int(self, setter_name: str, value: int, spinbox: QSpinBox) -> None:
        """Handle integer setting change from slider"""
        if self._updating_from_camera:
            return
        
        # Update spinbox
        spinbox.blockSignals(True)
        spinbox.setValue(value)
        spinbox.blockSignals(False)
        
        camera = self.ctx.camera
        if not camera:
            return
        
        try:
            setter = getattr(camera.settings, setter_name)
            setter(value)
            
            # Extract setting name from setter name (remove "set_" prefix)
            setting_name = setter_name.replace("set_", "")
            self._mark_setting_modified(setting_name, value)
            
            debug(f"Set {setter_name} to {value}")
        except Exception as e:
            error(f"Error setting {setter_name}: {e}")
    
    def _on_slider_changed_float(self, setter_name: str, slider_val: int, 
                                  spinbox: QDoubleSpinBox, meta) -> None:
        """Handle float setting change from slider"""
        if self._updating_from_camera:
            return
        
        # Convert slider value to float
        value = meta.min_value + (slider_val / 1000.0) * (meta.max_value - meta.min_value)
        
        # Update spinbox
        spinbox.blockSignals(True)
        spinbox.setValue(value)
        spinbox.blockSignals(False)
        
        camera = self.ctx.camera
        if not camera:
            return
        
        try:
            setter = getattr(camera.settings, setter_name)
            setter(value)
            
            # Extract setting name from setter name (remove "set_" prefix)
            setting_name = setter_name.replace("set_", "")
            self._mark_setting_modified(setting_name, value)
            
            debug(f"Set {setter_name} to {value}")
        except Exception as e:
            error(f"Error setting {setter_name}: {e}")
    
    def _on_dropdown_changed(self, setter_name: str, index: int, value) -> None:
        """Handle dropdown setting change
        
        Args:
            setter_name: Name of the setter method (e.g., 'set_resolution')
            index: Index of the selected item in the dropdown
            value: Value associated with the selected item
        """
        if self._updating_from_camera:
            return
        
        camera = self.ctx.camera
        if not camera:
            return
        
        try:
            setter = getattr(camera.settings, setter_name)
            # Pass both index and value to the setter
            setter(index=index, value=value)
            
            # Extract setting name from setter name (remove "set_" prefix)
            setting_name = setter_name.replace("set_", "")
            self._mark_setting_modified(setting_name, value)
            
            debug(f"Set {setter_name} to index={index}, value={value}")
        except Exception as e:
            error(f"Error setting {setter_name}: {e}")
    
    @Slot()
    def _save_settings(self) -> None:
        """Save current camera settings"""
        camera = self.ctx.camera
        if not camera:
            warning("No camera to save settings from")
            return
        
        try:
            camera.save_settings()
            info("Camera settings saved successfully")
            
            # Clear modification markers
            self._clear_all_modifications()
            
            self.ctx.toast.info("Settings saved successfully", duration=2000)
        except Exception as e:
            error(f"Error saving camera settings: {e}")
            self.ctx.toast.info(f"Error saving settings: {e}", duration=3000)
    
    @Slot()
    def _load_settings(self) -> None:
        """Load camera settings from file"""
        camera = self.ctx.camera
        if not camera:
            warning("No camera to load settings to")
            return
        
        # Open file picker for YAML files
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Camera Settings",
            "config/cameras",
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        
        # If no file was selected, do nothing
        if not file_path:
            return
        
        # Convert to Path for the callback
        selected_path = Path(file_path)
        
        def on_load_complete(success: bool, result):
            """Callback runs on camera thread - emit signal to UI thread"""
            self.settings_loaded.emit(success, (result, str(selected_path)))
        
        # Load settings with callback
        camera.load_settings(selected_path, on_complete=on_load_complete)
    
    @Slot(bool, object)
    def _on_settings_loaded(self, success: bool, data: tuple) -> None:
        """Handle settings loaded callback on UI thread"""
        result, file_path = data
        
        if success:
            info(f"Camera settings loaded successfully from {file_path}")
            
            # Refresh the display to show loaded values
            self._refresh_settings_display()
            
            self.ctx.toast.success("Settings loaded successfully", duration=2000)
        else:
            error(f"Error loading camera settings from {file_path}: {result}")
            self.ctx.toast.error(f"Error loading settings: {result}", duration=3000)
    
    @Slot()
    def _reset_settings(self) -> None:
        """Reset camera settings to defaults"""
        camera = self.ctx.camera
        if not camera:
            warning("No camera to reset settings on")
            return
        
        # Confirm reset with user
        reply = QMessageBox.question(
            self,
            "Reset to Defaults",
            "Are you sure you want to reset all camera settings to their default values? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            # Refresh from camera hardware (gets defaults)
            settings = camera.settings
            settings.refresh_from_camera(camera.underlying_camera)
            
            info("Camera settings reset to defaults")
            
            # Refresh the display
            self._refresh_settings_display()
            
            self.ctx.toast.info("Settings reset to defaults", duration=2000)
        except Exception as e:
            error(f"Error resetting camera settings: {e}")
            self.ctx.toast.info(f"Error resetting settings: {e}", duration=3000)


def camera_page(parent_dialog=None) -> QWidget:
    """Create and return the camera settings page widget"""
    return CameraSettingsWidget(parent_dialog=parent_dialog)