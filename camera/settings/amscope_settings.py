from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from pathlib import Path

from camera.settings.camera_settings import (
    CameraSettings,
    SettingMetadata,
    SettingType,
    RGBALevel,
    FileFormat,
)
from logger import info, error, exception, debug, warning

if TYPE_CHECKING:
    from camera.cameras.base_camera import BaseCamera, CameraResolution


@dataclass
class AmscopeSettings(CameraSettings):
    version: str = "0"
    auto_exposure: bool = True
    exposure: int = 128
    exposure_time: int = 50000
    preview_resolution: str = ""
    still_resolution: str = ""
    temp: int = 6500
    tint: int = 1000
    contrast: int = 0
    hue: int = 0
    saturation: int = 128
    brightness: int = 0
    gamma: int = 100
    gain: int = 100
    level_range_low: RGBALevel = RGBALevel(0, 0, 0, 0)
    level_range_high: RGBALevel = RGBALevel(255, 255, 255, 255)
    fformat: FileFormat = FileFormat.TIFF
    rotate: int = 0
    hflip: bool = False
    vflip: bool = False
    
    # Dark Field Correction
    dfc_enable: bool = False
    _dfc_initialized: bool = False  # Track if DFC has been captured or imported
    dfc_quantity: int = 10
    dfc_filepath: str = ""  # Path to the DFC file
    
    _camera: BaseCamera | None = field(default=None, repr=False, compare=False)
    _ui_update_callback: callable | None = field(default=None, repr=False, compare=False)
    
    def __post_init__(self) -> None:
        super().__post_init__()
    
    def get_metadata(self) -> list[SettingMetadata]:
        """
        Get metadata for all settings with dynamically populated resolution choices.
        """
        # Get available resolutions from camera
        resolutions = self.get_resolutions()
        resolution_choices = [f"{res.width}x{res.height}" for res in resolutions]

        still_resolutions = self.get_still_resolutions()
        still_resolution_choices = [f"{res.width}x{res.height}" for res in still_resolutions]

        return [
            SettingMetadata(
                name="preview_resolution",
                display_name="Preview Resolution",
                setting_type=SettingType.DROPDOWN,
                description="Camera preview resolution",
                choices=resolution_choices,
                group="Capture",
                runtime_changeable=False,
            ),
            SettingMetadata(
                name="still_resolution",
                display_name="Still Resolution",
                setting_type=SettingType.DROPDOWN,
                description="Resolution used when capturing a still image",
                choices=still_resolution_choices,
                group="Capture",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="fformat",
                display_name="File Format",
                setting_type=SettingType.DROPDOWN,
                description="Default file format for saved images",
                choices=self._file_formats,
                group="Capture",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="rotate",
                display_name="Rotation",
                setting_type=SettingType.DROPDOWN,
                description="Rotate the camera image clockwise. Requires camera restart to apply.",
                choices=["0", "90", "180", "270"],
                group="Capture",
                runtime_changeable=False,
            ),
            SettingMetadata(
                name="hflip",
                display_name="Flip Horizontal",
                setting_type=SettingType.BOOL,
                description="Mirror the image horizontally",
                group="Capture",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="vflip",
                display_name="Flip Vertical",
                setting_type=SettingType.BOOL,
                description="Mirror the image vertically",
                group="Capture",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="auto_exposure",
                display_name="Auto Exposure",
                setting_type=SettingType.BOOL,
                description="Enable automatic exposure control",
                group="Exposure",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="exposure",
                display_name="Exposure Target",
                setting_type=SettingType.RANGE,
                description="Target brightness for auto exposure",
                min_value=16,
                max_value=235,
                group="Exposure",
                runtime_changeable=True,
                controlled_by="auto_exposure",
                controlled_when=False,
            ),
            SettingMetadata(
                name="exposure_time",
                display_name="Exposure Time (µs)",
                setting_type=SettingType.RANGE,
                description="Manual exposure time in microseconds",
                min_value=1,
                max_value=1000000,
                group="Exposure",
                runtime_changeable=True,
                controlled_by="auto_exposure",
            ),
            SettingMetadata(
                name="gain",
                display_name="Gain",
                setting_type=SettingType.RANGE,
                description="Sensor gain",
                min_value=100,
                max_value=300,
                group="Exposure",
                runtime_changeable=True,
                controlled_by="auto_exposure",
            ),
            SettingMetadata(
                name="temp",
                display_name="Color Temperature",
                setting_type=SettingType.RANGE,
                description="White balance temperature in Kelvin",
                min_value=2000,
                max_value=15000,
                group="White Balance",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="tint",
                display_name="Tint",
                setting_type=SettingType.RANGE,
                description="White balance tint adjustment",
                min_value=200,
                max_value=2500,
                group="White Balance",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="hue",
                display_name="Hue",
                setting_type=SettingType.RANGE,
                description="Color hue adjustment",
                min_value=-180,
                max_value=180,
                group="Color",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="saturation",
                display_name="Saturation",
                setting_type=SettingType.RANGE,
                description="Color saturation",
                min_value=0,
                max_value=255,
                group="Color",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="brightness",
                display_name="Brightness",
                setting_type=SettingType.RANGE,
                description="Image brightness adjustment",
                min_value=-64,
                max_value=64,
                group="Color",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="contrast",
                display_name="Contrast",
                setting_type=SettingType.RANGE,
                description="Image contrast adjustment",
                min_value=-100,
                max_value=100,
                group="Color",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="gamma",
                display_name="Gamma",
                setting_type=SettingType.RANGE,
                description="Gamma correction",
                min_value=0,
                max_value=180,
                group="Color",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="level_range_low",
                display_name="Black Point",
                setting_type=SettingType.RGBA_LEVEL,
                description="Output level for darkest input values",
                group="Levels",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="level_range_high",
                display_name="White Point",
                setting_type=SettingType.RGBA_LEVEL,
                description="Output level for brightest input values",
                group="Levels",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="dfc_enable",
                display_name="Enable",
                setting_type=SettingType.BOOL,
                description="Enable dark field correction (must capture or import DFC data first)",
                group="Dark Field Correction",
                runtime_changeable=True,
                controlled_by="_dfc_initialized",
                controlled_when=False,
            ),
            SettingMetadata(
                name="dfc_capture",
                display_name="Capture",
                setting_type=SettingType.BUTTON,
                description="Capture dark field correction frames",
                group="Dark Field Correction",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="dfc_import",
                display_name="Import",
                setting_type=SettingType.FILE_PICKER_BUTTON,
                description="Import dark field correction from file",
                group="Dark Field Correction",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="dfc_export",
                display_name="Export",
                setting_type=SettingType.FILE_PICKER_BUTTON,
                description="Export dark field correction to file",
                group="Dark Field Correction",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="dfc_quantity",
                display_name="Quantity",
                setting_type=SettingType.NUMBER_PICKER,
                description="Number of frames to average for dark field correction",
                min_value=1,
                max_value=255,
                group="Dark Field Correction",
                runtime_changeable=True,
            ),
        ]
    
    def _get_metadata_map(self) -> dict[str, SettingMetadata]:
        return {m.name: m for m in self.get_metadata()}
    
    def _validate_range(self, name: str, value: int) -> None:
        meta = self._get_metadata_map().get(name)
        if meta and meta.setting_type == SettingType.RANGE:
            if not (meta.min_value <= value <= meta.max_value):
                raise ValueError(
                    f"{name} must be in [{meta.min_value}, {meta.max_value}], got {value}"
                )

    # ------------------------------------------------------------------
    # Live-value protocol
    # ------------------------------------------------------------------

    def get_live_values(self) -> dict[str, int]:
        """Return live hardware exposure_time and gain while auto_exposure is on."""
        if not self.auto_exposure:
            return {}
        if not (self._camera and hasattr(self._camera, '_hcam')):
            return {}
        try:
            hcam = self._camera._hcam
            return {
                "exposure_time": hcam.get_ExpoTime(),
                "gain": hcam.get_ExpoAGain(),
            }
        except Exception as e:
            error(f"Failed to read live exposure values: {e}")
            return {}

    def on_controller_disabled(self, controller_name: str) -> None:
        """Flush live exposure_time / gain into stored settings when auto_exposure turns off."""
        if controller_name != "auto_exposure":
            super().on_controller_disabled(controller_name)
            return

        if not (self._camera and hasattr(self._camera, '_hcam')):
            return
        try:
            hcam = self._camera._hcam
            self.exposure_time = hcam.get_ExpoTime()
            self.gain = hcam.get_ExpoAGain()
            debug(
                f"Flushed auto-exposure values: exposure_time={self.exposure_time}, gain={self.gain}"
            )
        except Exception as e:
            error(f"Failed to flush live exposure values: {e}")

    # ------------------------------------------------------------------
    
    def set_auto_exposure(self, enabled: bool) -> None:
        if not enabled and self.auto_exposure:
            # Flush hardware values before turning off so stored settings are up-to-date.
            self.on_controller_disabled("auto_exposure")
        self.auto_exposure = enabled
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_AutoExpoEnable(1 if enabled else 0)
    
    def set_exposure(self, value: int) -> None:
        self._validate_range("exposure", value)
        self.exposure = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_AutoExpoTarget(value)
    
    def set_exposure_time(self, time_us: int) -> bool:
        self._validate_range("exposure_time", time_us)
        self.exposure_time = time_us
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_ExpoTime(time_us)
        return True
    
    def set_gain(self, gain: int) -> None:
        self._validate_range("gain", gain)
        self.gain = gain
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_ExpoAGain(gain)
    
    def set_temp(self, value: int) -> None:
        self._validate_range("temp", value)
        self.temp = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_TempTint(value, self.tint)
    
    def set_tint(self, value: int) -> None:
        self._validate_range("tint", value)
        self.tint = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_TempTint(self.temp, value)
    
    def set_temp_tint(self, temp: int, tint: int) -> None:
        self._validate_range("temp", temp)
        self._validate_range("tint", tint)
        self.temp = temp
        self.tint = tint
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_TempTint(temp, tint)
    
    def set_hue(self, value: int) -> None:
        self._validate_range("hue", value)
        self.hue = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_Hue(value)
    
    def set_saturation(self, value: int) -> None:
        self._validate_range("saturation", value)
        self.saturation = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_Saturation(value)
    
    def set_brightness(self, value: int) -> None:
        self._validate_range("brightness", value)
        self.brightness = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_Brightness(value)
    
    def set_contrast(self, value: int) -> None:
        self._validate_range("contrast", value)
        self.contrast = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_Contrast(value)
    
    def set_gamma(self, value: int) -> None:
        self._validate_range("gamma", value)
        self.gamma = value
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_Gamma(value)
    
    def set_level_range(self, low: RGBALevel, high: RGBALevel) -> None:
        low.validate()
        high.validate()
        self.level_range_low = low
        self.level_range_high = high
        if self._camera:
            self._camera._hcam.put_LevelRange(
                (low.r, low.g, low.b, low.a),
                (high.r, high.g, high.b, high.a)
            )

    def set_level_range_low(self, low: RGBALevel) -> None:
        low.validate()
        if self._camera:
            high = self.level_range_high
            self._camera._hcam.put_LevelRange(
                (low.r, low.g, low.b, low.a),
                (high.r, high.g, high.b, high.a)
            )
            self.level_range_low = low

    def set_level_range_high(self, high: RGBALevel) -> None:
        high.validate()
        if self._camera:
            low = self.level_range_low
            self._camera._hcam.put_LevelRange(
                (low.r, low.g, low.b, low.a),
                (high.r, high.g, high.b, high.a)
            )
            self.level_range_high = high
    
    def set_rotate(self, value: int | str, index: int | None = None) -> bool:
        """
        Set the camera image rotation (0, 90, 180, 270 degrees clockwise).

        AMCAM_OPTION_ROTATE cannot be changed while the camera is running, so
        this method follows the same stop/restart pattern as set_preview_resolution.
        The dropdown supplies both the string label (e.g. "90") and the index of
        that label in the choices list.  When ``index`` is provided it is used
        directly; otherwise it is derived from ``value``.
        """
        valid_degrees = [0, 90, 180, 270]

        if index is not None:
            if not (0 <= index < len(valid_degrees)):
                error(f"Invalid rotation index: {index}. Valid range: 0-{len(valid_degrees) - 1}")
                return False
            degrees = valid_degrees[index]
        else:
            try:
                degrees = int(value)
            except (ValueError, TypeError):
                error(f"Invalid rotation value: {value!r}. Must be one of {valid_degrees}")
                return False
            if degrees not in valid_degrees:
                error(f"Invalid rotation value: {degrees}. Must be one of {valid_degrees}")
                return False

        try:
            self.rotate = degrees

            if not (self._camera and hasattr(self._camera, '_hcam')):
                return True

            camera_was_open = self._camera.is_open
            saved_callback = self._camera._callback
            saved_context = self._camera._callback_context

            if camera_was_open:
                debug("Camera is open, stopping to set rotation")
                self._camera.stop_capture()

            amcam = self._camera._get_sdk()
            self._camera._hcam.put_Option(amcam.AMCAM_OPTION_ROTATE, degrees)

            if camera_was_open:
                debug("Restarting camera after rotation change")
                self._camera.start_capture(saved_callback, saved_context)

            debug(f"Successfully changed rotation to {degrees} degrees")
            return True
        except Exception as e:
            error(f"Failed to set rotation: {e}")
            return False

    def set_hflip(self, enabled: bool) -> None:
        """Flip the image horizontally."""
        self.hflip = enabled
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_HFlip(1 if enabled else 0)

    def set_vflip(self, enabled: bool) -> None:
        """Flip the image vertically."""
        self.vflip = enabled
        if self._camera and hasattr(self._camera, '_hcam'):
            self._camera._hcam.put_VFlip(1 if enabled else 0)

    # Dark Field Correction methods
    def set_dfc_enable(self, enabled: bool) -> None:
        """Enable or disable dark field correction."""
        self.dfc_enable = enabled
        if self._camera and hasattr(self._camera, '_hcam'):
            try:
                amcam = self._camera._get_sdk()
                self._camera._hcam.put_Option(amcam.AMCAM_OPTION_DFC, 1 if enabled else 0)
                debug(f"Set DFC enable to {enabled}")
            except Exception as e:
                error(f"Failed to set DFC enable: {e}")
    
    def set_dfc_capture(self) -> None:
        """Capture dark field correction frames and save to config directory."""
        if self._camera and hasattr(self._camera, '_hcam'):
            try:                
                # Save whether DFC was enabled before capture
                dfc_was_enabled = self.dfc_enable
                
                # Reset initialized flag when starting new capture
                self._dfc_initialized = False
                
                # Reset DFC to clear any existing data before capturing new frames
                amcam = self._camera._get_sdk()
                self._camera._hcam.put_Option(amcam.AMCAM_OPTION_DFC, -1)
                info("Reset DFC before capturing new frames")
                
                # Set the average number
                self._camera._hcam.put_Option(amcam.AMCAM_OPTION_DFC, 0xff000000 | self.dfc_quantity)
                
                info(f"Starting DFC capture with {self.dfc_quantity} frames...")
                
                # Store completion handler on camera
                logged_sequences = set()
                
                def on_dfc_event():
                    """Called when DFC event fires - check if we're done"""
                    try:
                        # Query the current DFC state
                        dfc_val = self._camera._hcam.get_Option(amcam.AMCAM_OPTION_DFC)
                        dfc_state = dfc_val & 0xff  # 0=disabled, 1=enabled, 2=inited
                        dfc_sequence = (dfc_val & 0xff00) >> 8  # Current sequence number
                        
                        # Log frame capture if we haven't logged this sequence yet
                        if dfc_sequence > 0 and dfc_sequence not in logged_sequences:
                            info(f"DFC frame {dfc_sequence}/{self.dfc_quantity} captured")
                            logged_sequences.add(dfc_sequence)
                        
                        # Check if DFC is initialized (state == 2) - means all frames captured
                        if dfc_state == 2:
                            info(f"All {self.dfc_quantity} DFC frames captured and processed")
                            
                            # Generate timestamped filename
                            filename = f"dark_field_correction.dfc"
                            
                            # Get config directory
                            config_dir = Path("./config/cameras") / self._camera.model
                            config_dir.mkdir(parents=True, exist_ok=True)
                            
                            filepath = config_dir / filename
                            
                            # Export the captured DFC to the file
                            self._camera._hcam.DfcExport(str(filepath))
                            
                            # Store the filepath
                            self.dfc_filepath = str(filepath)
                            self._dfc_initialized = True
                            
                            info(f"DFC successfully exported to {filepath}")
                            
                            # Re-enable DFC if it was enabled before capture
                            if dfc_was_enabled:
                                self._camera._hcam.put_Option(amcam.AMCAM_OPTION_DFC, 1)
                                self.dfc_enable = True
                                info("Re-enabled DFC after capture completion")
                            
                            # Clean up - remove the callback
                            self._camera._dfc_completion_callback = None
                            
                            # Notify UI that _dfc_initialized has changed
                            # This will enable the dfc_enable checkbox in the UI
                            if self._ui_update_callback:
                                try:
                                    debug(f"Calling UI update callback for _dfc_initialized=True")
                                    self._ui_update_callback('_dfc_initialized', True)
                                    debug(f"UI update callback completed successfully")
                                except Exception as e:
                                    error(f"Failed to notify UI of DFC initialization: {e}")
                            else:
                                warning("No UI update callback registered - UI won't be notified of DFC initialization")
                            
                    except Exception as e:
                        error(f"Failed to process DFC completion: {e}")
                        # Clean up on error
                        self._camera._dfc_completion_callback = None
                
                # Register the callback
                self._camera._dfc_completion_callback = on_dfc_event
                
                # Trigger the capture (async - will complete via events)
                self._camera._hcam.DfcOnce()
                
                info("DFC capture started (will complete asynchronously)")
                
            except Exception as e:
                error(f"Failed to start DFC capture: {e}")
                raise
    
    def set_dfc_import(self, filepath: str) -> None:
        """Import dark field correction from file."""
        if self._camera and hasattr(self._camera, '_hcam'):
            try:
                self._camera._hcam.DfcImport(filepath)
                self.dfc_filepath = filepath
                self._dfc_initialized = True
                info(f"Imported DFC from {filepath} - DFC initialized")
                
                # Notify UI that _dfc_initialized has changed
                if self._ui_update_callback:
                    try:
                        self._ui_update_callback('_dfc_initialized', True)
                    except Exception as e:
                        error(f"Failed to notify UI of DFC initialization: {e}")
            except Exception as e:
                error(f"Failed to import DFC: {e}")
                raise  # Re-raise so UI can show error
    
    def set_dfc_export(self, filepath: str) -> None:
        """Export dark field correction to file."""
        if self._camera and hasattr(self._camera, '_hcam'):
            try:
                self._camera._hcam.DfcExport(filepath)
                info(f"Exported DFC to {filepath}")
            except Exception as e:
                error(f"Failed to export DFC: {e}")
    
    def set_dfc_quantity(self, value: int) -> None:
        """Set the number of frames to average for dark field correction."""
        if not (1 <= value <= 255):
            error(f"DFC quantity must be between 1 and 255, got {value}")
            return
        self.dfc_quantity = value
        debug(f"Set DFC quantity to {value}")

    def get_resolutions(self) -> list[CameraResolution]:
        if self._camera is None or not hasattr(self._camera, '_hcam'):
            return []
        
        from camera.cameras.base_camera import CameraResolution
        
        try:
            resolutions = []
            hcam = self._camera._hcam
            count = hcam.ResolutionNumber()
            for i in range(count):
                width, height = hcam.get_Resolution(i)
                resolutions.append(CameraResolution(width=width, height=height))
            return resolutions
        except Exception as e:
            error(f"Failed to get resolutions: {e}")
            return []
    
    def get_current_resolution(self) -> tuple[int, int, int]:
        if self._camera is None or not hasattr(self._camera, '_hcam'):
            return (0, 0, 0)
        
        try:
            hcam = self._camera._hcam
            index = hcam.get_eSize()
            width, height = hcam.get_Size()
            return (index, width, height)
        except Exception as e:
            error(f"Failed to get current resolution: {e}")
            return (0, 0, 0)

    def get_output_dimensions(self) -> tuple[int, int]:
        """
        Return the final (width, height) of frames delivered by the SDK.
        """
        if self._camera is None or not hasattr(self._camera, '_hcam'):
            return (0, 0)
        try:
            width, height = self._camera._hcam.get_FinalSize()
            return (width, height)
        except Exception:
            pass
        # Fallback: raw sensor resolution (no rotation compensation)
        _, width, height = self.get_current_resolution()
        return (width, height)
    
    def set_preview_resolution(self, value: str, index: int | None = None) -> bool:
        """
        Set camera preview resolution. Requires camera restart.
        """
        try:
            resolutions = self.get_resolutions()
            choices = [f"{r.width}x{r.height}" for r in resolutions]

            if index is None:
                if value not in choices:
                    error(f"Invalid resolution value: {value!r}. Available: {choices}")
                    return False
                index = choices.index(value)
            else:
                if not (0 <= index < len(choices)):
                    error(f"Invalid resolution index: {index}. Valid range: 0-{len(choices) - 1}")
                    return False
                value = choices[index]

            camera_was_open = self._camera.is_open
            saved_callback = self._camera._callback
            saved_context = self._camera._callback_context

            if camera_was_open:
                debug("Camera is open, stopping to set resolution")
                self._camera.stop_capture()

            self._camera._hcam.put_eSize(index)
            self.preview_resolution = value

            if camera_was_open:
                debug("Restarting camera after resolution change")
                self._camera.start_capture(saved_callback, saved_context)

            debug(f"Successfully changed preview resolution to {value} (index {index})")
            return True
        except Exception as e:
            error(f"Failed to set resolution: {e}")
            return False

    def set_still_resolution(self, value: str, index: int | None = None) -> bool:
        """
        Set the still-capture resolution.

        The dropdown supplies both the string label (e.g. "2592x1944") and the index
        of that label in the choices list.  When ``index`` is provided it is used
        directly; otherwise it is derived from ``value``.
        """
        try:
            still_resolutions = self.get_still_resolutions()
            choices = [f"{r.width}x{r.height}" for r in still_resolutions]

            if not choices:
                self.still_resolution = value
                return True

            if index is None:
                if value not in choices:
                    error(f"Invalid still resolution value: {value!r}. Available: {choices}")
                    return False
                index = choices.index(value)
            else:
                if not (0 <= index < len(choices)):
                    error(f"Invalid still resolution index: {index}. Valid range: 0-{len(choices) - 1}")
                    return False
                value = choices[index]

            self.still_resolution = value
            debug(f"Successfully changed still resolution to {value} (index {index})")
            return True
        except Exception as e:
            error(f"Failed to set still resolution: {e}")
            return False

    def get_still_resolutions(self) -> list[CameraResolution]:
        if self._camera is None or not hasattr(self._camera, '_hcam'):
            return []
        
        from camera.cameras.base_camera import CameraResolution
        
        try:
            resolutions = []
            hcam = self._camera._hcam
            count = hcam.StillResolutionNumber()
            for i in range(count):
                width, height = hcam.get_StillResolution(i)
                resolutions.append(CameraResolution(width=width, height=height))
            return resolutions
        except Exception as e:
            error(f"Failed to get still resolutions: {e}")
            return []
    
    def get_exposure_time(self) -> int:
        if self._camera is None or not hasattr(self._camera, '_hcam'):
            return self.exposure_time
        
        try:
            return self._camera._hcam.get_ExpoTime()
        except Exception as e:
            error(f"Failed to get exposure time: {e}")
            return self.exposure_time
    
    def apply_to_camera(self, camera: BaseCamera) -> None:
        self._camera = camera
        info(f"Applying settings to camera {camera.model}")
        
        try:
            if self.preview_resolution:
                self.set_preview_resolution(self.preview_resolution)
            if self.still_resolution:
                self.set_still_resolution(self.still_resolution)
            self.set_auto_exposure(self.auto_exposure)
            self.set_exposure(self.exposure)
            self.set_exposure_time(self.exposure_time)
            self.set_gain(self.gain)
            
            self.set_temp_tint(self.temp, self.tint)
            
            self.set_hue(self.hue)
            self.set_saturation(self.saturation)
            self.set_brightness(self.brightness)
            self.set_contrast(self.contrast)
            self.set_gamma(self.gamma)
            
            self.set_level_range(self.level_range_low, self.level_range_high)
            
            self.set_rotate(self.rotate)
            self.set_hflip(self.hflip)
            self.set_vflip(self.vflip)
            
            # Dark Field Correction
            # Load DFC file if filepath is set and file exists
            if self.dfc_filepath:
                dfc_path = Path(self.dfc_filepath)
                if dfc_path.exists():
                    try:
                        self._camera._hcam.DfcImport(str(dfc_path))
                        self._dfc_initialized = True
                        info(f"Loaded DFC from {dfc_path}")
                    except Exception as e:
                        error(f"Failed to load DFC from {dfc_path}: {e}")
                        self._dfc_initialized = False
                        self.dfc_filepath = ""
                else:
                    warning(f"DFC file not found: {dfc_path}")
                    self._dfc_initialized = False
                    self.dfc_filepath = ""
            else:
                self._dfc_initialized = False
            
            if self.dfc_enable and not self._dfc_initialized:
                warning("Cannot enable DFC: no DFC data available. Disabling DFC.")
                self.dfc_enable = False
            
            self.set_dfc_quantity(self.dfc_quantity)
            self.set_dfc_enable(self.dfc_enable)
            
            debug("Successfully applied all settings to camera")
        except Exception as e:
            exception(f"Failed to apply settings to camera: {e}")
            raise
    
    def refresh_from_camera(self, camera: BaseCamera) -> None:
        self._camera = camera
        info(f"Refreshing settings from camera {camera.model}")
        
        if not hasattr(camera, '_hcam') or camera._hcam is None:
            error("Camera not available for refresh")
            return
        
        hcam = camera._hcam
        
        try:
            self.auto_exposure = bool(hcam.get_AutoExpoEnable())
            self.exposure = hcam.get_AutoExpoTarget()
            self.exposure_time = hcam.get_ExpoTime()
            self.gain = hcam.get_ExpoAGain()
            temp, tint = hcam.get_TempTint()
            self.temp = temp
            self.tint = tint
            
            self.hue = hcam.get_Hue()
            self.saturation = hcam.get_Saturation()
            self.brightness = hcam.get_Brightness()
            self.contrast = hcam.get_Contrast()
            self.gamma = hcam.get_Gamma()
            
            low, high = hcam.get_LevelRange()
            self.level_range_low = RGBALevel(r=low[0], g=low[1], b=low[2], a=low[3])
            self.level_range_high = RGBALevel(r=high[0], g=high[1], b=high[2], a=high[3])
            
            index = hcam.get_eSize()
            resolutions = self.get_resolutions()
            if 0 <= index < len(resolutions):
                r = resolutions[index]
                self.preview_resolution = f"{r.width}x{r.height}"
            else:
                self.preview_resolution = ""

            still_resolutions = self.get_still_resolutions()
            if still_resolutions:
                r = still_resolutions[0]
                self.still_resolution = f"{r.width}x{r.height}"

            rotate_raw = hcam.get_Option(camera._get_sdk().AMCAM_OPTION_ROTATE)
            self.rotate = rotate_raw if rotate_raw in (0, 90, 180, 270) else 0
            self.hflip = bool(hcam.get_HFlip())
            self.vflip = bool(hcam.get_VFlip())
            
            # Dark Field Correction
            amcam = camera._get_sdk()
            dfc_val = hcam.get_Option(amcam.AMCAM_OPTION_DFC)
            dfc_state = dfc_val & 0xff
            self.dfc_enable = (dfc_state == 1)  # 0=disabled, 1=enabled, 2=inited
            self._dfc_initialized = (dfc_state >= 1)
            dfc_avg = (dfc_val & 0xff0000) >> 16
            if dfc_avg > 0:
                self.dfc_quantity = dfc_avg
            
            info("Successfully refreshed all settings from camera")
        except Exception as e:
            exception(f"Failed to refresh settings from camera: {e}")