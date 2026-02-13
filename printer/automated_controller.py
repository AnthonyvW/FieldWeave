import time
from typing import Callable, Optional, List, Tuple
import math 
from pathlib import Path

from .models import Position, FocusScore
from .base_controller import BasePrinterController
from .automation.autofocus_mixin import AutofocusMixin
from .automation.camera_calibration_mixin import CameraCalibrationMixin
from image_processing.machine_vision import MachineVision

from UI.list_frame import ListFrame
from UI.input.button import Button
from UI.input.toggle_button import ToggleButton
from UI.input.text_field import TextField

from forgeConfig import (
    ForgeSettings,
)
from .automation_config import (
    AutomationSettings,
    AutomationSettingsManager,
    ACTIVE_FILENAME as AUTO_ACTIVE_FILENAME,
    DEFAULT_FILENAME as AUTO_DEFAULT_FILENAME,
)

from .base_controller import command
from camera.image_name_formatter import ImageNameFormatter
from .automation_config import AutomationSettings, AutomationSettingsManager


class AutomatedPrinter(CameraCalibrationMixin, AutofocusMixin, BasePrinterController):
    """Extended printer controller with automation capabilities"""
    AUTOMATION_CONFIG_SUBDIR = ""
    def __init__(self, forgeConfig: ForgeSettings, camera):
        super().__init__(forgeConfig)

        AutomationSettingsManager.scope_dir(self.AUTOMATION_CONFIG_SUBDIR)
        self.automation_settings: AutomationSettings = AutomationSettingsManager.load(self.AUTOMATION_CONFIG_SUBDIR)
        
        # Initialize printer configurations
        self.camera = camera
        s = self.automation_settings
        self.machine_vision = MachineVision(
            camera,
            tile_size=s.tile_size,
            stride=s.stride,
            top_percent=s.top_percent,
            min_score=s.min_score,
            soft_min_score=s.soft_min_score,
            inset_left_pct=s.inset_left_pct,
            inset_top_pct=s.inset_top_pct,
            inset_right_pct=s.inset_right_pct,
            inset_bottom_pct=s.inset_bottom_pct,
            scale_factor= s.scale_factor
        )
        self.is_automated = False
        
        self.image_formatter = ImageNameFormatter(
            controller=self,
            pad_positions=self.automation_settings.zero_pad,
            position_decimals=0,
            delimiter=self.automation_settings.delimiter,
            template=self.automation_settings.image_name_template,
        )

        self.sample_list: ListFrame | None = None
        self.current_sample_index = 1
        self.live_plots_enabled: bool = False

        # Initialize autofocus handlers from mixin
        self._init_autofocus_handlers()
        
        # Initialize camera calibration handlers from mixin
        self._init_camera_calibration_handlers()

        
        # Automation Routines
        self.register_handler("SCAN_SAMPLE_BOUNDS", self.scan_sample_bounds)

    def get_automation_config_dir(self) -> Path:
        """Return the resolved config directory Path for automation settings."""
        scope = self.AUTOMATION_CONFIG_SUBDIR
        return AutomationSettingsManager.scope_dir(scope)

    def load_and_apply_automation_settings(self, filename: str = AUTO_ACTIVE_FILENAME):
        """
        Load automation settings from YAML and apply to live objects (MachineVision, ImageNameFormatter).
        If the active file is missing, fall back to default_settings.yaml, else built-ins.
        """
        scope = self.AUTOMATION_CONFIG_SUBDIR
        loaded = AutomationSettingsManager.load(scope)
        self.automation_settings = loaded
        self._apply_automation_settings(self.automation_settings)

    def _apply_automation_settings(self, settings: AutomationSettings):
        """
        Apply settings to runtime objects. Reuse existing instances if present; otherwise create them.
        This mirrors BaseCamera.apply_settings(...) calling a hardware hook.
        """
        mv = self.machine_vision
        mv.tile_size       = settings.tile_size
        mv.stride          = settings.stride
        mv.top_percent     = settings.top_percent
        mv.min_score       = settings.min_score
        mv.soft_min_score  = settings.soft_min_score
        mv.edge_left_pct   = settings.inset_left_pct
        mv.edge_top_pct    = settings.inset_top_pct
        mv.edge_right_pct  = settings.inset_right_pct
        mv.edge_bottom_pct = settings.inset_bottom_pct
        mv.scale_factor    = settings.scale_factor

        self.image_formatter.set_template(settings.image_name_template)

    def save_automation_settings(self):
        """
        Persist current automation settings to YAML in the scoped folder.
        Old version is backed up; most recent N backups are kept (per manager policy).
        """
        scope = self.AUTOMATION_CONFIG_SUBDIR
        AutomationSettingsManager.save(scope, self.automation_settings)

    def set_automation_settings(
        self,
        settings: AutomationSettings,
        persist: bool = False,
    ):
        """
        Replace entire automation settings object, apply immediately, optionally persist to disk.
        """
        self.automation_settings = settings
        self._apply_automation_settings(self.automation_settings)
        if persist:
            self.save_automation_settings()

    def update_automation_settings(
        self,
        persist: bool = False,
        **updates,
    ):
        """
        Update one or more attributes on the current automation settings, apply immediately,
        and optionally persist to disk. Unknown keys raise AttributeError (avoids silent typos).
        """
        #self.load_and_apply_automation_settings(filename=filename)

        for k, v in updates.items():
            if hasattr(self.automation_settings, k):
                setattr(self.automation_settings, k, v)
            else:
                raise AttributeError(f"Unknown automation setting '{k}'")

        self._apply_automation_settings(self.automation_settings)
        if persist:
            self.save_automation_settings()

    # ----- Defaults helpers (parity with camera) -----
    def get_automation_default_config_path(self) -> Path:
        return self.get_automation_config_dir() / AUTO_DEFAULT_FILENAME

    def write_default_automation_settings(self, settings: AutomationSettings | None = None) -> Path:
        """Write default_settings.yaml for automation (or built-ins if None)."""
        scope = self.AUTOMATION_CONFIG_SUBDIR
        return AutomationSettingsManager.write_defaults(scope, settings)

    def load_default_automation_settings(self) -> AutomationSettings:
        """
        Load defaults from default_settings.yaml (or built-ins if missing),
        apply but do NOT persist to active.
        """
        scope = self.AUTOMATION_CONFIG_SUBDIR
        defaults = AutomationSettingsManager.load_defaults(scope)
        self.set_automation_settings(defaults, persist=False)
        return defaults

    def restore_default_automation_settings(self, persist: bool = True) -> AutomationSettings:
        """
        Restore defaults into the active automation file (backup the current one),
        apply, and optionally persist.
        """
        scope = self.AUTOMATION_CONFIG_SUBDIR
        restored = AutomationSettingsManager.restore_defaults_into_active(scope)
        self.set_automation_settings(restored, persist=False)
        if persist:
            self.save_automation_settings()
        return restored


    def get_sample_position(self, index: int) -> Position:
        """
        Strictly load a sample (x,y,z) from the PRINTER config.
        Units in YAML are millimeters; we store 0.01 mm internally.
        Raises KeyError if the slot or any coordinate is missing.
        """

        entry = self.config.sample_positions[index]
        try:
            x_mm = float(entry["x"])
            y_mm = float(entry["y"])
            z_mm = float(entry["z"])
        except KeyError as e:
            missing = str(e).strip("'")
            raise KeyError(f"sample_positions[{index}] missing '{missing}'") from None

        return Position(
            x=int(x_mm * 100),
            y=int(y_mm * 100),
            z=int(z_mm * 100),
        )
    
    def get_num_slots(self)-> int:
        return len(self.config.sample_positions)

    def get_enabled_samples(self) -> List[Tuple[int, str]]:
        results: List[Tuple[int, str]] = []
        for i, row in enumerate(self.sample_list):
            toggle = row.find_child_of_type(ToggleButton)
            field  = row.find_child_of_type(TextField)

            if toggle and getattr(toggle, "is_on", False):
                name = (getattr(field, "text", "") or getattr(field, "placeholder", "") or "").strip()
                results.append((i, name))
        return results

    def status(self, msg: str, log: bool = True) -> None:
        self._handle_status(self.status_cmd(msg), log)

    # Automation
    # --- Handler --------------------------------------------------------------
    def scan_sample_bounds(self, cmd: command) -> None:
        STEP_MM  = 1.00
        Y_MAX_MM = 224.0

        def report(msg: str, log: bool = True) -> None:
            self._handle_status(self.status_cmd(msg), log)

        # Folder name to save images into (from command.value, fallback to current index)
        sample_folder = str(cmd.value).strip() if (cmd and getattr(cmd, "value", "")) else f"sample_{self.current_sample_index}"

        # --- capture start Y ---
        start_y = float(self.position.y) / 100
        start_z = float(self.position.z) / 100
        start_time = time.time()

        report(f"[SCAN_SAMPLE_BOUNDS] Start @ Y={start_y:.3f} mm")
        self.pause_point()

        # 1) Autofocus from above
        report("[SCAN_SAMPLE_BOUNDS] Running autofocus_descent_macro…")
        self.autofocus_descent_macro(cmd)
        self.pause_point()

        # --- measurement helper: color + focus counts ---
        def refine_and_measure(y_now: float) -> None:
            """
            Measure at current Y:
            - If focus is too weak (hard<10 and soft<15), skip fine_autofocus.
            - Otherwise run fine_autofocus, then capture a STILL, compute focus score,
            and save the image as: "sample_{index}/X{X} Y{Y} Z{Z} F{FOCUS}".
            - Stream color + focus counts to the live plots.
            """
            try:
                # Pre-check focus to decide whether to run fine AF
                pre = self.machine_vision.compute_focused_tiles()
                pre_hard = len(pre.get("hard", []))
                pre_soft = len(pre.get("soft", []))

                run_fine = not (pre_hard < 10 and pre_soft < 200)
                if run_fine:
                    # Do the fine AF
                    self.fine_autofocus(cmd)

                    # Immediately capture a STILL, score it, and save it with X/Y/Z/F in the filename.
                    # _af_score_still() captures a still internally, so we can save that same image.
                    focus_score = self._af_score_still()
                    try:
                        
                        # Save into the sample folder
                        filename = self.image_formatter.get_formatted_string(focus_score=focus_score)
                        self.camera.save_image(sample_folder, filename)
                        report(f"[SCAN_SAMPLE_BOUNDS] Saved image: {sample_folder}/{filename}", True)
                    except Exception as e_save:
                        report(f"[SCAN_SAMPLE_BOUNDS] Image save failed: {e_save}", True)
                else:
                    # Skip fine AF (settle briefly so measurements are stable)
                    time.sleep(0.4)

                # Read color
                r, g, b, ylum = self.machine_vision.get_average_color()

                # Compute focus tiles for reporting after optional fine AF
                all_tiles = self.machine_vision.compute_focused_tiles(filter_invalid=True)
                hard_tiles = len(all_tiles.get("hard", []))
                soft_tiles = len(all_tiles.get("soft", []))

                report(
                    f"[SCAN_SAMPLE_BOUNDS] Y={y_now:.3f} "
                    f"→ Avg(R,G,B,Y)=({r:.1f},{g:.1f},{b:.1f},{ylum:.3f})  "
                    f"Focus: hard={hard_tiles} soft={soft_tiles}  "
                    f"{'(fine AF skipped)' if not run_fine else ''}"
                )

            except Exception as e:
                report(f"[SCAN_SAMPLE_BOUNDS] Y={y_now:.3f} → measurement failed: {e}", True)

        # Make first measurement at start
        refine_and_measure(start_y)

        # 2) Sweep +Y
        y = start_y
        while y < Y_MAX_MM - 1e-9:
            y = min(y + STEP_MM, Y_MAX_MM)
            self._exec_gcode(f"G0 Y{y:.3f}")
            self.pause_point()
            refine_and_measure(y)

        # 3) Return to start **without drawing a connecting line**
        if abs(y - start_y) > 1e-9:
            self._exec_gcode(f"G0 Y{start_y:.3f} Z{start_z:.3f}")
            self.pause_point()
            report("[SCAN_SAMPLE_BOUNDS] Running autofocus_macro at start position…")
            self.autofocus_descent_macro(cmd)

        # Measure at start after autofocus_macro (with skip logic inside refine_and_measure)
        refine_and_measure(start_y)

        # 4) Sweep -Y until sample end or 0
        y = start_y
        sample_done = False
        while y > 0 + 1e-9 and not sample_done:
            y = max(y - STEP_MM, 0.0)
            self._exec_gcode(f"G0 Y{y:.3f}")
            self.pause_point()

            # --- measurement + early stop check ---
            try:
                pre = self.machine_vision.compute_focused_tiles()
                pre_hard = len(pre.get("hard", []))
                pre_soft = len(pre.get("soft", []))
                r, g, b, ylum = self.machine_vision.get_average_color()

                if pre_hard < 10 and pre_soft < 15 and ylum < 30:
                    report(f"[SCAN_SAMPLE_BOUNDS] End of sample reached at Y={y:.3f}")
                    sample_done = True
                else:
                    refine_and_measure(y)
            except Exception as e:
                report(f"[SCAN_SAMPLE_BOUNDS] Y={y:.3f} → measurement failed: {e}")


        total_time = time.time() - start_time
        report(f"[SCAN_SAMPLE_BOUNDS] Scan complete. Total time: {total_time:.2f} seconds")


    def start_scan_sample_bounds(self, folder_name: str | None = None) -> None:
        """
        Enqueue SCAN_SAMPLE_BOUNDS; if folder_name provided, it is used as command.value
        and thus determines the image save folder inside the handler.
        """
        self.enqueue_cmd(command(
            kind="SCAN_SAMPLE_BOUNDS",
            value=(folder_name or ""),
            message="Scan sample bounds",
            log=True
        ))


    def start_automation(self) -> None:
        """Home, then iterate enabled samples and scan each with progress messaging."""
        self.reset_after_stop()

        enabled = self.get_enabled_samples()  # -> List[Tuple[row_index, name]]
        total = len(enabled)

        if total == 0:
            self.status("No samples are enabled.", True)
            return

        steps: list[command] = []

        # 1) Home first

        # 2) For each enabled sample, home  -> move -> scan
        for k, (row_idx, sample_name) in enumerate(enabled, start=1):
            one_based = row_idx + 1
            pos = self.get_sample_position(one_based)

            # Percent complete (before running this sample)
            pct = int(round((k - 1) / total * 100.0))
            scan_msg = f"[{k}/{total} {pct}%] Scanning {sample_name or f'Sample {one_based}'}"
            
            # Home to get rid of error that builds up
            steps.append(self.printer_cmd("G28", message="Homing Printer. . .", log=True))

            # Move to that sample's position
            steps.append(self.printer_cmd(
                f"G0 X{pos.x/100:.2f} Y{pos.y/100:.2f} Z{pos.z/100:.2f}",
                message=f"Moving to sample {one_based} ({sample_name})",
                log=True
            ))

            # Run the SCAN_SAMPLE_BOUNDS with the sample's NAME as the value (folder)
            steps.append(command(
                kind="SCAN_SAMPLE_BOUNDS",
                value=(sample_name or f"sample_{one_based}"),
                message=scan_msg,
                log=True
            ))

        # Home the print head to signify that it's complete
        steps.append(self.printer_cmd("G28", message="Homing Printer. . .", log=True))

        steps.append(self.status_cmd(f"Scanning Complete : {total} Samples Scanned"))
        
        # 3) Wrap as a single macro so it runs as one logical unit
        macro = self.macro_cmd(
            steps,
            wait_printer=True,
            message="Automatic sample scans",
            log=True
        )

        # 4) Enqueue the macro
        self.enqueue_cmd(macro)

    def _get_range(self, start: int, end: int, step: int) -> range:
        """Get appropriate range based on start and end positions"""
        if start < end:
            return range(start, end + step, step)
        return range(start, end - step, -step)