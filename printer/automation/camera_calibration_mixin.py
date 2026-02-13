"""
Camera calibration and vision-guided movement for automated 3D printer control.

This module contains camera calibration and vision-based positioning methods
that can be mixed into the main AutomatedPrinter controller class.
"""

import time
import numpy as np
import cv2
from typing import Optional, Tuple

from printer.base_controller import command
from printer.models import Position


class CameraCalibrationMixin:
    """
    Mixin class containing camera calibration and vision-guided movement functionality.
    
    This class assumes it will be mixed into a controller that has:
    - self.camera (camera instance with capture_image, get_last_frame, is_taking_image)
    - self._exec_gcode(gcode, wait=False) method
    - self.status(message, log=True) method
    - self.pause_point() method that returns True if stopped
    - self.register_handler(kind, function) method
    - self.get_position() -> Position method
    - self.get_max_x/y/z() methods
    - self.enqueue_cmd(command) method
    """
    
    def _init_camera_calibration_handlers(self):
        """Register camera calibration command handlers. Call this from __init__."""
        self.register_handler("CAMERA_CALIBRATE", self._handle_camera_calibrate)
        self.register_handler("MOVE_TO_VISION_POINT", self._handle_move_to_vision_point)
        
        # Initialize calibration state
        self.M_est = None  # 2x2 estimated mapping matrix (pixels = M * world_delta)
        self.M_inv = None  # Inverse mapping (world_delta = M_inv * pixel_delta)
        self._cal_ref_pos = None  # Position where calibration was performed
        self._cal_image_width = None  # Image width used during calibration
        self._cal_image_height = None  # Image height used during calibration
        self._cal_dpi = None  # Computed DPI (dots per inch) from calibration
        
        # Calibration parameters (can be overridden)
        self._cal_move_x_ticks = 100  # 1.00mm in 0.01mm units
        self._cal_move_y_ticks = 100  # 1.00mm in 0.01mm units
        
        # Try to load saved calibration
        self._load_camera_calibration()
    
    # ========================================================================
    # Save/Load calibration methods
    # ========================================================================
    
    def _save_camera_calibration(self) -> None:
        """Save the current calibration matrix to printer config."""
        if self.M_est is None or self.M_inv is None:
            return
        
        calibration_data = {
            'M_est': self.M_est.tolist(),
            'M_inv': self.M_inv.tolist(),
            'ref_pos_x': int(self._cal_ref_pos.x) if self._cal_ref_pos else None,
            'ref_pos_y': int(self._cal_ref_pos.y) if self._cal_ref_pos else None,
            'ref_pos_z': int(self._cal_ref_pos.z) if self._cal_ref_pos else None,
            'image_width': self._cal_image_width,
            'image_height': self._cal_image_height,
            'move_x_ticks': self._cal_move_x_ticks,
            'move_y_ticks': self._cal_move_y_ticks,
            'dpi': float(self._cal_dpi) if self._cal_dpi is not None else None,
        }
        
        # Save to printer config
        self.config.camera_calibration = calibration_data
        
        # Persist to disk using the PrinterSettingsManager
        from printer.printerConfig import PrinterSettingsManager
        PrinterSettingsManager.save(self.CONFIG_SUBDIR, self.config)
        
        self.status("Camera calibration saved to config", True)
    
    def _load_camera_calibration(self) -> bool:
        """
        Load saved calibration from printer config.
        Returns True if calibration was loaded successfully.
        """
        if not hasattr(self.config, 'camera_calibration'):
            return False
        
        cal_data = self.config.camera_calibration
        if not cal_data or not isinstance(cal_data, dict):
            return False
        
        try:
            # Load matrices
            M_est_list = cal_data.get('M_est')
            M_inv_list = cal_data.get('M_inv')
            
            if M_est_list is None or M_inv_list is None:
                return False
            
            self.M_est = np.array(M_est_list, dtype=np.float64)
            self.M_inv = np.array(M_inv_list, dtype=np.float64)
            
            # Load reference position
            ref_x = cal_data.get('ref_pos_x')
            ref_y = cal_data.get('ref_pos_y')
            ref_z = cal_data.get('ref_pos_z')
            
            if ref_x is not None and ref_y is not None and ref_z is not None:
                self._cal_ref_pos = Position(x=ref_x, y=ref_y, z=ref_z)
            
            # Load image dimensions
            self._cal_image_width = cal_data.get('image_width')
            self._cal_image_height = cal_data.get('image_height')
            
            # Load calibration parameters
            self._cal_move_x_ticks = cal_data.get('move_x_ticks', 100)
            self._cal_move_y_ticks = cal_data.get('move_y_ticks', 100)
            
            # Load or calculate DPI
            self._cal_dpi = cal_data.get('dpi')
            if self._cal_dpi is None:
                # Calculate DPI if not saved
                self._calculate_dpi()
            
            dpi_str = f" (DPI: {self._cal_dpi:.1f})" if self._cal_dpi is not None else ""
            self.status(f"Camera calibration loaded from config{dpi_str}", True)
            return True
            
        except Exception as e:
            self.status(f"Failed to load camera calibration: {e}", True)
            self.M_est = None
            self.M_inv = None
            self._cal_ref_pos = None
            self._cal_image_width = None
            self._cal_image_height = None
            self._cal_dpi = None
            return False
    
    def clear_camera_calibration(self) -> None:
        """Clear the saved camera calibration from config and memory."""
        self.M_est = None
        self.M_inv = None
        self._cal_ref_pos = None
        self._cal_image_width = None
        self._cal_image_height = None
        self._cal_dpi = None
        
        # Clear from config
        self.config.camera_calibration = {}
        
        # Persist to disk
        from printer.printerConfig import PrinterSettingsManager
        PrinterSettingsManager.save(self.CONFIG_SUBDIR, self.config)
        
        self.status("Camera calibration cleared", True)
    
    # ========================================================================
    # Camera calibration helper methods
    # ========================================================================
    
    def _surface_to_gray_cv(self, arr: np.ndarray) -> np.ndarray:
        """Convert RGB numpy array to grayscale for OpenCV."""
        if arr.ndim == 2:
            return arr
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return gray
    
    def _edges_canny(self, gray_u8: np.ndarray) -> np.ndarray:
        """Compute normalized Canny edges."""
        g = cv2.GaussianBlur(gray_u8, (5, 5), 0)
        e = cv2.Canny(g, 60, 180)
        ef = e.astype(np.float32)
        ef -= ef.mean()
        ef /= (ef.std() + 1e-6)
        return ef
    
    def _phase_corr_shift(
        self,
        img_a_f32: np.ndarray,
        img_b_f32: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute phase correlation shift between two images."""
        win = cv2.createHanningWindow(
            (img_a_f32.shape[1], img_a_f32.shape[0]),
            cv2.CV_32F
        )
        (dx, dy), response = cv2.phaseCorrelate(img_a_f32, img_b_f32, win)
        return float(dx), float(dy), float(response)
    
    def _calculate_dpi(self) -> None:
        """
        Calculate DPI (dots per inch) from the calibration matrix.
        DPI represents the average resolution of the camera image.
        
        The calculation uses the calibration matrix M_est to determine how many
        pixels correspond to physical movement. We compute the average scaling
        factor from both X and Y axes and convert to DPI (pixels per inch).
        """
        if self.M_est is None:
            self._cal_dpi = None
            return
        
        try:
            # M_est maps world deltas (in 0.01mm ticks) to pixel deltas
            # M_est[0,0] and M_est[1,1] are the diagonal elements (x->px, y->py)
            # Extract pixels per tick for both axes
            px_per_tick_x = abs(self.M_est[0, 0])  # pixels per 0.01mm in X
            px_per_tick_y = abs(self.M_est[1, 1])  # pixels per 0.01mm in Y
            
            # Average the two axes
            px_per_tick_avg = (px_per_tick_x + px_per_tick_y) / 2.0
            
            # Convert to pixels per mm (1 tick = 0.01mm)
            px_per_mm = px_per_tick_avg * 100.0
            
            # Convert to DPI (1 inch = 25.4 mm)
            dpi = px_per_mm * 25.4
            
            self._cal_dpi = dpi
            
        except Exception as e:
            self.status(f"Failed to calculate DPI: {e}", False)
            self._cal_dpi = None
    
    def _capture_and_process_edges(self) -> Optional[np.ndarray]:
        """Capture a still image and return its edge map."""
        try:
            # Capture still
            self.camera.capture_image()
            while self.camera.is_taking_image:
                time.sleep(0.01)
            
            # Get frame as numpy array
            arr = self.camera.get_last_frame(prefer="still", wait_for_still=False)
            if arr is None:
                return None
            
            # Store the calibration image resolution
            if self._cal_image_height is None:
                self._cal_image_height = arr.shape[0]
                self._cal_image_width = arr.shape[1]
                self.status(
                    f"Calibration using image resolution: "
                    f"{self._cal_image_width}x{self._cal_image_height}",
                    True
                )
            
            # Convert to grayscale and compute edges
            gray = self._surface_to_gray_cv(arr)
            edges = self._edges_canny(gray)
            return edges
        except Exception as e:
            self.status(f"Edge capture failed: {e}", True)
            return None
    
    # ========================================================================
    # Main calibration routine
    # ========================================================================
    
    def _handle_camera_calibrate(self, cmd: command) -> None:
        """
        Run the calibration routine to determine the mapping between 
        image pixels and world coordinates using phase correlation.
        """
        self.status("Starting camera calibration...", True)
        
        # Reset calibration state
        self.M_est = None
        self.M_inv = None
        self._cal_image_width = None
        self._cal_image_height = None
        
        # Allow pausing/stopping
        if self.pause_point():
            self.status("Calibration cancelled.", True)
            return
        
        # Step 1: Capture base image at current position
        self.status("Capturing base image...", True)
        self._exec_gcode("M400", wait=True)
        time.sleep(0.6)  # Settle time
        
        cal_base_pos = self.get_position()
        edges_base = self._capture_and_process_edges()
        
        if edges_base is None:
            self.status("Failed to capture base image.", True)
            return
        
        if self.pause_point():
            self.status("Calibration cancelled.", True)
            return
        
        # Step 2: Move +X and capture
        dx_mm = self._cal_move_x_ticks / 100.0
        self.status(f"Moving +X by {dx_mm:.2f}mm...", True)
        self._exec_gcode(f"G91", wait=True)  # Relative mode
        self._exec_gcode(f"G0 X{dx_mm:.2f}", wait=True)
        self._exec_gcode(f"G90", wait=True)  # Back to absolute
        time.sleep(0.6)
        
        edges_x = self._capture_and_process_edges()
        if edges_x is None:
            self.status("Failed to capture +X image.", True)
            return
        
        # Compute phase correlation for +X move
        dpx_x, dpy_x, resp_x = self._phase_corr_shift(edges_base, edges_x)
        self.status(
            f"+X move: pixel shift=({dpx_x:.2f}, {dpy_x:.2f}), "
            f"response={resp_x:.3f}",
            True
        )
        
        if self.pause_point():
            self.status("Calibration cancelled.", True)
            return
        
        # Step 3: Return to base, then move +Y and capture
        self.status("Returning to base position...", True)
        self._exec_gcode(
            f"G0 X{cal_base_pos.x/100:.2f} Y{cal_base_pos.y/100:.2f}",
            wait=True
        )
        time.sleep(0.6)
        
        dy_mm = self._cal_move_y_ticks / 100.0
        self.status(f"Moving +Y by {dy_mm:.2f}mm...", True)
        self._exec_gcode(f"G91", wait=True)
        self._exec_gcode(f"G0 Y{dy_mm:.2f}", wait=True)
        self._exec_gcode(f"G90", wait=True)
        time.sleep(0.6)
        
        edges_y = self._capture_and_process_edges()
        if edges_y is None:
            self.status("Failed to capture +Y image.", True)
            return
        
        # Compute phase correlation for +Y move
        dpx_y, dpy_y, resp_y = self._phase_corr_shift(edges_base, edges_y)
        self.status(
            f"+Y move: pixel shift=({dpx_y:.2f}, {dpy_y:.2f}), "
            f"response={resp_y:.3f}",
            True
        )
        
        # Step 4: Return to base
        self.status("Returning to base position...", True)
        self._exec_gcode(
            f"G0 X{cal_base_pos.x/100:.2f} Y{cal_base_pos.y/100:.2f}",
            wait=True
        )
        
        # Step 5: Build calibration matrix
        # M * [dx_world, dy_world] = [dpx_pixel, dpy_pixel]
        # We have two observations:
        #   M * [dx_ticks, 0] = [dpx_x, dpy_x]
        #   M * [0, dy_ticks] = [dpx_y, dpy_y]
        
        world_x = np.array([[self._cal_move_x_ticks], [0.0]])
        world_y = np.array([[0.0], [self._cal_move_y_ticks]])
        pixel_x = np.array([[dpx_x], [dpy_x]])
        pixel_y = np.array([[dpx_y], [dpy_y]])
        
        # M = [pixel_x, pixel_y] * [world_x, world_y]^-1
        world_mat = np.hstack([world_x, world_y])
        pixel_mat = np.hstack([pixel_x, pixel_y])
        
        try:
            world_inv = np.linalg.inv(world_mat)
            self.M_est = pixel_mat @ world_inv
            self.M_inv = np.linalg.inv(self.M_est)
            
            # Store calibration reference position (center of image at cal_base_pos)
            self._cal_ref_pos = cal_base_pos
            
            # Calculate DPI from the calibration matrix
            self._calculate_dpi()
            
            dpi_str = f", DPI: {self._cal_dpi:.1f}" if self._cal_dpi is not None else ""
            self.status(
                f"Calibration complete. Matrix M_est:\n{self.M_est}\n"
                f"Inverse M_inv:\n{self.M_inv}{dpi_str}",
                True
            )
            
            # Save calibration to config
            self._save_camera_calibration()
            
        except np.linalg.LinAlgError:
            self.status("Calibration failed: singular matrix.", True)
            return
    
    # ========================================================================
    # Vision-guided movement
    # ========================================================================
    
    def _pixel_to_world_delta(
        self,
        pixel_x: float,
        pixel_y: float,
        image_center_x: Optional[float] = None,
        image_center_y: Optional[float] = None
    ) -> Optional[Tuple[float, float]]:
        """
        Convert pixel coordinates to world coordinate delta from calibration reference.
        
        Args:
            pixel_x: X coordinate in image pixels
            pixel_y: Y coordinate in image pixels
            image_center_x: Center X of image (defaults to _cal_image_width/2)
            image_center_y: Center Y of image (defaults to _cal_image_height/2)
            
        Returns:
            (dx_ticks, dy_ticks) relative to calibration reference position,
            or None if calibration is not available
        """
        if self.M_inv is None:
            return None
        
        if image_center_x is None:
            image_center_x = (self._cal_image_width or 0) / 2.0
        if image_center_y is None:
            image_center_y = (self._cal_image_height or 0) / 2.0
        
        # Pixel delta from image center
        pixel_delta = np.array([[pixel_x - image_center_x], [pixel_y - image_center_y]])
        
        # Convert to world delta
        world_delta = self.M_inv @ pixel_delta
        
        # NOTE: We negate both X and Y to convert from image coordinates to stage coordinates.
        # Image coords: origin at top-left, X increases right, Y increases down
        # Stage coords: X and Y both increase in positive directions
        # The calibration matrix M is built to map stage deltas to pixel deltas,
        # so M_inv maps pixel deltas to stage deltas, but we need to flip signs
        # to account for the image coordinate system.
        dx_ticks = -float(world_delta[0, 0])
        dy_ticks = -float(world_delta[1, 0])
        
        return dx_ticks, dy_ticks
    
    def _handle_move_to_vision_point(self, cmd: command) -> None:
        """
        Move to a position specified by vision coordinates.
        
        cmd.value should be a dict with:
            - 'pixel_x': X coordinate in image
            - 'pixel_y': Y coordinate in image
            - 'relative': bool, if True move relative to current position,
                         if False move relative to calibration reference
        """
        if self.M_inv is None:
            self.status("Cannot move: calibration required first", True)
            return
        
        try:
            params = cmd.value
            pixel_x = float(params['pixel_x'])
            pixel_y = float(params['pixel_y'])
            relative = params.get('relative', True)
        except (TypeError, KeyError, ValueError) as e:
            self.status(f"Invalid MOVE_TO_VISION_POINT parameters: {e}", True)
            return
        
        # Convert pixel coords to world delta
        result = self._pixel_to_world_delta(pixel_x, pixel_y)
        if result is None:
            self.status("Pixel-to-world conversion failed", True)
            return
        
        dx_ticks, dy_ticks = result
        
        # Determine target position
        if relative:
            # Move relative to current position
            current_pos = self.get_position()
            new_x_ticks = current_pos.x + int(round(dx_ticks))
            new_y_ticks = current_pos.y + int(round(dy_ticks))
        else:
            # Move relative to calibration reference
            if self._cal_ref_pos is None:
                self.status("No calibration reference position available", True)
                return
            new_x_ticks = self._cal_ref_pos.x + int(round(dx_ticks))
            new_y_ticks = self._cal_ref_pos.y + int(round(dy_ticks))
        
        # Convert to mm
        new_x_mm = new_x_ticks / 100.0
        new_y_mm = new_y_ticks / 100.0
        
        # Bounds check
        max_x = self.get_max_x()
        max_y = self.get_max_y()
        
        if not (0 <= new_x_mm <= max_x and 0 <= new_y_mm <= max_y):
            self.status(
                f"Vision target out of bounds: ({new_x_mm:.2f}, {new_y_mm:.2f})",
                True
            )
            return
        
        # Execute move
        self._exec_gcode(
            f"G0 X{new_x_mm:.2f} Y{new_y_mm:.2f}",
            wait=True,
            message=f"Moving to vision point: X={new_x_mm:.2f}, Y={new_y_mm:.2f}",
            log=True
        )
    
    # ========================================================================
    # Public convenience methods
    # ========================================================================
    
    def set_calibration_moves(self, x_ticks: int, y_ticks: int) -> None:
        """
        Set the calibration move distances in ticks (0.01mm units).
        
        Args:
            x_ticks: Distance to move in X during calibration
            y_ticks: Distance to move in Y during calibration
        """
        self._cal_move_x_ticks = x_ticks
        self._cal_move_y_ticks = y_ticks
    
    def start_camera_calibration(self) -> None:
        """Enqueue a camera calibration command."""
        self.reset_after_stop()
        self.enqueue_cmd(command(
            kind="CAMERA_CALIBRATE",
            value="",
            message="Starting camera calibration",
            log=True
        ))
    
    def go_to_calibration_pattern(self, position: Optional[Position] = None) -> None:
        """
        Move to a known calibration pattern position.
        
        This is useful for setting up before running calibration, allowing you to
        position the camera over a calibration target (e.g., a grid or known feature).
        
        Args:
            position: Position to move to (in 0.01mm ticks).
                     If None, looks for 'calibration_pattern_position' in printer config.
        """
        if position is None:
            # Try to load from printer config (same pattern as get_sample_position)
            if hasattr(self.config, 'calibration_pattern_position'):
                try:
                    entry = self.config.calibration_pattern_position
                    x_mm = float(entry["x"])
                    y_mm = float(entry["y"])
                    z_mm = float(entry["z"])
                    position = Position(
                        x=int(x_mm * 100),
                        y=int(y_mm * 100),
                        z=int(z_mm * 100),
                    )
                except (KeyError, ValueError, TypeError) as e:
                    self.status(
                        f"Invalid calibration_pattern_position in printer config: {e}",
                        True
                    )
                    return
            else:
                self.status(
                    "No calibration pattern position provided or configured in printer config",
                    True
                )
                return
        
        # Move to the position
        x_mm = position.x / 100.0
        y_mm = position.y / 100.0
        z_mm = position.z / 100.0
        
        self.enqueue_printer(
            f"G0 X{x_mm:.2f} Y{y_mm:.2f} Z{z_mm:.2f}",
            message=f"Moving to calibration pattern at X={x_mm:.2f}, Y={y_mm:.2f}, Z={z_mm:.2f}",
            log=True
        )
    
    def move_to_vision_point(
        self,
        pixel_x: float,
        pixel_y: float,
        relative: bool = True
    ) -> None:
        """
        Move to a point identified by vision coordinates.
        
        Args:
            pixel_x: X coordinate in image pixels
            pixel_y: Y coordinate in image pixels
            relative: If True, move relative to current position;
                     if False, move relative to calibration reference
        """
        self.enqueue_cmd(command(
            kind="MOVE_TO_VISION_POINT",
            value={
                'pixel_x': pixel_x,
                'pixel_y': pixel_y,
                'relative': relative
            },
            message=f"Moving to vision point ({pixel_x:.1f}, {pixel_y:.1f})",
            log=True
        ))
    
    def get_calibration_status(self) -> dict:
        """
        Get current calibration status.
        
        Returns:
            Dict with calibration state information
        """
        return {
            'calibrated': self.M_inv is not None,
            'image_width': self._cal_image_width,
            'image_height': self._cal_image_height,
            'reference_position': self._cal_ref_pos,
            'matrix_M': self.M_est.tolist() if self.M_est is not None else None,
            'matrix_M_inv': self.M_inv.tolist() if self.M_inv is not None else None,
            'dpi': self._cal_dpi,
        }
    
    def is_calibrated(self) -> bool:
        """
        Check if camera is calibrated and ready for vision-guided movement.
        
        Returns:
            True if calibration matrices are loaded and valid
        """
        return (self.M_est is not None and 
                self.M_inv is not None and 
                self._cal_ref_pos is not None)