"""
Z-stack scan automation routine.

Moves the stage between two Z positions, capturing an image at each step.
Images are saved with X / Y / Z position metadata embedded, and the file
name is the Z position in nanometres.

Usage::

    from common.app_context import get_app_context
    from motion.automations.z_stack_scan import ZStackScan

    ctx = get_app_context()
    routine = ZStackScan(
        motion=ctx.motion,
        z_start_nm=0,
        z_end_nm=5_000_000,    # 5 mm
        step_nm=500_000,        # 0.5 mm steps
        output_folder="/data/scans/run1",
    )
    routine.start()
    # …later…
    routine.pause()
    routine.resume()
    routine.stop()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator

from common.app_context import get_app_context
from common.logger import info, warning, error
from motion.motion_controller_manager import MotionControllerManager
from motion.models import Position

from motion.routines.automation_routine import AutomationRoutine

_NM_PER_MM = 1_000_000


class ZStackScan(AutomationRoutine):
    """
    Capture images at evenly-spaced Z positions between two Z locations.

    The stage travels to whichever of *z_start_nm* / *z_end_nm* is closest
    to the current Z position first, then steps toward the other end,
    capturing one image per step.

    Parameters
    ----------
    motion:
        Active :class:`MotionControllerManager`.
    z_start_nm:
        One end of the Z range, in nanometres.
    z_end_nm:
        The other end of the Z range, in nanometres.
    step_nm:
        Distance between capture positions, in nanometres.  Must be > 0.
    output_folder:
        Directory in which captured images are saved.  Created automatically
        if it does not exist.
    capture_timeout_ms:
        How long (ms) to wait for each image capture to complete.
    """

    def __init__(
        self,
        motion: MotionControllerManager,
        z_start_nm: int,
        z_end_nm: int,
        step_nm: int,
        output_folder: str | Path,
        capture_timeout_ms: int = 5000,
    ) -> None:
        super().__init__(motion)

        if step_nm <= 0:
            raise ValueError(f"step_nm must be positive, got {step_nm}")
        if z_start_nm == z_end_nm:
            raise ValueError("z_start_nm and z_end_nm must be different")

        self._z_start_nm = z_start_nm
        self._z_end_nm = z_end_nm
        self._step_nm = step_nm
        self._output_folder = Path(output_folder)
        self._capture_timeout_ms = capture_timeout_ms

    # ------------------------------------------------------------------
    # AutomationRoutine implementation
    # ------------------------------------------------------------------

    def steps(self) -> Generator[None, None, None]:
        ctx = get_app_context()
        camera = ctx.camera

        if camera is None:
            error("[ZStackScan] No camera available — aborting")
            return

        # Ensure output directory exists
        self._output_folder.mkdir(parents=True, exist_ok=True)

        # Determine travel direction: go to closest Z first
        current_z = self.motion.get_position().z
        if abs(current_z - self._z_start_nm) <= abs(current_z - self._z_end_nm):
            z_near = self._z_start_nm
            z_far = self._z_end_nm
        else:
            z_near = self._z_end_nm
            z_far = self._z_start_nm

        direction = 1 if z_far > z_near else -1

        # Build list of Z positions to visit
        z_positions: list[int] = []
        z = z_near
        while (direction == 1 and z <= z_far) or (direction == -1 and z >= z_far):
            z_positions.append(z)
            z += direction * self._step_nm

        total = len(z_positions)
        info(f"[ZStackScan] {total} positions from {z_near} nm to {z_far} nm, step {self._step_nm} nm")
        info(f"[ZStackScan] Output folder: {self._output_folder}")

        scan_start_time = time.monotonic()
        capture_times: list[float] = []
        captured_positions: list[int] = []

        for idx, target_z_nm in enumerate(z_positions):
            if self._check_stop():
                break

            target_z_mm = target_z_nm / _NM_PER_MM
            current_pos = self.motion.get_position()
            target_pos = Position(
                x=current_pos.x,
                y=current_pos.y,
                z=target_z_nm,
            )

            info(f"[ZStackScan] Step {idx + 1}/{total}: moving to Z={target_z_mm:.6f} mm")
            self.motion.move_to_position(target_pos)

            # Wait for the move to finish by polling position
            # We yield between polling so the pause/stop checks stay responsive
            yield  # pause/stop point: after enqueuing the move

            if self._check_stop():
                break

            # Poll until Z reaches the target (within 1 µm tolerance)
            _tolerance_nm = 1_000
            _poll_interval = 0.05
            _move_timeout = 30.0
            elapsed = 0.0
            while elapsed < _move_timeout:
                actual_z = self.motion.get_position().z
                if abs(actual_z - target_z_nm) <= _tolerance_nm:
                    break
                time.sleep(_poll_interval)
                elapsed += _poll_interval
                if self._check_stop():
                    break
            else:
                warning(
                    f"[ZStackScan] Timed out waiting for Z to reach {target_z_nm} nm "
                    f"(actual: {self.motion.get_position().z} nm)"
                )

            if self._check_stop():
                break

            yield  # pause/stop point: after settling at position

            # Capture image
            actual_pos = self.motion.get_position()
            filepath = self._output_folder / f"{actual_pos.z}.jpg"

            info(f"[ZStackScan] Capturing image: {filepath}")

            capture_success: bool | None = None
            capture_error: Exception | None = None

            capture_done = __import__("threading").Event()

            def _on_complete(success: bool, result, _done=capture_done) -> None:
                nonlocal capture_success, capture_error
                capture_success = success
                if not success:
                    capture_error = result if isinstance(result, Exception) else Exception(str(result))
                _done.set()

            capture_start = time.monotonic()
            camera.capture_and_save_still(
                filepath=filepath,
                resolution_index=0,
                additional_metadata={
                    "x_position_nm": actual_pos.x,
                    "y_position_nm": actual_pos.y,
                    "z_position_nm": actual_pos.z,
                    "x_position_mm": actual_pos.x / _NM_PER_MM,
                    "y_position_mm": actual_pos.y / _NM_PER_MM,
                    "z_position_mm": actual_pos.z / _NM_PER_MM,
                    "source": "z_stack_scan",
                    "step_index": idx,
                    "total_steps": total,
                },
                timeout_ms=self._capture_timeout_ms,
                on_complete=_on_complete,
                wait=True,
            )

            file_exists = filepath.exists()

            if not capture_success and not file_exists:
                warning(f"[ZStackScan] Capture failed at Z={actual_pos.z} nm: {capture_error}")
            else:
                if not capture_success and file_exists:
                    warning(
                        f"[ZStackScan] Capture callback reported failure at Z={actual_pos.z} nm "
                        f"but file exists on disk — treating as success: {capture_error}"
                    )
                capture_times.append(time.monotonic() - capture_start)
                captured_positions.append(actual_pos.z)
                info(f"[ZStackScan] Saved {filepath}")

            yield  # pause/stop point: after capture

        total_elapsed = time.monotonic() - scan_start_time
        n_captured = len(capture_times)

        info("[ZStackScan] Scan complete")
        info(f"[ZStackScan] Total duration:    {total_elapsed:.3f} s")
        info(f"[ZStackScan] Images captured:   {n_captured} / {total}")
        info(
            f"[ZStackScan] Z range:           {z_near / _NM_PER_MM:.6f} mm"
            f" to {z_far / _NM_PER_MM:.6f} mm"
            f"  ({(z_far - z_near) / _NM_PER_MM:.6f} mm span)"
        )
        info(f"[ZStackScan] Step size:         {self._step_nm / _NM_PER_MM:.6f} mm  ({self._step_nm} nm)")

        if captured_positions:
            positions_mm = ", ".join(f"{z / _NM_PER_MM:.6f}" for z in captured_positions)
            info(f"[ZStackScan] Captured at (mm):  [{positions_mm}]")

        if capture_times:
            info(
                f"[ZStackScan] Capture time (s):  "
                f"min={min(capture_times):.3f}  "
                f"max={max(capture_times):.3f}  "
                f"avg={sum(capture_times) / len(capture_times):.3f}"
            )