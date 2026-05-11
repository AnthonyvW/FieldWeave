"""
Z-stack scan automation routine.

Moves the stage between two Z positions, capturing an image at each step.
Images are saved with X / Y / Z position metadata embedded, and the file
name is the Z position in nanometres.

If a :class:`FocusStackRoutineConfig` is supplied the routine will automatically
launch a focus stack via the application's :class:`PostProcessingManager` once
all images have been captured (or, in live mode, as images arrive).

Two focus stacking modes are available via the *live* parameter:

- ``live=False`` (default): a :class:`QueuedFocusStackRoutine` is started after
  all frames have been saved.  Faster for large batches on capable hardware.
- ``live=True``: a :class:`StreamingFocusStackRoutine` is started before
  capture begins and receives each frame as it is taken.  Alignment and culling
  run concurrently with acquisition, reducing total wall time.

The stacked output is written to ``<output_folder>/stacked.<ext>`` where the
extension comes from ``focus_stack_config.output_extension``.

Usage::

    from common.app_context import get_app_context
    from motion.automations.z_stack_scan import ZStackScan
    from post_processing.routines.focus_stack_routine import FocusStackRoutineConfig

    ctx = get_app_context()
    cfg = FocusStackRoutineConfig()

    routine = ZStackScan(
        motion=ctx.motion,
        z_start_nm=0,
        z_end_nm=5_000_000,
        step_nm=500_000,
        output_folder="/data/scans/run1",
        focus_stack_config=cfg,
        live=False,
    )
    routine.start()
"""

from __future__ import annotations

import threading

import numpy as np
import time
from pathlib import Path
from typing import Callable, Generator, TYPE_CHECKING

from common.app_context import get_app_context
from common.logger import info, warning, error
from motion.motion_controller_manager import MotionControllerManager
from motion.models import Position

from motion.routines.automation_routine import AutomationRoutine

if TYPE_CHECKING:
    from post_processing.post_processing_manager import PostProcessingManager
    from post_processing.routines.focus_stack_routine import FocusStackRoutineConfig
    from focusweave.streaming_stack import PreviewCallback

_NM_PER_MM = 1_000_000


class ZStackScan(AutomationRoutine):
    """
    Capture images at evenly-spaced Z positions between two Z locations.

    The stage travels to whichever of *z_start_nm* / *z_end_nm* is closest
    to the current Z position first, then steps toward the other end,
    capturing one image per step.

    If *approach_distance_nm* is non-zero, the stage will overshoot the near
    end by that amount before returning to it. This eliminates backlash and
    direction-change wobble at the start of the sweep.

    If *focus_stack_config* is provided, a focus stack is performed after (or
    during) capture depending on *live*:

    - ``live=False``: a :class:`QueuedFocusStackRoutine` is launched after all
      frames are saved to disk.
    - ``live=True``: a :class:`StreamingFocusStackRoutine` is launched before
      capture begins and each captured frame is fed to it as it arrives.
      Alignment runs concurrently with acquisition.

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
    focus_stack_config:
        When supplied, a focus-stack post-processing job is run after capture.
        Pass ``None`` (the default) to skip post-processing.
    live:
        When True and *focus_stack_config* is set, use streaming focus stacking
        so alignment runs concurrently with acquisition.  When False (default),
        use the queued pipeline which runs after all frames are saved.

    Settings read at runtime
    ------------------------
    ``ctx.settings.motion.automation.capture_timeout_ms``:
        How long (ms) to wait for each image capture to complete.
    ``ctx.settings.motion.z_stack_scan.approach_distance_nm``:
        Before the scan begins, the stage overshoots the near end by this
        distance in the scan direction, then returns to it.  0 disables
        the approach move.
    """

    job_name = "Z-Stack Scan"

    def __init__(
        self,
        motion: MotionControllerManager,
        z_start_nm: int,
        z_end_nm: int,
        step_nm: int,
        output_folder: str | Path,
        focus_stack_config: FocusStackRoutineConfig | None = None,
        live: bool = False,
        on_preview_frame: Callable[[np.ndarray, int], None] | None = None,
    ) -> None:
        super().__init__(motion)

        if step_nm <= 0:
            raise ValueError(f"step_nm must be positive, got {step_nm}")
        self._z_start_nm = z_start_nm
        self._z_end_nm = z_end_nm
        self._step_nm = step_nm
        self._output_folder = Path(output_folder)
        self._focus_stack_config = focus_stack_config
        self._live = live
        self._on_preview_frame = on_preview_frame

    # ------------------------------------------------------------------
    # AutomationRoutine implementation
    # ------------------------------------------------------------------

    def steps(self) -> Generator[None, None, None]:
        ctx = get_app_context()
        camera = ctx.camera
        post_processing = ctx.post_processing

        motion_settings = ctx.motion.settings
        capture_timeout_ms = motion_settings.automation.capture_timeout_ms
        approach_distance_nm = motion_settings.z_stack_scan.approach_distance_nm

        self._set_activity("Initialising")

        if camera is None:
            error("[ZStackScan] No camera available — aborting")
            return

        self._output_folder.mkdir(parents=True, exist_ok=True)

        current_z = self.motion.get_position().z
        if abs(current_z - self._z_start_nm) <= abs(current_z - self._z_end_nm):
            z_near = self._z_start_nm
            z_far = self._z_end_nm
        else:
            z_near = self._z_end_nm
            z_far = self._z_start_nm

        z_positions: list[int] = []
        if z_near == z_far:
            z_positions = [z_near]
        else:
            direction = 1 if z_far > z_near else -1
            z = z_near
            while (direction == 1 and z <= z_far) or (direction == -1 and z >= z_far):
                z_positions.append(z)
                z += direction * self._step_nm

        total = len(z_positions)
        info(f"[ZStackScan] {total} positions from {z_near} nm to {z_far} nm, step {self._step_nm} nm")
        info(f"[ZStackScan] Output folder: {self._output_folder}")

        if approach_distance_nm > 0:
            approach_z_nm = z_near - direction * approach_distance_nm
            approach_z_mm = approach_z_nm / _NM_PER_MM
            self._set_activity(f"Approaching  —  Z={approach_z_mm:.3f} mm")
            info(f"[ZStackScan] Approach overshoot to Z={approach_z_mm:.6f} mm")
            current_pos = self.motion.get_position()
            self.motion.move_to_position(
                Position(x=current_pos.x, y=current_pos.y, z=approach_z_nm), wait=True
            )

            yield  # pause/stop point: after overshoot

            if self._check_stop():
                return

            near_z_mm = z_near / _NM_PER_MM
            self._set_activity(f"Returning to start  —  Z={near_z_mm:.3f} mm")
            info(f"[ZStackScan] Returning to Z={near_z_mm:.6f} mm")
            current_pos = self.motion.get_position()
            self.motion.move_to_position(
                Position(x=current_pos.x, y=current_pos.y, z=z_near), wait=True
            )

            yield  # pause/stop point: after return

            if self._check_stop():
                return

        self._set_progress(0, total)

        total_start_time = time.monotonic()
        scan_start_time = time.monotonic()
        capture_times: list[float] = []
        captured_positions: list[int] = []

        pending_saves: list[tuple[int, Path, threading.Event, list[bool]]] = []

        use_live = self._live and self._focus_stack_config is not None
        streaming_routine = None

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

            self._set_status(f"Step {idx + 1}/{total}  —  Z={target_z_mm:.3f} mm", idx, total)
            info(f"[ZStackScan] Step {idx + 1}/{total}: moving to Z={target_z_mm:.6f} mm")
            self.motion.move_to_position(target_pos, wait=True)

            yield  # pause/stop point: after move

            if self._check_stop():
                break

            actual_pos = self.motion.get_position()
            filepath = self._output_folder / f"{actual_pos.z}.jpg"

            info(f"[ZStackScan] Capturing image: {filepath}")

            save_done = threading.Event()
            success_cell: list[bool] = [False]
            captured_rgb: list[np.ndarray | None] = [None]

            def _on_complete(
                success: bool,
                _done=save_done,
                _cell=success_cell,
            ) -> None:
                _cell[0] = success
                _done.set()

            def _on_image(
                image: np.ndarray,
                _rgb=captured_rgb,
            ) -> None:
                _rgb[0] = image

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
                timeout_ms=capture_timeout_ms,
                on_complete=_on_complete,
                on_image=_on_image if use_live else None,
                wait=True,
            )

            capture_times.append(time.monotonic() - capture_start)
            captured_positions.append(actual_pos.z)
            pending_saves.append((actual_pos.z, filepath, save_done, success_cell))
            self._set_progress(idx + 1, total)

            if use_live:
                save_done.wait(timeout=capture_timeout_ms / 1000.0)
                rgb = captured_rgb[0]
                if rgb is not None:
                    if streaming_routine is None:
                        h, w = rgb.shape[:2]
                        streaming_routine = self._start_streaming_focus_stack(
                            post_processing, (w, h)
                        )
                    if streaming_routine is not None:
                        streaming_routine.add_image(rgb)
                else:
                    warning(f"[ZStackScan] No RGB data for frame {idx + 1} — skipping live feed")

            yield  # pause/stop point: after capture

        scan_elapsed = time.monotonic() - scan_start_time

        save_timeout_s = capture_timeout_ms / 1000.0
        n_saved = 0
        for z_nm, filepath, save_done, success_cell in pending_saves:
            save_done.wait(timeout=save_timeout_s)
            if success_cell[0]:
                n_saved += 1
                info(f"[ZStackScan] Saved {filepath}")
            else:
                warning(f"[ZStackScan] Save failed at Z={z_nm} nm")

        n_captured = len(capture_times)

        info("[ZStackScan] Scan complete")
        info(f"[ZStackScan] Scan duration:     {scan_elapsed:.3f} s")
        info(f"[ZStackScan] Images captured:   {n_captured} / {total}")
        info(f"[ZStackScan] Images saved:      {n_saved} / {n_captured}")
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

        if use_live and streaming_routine is not None:
            if self._check_stop():
                post_processing.stop_routine()
                return
            self._run_streaming_focus_stack_finish(streaming_routine, post_processing, total_start_time)
        elif self._focus_stack_config is not None and n_saved > 0:
            self._run_queued_focus_stack(post_processing, total_start_time)
        else:
            info(f"[ZStackScan] Total time:        {time.monotonic() - total_start_time:.3f} s")

    # ------------------------------------------------------------------
    # Live (streaming) focus stack helpers
    # ------------------------------------------------------------------

    def _start_streaming_focus_stack(
        self,
        post_processing: PostProcessingManager | None,
        reference_size: tuple[int, int],
    ):
        if post_processing is None:
            error("[ZStackScan] No post_processing manager available — skipping live focus stack")
            return None

        from post_processing.routines.focus_stack_routine import StreamingFocusStackRoutine

        cfg = self._focus_stack_config
        ext = cfg.output_extension
        output_path = str(self._output_folder / f"stacked.{ext}")

        info(f"[ZStackScan] Starting streaming focus stack — output: {output_path}")

        routine = StreamingFocusStackRoutine(
            settings=post_processing.settings,
            output_path=output_path,
            reference_size=reference_size,
            config=cfg,
            progress_start=50,
            progress_end=100,
        )
        if self._on_preview_frame is not None:
            routine.on_preview = self._on_preview_frame
        post_processing.start_routine(routine)
        return routine

    def _run_streaming_focus_stack_finish(
        self,
        routine,
        post_processing: PostProcessingManager,
        total_start_time: float,
    ) -> None:

        focus_stack_start_time = time.monotonic()
        info("[ZStackScan] Signalling streaming focus stack to finalise")
        routine.finish()

        while routine.is_running:
            if self._check_stop():
                post_processing.stop_routine()
                return
            self._set_status(
                f"Focus stacking — {routine.activity}",
                routine.progress_current,
                routine.progress_total,
            )
            time.sleep(0.25)

        routine.wait()

        info(f"[ZStackScan] Focus stack time:  {time.monotonic() - focus_stack_start_time:.3f} s")
        info(f"[ZStackScan] Total time:        {time.monotonic() - total_start_time:.3f} s")

    # ------------------------------------------------------------------
    # Queued focus stack helper
    # ------------------------------------------------------------------

    def _run_queued_focus_stack(
        self,
        post_processing: PostProcessingManager | None,
        total_start_time: float,
    ) -> None:
        if post_processing is None:
            error("[ZStackScan] No post_processing manager available — skipping focus stack")
            info(f"[ZStackScan] Total time:        {time.monotonic() - total_start_time:.3f} s")
            return

        from post_processing.routines.focus_stack_routine import QueuedFocusStackRoutine

        cfg = self._focus_stack_config
        ext = cfg.output_extension
        output_path = str(self._output_folder / f"stacked.{ext}")

        info(f"[ZStackScan] Starting queued focus stack — output: {output_path}")
        focus_stack_start_time = time.monotonic()

        focus_routine = QueuedFocusStackRoutine(
            settings=post_processing.settings,
            input_folder=str(self._output_folder),
            output_path=output_path,
            config=cfg,
            progress_start=50,
            progress_end=100,
        )
        post_processing.start_routine(focus_routine)

        while focus_routine.is_running:
            if self._check_stop():
                post_processing.stop_routine()
                return
            self._set_status(
                f"Focus stacking — {focus_routine.activity}",
                focus_routine.progress_current,
                focus_routine.progress_total,
            )
            time.sleep(0.25)

        focus_routine.wait()

        info(f"[ZStackScan] Focus stack time:  {time.monotonic() - focus_stack_start_time:.3f} s")
        info(f"[ZStackScan] Total time:        {time.monotonic() - total_start_time:.3f} s")