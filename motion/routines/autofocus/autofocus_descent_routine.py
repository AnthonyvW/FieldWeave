"""
Descent-only autofocus automation routine.

Starts at the current Z position and marches downward toward the Z floor,
scoring each coarse step. A fine-polish pass then refines around the best
coarse position using still captures.

Usage::

    from common.app_context import get_app_context
    from motion.routines.autofocus_descent import AutofocusDescent

    ctx = get_app_context()
    routine = AutofocusDescent(motion=ctx.motion)
    routine.start()
    routine.wait()
"""

from __future__ import annotations

import time
from typing import Generator

from common.app_context import get_app_context
from common.logger import info, error
from motion.motion_controller_manager import MotionControllerManager
from motion.routines.automation_routine import AutomationRoutine
from motion.routines.autofocus.autofocus_utils import move_z_and_wait, capture_still_frame

_NM_PER_MM = 1_000_000


def _quantize(z_nm: int, step_nm: int) -> int:
    """Snap *z_nm* down to the nearest multiple of *step_nm*."""
    return (z_nm // step_nm) * step_nm


class AutofocusDescent(AutomationRoutine):
    """
    Descent-only autofocus: coarse downward march, then fine polish.

    The routine descends from the current Z position toward the Z floor in
    coarse steps, scoring each position via the machine-vision manager. Once
    the coarse sweep is complete (or an early-stop condition is met) a fine
    bidirectional polish is performed around the best coarse position.

    All distance parameters are in millimetres for readability at the call
    site; they are converted to nanometres internally to match the motion
    system's coordinate space. The fine step size is not a parameter — it is
    read from ``motion.settings.step_size`` at runtime so it always reflects
    the configured minimum step for the connected machine.

    Parameters
    ----------
    motion:
        Active :class:`MotionControllerManager`.
    focus_preview_threshold:
        If the baseline still-score is *below* this value the coarse sweep
        uses the faster preview scorer. Above it, stills are used throughout.
    z_floor_mm:
        Hard lower Z limit in mm. The routine will not descend past this.
    coarse_step_mm:
        Distance between coarse evaluation positions in mm.
    max_offset_mm:
        Maximum total descent from the starting Z position in mm.
    drop_stop_peak:
        If the score drops more than this below the running peak, abort the
        coarse sweep early.
    drop_stop_base:
        If the score drops more than this below the baseline *and* no position
        better than the start has been found yet, abort early.
    settle_still_s:
        Seconds to wait after the move settles before capturing a still.
    settle_preview_s:
        Seconds to wait after the move settles before scoring the preview.
    fine_allow_preview:
        When True the fine pass may use the preview scorer if the baseline
        was below *focus_preview_threshold*. Defaults to False (always stills).
    fine_no_improve_limit:
        Number of consecutive non-improving fine steps before stopping each
        climb direction.
    """

    job_name = "Autofocus (Descent)"

    def __init__(
        self,
        motion: MotionControllerManager,
        *,
        focus_preview_threshold: float = 0.9,
        z_floor_mm: float = 0.0,
        coarse_step_mm: float = 0.20,
        max_offset_mm: float = 5.60,
        drop_stop_peak: float = 0.02,
        drop_stop_base: float = 0.03,
        settle_still_s: float = 0.4,
        settle_preview_s: float = 0.4,
        fine_allow_preview: bool = False,
        fine_no_improve_limit: int = 2,
    ) -> None:
        super().__init__(motion)

        self._focus_preview_threshold = focus_preview_threshold
        self._z_floor_nm = int(round(z_floor_mm * _NM_PER_MM))
        self._coarse_step_nm = int(round(coarse_step_mm * _NM_PER_MM))
        self._max_offset_nm = int(round(max_offset_mm * _NM_PER_MM))
        self._drop_stop_peak = drop_stop_peak
        self._drop_stop_base = drop_stop_base
        self._settle_still_s = settle_still_s
        self._settle_preview_s = settle_preview_s
        self._fine_allow_preview = fine_allow_preview
        self._fine_no_improve_limit = fine_no_improve_limit

    # ------------------------------------------------------------------
    # AutomationRoutine implementation
    # ------------------------------------------------------------------

    def steps(self) -> Generator[None, None, None]:
        ctx = get_app_context()
        camera_manager = ctx.camera_manager
        mv = ctx.machine_vision

        self._set_activity("Initialising")

        if not ctx.has_camera:
            error("[AutofocusDescent] No camera available — aborting")
            return

        # Read the minimum step size from the motion config at runtime so any
        # settings reload is reflected without reconstructing the routine.
        settings = self.motion.settings
        fine_step_nm: int = settings.step_size if settings is not None else 40_000

        z_floor = self._z_floor_nm

        # ----------------------------------------------------------------
        # Scorer callables
        # ----------------------------------------------------------------

        def score_still() -> float:
            if self._settle_still_s > 0:
                time.sleep(self._settle_still_s)
            frame = capture_still_frame(camera_manager)
            if frame is None:
                return float("-inf")
            try:
                future = mv.request_focus_analysis_async(frame, frame.shape[1], frame.shape[0])
                return float(future.result(timeout=10.0).scores.peak)
            except Exception:
                return float("-inf")

        def score_preview() -> float:
            if self._settle_preview_s > 0:
                time.sleep(self._settle_preview_s)
            frame = camera_manager.copy_current_frame_to_numpy()
            if frame is None:
                return float("-inf")
            try:
                future = mv.request_focus_analysis_async(frame, frame.shape[1], frame.shape[0])
                return float(future.result(timeout=10.0).scores.peak)
            except Exception:
                return float("-inf")

        # ----------------------------------------------------------------
        # Motion / cache helpers
        # ----------------------------------------------------------------

        def move(z_nm: int) -> None:
            move_z_and_wait(self.motion, z_nm, z_floor)

        def within_env(z_nm: int) -> bool:
            return (start_nm - self._max_offset_nm) <= z_nm <= start_nm and z_nm >= z_floor

        def score_at(z_nm: int, cache: dict[int, float], scorer) -> float:
            z_nm = _quantize(z_nm, fine_step_nm)
            if z_nm < z_floor or not within_env(z_nm):
                return float("-inf")
            if z_nm in cache:
                return cache[z_nm]
            move(z_nm)
            s = scorer()
            cache[z_nm] = s
            return s

        # ----------------------------------------------------------------
        # Establish start position & baseline
        # ----------------------------------------------------------------

        start_nm = _quantize(self.motion.get_position().z, fine_step_nm)
        self._set_activity(f"Baseline  Z={start_nm / _NM_PER_MM:.3f} mm")

        scores: dict[int, float] = {}

        move(start_nm)
        baseline = score_still()
        scores[start_nm] = baseline
        best_z = start_nm
        best_s = baseline
        info(f"[AutofocusDescent] Baseline Z={start_nm / _NM_PER_MM:.3f} mm  score={baseline:.3f}")

        coarse_scorer = (
            score_preview if baseline < self._focus_preview_threshold else score_still
        )
        info(
            f"[AutofocusDescent] Coarse scorer: "
            f"{'PREVIEW' if coarse_scorer is score_preview else 'STILL'} "
            f"(baseline={baseline:.3f})"
        )

        yield  # pause/stop point: after baseline

        # ----------------------------------------------------------------
        # Coarse descent
        # ----------------------------------------------------------------

        steps_max = min(
            self._max_offset_nm // self._coarse_step_nm,
            (start_nm - z_floor) // self._coarse_step_nm,
        )
        peak_s = baseline
        total_coarse = steps_max
        self._set_progress(0, total_coarse + 1)  # +1 accounts for the fine pass

        for k in range(1, steps_max + 1):
            if self._check_stop():
                return

            target = _quantize(start_nm - k * self._coarse_step_nm, fine_step_nm)
            target = max(target, z_floor)

            self._set_status(
                f"Coarse ↓  Z={target / _NM_PER_MM:.3f} mm  ({k}/{steps_max})",
                k, total_coarse + 1,
            )

            s = score_at(target, scores, coarse_scorer)
            info(
                f"[AutofocusDescent] ↓{self._coarse_step_nm / _NM_PER_MM:.3f} mm "
                f"Z={target / _NM_PER_MM:.3f}  score={s:.3f}  Δbase={(s - baseline):+.3f}"
            )

            if s > best_s:
                best_s, best_z = s, target
            if s > peak_s:
                peak_s = s

            if best_z == start_nm and (baseline - s) >= self._drop_stop_base:
                info("[AutofocusDescent] Early stop (baseline-drop)")
                break
            if (peak_s - s) >= self._drop_stop_peak:
                info("[AutofocusDescent] Early stop (peak-drop)")
                break
            if target == z_floor:
                break

            yield  # pause/stop point: between coarse steps

        # ----------------------------------------------------------------
        # Fine polish
        # ----------------------------------------------------------------

        if self._check_stop():
            return

        if self._fine_allow_preview and baseline < self._focus_preview_threshold:
            fine_scorer = score_preview
            scorer_name = "PREVIEW"
        else:
            fine_scorer = score_still
            scorer_name = "STILL"

        info(
            f"[AutofocusDescent] Fine search using {scorer_name} "
            f"(step={fine_step_nm / _NM_PER_MM:.4f} mm)"
        )
        self._set_activity(f"Fine polish  Z≈{best_z / _NM_PER_MM:.3f} mm  ({scorer_name})")

        def _climb(start_z: int, step: int) -> tuple[int, float]:
            zt = start_z
            best_lz = start_z
            best_ls = scores.get(start_z, score_at(start_z, scores, fine_scorer))
            no_imp = 0
            while True:
                nxt = _quantize(zt + step, fine_step_nm)
                if nxt < z_floor or not within_env(nxt):
                    break
                s = score_at(nxt, scores, fine_scorer)
                info(
                    f"[AutofocusDescent] Fine {'↑' if step > 0 else '↓'}"
                    f"{abs(step) / _NM_PER_MM:.4f} mm  Z={nxt / _NM_PER_MM:.3f}  score={s:.3f}"
                )
                if s > best_ls + 1e-6:
                    best_lz, best_ls = nxt, s
                    zt = nxt
                    no_imp = 0
                else:
                    no_imp += 1
                    if no_imp >= self._fine_no_improve_limit:
                        break
                    zt = nxt
            return best_lz, best_ls

        up_z, up_s = _climb(best_z, fine_step_nm)
        down_z, down_s = _climb(best_z, -fine_step_nm)
        local_z, local_s = (up_z, up_s) if (up_s, up_z) >= (down_s, down_z) else (down_z, down_s)
        if local_s > best_s:
            best_z, best_s = local_z, local_s

        yield  # pause/stop point: after fine pass

        if self._check_stop():
            return

        move(best_z)
        self._set_status(
            f"Done  Z={best_z / _NM_PER_MM:.3f} mm  score={best_s:.3f}",
            total_coarse + 1, total_coarse + 1,
        )
        info(
            f"[AutofocusDescent] Complete: "
            f"Z={best_z / _NM_PER_MM:.3f} mm  score={best_s:.3f}  "
            f"Δbase={(best_s - baseline):+.3f}  "
            f"coarse={'PREVIEW' if coarse_scorer is score_preview else 'STILL'}  "
            f"fine={scorer_name}"
        )