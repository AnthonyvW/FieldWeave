"""
Bidirectional autofocus automation routine.

Performs a coarse alternating sweep outward from the current Z position
(with bias toward the side that shows early improvement), a refine march
in the better direction, and a final fine-step polish pass.

Usage::

    from common.app_context import get_app_context
    from motion.routines.autofocus import Autofocus

    ctx = get_app_context()
    routine = Autofocus(motion=ctx.motion)
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


class Autofocus(AutomationRoutine):
    """
    Bidirectional autofocus: alternating coarse sweep → refine march → fine polish.

    Starting from the current Z, the routine probes both above and below in
    coarse steps, biasing toward whichever side shows early improvement. After
    the coarse sweep a refine march continues in the best direction, and
    finally a fine bidirectional polish locks onto the peak.

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
        Baseline still-scores below this use the faster preview scorer for
        the coarse sweep.
    coarse_improve_thresh:
        Score improvement over baseline required to lock in a bias direction.
    coarse_drop_stop_peak:
        Abort the coarse sweep when the score drops this much below the
        running peak.
    coarse_drop_stop_base:
        Abort when the score drops this much below the baseline and no
        improvement over the start has been found yet.
    z_floor_mm:
        Hard lower Z limit in mm.
    coarse_step_mm:
        Coarse evaluation step size in mm.
    refine_step_mm:
        Step size for the refine march between coarse and fine passes in mm.
    max_offset_mm:
        Maximum distance from the starting Z in either direction in mm.
    settle_still_s:
        Settle time after a move before a still capture.
    settle_preview_s:
        Settle time after a move before a preview score.
    fine_no_improve_limit:
        Consecutive non-improving steps before stopping a fine-climb direction.
    """

    job_name = "Autofocus"

    def __init__(
        self,
        motion: MotionControllerManager,
        *,
        focus_preview_threshold: float = 0.9,
        coarse_improve_thresh: float = 0.01,
        coarse_drop_stop_peak: float = 0.02,
        coarse_drop_stop_base: float = 0.03,
        z_floor_mm: float = 0.0,
        coarse_step_mm: float = 0.20,
        refine_step_mm: float = 0.12,
        max_offset_mm: float = 5.60,
        settle_still_s: float = 0.4,
        settle_preview_s: float = 0.4,
        fine_no_improve_limit: int = 2,
    ) -> None:
        super().__init__(motion)

        self._focus_preview_threshold = focus_preview_threshold
        self._coarse_improve_thresh = coarse_improve_thresh
        self._coarse_drop_stop_peak = coarse_drop_stop_peak
        self._coarse_drop_stop_base = coarse_drop_stop_base
        self._z_floor_nm = int(round(z_floor_mm * _NM_PER_MM))
        self._coarse_step_nm = int(round(coarse_step_mm * _NM_PER_MM))
        self._refine_step_nm = int(round(refine_step_mm * _NM_PER_MM))
        self._max_offset_nm = int(round(max_offset_mm * _NM_PER_MM))
        self._settle_still_s = settle_still_s
        self._settle_preview_s = settle_preview_s
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
            error("[Autofocus] No camera available — aborting")
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
            return (
                (start_nm - self._max_offset_nm) <= z_nm <= (start_nm + self._max_offset_nm)
                and z_nm >= z_floor
            )

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
        info(f"[Autofocus] Baseline Z={start_nm / _NM_PER_MM:.3f} mm  score={baseline:.3f}")

        coarse_scorer = (
            score_preview if baseline < self._focus_preview_threshold else score_still
        )
        info(
            f"[Autofocus] Coarse scorer: "
            f"{'PREVIEW' if coarse_scorer is score_preview else 'STILL'} "
            f"(baseline={baseline:.3f})"
        )

        yield  # pause/stop point: after baseline

        # ----------------------------------------------------------------
        # Coarse alternating sweep with bias
        # ----------------------------------------------------------------

        max_k = self._max_offset_nm // self._coarse_step_nm
        left_max_safe = min(max_k, (start_nm - z_floor) // self._coarse_step_nm)
        right_max_safe = max_k

        k_right = 1
        k_left = 1
        bias_side: str | None = None
        last_side: str | None = None
        peak_on_bias = baseline

        total_steps = max_k * 2 + 2  # rough upper bound for progress display
        sweep_step = 0

        while True:
            if self._check_stop():
                return

            right_has = k_right <= right_max_safe
            left_has = k_left <= left_max_safe
            if not right_has and not left_has:
                break

            # Choose side
            if bias_side:
                if bias_side == "right" and right_has:
                    side = "right"
                elif bias_side == "left" and left_has:
                    side = "left"
                else:
                    side = "right" if right_has else "left"
            else:
                if last_side == "left" and right_has:
                    side = "right"
                elif last_side == "right" and left_has:
                    side = "left"
                elif right_has:
                    side = "right"
                else:
                    side = "left"

            k = k_right if side == "right" else k_left
            offset = k * self._coarse_step_nm if side == "right" else -k * self._coarse_step_nm
            target = _quantize(start_nm + offset, fine_step_nm)

            if side == "left" and target < z_floor:
                info("[Autofocus] Coarse: reached Z floor, stopping left")
                k_left = left_max_safe + 1
                last_side = side
                continue

            sweep_step += 1
            self._set_status(
                f"Coarse {'↑' if side == 'right' else '↓'}  Z={target / _NM_PER_MM:.3f} mm",
                sweep_step, total_steps,
            )

            s = score_at(target, scores, coarse_scorer)
            improv = s - baseline
            info(
                f"[Autofocus] Coarse side={side:<5}  Z={target / _NM_PER_MM:.3f}  "
                f"score={s:.3f}  Δbase={improv:+.3f}"
            )

            if s > best_s:
                best_s, best_z = s, target

            if best_z == start_nm and (baseline - s) >= self._coarse_drop_stop_base:
                info("[Autofocus] Early stop (baseline-drop)")
                break

            if not bias_side and improv >= self._coarse_improve_thresh:
                bias_side = side
                peak_on_bias = s
                info(f"[Autofocus] Bias → {bias_side.upper()}")

            if bias_side and side == bias_side:
                if s > peak_on_bias:
                    peak_on_bias = s
                elif (peak_on_bias - s) >= self._coarse_drop_stop_peak:
                    info("[Autofocus] Early stop (peak-drop)")
                    break

            if side == "right":
                k_right += 1
            else:
                k_left += 1
            last_side = side

            if bias_side and (
                (bias_side == "right" and k_right > max_k)
                or (bias_side == "left" and k_left > max_k)
            ):
                break

            yield  # pause/stop point: between coarse steps

        # ----------------------------------------------------------------
        # Refine march in the best direction
        # ----------------------------------------------------------------

        if self._check_stop():
            return

        self._set_activity(f"Refine march  Z≈{best_z / _NM_PER_MM:.3f} mm")

        up_nm = _quantize(best_z + self._refine_step_nm, fine_step_nm)
        down_nm = _quantize(best_z - self._refine_step_nm, fine_step_nm)
        up_s = score_at(up_nm, scores, coarse_scorer)
        down_s = score_at(down_nm, scores, coarse_scorer)

        if up_s >= down_s:
            refine_dir, refine_z, refine_s = "up", up_nm, up_s
        else:
            refine_dir, refine_z, refine_s = "down", down_nm, down_s

        info(
            f"[Autofocus] Refine probe {self._refine_step_nm / _NM_PER_MM:.3f} mm {refine_dir}: "
            f"Z={refine_z / _NM_PER_MM:.3f}  score={refine_s:.3f}"
        )
        if refine_s > best_s:
            best_s, best_z = refine_s, refine_z

        current_z, prev_s = refine_z, refine_s
        while True:
            if self._check_stop():
                return
            step = self._refine_step_nm if refine_dir == "up" else -self._refine_step_nm
            nxt = _quantize(current_z + step, fine_step_nm)
            if nxt < z_floor or not within_env(nxt):
                break
            s = score_at(nxt, scores, coarse_scorer)
            info(
                f"[Autofocus] Refine {self._refine_step_nm / _NM_PER_MM:.3f} mm {refine_dir}: "
                f"Z={nxt / _NM_PER_MM:.3f}  score={s:.3f}"
            )
            if s > best_s:
                best_s, best_z = s, nxt
            if s + 1e-6 >= prev_s:
                current_z, prev_s = nxt, s
            else:
                break
            yield  # pause/stop point: between refine steps

        # ----------------------------------------------------------------
        # Fine polish (always stills)
        # ----------------------------------------------------------------

        if self._check_stop():
            return

        self._set_activity(f"Fine polish  Z≈{best_z / _NM_PER_MM:.3f} mm")
        info(f"[Autofocus] Fine search (step={fine_step_nm / _NM_PER_MM:.4f} mm, always STILL)")

        def _climb(start_z: int, step: int) -> tuple[int, float]:
            zt = start_z
            best_lz = start_z
            best_ls = scores.get(start_z, score_at(start_z, scores, score_still))
            no_imp = 0
            while True:
                nxt = _quantize(zt + step, fine_step_nm)
                if nxt < z_floor or not within_env(nxt):
                    break
                s = score_at(nxt, scores, score_still)
                info(
                    f"[Autofocus] Fine {'↑' if step > 0 else '↓'}"
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
            total_steps, total_steps,
        )
        info(
            f"[Autofocus] Complete: Z={best_z / _NM_PER_MM:.3f} mm  "
            f"score={best_s:.3f}  Δbase={(best_s - baseline):+.3f}"
        )