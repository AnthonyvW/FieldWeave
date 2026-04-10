"""
Fine autofocus automation routine.

Performs a narrow bidirectional fine-step search around the current Z
position. Intended as a quick touch-up when the stage is already close
to focus.

Usage::

    from common.app_context import get_app_context
    from motion.routines.autofocus_fine import AutofocusFine

    ctx = get_app_context()
    routine = AutofocusFine(motion=ctx.motion)
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
_AF_ZFLOOR_NM = 0


def _quantize(z_nm: int, step_nm: int) -> int:
    """Snap *z_nm* down to the nearest multiple of *step_nm*."""
    return (z_nm // step_nm) * step_nm


class AutofocusFine(AutomationRoutine):
    """
    Fine-only autofocus within a narrow window around the current Z.

    Takes a baseline still at the current position, optionally switches to
    the preview scorer for speed if the baseline is low, then climbs both up
    and down in fine steps, stopping after a configurable number of consecutive
    non-improving steps.

    The step size is not a parameter — it is read from
    ``motion.settings.step_size`` at runtime so it always reflects the
    configured minimum step for the connected machine.

    Parameters
    ----------
    motion:
        Active :class:`MotionControllerManager`.
    window_mm:
        Half-width of the search window in mm. The routine will not move more
        than this distance from the starting Z in either direction.
        When ``None`` (the default), the window is derived automatically at
        runtime as ``4 × step_size``, keeping it always proportional to the
        machine's minimum step.
    no_improve_limit:
        Consecutive non-improving steps before stopping a climb direction.
    use_preview_if_below:
        When True and the baseline still-score is below
        *focus_preview_threshold*, the search uses the faster preview scorer.
    focus_preview_threshold:
        Score threshold that gates switching to the preview scorer.
    settle_still_s:
        Seconds to wait after a move settles before a still capture.
    settle_preview_s:
        Seconds to wait after a move settles before a preview score.
    """

    job_name = "Autofocus (Fine)"

    def __init__(
        self,
        motion: MotionControllerManager,
        *,
        window_mm: float | None = None,
        no_improve_limit: int = 1,
        use_preview_if_below: bool = False,
        focus_preview_threshold: float = 90_000.0,
        settle_still_s: float = 0.4,
        settle_preview_s: float = 0.4,
    ) -> None:
        super().__init__(motion)

        # None means "derive from step size at runtime" (4 × step).
        # A caller-supplied value is converted immediately and used as-is.
        self._window_mm_override: float | None = window_mm
        self._window_nm: int = (
            int(round(window_mm * _NM_PER_MM)) if window_mm is not None else 0
        )
        self._no_improve_limit = no_improve_limit
        self._use_preview_if_below = use_preview_if_below
        self._focus_preview_threshold = focus_preview_threshold
        self._settle_still_s = settle_still_s
        self._settle_preview_s = settle_preview_s

    # ------------------------------------------------------------------
    # AutomationRoutine implementation
    # ------------------------------------------------------------------

    def steps(self) -> Generator[None, None, None]:
        ctx = get_app_context()
        camera_manager = ctx.camera_manager
        mv = ctx.machine_vision

        self._set_activity("Initialising")

        if not ctx.has_camera:
            error("[AutofocusFine] No camera available — aborting")
            return

        # Read the minimum step size from the motion config at runtime so any
        # settings reload is reflected without reconstructing the routine.
        settings = self.motion.settings
        fine_step_nm: int = settings.step_size if settings is not None else 40_000
        info(f"[AutofocusFine] settings.step_size={settings.step_size}  step_presets={settings.step_presets}")
        # Derive the search window from the step size when the caller did not
        # supply an explicit value.  4 steps in each direction gives enough
        # range for a fine touch-up while keeping the search fast.
        if self._window_mm_override is None:
            self._window_nm = fine_step_nm * 4

        # ----------------------------------------------------------------
        # Scorer callables
        # ----------------------------------------------------------------

        def score_still() -> float:
            if self._settle_still_s > 0:
                time.sleep(self._settle_still_s)
            frame = capture_still_frame(camera_manager)
            if frame is None:
                error("Autofocus Fine : No frame from capture still frame")
                return float("-inf")
            try:
                future = mv.request_focus_analysis_async(frame, frame.shape[1], frame.shape[0])
                return float(future.result(timeout=10.0).scores.peak)
            except Exception as exc:
                error(f"[AutofocusFine] score_still Focus analysis failed: {exc!r}")
                return float("-inf")

        def score_preview() -> float:
            if self._settle_preview_s > 0:
                time.sleep(self._settle_preview_s)
            frame = camera_manager.copy_current_frame_to_numpy()
            if frame is None:
                error("Autofocus Fine : No frame in score preview")
                return float("-inf")
            try:
                future = mv.request_focus_analysis_async(frame, frame.shape[1], frame.shape[0])
                return float(future.result(timeout=10.0).scores.peak)
            except Exception as exc:
                error(f"[AutofocusFine] score_preview Focus analysis failed: {exc!r}")
                return float("-inf")

        # ----------------------------------------------------------------
        # Motion / cache helpers
        # ----------------------------------------------------------------

        def move(z_nm: int) -> None:
            move_z_and_wait(self.motion, z_nm, _AF_ZFLOOR_NM)

        def within_window(z_nm: int) -> bool:
            return (
                (center_nm - self._window_nm) <= z_nm <= (center_nm + self._window_nm)
                and z_nm >= _AF_ZFLOOR_NM
            )

        def score_at(z_nm: int, cache: dict[int, float], scorer) -> float:
            z_nm = _quantize(z_nm, fine_step_nm)
            if not within_window(z_nm):
                info("not within window")
                return float("-inf")
            if z_nm in cache:
                return cache[z_nm]
            move(z_nm)
            s = scorer()
            cache[z_nm] = s
            return s

        # ----------------------------------------------------------------
        # Establish center position
        # ----------------------------------------------------------------

        center_nm = _quantize(self.motion.get_position().z, fine_step_nm)
        window_mm = self._window_nm / _NM_PER_MM
        step_mm = fine_step_nm / _NM_PER_MM

        self._set_activity(
            f"Baseline  Z={center_nm / _NM_PER_MM:.3f} mm  window=±{window_mm:.3f} mm"
        )
        info(
            f"[AutofocusFine] Center Z={center_nm / _NM_PER_MM:.3f} mm  "
            f"window=±{window_mm:.3f} mm  step={step_mm:.4f} mm"
        )

        scores: dict[int, float] = {}

        move(center_nm)
        baseline = score_still()
        scores[center_nm] = baseline
        info(f"[AutofocusFine] Baseline score={baseline:.3f}")

        if baseline == float("-inf"):
            error("[AutofocusFine] Baseline capture failed (score=-inf) — aborting")
            return

        # Choose scorer for the search pass
        if self._use_preview_if_below and baseline < self._focus_preview_threshold:
            search_scorer = score_preview
            scorer_name = "PREVIEW"
        else:
            search_scorer = score_still
            scorer_name = "STILL"

        info(
            f"[AutofocusFine] Using {scorer_name} scorer "
            f"(baseline={baseline:.3f}  thresh={self._focus_preview_threshold:.3f})"
        )

        yield  # pause/stop point: after baseline

        # ----------------------------------------------------------------
        # Fine bidirectional search
        # ----------------------------------------------------------------

        if self._check_stop():
            return

        total_steps = (self._window_nm // fine_step_nm) * 2 + 1
        self._set_progress(0, total_steps)

        def _climb(start_z: int, step: int) -> tuple[int, float]:
            zt = start_z
            best_lz = start_z
            best_ls = scores.get(start_z, score_at(start_z, scores, search_scorer))
            no_imp = 0
            step_count = 0
            while True:
                nxt = _quantize(zt + step, fine_step_nm)
                if not within_window(nxt):
                    break
                s = score_at(nxt, scores, search_scorer)
                step_count += 1
                self._set_progress(step_count, total_steps)
                info(
                    f"[AutofocusFine] {'↑' if step > 0 else '↓'}"
                    f"{abs(step) / _NM_PER_MM:.4f} mm  Z={nxt / _NM_PER_MM:.3f}  "
                    f"score={s:.3f}  Δbase={(s - baseline):+.3f}"
                )
                if s > best_ls + 1e-6:
                    best_lz, best_ls = nxt, s
                    zt = nxt
                    no_imp = 0
                else:
                    no_imp += 1
                    zt = nxt
                    if no_imp >= self._no_improve_limit:
                        break
            return best_lz, best_ls

        up_z, up_s = _climb(center_nm, fine_step_nm)

        if self._check_stop():
            return

        down_z, down_s = _climb(center_nm, -fine_step_nm)

        best_z = up_z if up_s >= down_s else down_z
        best_s = up_s if up_s >= down_s else down_s

        yield  # pause/stop point: after search

        if self._check_stop():
            return

        move(best_z)
        self._set_status(
            f"Done  Z={best_z / _NM_PER_MM:.3f} mm  score={best_s:.3f}",
            total_steps, total_steps,
        )
        info(
            f"[AutofocusFine] Complete: Z={best_z / _NM_PER_MM:.3f} mm  "
            f"score={best_s:.3f}  Δbase={(best_s - baseline):+.3f}  "
            f"scorer={scorer_name}  step={step_mm:.4f} mm  window=±{window_mm:.3f} mm"
        )