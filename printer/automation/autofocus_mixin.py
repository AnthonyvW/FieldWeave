"""
Autofocus functionality for automated 3D printer control.

This module contains all autofocus-related methods that can be mixed into
the main AutomatedPrinter controller class.
"""

import time
from typing import Optional, Callable

from printer.base_controller import command


# Autofocus constants (ticks per mm = 100, meaning 0.01 mm units)
_AFTPM = 100          # ticks/mm (0.01 mm units)
_AFSTEP = 4           # 0.04 mm (printer min step)
_AF_ZFLOOR = 0        # 0.00 mm -> 0 ticks


class AutofocusMixin:
    """
    Mixin class containing autofocus functionality.
    
    This class assumes it will be mixed into a controller that has:
    - self.machine_vision (MachineVision instance)
    - self.camera (camera instance with capture_image, get_last_frame, is_taking_image)
    - self._exec_gcode(gcode, wait=False) method
    - self.status(message, log=True) method
    - self.pause_point() method that returns True if stopped
    - self.register_handler(kind, function) method
    """
    
    def _init_autofocus_handlers(self):
        """Register autofocus command handlers. Call this from __init__."""
        self.register_handler("AUTOFOCUS_DESCENT", self.autofocus_descent_macro)
        self.register_handler("AUTOFOCUS", self.autofocus_macro)
        self.register_handler("FINE_AUTOFOCUS", self.fine_autofocus)
    
    # ========================================================================
    # Core autofocus helper methods
    # ========================================================================
    
    def _af_quantize(self, z_ticks: int) -> int:
        """Quantize Z position to printer's minimum step size."""
        return int(round(z_ticks / _AFSTEP) * _AFSTEP)

    def _af_move_to_ticks(self, z_ticks: int) -> None:
        """Move Z axis to specified position in ticks."""
        z_ticks = max(z_ticks, _AF_ZFLOOR)
        z_mm = z_ticks / _AFTPM
        self._exec_gcode(f"G0 Z{z_mm:.2f}", wait=True)

    def _af_score_still(self) -> float:
        """Capture a STILL image and return its focus score."""
        self._exec_gcode("M400", wait=True)
        time.sleep(0.1)  # vibration settle for stills
        self.camera.capture_image()
        while self.camera.is_taking_image:
            time.sleep(0.01)
        if self.machine_vision.is_black(source="still"):
            return float("-inf")
        try:
            img = self.camera.get_last_frame(prefer="still", wait_for_still=False)
            res = self.machine_vision.analyze_focus()
            return float(getattr(res, "focus_score", float("-inf")))
        except Exception:
            return float("-inf")

    def _af_score_preview(self) -> float:
        """Score the live preview/stream (no still capture). Much faster."""
        self._exec_gcode("M400", wait=True)
        time.sleep(0.05)  # tiny settle is enough for stream
        if self.machine_vision.is_black(source="stream"):
            return float("-inf")
        try:
            img = self.camera.get_last_frame(prefer="stream", wait_for_still=False)
            res = self.machine_vision.analyze_focus()
            return float(getattr(res, "focus_score", float("-inf")))
        except Exception:
            return float("-inf")

    def _af_score_at(
        self,
        zt: int,
        cache: dict[int, float],
        bounds_ok: Optional[Callable[[int], bool]] = None,
        scorer: Optional[Callable[[], float]] = None,
    ) -> float:
        """
        Quantize, check bounds, check cache, move, and score using the provided scorer.
        Defaults to STILL scorer if not provided.
        """
        scorer = scorer or self._af_score_still
        zt = self._af_quantize(zt)
        if zt < _AF_ZFLOOR:
            return float("-inf")
        if bounds_ok and not bounds_ok(zt):
            return float("-inf")
        if zt in cache:
            return cache[zt]
        self._af_move_to_ticks(zt)
        s = scorer(zt, cache, bounds_ok)
        cache[zt] = s
        return s

    def _af_climb_fine(
        self,
        start: int,
        step_ticks: int,
        cache: dict[int, float],
        bounds_ok: Optional[Callable[[int], bool]] = None,
        no_improve_limit: int = 2,
        scorer: Optional[Callable[[], float]] = None,
        baseline: Optional[float] = None,
    ) -> tuple[int, float]:
        """
        Climb in one direction with fine steps until no improvement is found.
        Returns (best_z, best_score).
        """
        scorer = scorer or self._af_score_still
        zt = start
        best_z = start
        best_s = cache.get(start, self._af_score_at(start, cache, bounds_ok, scorer))
        no_imp = 0
        
        while True:
            nxt = self._af_quantize(zt + step_ticks)
            if nxt < _AF_ZFLOOR or (bounds_ok and not bounds_ok(nxt)):
                break
            s = self._af_score_at(nxt, cache, bounds_ok, scorer)
            delta = f"  Δbase={s - baseline:+.1f}" if baseline is not None else ""
            self.status(
                f"[AF-Fine] {step_ticks/_AFTPM:.2f}mm step {'up' if step_ticks>0 else 'down'}: "
                f"Z={nxt / _AFTPM:.2f}  score={s:.1f}{delta}",
                False
            )
            if s > best_s + 1e-6:
                best_z, best_s = nxt, s
                zt = nxt
                no_imp = 0
            else:
                no_imp += 1
                zt = nxt
                if no_imp >= no_improve_limit:
                    break
        return best_z, best_s

    def _af_refine_around(
        self,
        center: int,
        cache: dict[int, float],
        bounds_ok: Optional[Callable[[int], bool]] = None,
        fine_step_ticks: int = _AFSTEP,
        no_improve_limit: int = 2,
        scorer: Optional[Callable[[], float]] = None,
        baseline: Optional[float] = None,
    ) -> tuple[int, float]:
        """
        Refine focus by climbing both up and down from center position.
        Returns (best_z, best_score) from both directions.
        """
        scorer = scorer or self._af_score_still
        up_z, up_s = self._af_climb_fine(
            center, fine_step_ticks, cache, bounds_ok, no_improve_limit, scorer, baseline
        )
        down_z, down_s = self._af_climb_fine(
            center, -fine_step_ticks, cache, bounds_ok, no_improve_limit, scorer, baseline
        )
        return (up_z, up_s) if up_s >= down_s else (down_z, down_s)

    # ========================================================================
    # Main autofocus macros
    # ========================================================================

    def autofocus_descent_macro(self, cmd: command) -> None:
        """
        Descent-only autofocus with configurable envelope, step sizes, and scoring.
        Coarse: fixed downward march from the start position toward Z floor.
        Refine: fine polish around the best coarse Z.
        """
        # Tunables
        FOCUS_PREVIEW_THRESHOLD = 90000.0
        Z_FLOOR_MM = 0.00
        COARSE_STEP_MM = 0.20
        FINE_STEP_MM = 0.04
        MAX_OFFSET_MM = 5.60
        DROP_STOP_PEAK = 5000.0
        DROP_STOP_BASE = 3000.0
        SETTLE_STILL_S = 0.4
        SETTLE_PREVIEW_S = 0.4
        FINE_NO_IMPROVE_LIMIT = 2
        FINE_ALLOW_PREVIEW = False
        LOG_VERBOSE = True

        # Derived constants
        _AF_ZFLOOR = int(round(Z_FLOOR_MM * _AFTPM))
        COARSE_STEP = int(round(COARSE_STEP_MM * _AFTPM))
        _AFSTEP = int(round(FINE_STEP_MM * _AFTPM))
        MAX_OFFSET = int(round(MAX_OFFSET_MM * _AFTPM))

        def quantize(zt: int) -> int:
            step = 4
            return (zt // step) * step

        def within_env(zt: int) -> bool:
            return (start - MAX_OFFSET) <= zt <= start and zt >= _AF_ZFLOOR

        # Scorers
        def score_still_lambda(_z, _c, _b) -> float:
            self._exec_gcode("M400", wait=True)
            if SETTLE_STILL_S > 0:
                time.sleep(SETTLE_STILL_S)
            self.camera.capture_image()
            while self.camera.is_taking_image:
                time.sleep(0.01)
            if self.machine_vision.is_black(source="still"):
                return float("-inf")
            try:
                img = self.camera.get_last_frame(prefer="still", wait_for_still=False)
                res = self.machine_vision.analyze_focus()
                return float(res.focus_score)
            except Exception:
                return float("-inf")

        def score_preview_lambda(_z, _c, _b) -> float:
            self._exec_gcode("M400", wait=True)
            if SETTLE_PREVIEW_S > 0:
                time.sleep(SETTLE_PREVIEW_S)
            if self.machine_vision.is_black(source="stream"):
                return float("-inf")
            try:
                img = self.camera.get_last_frame(prefer="stream", wait_for_still=False)
                res = self.machine_vision.analyze_focus()
                return float(res.focus_score)
            except Exception:
                return float("-inf")

        # Start
        self.status(cmd.message or "Autofocus (descent) starting...", cmd.log)
        if self.pause_point():
            return

        pos = self.get_position()
        start = quantize(int(round(getattr(pos, "z", 1600))))
        self.status(f"Start @ Z={start / _AFTPM:.2f} mm (descent expected)", cmd.log)

        scores: dict[int, float] = {}

        # Baseline STILL
        self._af_move_to_ticks(start)
        baseline = self._af_score_at(start, scores, within_env, scorer=score_still_lambda)
        scores[start] = baseline
        best_z = start
        best_s = baseline
        self.status(f"[AF-Descent] Baseline Z={start / _AFTPM:.2f}  score={baseline:.1f}", LOG_VERBOSE)

        # Choose coarse scorer
        coarse_scorer = (
            score_preview_lambda if (baseline < FOCUS_PREVIEW_THRESHOLD) else score_still_lambda
        )
        self.status(
            f"[AF-Descent] Coarse scorer: "
            f"{'PREVIEW' if coarse_scorer is score_preview_lambda else 'STILL'} "
            f"(baseline={baseline:.1f} < thresh={FOCUS_PREVIEW_THRESHOLD:.1f})",
            LOG_VERBOSE
        )

        # Coarse descent
        peak_s = baseline
        peak_z = start
        steps = min(MAX_OFFSET // COARSE_STEP, (start - _AF_ZFLOOR) // COARSE_STEP)

        for k in range(1, steps + 1):
            if self.pause_point():
                self.status("Autofocus paused/stopped.", True)
                return

            target = quantize(start - k * COARSE_STEP)
            if target <= _AF_ZFLOOR:
                target = _AF_ZFLOOR

            s = self._af_score_at(target, scores, within_env, scorer=coarse_scorer)
            d_base = s - baseline
            self.status(
                f"[AF-Descent] ↓{COARSE_STEP_MM:.2f}mm  Z={target / _AFTPM:.2f}"
                f"{' (FLOOR)' if target == _AF_ZFLOOR else ''}  score={s:.1f}  Δbase={d_base:+.1f}",
                LOG_VERBOSE
            )

            if s > best_s:
                best_s, best_z = s, target
            if s > peak_s:
                peak_s, peak_z = s, target

            if best_z == start and (baseline - s) >= DROP_STOP_BASE:
                self.status("[AF-Descent] Early stop (baseline-drop)", LOG_VERBOSE)
                break
            if (peak_s - s) >= DROP_STOP_PEAK:
                self.status("[AF-Descent] Early stop (peak-drop)", LOG_VERBOSE)
                break
            if target == _AF_ZFLOOR:
                break

        # Fine polish
        if self.pause_point():
            self.status("Autofocus paused/stopped.", True)
            return

        if FINE_ALLOW_PREVIEW and baseline < FOCUS_PREVIEW_THRESHOLD:
            fine_scorer = score_preview_lambda
            scorer_name = "PREVIEW"
        else:
            fine_scorer = score_still_lambda
            scorer_name = "STILL"

        self.status(
            f"[AF-Descent] Fine search using {scorer_name} (step={FINE_STEP_MM:.2f}mm)",
            LOG_VERBOSE
        )

        local_z, local_s = self._af_refine_around(
            center=best_z,
            cache=scores,
            bounds_ok=within_env,
            fine_step_ticks=_AFSTEP,
            no_improve_limit=FINE_NO_IMPROVE_LIMIT,
            scorer=fine_scorer,
            baseline=baseline
        )
        if local_s > best_s:
            best_z, best_s = local_z, local_s

        if self.pause_point():
            return
        self._af_move_to_ticks(best_z)
        self.status(
            f"Autofocus (descent) complete: Best Z={best_z / _AFTPM:.2f} mm  "
            f"Score={best_s:.1f}  Δbase={(best_s - baseline):+.1f}  "
            f"(coarse={'PREVIEW' if coarse_scorer is score_preview_lambda else 'STILL'}, "
            f"fine={scorer_name}, step={FINE_STEP_MM:.2f}mm, max_offset={MAX_OFFSET_MM:.2f}mm)",
            True
        )

    def fine_autofocus(self, cmd: command) -> None:
        """
        Fine autofocus around current Z with configurable window, step, and scoring.
        """
        # Tunables
        WINDOW_MM = 0.16
        FINE_STEP_MM = 0.04
        NO_IMPROVE_LIMIT = 1
        USE_PREVIEW_IF_BELOW = False
        FOCUS_PREVIEW_THRESHOLD = 90000.0
        LOG_VERBOSE = True

        # Derived constants
        _AF_ZFLOOR = 0
        FINE_STEP_TICKS = int(round(FINE_STEP_MM * _AFTPM))
        WINDOW_TICKS = int(round(WINDOW_MM * _AFTPM))

        def within_window(zt: int, center: int) -> bool:
            return (center - WINDOW_TICKS) <= zt <= (center + WINDOW_TICKS) and zt >= _AF_ZFLOOR

        # Start
        self.status(cmd.message or "Fine autofocus...", cmd.log)

        pos = self.get_position()
        center = self._af_quantize(int(round(getattr(pos, "z", 1600))))
        self.status(
            f"[AF-Fine] Center Z={center / _AFTPM:.2f} mm  Window=±{WINDOW_MM:.2f} mm  "
            f"Step={FINE_STEP_MM:.2f} mm",
            LOG_VERBOSE
        )

        scores: dict[int, float] = {}

        # Baseline with STILL
        baseline = self._af_score_at(
            center, scores, lambda z: within_window(z, center),
            scorer=lambda _z, _c, _b: self._af_score_still()
        )

        # Choose scorer
        if USE_PREVIEW_IF_BELOW and baseline < FOCUS_PREVIEW_THRESHOLD:
            fine_scorer = lambda _z, _c, _b: self._af_score_preview()
            scorer_name = "PREVIEW"
        else:
            fine_scorer = lambda _z, _c, _b: self._af_score_still()
            scorer_name = "STILL"

        self.status(
            f"[AF-Fine] Using {scorer_name} scorer for search "
            f"(baseline={baseline:.1f}  thresh={FOCUS_PREVIEW_THRESHOLD:.1f})",
            LOG_VERBOSE
        )

        # Fine search
        if self.pause_point():
            return

        best_z, best_s = self._af_refine_around(
            center=center,
            cache=scores,
            bounds_ok=lambda z: within_window(z, center),
            fine_step_ticks=FINE_STEP_TICKS,
            no_improve_limit=NO_IMPROVE_LIMIT,
            scorer=fine_scorer,
            baseline=baseline
        )

        if self.pause_point():
            return

        self._af_move_to_ticks(best_z)
        self.status(
            f"[AF-Fine] Best Z={best_z / _AFTPM:.2f} mm  "
            f"Score={best_s:.1f}  Δbase={(best_s - baseline):+.1f}  "
            f"(search={scorer_name}, step={FINE_STEP_MM:.2f}mm, window=±{WINDOW_MM:.2f}mm, "
            f"no_improve_limit={NO_IMPROVE_LIMIT})",
            True
        )

    def autofocus_macro(self, cmd: command) -> None:
        """
        Coarse (0.40 mm) alternating with bias, then 0.20 mm refine march,
        then 0.04 mm fine polish.
        """
        # Tunables
        FOCUS_PREVIEW_THRESHOLD = 90000.0
        COARSE_IMPROVE_THRESH = 1000.0
        COARSE_DROP_STOP_PEAK = 2000.0
        COARSE_DROP_STOP_BASE = 3000.0
        Z_FLOOR_MM = 0.00
        COARSE_STEP_MM = 0.20
        REFINE_COARSE_MM = 0.12
        FINE_STEP_MM = 0.04
        MAX_OFFSET_MM = 5.60
        SETTLE_STILL_S = 0.4
        SETTLE_PREVIEW_S = 0.4
        FINE_NO_IMPROVE_LIMIT = 2
        LOG_VERBOSE = True

        # Derived constants
        _AF_ZFLOOR = int(round(Z_FLOOR_MM * _AFTPM))
        COARSE_STEP = int(round(COARSE_STEP_MM * _AFTPM))
        REFINE_COARSE = int(round(REFINE_COARSE_MM * _AFTPM))
        _AFSTEP = int(round(FINE_STEP_MM * _AFTPM))
        MAX_OFFSET = int(round(MAX_OFFSET_MM * _AFTPM))

        def quantize(zt: int) -> int:
            step = 4
            return (zt // step) * step

        def within_env(zt: int) -> bool:
            return (start - MAX_OFFSET) <= zt <= (start + MAX_OFFSET) and zt >= _AF_ZFLOOR

        def score_still() -> float:
            self._exec_gcode("M400", wait=True)
            if SETTLE_STILL_S > 0:
                time.sleep(SETTLE_STILL_S)
            self.camera.capture_image()
            while self.camera.is_taking_image:
                time.sleep(0.01)
            if self.machine_vision.is_black(source="still"):
                return float("-inf")
            img = self.camera.get_last_frame(prefer="still", wait_for_still=False)
            res = self.machine_vision.analyze_focus()
            return float(res.focus_score)

        def score_preview() -> float:
            self._exec_gcode("M400", wait=True)
            if SETTLE_PREVIEW_S > 0:
                time.sleep(SETTLE_PREVIEW_S)
            if self.machine_vision.is_black(source="stream"):
                return float("-inf")
            img = self.camera.get_last_frame(prefer="stream", wait_for_still=False)
            res = self.machine_vision.analyze_focus()
            return float(res.focus_score)

        def score_at(zt: int, cache: dict, scorer) -> float:
            zt = quantize(zt)
            if zt < _AF_ZFLOOR or not within_env(zt):
                return float("-inf")
            if zt in cache:
                return cache[zt]
            self._af_move_to_ticks(zt)
            s = scorer()
            cache[zt] = s
            return s

        # Start
        self.status(cmd.message or "Autofocus starting...", cmd.log)
        if self.pause_point():
            return

        pos = self.get_position()
        start = quantize(int(round(getattr(pos, "z", 1600))))
        self.status(f"Start @ Z={start / _AFTPM:.2f} mm", cmd.log)

        scores: dict[int, float] = {}

        # Baseline STILL
        self._af_move_to_ticks(start)
        baseline = score_still()
        scores[start] = baseline
        best_z = start
        best_s = baseline
        self.status(f"[AF] Baseline Z={start / _AFTPM:.2f}  score={baseline:.1f}", LOG_VERBOSE)

        coarse_scorer = score_preview if (baseline < FOCUS_PREVIEW_THRESHOLD) else score_still
        self.status(
            f"[AF] Coarse scorer: "
            f"{'PREVIEW' if coarse_scorer is score_preview else 'STILL'} "
            f"(baseline={baseline:.1f} < thresh={FOCUS_PREVIEW_THRESHOLD:.1f})",
            LOG_VERBOSE
        )

        # Coarse alternating with bias
        k_right = 1
        k_left = 1
        max_k = MAX_OFFSET // COARSE_STEP
        left_max_safe = min(max_k, (start - _AF_ZFLOOR) // COARSE_STEP)
        right_max_safe = max_k
        bias_side = None
        last_side = None
        peak_on_bias = baseline

        while True:
            if self.pause_point():
                self.status("Autofocus paused/stopped.", True)
                return

            right_has = k_right <= right_max_safe
            left_has = k_left <= left_max_safe
            if not right_has and not left_has:
                break

            # Choose side
            if bias_side:
                if bias_side == 'right' and right_has:
                    side = 'right'
                elif bias_side == 'left' and left_has:
                    side = 'left'
                else:
                    side = 'right' if right_has else 'left'
            else:
                if last_side == 'left' and right_has:
                    side = 'right'
                elif last_side == 'right' and left_has:
                    side = 'left'
                elif right_has:
                    side = 'right'
                else:
                    side = 'left'

            target = quantize(
                start + (k_right * COARSE_STEP if side == 'right' else -k_left * COARSE_STEP)
            )
            if side == 'left' and target < _AF_ZFLOOR:
                self.status("[AF-Coarse] Reached Z floor; stop left.", LOG_VERBOSE)
                k_left = left_max_safe + 1
                last_side = side
                continue

            s = score_at(target, scores, coarse_scorer)
            if s > best_s:
                best_s, best_z = s, target

            improv = s - baseline
            self.status(
                f"[AF-Coarse] side={side:<5} Z={target / _AFTPM:.2f}  "
                f"score={s:.1f}  Δbase={improv:+.1f}",
                LOG_VERBOSE
            )

            if best_z == start and (baseline - s) >= COARSE_DROP_STOP_BASE:
                self.status("[AF-Coarse] Early stop (baseline-drop)", LOG_VERBOSE)
                break

            if not bias_side and improv >= COARSE_IMPROVE_THRESH:
                bias_side = side
                peak_on_bias = s
                self.status(
                    f"[AF-Coarse] Bias → {bias_side.upper()} (≥+{COARSE_IMPROVE_THRESH:.0f})",
                    LOG_VERBOSE
                )

            if bias_side and side == bias_side:
                if s > peak_on_bias:
                    peak_on_bias = s
                elif (peak_on_bias - s) >= COARSE_DROP_STOP_PEAK:
                    self.status("[AF-Coarse] Early stop (peak-drop)", LOG_VERBOSE)
                    break

            if side == 'right':
                k_right += 1
            else:
                k_left += 1
            last_side = side

            if bias_side and ((bias_side == 'right' and not (k_right <= max_k)) or
                            (bias_side == 'left' and not (k_left <= max_k))):
                break

        # Refine march (0.20 mm)
        if self.pause_point():
            self.status("Autofocus paused/stopped.", True)
            return

        up_zt = quantize(best_z + REFINE_COARSE)
        down_zt = quantize(best_z - REFINE_COARSE)
        up_s = score_at(up_zt, scores, coarse_scorer)
        down_s = score_at(down_zt, scores, coarse_scorer)
        dir1, z1, s1 = (('up', up_zt, up_s) if up_s >= down_s else ('down', down_zt, down_s))
        self.status(
            f"[AF-Refine] Probe {REFINE_COARSE_MM:.2f}mm {dir1}: Z={z1 / _AFTPM:.2f}  score={s1:.1f}",
            LOG_VERBOSE
        )
        if s1 > best_s:
            best_s, best_z = s1, z1

        current, prev = z1, s1
        while True:
            if self.pause_point():
                self.status("Autofocus paused/stopped.", True)
                return
            step = REFINE_COARSE if dir1 == 'up' else -REFINE_COARSE
            nxt = quantize(current + step)
            if nxt < _AF_ZFLOOR or not within_env(nxt):
                break
            s = score_at(nxt, scores, coarse_scorer)
            self.status(
                f"[AF-Refine] {REFINE_COARSE_MM:.2f}mm step {dir1}: Z={nxt / _AFTPM:.2f}  score={s:.1f}",
                LOG_VERBOSE
            )
            if s > best_s:
                best_s, best_z = s, nxt
            if s + 1e-6 >= prev:
                current, prev = nxt, s
            else:
                break

        # Fine polish (ALWAYS STILLs)
        def climb_fine(start_zt: int, step_ticks: int) -> tuple[int, float]:
            zt = start_zt
            best_local_z = start_zt
            best_local_s = scores.get(start_zt, score_at(start_zt, scores, score_still))
            no_imp = 0
            while True:
                nxt = quantize(zt + step_ticks)
                if nxt < _AF_ZFLOOR or not within_env(nxt):
                    break
                s = score_at(nxt, scores, score_still)
                self.status(
                    f"[AF-Fine] {FINE_STEP_MM:.2f}mm step {'up' if step_ticks>0 else 'down'}: "
                    f"Z={nxt / _AFTPM:.2f}  score={s:.1f}",
                    LOG_VERBOSE
                )
                if s > best_local_s + 1e-6:
                    best_local_z, best_local_s = nxt, s
                    zt = nxt
                    no_imp = 0
                else:
                    no_imp += 1
                    zt = nxt
                    if no_imp >= FINE_NO_IMPROVE_LIMIT:
                        break
            return best_local_z, best_local_s

        up_z, up_s = climb_fine(best_z, _AFSTEP)
        down_z, down_s = climb_fine(best_z, -_AFSTEP)
        if (up_s, up_z) >= (down_s, down_z):
            local_z, local_s = up_z, up_s
        else:
            local_z, local_s = down_z, down_s
        if local_s > best_s:
            best_z, best_s = local_z, local_s

        if self.pause_point():
            return
        self._af_move_to_ticks(best_z)
        self.status(
            f"Autofocus complete: Best Z={best_z / _AFTPM:.2f} mm  Score={best_s:.1f}",
            True
        )

    # ========================================================================
    # Public convenience methods
    # ========================================================================

    def start_autofocus(self) -> None:
        """Start the autofocus macro."""
        self.reset_after_stop()
        self.enqueue_cmd(command(
            kind="AUTOFOCUS",
            value="",
            message="Beginning Autofocus Macro",
            log=True
        ))

    def start_fine_autofocus(self) -> None:
        """Start the fine autofocus macro."""
        self.reset_after_stop()
        self.enqueue_cmd(command(
            kind="FINE_AUTOFOCUS",
            value="",
            message="Beginning Fine Autofocus Macro",
            log=True
        ))