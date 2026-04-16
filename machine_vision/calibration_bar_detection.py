"""
calibration_bar_detection.py

Library for detecting a calibration bar (ruler/scale bar) in a single BGR
frame.  Determines the bar's orientation (horizontal or vertical), locates
its tick marks, and reports whether end-caps are present at each terminus.

Public API
----------
  AxisState          -- per-session hysteresis state; preserve across frames
  BarDetectionResult -- result returned by process_frame
  process_frame      -- run the full pipeline on one BGR frame
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

END_SEARCH_FRACTION  = 0.15
EDGE_MARGIN_FRACTION = 0.05
LOW_TICK_THRESHOLD   = 2
HIGH_TICK_THRESHOLD  = 5
CONFIRM_STREAK       = 3


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------

@dataclass
class AxisState:
    """
    Hysteresis state for axis detection across a sequence of frames.

    Tracks a confirmation streak for the detected axis.  Once the same axis
    has been confirmed for ``CONFIRM_STREAK`` consecutive frames with enough
    ticks, the state is ``bypassed``: subsequent frames skip the expensive
    full ``select_axis`` search and use a fast argmax instead.  The bypass
    is released when the tick count drops back below ``HIGH_TICK_THRESHOLD``.
    """
    confirmed_axis: str | None = None
    streak: int = 0
    bypassed: bool = False

    def update(self, axis: str, tick_count: int) -> None:
        if self.bypassed:
            if tick_count < HIGH_TICK_THRESHOLD:
                self.bypassed = False
                self.confirmed_axis = None
                self.streak = 0
        else:
            if self.confirmed_axis == axis:
                self.streak += 1
            else:
                self.confirmed_axis = axis
                self.streak = 1
            if self.streak >= CONFIRM_STREAK and tick_count >= HIGH_TICK_THRESHOLD:
                self.bypassed = True

    @property
    def mode(self) -> str:
        if self.bypassed:
            return "bypassed"
        if self.streak >= CONFIRM_STREAK:
            return "confirmed"
        return "searching"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class BarDetectionResult:
    """Result of a single calibration-bar detection pass."""

    axis: str
    """Detected bar orientation: ``"horizontal"`` or ``"vertical"``."""

    tick_count: int
    """Number of tick-mark clusters detected along the bar."""

    start_present: bool
    """``True`` when an end-cap tick is present at the start of the bar."""

    end_present: bool
    """``True`` when an end-cap tick is present at the end of the bar."""

    mode: str
    """Axis-state mode after this frame: ``"searching"``, ``"confirmed"``, or ``"bypassed"``."""

    elapsed_ms: float
    """Wall-clock time spent in the detection pipeline (ms), excluding BGR conversion."""

    # Internal fields used by the overlay for coordinate reprojection.
    _baseline_index: int = field(repr=False)
    _bar_start: int = field(repr=False)
    _bar_end: int = field(repr=False)
    _clusters: list[tuple[int, int]] = field(repr=False)
    _downsample: int = field(repr=False)


# ---------------------------------------------------------------------------
# Internal pipeline functions
# ---------------------------------------------------------------------------

def _binarize(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _ranked_candidates(sums: np.ndarray, dim: int) -> list[int]:
    margin = max(1, int(dim * EDGE_MARGIN_FRACTION))
    order = np.argsort(sums)[::-1]
    return [int(i) for i in order if margin <= i < dim - margin]


def _best_candidate(sums: np.ndarray, dim: int) -> int | None:
    margin = max(1, int(dim * EDGE_MARGIN_FRACTION))
    if margin >= dim - margin:
        return None
    masked = sums[margin:dim - margin]
    if masked.size == 0:
        return None
    return int(masked.argmax()) + margin


def _count_ticks_for_candidate(
    binary: np.ndarray,
    axis: str,
    baseline_index: int,
    tick_min_length: int,
) -> int:
    h, w = binary.shape
    baseline_band = 3

    if axis == "horizontal":
        y0 = max(0, baseline_index - baseline_band)
        y1 = min(h, baseline_index + baseline_band + 1)
        presence = binary[y0:y1].max(axis=0)
        nz = np.where(presence > 0)[0]
        if len(nz) == 0:
            return 0
        bar_start, bar_end = int(nz[0]), int(nz[-1])
        baseline_presence = binary[y0:y1, bar_start:bar_end + 1].max(axis=0)
        perp_counts = np.count_nonzero(binary[:, bar_start:bar_end + 1], axis=0)
    else:
        x0 = max(0, baseline_index - baseline_band)
        x1 = min(w, baseline_index + baseline_band + 1)
        presence = binary[:, x0:x1].max(axis=1)
        nz = np.where(presence > 0)[0]
        if len(nz) == 0:
            return 0
        bar_start, bar_end = int(nz[0]), int(nz[-1])
        baseline_presence = binary[bar_start:bar_end + 1, x0:x1].max(axis=1)
        perp_counts = np.count_nonzero(binary[bar_start:bar_end + 1, :], axis=1)

    mask = (baseline_presence > 0) & (perp_counts >= tick_min_length)
    raw = (np.where(mask)[0] + bar_start).tolist()
    return len(_cluster_ticks(raw))


def _select_axis(binary: np.ndarray, tick_min_length: int) -> tuple[str, int]:
    h, w = binary.shape
    row_sums = binary.sum(axis=1)
    col_sums = binary.sum(axis=0)

    h_candidates = _ranked_candidates(row_sums, h)
    v_candidates = _ranked_candidates(col_sums, w)

    best_h = h_candidates[0] if h_candidates else None
    best_v = v_candidates[0] if v_candidates else None

    if best_h is None and best_v is None:
        return "horizontal", int(row_sums.argmax())
    if best_h is None:
        return "vertical", best_v
    if best_v is None:
        return "horizontal", best_h

    if row_sums[best_h] >= col_sums[best_v]:
        axis, index = "horizontal", best_h
        alt_axis, alt_index = "vertical", best_v
    else:
        axis, index = "vertical", best_v
        alt_axis, alt_index = "horizontal", best_h

    ticks = _count_ticks_for_candidate(binary, axis, index, tick_min_length)
    if ticks <= LOW_TICK_THRESHOLD:
        alt_ticks = _count_ticks_for_candidate(binary, alt_axis, alt_index, tick_min_length)
        if alt_ticks > ticks:
            return alt_axis, alt_index

    return axis, index


def _resolve_axis(
    binary: np.ndarray,
    state: AxisState,
    tick_min_length: int,
) -> tuple[str, int]:
    if state.bypassed:
        h, w = binary.shape
        if state.confirmed_axis == "horizontal":
            sums = binary.sum(axis=1)
            index = _best_candidate(sums, h) or int(sums.argmax())
        else:
            sums = binary.sum(axis=0)
            index = _best_candidate(sums, w) or int(sums.argmax())
        return state.confirmed_axis, index

    return _select_axis(binary, tick_min_length)


def _find_baseline_extent(
    binary: np.ndarray,
    axis: str,
    baseline_index: int,
    baseline_band: int = 3,
) -> tuple[int, int]:
    h, w = binary.shape
    if axis == "horizontal":
        y0 = max(0, baseline_index - baseline_band)
        y1 = min(h, baseline_index + baseline_band + 1)
        presence = binary[y0:y1].max(axis=0)
    else:
        x0 = max(0, baseline_index - baseline_band)
        x1 = min(w, baseline_index + baseline_band + 1)
        presence = binary[:, x0:x1].max(axis=1)
    nz = np.where(presence > 0)[0]
    if len(nz) == 0:
        return 0, (w if axis == "horizontal" else h)
    return int(nz[0]), int(nz[-1])


def _collect_tick_positions(
    binary: np.ndarray,
    axis: str,
    baseline_index: int,
    bar_start: int,
    bar_end: int,
    baseline_band: int = 3,
    tick_min_length: int = 200,
) -> list[int]:
    h, w = binary.shape

    if axis == "horizontal":
        y0 = max(0, baseline_index - baseline_band)
        y1 = min(h, baseline_index + baseline_band + 1)
        baseline_presence = binary[y0:y1, bar_start:bar_end + 1].max(axis=0)
        py0 = max(0, baseline_index - tick_min_length)
        py1 = min(h, baseline_index + tick_min_length + 1)
        perp_counts = np.count_nonzero(binary[py0:py1, bar_start:bar_end + 1], axis=0)
    else:
        x0 = max(0, baseline_index - baseline_band)
        x1 = min(w, baseline_index + baseline_band + 1)
        baseline_presence = binary[bar_start:bar_end + 1, x0:x1].max(axis=1)
        px0 = max(0, baseline_index - tick_min_length)
        px1 = min(w, baseline_index + tick_min_length + 1)
        perp_counts = np.count_nonzero(binary[bar_start:bar_end + 1, px0:px1], axis=1)

    mask = (baseline_presence > 0) & (perp_counts >= tick_min_length)
    return (np.where(mask)[0] + bar_start).tolist()


def _cluster_ticks(positions: list[int], gap: int = 5) -> list[tuple[int, int]]:
    if not positions:
        return []
    clusters: list[tuple[int, int]] = []
    s = e = positions[0]
    for p in positions[1:]:
        if p - e <= gap:
            e = p
        else:
            clusters.append((s, e))
            s = e = p
    clusters.append((s, e))
    return clusters


def _detect_ends(
    binary: np.ndarray,
    axis: str,
    baseline_index: int,
    bar_start: int,
    bar_end: int,
    tick_min_length: int,
) -> tuple[bool, bool, list[tuple[int, int]]]:
    bar_len = bar_end - bar_start
    search_window = max(1, int(bar_len * END_SEARCH_FRACTION))

    raw = _collect_tick_positions(
        binary, axis, baseline_index, bar_start, bar_end,
        tick_min_length=tick_min_length,
    )
    clusters = _cluster_ticks(raw)
    centres = [(s + e) // 2 for s, e in clusters]

    start_present = any(c <= bar_start + search_window for c in centres)
    end_present   = any(c >= bar_end   - search_window for c in centres)

    return start_present, end_present, clusters


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_frame(
    bgr: np.ndarray,
    state: AxisState,
    downsample: int | None = None,
    tick_min_length: int = 200,
) -> BarDetectionResult:
    """
    Run a full calibration-bar detection pass on a single BGR frame.

    Converts to greyscale, optionally downsamples, binarizes, then detects
    the bar's axis, extent, tick clusters, and end-cap presence.  Updates
    *state* in place so hysteresis carries over to the next call.

    Parameters
    ----------
    bgr:
        Full-resolution BGR image (e.g. from ``cv2.cvtColor(rgb, COLOR_RGB2BGR)``).
    state:
        Per-session ``AxisState``.  Create once and reuse across frames.
        Call ``state = AxisState()`` to reset between sessions.
    downsample:
        Integer downscale factor applied before processing.  ``None`` or ``1``
        disables downscaling.  Use a value like ``2`` for large full-resolution
        images where speed matters more than precision.
    tick_min_length:
        Minimum perpendicular pixel run (full-resolution pixels) to count as
        a tick.  Scaled by ``downsample`` internally when downscaling is active.

    Returns
    -------
    BarDetectionResult with detection summary and internal geometry for
    coordinate reprojection by the overlay.
    """
    t_start = time.perf_counter()

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    effective_downsample = downsample if (downsample is not None and downsample > 1) else 1
    if effective_downsample > 1:
        h, w = gray.shape
        small = cv2.resize(gray, (w // effective_downsample, h // effective_downsample), interpolation=cv2.INTER_AREA)
    else:
        small = gray

    binary = _binarize(small)
    scaled_tick_min = max(1, tick_min_length // effective_downsample)

    axis, baseline_index = _resolve_axis(binary, state, scaled_tick_min)
    bar_start, bar_end = _find_baseline_extent(binary, axis, baseline_index)
    start_present, end_present, clusters = _detect_ends(
        binary, axis, baseline_index, bar_start, bar_end,
        tick_min_length=scaled_tick_min,
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    state.update(axis, len(clusters))

    return BarDetectionResult(
        axis=axis,
        tick_count=len(clusters),
        start_present=start_present,
        end_present=end_present,
        mode=state.mode,
        elapsed_ms=elapsed_ms,
        _baseline_index=baseline_index,
        _bar_start=bar_start,
        _bar_end=bar_end,
        _clusters=clusters,
        _downsample=effective_downsample,
    )