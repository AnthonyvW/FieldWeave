from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS     = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
END_SEARCH_FRACTION  = 0.15
EDGE_MARGIN_FRACTION = 0.05
LOW_TICK_THRESHOLD   = 2
HIGH_TICK_THRESHOLD  = 5
CONFIRM_STREAK       = 3
DEFAULT_DOWNSAMPLE   = 2

# Colour palette (BGR)
COL_BASELINE = (0, 200, 255)
COL_TICK     = (0, 255, 0)
COL_ABSENT   = (0, 0, 255)
COL_PRESENT  = (0, 255, 0)
COL_TEXT     = (255, 255, 255)
COL_SHADOW   = (0, 0, 0)


@dataclass
class AxisState:
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


@dataclass
class StepTimes:
    steps: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def record(self, step: str, ms: float) -> None:
        self.steps[step].append(ms)

    def report(self) -> None:
        if not self.steps:
            return
        col = 28
        header = f"  {'step':<{col}}  {'count':>6}  {'min':>8}  {'mean':>8}  {'max':>8}"
        print()
        print("Profile summary")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for step, times in self.steps.items():
            n = len(times)
            print(f"  {step:<{col}}  {n:>6}  {min(times):>7.2f}ms  {sum(times)/n:>7.2f}ms  {max(times):>7.2f}ms")
        totals = [sum(v[i] for v in self.steps.values()) for i in range(len(next(iter(self.steps.values()))))]
        print("-" * len(header))
        n = len(totals)
        print(f"  {'TOTAL':<{col}}  {n:>6}  {min(totals):>7.2f}ms  {sum(totals)/n:>7.2f}ms  {max(totals):>7.2f}ms")


def load_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path))


def binarize(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def ranked_candidates(sums: np.ndarray, dim: int) -> list[int]:
    """Full sorted list — used by select_axis which needs top-N candidates."""
    margin = max(1, int(dim * EDGE_MARGIN_FRACTION))
    order = np.argsort(sums)[::-1]
    return [int(i) for i in order if margin <= i < dim - margin]


def best_candidate(sums: np.ndarray, dim: int) -> int | None:
    """Argmax with edge margin — O(n) alternative for when only the top index is needed."""
    margin = max(1, int(dim * EDGE_MARGIN_FRACTION))
    if margin >= dim - margin:
        return None
    masked = sums[margin:dim - margin]
    if masked.size == 0:
        return None
    return int(masked.argmax()) + margin


def count_ticks_for_candidate(
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
    return len(cluster_ticks(raw))


def select_axis(binary: np.ndarray, tick_min_length: int) -> tuple[str, int]:
    h, w = binary.shape
    row_sums = binary.sum(axis=1)
    col_sums = binary.sum(axis=0)

    h_candidates = ranked_candidates(row_sums, h)
    v_candidates = ranked_candidates(col_sums, w)

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

    ticks = count_ticks_for_candidate(binary, axis, index, tick_min_length)
    if ticks <= LOW_TICK_THRESHOLD:
        alt_ticks = count_ticks_for_candidate(binary, alt_axis, alt_index, tick_min_length)
        if alt_ticks > ticks:
            return alt_axis, alt_index

    return axis, index


def resolve_axis(
    binary: np.ndarray,
    state: AxisState,
    tick_min_length: int,
) -> tuple[str, int]:
    if state.bypassed:
        h, w = binary.shape
        if state.confirmed_axis == "horizontal":
            sums = binary.sum(axis=1)
            index = best_candidate(sums, h) or int(sums.argmax())
        else:
            sums = binary.sum(axis=0)
            index = best_candidate(sums, w) or int(sums.argmax())
        return state.confirmed_axis, index

    return select_axis(binary, tick_min_length)


def find_baseline_extent(
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


def collect_tick_positions(
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
        # Crop perp scan to a band that is tick_min_length deep on each side of the
        # baseline. Any genuine tick will have at least tick_min_length pixels inside
        # this band, so the threshold comparison is exact.
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


def cluster_ticks(positions: list[int], gap: int = 5) -> list[tuple[int, int]]:
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


def detect_ends(
    binary: np.ndarray,
    axis: str,
    baseline_index: int,
    bar_start: int,
    bar_end: int,
    tick_min_length: int,
) -> tuple[bool, bool, list[tuple[int, int]]]:
    bar_len = bar_end - bar_start
    search_window = max(1, int(bar_len * END_SEARCH_FRACTION))

    raw = collect_tick_positions(
        binary, axis, baseline_index, bar_start, bar_end,
        tick_min_length=tick_min_length,
    )
    clusters = cluster_ticks(raw)
    centres = [(s + e) // 2 for s, e in clusters]

    start_present = any(c <= bar_start + search_window for c in centres)
    end_present   = any(c >= bar_end   - search_window for c in centres)

    return start_present, end_present, clusters


def put_text_shadowed(
    img: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    colour: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x, y = origin
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, COL_SHADOW, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def draw_debug(
    original: np.ndarray,
    axis: str,
    baseline_index: int,
    bar_start: int,
    bar_end: int,
    tick_clusters: list[tuple[int, int]],
    start_present: bool,
    end_present: bool,
    elapsed_ms: float,
    mode: str,
    downsample: int,
) -> np.ndarray:
    vis = original.copy() if len(original.shape) == 3 else cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]

    # Scale all pipeline coordinates back to full-res
    s = downsample
    bl   = baseline_index * s
    bst  = bar_start * s
    ben  = bar_end * s
    scaled_clusters = [(ts * s, te * s) for ts, te in tick_clusters]

    if axis == "horizontal":
        cv2.line(vis, (bst, bl), (ben, bl), COL_BASELINE, 2)
        for (ts, te) in scaled_clusters:
            cx = (ts + te) // 2
            cv2.line(vis, (cx, 0), (cx, h), COL_TICK, 1)
        put_text_shadowed(vis, "END" if start_present else "OPEN", (bst + 4, bl - 14), 0.7, COL_PRESENT if start_present else COL_ABSENT)
        put_text_shadowed(vis, "END" if end_present   else "OPEN", (ben - 60, bl - 14), 0.7, COL_PRESENT if end_present   else COL_ABSENT)
        put_text_shadowed(vis, "HORIZONTAL", (8, 28), 0.8, COL_TEXT)
    else:
        cv2.line(vis, (bl, bst), (bl, ben), COL_BASELINE, 2)
        for (ts, te) in scaled_clusters:
            cy = (ts + te) // 2
            cv2.line(vis, (0, cy), (w, cy), COL_TICK, 1)
        put_text_shadowed(vis, "END" if start_present else "OPEN", (bl + 8, bst + 20), 0.7, COL_PRESENT if start_present else COL_ABSENT)
        put_text_shadowed(vis, "END" if end_present   else "OPEN", (bl + 8, ben - 8),  0.7, COL_PRESENT if end_present   else COL_ABSENT)
        put_text_shadowed(vis, "VERTICAL", (8, 28), 0.8, COL_TEXT)

    put_text_shadowed(vis, f"ticks: {len(tick_clusters)}", (8, 56),  0.65, COL_TEXT)
    put_text_shadowed(vis, f"vision: {elapsed_ms:.1f}ms",  (8, 80),  0.65, COL_TEXT)
    put_text_shadowed(vis, f"mode: {mode}",                (8, 104), 0.65, COL_TEXT)
    put_text_shadowed(vis, f"ds: {s}x",                   (8, 128), 0.65, COL_TEXT)

    return vis


def _ms(t: float) -> float:
    return (time.perf_counter() - t) * 1000


def process_image(
    path: Path,
    output_dir: Path,
    tick_min_length: int,
    state: AxisState,
    downsample: int,
    profiler: StepTimes | None = None,
) -> None:
    original = load_image(path)
    if original is None:
        print(f"  WARNING: could not read {path.name}, skipping.", file=sys.stderr)
        return

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    t_total = time.perf_counter()

    t = time.perf_counter()
    if downsample > 1:
        h, w = gray.shape
        small = cv2.resize(gray, (w // downsample, h // downsample), interpolation=cv2.INTER_AREA)
    else:
        small = gray
    binary = binarize(small)
    if profiler: profiler.record("downsample+binarize", _ms(t))

    scaled_tick_min = max(1, tick_min_length // downsample)

    t = time.perf_counter()
    axis, baseline_index = resolve_axis(binary, state, scaled_tick_min)
    if profiler: profiler.record("resolve_axis", _ms(t))

    t = time.perf_counter()
    bar_start, bar_end = find_baseline_extent(binary, axis, baseline_index)
    if profiler: profiler.record("find_baseline_extent", _ms(t))

    t = time.perf_counter()
    start_present, end_present, clusters = detect_ends(
        binary, axis, baseline_index, bar_start, bar_end,
        tick_min_length=scaled_tick_min,
    )
    if profiler: profiler.record("detect_ends", _ms(t))

    elapsed_ms = _ms(t_total)

    state.update(axis, len(clusters))

    vis = draw_debug(
        original, axis, baseline_index,
        bar_start, bar_end, clusters,
        start_present, end_present,
        elapsed_ms, state.mode, downsample,
    )

    out_path = output_dir / (path.stem + "_inspect.png")
    cv2.imwrite(str(out_path), vis)

    start_str = "END" if start_present else "OPEN"
    end_str   = "END" if end_present   else "OPEN"
    print(f"  {path.name}: {axis:10s}  start={start_str}  stop={end_str}  ticks={len(clusters):3d}  vision={elapsed_ms:.1f}ms  [{state.mode}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect calibration-bar images: detect orientation and which ends are present."
    )
    parser.add_argument("input_dir",  type=Path, help="Folder of input images.")
    parser.add_argument("output_dir", type=Path, help="Folder for annotated debug images.")
    parser.add_argument(
        "--tick-min-length", type=int, default=200,
        help="Minimum perpendicular pixel run to count as a tick (default: 200).",
    )
    parser.add_argument(
        "--downsample", type=int, default=DEFAULT_DOWNSAMPLE, metavar="N",
        help=f"Shrink each axis by N before vision processing (default: {DEFAULT_DOWNSAMPLE}). Use 1 to disable.",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Print a per-step timing breakdown after processing all images.",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: input directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    if not images:
        print(f"No images found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(images)} image(s) from {args.input_dir}  [downsample={args.downsample}x]")
    state    = AxisState()
    profiler = StepTimes() if args.profile else None
    for img_path in images:
        process_image(img_path, args.output_dir, args.tick_min_length, state, args.downsample, profiler)

    print(f"Done. Debug images written to {args.output_dir}")
    if profiler:
        profiler.report()


if __name__ == "__main__":
    main()