#!/usr/bin/env python3
"""
Stress-tests pyramid TIFF tile decoding against a clean single-threaded
reference, looking for tiles whose concurrently-decoded content doesn't
match a fresh, uncontended re-decode of the exact same region.

Run directly against the real file that's showing misplaced-tile content:

    python diagnose_tile_mismatch.py path/to/cool_rock.tif

Prints one line per mismatch found, with the level/tile/box involved and
how much of the tile actually differs, then a summary count at the end.
Takes a few minutes on a large file -- it decodes every tile at every
level twice, once through a worker pool (mimicking the app) and once
sequentially on the main thread as a reference.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from UI.widgets.preview_overlay.large_image_source import _PyramidTiffBackend  # noqa: E402
import tifffile  # noqa: E402


def iter_tiles(backend: _PyramidTiffBackend, level_index: int, tile_size: int = 512):
    level = backend._levels[level_index]
    scale = level["scale_x"]
    lw, lh = level["width"], level["height"]
    for ty in range(0, lh, tile_size):
        for tx in range(0, lw, tile_size):
            box = (
                round(tx * scale), round(ty * scale),
                round(min(lw, tx + tile_size) * scale), round(min(lh, ty + tile_size) * scale),
            )
            yield tx, ty, box


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--levels", type=int, default=None, help="only test the first N levels (default: all)")
    args = parser.parse_args()

    print(f"Opening {args.path} ...")
    tf = tifffile.TiffFile(str(args.path))
    series = tf.series[0]
    if not series.is_pyramidal:
        print("Not detected as a pyramidal TIFF -- nothing to test.")
        return

    tf_ref = tifffile.TiffFile(str(args.path))
    backend_ref = _PyramidTiffBackend(tf_ref, series.pages[0].imagewidth, series.pages[0].imagelength)

    tf_pool = tifffile.TiffFile(str(args.path))
    backend_pool = _PyramidTiffBackend(tf_pool, series.pages[0].imagewidth, series.pages[0].imagelength)

    n_levels = len(backend_ref._levels)
    if args.levels is not None:
        n_levels = min(n_levels, args.levels)
    print(f"{n_levels} pyramid level(s) to test, {args.workers} workers for the concurrent pass")

    total_tiles = 0
    mismatches = 0
    t_start = time.time()

    executor = ThreadPoolExecutor(max_workers=args.workers)

    for level_index in range(n_levels):
        tiles = list(iter_tiles(backend_pool, level_index))
        print(f"level {level_index} ({backend_ref._levels[level_index]['width']}x"
              f"{backend_ref._levels[level_index]['height']}): {len(tiles)} tiles")

        def decode_pool(item):
            tx, ty, box = item
            return tx, ty, box, backend_pool.decode_region(level_index, box)

        futures = [executor.submit(decode_pool, item) for item in tiles]
        pool_results = {}
        for fut in futures:
            tx, ty, box, arr = fut.result()
            pool_results[(tx, ty)] = (box, arr)

        for tx, ty, box in tiles:
            total_tiles += 1
            _, pool_arr = pool_results[(tx, ty)]
            ref_arr = backend_ref.decode_region(level_index, box)
            if pool_arr.shape != ref_arr.shape or not np.array_equal(pool_arr, ref_arr):
                mismatches += 1
                diff = np.abs(pool_arr.astype(int) - ref_arr.astype(int)).sum(axis=2)
                bad_frac = (diff > 0).mean()
                bad_cols = np.where(diff.sum(axis=0) > 0)[0]
                bad_rows = np.where(diff.sum(axis=1) > 0)[0]
                col_range = f"{bad_cols.min()}-{bad_cols.max()}" if len(bad_cols) else "none"
                row_range = f"{bad_rows.min()}-{bad_rows.max()}" if len(bad_rows) else "none"
                print(
                    f"  MISMATCH level={level_index} tile=({tx},{ty}) box={box} "
                    f"bad_pixel_frac={bad_frac:.3f} bad_cols={col_range} bad_rows={row_range}"
                )

    executor.shutdown(wait=True)
    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s -- {mismatches}/{total_tiles} tiles mismatched between "
          f"concurrent and reference decode.")
    if mismatches == 0:
        print("No mismatches found in this run. The bug may need repeated runs to catch "
              "(if it's a rare race) -- try running this a few more times.")

    backend_ref.close()
    backend_pool.close()


if __name__ == "__main__":
    main()
