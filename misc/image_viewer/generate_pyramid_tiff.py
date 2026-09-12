#!/usr/bin/env python3
"""
Generate a pyramidal TIFF from an input image using libvips.

libvips processes the image as a demand-driven pipeline rather than
decoding it fully into RAM, so peak memory stays roughly constant
regardless of image size -- unlike a numpy/tifffile approach, which has
to hold the whole decoded raster (and every pyramid level) resident at
once. This is what vips's own tiffsave does, equivalent to running:

    vips tiffsave INPUT OUTPUT --tile --pyramid --compression deflate

Defaults to deflate: lossless (jpeg's default produced visible blocking
on sharp edges -- it discards data) and, unlike zstd, built directly
into libtiff rather than linked as an optional external codec, so it
works on every libvips build without needing one compiled with zstd
support (not guaranteed -- the prebuilt pyvips wheel on Windows lacks
it, for one).

Requires libvips itself, not just the pyvips Python binding:
    Debian/Ubuntu: sudo apt install libvips
    macOS:         brew install vips
    pip:           pip install pyvips

Usage:
    python generate_pyramid_tiff.py INPUT_IMAGE OUTPUT.tiff
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pyvips

# Matches TILE_SIZE in UI/widgets/preview_overlay/large_image_source.py --
# when they agree, most of that reader's virtual tile requests land on
# exactly one on-disk segment instead of needing several to stitch
# together, which under concurrent load means far fewer separate
# lock-protected file reads to decode the same view.
DEFAULT_TILE_SIZE = 512
DEFAULT_QUALITY = 90

# The first percent point consistently completes far faster than the
# steady-state rate (initial thread/buffer warm-up), which skews an
# average built from only one or two samples. Withhold the estimate
# until enough samples have diluted that outlier -- real runs show the
# average-rate error dropping from ~40-60% at 1-2% to under 10% by here.
MIN_PERCENT_FOR_ESTIMATE = 3


class _ProgressReporter:
    def __init__(self) -> None:
        self.start_time = time.monotonic()
        self.last_time = self.start_time
        self.last_percent = 0

    def __call__(self, image: pyvips.Image, progress: pyvips.VipsProgress) -> None:
        percent = progress.percent
        # The same percent otherwise repeats across several callbacks --
        # only report a new line once it actually advances, so each line
        # shows how long that step took instead of overwriting the last one.
        if percent <= self.last_percent:
            return

        now = time.monotonic()
        step_duration = now - self.last_time
        elapsed = now - self.start_time
        self.last_time = now
        self.last_percent = percent

        if percent < MIN_PERCENT_FOR_ESTIMATE:
            eta = "estimating remaining time..."
        else:
            # Per-step timing is noisy enough that extrapolating from just
            # the last step swings wildly once multiplied by (100 - percent).
            # The average rate since start is far steadier while still
            # tracking a process whose overall rate drifts over the run.
            seconds_remaining = (elapsed / percent) * (100 - percent)
            eta = f"~{seconds_remaining:.0f}s remaining"

        print(f"Generating pyramidal TIFF: {percent:3d}% (step {step_duration:.1f}s, elapsed {elapsed:.1f}s, {eta})")


def write_pyramid_tiff(
    input_path: Path, output_path: Path, tile_size: int, compression: str, quality: int,
) -> None:
    image = pyvips.Image.new_from_file(str(input_path), access="sequential")
    image.set_progress(True)
    image.signal_connect("eval", _ProgressReporter())
    image.tiffsave(
        str(output_path),
        tile=True,
        tile_width=tile_size,
        tile_height=tile_size,
        pyramid=True,
        compression=compression,
        Q=quality,
        bigtiff=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a pyramidal TIFF from an input image using libvips")
    parser.add_argument("input", type=Path, help="Path to the source image")
    parser.add_argument("output", type=Path, help="Path to write the pyramidal TIFF")
    parser.add_argument(
        "--tile-size", type=int, default=DEFAULT_TILE_SIZE,
        help="Tile size for every pyramid level (default: %(default)s)",
    )
    parser.add_argument(
        "--compression", default="deflate",
        help="libvips TIFF compression: deflate, lzw, none, zstd (all lossless -- zstd may "
             "not be available on every libvips build), or jpeg (lossy -- smaller files but "
             "visible artifacts on sharp edges) (default: %(default)s)",
    )
    parser.add_argument(
        "--quality", type=int, default=DEFAULT_QUALITY,
        help="JPEG quality, only used with --compression jpeg (default: %(default)s)",
    )
    args = parser.parse_args()

    write_pyramid_tiff(args.input, args.output, args.tile_size, args.compression, args.quality)
    print(f"Wrote pyramidal TIFF to {args.output}")


if __name__ == "__main__":
    main()
