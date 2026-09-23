"""
Detecting, generating, and replacing pyramidal TIFFs -- the app's
version of the misc/image_viewer/generate_pyramid_tiff.py experiment.
Deliberately UI-free: CaptureControlWidget supplies every prompt and
decides what to do when a step here fails.

Generation uses libvips.

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

pyvips is imported lazily so the rest of the app still runs on an
install that predates it being added to requirements.txt -- callers
check is_available() before offering generation.
"""

from __future__ import annotations

import math
import os
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import tifffile

from common.logger import info, warning

# Matches TILE_SIZE in UI/widgets/preview_overlay/large_image_source.py --
# when they agree, most of that reader's virtual tile requests land on
# exactly one on-disk segment instead of needing several to stitch
# together, which under concurrent load means far fewer separate
# lock-protected file reads to decode the same view.
DEFAULT_TILE_SIZE = 512
DEFAULT_QUALITY = 90
DEFAULT_COMPRESSION = "deflate"

BYTES_PER_GIB = 1024**3

# Opening a flat image this size already takes long enough that adding
# pyramids is worth offering for next time.
OFFER_MIN_BYTES = 200 * 1024 * 1024
TIFF_SUFFIXES = frozenset({".tif", ".tiff"})

# Rough heuristic from observed runs: per-percent step time tracks input
# file size at roughly (GiB rounded up) + 1 seconds. Used as a prior on
# the per-percent rate, weighted as if it were this many percentage
# points of real samples -- real runs show that weight correcting the
# first percent point's misleadingly fast reading (thread/buffer
# warm-up) at the start without still biasing the estimate once actual
# samples dominate.
PRIOR_WEIGHT_PERCENT = 2

# Real multi-minute runs consistently run a bit longer than the average
# rate predicts in the second half (final pyramid levels and file close
# apparently cost more than the linear-in-percent model assumes), so the
# estimate quietly underestimates there. Padding it out corrects that and
# is a safer direction to be wrong in than promising an early finish.
SAFETY_MARGIN = 1.15

# (percent, step_seconds, elapsed_seconds, remaining_seconds)
ProgressCallback = Callable[[int, float, float, float], None]


def is_available() -> bool:
    try:
        import pyvips  # noqa: F401
    except (ImportError, OSError):
        # OSError: the binding installed but the libvips shared library
        # itself couldn't be found/loaded.
        return False
    return True


def is_pyramidal_tiff(path: Path) -> bool:
    if path.suffix.lower() not in TIFF_SUFFIXES:
        return False
    try:
        with tifffile.TiffFile(path) as tf:
            return bool(tf.series[0].is_pyramidal)
    except (OSError, ValueError, KeyError, IndexError):
        return False


def estimate_seconds_per_percent(file_size_bytes: int) -> float:
    size_gib = file_size_bytes / BYTES_PER_GIB
    return math.ceil(size_gib) + 1


def estimate_total_seconds(file_size_bytes: int) -> float:
    return estimate_seconds_per_percent(file_size_bytes) * 100


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {secs} s" if secs else f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


class _ProgressReporter:
    def __init__(self, prior_seconds_per_percent: float, callback: ProgressCallback) -> None:
        self.start_time = time.monotonic()
        self.last_time = self.start_time
        self.last_percent = 0
        self.prior_seconds = prior_seconds_per_percent * PRIOR_WEIGHT_PERCENT
        self.callback = callback

    def __call__(self, image, progress) -> None:
        percent = progress.percent
        # The same percent otherwise repeats across several callbacks --
        # only report once it actually advances, so each report shows how
        # long that step took instead of overwriting the last one.
        if percent <= self.last_percent:
            return

        now = time.monotonic()
        step_duration = now - self.last_time
        elapsed = now - self.start_time
        self.last_time = now
        self.last_percent = percent

        seconds_per_percent = (self.prior_seconds + elapsed) / (PRIOR_WEIGHT_PERCENT + percent)
        seconds_remaining = seconds_per_percent * (100 - percent) * SAFETY_MARGIN

        self.callback(percent, step_duration, elapsed, seconds_remaining)


def write_pyramid_tiff(
    input_path: Path,
    output_path: Path,
    tile_size: int = DEFAULT_TILE_SIZE,
    compression: str = DEFAULT_COMPRESSION,
    quality: int = DEFAULT_QUALITY,
    progress: ProgressCallback | None = None,
) -> None:
    """Raises pyvips.Error if libvips can't read the input or write the output."""
    import pyvips

    # libvips' operation cache otherwise keeps the input file open after
    # the save returns, which on Windows blocks overwriting the original
    # with the result.
    pyvips.cache_set_max(0)
    image = pyvips.Image.new_from_file(str(input_path), access="sequential")
    if progress is not None:
        prior_seconds_per_percent = estimate_seconds_per_percent(input_path.stat().st_size)
        image.set_progress(True)
        image.signal_connect("eval", _ProgressReporter(prior_seconds_per_percent, progress))
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


def offer_size(path: Path) -> int | None:
    """*path*'s size in bytes if adding pyramids to it is worth offering, else None."""
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size <= OFFER_MIN_BYTES or is_pyramidal_tiff(path):
        return None
    if not is_available():
        info("pyramid_tiff: pyvips/libvips unavailable, not offering to add pyramids")
        return None
    return size


def overwrite_target(path: Path) -> Path:
    # A pyramid is always a TIFF, so "overwriting" a PNG/JPEG means
    # replacing it with a same-named .tif rather than keeping a
    # misleading extension.
    return path if path.suffix.lower() in TIFF_SUFFIXES else path.with_suffix(".tif")


def default_new_target(path: Path) -> Path:
    return path.with_name(f"{path.stem}_pyramids.tif")


def with_tiff_suffix(path: Path) -> Path:
    return path if path.suffix.lower() in TIFF_SUFFIXES else path.with_name(f"{path.name}.tif")


def measurements_sidecar(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


def same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def is_locked(path: Path) -> bool:
    """Whether another program holds *path* open in a way that blocks overwriting it (Windows share modes -- never true on Linux/macOS)."""
    try:
        with open(path, "r+b"):
            return False
    except PermissionError:
        return True
    except OSError:
        return False


@dataclass
class PyramidJob:
    source: Path
    target: Path
    # A PNG/JPEG overwritten with a pyramid becomes a same-named .tif, so
    # the original has to be removed separately afterwards.
    delete_source: bool
    copy_measurements_from: Path | None = None

    @classmethod
    def overwrite(cls, source: Path) -> PyramidJob:
        target = overwrite_target(source)
        return cls(source, target, delete_source=target != source)

    @property
    def partial(self) -> Path:
        """Where generation writes, so a failure part-way never leaves a truncated image at the target."""
        return self.target.with_name(f"{self.target.stem}.partial{self.target.suffix}")

    @property
    def target_sidecar(self) -> Path:
        return measurements_sidecar(self.target)

    def carryable_measurements(self) -> Path | None:
        """The source's measurements file if it exists and wouldn't already match the target's name."""
        sidecar = measurements_sidecar(self.source)
        if not sidecar.exists() or sidecar == self.target_sidecar:
            return None
        return sidecar

    def files_to_replace(self) -> list[Path]:
        """Existing files this job will overwrite or delete."""
        candidates = [self.target] + ([self.source] if self.delete_source else [])
        return [p for p in candidates if p.exists()]

    def install(self) -> None:
        """Move the finished partial file over the target. Raises PermissionError if another program has the target locked."""
        os.replace(self.partial, self.target)

    def remove_source(self) -> None:
        self.source.unlink()

    def copy_measurements(self) -> None:
        if self.copy_measurements_from is not None:
            shutil.copyfile(self.copy_measurements_from, self.target_sidecar)

    def discard_partial(self) -> None:
        try:
            self.partial.unlink(missing_ok=True)
        except OSError as exc:
            warning(f"pyramid_tiff: failed to remove {self.partial}: {exc}")


class PyramidConversion:
    """
    Writes a job's partial file on a background thread. Every attribute
    is plain data written only by that thread -- a UI polls ``done``,
    ``percent`` and ``remaining`` rather than being called back, so
    nothing here ever runs on or touches the UI thread. Installing the
    result is left to the caller (see PyramidJob.install), since that
    may need to wait on the user closing a program holding the target.
    """

    def __init__(self, job: PyramidJob) -> None:
        self.job = job
        self.done = False
        self.error: str | None = None
        self.percent = 0
        self.remaining: float | None = None

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            write_pyramid_tiff(self.job.source, self.job.partial, progress=self._on_progress)
        except Exception as exc:
            # Anything escaping would kill this thread and leave the
            # poller waiting on `done` forever.
            self.error = f"{type(exc).__name__}: {exc}"
            self.job.discard_partial()
        self.done = True

    def _on_progress(self, percent: int, step: float, elapsed: float, remaining: float) -> None:
        self.percent = percent
        self.remaining = remaining
