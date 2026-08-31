"""
focus_stack_routine.py

Post-processing routines for focus stacking images using the focusweave library.

Two implementations are provided:

- :class:`QueuedFocusStackRoutine` — reads a completed folder of images and
  stacks them in one pass using the standard ``focusweave.run`` pipeline.
  Faster for large batches where all images are already on disk (e.g. unattended
  long imaging sessions).

- :class:`StreamingFocusStackRoutine` — accepts images fed incrementally via
   `~StreamingFocusStackRoutine.add_image`, performing alignment and
  culling upfront as each frame arrives.  Slower overall for already-captured
  images but reduces total wall time when images can be pushed in as they are
  acquired.

Typical usage — queued::

    routine = QueuedFocusStackRoutine(
        settings,
        input_folder="/path/to/stack/input",
        output_path="/path/to/stack/output.jpeg",
    )
    manager.start_routine(routine)

Typical usage — streaming::

    routine = StreamingFocusStackRoutine(
        settings,
        output_path="/path/to/stack/output.jpeg",
        reference_size=(width, height),
    )
    manager.start_routine(routine)

    for frame in acquisition_loop():
        routine.add_image(frame)

    routine.finish()
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generator

import numpy as np
from PIL import Image

from focusweave import FocusStackConfig, RunResult, run
from focusweave.streaming_stack import StreamingFocusStacker

from common.fieldweaveConfig import FieldWeaveSettings
from common.app_context import get_app_context
from common.read_metadata import build_exif_bytes, build_png_info, extract_dpi, read_metadata
from common.setting_types import FileFormat
from post_processing.routines.post_processing_routine import PostProcessingRoutine
from common.logger import info, warning, error


def _save_stack_result(
    image: np.ndarray,
    out_path: Path,
    fmt: FileFormat,
    jpeg_quality: int,
    metadata: dict[str, Any] | None = None,
) -> bool:
    # Let Pillow infer the format from out_path's extension instead of passing
    # a format string ourselves -- avoids the FileFormat/Pillow name mismatch
    # entirely (e.g. "jpg" vs the "JPEG" identifier Pillow expects).
    save_kwargs: dict[str, Any] = {"quality": jpeg_quality} if fmt in (FileFormat.JPEG, FileFormat.JPG) else {}

    if metadata:
        dpi = extract_dpi(metadata)
        if dpi is not None:
            save_kwargs["dpi"] = (dpi, dpi)
        if fmt == FileFormat.PNG:
            save_kwargs["pnginfo"] = build_png_info(metadata)
        else:
            exif_bytes = build_exif_bytes(metadata)
            if exif_bytes is not None:
                save_kwargs["exif"] = exif_bytes

    try:
        Image.fromarray(image).save(out_path, **save_kwargs)
    except OSError as exc:
        error(f"Failed to save stacked image to {out_path}: {exc}")
        return False

    return True


# ---------------------------------------------------------------------------
# Shared configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class FocusStackRoutineConfig:
    """
    Tunable parameters shared by both focus stack routine implementations.

    Maps directly onto :class:`focusweave.FocusStackConfig` and
    :class:`focusweave.streaming_stack.StreamingFocusStacker`.

    Parameters
    ----------
    no_align:
        Skip ECC alignment. Use when images are already registered.
    keep_size:
        Keep output the same size as the input images.
    crop:
        Crop output to the intersection of all transformed image extents.
    no_fill:
        Fill border regions with black instead of reflecting edge pixels.
    reference:
        Index of the alignment reference image. -1 = middle image.
    cull:
        Discard out-of-focus frames before stacking. Value is the fraction
        of the peak score below which a frame is dropped. None = disabled.
    global_align:
        Align every image directly to the reference instead of chaining.
    no_rotation:
        Suppress rotation correction during alignment.
    no_scale:
        Suppress scale correction during alignment.
    no_shear:
        Suppress shear correction during alignment.
    no_translation:
        Suppress translation correction during alignment.
    full_res:
        Run fine ECC alignment at full resolution instead of the 2048px cap.
    min_shift:
        Minimum shift in pixels before alignment is applied.
    levels:
        Laplacian pyramid levels. 0 = auto-detect from image size.
    sharpness:
        Weight sharpness exponent. Higher = harder winner-take-all selection.
    workers:
        Number of parallel workers for stacking.
    slab:
        (size, overlap) tuple to enable slabbing, or None to disable.
    recursive_slab:
        Recursively slab the slab results until they fit in one pass.
    output_extension:
        File extension used when constructing the output path automatically.
    jpeg_quality:
        Quality setting used when saving JPEG output.
    """

    no_align: bool = False
    keep_size: bool = False
    crop: bool = False
    no_fill: bool = False
    reference: int = -1
    cull: float | None = None
    global_align: bool = False
    no_rotation: bool = False
    no_scale: bool = False
    no_shear: bool = False
    no_translation: bool = False
    full_res: bool = False
    min_shift: float = 5.0
    levels: int = 0
    sharpness: float = 4.0
    workers: int = 3
    slab: tuple[int, int] | None = None
    recursive_slab: bool = False
    output_extension: str = "jpeg"
    jpeg_quality: int = 95


# ---------------------------------------------------------------------------
# Shared result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FocusStackResult:
    """
    Produced by either focus stack routine on successful completion.

    Retrieved via ``routine.result.get("focus_stack")`` after
     `~PostProcessingRoutine.wait` returns, or from the
    :class:`RoutineResult` passed to the ``on_complete`` callback.
    """

    output_path: str
    image_width: int
    image_height: int
    frame_count: int
    result_rgb: np.ndarray
    """Final stacked image in RGB888 order, shape (H, W, 3), dtype uint8."""


# ---------------------------------------------------------------------------
# Queued routine
# ---------------------------------------------------------------------------

class QueuedFocusStackRoutine(PostProcessingRoutine):
    """
    Focus-stack a completed folder of images using the standard focusweave pipeline.

    Reads all images from *input_folder* and stacks them in one pass via
    ``focusweave.run``.  Prefer this when all images are already on disk and
    throughput matters more than latency (e.g. unattended long imaging sessions).

    Parameters
    ----------
    settings:
        Application-wide settings.
    input_folder:
        Directory containing the input images.
    output_path:
        Destination file for the stacked result (e.g. ``output.jpeg``).
    config:
        Tunable stacking parameters. If not supplied, defaults are used.
    progress_start:
        Progress value (out of 100) at the start of this routine.
        Adjust when this routine is a sub-phase of a larger operation.
    progress_end:
        Progress value (out of 100) at the end of this routine.
    """

    job_name = "Focus Stack"

    def __init__(
        self,
        settings: FieldWeaveSettings,
        input_folder: str,
        output_path: str,
        config: FocusStackRoutineConfig | None = None,
        progress_start: int = 0,
        progress_end: int = 100,
    ) -> None:
        super().__init__(settings)
        self.input_folder = input_folder
        self.output_path = output_path
        self.config = config or FocusStackRoutineConfig()
        self._progress_start = progress_start
        self._progress_end = progress_end

    def _map_progress(self, fraction: float) -> int:
        span = self._progress_end - self._progress_start
        return self._progress_start + int(fraction * span)

    def steps(self) -> Generator[None, None, None]:
        cfg = self.config
        input_dir = Path(self.input_folder)
        out_path = Path(self.output_path)

        self._set_status("Stacking images", self._progress_start, 100)

        def _progress(fraction: float, stage: str, message: str) -> None:
            capped = min(fraction, 0.99)
            self._set_status(message or stage, self._map_progress(capped), 100)

        def _interrupt() -> bool:
            return self._check_stop()

        fw_cfg = FocusStackConfig(
            images=input_dir,
            no_align=cfg.no_align,
            keep_size=cfg.keep_size,
            crop=cfg.crop,
            no_fill=cfg.no_fill,
            reference=cfg.reference,
            cull=cfg.cull,
            global_align=cfg.global_align,
            no_rotation=cfg.no_rotation,
            no_scale=cfg.no_scale,
            no_shear=cfg.no_shear,
            no_translation=cfg.no_translation,
            full_res=cfg.full_res,
            min_shift=cfg.min_shift,
            levels=cfg.levels,
            sharpness=cfg.sharpness,
            workers=cfg.workers,
            slab=cfg.slab,
            only_slab=False,
            recursive_slab=cfg.recursive_slab,
            interrupt=_interrupt,
        )

        result: RunResult = run(fw_cfg, progress=_progress)

        if self._check_stop() or result.image is None:
            self._set_result(success=False)
            return

        input_files = sorted(p for p in input_dir.iterdir() if p.is_file())
        frame_count = len(input_files)
        metadata = read_metadata(input_files[0]) if input_files else None

        yield

        save_progress = self._map_progress(0.99)
        self._set_status("Saving result", save_progress, 100)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        fmt = get_app_context().camera.settings.fformat
        if not _save_stack_result(result.image, out_path, fmt, cfg.jpeg_quality, metadata):
            self._set_result(success=False)
            return

        h, w = result.image.shape[:2]

        self._set_result(
            success=True,
            focus_stack=FocusStackResult(
                output_path=str(out_path.resolve()),
                image_width=w,
                image_height=h,
                frame_count=frame_count,
                result_rgb=result.image,
            ),
        )

        self._set_status("Done", self._progress_end, 100)
        yield


# ---------------------------------------------------------------------------
# Streaming routine
# ---------------------------------------------------------------------------

class StreamingFocusStackRoutine(PostProcessingRoutine):
    """
    Focus-stack images fed incrementally via  `add_image`.

    Alignment and culling are performed upfront as each frame arrives, reducing
    the work left to do at finalisation time.  This is slower than
    :class:`QueuedFocusStackRoutine` when all images are already on disk, but
    reduces total wall time when images can be pushed in as they are acquired
    (e.g. live during a Z-stack scan).

    Usage pattern::

        routine = StreamingFocusStackRoutine(
            settings,
            output_path="output.jpeg",
            reference_size=(width, height),
        )
        manager.start_routine(routine)

        for frame in acquisition_loop():
            routine.add_image(frame)

        routine.finish()

     `add_image` is safe to call from any thread (including the acquisition
    thread) while the routine is running.   `finish` signals that no more
    images are coming and blocks until the stacker finalises and saves the result.
    Do not call  `add_image` after  `finish`.

    Parameters
    ----------
    settings:
        Application-wide settings.
    output_path:
        Destination file for the stacked result (e.g. ``output.jpeg``).
    reference_size:
        ``(width, height)`` of the input frames. Required by
        :class:`~focusweave.streaming_stack.StreamingFocusStacker` to
        pre-allocate its internal buffers.
    reference_metadata:
        Metadata (from  `read_metadata.read_metadata`) to carry onto the
        saved output, e.g. DPI and camera settings. Streaming frames arrive
        as bare arrays with no file to read this from, so the caller must
        supply it up front. None = no metadata written.
    config:
        Tunable stacking parameters. If not supplied, defaults are used.
    progress_start:
        Progress value (out of 100) at the start of this routine.
    progress_end:
        Progress value (out of 100) at the end of this routine.
    """

    job_name = "Focus Stack"

    def __init__(
        self,
        settings: FieldWeaveSettings,
        output_path: str,
        reference_size: tuple[int, int],
        reference_metadata: dict[str, Any] | None = None,
        config: FocusStackRoutineConfig | None = None,
        progress_start: int = 0,
        progress_end: int = 100,
    ) -> None:
        super().__init__(settings)
        self.output_path = output_path
        self.reference_size = reference_size
        self.reference_metadata = reference_metadata
        self.config = config or FocusStackRoutineConfig()
        self._progress_start = progress_start
        self._progress_end = progress_end

        self._image_queue: list[np.ndarray] = []
        self._queue_lock = threading.Lock()
        self._queue_event = threading.Event()
        self._finish_requested = threading.Event()
        self._frame_count = 0

        # Wired by the caller after construction, before start().
        # Signature: (frame: np.ndarray, count: int) -> None
        # Called from the post-processing thread after each frame is added.
        self.on_preview: Callable[[np.ndarray, int], None] | None = None
        self.preview_scale: float = 1.0

    def _map_progress(self, fraction: float) -> int:
        span = self._progress_end - self._progress_start
        return self._progress_start + int(fraction * span)

    def add_image(self, image: np.ndarray) -> None:
        """Push a frame into the stacker.

        Safe to call from any thread while the routine is running.  The image
        must be a uint8 or uint16 RGB ndarray with the dimensions matching
        *reference_size*.

        Do not call after  `finish`.
        """
        with self._queue_lock:
            self._image_queue.append(image)
        self._queue_event.set()

    def finish(self) -> None:
        """Signal that all images have been added and trigger finalisation.

        The routine will drain any remaining queued images, call
        ``stacker.finish()``, save the result, and complete.  This method
        returns immediately; use  `~PostProcessingRoutine.wait` to block
        until the routine has fully finished.
        """
        self._finish_requested.set()
        self._queue_event.set()

    def steps(self) -> Generator[None, None, None]:
        cfg = self.config
        out_path = Path(self.output_path)

        self._set_status("Waiting for images", self._progress_start, 100)

        stacker = StreamingFocusStacker(
            reference_size=self.reference_size,
            reference=cfg.reference,
            cull_threshold=cfg.cull,
            workers=cfg.workers,
            slab=cfg.slab,
            only_slab=False,
            recursive_slab=cfg.recursive_slab,
            on_preview=self.on_preview,
            preview_scale=self.preview_scale,
        )

        # Drain the queue until finish() is called and no images remain.
        # Yield after each batch so the base runner can honour pause/stop
        # between iterations rather than blocking inside a single next() call.
        while not (self._finish_requested.is_set() and self._image_queue == []):
            if self._check_stop():
                self._set_result(success=False)
                return

            self._queue_event.wait(timeout=0.1)
            self._queue_event.clear()

            with self._queue_lock:
                batch = self._image_queue[:]
                self._image_queue.clear()

            for image in batch:
                if self._check_stop():
                    self._set_result(success=False)
                    return
                stacker.add_image(image)
                self._frame_count += 1
                self._set_status(
                    f"Acquired {self._frame_count} frame(s)",
                    self._map_progress(0.0),
                    100,
                )

            yield

        if self._check_stop():
            self._set_result(success=False)
            return

        self._set_status("Stacking", self._map_progress(0.01), 100)

        def _progress(fraction: float, stage: str, message: str) -> None:
            capped = min(fraction, 0.99)
            self._set_status(
                message or stage,
                self._map_progress(0.01 + capped * 0.98),
                100,
            )

        result: RunResult = stacker.finish(
            keep_size=cfg.keep_size,
            progress=_progress,
        )

        if self._check_stop() or result.image is None:
            self._set_result(success=False)
            return

        yield

        save_progress = self._map_progress(0.99)
        self._set_status("Saving result", save_progress, 100)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        fmt = get_app_context().camera.settings.fformat
        if not _save_stack_result(result.image, out_path, fmt, cfg.jpeg_quality, self.reference_metadata):
            self._set_result(success=False)
            return

        h, w = result.image.shape[:2]

        self._set_result(
            success=True,
            focus_stack=FocusStackResult(
                output_path=str(out_path.resolve()),
                image_width=w,
                image_height=h,
                frame_count=self._frame_count,
                result_rgb=result.image,
            ),
        )

        self._set_status("Done", self._progress_end, 100)
        yield