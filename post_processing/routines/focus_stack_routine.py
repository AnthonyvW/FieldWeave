"""
focus_stack_routine.py

Post-processing routine that focus-stacks a folder of images using the
focusweave library.

Typical usage::

    routine = FocusStackRoutine(
        settings,
        input_folder="/path/to/stack/input",
        output_path="/path/to/stack/output.jpeg",
    )
    manager.start_routine(routine)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
from PIL import Image

from focusweave import FocusStackConfig, RunResult, run

from common.fieldweaveConfig import FieldWeaveSettings
from post_processing.routines.post_processing_routine import PostProcessingRoutine


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class FocusStackRoutineConfig:
    """
    Tunable parameters for :class:`FocusStackRoutine`.

    Maps directly onto :class:`focusweave.FocusStackConfig`.

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
    dark_threshold:
        Luminance threshold below which chroma is suppressed toward neutral.
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
    dark_threshold: float = 30.0
    workers: int = 3
    slab: tuple[int, int] | None = None
    recursive_slab: bool = False
    output_extension: str = "jpeg"
    jpeg_quality: int = 95


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FocusStackResult:
    """
    Produced by :class:`FocusStackRoutine` on successful completion.

    Retrieved via ``routine.result.get("focus_stack")`` after
    :meth:`~PostProcessingRoutine.wait` returns, or from the
    :class:`RoutineResult` passed to the ``on_complete`` callback.
    """

    output_path: str
    image_width: int
    image_height: int
    frame_count: int
    result_rgb: np.ndarray
    """Final stacked image in RGB888 order, shape (H, W, 3), dtype uint8."""


# ---------------------------------------------------------------------------
# Routine
# ---------------------------------------------------------------------------

class FocusStackRoutine(PostProcessingRoutine):
    """
    Focus-stack a folder of images using the focusweave library.

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
        Progress value (out of 100) that represents 0% into this routine.
        Use when this routine runs as a sub-phase of a larger operation so
        the caller's progress bar advances smoothly through the correct slice.
    progress_end:
        Progress value (out of 100) that represents 100% into this routine.
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
        """Map a 0.0–1.0 fraction into the [progress_start, progress_end] range."""
        span = self._progress_end - self._progress_start
        return self._progress_start + int(fraction * span)

    def steps(self) -> Generator[None, None, None]:
        cfg = self.config
        input_dir = Path(self.input_folder)
        out_path = Path(self.output_path)

        # ----------------------------------------------------------------
        # Step 1: run focusweave pipeline
        # ----------------------------------------------------------------
        self._set_status("Stacking images", self._progress_start, 100)

        def _progress(fraction: float, stage: str, message: str) -> None:
            # Reserve the top of our range for the save step
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
            dark_threshold=cfg.dark_threshold,
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

        frame_count = len(list(input_dir.iterdir()))

        yield

        # ----------------------------------------------------------------
        # Step 2: save output
        # ----------------------------------------------------------------
        save_progress = self._map_progress(0.99)
        self._set_status("Saving result", save_progress, 100)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = out_path.suffix.lower().lstrip(".")
        fmt = "jpeg" if suffix in ("jpg", "jpeg") else suffix.upper()
        save_kwargs: dict = {"quality": cfg.jpeg_quality} if fmt == "jpeg" else {}
        Image.fromarray(result.image).save(out_path, fmt, **save_kwargs)

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