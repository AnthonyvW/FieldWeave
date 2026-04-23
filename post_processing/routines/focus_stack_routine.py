"""
focus_stack_routine.py

Post-processing routine that focus-stacks a folder of images using the
Tenengrad hard-selection pipeline from :mod:`focus_stack`.

Typical usage::

    routine = FocusStackRoutine(
        settings,
        input_folder="/path/to/stack/input",
        output_path="/path/to/stack/output.jpeg",
    )
    manager.start_routine(routine)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

from common.fieldweaveConfig import FieldWeaveSettings
from post_processing.routines.post_processing_routine import PostProcessingRoutine
from post_processing.routines.focus_stack import (
    collect_paths,
    load_images,
    cull_unfocused_images,
    align_images,
    fill_alignment_gaps,
    crop_to_valid_union,
    compute_focus_maps,
    select_best_pixels,
    save_depth_map,
    save_bg_mask,
    save_score_maps,
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class FocusStackConfig:
    """
    All tunable parameters for :class:`FocusStackRoutine`.

    Defaults mirror the recommended command-line invocation::

        python main.py <input> -o <output>
            --depth-radius 1
            --no-align
            --warp euclidean
            --smooth-source 15

    Parameters
    ----------
    depth_map_path:
        If set, save a greyscale source-frame depth map to this path.
        Pass an empty string to explicitly disable depth map saving when
        ``ZStackScan`` would otherwise set a default path.
    output_extension:
        File extension (without leading dot) used when ``ZStackScan``
        constructs the stacked output path automatically (default: ``jpeg``).
    sigma:
        Gaussian smoothing radius for focus maps. Larger = smoother region
        boundaries.
    sobel_ksize:
        Sobel kernel size for Tenengrad (1, 3, 5, or 7).
    score_power:
        Exponent applied to the raw Tenengrad score before smoothing. Increase
        to 3-4 if halo contamination persists.
    no_align:
        Skip ECC alignment. Use when images are already registered.
    warp:
        ECC warp model used when alignment is enabled.
    crop:
        Trim output to the bounding box where every frame contributed real
        pixel data (intersection). Without this, the union is used and gap
        pixels are filled from nearest-neighbour frames.
    depth_radius:
        Restrict pixel selection to a window of [peak-R, peak+R] frames around
        each pixel's best-focus frame. Prevents distant frames from stealing
        pixels due to diffraction halos.
    smooth_source:
        Apply two passes of median filtering (radius R) to the source-frame map
        after selection. Removes isolated outlier frame assignments.
    blend_low_confidence:
        Blend frames by focus-score weight for pixels whose winner confidence
        falls below this threshold. Near 0 in featureless regions where hard
        selection introduces noise.
    fill_quiet:
        Fill featureless low-detail pixels by propagating the frame assignment
        from their nearest sharp neighbours rather than using argmax.
    fill_quiet_search_radius:
        Gaussian propagation radius in pixels for fill_quiet.
    bg_percentile:
        Enable background halo suppression. Pass a float for a percentile lower
        bound, or None to use pure Otsu thresholding. Leave as _DISABLED to
        disable entirely.
    bg_search_radius:
        Search radius (pixels) when locating a halo-safe background frame.
    bg_blend_radius:
        Feather the subject/background boundary by this many pixels. Requires
        bg_percentile to be set.
    bg_mask_path:
        If set, save the subject/background classification mask to this path.
    cull:
        Discard wholly out-of-focus frames before stacking.
    cull_threshold:
        Fraction of the sharpest frame's focus score below which a frame is
        culled when cull is True.
    debug_scores:
        Save per-frame Tenengrad score maps to a focus_scores/ subdirectory
        next to the output file.
    """

    # Sentinel used to distinguish "disabled" from None (which means "otsu mode")
    # for bg_percentile. Not part of the public interface.
    _BG_DISABLED: float | None = field(default=object(), init=False, repr=False, compare=False)  # type: ignore[assignment]

    depth_map_path: str | None = None
    output_extension: str = "jpeg"

    sigma: float = 5.0
    sobel_ksize: int = 5
    score_power: float = 2.0

    no_align: bool = True
    warp: str = "euclidean"
    crop: bool = False

    depth_radius: int | None = 1
    smooth_source: int | None = 15
    blend_low_confidence: float | None = None

    fill_quiet: bool = False
    fill_quiet_search_radius: int = 32

    # bg_percentile sentinel: use the class-level _BG_DISABLED to mean "off".
    # Set to None for pure-Otsu mode; set to a float for percentile+Otsu mode.
    bg_percentile: float | None | object = field(default=None, init=False)
    bg_search_radius: int = 64
    bg_blend_radius: int | None = None
    bg_mask_path: str | None = None

    cull: bool = False
    cull_threshold: float = 0.19
    debug_scores: bool = False

    def __post_init__(self) -> None:
        # Default bg_percentile to the disabled sentinel so callers opt in explicitly.
        self.bg_percentile = self._BG_DISABLED  # type: ignore[assignment]

    @property
    def bg_enabled(self) -> bool:
        return self.bg_percentile is not self._BG_DISABLED

    @property
    def bg_percentile_value(self) -> float | None:
        """Numeric percentile threshold, or None for pure-Otsu mode."""
        if not self.bg_enabled:
            return None
        val = self.bg_percentile
        if val is None or val is self._BG_DISABLED:
            return None
        return float(val)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FocusStackResult:
    """
    Produced by :class:`FocusStackRoutine` on successful completion.

    Attach a done-callback or inspect this attribute after
    :meth:`~PostProcessingRoutine.wait` returns.
    """

    output_path: str
    """Absolute path to the saved focus-stacked image."""

    depth_map_path: str | None
    """Absolute path to the saved depth map, or None if not requested."""

    image_width: int
    image_height: int

    frame_count: int
    """Number of input frames that were stacked."""

    result_rgb: np.ndarray
    """
    Final stacked image in RGB888 order, shape (H, W, 3), dtype uint8.
    Safe to wrap in QImage directly on the GUI thread after wait() returns.
    """


# ---------------------------------------------------------------------------
# Routine
# ---------------------------------------------------------------------------

class FocusStackRoutine(PostProcessingRoutine):
    """
    Focus-stack a folder of images using Tenengrad hard-selection.

    Parameters
    ----------
    settings:
        Application-wide settings.
    input_folder:
        Directory containing the input images (all supported extensions).
    output_path:
        Destination file for the stacked result (e.g. ``output.jpeg``).
    config:
        Tunable stacking parameters. Defaults match the recommended preset.
        If not supplied, a :class:`FocusStackConfig` with default values is
        used.
    """

    job_name = "Focus Stack"

    def __init__(
        self,
        settings: FieldWeaveSettings,
        input_folder: str,
        output_path: str,
        config: FocusStackConfig | None = None,
    ) -> None:
        super().__init__(settings)
        self.input_folder = input_folder
        self.output_path = output_path
        self.config = config or FocusStackConfig()
        self.result: FocusStackResult | None = None

    def steps(self) -> Generator[None, None, None]:
        cfg = self.config
        input_dir = Path(self.input_folder)
        out_path = Path(self.output_path)

        # ----------------------------------------------------------------
        # Step 1: collect and load images
        # ----------------------------------------------------------------
        self._set_status("Collecting images", 0, 6)

        image_paths = collect_paths([str(input_dir)])
        images = load_images(image_paths)

        yield

        # ----------------------------------------------------------------
        # Step 2 (optional): cull wholly out-of-focus frames
        # ----------------------------------------------------------------
        self._set_status("Culling frames", 1, 6)

        if cfg.cull:
            images, image_paths = cull_unfocused_images(
                images,
                image_paths,
                threshold=cfg.cull_threshold,
                ksize=cfg.sobel_ksize,
            )

        ref_shape = images[0].shape[:2]
        for i, img in enumerate(images[1:], start=1):
            if img.shape[:2] != ref_shape:
                raise ValueError(
                    f"Image {image_paths[i]} has shape {img.shape[:2]} but "
                    f"reference is {ref_shape}. All images must have identical dimensions."
                )

        yield

        # ----------------------------------------------------------------
        # Step 3: align
        # ----------------------------------------------------------------
        self._set_status("Aligning images", 2, 6)

        valid_union: np.ndarray | None = None
        valid_intersection: np.ndarray | None = None

        if cfg.no_align:
            aligned = images
        else:
            warp_mode = {
                "translation": cv2.MOTION_TRANSLATION,
                "euclidean":   cv2.MOTION_EUCLIDEAN,
                "affine":      cv2.MOTION_AFFINE,
                "homography":  cv2.MOTION_HOMOGRAPHY,
            }[cfg.warp]
            aligned, valid_masks = align_images(images, warp_mode=warp_mode)
            aligned = fill_alignment_gaps(aligned, valid_masks)
            aligned, valid_union, valid_intersection = crop_to_valid_union(
                aligned, valid_masks,
            )

        yield

        # ----------------------------------------------------------------
        # Step 4: compute focus maps
        # ----------------------------------------------------------------
        self._set_status("Computing focus maps", 3, 6)

        focus_maps = compute_focus_maps(
            aligned,
            sigma=cfg.sigma,
            ksize=cfg.sobel_ksize,
            power=cfg.score_power,
            valid_mask=valid_intersection if not cfg.no_align else None,
        )

        yield

        # ----------------------------------------------------------------
        # Step 5: composite
        # ----------------------------------------------------------------
        self._set_status("Compositing frames", 4, 6)

        result_img, source_map, bg_mask = select_best_pixels(
            aligned,
            focus_maps,
            depth_radius=cfg.depth_radius,
            smooth_radius=cfg.smooth_source,
            blend_low_confidence=cfg.blend_low_confidence,
            bg_threshold_percentile=cfg.bg_percentile_value if cfg.bg_enabled else None,
            bg_halo_suppression=cfg.bg_enabled,
            bg_search_radius=cfg.bg_search_radius,
            bg_blend_radius=cfg.bg_blend_radius if cfg.bg_enabled else None,
            fill_quiet=cfg.fill_quiet,
            fill_quiet_search_radius=cfg.fill_quiet_search_radius,
        )

        # Optional crop to valid intersection
        if cfg.crop and not cfg.no_align and valid_intersection is not None:
            irows = np.any(valid_intersection, axis=1)
            icols = np.any(valid_intersection, axis=0)
            if irows.any() and icols.any():
                iy0 = int(np.argmax(irows))
                iy1 = int(len(irows) - 1 - np.argmax(irows[::-1]))
                ix0 = int(np.argmax(icols))
                ix1 = int(len(icols) - 1 - np.argmax(icols[::-1]))
                result_img = result_img[iy0:iy1 + 1, ix0:ix1 + 1]
                source_map = source_map[iy0:iy1 + 1, ix0:ix1 + 1]
                if bg_mask is not None:
                    bg_mask = bg_mask[iy0:iy1 + 1, ix0:ix1 + 1]

        yield

        # ----------------------------------------------------------------
        # Step 6: save outputs
        # ----------------------------------------------------------------
        self._set_status("Saving results", 5, 6)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        params: list[int] = []
        suffix = out_path.suffix.lower()
        if suffix in {".tif", ".tiff"}:
            params = [cv2.IMWRITE_TIFF_COMPRESSION, 5]
        elif suffix == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 6]

        cv2.imwrite(str(out_path), result_img, params)

        n = len(images)
        saved_depth_map_path: str | None = None

        if cfg.depth_map_path:
            save_depth_map(source_map, n, Path(cfg.depth_map_path))
            saved_depth_map_path = cfg.depth_map_path

        if cfg.bg_mask_path:
            if bg_mask is not None:
                save_bg_mask(bg_mask, Path(cfg.bg_mask_path))

        if cfg.debug_scores:
            scores_dir = out_path.parent / "focus_scores"
            save_score_maps(focus_maps, scores_dir, image_paths)

        h, w = result_img.shape[:2]

        channels = result_img.shape[2] if result_img.ndim == 3 else 1
        if channels >= 3:
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        else:
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)

        self.result = FocusStackResult(
            output_path=str(out_path.resolve()),
            depth_map_path=saved_depth_map_path,
            image_width=w,
            image_height=h,
            frame_count=n,
            result_rgb=result_rgb,
        )

        self._set_status("Done", 6, 6)
        yield