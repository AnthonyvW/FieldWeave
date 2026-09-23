"""
tree_core_stitch_routine.py

Stitches the frames of one tree core into a single long image.

The stitching itself is a port of ``misc/image_stitching/stitchCli.py``: SIFT +
FLANN registration inside the expected overlap region of each neighbouring
pair, followed by a hard-seam (no blend, no warp) composite.  Pairs that
cannot be registered, such as the out-of-focus tray past either end of the
core, split the sequence into runs and the longest run is kept.

Unlike the CLI, nothing about the layout has to be supplied:

- Frame order, sample orientation (``--vertical-core``) and direction
  (``--reverse``) come from the camera calibration together with each frame's
  stage position.  Without either, they are measured by matching features
  between whole neighbouring frames.
- The overlap is derived the same way (stage step / calibrated field of view,
  or the measured frame-to-frame shift) unless one is passed explicitly.
- The calibration slide is prepended when a calibration slide folder is
  given.  The composite is never cropped unless ``StitchConfig.crop`` is set.

Frames may be handed over before they exist on disk: a :class:`StitchFrame`
whose ``source`` is a :class:`QueuedFocusStackRoutine` is waited on and its
stacked output used, so stitching never starts before focus stacking is done.

Typical usage::

    routine = TreeCoreStitchRoutine(
        settings,
        tree_core_folder="/path/to/run/core_A",
        frames=[StitchFrame(stage_nm=pos, source=stack_routine), ...],
        axis="y",
        calibration_slide_folder="/path/to/run/calibration_slide",
    )
    manager.queue_routine(routine)

Result data keys
----------------
- ``output_path`` (:class:`str`): the saved composite.
- ``image_width`` / ``image_height`` (:class:`int`): composite size in pixels.
- ``frames_stitched`` (:class:`int`): frames in the stitched run.
- ``frames_total`` (:class:`int`): frames that were available to stitch.
- ``overlap`` (:class:`float`): overlap fraction used.
- ``vertical_core`` / ``reverse`` (:class:`bool`): the detected layout.
"""

from __future__ import annotations

import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from PIL import Image

from common.fieldweaveConfig import FieldWeaveSettings
from common.logger import debug, error, info, warning
from common.read_metadata import extract_dpi, read_metadata
from post_processing.routines.post_processing_routine import PostProcessingRoutine

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
STACKED_DIRNAME = "focus_stacked"
CALIBRATION_STEM = "calibration_slide"

_DEFAULT_OVERLAP = 0.35
_MIN_OVERLAP = 0.05
_MAX_OVERLAP = 0.95
_LAYOUT_SAMPLE_PAIRS = 5
_LAYOUT_MIN_MATCHES = 8

# Pillow refuses images above this by default as a decompression-bomb guard;
# a full-length core composite legitimately exceeds it.
Image.MAX_IMAGE_PIXELS = None


@dataclass
class StitchConfig:
    """Tunable stitching parameters, matching the CLI's options."""

    max_features: int = 500
    scale_factor: float = 0.25
    flann_checks: int = 12
    mask: bool = False
    crop: bool = False


@dataclass
class StitchFrame:
    """
    One frame to stitch.

    Parameters
    ----------
    path:
        The image file.  May be None when *source* will produce it.
    stage_nm:
        Stage coordinate along the scan axis where the frame was captured, or
        None if unknown.  Frames are only ordered by position when every frame
        has one; otherwise natural filename order is used.
    source:
        Routine that writes the image, e.g. a :class:`QueuedFocusStackRoutine`.
        It is waited on before stitching starts and its
        ``focus_stack.output_path`` result replaces *path*.  A failed source
        drops the frame.
    """

    path: Path | None = None
    stage_nm: int | None = None
    source: PostProcessingRoutine | None = None


@dataclass
class StitchLayout:
    paths: list[Path]
    vertical_core: bool
    reverse: bool
    overlap: float
    basis: str


class RegistrationError(RuntimeError):
    """A pair carries too little detail in its overlap to align.  Past either
    end of a core that means the out-of-focus tray, not a stitching fault."""


def natural_sort_key(path: Path) -> list[int | str]:
    return [int(chunk) if chunk.isdigit() else chunk.lower()
            for chunk in re.split(r"(\d+)", path.name)]


def collect_frames(tree_core_folder: Path) -> list[StitchFrame]:
    """Frames from ``<folder>/focus_stacked`` if present, else the folder itself."""
    stacked = tree_core_folder / STACKED_DIRNAME
    image_dir = stacked if stacked.is_dir() else tree_core_folder
    paths = sorted(
        (p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=natural_sort_key,
    )
    return [StitchFrame(path=p) for p in paths]


def find_calibration_slide(cal_dir: Path) -> tuple[Path | None, bool]:
    """Return ``(slide_path, already_horizontal)`` for the slide in *cal_dir*,
    or ``(None, False)`` when the folder or the image is absent."""
    if not cal_dir.is_dir():
        return None, False

    entries = sorted(cal_dir.iterdir())
    slide_path = next(
        (e for e in entries if e.stem.lower() == CALIBRATION_STEM and e.suffix.lower() in IMAGE_EXTENSIONS),
        None,
    )
    if slide_path is None:
        return None, False

    # DPI.txt is only written once the slide has been measured, a pass that
    # leaves the image horizontal regardless of how the core was scanned.
    already_horizontal = any(e.name.lower() == "dpi.txt" for e in entries)
    return slide_path, already_horizontal


def prepend_calibration_slide(composite: np.ndarray, slide: np.ndarray) -> np.ndarray:
    """Butt the slide against the left edge of the composite, vertically centred."""
    height = max(composite.shape[0], slide.shape[0])
    width = slide.shape[1] + composite.shape[1]
    canvas = np.zeros((height, width, 3), np.uint8)

    slide_y = (height - slide.shape[0]) // 2
    canvas[slide_y:slide_y + slide.shape[0], 0:slide.shape[1]] = slide

    comp_y = (height - composite.shape[0]) // 2
    canvas[comp_y:comp_y + composite.shape[0], slide.shape[1]:width] = composite
    return canvas


def load_image(path: Path, vertical_core: bool) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    if vertical_core:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _whole_frame_shift(path_a: Path, path_b: Path, cfg: StitchConfig) -> tuple[float, float, int, int] | None:
    """
    Median displacement of scene content from frame *a* to frame *b*, as
    ``(dx, dy, width, height)`` at the downscaled size, or None if the pair
    has too few matches.  Positive dx means *b* shows the scene to the right
    of *a*; positive dy means below.
    """
    grey = []
    for path in (path_a, path_b):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        grey.append(cv2.resize(img, (0, 0), fx=cfg.scale_factor, fy=cfg.scale_factor))

    sift = cv2.SIFT_create(nfeatures=cfg.max_features * 4)
    kp_a, des_a = sift.detectAndCompute(grey[0], None)
    kp_b, des_b = sift.detectAndCompute(grey[1], None)
    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return None

    matcher = cv2.FlannBasedMatcher({"algorithm": 0, "trees": 5}, {"checks": cfg.flann_checks * 4})
    good = [m for m, n in (pair for pair in matcher.knnMatch(des_a, des_b, k=2) if len(pair) == 2)
            if m.distance < 0.7 * n.distance]
    if len(good) < _LAYOUT_MIN_MATCHES:
        return None

    dx = float(np.median([kp_a[m.queryIdx].pt[0] - kp_b[m.trainIdx].pt[0] for m in good]))
    dy = float(np.median([kp_a[m.queryIdx].pt[1] - kp_b[m.trainIdx].pt[1] for m in good]))
    h, w = grey[0].shape[:2]
    return dx, dy, w, h


def estimate_layout_from_images(paths: list[Path], cfg: StitchConfig) -> tuple[bool, bool, float] | None:
    """
    Measure ``(vertical_core, reverse, overlap)`` by matching whole
    neighbouring frames.  Pairs are sampled from the middle of the sequence
    since the ends are often bare tray.  Returns None if no pair matched.
    """
    n_pairs = len(paths) - 1
    first = max(0, (n_pairs - _LAYOUT_SAMPLE_PAIRS) // 2)
    shifts = [
        s for s in (
            _whole_frame_shift(paths[i], paths[i + 1], cfg)
            for i in range(first, min(n_pairs, first + _LAYOUT_SAMPLE_PAIRS))
        )
        if s is not None
    ]
    if not shifts:
        return None

    dx = statistics.median(s[0] for s in shifts)
    dy = statistics.median(s[1] for s in shifts)
    width, height = shifts[0][2], shifts[0][3]

    vertical = abs(dy) > abs(dx)
    along = dy if vertical else dx
    extent = height if vertical else width
    return vertical, along < 0, 1.0 - abs(along) / extent


class _Stitcher:
    """SIFT + FLANN registration in the overlap region, hard-seam composite."""

    def __init__(self, overlap: float, cfg: StitchConfig) -> None:
        self.overlap = overlap
        self.cfg = cfg
        self.max_offset = 0.0
        self.runs: list[tuple[int, int]] = []
        self.stitched_range: tuple[int, int] | None = None
        self.composite: np.ndarray | None = None

    def calculate_offset(self, img1: np.ndarray, img2: np.ndarray) -> tuple[float, float]:
        cfg = self.cfg
        overlap_px = int(img2.shape[1] * self.overlap)

        i1 = cv2.cvtColor(
            cv2.resize(img1[:, -overlap_px:, :], (0, 0), fx=cfg.scale_factor, fy=cfg.scale_factor),
            cv2.COLOR_BGR2GRAY,
        )
        i2 = cv2.cvtColor(
            cv2.resize(img2[:, :overlap_px, :], (0, 0), fx=cfg.scale_factor, fy=cfg.scale_factor),
            cv2.COLOR_BGR2GRAY,
        )

        mask = None
        if cfg.mask:
            height, width = i1.shape[:2]
            mask = np.zeros(i1.shape[:2], np.uint8)
            quarter = round(height / 4)
            mask[quarter:height - quarter, 0:width] = 255

        sift = cv2.SIFT_create(nfeatures=cfg.max_features)
        kp1, des1 = sift.detectAndCompute(i1, mask)
        kp2, des2 = sift.detectAndCompute(i2, mask)
        n1 = 0 if des1 is None else len(des1)
        n2 = 0 if des2 is None else len(des2)
        if n1 < 2 or n2 < 2:
            raise RegistrationError(f"not enough SIFT keypoints in the overlap region ({n1} and {n2} descriptors)")

        flann = cv2.FlannBasedMatcher({"algorithm": 0, "trees": 5}, {"checks": cfg.flann_checks})
        matches = flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in (pair for pair in matches if len(pair) == 2) if m.distance < 0.7 * n.distance]
        if not good:
            raise RegistrationError(
                f"no SIFT match survived the ratio test ({n1} and {n2} descriptors, {len(matches)} raw matches)"
            )

        src = np.float32([kp1[m.queryIdx].pt for m in good])
        dst = np.float32([kp2[m.trainIdx].pt for m in good])
        x_offset = int(np.median(src[:, 0] - dst[:, 0]))
        y_offset = int(np.median(src[:, 1] - dst[:, 1]))
        return x_offset / cfg.scale_factor, y_offset / cfg.scale_factor

    def stitch_pair(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        x_offset, y_offset = self.calculate_offset(img1, img2)

        # Seam halfway through the overlap so both images lose an equal share.
        x_seam = int(img1.shape[1] - (img2.shape[1] * self.overlap * 0.5) + x_offset)
        partial_image = int(img2.shape[1] * self.overlap * 0.5)

        if x_seam < 0 or x_seam > img1.shape[1]:
            raise RuntimeError(
                f"computed seam position ({x_seam} px) falls outside image 1's width ({img1.shape[1]} px); "
                f"SIFT measured an x offset of {x_offset:.1f} px, which is not consistent with an overlap "
                f"of {self.overlap:.2f}"
            )

        self.max_offset = max(self.max_offset, y_offset)
        width = x_seam + (img2.shape[1] - partial_image)
        height = img2.shape[0] + abs(int(self.max_offset))
        right = slice(x_seam, x_seam + img2.shape[1] - partial_image)

        if y_offset < 0.0:
            shift = int(abs(y_offset))
            comp = np.zeros((height + shift, width, 3), np.uint8)
            self.max_offset += shift
            comp[shift:img1.shape[0] + shift, 0:x_seam] = img1[:, 0:x_seam]
            comp[0:img2.shape[0], right] = img2[:, partial_image:]
        else:
            comp = np.zeros((height, width, 3), np.uint8)
            comp[0:img1.shape[0], 0:x_seam] = img1[:, 0:x_seam]
            comp[int(y_offset):img2.shape[0] + int(y_offset), right] = img2[:, partial_image:]
        return comp

    def _close_run(self, best: dict | None, composite: np.ndarray | None, first: int, last: int) -> dict | None:
        """Keep whichever of *best* and the run just closed covers more images.
        Only the winner's composite is retained since each can be hundreds of MB."""
        if composite is None:
            return best
        self.runs.append((first, last))
        if best is None or last - first > best["last"] - best["first"]:
            return {"first": first, "last": last, "offset": self.max_offset, "composite": composite}
        return best

    def stitch(self, paths: list[Path], vertical_core: bool) -> Generator[int, None, None]:
        """
        Stitch the longest run of consecutive images that align, yielding the
        number of pairs processed after each one.  The result is left in
        :attr:`composite`.
        """
        self.runs = []
        best: dict | None = None
        composite: np.ndarray | None = None
        first = 0
        index = 0
        self.max_offset = 0.0

        while index + 1 < len(paths):
            left = composite if composite is not None else load_image(paths[index], vertical_core)
            right = load_image(paths[index + 1], vertical_core)

            try:
                composite = self.stitch_pair(left, right)
            except RegistrationError as exc:
                warning(
                    f"[TreeCoreStitch] Could not align {paths[index].name} with {paths[index + 1].name}: {exc}"
                )
                best = self._close_run(best, composite, first, index)
                composite = None
                self.max_offset = 0.0
                first = index + 1

            index += 1
            yield index

        best = self._close_run(best, composite, first, index)
        if best is None:
            raise RuntimeError("no pair of images could be aligned; every overlap region was too featureless")

        self.stitched_range = (best["first"], best["last"])
        self.max_offset = best["offset"]
        composite = best["composite"]

        kept = best["last"] - best["first"] + 1
        if kept < len(paths):
            dropped = [r for r in self.runs if r != self.stitched_range]
            unaligned = [paths[i].name for i in range(len(paths)) if not any(f <= i <= l for f, l in self.runs)]
            if unaligned:
                warning(f"[TreeCoreStitch] Dropped {len(unaligned)} image(s) that aligned with nothing: {', '.join(unaligned)}")
            if dropped:
                warning(
                    f"[TreeCoreStitch] Dropped {len(dropped)} shorter aligned run(s), so a section of the core is "
                    f"missing: {'; '.join(f'images {f + 1}-{l + 1}' for f, l in dropped)}"
                )
            info(f"[TreeCoreStitch] Stitched images {best['first'] + 1}-{best['last'] + 1} ({kept} of {len(paths)})")

        if self.cfg.crop and self.max_offset:
            drift = int(self.max_offset)
            composite = composite[drift:composite.shape[0] - drift, :]

        self.composite = composite


class TreeCoreStitchRoutine(PostProcessingRoutine):
    """
    Stitch one tree core's frames into a single image.

    Parameters
    ----------
    settings:
        Application-wide settings.
    tree_core_folder:
        The core's output folder.  The composite is written here as
        ``<folder name>.tiff`` unless *output_path* is given.
    frames:
        Frames to stitch.  When None they are read from
        ``<tree_core_folder>/focus_stacked`` (or the folder itself) in natural
        filename order, and the layout is measured from the images.
    axis:
        Stage axis the frames were captured along (``"x"`` or ``"y"``).
        Used with the camera calibration to derive the layout.
    calibration_slide_folder:
        Folder holding ``calibration_slide.<ext>`` to prepend to the left of
        the core, or None to leave it off.
    overlap:
        Overlap fraction between neighbouring frames.  None derives it
        automatically.
    output_path:
        Where to write the composite.
    config:
        Stitching parameters.  Defaults to :class:`StitchConfig` (no crop).
    """

    job_name = "Stitch Tree Core"

    def __init__(
        self,
        settings: FieldWeaveSettings,
        tree_core_folder: str | Path,
        frames: list[StitchFrame] | None = None,
        *,
        axis: str = "y",
        calibration_slide_folder: str | Path | None = None,
        overlap: float | None = None,
        output_path: str | Path | None = None,
        config: StitchConfig | None = None,
    ) -> None:
        super().__init__(settings)
        self.tree_core_folder = Path(tree_core_folder)
        self.frames = frames
        self.axis = axis.lower()
        self.calibration_slide_folder = Path(calibration_slide_folder) if calibration_slide_folder is not None else None
        self.overlap = overlap
        self.output_path = (
            Path(output_path) if output_path is not None
            else self.tree_core_folder / f"{self.tree_core_folder.name}.tiff"
        )
        self.config = config or StitchConfig()
        self.job_name = f"Stitch ({self.tree_core_folder.name})"

    def steps(self) -> Generator[None, None, None]:
        frames = self.frames if self.frames is not None else collect_frames(self.tree_core_folder)

        pending = [f for f in frames if f.source is not None]
        for i, frame in enumerate(pending):
            self._set_status(f"Waiting for focus stacking ({i}/{len(pending)})", 0, 100)
            while not frame.source.wait(timeout=0.25):
                yield
            result = frame.source.result
            stacked = result.get("focus_stack") if result is not None and result.success else None
            if stacked is None:
                warning(f"[TreeCoreStitch] Focus stack '{frame.source.job_name}' failed; leaving it out")
            frame.path = Path(stacked.output_path) if stacked is not None else None

        frames = [f for f in frames if f.path is not None and f.path.is_file()]
        if len(frames) < 2:
            error(f"[TreeCoreStitch] Need at least 2 images to stitch {self.tree_core_folder}, found {len(frames)}")
            self._set_result(success=False)
            return

        self._set_status("Determining layout", 5, 100)
        layout = self._determine_layout(frames)
        yield

        info(
            f"[TreeCoreStitch] {len(layout.paths)} images, "
            f"{'vertical' if layout.vertical_core else 'horizontal'} core"
            f"{', reversed' if layout.reverse else ''}, overlap {layout.overlap:.1%} ({layout.basis})"
        )

        stitcher = _Stitcher(layout.overlap, self.config)
        n_pairs = len(layout.paths) - 1
        try:
            for done in stitcher.stitch(layout.paths, layout.vertical_core):
                self._set_status(f"Stitching pair {done}/{n_pairs}", 10 + int(80 * done / n_pairs), 100)
                yield
        except RuntimeError as exc:
            error(f"[TreeCoreStitch] Stitching {self.tree_core_folder.name} failed: {exc}")
            self._set_result(success=False)
            return

        composite = stitcher.composite
        if composite is None:
            self._set_result(success=False)
            return

        if self.calibration_slide_folder is not None:
            composite = self._prepend_slide(composite, layout.vertical_core)

        self._set_status("Saving", 92, 100)
        yield

        dpi = extract_dpi(read_metadata(layout.paths[0]) or {})
        if not self._save(composite, dpi):
            self._set_result(success=False)
            return

        h, w = composite.shape[:2]
        first, last = stitcher.stitched_range
        info(f"[TreeCoreStitch] Wrote {w}x{h} composite to {self.output_path}")
        self._set_result(
            success=True,
            output_path=str(self.output_path.resolve()),
            image_width=w,
            image_height=h,
            frames_stitched=last - first + 1,
            frames_total=len(layout.paths),
            overlap=layout.overlap,
            vertical_core=layout.vertical_core,
            reverse=layout.reverse,
        )
        self._set_status("Done", 100, 100)
        yield

    def _determine_layout(self, frames: list[StitchFrame]) -> StitchLayout:
        positioned = all(f.stage_nm is not None for f in frames)
        if positioned:
            frames = sorted(frames, key=lambda f: f.stage_nm)
        paths = [f.path for f in frames]

        from common.app_context import get_app_context
        mv = get_app_context().machine_vision
        calibration = mv.calibration if mv is not None else None

        if positioned and calibration is not None:
            geometry = calibration.stage_axis_imaging(self.axis)
            vertical = geometry.image_axis == "vertical"
            # Stitching runs left to right (top to bottom before rotation), so
            # the next frame must show the scene to the right; that happens at
            # increasing stage positions only when content moves the other way.
            reverse = geometry.content_moves_forward
            step_nm = statistics.median(abs(b.stage_nm - a.stage_nm) for a, b in zip(frames, frames[1:]))
            measured = 1.0 - step_nm / geometry.fov_nm
            basis = "camera calibration"
        else:
            estimate = estimate_layout_from_images(paths, self.config)
            if estimate is None:
                warning("[TreeCoreStitch] Could not measure the layout from the images; assuming a horizontal core in order")
                vertical, reverse, measured = False, False, _DEFAULT_OVERLAP
                basis = "default"
            else:
                vertical, reverse, measured = estimate
                basis = "image matching"

        if self.overlap is not None:
            overlap = self.overlap
            basis = "manual"
        else:
            overlap = min(max(measured, _MIN_OVERLAP), _MAX_OVERLAP)

        if reverse:
            paths.reverse()
        return StitchLayout(paths, vertical, reverse, overlap, basis)

    def _prepend_slide(self, composite: np.ndarray, vertical_core: bool) -> np.ndarray:
        slide_path, already_horizontal = find_calibration_slide(self.calibration_slide_folder)
        if slide_path is None:
            warning(f"[TreeCoreStitch] No calibration slide found in {self.calibration_slide_folder}; skipping")
            return composite
        slide = cv2.imread(str(slide_path))
        if slide is None:
            warning(f"[TreeCoreStitch] Could not read calibration slide {slide_path}; skipping")
            return composite
        if vertical_core and not already_horizontal:
            slide = cv2.rotate(slide, cv2.ROTATE_90_COUNTERCLOCKWISE)
        debug(f"[TreeCoreStitch] Prepending calibration slide {slide_path}")
        return prepend_calibration_slide(composite, slide)

    def _save(self, composite: np.ndarray, dpi: float | None) -> bool:
        save_kwargs: dict[str, object] = {}
        if dpi is not None:
            save_kwargs["dpi"] = (dpi, dpi)
        if self.output_path.suffix.lower() in (".tif", ".tiff"):
            save_kwargs["compression"] = "tiff_lzw"
        try:
            os.makedirs(self.output_path.parent, exist_ok=True)
            Image.fromarray(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)).save(self.output_path, **save_kwargs)
        except OSError as exc:
            error(f"[TreeCoreStitch] Failed to save {self.output_path}: {exc}")
            return False
        return True
