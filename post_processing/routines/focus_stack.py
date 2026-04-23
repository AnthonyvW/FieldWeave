"""
Focus stacking using Tenengrad focus detection with hard pixel selection.

Pipeline:
  1. Load images
  2. Align all images to the reference (first) frame using ECC (Enhanced Correlation
     Coefficient) with a homography warp — handles translation, rotation, and minor
     perspective distortion.
  3. Compute per-pixel Tenengrad focus score for every aligned image by squaring the
     Sobel gradients in x and y.  Scores are smoothed with a Gaussian to avoid
     single-pixel noise spikes pulling in wrong source frames.
  4. For every pixel, pick the source image with the highest focus score.  In regions
     where no frame has strong focus (e.g. featureless backgrounds) the scores across
     frames are nearly equal, so hard selection produces a noisy result.  When
     blend_low_confidence is set, such pixels are instead composited by blending all
     frames weighted by their normalised focus scores, which averages out the noise.
  5. Save the result.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def load_images(paths: list[Path]) -> list[np.ndarray]:
    """Load images preserving bit-depth (16-bit TIFF etc.)."""
    images: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Could not read image: {p}")
        images.append(img)
    print(f"Loaded {len(images)} images  ({images[0].shape[1]}x{images[0].shape[0]}, "
          f"depth={images[0].dtype})")
    return images


def _numeric_sort_key(p: Path) -> tuple[list[int | str], str]:
    """
    Sort key that orders filenames by the sequence of integers embedded in them,
    falling back to lexicographic order for non-numeric parts.

    Examples (ascending):
        z1.tif, z2.tif, z10.tif   -- not z1, z10, z2
        frame_001.tif ... frame_100.tif
        0001.tif ... 9999.tif
    """
    parts: list[int | str] = [
        int(tok) if tok.isdigit() else tok
        for tok in re.split(r"(\d+)", p.stem)
    ]
    return parts, p.suffix.lower()


def collect_paths(inputs: list[str]) -> list[Path]:
    """
    Accept a mix of file paths and directories; return numerically sorted file list.

    Numeric sorting ensures that filenames such as z1.tif, z2.tif ... z10.tif are
    ordered by depth rather than lexicographically (which would give z1, z10, z2).
    When explicit file paths are supplied on the command line their order is
    preserved -- the caller is assumed to have already ordered them by depth.
    """
    dir_paths: list[Path] = []
    explicit_paths: list[Path] = []

    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            found = sorted(
                (f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS),
                key=_numeric_sort_key,
            )
            if not found:
                raise ValueError(f"No supported images found in directory: {p}")
            dir_paths.extend(found)
        elif p.is_file():
            if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {p}")
            explicit_paths.append(p)
        else:
            raise FileNotFoundError(f"Path does not exist: {p}")

    # Explicit files keep caller order (depth-ordered by convention);
    # directory files are numerically sorted.
    paths = explicit_paths + dir_paths
    if len(paths) < 2:
        raise ValueError("At least 2 images are required for focus stacking.")
    return paths


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def to_gray_8bit(img: np.ndarray) -> np.ndarray:
    """Convert any image to 8-bit grayscale for feature / ECC work."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    if gray.dtype != np.uint8:
        # Scale to 0-255
        mn, mx = gray.min(), gray.max()
        if mx > mn:
            gray = ((gray.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            gray = np.zeros_like(gray, dtype=np.uint8)
    return gray


def _forward_matrix(
    inv_matrix: np.ndarray,
    is_homography: bool,
) -> np.ndarray:
    """
    Convert a dst->src warp matrix (as returned by findTransformECC, which uses
    WARP_INVERSE_MAP convention) to a forward src->dst matrix.

    For affine/euclidean/translation: the input is 2x3; we embed it in 3x3,
    invert, and return the top 2 rows as a 2x3 forward matrix.
    For homography: the input is 3x3; we simply invert it.
    """
    if is_homography:
        return np.linalg.inv(inv_matrix.astype(np.float64)).astype(np.float32)
    M = np.vstack([inv_matrix.astype(np.float64), [0.0, 0.0, 1.0]])
    return np.linalg.inv(M)[:2].astype(np.float32)


def _transform_corners(
    w: int,
    h: int,
    fwd: np.ndarray,
    is_homography: bool,
) -> np.ndarray:
    """
    Map the four corners of a w x h canvas through a forward (src->dst) warp
    matrix and return them as a (4, 2) float32 array.
    """
    corners = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32,
    ).reshape(1, -1, 2)
    if is_homography:
        return cv2.perspectiveTransform(corners, fwd).reshape(-1, 2)
    return cv2.transform(corners, fwd).reshape(-1, 2)


def align_images(
    images: list[np.ndarray],
    *,
    warp_mode: int = cv2.MOTION_TRANSLATION,
    max_iterations: int = 200,
    termination_eps: float = 1e-4,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Align all images to the first image (reference) using ECC, expanding the
    output canvas so that no content is clipped by rotation or translation.

    ECC is robust to photometric differences (brightness/contrast variation
    across focus planes) which makes it well-suited for microscopy stacks.

    Canvas expansion
    ----------------
    findTransformECC returns a dst->src matrix (WARP_INVERSE_MAP convention).
    To expand the canvas correctly we need the *forward* (src->dst) matrix so
    we can project each frame's corners into output space and find a union
    bounding box.  The procedure is:

    Pass 1 -- solve all ECC warps; invert each matrix to get the forward map;
    project corners; accumulate the union bounding box (the reference frame's
    forward map is the identity so its corners are just the canvas corners).

    Pass 2 -- prepend a canvas-offset translation to each *forward* matrix so
    that the top-left corner of the union bbox maps to (0, 0); apply the
    adjusted forward matrix directly (no WARP_INVERSE_MAP flag).  The reference
    frame is padded via copyMakeBorder with the same offset so all frames are
    spatially registered on the enlarged canvas.

    Returns a tuple of:
        aligned     - images on the expanded canvas, spatially registered.
        valid_masks - (H, W) bool arrays; True = real pixel data, False = fill.
                      Built by warping an all-ones sentinel with INTER_NEAREST
                      (hard boundary) then eroding 1 px to strip the
                      anti-aliased fringe from INTER_LINEAR image warping and
                      the unreliable 1-px canvas edge present on every frame.
    """
    mode_name = {
        cv2.MOTION_TRANSLATION: "translation",
        cv2.MOTION_EUCLIDEAN:   "euclidean",
        cv2.MOTION_AFFINE:      "affine",
        cv2.MOTION_HOMOGRAPHY:  "homography",
    }.get(warp_mode, str(warp_mode))
    print(f"Aligning images to reference frame (ECC, warp={mode_name})...")

    ref_gray = to_gray_8bit(images[0])
    h, w = ref_gray.shape
    is_homography = warp_mode == cv2.MOTION_HOMOGRAPHY

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max_iterations,
        termination_eps,
    )

    # -- Pass 1: solve warps, project corners, find union bbox ------------------
    fwd_matrices: list[np.ndarray] = []
    ref_corners = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32,
    )
    all_corners: list[np.ndarray] = [ref_corners]

    for i, img in enumerate(images[1:], start=1):
        src_gray = to_gray_8bit(img)

        if is_homography:
            inv_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            inv_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            _, inv_matrix = cv2.findTransformECC(
                ref_gray, src_gray, inv_matrix, warp_mode, criteria,
                inputMask=None, gaussFiltSize=5,
            )
        except cv2.error as exc:
            print(f"  [warn] ECC failed for image {i} ({exc}); using identity warp.")

        fwd = _forward_matrix(inv_matrix, is_homography)
        fwd_matrices.append(fwd)
        all_corners.append(_transform_corners(w, h, fwd, is_homography))
        print(f"  ECC solved image {i + 1}/{len(images)}")

    all_pts = np.concatenate(all_corners, axis=0)
    x_min = float(np.floor(all_pts[:, 0].min()))
    y_min = float(np.floor(all_pts[:, 1].min()))
    x_max = float(np.ceil(all_pts[:, 0].max()))
    y_max = float(np.ceil(all_pts[:, 1].max()))

    out_w = int(round(x_max - x_min)) + 1
    out_h = int(round(y_max - y_min)) + 1
    tx = -x_min
    ty = -y_min
    print(f"  Expanded canvas: {w}x{h} -> {out_w}x{out_h} "
          f"(offset tx={tx:.1f}px, ty={ty:.1f}px)")

    # -- Pass 2: warp every frame onto the expanded canvas ----------------------
    erode_kernel = np.ones((3, 3), dtype=np.uint8)
    ones_src = np.ones((h, w), dtype=np.uint8)

    pad_top    = int(round(ty))
    pad_left   = int(round(tx))
    pad_bottom = out_h - h - pad_top
    pad_right  = out_w - w - pad_left
    ref_padded = cv2.copyMakeBorder(
        images[0],
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0,
    )
    aligned: list[np.ndarray] = [ref_padded]

    ref_sentinel = np.zeros((out_h, out_w), dtype=np.uint8)
    ref_sentinel[pad_top:pad_top + h, pad_left:pad_left + w] = 1
    valid_masks: list[np.ndarray] = [
        cv2.erode(ref_sentinel, erode_kernel, iterations=1).astype(bool)
    ]

    for i, fwd in enumerate(fwd_matrices, start=1):
        if is_homography:
            T = np.array(
                [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            adj_fwd = T @ fwd
            warped = cv2.warpPerspective(
                images[i], adj_fwd, (out_w, out_h), flags=cv2.INTER_LINEAR,
            )
            sentinel_warped = cv2.warpPerspective(
                ones_src, adj_fwd, (out_w, out_h), flags=cv2.INTER_NEAREST,
            )
        else:
            T = np.array(
                [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            fwd3 = np.vstack([fwd.astype(np.float64), [0.0, 0.0, 1.0]])
            adj_fwd = (T @ fwd3)[:2].astype(np.float32)
            warped = cv2.warpAffine(
                images[i], adj_fwd, (out_w, out_h), flags=cv2.INTER_LINEAR,
            )
            sentinel_warped = cv2.warpAffine(
                ones_src, adj_fwd, (out_w, out_h), flags=cv2.INTER_NEAREST,
            )

        eroded = cv2.erode(sentinel_warped, erode_kernel, iterations=1)
        valid_masks.append(eroded.astype(bool))
        aligned.append(warped)
        print(f"  Warped image {i + 1}/{len(images)}")

    return aligned, valid_masks


def crop_to_valid_union(
    images: list[np.ndarray],
    valid_masks: list[np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Crop all images to the bounding box of the *union* of all valid masks,
    removing only the outer padding on the expanded canvas where no frame has
    any real pixel data.

    Also returns the union mask and the intersection mask, both in the
    coordinate space of the cropped canvas, so the caller can:
      - pass valid_intersection to compute_focus_maps to keep border scores zeroed
      - use valid_intersection bbox to implement --crop (trim to every-frame region)

    The uncropped output canvas will therefore be as large as the largest single
    frame, with corner/edge regions filled by fill_alignment_gaps where only some
    frames cover them.

    Returns
    -------
    cropped              : images cropped to the union bounding box.
    valid_union_cropped  : (H, W) bool -- union mask in cropped-canvas space.
    valid_isect_cropped  : (H, W) bool -- intersection mask in cropped-canvas space.
    """
    union = valid_masks[0].copy()
    for m in valid_masks[1:]:
        union |= m

    intersection = valid_masks[0].copy()
    for m in valid_masks[1:]:
        intersection &= m

    rows = np.any(union, axis=1)
    cols = np.any(union, axis=0)

    if not rows.any() or not cols.any():
        raise ValueError(
            "No frame has any valid pixels -- "
            "check input images or use --no-align."
        )

    y0 = int(np.argmax(rows))
    y1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(len(cols) - 1 - np.argmax(cols[::-1]))

    cropped = [img[y0:y1 + 1, x0:x1 + 1] for img in images]
    union_cropped  = union       [y0:y1 + 1, x0:x1 + 1]
    isect_cropped  = intersection[y0:y1 + 1, x0:x1 + 1]

    orig_h, orig_w = valid_masks[0].shape
    print(f"Cropped to valid union: {orig_w}x{orig_h} -> "
          f"{x1 - x0 + 1}x{y1 - y0 + 1}")
    return cropped, union_cropped, isect_cropped


def fill_alignment_gaps(
    aligned: list[np.ndarray],
    valid_masks: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Fill invalid border pixels in each aligned frame by copying from the nearest
    frame (by index distance) that has valid data at each gap position.

    After warping, pixels that fall outside the source image boundary are filled
    with zeros.  The eroded valid masks also mark the outermost pixel row/column
    of every frame as gaps.  Without this function those zero regions produce
    large Sobel gradients that corrupt focus scores at the boundary, and the
    Gaussian smoothing then spreads that corruption inward.

    For each frame i, gap pixels are filled by walking outward in both directions
    (i-1, i+1, i-2, i+2, ...) and copying from the first frame that has valid
    data at each position.  Adjacent frames share nearly identical scene content
    in a focus stack so nearest-neighbour donation minimises visible differences.

    Pixels that are invalid in every frame retain the zero fill and a warning is
    printed.
    """
    n = len(aligned)
    filled = [img.copy() for img in aligned]

    for i in range(n):
        gap_mask = ~valid_masks[i]
        if not gap_mask.any():
            continue

        remaining = gap_mask.copy()

        for dist in range(1, n):
            if not remaining.any():
                break
            for j in (i - dist, i + dist):
                if j < 0 or j >= n:
                    continue
                can_fill = remaining & valid_masks[j]
                if not can_fill.any():
                    continue
                filled[i][can_fill] = aligned[j][can_fill]
                remaining[can_fill] = False

        if remaining.any():
            print(f"  [warn] Frame {i + 1}: {int(remaining.sum())} gap pixel(s) could "
                  f"not be filled (no valid neighbour in any frame); retaining zero fill.")

    return filled


# ---------------------------------------------------------------------------
# Tenengrad focus measure
# ---------------------------------------------------------------------------

def tenengrad_score(
    img: np.ndarray,
    ksize: int = 5,
    power: float = 2.0,
) -> np.ndarray:
    """
    Compute the Tenengrad focus score for every pixel.

    Tenengrad = (Gx² + Gy²) ^ power

    Two measures are taken against diffraction artefacts — the defocus halos that
    form around high-contrast edges (e.g. cell walls) and project spurious gradient
    energy into blurry interior regions:

    1.  CLAHE pre-normalisation.  Contrast-limited adaptive histogram equalisation
        is applied to the grayscale image before Sobel.  This compresses the large
        intensity difference at a bright edge so that its out-of-focus halo (which
        owes its apparent contrast entirely to that edge) contributes far less to the
        gradient score in the adjacent region.

    2.  Power exponentiation.  Raising the raw Tenengrad score to `power` (default 2)
        sharpens the score distribution: genuinely in-focus edges produce a compact
        spike of very high gradient magnitude, whereas defocus halos produce a
        broader plateau of moderate magnitude.  Squaring that difference makes the
        spike stand out much more strongly and suppresses the plateau.

    ksize=5 is the default (vs the classical ksize=3) because a slightly wider Sobel
    kernel is less sensitive to single-pixel shot noise, which otherwise competes
    with real edge responses.

    Returns a float32 array of the same H×W shape.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Normalise to 8-bit for CLAHE (which requires uint8)
    if gray.dtype != np.uint8:
        mn, mx = gray.min(), gray.max()
        if mx > mn:
            gray8 = ((gray.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            return np.zeros(gray.shape, dtype=np.float32)
    else:
        gray8 = gray

    # CLAHE: tileGridSize matches typical cell/feature scale; clipLimit caps
    # amplification so genuine flat regions don't get boosted into fake edges.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray8).astype(np.float32) / 255.0

    gx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=ksize)
    score = gx ** 2 + gy ** 2

    if power != 1.0:
        score = score ** power

    return score


def compute_focus_maps(
    images: list[np.ndarray],
    *,
    sigma: float = 5.0,
    ksize: int = 5,
    power: float = 2.0,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return a (N, H, W) float32 array of smoothed Tenengrad scores.

    Gaussian smoothing (radius = sigma) is applied to the raw score map so that
    a single sharp edge does not cause a neighbouring region to pick a different
    source frame than its surroundings.  Larger sigma = smoother region boundaries
    but slightly reduced spatial resolution of focus selection.

    The default sigma is 5 (up from 3) to better absorb the residual spread of
    diffraction halos after power exponentiation.
    """
    print(f"Computing Tenengrad focus maps (ksize={ksize}, power={power}, sigma={sigma})...")
    h, w = images[0].shape[:2]
    maps = np.zeros((len(images), h, w), dtype=np.float32)

    for i, img in enumerate(images):
        score = tenengrad_score(img, ksize=ksize, power=power)
        if valid_mask is not None:
            score[~valid_mask] = 0.0
        k = int(sigma * 6) | 1  # next odd integer >= 6σ
        smoothed = cv2.GaussianBlur(score, (k, k), sigma)
        if valid_mask is not None:
            smoothed[~valid_mask] = 0.0
        maps[i] = smoothed
        print(f"  Focus map {i + 1}/{len(images)}")

    return maps


# ---------------------------------------------------------------------------
# Culling: remove wholly out-of-focus frames before stacking
# ---------------------------------------------------------------------------

def _image_focus_score(img: np.ndarray, ksize: int = 5, top_percentile: float = 95.0) -> float:
    """
    Return a single scalar focus score for an image.

    Rather than using the mean of the whole score map (which is dominated by
    featureless background pixels with near-zero scores), we take the mean of
    the top `top_percentile` percent of pixel scores.  This measures how sharp
    the sharpest parts of the image are, which is what matters for culling:
    an image that has *any* well-focused region should be kept.

    Steps
    -----
    1. Compute the raw (un-smoothed) Tenengrad score so we are measuring
       per-pixel sharpness, not the smoothed version used for selection.
    2. Take the mean of scores at or above the given percentile threshold.
    """
    score = tenengrad_score(img, ksize=ksize, power=1.0)  # power=1 for raw magnitude
    threshold = float(np.percentile(score, top_percentile))
    top_scores = score[score >= threshold]
    return float(top_scores.mean()) if top_scores.size > 0 else 0.0


def cull_unfocused_images(
    images: list[np.ndarray],
    paths: list[Path],
    *,
    threshold: float = 0.05,
    ksize: int = 5,
    top_percentile: float = 95.0,
) -> tuple[list[np.ndarray], list[Path]]:
    """
    Remove images that have no sufficiently sharp region from the stack.

    Each image receives a focus score based on the mean Tenengrad response in
    its top `top_percentile` percent of pixels (see `_image_focus_score`).  An
    image is culled if its score falls below `threshold` × (score of the
    sharpest image in the stack).

    Parameters
    ----------
    images       : loaded image arrays.
    paths        : corresponding file paths (same order as images).
    threshold    : fraction of the peak score below which a frame is culled
                   (default 0.05 = 5 %).  Raise toward 1.0 to cull more
                   aggressively; lower toward 0.0 to keep almost everything.
    ksize        : Sobel kernel size passed to the focus scorer (default 5).
    top_percentile : percentile used when summarising each frame's score map
                   (default 95.0).

    Returns
    -------
    Filtered (images, paths) with culled frames removed.  At least 2 images
    are always retained (the two sharpest) to ensure the stack can proceed.

    Raises
    ------
    ValueError if fewer than 2 images survive culling (should not happen given
    the 2-image safety floor, but guards against degenerate inputs).
    """
    print(f"Culling unfocused images (threshold={threshold:.0%}, "
          f"top_percentile={top_percentile:.0f}th)...")

    scores = [_image_focus_score(img, ksize=ksize, top_percentile=top_percentile)
              for img in images]

    peak = max(scores)
    if peak == 0.0:
        print("  [warn] All images have zero focus score; skipping cull.")
        return images, paths

    cutoff = threshold * peak
    keep_flags = [s >= cutoff for s in scores]

    # Safety: always retain at least the two sharpest frames so the stack
    # has something to work with even when the threshold is set very high.
    if sum(keep_flags) < 2:
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for idx in ranked[:2]:
            keep_flags[idx] = True

    kept_images: list[np.ndarray] = []
    kept_paths: list[Path] = []
    n_culled = 0

    for img, path, score, keep in zip(images, paths, scores, keep_flags):
        status = "keep" if keep else "CULL"
        print(f"  [{status}] {path.name}  (score={score:.4g},"
              f" cutoff={cutoff:.4g})")
        if keep:
            kept_images.append(img)
            kept_paths.append(path)
        else:
            n_culled += 1

    print(f"Culled {n_culled}/{len(images)} image(s); "
          f"{len(kept_images)} frame(s) remaining.")

    if len(kept_images) < 2:
        raise ValueError(
            "Fewer than 2 images survived culling.  "
            "Lower --cull-threshold or disable --cull."
        )

    return kept_images, kept_paths


# ---------------------------------------------------------------------------
# Hard pixel selection (no blending)
# ---------------------------------------------------------------------------

def smooth_source_map(
    source_map: np.ndarray,
    radius: int,
) -> np.ndarray:
    """
    Clean up the source map using an iterative majority-vote (median) filter.

    Each pixel is replaced with the most common frame index in a square
    neighbourhood of side (2*radius + 1).  This removes isolated outlier
    frame assignments — pixels that were pulled to a distant frame by a
    noise spike or diffraction halo — without blurring the boundaries
    between legitimate in-focus bands, which span thousands of pixels.

    A single pass of median filtering on the integer index map is
    mathematically equivalent to majority voting over the neighbourhood,
    making it the right operation here (as opposed to a Gaussian blur,
    which would create non-integer interpolated frame indices that have
    no physical meaning).

    Two passes are used so that small clusters (not just single pixels)
    are also absorbed into their dominant surroundings, while large
    coherent bands are left untouched.
    """
    ksize = 2 * radius + 1
    # cv2.medianBlur requires uint8 or uint16; source_map is already uint8.
    smoothed = cv2.medianBlur(source_map, ksize)
    smoothed = cv2.medianBlur(smoothed, ksize)
    return smoothed


def _confidence_map(focus_maps: np.ndarray) -> np.ndarray:
    """
    Compute a per-pixel confidence score in [0, 1] that indicates how decisive
    the focus winner is relative to the rest of the stack.

    The metric is the normalised gap between the best and second-best frame score:

        confidence = (best - second) / (best + epsilon)

    A value near 1.0 means one frame is vastly sharper than all others (clear
    winner, hard selection is reliable).  A value near 0.0 means scores are nearly
    tied across frames (featureless region, blending is more appropriate).

    Returns a float32 (H, W) array.
    """
    # Partially sort: we only need the top-2 values along the frame axis.
    # np.partition is O(N) vs O(N log N) for a full sort.
    top2 = np.partition(focus_maps, -2, axis=0)[-2:]  # (2, H, W): [second, best]
    second = top2[0]
    best   = top2[1]
    eps = np.finfo(np.float32).eps
    confidence = (best - second) / (best + eps)
    return confidence.astype(np.float32)


def _blend_weighted(
    images: list[np.ndarray],
    focus_maps: np.ndarray,
) -> np.ndarray:
    """
    Composite all frames using per-pixel focus-score weights (soft selection).

    For each pixel the weight of frame i is:

        w_i = score_i / sum_j(score_j)

    This is the correct fallback for low-detail regions: when all frame scores
    are near zero and roughly equal, the weighted average simply returns the
    mean of all frames, which suppresses the random frame-selection noise that
    hard argmax would produce.

    Returns an array of the same dtype and shape as the input images.
    """
    src_dtype = images[0].dtype
    stack = np.stack(images, axis=0).astype(np.float64)  # (N, H, W[, C])

    # Normalise focus maps to per-pixel probability weights
    total = focus_maps.sum(axis=0, keepdims=True)  # (1, H, W)
    total = np.where(total > 0, total, 1.0)
    weights = (focus_maps / total).astype(np.float64)  # (N, H, W)

    if stack.ndim == 4:
        weights = weights[:, :, :, np.newaxis]  # broadcast over channels

    blended = (stack * weights).sum(axis=0)

    # Round and clip back to original integer dtype
    if np.issubdtype(src_dtype, np.integer):
        info = np.iinfo(src_dtype)
        blended = blended.round().clip(info.min, info.max)
    return blended.astype(src_dtype)


def _build_background_mask(
    focus_maps: np.ndarray,
    bg_threshold_percentile: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return a background mask, the per-pixel max-score map, and the effective
    threshold used, so the caller can build a soft transition alpha.

    Strategy
    --------
    A pixel is considered background when even its *best* focus score across the
    entire stack is low — i.e. it is never sharply in focus in any frame.  Subject
    pixels have at least one frame where they score highly.

    Thresholding
    ------------
    The per-pixel max-score distribution is typically bimodal: a low-score cluster
    (background, never sharply focused) and a high-score cluster (subject, sharply
    focused in at least one frame).  Otsu's method finds the threshold that
    maximises between-class variance, i.e. sits in the valley between the two
    modes.  This is far more robust than a fixed percentile because it adapts to
    the actual score distribution rather than assuming background occupies a fixed
    fraction of the image.

    The raw float scores are log-compressed and normalised to 8-bit before Otsu so
    that the extreme dynamic range of the Tenengrad measure (scores span many orders
    of magnitude) does not compress all the useful information into a few high bins.

    ``bg_threshold_percentile``, when given, is used as a *lower bound* on the
    Otsu threshold expressed as a percentile of the max-score distribution.  This
    lets you force more pixels into the background class when Otsu's split sits too
    high (e.g. if the background has some mildly textured regions that pull the
    valley upward).  Leave as None to use the pure Otsu threshold.

    Returns
    -------
    bg_mask      : (H, W) bool  — True where the pixel is classified as background.
    max_score    : (H, W) float32 — per-pixel peak focus score across all frames.
    effective_threshold : float — the threshold in raw score space (used to build
                          the soft transition alpha in the caller).
    """
    # Per-pixel maximum focus score across all frames.
    max_score = focus_maps.max(axis=0)  # (H, W) float32

    # Log-compress to 8-bit for Otsu (compresses the extreme dynamic range so the
    # histogram has meaningful structure across the full 0-255 range).
    log_score = np.log1p(max_score)
    log_min, log_max = float(log_score.min()), float(log_score.max())
    if log_max > log_min:
        norm8 = ((log_score - log_min) / (log_max - log_min) * 255).astype(np.uint8)
    else:
        norm8 = np.zeros_like(log_score, dtype=np.uint8)

    otsu_thresh_8bit, _ = cv2.threshold(norm8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert Otsu threshold back to log-score space, then to raw score space.
    otsu_log = float(otsu_thresh_8bit) / 255.0 * (log_max - log_min) + log_min
    otsu_raw = float(np.expm1(otsu_log))

    # Optional percentile lower bound.
    if bg_threshold_percentile is not None:
        percentile_raw = float(np.percentile(max_score, bg_threshold_percentile))
        effective_raw = max(otsu_raw, percentile_raw)
        print(f"  Otsu threshold: {otsu_raw:.4g}  |  "
              f"percentile-{bg_threshold_percentile} lower bound: {percentile_raw:.4g}  |  "
              f"using: {effective_raw:.4g}")
    else:
        effective_raw = otsu_raw
        print(f"  Otsu threshold: {otsu_raw:.4g}")

    bg_mask = max_score <= effective_raw

    pct = bg_mask.mean() * 100
    print(f"  Background mask: {pct:.1f}% of pixels classified as background")
    return bg_mask, max_score, effective_raw


def _halo_safe_source_for_background(
    focus_maps: np.ndarray,
    bg_mask: np.ndarray,
    source_map: np.ndarray,
    depth_radius: int,
    local_search_radius: int = 64,
) -> np.ndarray:
    """
    For background pixels, choose the frame that is *least likely to carry a halo*,
    staying as close as possible to the frame already chosen by adjacent subject edges.

    Two-stage approach
    ------------------
    Stage 1 — Edge propagation (primary):
        The subject pixels at the boundary of the background already have correct,
        halo-free frame assignments in ``source_map`` (they are real in-focus pixels
        at the edge of the subject, which is exactly the halo-free depth we want for
        adjacent background).  We propagate those assignments outward into background
        pixels using a normalised Gaussian blur — the same weighted-blur trick used
        for depth estimation.  Background pixels close to a subject edge inherit its
        exact frame assignment; pixels further away get a smoothly interpolated
        (rounded) version.

    Stage 2 — Edge-step fallback (secondary):
        For background pixels that are far from any subject edge (blurred weight < eps)
        there are no nearby assignments to inherit.  These fall back to the previous
        behaviour: estimate the local subject peak depth via Gaussian-blurred argmax,
        step ``depth_radius`` away in both directions, and pick the edge frame with the
        lower focus score.

    The result is that background pixels immediately adjacent to the subject use the
    same frame as their neighbouring edge pixels (maximising visual continuity), while
    isolated background regions far from the subject still get the most halo-free frame
    available.

    Parameters
    ----------
    focus_maps          : (N, H, W) float32 — smoothed Tenengrad scores.
    bg_mask             : (H, W) bool — True for background pixels.
    source_map          : (H, W) uint8 — frame assignments for subject pixels
                          (background pixel values will be overwritten by the caller).
    depth_radius        : int — used by the fallback path; same as the subject
                          selection depth_radius.
    local_search_radius : int — sigma of the Gaussian used to propagate frame
                          assignments and peak depths outward (pixels).  Should be
                          large enough to reach every background pixel from the nearest
                          subject pixel.  Default 64; increase for images with large
                          featureless background areas.

    Returns
    -------
    bg_source : (H, W) uint8 — frame index to use for each background pixel.
                Non-background pixels are set to 0 (they will be overwritten by
                the caller).
    """
    n = focus_maps.shape[0]
    subject_mask = ~bg_mask

    # Compute per-pixel argmax over subject pixels only, smooth outliers at the
    # boundary before propagating.
    subject_peak_map = np.argmax(focus_maps, axis=0).astype(np.float32)  # (H, W)
    subject_peak_clean = cv2.medianBlur(
        np.where(subject_mask, subject_peak_map, np.float32(0)), 5,
    )
    subject_peaks_f = np.where(subject_mask, subject_peak_clean, 0.0).astype(np.float32)
    subject_weights = subject_mask.astype(np.float32)

    # Propagate subject peak depths outward using a single Gaussian whose sigma
    # is large enough to reach every background pixel.  We use distanceTransform
    # to find the exact maximum distance from any background pixel to the nearest
    # subject pixel, then set sigma = max(local_search_radius, max_dist / 3) so
    # that the 3-sigma point of the Gaussian covers the furthest pixel.
    # This gives a single smooth, continuous propagation that honours the
    # local_search_radius near the subject and fills the entire background without
    # a hard switch to a global scalar fallback.
    if bg_mask.any():
        dist_to_subject = cv2.distanceTransform(
            bg_mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
        )
        max_dist = float(dist_to_subject.max())
    else:
        max_dist = 0.0

    sigma = max(float(local_search_radius), max_dist / 3.0)
    k = int(sigma * 6) | 1
    blurred_peaks   = cv2.GaussianBlur(subject_peaks_f, (k, k), sigma)
    blurred_weights = cv2.GaussianBlur(subject_weights,  (k, k), sigma)

    peak_estimate = blurred_peaks / np.where(blurred_weights > 1e-6, blurred_weights, 1.0)

    # Determine step direction once from the global median to avoid per-pixel
    # lo/hi flipping that would produce salt-and-pepper noise in the depth map.
    if subject_mask.any():
        global_peak = int(np.median(subject_peak_map[subject_mask]))
    else:
        global_peak = n // 2

    edge_lo = max(0,     global_peak - depth_radius)
    edge_hi = min(n - 1, global_peak + depth_radius)
    use_lo_globally = (global_peak - edge_lo) >= (edge_hi - global_peak)

    # Step depth_radius away from the local peak estimate in the chosen direction.
    if use_lo_globally:
        halo_safe = np.clip(peak_estimate - depth_radius, 0, n - 1)
    else:
        halo_safe = np.clip(peak_estimate + depth_radius, 0, n - 1)

    bg_source = np.clip(np.round(halo_safe), 0, n - 1).astype(np.uint8)

    print(f"  Background pixels filled: {int(bg_mask.sum())} "
          f"(propagation sigma={sigma:.1f}px, max dist to subject={max_dist:.1f}px)")

    return bg_source


def _propagate_from_sharp_neighbours(
    focus_maps: np.ndarray,
    quiet_mask: np.ndarray,
    source_map: np.ndarray,
    search_radius: int = 32,
) -> np.ndarray:
    """
    Fill quiet featureless pixels by propagating frame assignments from the
    nearest sharp neighbours.

    For pixels where no frame has genuine structural information (``quiet_mask``
    is True), the correct frame to use is the one already chosen by the
    surrounding sharp region — that region is at the right depth, and the quiet
    pixel has no signal of its own to argue otherwise.

    The propagation uses the same normalised Gaussian blur trick used by
    ``_halo_safe_source_for_background``: the source_map values of sharp pixels
    (where ``quiet_mask`` is False) are blurred with a Gaussian of sigma
    ``search_radius``, divided by a similarly-blurred mask of ones to normalise.
    Quiet pixels that fall within the influence of at least one sharp neighbour
    receive its (weighted, rounded) frame assignment.

    For quiet pixels that are entirely surrounded by other quiet pixels with no
    sharp neighbour within range, the original source_map value (argmax) is
    kept as a fallback, since there is genuinely no local depth reference.

    Parameters
    ----------
    focus_maps    : (N, H, W) float32 — used only to size the frame index range.
    quiet_mask    : (H, W) bool — True for pixels with no reliable focus signal.
    source_map    : (H, W) uint8 — frame assignments from argmax; quiet pixels
                    will have their values overwritten.
    search_radius : int — sigma of the Gaussian propagation kernel in pixels.
                    Should be large enough to bridge the widest quiet region.
                    Larger values propagate over greater distances but blur the
                    depth boundary more.  Default 32; increase for images with
                    large featureless areas.

    Returns
    -------
    filled : (H, W) uint8 — source_map with quiet pixels replaced by their
             propagated neighbour frame assignment wherever a sharp neighbour
             was reachable.
    """
    n = focus_maps.shape[0]
    sharp_mask = ~quiet_mask

    # Sigma large enough to cover the furthest quiet pixel from any sharp one.
    if quiet_mask.any() and sharp_mask.any():
        max_dist = float(cv2.distanceTransform(
            quiet_mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
        ).max())
    else:
        max_dist = 0.0
    sigma = max(float(search_radius), max_dist / 3.0)
    k = int(sigma * 6) | 1

    # Weighted Gaussian propagation: blur (sharp_source * sharp_weight) and
    # (sharp_weight) separately, then divide to get the normalised estimate.
    sharp_source_f = np.where(sharp_mask, source_map.astype(np.float32), 0.0)
    sharp_weight   = sharp_mask.astype(np.float32)

    blurred_source  = cv2.GaussianBlur(sharp_source_f, (k, k), sigma)
    blurred_weights = cv2.GaussianBlur(sharp_weight,   (k, k), sigma)

    # Where at least one sharp neighbour contributed weight, take its propagated
    # assignment.  Where weight is effectively zero, fall back to argmax.
    has_neighbour = blurred_weights > 1e-6
    propagated = blurred_source / np.where(has_neighbour, blurred_weights, 1.0)
    propagated_idx = np.clip(np.round(propagated), 0, n - 1).astype(np.uint8)

    n_filled = int((quiet_mask & has_neighbour).sum())
    n_fallback = int((quiet_mask & ~has_neighbour).sum())
    print(f"  Quiet-fill: {n_filled} pixels filled from neighbours "
          f"(sigma={sigma:.1f}px), {n_fallback} fallback to argmax")

    return np.where(quiet_mask & has_neighbour, propagated_idx, source_map).astype(np.uint8)


def select_best_pixels(
    images: list[np.ndarray],
    focus_maps: np.ndarray,
    depth_radius: int | None = None,
    smooth_radius: int | None = None,
    blend_low_confidence: float | None = None,
    bg_threshold_percentile: float | None = None,
    bg_halo_suppression: bool = False,
    bg_search_radius: int = 64,
    bg_blend_radius: int | None = None,
    fill_quiet: bool = False,
    fill_quiet_search_radius: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    For each pixel, pick the source image with the highest focus score.

    depth_radius
        When given, restricts candidate frames for each pixel to a window of
        [peak - depth_radius, peak + depth_radius] (inclusive) centred on the
        frame that already has the global maximum score for that pixel.

        Rationale: images are sorted by Z depth.  For a mostly-flat subject the
        in-focus plane sweeps monotonically through the stack, so a pixel whose
        sharpest frame is index 10 should never source content from frame 1 or 50.
        Limiting the window prevents diffraction halos or noise spikes on distant
        frames from stealing pixels away from the correct depth, and narrows the
        effective competition so the true sharp frame wins more decisively against
        residual artefacts in adjacent frames.

        A radius of 2-4 is appropriate for thin, flat subjects (e.g. mineral
        surfaces, flat biological specimens).  Increase for subjects with
        significant relief or a deeper depth of field.  None disables the
        restriction (considers all frames).

    smooth_radius
        When given, applies two passes of median filtering to the source map
        after selection to remove spatially isolated outlier frame assignments.

        In-focus bands span large contiguous regions; the spurious "islands"
        visible in the depth map (pixels assigned to a distant frame due to a
        noise spike or diffraction halo) are small isolated clusters.  A median
        filter over frame indices is a majority vote: it replaces each outlier
        pixel with the dominant frame index of its neighbourhood, restoring band
        coherence without blurring real band boundaries.

        Recommended starting value: 5 (neighbourhood of 11x11).  Increase if
        visible patches remain; decrease to preserve fine band detail.  Good
        values depend on the physical size of the spurious patches in pixels —
        the radius should be larger than half the largest patch you want removed.

    blend_low_confidence
        When given (a float in (0, 1)), pixels whose confidence score falls below
        this threshold are composited by blending all frames weighted by their
        focus scores rather than by hard winner-takes-all selection.

        Confidence is defined as (best_score - second_best_score) / best_score.
        It is near 1 where one frame is clearly sharper (textured subject matter)
        and near 0 where all frames score similarly (featureless background,
        uniform regions).  Hard selection in the latter case picks a winner based
        on noise, creating a splotchy source map and visible artefacts.  Blending
        averages those noisy contributions away.

        Recommended starting value: 0.3.  Raise toward 0.6-0.8 if background
        banding is still visible; lower toward 0.1 if soft blending is leaking
        into clearly focused subject regions.

    bg_threshold_percentile
        When given (a float in (0, 100)), enables background halo suppression.
        Pixels whose peak focus score (across all frames) falls at or below this
        percentile of the stack-wide peak-score distribution are classified as
        background.  For these pixels, instead of selecting the frame with the
        highest (potentially halo-contaminated) score, the algorithm selects the
        frame at the *edge* of the in-focus band — i.e. exactly ``depth_radius``
        steps away from the dominant subject focus depth.  Those edge frames are
        far enough from the subject's sharpest plane that diffraction halos have
        dissipated, so they provide a clean, un-haloed background.

        Requires ``depth_radius`` to be set; if ``depth_radius`` is None the
        option is ignored with a warning.

        Start with 20 (bottom fifth of peak scores).  Raise toward 40-50 if
        halo-contaminated background regions are still being selected as subject;
        lower toward 5-10 if genuine subject detail near the edges is being
        suppressed.

    bg_blend_radius
        When given (in pixels), feathers the subject/background boundary by
        blending subject-sourced pixels and background-sourced pixels with a
        Gaussian alpha of this sigma.  Without this, the hard switch at the
        bg_mask boundary can produce a visible seam where the frame index jumps
        abruptly.  The alpha is 1 inside the subject (fully uses source_map)
        and decays to 0 deep in the background (fully uses bg_source), so the
        transition runs outward from the subject edge.  A value of 10-30 pixels
        is usually enough to hide the seam; increase if a step is still visible,
        but be aware that very large values will let the subject frame bleed far
        into the background.  None disables feathering (hard edge, default).

    fill_quiet
        When True, pixels whose peak focus score across the stack is below the
        Otsu threshold (i.e. no frame has genuine structural information there)
        have their argmax frame assignment replaced by the frame assignment of
        their nearest sharp neighbours, propagated via normalised Gaussian blur.

        This fixes the failure mode where a featureless region that is smooth
        when in focus gets assigned to a noisy out-of-focus frame by argmax:
        the quiet pixel has no reliable signal of its own, so it should simply
        use whatever depth the surrounding in-focus detail dictates.

        Unlike ``--quiet-wins-auto``, this does not try to guess the correct
        frame from the quiet pixel's own scores.  It copies the answer from the
        sharp neighbourhood, which already has it right.  The result is that
        quiet regions take on the depth of the closest sharp content —
        exactly what you observe visually when the surrounding detail is in focus.

        Use ``--fill-quiet-search-radius`` to control how far the propagation
        reaches (default 32 px).

    fill_quiet_search_radius
        Gaussian sigma in pixels for the neighbour propagation in ``fill_quiet``
        (default: 32).  Increase if quiet regions are large and not fully filled;
        decrease to keep the depth assignment more local.

    Returns:
        result     - the composited image (same dtype/channels as inputs)
        source_map - uint8 index array (0..N-1) indicating which image was chosen
                     for each pixel.
    """
    n = len(images)

    # --- Background halo suppression --------------------------------------------
    # Compute a background mask before the normal selection so we can route
    # background pixels to halo-safe frames instead of the (potentially
    # halo-contaminated) argmax frame.
    bg_mask: np.ndarray | None = None
    _bg_max_score: np.ndarray | None = None
    _bg_threshold: float | None = None
    if bg_halo_suppression:
        if depth_radius is None:
            print("  [warn] --bg-percentile requires --depth-radius; ignoring.")
        else:
            print(f"Background halo suppression enabled "
                  f"(depth_radius={depth_radius}, percentile lower bound={bg_threshold_percentile})...")
            bg_mask, _bg_max_score, _bg_threshold = _build_background_mask(focus_maps, bg_threshold_percentile)
    if depth_radius is not None and depth_radius < n:
        print(f"Compositing: depth-restricted selection (radius={depth_radius})...")
        # Step 1: find the unconstrained peak frame per pixel
        peak = np.argmax(focus_maps, axis=0)  # (H, W) int64

        # Step 2: mask out-of-window frames to -inf so they cannot win the argmax.
        # frame_idx broadcasts against the (H, W) peak map.
        frame_idx = np.arange(n, dtype=np.int64)[:, None, None]  # (N, 1, 1)
        lo = (peak[None] - depth_radius).clip(0, n - 1)           # (1, H, W)
        hi = (peak[None] + depth_radius).clip(0, n - 1)           # (1, H, W)
        in_window = (frame_idx >= lo) & (frame_idx <= hi)          # (N, H, W)

        masked = np.where(in_window, focus_maps, -np.inf)
        source_map = np.argmax(masked, axis=0).astype(np.uint8)
    else:
        if depth_radius is not None:
            print("Compositing: depth_radius >= stack size, ignoring restriction...")
        else:
            print("Compositing: selecting best-focus pixel per location...")
        source_map = np.argmax(focus_maps, axis=0).astype(np.uint8)

    # --- Fill quiet pixels from sharp neighbours --------------------------------
    # Done immediately after argmax, before smoothing and background handling,
    # so that smoothing and bg propagation operate on already-filled values.
    if fill_quiet:
        print("Filling quiet pixels from sharp neighbours...")
        # Reuse _build_background_mask's Otsu logic to identify quiet pixels:
        # pixels whose peak score across all frames falls in the low-score cluster
        # of the bimodal distribution.  No percentile adjustment needed here —
        # we just want the natural split between "has real detail" and "doesn't".
        quiet_mask, _, qw_thresh = _build_background_mask(focus_maps, bg_threshold_percentile=None)
        pct = quiet_mask.mean() * 100
        print(f"  Quiet mask: {pct:.1f}% of pixels (peak score < {qw_thresh:.4g})")
        source_map = _propagate_from_sharp_neighbours(
            focus_maps, quiet_mask, source_map, search_radius=fill_quiet_search_radius,
        )

    # Compute halo-safe background frame assignments.
    bg_source: np.ndarray | None = None
    if bg_mask is not None and bg_mask.any():
        bg_source = _halo_safe_source_for_background(focus_maps, bg_mask, source_map, depth_radius, bg_search_radius)  # type: ignore[arg-type]
        # Patch source_map for depth-map visualisation and smooth_source_map, but
        # keep the original subject-only map for pixel gathering so the transition
        # blend has two genuinely different inputs to interpolate between.
        source_map_for_viz = np.where(bg_mask, bg_source, source_map).astype(np.uint8)
    else:
        source_map_for_viz = source_map

    if smooth_radius is not None:
        print(f"Smoothing source map (median radius={smooth_radius})...")
        source_map_for_viz = smooth_source_map(source_map_for_viz, smooth_radius)

    h, w = source_map.shape
    stack = np.stack(images, axis=0)  # (N, H, W) or (N, H, W, C)
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]

    # Use the viz map as the canonical source_map for depth-map output and stats.
    source_map = source_map_for_viz

    # Gather subject pixels from the (now smoothed) source_map so that the
    # composited pixels are consistent with the depth map visualisation.
    subject_pixels = stack[source_map, ys, xs]

    # --- Confidence-weighted blend for low-detail regions -----------------------
    if blend_low_confidence is not None:
        print(f"Blending low-confidence pixels (threshold={blend_low_confidence})...")

        confidence = _confidence_map(focus_maps)  # (H, W)
        conf_smooth = cv2.GaussianBlur(confidence, (0, 0), sigmaX=2.0)

        low_conf_mask = conf_smooth < blend_low_confidence  # (H, W) bool
        pct_blended = low_conf_mask.mean() * 100
        print(f"  {pct_blended:.1f}% of pixels will be blended (confidence < {blend_low_confidence})")

        if low_conf_mask.any():
            blended_result = _blend_weighted(images, focus_maps)
            conf_alpha = (conf_smooth / blend_low_confidence).clip(0.0, 1.0)

            if subject_pixels.ndim == 3:
                conf_alpha_3d = conf_alpha[:, :, np.newaxis]
            else:
                conf_alpha_3d = conf_alpha

            conf_blended = (
                conf_alpha_3d * subject_pixels.astype(np.float64)
                + (1.0 - conf_alpha_3d) * blended_result.astype(np.float64)
            )
            src_dtype = images[0].dtype
            if np.issubdtype(src_dtype, np.integer):
                info = np.iinfo(src_dtype)
                conf_blended = conf_blended.round().clip(info.min, info.max)
            subject_pixels = conf_blended.astype(src_dtype)

    # --- Composite: subject pixels + bg_source pixels --------------------------
    # Background pixels use bg_source (halo-safe frame from
    # _halo_safe_source_for_background); subject pixels use source_map.
    # With bg_blend_radius=None this is a hard switch at the mask boundary.
    # With bg_blend_radius set, the subject mask is Gaussian-blurred to produce
    # a soft alpha that feathers the boundary: alpha=1 inside the subject
    # (uses source_map), fading to 0 deep in the background (uses bg_source).
    if bg_source is not None and bg_mask is not None:
        if bg_blend_radius is not None:
            sig = float(bg_blend_radius)
            k = int(sig * 6) | 1
            subject_mask_f = (~bg_mask).astype(np.float32)
            alpha = cv2.GaussianBlur(subject_mask_f, (k, k), sig)
            alpha = np.clip(alpha, 0.0, 1.0)
            blended_idx = (
                alpha         * source_map_for_viz.astype(np.float32)
                + (1.0 - alpha) * bg_source.astype(np.float32)
            )
            composite_idx = np.clip(np.round(blended_idx), 0, len(images) - 1).astype(np.uint8)
        else:
            composite_idx = np.where(bg_mask, bg_source, source_map_for_viz).astype(np.uint8)
        hard_result = stack[composite_idx, ys, xs]
        source_map = composite_idx
    else:
        hard_result = subject_pixels
        source_map = source_map_for_viz

    return hard_result, source_map, bg_mask


# ---------------------------------------------------------------------------
# Depth / source map visualisation
# ---------------------------------------------------------------------------

def save_depth_map(source_map: np.ndarray, n_images: int, path: Path) -> None:
    """Save a greyscale PNG showing which frame was chosen per pixel.

    Frame 0 (nearest / first in stack) maps to black (0); the last frame maps
    to white (255).  Linear interpolation between those extremes means brighter
    pixels were sourced from frames deeper in the stack.
    """
    grey = (source_map.astype(np.float32) / max(n_images - 1, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(path), grey)
    print(f"Depth map saved -> {path}")


def save_bg_mask(bg_mask: np.ndarray, path: Path) -> None:
    """Save the subject/background classification mask as a greyscale PNG.

    White (255) = background pixels (classified as not in-focus subject).
    Black (0)   = subject pixels (classified as genuinely in-focus at some depth).

    Use this to diagnose --bg-percentile: if the mask shows hot-spot regions
    in white they are correctly classified as background and will be handled by
    the halo-suppression path.  If they appear black, they are being treated as
    subject and the hot spots will persist — raise --bg-percentile until they
    turn white.
    """
    out = np.where(bg_mask, np.uint8(255), np.uint8(0))
    cv2.imwrite(str(path), out)
    print(f"Background mask saved -> {path}")


def save_score_maps(
    focus_maps: np.ndarray,
    out_dir: Path,
    image_paths: list[Path],
) -> None:
    """Save per-frame Tenengrad score maps as normalised 16-bit PNGs.

    A log transform is applied before normalisation to compress the extreme
    dynamic range produced by power exponentiation (squaring already-small
    Sobel magnitudes yields values spread across many orders of magnitude).
    Without it, global min/max normalisation crushes nearly all pixels to
    black.  Log scaling preserves cross-frame comparability — a brighter
    pixel genuinely indicates a higher focus score relative to the rest of
    the stack — while making the spatial structure visible.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_maps = np.log1p(focus_maps)  # log(1 + x); safe for zero values

    # Exclude warp border pixels from the global normalisation range.  Borders
    # are filled with zero by warpAffine/warpPerspective; the hard edge they
    # create produces extreme Sobel responses that dominate the global max and
    # crush all real content toward black.  We find valid pixels by taking the
    # union of non-border regions across all frames (frame 0 is never warped so
    # it has no border, but including it in the union is harmless).
    # A pixel is considered a border pixel if its score in any frame is at the
    # very bottom of the log-score distribution (i.e., the raw focus map was
    # effectively zero there due to the constant fill).
    valid_mask = np.ones(focus_maps.shape[1:], dtype=bool)
    for i in range(focus_maps.shape[0]):
        # Zero-filled border regions have a focus score of exactly 0 after
        # Gaussian smoothing only if the entire kernel landed on zeros, which
        # is true well inside the border strip.  A small epsilon is used to
        # catch near-zero values at the edge of the smoothing kernel.
        valid_mask &= focus_maps[i] > 1e-8

    valid_values = log_maps[:, valid_mask]
    if valid_values.size > 0:
        global_min = valid_values.min()
        global_max = valid_values.max()
    else:
        global_min = log_maps.min()
        global_max = log_maps.max()
    for i, p in enumerate(image_paths):
        score = log_maps[i]
        if global_max > global_min:
            norm = ((score - global_min) / (global_max - global_min) * 65535).astype(np.uint16)
        else:
            norm = np.zeros_like(score, dtype=np.uint16)
        out_path = out_dir / f"score_{p.stem}.png"
        cv2.imwrite(str(out_path), norm)
    print(f"Focus score maps saved -> {out_dir}/")