from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d, uniform_filter


KERNEL_1D = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_images(folder: Path) -> tuple[list[np.ndarray], tuple[int, int]]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(paths) < 2:
        print(f"Error: need at least 2 images in '{folder}', found {len(paths)}.")
        sys.exit(1)

    images = []
    reference_size: tuple[int, int] | None = None
    for path in paths:
        img = Image.open(path).convert("RGB")
        if reference_size is None:
            reference_size = img.size
        elif img.size != reference_size:
            img = img.resize(reference_size, Image.LANCZOS)
        images.append(np.array(img, dtype=np.float64))
        print(f"  Loaded: {path.name}")

    return images, reference_size  # type: ignore[return-value]


def _to_gray_cv(image: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB (H,W,3) to uint8 grayscale using OpenCV."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _ecc_align(ref_gray: np.ndarray, src_gray: np.ndarray,
               max_resolution: int, rough: bool) -> np.ndarray:
    """Single ECC alignment pass, mirroring task_align.cc match_transform().

    Downscales both images so the longer edge is at most max_resolution,
    runs findTransformECC with MOTION_AFFINE, then rescales the translation
    component back to full resolution and returns the 2x3 affine matrix.
    """
    h, w = ref_gray.shape
    resolution = max(h, w)
    if resolution > max_resolution:
        scale = max_resolution / resolution
        ref_small = cv2.resize(ref_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        src_small = cv2.resize(src_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        ref_small = ref_gray
        src_small = src_gray

    warp = np.eye(2, 3, dtype=np.float32)
    if rough:
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 25, 0.01)
        gauss_levels = 1
    else:
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 0.001)
        gauss_levels = 3

    cv2.findTransformECC(src_small.astype(np.float32),
                         ref_small.astype(np.float32),
                         warp, cv2.MOTION_AFFINE, criteria,
                         None, gauss_levels)

    warp[0, 2] /= scale
    warp[1, 2] /= scale
    return warp


def _chain_affines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two 2x3 affine matrices: apply b then a."""
    a3 = np.vstack([a, [0, 0, 1]])
    b3 = np.vstack([b, [0, 0, 1]])
    return (a3 @ b3)[:2]


def align_images(images: list[np.ndarray], min_shift: float = 5.0) -> list[np.ndarray]:
    """Align all images using OpenCV ECC, mirroring task_align.cc.

    Each image is aligned against its neighbour (not the global reference)
    and the per-step affine transforms are chained back to the reference frame.
    This matches the default behaviour of the focus-stack C++ program, which
    handles deep stacks where extreme images are too blurry to align directly
    against the centre.

    Two ECC passes are used per pair, mirroring match_transform():
      - Rough pass at 256px (25 iterations, eps=0.01, Gaussian pyramid level 1)
      - Fine pass at 2048px (50 iterations, eps=0.001, Gaussian pyramid level 3)

    Transforms with a translation below min_shift pixels AND negligible
    rotation/scale are skipped to avoid applying unnecessary resampling.
    """
    grays = [_to_gray_cv(np.clip(img, 0, 255).astype(np.uint8)) for img in images]
    h, w = images[0].shape[:2]
    identity = np.eye(2, 3, dtype=np.float32)

    aligned = [images[0]]
    cumulative_warp = identity.copy()

    for i in range(1, len(images)):
        ref_gray = grays[i - 1]
        src_gray = grays[i]

        warp = identity.copy()
        converged = True
        for max_res, rough in [(256, True), (2048, False)]:
            try:
                warp = _ecc_align(ref_gray, src_gray, max_res, rough)
            except cv2.error:
                converged = False
                break

        if not converged:
            print(f"  Image {i + 1}: ECC did not converge — using previous transform")
            aligned.append(images[i])
            continue

        cumulative_warp = _chain_affines(cumulative_warp, warp)

        translation = np.linalg.norm(cumulative_warp[:, 2])
        linear_part = cumulative_warp[:, :2]
        is_identity_linear = np.allclose(linear_part, np.eye(2), atol=1e-3)
        if translation < min_shift and is_identity_linear:
            print(f"  Image {i + 1}: transform negligible — skipped")
            aligned.append(images[i])
            continue

        angle = np.degrees(np.arctan2(cumulative_warp[1, 0], cumulative_warp[0, 0]))
        tx, ty = cumulative_warp[0, 2], cumulative_warp[1, 2]
        print(f"  Image {i + 1}: rotation {angle:+.2f}°  shift ({ty:+.1f}, {tx:+.1f}) px")

        src_u8 = np.clip(images[i], 0, 255).astype(np.uint8)
        warped = cv2.warpAffine(src_u8, cumulative_warp, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REFLECT)
        aligned.append(warped.astype(np.float64))

    return aligned


def _smooth(image: np.ndarray) -> np.ndarray:
    """Apply separable 5-tap Gaussian smoothing."""
    return convolve1d(convolve1d(image, KERNEL_1D, axis=0), KERNEL_1D, axis=1)


def reduce(image: np.ndarray) -> np.ndarray:
    return _smooth(image)[::2, ::2]


def expand(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = image.shape
    upsampled = np.zeros((h * 2, w * 2), dtype=np.float64)
    upsampled[::2, ::2] = image
    expanded = convolve1d(convolve1d(upsampled, KERNEL_1D * 2, axis=0), KERNEL_1D * 2, axis=1)
    return expanded[: target_shape[0], : target_shape[1]]


def gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    pyramid = [image.copy()]
    for _ in range(levels):
        pyramid.append(reduce(pyramid[-1]))
    return pyramid


def laplacian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    gauss = gaussian_pyramid(image, levels)
    lp = []
    for i in range(levels):
        lp.append(gauss[i] - expand(gauss[i + 1], gauss[i].shape))
    lp.append(gauss[-1])
    return lp


def reconstruct(lp: list[np.ndarray]) -> np.ndarray:
    image = lp[-1].copy()
    for level in reversed(lp[:-1]):
        image = expand(image, level.shape) + level
    return image


def region_energy(lp_level: np.ndarray, window: int = 3) -> np.ndarray:
    return uniform_filter(lp_level ** 2, size=window)


def region_deviation(image: np.ndarray, window: int = 3) -> np.ndarray:
    mean = uniform_filter(image, size=window)
    mean_sq = uniform_filter(image ** 2, size=window)
    return np.sqrt(np.maximum(mean_sq - mean ** 2, 0))


def region_entropy(image: np.ndarray, window: int = 8) -> np.ndarray:
    normed = (image - image.min()) / (image.max() - image.min() + 1e-10)
    eps = 1e-10
    ent = -normed * np.log2(normed + eps) - (1 - normed) * np.log2(1 - normed + eps)
    return uniform_filter(ent, size=window)


def compute_energy_weights(lp_levels: list[np.ndarray], sharpness: float) -> np.ndarray:
    """Compute normalised per-pixel weights from Laplacian energy.

    The sharpness exponent controls how peaked the distribution is.
    At 1.0 weights are proportional to energy (soft blend). Higher values
    increasingly favour the dominant image, approaching argmax at the limit.
    Returns an array of shape (N, H, W) that sums to 1 along axis 0.
    """
    energies = np.stack([region_energy(lv) for lv in lp_levels], axis=0)
    energies = energies ** sharpness
    total = energies.sum(axis=0, keepdims=True) + 1e-10
    return energies / total


def compute_top_level_weights(levels: list[np.ndarray], sharpness: float) -> np.ndarray:
    """Compute normalised weights for the coarsest pyramid level.

    Uses local standard deviation and entropy, both derived from luminance,
    to produce a (N, H, W) weight array that sums to 1 along axis 0.
    Average of the two metrics is taken before normalising so neither
    dominates. The sharpness exponent is applied to the combined score.
    """
    devs = np.stack([region_deviation(lv) for lv in levels], axis=0)
    ents = np.stack([region_entropy(lv) for lv in levels], axis=0)
    combined = ((devs + ents) / 2.0) ** sharpness
    total = combined.sum(axis=0, keepdims=True) + 1e-10
    return combined / total


def fuse_with_weights(levels: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Blend pyramid levels using precomputed per-pixel weights.

    levels  : list of N arrays each shaped (H, W)
    weights : array shaped (N, H, W), summing to 1 along axis 0
    """
    stacked = np.stack(levels, axis=0)
    return (stacked * weights).sum(axis=0)


def compute_level_weights(
    lum_pyramids: list[list[np.ndarray]],
    levels: int,
    sharpness: float,
) -> list[np.ndarray]:
    """Compute fusion weight maps for every pyramid level from luminance pyramids.

    Returns a list of (N, H, W) weight arrays, one per pyramid level including
    the coarsest residual.
    """
    all_weights = []
    for i in range(levels + 1):
        layer = [p[i] for p in lum_pyramids]
        if i == levels:
            all_weights.append(compute_top_level_weights(layer, sharpness))
        else:
            all_weights.append(compute_energy_weights(layer, sharpness))
    return all_weights


def fuse_channel(
    images: list[np.ndarray],
    levels: int,
    weights: list[np.ndarray],
    prebuilt_pyramids: list[list[np.ndarray]] | None = None,
) -> np.ndarray:
    """Fuse a single channel using precomputed luminance-derived weight maps."""
    pyramids = prebuilt_pyramids if prebuilt_pyramids is not None else [laplacian_pyramid(img, levels) for img in images]
    fused_lp = [
        fuse_with_weights([p[i] for p in pyramids], weights[i])
        for i in range(levels + 1)
    ]
    return reconstruct(fused_lp)


def _rgb_to_lab(images: list[np.ndarray]) -> list[np.ndarray]:
    result = []
    for img in images:
        u8 = np.clip(img, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2Lab).astype(np.float64)
        result.append(lab)
    return result


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    clipped = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(clipped, cv2.COLOR_Lab2RGB)


def _suppress_dark_chroma(fused_lab: np.ndarray, threshold: float) -> np.ndarray:
    """Lerp a/b channels toward neutral (128) in dark regions.

    In Lab (OpenCV uint8 encoding) a neutral/achromatic pixel has a=128, b=128.
    Floating point drift during pyramid reconstruction can push dark pixels away
    from neutral, producing visible color casts in areas that should be black.
    The mask is 0 where L=0 and ramps linearly to 1 at L=threshold, clamped
    to 1 above that, so only genuinely dark pixels are affected.
    """
    l = fused_lab[:, :, 0]
    mask = np.clip(l / (threshold + 1e-10), 0.0, 1.0)
    result = fused_lab.copy()
    result[:, :, 1] = 128.0 + (fused_lab[:, :, 1] - 128.0) * mask
    result[:, :, 2] = 128.0 + (fused_lab[:, :, 2] - 128.0) * mask
    return result


def stack_images(images: list[np.ndarray], levels: int, sharpness: float, dark_threshold: float) -> np.ndarray:
    print("  Converting to Lab...")
    lab_images = _rgb_to_lab(images)

    lum_channels = [img[:, :, 0] for img in lab_images]
    lum_pyramids = [laplacian_pyramid(lum, levels) for lum in lum_channels]

    print(f"  Computing luminance-based fusion weights (sharpness={sharpness})...")
    weights = compute_level_weights(lum_pyramids, levels, sharpness)

    fused_lab = np.zeros_like(lab_images[0])
    channel_names = ("L", "a", "b")
    for c in range(3):
        print(f"  Fusing {channel_names[c]} channel across {len(lab_images)} images...")
        channel_slices = [img[:, :, c] for img in lab_images]
        prebuilt = lum_pyramids if c == 0 else None
        fused_lab[:, :, c] = fuse_channel(channel_slices, levels, weights, prebuilt)

    print("  Suppressing chroma in dark regions...")
    fused_lab = _suppress_dark_chroma(fused_lab, dark_threshold)

    print("  Converting back to RGB...")
    return _lab_to_rgb(fused_lab)


def compute_levels(shape: tuple[int, int], max_levels: int = 6) -> int:
    min_dim = min(shape[0], shape[1])
    levels = 0
    size = min_dim
    while size > 16 and levels < max_levels:
        size //= 2
        levels += 1
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focus stack a folder of images using Laplacian pyramid fusion."
    )
    parser.add_argument("folder", type=Path, help="Folder containing input images.")
    parser.add_argument(
        "--no-align", action="store_true",
        help="Skip DFT alignment (use when images are already registered).",
    )
    parser.add_argument(
        "--min-shift", type=float, default=5.0,
        help="Minimum shift magnitude in pixels before alignment is applied (default: 5.0).",
    )
    parser.add_argument(
        "--levels", type=int, default=0,
        help="Pyramid levels (0 = auto-detect from image size).",
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG output quality 1-95 (default: 95).",
    )
    parser.add_argument(
        "--sharpness", type=float, default=4.0,
        help=(
            "Weight sharpness exponent (default: 4.0). "
            "Higher values favour the sharpest image more aggressively at each pixel, "
            "approaching a hard winner-take-all selection. "
            "Lower values blend more smoothly across images. "
            "Useful range is roughly 1.0 (soft) to 16.0 (near-hard)."
        ),
    )
    parser.add_argument(
        "--dark-threshold", type=float, default=30.0,
        help=(
            "Luminance threshold (0-255) below which chroma is suppressed toward neutral (default: 30.0). "
            "Pixels with L below this value have their a/b channels lerped toward 128 (achromatic), "
            "preventing color drift in dark/black regions caused by floating point reconstruction error. "
            "Raise if color casts remain in shadows; lower if legitimate dark colors are being desaturated."
        ),
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"Error: '{args.folder}' is not a directory.")
        sys.exit(1)

    print(f"Loading images from '{args.folder}'...")
    images, size = load_images(args.folder)
    h, w = images[0].shape[:2]

    levels = args.levels if args.levels > 0 else compute_levels((h, w))
    print(f"Image size: {w}x{h}  |  Images: {len(images)}  |  Pyramid levels: {levels}  |  Sharpness: {args.sharpness}  |  Dark threshold: {args.dark_threshold}")

    if not args.no_align:
        print("Aligning (ECC affine, neighbour-chained)...")
        images = align_images(images, min_shift=args.min_shift)
    else:
        print("Skipping alignment.")

    print("Stacking...")
    result = stack_images(images, levels, args.sharpness, args.dark_threshold)

    out_path = args.folder / "stacked.jpg"
    Image.fromarray(result).save(out_path, "JPEG", quality=args.quality)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()