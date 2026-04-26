from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d, uniform_filter


KERNEL_1D = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.2f}s"


class _Checkpoint:
    __slots__ = ("name", "elapsed", "current_mib", "peak_mib")

    def __init__(self, name: str, elapsed: float, current_mib: float, peak_mib: float) -> None:
        self.name = name
        self.elapsed = elapsed
        self.current_mib = current_mib
        self.peak_mib = peak_mib


def _snap(name: str, step_start: float, checkpoints: list[_Checkpoint]) -> float:
    """Record elapsed time and current tracemalloc stats, return now for the next timer."""
    now = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    checkpoints.append(_Checkpoint(name, now - step_start, current / 2**20, peak / 2**20))
    return now


def _print_report(checkpoints: list[_Checkpoint], total_elapsed: float) -> None:
    name_w = max(len(c.name) for c in checkpoints)
    header = f"  {'Step':<{name_w}}  {'Time':>8}  {'Current MiB':>12}  {'Peak MiB':>10}"
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))
    for c in checkpoints:
        print(f"  {c.name:<{name_w}}  {c.elapsed:>7.2f}s  {c.current_mib:>11.1f}  {c.peak_mib:>9.1f}")
    print("  " + "-" * (len(header) - 2))
    print(f"  {'Total':<{name_w}}  {total_elapsed:>7.2f}s")

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
        images.append(np.array(img, dtype=np.float32))
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
    cumulative_warp = identity.copy()

    for i in range(1, len(images)):
        warp = identity.copy()
        converged = True
        for max_res, rough in [(256, True), (2048, False)]:
            try:
                warp = _ecc_align(grays[i - 1], grays[i], max_res, rough)
            except cv2.error:
                converged = False
                break

        if not converged:
            print(f"  Image {i + 1}: ECC did not converge — using previous transform")
            continue

        cumulative_warp = _chain_affines(cumulative_warp, warp)

        translation = np.linalg.norm(cumulative_warp[:, 2])
        is_identity_linear = np.allclose(cumulative_warp[:, :2], np.eye(2), atol=1e-3)
        if translation < min_shift and is_identity_linear:
            print(f"  Image {i + 1}: transform negligible — skipped")
            continue

        angle = np.degrees(np.arctan2(cumulative_warp[1, 0], cumulative_warp[0, 0]))
        tx, ty = cumulative_warp[0, 2], cumulative_warp[1, 2]
        print(f"  Image {i + 1}: rotation {angle:+.2f}°  shift ({ty:+.1f}, {tx:+.1f}) px")

        src_u8 = np.clip(images[i], 0, 255).astype(np.uint8)
        images[i] = cv2.warpAffine(src_u8, cumulative_warp, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REFLECT).astype(np.float32)

    return images


def _smooth(image: np.ndarray) -> np.ndarray:
    """Apply separable 5-tap Gaussian smoothing."""
    return convolve1d(convolve1d(image, KERNEL_1D, axis=0), KERNEL_1D, axis=1)


def reduce(image: np.ndarray) -> np.ndarray:
    return _smooth(image)[::2, ::2]


def expand(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = image.shape
    upsampled = np.zeros((h * 2, w * 2), dtype=np.float32)
    upsampled[::2, ::2] = image
    expanded = convolve1d(convolve1d(upsampled, KERNEL_1D * 2, axis=0), KERNEL_1D * 2, axis=1)
    return expanded[: target_shape[0], : target_shape[1]]



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




def _fuse_level_low_memory(
    gauss: np.ndarray,
    next_gauss: np.ndarray,
    weight_buf: np.ndarray,
    weight_copy: np.ndarray,
    sharpness: float,
) -> np.ndarray:
    """Low-memory fusion path for one pyramid level.

    Computes expand(next_gauss[k, L]) twice per image — once for weights,
    once during accumulation — to avoid caching the expanded luminance bands.
    Peak extra allocation per call is one (H, W) tmp array.
    """
    n = gauss.shape[0]
    cur_shape = gauss.shape[1:3]

    for k in range(n):
        lum_band = gauss[k, :, :, 0] - expand(next_gauss[k, :, :, 0], cur_shape)
        weight_buf[k] = region_energy(lum_band) ** sharpness
    weight_buf /= weight_buf.sum(axis=0, keepdims=True) + 1e-10
    np.copyto(weight_copy, weight_buf)

    fused_band = np.zeros(cur_shape + (3,), dtype=np.float32)
    tmp = np.empty(cur_shape, dtype=np.float32)
    for k in range(n):
        for c in range(3):
            np.subtract(gauss[k, :, :, c], expand(next_gauss[k, :, :, c], cur_shape), out=tmp)
            fused_band[:, :, c] += tmp * weight_copy[k]
    return fused_band


def _fuse_level_high_performance(
    gauss: np.ndarray,
    next_gauss: np.ndarray,
    weight_buf: np.ndarray,
    weight_copy: np.ndarray,
    sharpness: float,
) -> np.ndarray:
    """High-performance fusion path for one pyramid level.

    Caches all N expanded luminance bands so expand() is called once per image
    instead of twice. Trades one extra (N, H, W) allocation for fewer convolutions.
    """
    n = gauss.shape[0]
    cur_shape = gauss.shape[1:3]

    expanded_lum = np.empty((n,) + cur_shape, dtype=np.float32)
    for k in range(n):
        expanded_lum[k] = expand(next_gauss[k, :, :, 0], cur_shape)
        weight_buf[k] = region_energy(gauss[k, :, :, 0] - expanded_lum[k]) ** sharpness
    weight_buf /= weight_buf.sum(axis=0, keepdims=True) + 1e-10
    np.copyto(weight_copy, weight_buf)

    fused_band = np.zeros(cur_shape + (3,), dtype=np.float32)
    tmp = np.empty(cur_shape, dtype=np.float32)
    for k in range(n):
        np.subtract(gauss[k, :, :, 0], expanded_lum[k], out=tmp)
        fused_band[:, :, 0] += tmp * weight_copy[k]
        for c in range(1, 3):
            np.subtract(gauss[k, :, :, c], expand(next_gauss[k, :, :, c], cur_shape), out=tmp)
            fused_band[:, :, c] += tmp * weight_copy[k]
    return fused_band


def stack_images(
    images: list[np.ndarray],
    levels: int,
    sharpness: float,
    dark_threshold: float,
    high_performance: bool = False,
) -> np.ndarray:
    """Fuse a stack of aligned RGB images using Laplacian pyramid blending in Lab space.

    When high_performance=False (default), expand() is called twice per image per
    level for the luminance channel to avoid caching expanded bands, minimising
    peak memory at the cost of redundant convolutions.

    When high_performance=True, expanded luminance bands are cached in an extra
    (N, H, W) buffer so expand() is only called once per image per level.
    """
    n = len(images)
    fuse_level = _fuse_level_high_performance if high_performance else _fuse_level_low_memory

    t = time.perf_counter()
    print("  Converting to Lab...")
    gauss = np.empty((n,) + images[0].shape[:2] + (3,), dtype=np.float32)
    for k in range(n):
        img = images[k]
        images[k] = None  # type: ignore[assignment]  # release RGB array immediately
        u8 = np.clip(img, 0, 255).astype(np.uint8)
        gauss[k] = cv2.cvtColor(u8, cv2.COLOR_RGB2Lab).astype(np.float32)
    images.clear()
    print(f"    done ({_elapsed(t)})")

    t = time.perf_counter()
    mode = "high-performance" if high_performance else "low-memory"
    print(f"  Fusing channels using streaming pyramid (levels={levels}, mode={mode})...")

    fused_lp: list[np.ndarray] = []
    weight_buf: np.ndarray | None = None
    weight_copy: np.ndarray | None = None

    for _ in range(levels):
        cur_shape = gauss.shape[1:3]

        if weight_buf is None or weight_buf.shape[1:] != cur_shape:
            weight_buf = np.empty((n,) + cur_shape, dtype=np.float32)
            weight_copy = np.empty_like(weight_buf)

        next_h = (cur_shape[0] + 1) // 2
        next_w = (cur_shape[1] + 1) // 2
        next_gauss = np.empty((n, next_h, next_w, 3), dtype=np.float32)
        for k in range(n):
            for c in range(3):
                next_gauss[k, :, :, c] = reduce(gauss[k, :, :, c])

        fused_lp.append(fuse_level(gauss, next_gauss, weight_buf, weight_copy, sharpness))
        gauss = next_gauss

    # Coarsest residual level — deviation + entropy weights on luminance.
    coarse_shape = gauss.shape[1:3]
    if weight_buf is None or weight_buf.shape[1:] != coarse_shape:
        weight_buf = np.empty((n,) + coarse_shape, dtype=np.float32)
        weight_copy = np.empty_like(weight_buf)

    for k in range(n):
        lv = gauss[k, :, :, 0]
        weight_buf[k] = ((region_deviation(lv) + region_entropy(lv)) * 0.5) ** sharpness
    weight_buf /= weight_buf.sum(axis=0, keepdims=True) + 1e-10
    np.copyto(weight_copy, weight_buf)

    fused_band = np.zeros(coarse_shape + (3,), dtype=np.float32)
    for k in range(n):
        for c in range(3):
            fused_band[:, :, c] += gauss[k, :, :, c] * weight_copy[k]
    fused_lp.append(fused_band)

    print(f"    done ({_elapsed(t)})")

    t = time.perf_counter()
    print("  Reconstructing from fused pyramid...")
    # fused_lp entries are (H, W, 3); reconstruct each channel independently.
    image = fused_lp[-1].copy()
    for band in reversed(fused_lp[:-1]):
        expanded = np.stack(
            [expand(image[:, :, c], band.shape[:2]) for c in range(3)], axis=-1
        )
        image = expanded + band
    fused_lab = image
    print(f"    done ({_elapsed(t)})")

    t = time.perf_counter()
    print("  Suppressing chroma in dark regions...")
    fused_lab = _suppress_dark_chroma(fused_lab, dark_threshold)
    print(f"    done ({_elapsed(t)})")

    t = time.perf_counter()
    print("  Converting back to RGB...")
    clipped = np.clip(fused_lab, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(clipped, cv2.COLOR_Lab2RGB)
    print(f"    done ({_elapsed(t)})")

    return result


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
    parser.add_argument(
        "--performance", choices=["low", "high"], default="low",
        help=(
            "Memory/performance trade-off (default: low). "
            "'low' minimises peak memory by recomputing expanded luminance bands during accumulation. "
            "'high' caches expanded luminance bands in an extra (N, H, W) buffer to halve convolution "
            "calls on the luminance channel, at the cost of higher peak memory."
        ),
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"Error: '{args.folder}' is not a directory.")
        sys.exit(1)

    checkpoints: list[_Checkpoint] = []
    t_total = time.perf_counter()
    tracemalloc.start()
    t = t_total

    print(f"Loading images from '{args.folder}'...")
    images, size = load_images(args.folder)
    t = _snap("Load", t, checkpoints)

    h, w = images[0].shape[:2]
    levels = args.levels if args.levels > 0 else compute_levels((h, w))
    print(f"Image size: {w}x{h}  |  Images: {len(images)}  |  Pyramid levels: {levels}  |  Sharpness: {args.sharpness}  |  Dark threshold: {args.dark_threshold}")

    if not args.no_align:
        print("Aligning (ECC affine, neighbour-chained)...")
        images = align_images(images, min_shift=args.min_shift)
        t = _snap("Align", t, checkpoints)
    else:
        print("Skipping alignment.")

    print("Stacking...")
    result = stack_images(images, levels, args.sharpness, args.dark_threshold, args.performance == "high")
    del images
    t = _snap("Stack", t, checkpoints)

    out_path = args.folder / "stacked.jpg"
    Image.fromarray(result).save(out_path, "JPEG", quality=args.quality)
    t = _snap("Save", t, checkpoints)

    tracemalloc.stop()

    print(f"Saved: {out_path}")
    _print_report(checkpoints, time.perf_counter() - t_total)



if __name__ == "__main__":
    main()