from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

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
    tracemalloc.reset_peak()
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

def load_images(folder: Path) -> tuple[list[Path], tuple[int, int]]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(paths) < 2:
        print(f"Error: need at least 2 images in '{folder}', found {len(paths)}.")
        sys.exit(1)

    img0 = Image.open(paths[0]).convert("RGB")
    reference_size: tuple[int, int] = img0.size
    n = len(paths)
    print(f"  Found {n} images, reference size: {reference_size[0]}x{reference_size[1]}")
    return paths, reference_size


def _to_gray_cv(image: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB (H,W,3) to uint8 grayscale using OpenCV."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _ecc_align(ref_gray: np.ndarray, src_gray: np.ndarray,
               max_resolution: int, rough: bool) -> np.ndarray:
    """Single cv2 ECC alignment pass.

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


def align_images(
    src_paths: list[Path],
    reference_size: tuple[int, int],
    min_shift: float = 5.0,
) -> list[np.ndarray]:
    """Compute cumulative affine warps for all images using neighbour-chained ECC.

    ECC is run on raw unwarped grayscale neighbor pairs so that interpolation
    error from warp application never feeds back into subsequent alignment steps.
    Warp matrices are chained purely mathematically. Only two raw grayscale
    arrays are live at any time.

    Returns a list of 2×3 float32 warp matrices, one per image. The first is
    always identity. Images that fail to converge or fall below min_shift get
    the previous cumulative warp (not identity) so the chain remains consistent.
    """
    ref_w, ref_h = reference_size
    identity = np.eye(2, 3, dtype=np.float32)
    warps: list[np.ndarray] = [identity.copy()]
    cumulative_warp = identity.copy()

    def _load_raw_gray(path: Path) -> np.ndarray:
        img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        if img.shape[1] != ref_w or img.shape[0] != ref_h:
            img = cv2.resize(img, (ref_w, ref_h), interpolation=cv2.INTER_AREA)
        return _to_gray_cv(img)

    prev_gray = _load_raw_gray(src_paths[0])

    for i in range(1, len(src_paths)):
        src_gray = _load_raw_gray(src_paths[i])

        warp = identity.copy()
        converged = True
        for max_res, rough in [(256, True), (2048, False)]:
            try:
                warp = _ecc_align(prev_gray, src_gray, max_res, rough)
            except cv2.error:
                converged = False
                break

        prev_gray = src_gray

        if not converged:
            print(f"  Image {i + 1}: ECC did not converge — using previous transform")
            warps.append(cumulative_warp.copy())
            continue

        cumulative_warp = _chain_affines(cumulative_warp, warp)

        translation = np.linalg.norm(cumulative_warp[:, 2])
        is_identity_linear = np.allclose(cumulative_warp[:, :2], np.eye(2), atol=1e-3)
        if translation < min_shift and is_identity_linear:
            print(f"  Image {i + 1}: transform negligible — skipped")
            warps.append(identity.copy())
            continue

        angle = np.degrees(np.arctan2(cumulative_warp[1, 0], cumulative_warp[0, 0]))
        tx, ty = cumulative_warp[0, 2], cumulative_warp[1, 2]
        print(f"  Image {i + 1}: rotation {angle:+.2f}°  shift ({ty:+.1f}, {tx:+.1f}) px")
        warps.append(cumulative_warp.copy())

    return warps



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




def _image_to_lab(img: np.ndarray) -> np.ndarray:
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2Lab).astype(np.float32)


def _lab_lap_pyramid(lab: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build full 3-channel Laplacian pyramid bands for a Lab image.

    Returns levels+1 entries of shape (H_l, W_l, 3). Builds one level at a
    time, keeping only the current and next Gaussian in memory simultaneously
    so the full Gaussian stack is never allocated at once.
    The luminance Laplacian bands are the [:,:,0] slice of each entry.
    """
    bands: list[np.ndarray] = []
    current = lab
    for _ in range(levels):
        nxt = np.empty((
            (current.shape[0] + 1) // 2,
            (current.shape[1] + 1) // 2,
            3,
        ), dtype=np.float32)
        for c in range(3):
            nxt[:, :, c] = reduce(current[:, :, c])
        exp = np.empty_like(current)
        for c in range(3):
            exp[:, :, c] = expand(nxt[:, :, c], current.shape[:2])
        bands.append(current - exp)
        current = nxt
    bands.append(current)
    return bands


def _load_and_warp(path: Path, warp: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Load an image from disk, resize if needed, and apply a pre-computed affine warp.

    size is (w, h) as expected by cv2.warpAffine.
    Returns a float32 RGB array.
    """
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    if img.shape[1] != size[0] or img.shape[0] != size[1]:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if not np.array_equal(warp, np.eye(2, 3, dtype=np.float32)):
        u8 = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.warpAffine(u8, warp, size,
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    return img


def stack_images(
    src_paths: list[Path],
    warps: list[np.ndarray],
    levels: int,
    sharpness: float,
    dark_threshold: float,
    workers: int = 3,
) -> np.ndarray:
    """Fuse a stack of images using single-pass unnormalized Laplacian pyramid fusion.

    Accumulates energy-weighted Lab bands and energy sums in one disk read per
    image, then divides at the end. Mathematically identical to normalized fusion:
      Σ(e_k / Σe_k · x_k) = Σ(e_k · x_k) / Σe_k

    workers controls how many images are processed concurrently. Peak RAM scales
    with workers × ~100 MiB per image plus the fixed fused_lp accumulator.
    Default of 3 workers balances speed and memory for most systems.
    """

    img0 = np.array(Image.open(src_paths[0]).convert("RGB"), dtype=np.float32)
    h, w = img0.shape[:2]
    del img0
    cv2_size = (w, h)

    n_workers = min(len(src_paths), workers if workers > 0 else (os.cpu_count() or 4))

    def _process_image(args: tuple[Path, np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        path, warp = args
        lab = _image_to_lab(_load_and_warp(path, warp, cv2_size))
        lap = _lab_lap_pyramid(lab, levels)
        del lab
        energies: list[np.ndarray] = []
        for i in range(levels):
            energies.append(region_energy(lap[i][:, :, 0]) ** sharpness)
        lv = lap[-1][:, :, 0]
        energies.append(((region_deviation(lv) + region_entropy(lv)) * 0.5) ** sharpness)
        return energies, lap

    t = time.perf_counter()
    print(f"  Fusing ({n_workers} workers)...")
    energy_sums: list[np.ndarray | None] = [None] * (levels + 1)
    fused_lp: list[np.ndarray | None] = [None] * (levels + 1)
    weighted_buf: np.ndarray | None = None

    items = list(zip(src_paths, warps))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        pending: dict = {}
        submit_idx = 0

        while submit_idx < len(items) and len(pending) < n_workers:
            f = pool.submit(_process_image, items[submit_idx])
            pending[f] = submit_idx
            submit_idx += 1

        while pending:
            done = next(as_completed(pending))
            pending.pop(done)

            if submit_idx < len(items):
                f = pool.submit(_process_image, items[submit_idx])
                pending[f] = submit_idx
                submit_idx += 1

            energies, lap = done.result()
            for i in range(levels + 1):
                e = energies[i]
                band = lap[i]
                lap[i] = None  # type: ignore[call-overload]
                energy_sums[i] = e if energy_sums[i] is None else energy_sums[i] + e
                if weighted_buf is None or weighted_buf.shape != band.shape:
                    weighted_buf = np.empty_like(band)
                np.multiply(band, e[:, :, np.newaxis], out=weighted_buf)
                if fused_lp[i] is None:
                    fused_lp[i] = weighted_buf.copy()
                else:
                    fused_lp[i] += weighted_buf  # type: ignore[operator]

    for i in range(levels + 1):
        fused_lp[i] /= energy_sums[i][:, :, np.newaxis] + 1e-10  # type: ignore[operator,index]

    print(f"    done ({_elapsed(t)})")

    t = time.perf_counter()
    print("  Reconstructing...")
    image = fused_lp[-1].copy()  # type: ignore[union-attr]
    for band in reversed(fused_lp[:-1]):
        cur_shape = band.shape[:2]  # type: ignore[union-attr]
        exp = np.stack([expand(image[:, :, c], cur_shape) for c in range(3)], axis=-1)
        image = exp + band  # type: ignore[operator]
    fused_lab = image
    print(f"    done ({_elapsed(t)})")

    t = time.perf_counter()
    print("  Suppressing chroma in dark regions...")
    fused_lab = _suppress_dark_chroma(fused_lab, dark_threshold)
    print(f"    done ({_elapsed(t)})")

    t = time.perf_counter()
    print("  Converting back to RGB...")
    result = cv2.cvtColor(np.clip(fused_lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2RGB)
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
        help="Skip ECC alignment (use when images are already registered).",
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
        "--workers", type=int, default=3,
        help=(
            "Number of parallel workers for stacking (default: 3). "
            "Higher values are faster but increase peak RAM by ~100 MiB per additional worker. "
            "Set to 0 to use all CPU cores."
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
    src_paths, reference_size = load_images(args.folder)
    t = _snap("Load", t, checkpoints)

    ref_w, ref_h = reference_size
    levels = args.levels if args.levels > 0 else compute_levels((ref_h, ref_w))
    print(f"Image size: {ref_w}x{ref_h}  |  Images: {len(src_paths)}  |  Pyramid levels: {levels}  |  Sharpness: {args.sharpness}  |  Dark threshold: {args.dark_threshold}")

    identity = np.eye(2, 3, dtype=np.float32)
    if not args.no_align:
        print("Aligning (ECC affine, neighbour-chained)...")
        warps = align_images(src_paths, reference_size, min_shift=args.min_shift)
        t = _snap("Align", t, checkpoints)
    else:
        print("Skipping alignment.")
        warps = [identity.copy() for _ in src_paths]

    print("Stacking...")
    result = stack_images(src_paths, warps, levels, args.sharpness, args.dark_threshold, args.workers)
    t = _snap("Stack", t, checkpoints)

    out_path = args.folder / "stacked.jpg"
    Image.fromarray(result).save(out_path, "JPEG", quality=args.quality)
    t = _snap("Save", t, checkpoints)

    tracemalloc.stop()
    print(f"Saved: {out_path}")
    _print_report(checkpoints, time.perf_counter() - t_total)



if __name__ == "__main__":
    main()