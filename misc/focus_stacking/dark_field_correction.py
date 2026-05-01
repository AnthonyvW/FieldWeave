from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter, maximum_filter, binary_dilation


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_HOT_PIXEL_SIGMA = 5.0
HOT_PIXEL_FILTER_SIZE = 5

FIXED_SATURATION_FRACTION = 0.99

PSF_CENTRE_THRESHOLD_FACTOR = 3.0
PSF_REPLACEMENT_RADIUS = 2


def load_image(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (float32 array, original array, image info). Single file open."""
    img = Image.open(path)
    info = img.info
    original = np.array(img)
    return original.astype(np.float32), original, info


def save_image(array: np.ndarray, original: np.ndarray, output_path: Path, info: dict) -> None:
    dtype = original.dtype
    mode = Image.fromarray(original).mode

    if np.issubdtype(dtype, np.unsignedinteger):
        max_val = np.iinfo(dtype).max
        clipped = np.clip(array, 0, max_val).astype(dtype)
    elif np.issubdtype(dtype, np.floating):
        clipped = array.astype(dtype)
    else:
        clipped = np.clip(array, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(clipped, mode=mode).save(
        output_path, **{k: v for k, v in info.items() if k in ("dpi", "compress_level")}
    )


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_dark_stack(dark_paths: list[Path]) -> tuple[np.ndarray, np.dtype]:
    """Return (float32 stack, original dtype). Loads each frame once."""
    print(f"Loading {len(dark_paths)} dark frame(s)...")
    frames = []
    original_dtype = None
    for p in dark_paths:
        float_arr, original, _ = load_image(p)
        frames.append(float_arr)
        if original_dtype is None:
            original_dtype = original.dtype
    return np.stack(frames, axis=0), original_dtype


def build_master_dark(stack: np.ndarray, method: str) -> np.ndarray:
    if method == "median":
        print("Computing median dark frame...")
        return np.median(stack, axis=0).astype(np.float32)
    else:
        print("Computing mean dark frame...")
        return np.mean(stack, axis=0).astype(np.float32)


def infer_max_value(dtype: np.dtype) -> float:
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


def build_fixed_hot_pixel_map(stack: np.ndarray, original_dtype: np.dtype) -> np.ndarray:
    """Flag pixels at or above FIXED_SATURATION_FRACTION of max in every dark frame."""
    threshold = FIXED_SATURATION_FRACTION * infer_max_value(original_dtype)
    saturated_in_all = np.all(stack >= threshold, axis=0)
    if saturated_in_all.ndim == 3:
        return np.any(saturated_in_all, axis=2)
    return saturated_in_all


def _psf_centres_single_channel(channel: np.ndarray, threshold_factor: float, window: int) -> np.ndarray:
    background = float(np.median(channel))
    threshold = background + threshold_factor * max(background, 1.0)
    local_max = maximum_filter(channel, size=window)
    return (channel >= threshold) & (channel == local_max)


def build_psf_hot_pixel_map(
    master_dark: np.ndarray, threshold_factor: float, radius: int
) -> tuple[np.ndarray, int]:
    """Detect PSF hot pixel centres, dilate into replacement footprints."""
    window = 2 * radius + 1

    if master_dark.ndim == 3:
        centre_map = np.zeros(master_dark.shape[:2], dtype=bool)
        for c in range(master_dark.shape[2]):
            centre_map |= _psf_centres_single_channel(master_dark[..., c], threshold_factor, window)
    else:
        centre_map = _psf_centres_single_channel(master_dark, threshold_factor, window)

    n_centres = int(np.sum(centre_map))
    struct = np.ones((window, window), dtype=bool)
    expanded = binary_dilation(centre_map, structure=struct)
    return expanded, n_centres


def apply_hot_pixel_map(
    image: np.ndarray,
    centre_coords: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Replace a radius×radius region around each hot pixel centre with the local
    neighborhood median. Operates on small per-centre patches rather than the full
    image, which is orders of magnitude faster when hot pixels are sparse.

    centre_coords: (N, 2) array of (row, col) centre positions.
    """
    if len(centre_coords) == 0:
        return image

    h, w = image.shape[:2]
    pad = HOT_PIXEL_FILTER_SIZE // 2
    result = image.copy()

    for cy, cx in centre_coords:
        # Fetch a patch with enough border for the median filter to be accurate.
        sr0 = max(0, cy - radius - pad)
        sr1 = min(h, cy + radius + pad + 1)
        sc0 = max(0, cx - radius - pad)
        sc1 = min(w, cx + radius + pad + 1)

        # Replacement slice within the patch.
        rr0 = cy - radius - sr0
        rr1 = cy + radius + 1 - sr0
        rc0 = cx - radius - sc0
        rc1 = cx + radius + 1 - sc0

        if image.ndim == 3:
            for c in range(image.shape[2]):
                filtered = median_filter(image[sr0:sr1, sc0:sc1, c], size=HOT_PIXEL_FILTER_SIZE)
                result[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1, c] = filtered[rr0:rr1, rc0:rc1]
        else:
            filtered = median_filter(image[sr0:sr1, sc0:sc1], size=HOT_PIXEL_FILTER_SIZE)
            result[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1] = filtered[rr0:rr1, rc0:rc1]

    return result


def apply_hot_pixel_map_mask(image: np.ndarray, hot_map: np.ndarray) -> np.ndarray:
    """Mask-based replacement used by spatial and fixed modes, which don't have
    discrete centre coordinates to iterate over."""
    if not np.any(hot_map):
        return image

    result = image.copy()
    if image.ndim == 3:
        for c in range(image.shape[2]):
            filtered = median_filter(image[..., c], size=HOT_PIXEL_FILTER_SIZE)
            result[..., c][hot_map] = filtered[hot_map]
    else:
        filtered = median_filter(image, size=HOT_PIXEL_FILTER_SIZE)
        result[hot_map] = filtered[hot_map]
    return result


def clean_hot_pixels_spatial(dark: np.ndarray, sigma: float) -> tuple[np.ndarray, int]:
    if dark.ndim == 3:
        cleaned = np.empty_like(dark)
        total_count = 0
        for c in range(dark.shape[2]):
            cleaned[..., c], count = clean_hot_pixels_spatial(dark[..., c], sigma)
            total_count += count
        return cleaned, total_count

    local_median = median_filter(dark, size=HOT_PIXEL_FILTER_SIZE)
    residual = dark - local_median
    hot_mask = residual > sigma * float(np.std(residual))
    cleaned = dark.copy()
    cleaned[hot_mask] = local_median[hot_mask]
    return cleaned, int(np.sum(hot_mask))


def correct_image(image: np.ndarray, master_dark: np.ndarray) -> np.ndarray:
    np.subtract(image, master_dark, out=image)
    np.maximum(image, 0, out=image)
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dark field correction: subtract a master dark frame from a folder of images."
    )
    parser.add_argument("--dark-dir", required=True, type=Path,
                        help="Folder containing lens-cap (dark) images.")
    parser.add_argument("--image-dir", required=True, type=Path,
                        help="Folder containing images to correct.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Folder to write corrected images into.")
    parser.add_argument("--method", choices=["median", "mean"], default=None,
                        help="How to combine dark frames into a master dark. Prompted if not provided.")
    parser.add_argument(
        "--hot-pixel-mode",
        choices=["spatial", "fixed", "psf"],
        default="psf",
        help=(
            "How to detect and remove hot pixels. "
            "'psf' (default): detects hot pixel centres as local maxima above background in the master dark, "
            "then replaces a PSF-sized region around each centre in the science images. "
            "Best for demosaiced cameras where hot pixels spread into neighboring pixels. "
            "'spatial': flags pixels that are outliers relative to their neighborhood in the master dark. "
            "'fixed': flags pixels at or above FIXED_SATURATION_FRACTION of the sensor maximum in every dark frame."
        ),
    )
    parser.add_argument(
        "--hot-pixel-sigma", type=float, default=DEFAULT_HOT_PIXEL_SIGMA, metavar="SIGMA",
        help=f"Sigma threshold for spatial hot pixel detection (default: {DEFAULT_HOT_PIXEL_SIGMA}). Only used in spatial mode.",
    )
    parser.add_argument(
        "--psf-threshold-factor", type=float, default=PSF_CENTRE_THRESHOLD_FACTOR, metavar="FACTOR",
        help=f"PSF mode: a pixel is a hot pixel centre if it exceeds background × this factor (default: {PSF_CENTRE_THRESHOLD_FACTOR}).",
    )
    parser.add_argument(
        "--psf-radius", type=int, default=PSF_REPLACEMENT_RADIUS, metavar="RADIUS",
        help=f"PSF mode: radius in pixels of the region replaced around each hot pixel centre (default: {PSF_REPLACEMENT_RADIUS}, giving a 5×5 region).",
    )
    return parser.parse_args()


def prompt_method() -> str:
    while True:
        choice = input("Combine dark frames using [m]edian or [a]verage (mean)? ").strip().lower()
        if choice in {"m", "median"}:
            return "median"
        if choice in {"a", "mean", "average"}:
            return "mean"
        print("Please enter 'm' for median or 'a' for mean.")


def main() -> None:
    args = parse_args()

    if not args.dark_dir.is_dir():
        print(f"Dark directory not found: {args.dark_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.image_dir.is_dir():
        print(f"Image directory not found: {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    method = args.method if args.method else prompt_method()

    dark_paths = collect_images(args.dark_dir)
    if not dark_paths:
        print(f"No supported images found in dark directory: {args.dark_dir}", file=sys.stderr)
        sys.exit(1)

    image_paths = collect_images(args.image_dir)
    if not image_paths:
        print(f"No supported images found in image directory: {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {args.output_dir}")

    stack, original_dtype = load_dark_stack(dark_paths)
    master_dark = build_master_dark(stack, method)

    # centre_coords is used by psf mode; hot_map is used by spatial/fixed modes.
    centre_coords: np.ndarray | None = None
    hot_map: np.ndarray | None = None
    psf_radius = args.psf_radius

    if args.hot_pixel_mode == "psf":
        print(f"Building PSF hot pixel map (threshold factor={args.psf_threshold_factor}, radius={psf_radius})...")
        expanded_map, n_centres = build_psf_hot_pixel_map(master_dark, args.psf_threshold_factor, psf_radius)
        centre_coords = np.argwhere(
            _psf_centres_single_channel(
                master_dark.max(axis=2) if master_dark.ndim == 3 else master_dark,
                args.psf_threshold_factor,
                2 * psf_radius + 1,
            )
        )
        print(f"  Found {n_centres} hot pixel centre(s).")
    elif args.hot_pixel_mode == "fixed":
        print(f"Building fixed hot pixel map (saturation fraction={FIXED_SATURATION_FRACTION})...")
        hot_map = build_fixed_hot_pixel_map(stack, original_dtype)
        print(f"  Flagged {int(np.sum(hot_map))} fixed hot pixel location(s).")
        master_dark = apply_hot_pixel_map_mask(master_dark, hot_map)
    else:
        print(f"Cleaning hot pixels from master dark (spatial, sigma={args.hot_pixel_sigma})...")
        master_dark, hot_count = clean_hot_pixels_spatial(master_dark, args.hot_pixel_sigma)
        print(f"  Replaced {hot_count} hot pixel(s).")

    del stack

    print(f"\nCorrecting {len(image_paths)} image(s)...")
    for i, path in enumerate(image_paths, 1):
        print(f"  [{i}/{len(image_paths)}] {path.name}")
        image, original, info = load_image(path)

        if image.shape != master_dark.shape:
            print(
                f"    WARNING: shape mismatch — image {image.shape} vs dark {master_dark.shape}, skipping.",
                file=sys.stderr,
            )
            continue

        corrected = correct_image(image, master_dark)

        if centre_coords is not None:
            corrected = apply_hot_pixel_map(corrected, centre_coords, psf_radius)
        elif hot_map is not None:
            corrected = apply_hot_pixel_map_mask(corrected, hot_map)

        save_image(corrected, original, args.output_dir / path.name, info)

    print(f"\nDone. Corrected images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()