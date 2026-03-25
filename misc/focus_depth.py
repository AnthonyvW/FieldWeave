"""
Focus Stack Depth Map Generator
================================
Given a folder of focus-stacked images whose filenames are the focal distance
in nanometers (e.g. "2400000.jpg") and the combined focus-stacked result image,
this script:

  1. Computes a per-pixel sharpness map for each image in the stack.
  2. Assigns each pixel the distance (nm) of its sharpest frame.
     detection + border flood-fill.
  4. Opens an interactive mask editor so you can hand-paint additional exclusion
     regions (e.g. noisy background blobs) over the stacked image.
       - Left-drag  : paint exclusion (red overlay)
       - Right-drag : erase exclusion
       - Scroll     : change brush size
       - S          : save mask PNG and continue to 3-D plot
       - R          : reset hand-drawn mask to all-clear
       - Q / Escape : quit without plotting
     The hand-drawn mask is saved as a PNG (255 = exclude, 0 = keep) and can
     be reloaded on subsequent runs with --mask so you don't have to redraw.
  5. Applies Gaussian blur to the depth map within the surviving foreground
     region to smooth sensor-noise fluctuations.
  6. Saves a PNG depth map (excluded pixels = black).
  7. Shows an interactive 3-D surface; press T to toggle between RGB texture
     (from the stacked image) and INFERNO depth colormap.

Usage:
    python focus_depth_map.py <input_folder> --stacked <path> [options]

Options:
    --stacked PATH        Combined focus-stacked image  [required]
    --mask PATH           Previously saved hand-drawn mask PNG to reload
    --mask-output PATH    Where to save the hand-drawn mask
                          (default: <stacked_image_stem>_mask.png)
    --output PATH         Output depth-map PNG  (default: <folder>/depth_map.png)
    --operator OP         laplacian | tenengrad | glvn  (default: laplacian)
    --blur-sigma FLOAT    Pre-blur before focus measure  (default: 1.0, 0=off)
    --smooth-sigma FLOAT  Smoothing on each raw focus map  (default: 3.0, 0=off)
    --canny-low FLOAT     Canny lower threshold  (default: 30)
    --blur-depth FLOAT    Gaussian sigma for depth-map smoothing after masking
                          (default: 3.0, 0=off)
    --brush-size INT      Initial brush radius in pixels  (default: 20)
    --colormap            Save false-color (INFERNO) PNG
    --downsample INT      Pixel stride for 3-D plot  (default: 4)
    --no-editor           Skip the mask editor (use --mask or auto-mask only)
    --no-save             Skip saving the depth map PNG
    --no-plot             Skip the interactive 3-D viewer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Focus measure operators
# ---------------------------------------------------------------------------

def focus_laplacian(gray: np.ndarray, blur_sigma: float) -> np.ndarray:
    if blur_sigma > 0:
        k = int(blur_sigma * 6) | 1
        gray = cv2.GaussianBlur(gray, (k, k), blur_sigma)
    return np.abs(cv2.Laplacian(gray, cv2.CV_64F))


def focus_tenengrad(gray: np.ndarray, blur_sigma: float) -> np.ndarray:
    if blur_sigma > 0:
        k = int(blur_sigma * 6) | 1
        gray = cv2.GaussianBlur(gray, (k, k), blur_sigma)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return gx ** 2 + gy ** 2


def focus_glvn(gray: np.ndarray, blur_sigma: float) -> np.ndarray:
    if blur_sigma > 0:
        k = int(blur_sigma * 6) | 1
        gray = cv2.GaussianBlur(gray, (k, k), blur_sigma)
    gray_f = gray.astype(np.float64)
    mu  = cv2.blur(gray_f,      (5, 5))
    mu2 = cv2.blur(gray_f ** 2, (5, 5))
    return np.maximum(mu2 - mu ** 2, 0)


OPERATORS: dict[str, object] = {
    "laplacian": focus_laplacian,
    "tenengrad": focus_tenengrad,
    "glvn":      focus_glvn,
}


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _is_date_stem(stem: str) -> bool:
    """Return True for stems that look like datestamps rather than focal distances.

    Recognised patterns (all must start with '2026'):
      - 8 digits            e.g. '20260226'
      - YYYYMMDD_HHMMSS     e.g. '20260217_152840'
    """
    if not stem.startswith("2026"):
        return False
    # plain 8-digit date
    if len(stem) == 8 and stem.isdigit():
        return True
    # datetime with underscore separator: 8 digits + '_' + 6 digits
    if (len(stem) == 15 and stem[8] == "_"
            and stem[:8].isdigit() and stem[9:].isdigit()):
        return True
    return False


def _parse_distance_nm(path: Path) -> float | None:
    stem = path.stem
    if _is_date_stem(stem):
        return None
    try:
        return float(stem)
    except ValueError:
        return None


def load_images(folder: Path) -> tuple[list[np.ndarray], list[float]]:
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    candidates = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]
    if not candidates:
        print(f"Error: no supported images found in '{folder}'", file=sys.stderr)
        sys.exit(1)

    # Split into numeric-named (have z distance) and non-numeric (no z info)
    named:   list[tuple[float, Path]] = []
    unnamed: list[Path]               = []
    for p in candidates:
        nm = _parse_distance_nm(p)
        if nm is not None:
            named.append((nm, p))
        else:
            unnamed.append(p)

    STEP_NM = 200_000.0  # 0.2 mm expressed in nanometres

    if named and unnamed:
        # Mixed — only use the numeric ones and warn about the rest
        for p in unnamed:
            print(f"Warning: '{p.name}' has no numeric distance — skipping.", file=sys.stderr)
        paths_sorted = [p for _, p in sorted(named, key=lambda t: t[0])]
        distances    = [nm for nm, _ in sorted(named, key=lambda t: t[0])]
    elif named:
        # All files have numeric names
        pairs        = sorted(named, key=lambda t: t[0])
        paths_sorted = [p for _, p in pairs]
        distances    = [nm for _, p in pairs]
    else:
        # No numeric names at all — sort alphabetically and assign 0.2 mm steps
        paths_sorted = sorted(unnamed)
        distances    = [i * STEP_NM for i in range(len(paths_sorted))]
        print(
            f"No numeric filenames found — assigning distances "
            f"0 to {distances[-1]:.0f} nm in {STEP_NM:.0f} nm (0.2 mm) steps."
        )

    images: list[np.ndarray] = []
    kept_distances: list[float] = []
    for nm, p in zip(distances, paths_sorted):
        img = cv2.imread(str(p))
        if img is None:
            print(f"Warning: could not read '{p.name}', skipping.", file=sys.stderr)
            continue
        images.append(img)
        kept_distances.append(nm)

    if not images:
        print("Error: failed to load any images.", file=sys.stderr)
        sys.exit(1)

    h, w = images[0].shape[:2]
    for i, img in enumerate(images[1:], start=2):
        if img.shape[:2] != (h, w):
            print(
                f"Error: image {i} has shape {img.shape[:2]} but expected ({h}, {w}).",
                file=sys.stderr,
            )
            sys.exit(1)

    print(
        f"Loaded {len(images)} images ({w}x{h})  "
        f"| distance range: {kept_distances[0]:.0f} \u2013 {kept_distances[-1]:.0f} nm"
    )
    return images, kept_distances


def load_stacked(path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        print(f"Error: could not read stacked image '{path}'", file=sys.stderr)
        sys.exit(1)
    h, w = target_shape
    if img.shape[:2] != (h, w):
        print(f"  Resizing stacked image from {img.shape[1]}x{img.shape[0]} to {w}x{h}.")
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------
# Depth map computation
# ---------------------------------------------------------------------------

def compute_depth_map(
    images: list[np.ndarray],
    distances_nm: list[float],
    operator: str,
    blur_sigma: float,
    smooth_sigma: float,
) -> np.ndarray:
    """Returns float32 (H, W) — nm distance of the best-focus frame per pixel."""
    focus_fn    = OPERATORS[operator]
    n           = len(images)
    h, w        = images[0].shape[:2]
    focus_stack = np.zeros((n, h, w), dtype=np.float64)

    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        fm   = focus_fn(gray, blur_sigma)
        if smooth_sigma > 0:
            k  = int(smooth_sigma * 6) | 1
            fm = cv2.GaussianBlur(fm, (k, k), smooth_sigma)
        focus_stack[i] = fm
        print(f"  [{i + 1:>{len(str(n))}}/{n}] focus computed  ({distances_nm[i]:.0f} nm)")

    best_idx   = np.argmax(focus_stack, axis=0)
    dist_array = np.array(distances_nm, dtype=np.float32)
    return dist_array[best_idx]
# ---------------------------------------------------------------------------
# Depth smoothing
# ---------------------------------------------------------------------------

def smooth_depth(
    depth_nm: np.ndarray,
    keep_mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Gaussian blur of depth values within *keep_mask*.
    Background pixels are filled with the foreground median during the blur
    so border pixels aren't contaminated, then restored to NaN-equivalent.
    """
    if sigma <= 0:
        return depth_nm.copy()

    depth   = depth_nm.astype(np.float64)
    bg_fill = float(np.median(depth[keep_mask])) if keep_mask.any() else 0.0
    tmp     = np.where(keep_mask, depth, bg_fill)
    tmp     = gaussian_filter(tmp, sigma=sigma)
    return np.where(keep_mask, tmp, depth).astype(np.float32)


# ---------------------------------------------------------------------------
# PNG output
# ---------------------------------------------------------------------------

def save_depth_map(
    depth_nm: np.ndarray,
    keep_mask: np.ndarray,
    output_path: Path,
    use_colormap: bool,
) -> None:
    fg = depth_nm[keep_mask]
    if fg.size == 0:
        print("Warning: mask excludes all pixels — depth map will be blank.")
        cv2.imwrite(str(output_path), np.zeros(depth_nm.shape, dtype=np.uint8))
        return

    mn, mx = float(fg.min()), float(fg.max())
    span   = mx - mn if mx != mn else 1.0

    normalized          = np.zeros(depth_nm.shape, dtype=np.uint8)
    normalized[keep_mask] = ((depth_nm[keep_mask] - mn) / span * 255).astype(np.uint8)

    if use_colormap:
        out               = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        out[~keep_mask]   = 0
    else:
        out = normalized

    cv2.imwrite(str(output_path), out)
    print(f"Depth map saved \u2192 '{output_path}'")


# ---------------------------------------------------------------------------
# 3-D viewer helpers
# ---------------------------------------------------------------------------

def _best_unit(span_nm: float) -> tuple[str, str, float]:
    if span_nm >= 1e6:
        return "mm", "mm", 1e6
    if span_nm >= 1e3:
        return "um", "\u00b5m", 1e3
    return "nm", "nm", 1.0


def _build_rgb_facecolors(
    stacked_bgr: np.ndarray,
    keep_mask: np.ndarray,
    downsample: int,
) -> np.ndarray:
    d   = max(1, downsample)
    rgb = cv2.cvtColor(stacked_bgr, cv2.COLOR_BGR2RGB)[::d, ::d].astype(np.float32) / 255.0
    a   = keep_mask[::d, ::d].astype(np.float32)
    return np.dstack([rgb, a])


def _build_depth_facecolors(
    z: np.ndarray,
    z_min: float,
    z_max: float,
    keep_ds: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(z) & keep_ds
    z_norm = np.zeros_like(z)
    span   = max(z_max - z_min, 1.0)
    z_norm[finite] = (z[finite] - z_min) / span
    fc             = cm.inferno(z_norm).astype(np.float32)
    fc[~finite, 3] = 0.0
    return fc


# ---------------------------------------------------------------------------
# 3-D interactive viewer
# ---------------------------------------------------------------------------

def show_3d(
    depth_nm: np.ndarray,
    keep_mask: np.ndarray,
    stacked_bgr: np.ndarray,
    downsample: int,
) -> None:
    """
    Interactive 3-D surface. Press T to toggle RGB texture / INFERNO depth.
    Default view: elev=25, azim=315.
    """
    d = max(1, downsample)

    depth_masked          = depth_nm.astype(np.float64).copy()
    depth_masked[~keep_mask] = np.nan
    z = depth_masked[::d, ::d]

    h_ds, w_ds = z.shape
    X, Y       = np.meshgrid(np.arange(w_ds) * d, np.arange(h_ds) * d)

    valid = z[np.isfinite(z)]
    if valid.size == 0:
        print("Warning: no foreground pixels after masking — nothing to plot.")
        return

    z_min, z_max = float(valid.min()), float(valid.max())
    span_nm      = z_max - z_min
    _, unit_label, divisor = _best_unit(span_nm)

    keep_ds  = keep_mask[::d, ::d]
    fc_rgb   = _build_rgb_facecolors(stacked_bgr, keep_mask, downsample)
    fc_depth = _build_depth_facecolors(z, z_min, z_max, keep_ds)

    fig = plt.figure(figsize=(13, 8), facecolor="#0d0d0d")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#0d0d0d")

    surf = ax.plot_surface(
        X, Y, z,
        facecolors=fc_rgb,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#2a2a2a")

    ax.tick_params(colors="#777777", labelsize=8)
    ax.set_xlabel("X  (px)",                color="#aaaaaa", labelpad=8,  fontsize=9)
    ax.set_ylabel("Y  (px)",                color="#aaaaaa", labelpad=8,  fontsize=9)
    ax.set_zlabel(f"Depth  ({unit_label})",  color="#aaaaaa", labelpad=10, fontsize=9)
    ax.zaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / divisor:.3g}"))

    mappable = cm.ScalarMappable(cmap="inferno")
    mappable.set_array(valid)
    mappable.set_clim(z_min, z_max)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.08, aspect=25)
    cbar.set_label(f"Distance  ({unit_label})", color="#aaaaaa", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / divisor:.3g}"))
    cbar.ax.tick_params(colors="#777777", labelsize=8)

    ax.view_init(elev=25, azim=315)

    fg_pct     = np.isfinite(depth_masked).mean() * 100
    mode_label = fig.text(
        0.5, 0.97,
        "Color mode: RGB texture  [press T to toggle]",
        ha="center", color="#dddddd", fontsize=11, fontweight="bold",
    )
    fig.text(
        0.5, 0.01,
        f"Z range: {z_min:.0f} \u2013 {z_max:.0f} nm  |  "
        f"foreground: {fg_pct:.1f}%  |  "
        f"downsample \u00d7{d}  |  drag to rotate",
        ha="center", color="#555555", fontsize=8,
    )

    state = {"rgb": True}

    def on_key(event: object) -> None:
        if getattr(event, "key", "").lower() != "t":
            return
        state["rgb"] = not state["rgb"]
        new_fc = fc_rgb if state["rgb"] else fc_depth
        surf.set_facecolors(new_fc.reshape(-1, 4))
        label = "RGB texture" if state["rgb"] else "INFERNO depth"
        mode_label.set_text(f"Color mode: {label}  [press T to toggle]")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate a depth map from a focus stack whose filenames encode "
            "focal distance in nanometers, with an interactive mask editor to "
            "exclude noisy regions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_folder", type=Path,
                   help="Folder containing the individual focus-stack frames.")
    p.add_argument("--stacked", type=Path, required=True, metavar="PATH",
                   help="Path to the combined focus-stacked image.")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output depth-map PNG (default: <folder>/depth_map.png).")
    p.add_argument("--operator", choices=list(OPERATORS.keys()), default="laplacian",
                   help="Focus measure operator.")
    p.add_argument("--blur-sigma", type=float, default=1.0,
                   help="Gaussian pre-blur sigma before focus measurement (0=off).")
    p.add_argument("--smooth-sigma", type=float, default=3.0,
                   help="Smoothing sigma on each raw focus map (0=off).")
    p.add_argument("--blur-depth", type=float, default=3.0,
                   help="Gaussian sigma for depth-map smoothing after masking (0=off).")
    p.add_argument("--colormap", action="store_true",
                   help="Save false-color (INFERNO) PNG instead of grayscale.")
    p.add_argument("--downsample", type=int, default=4,
                   help="Pixel stride for the 3-D surface plot (higher = faster).")
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving the depth map PNG.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the interactive 3-D viewer.")
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    folder: Path = args.input_folder
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    output: Path      = args.output or folder / "depth_map.png"

    print(f"Input folder  : {folder}")
    print(f"Stacked image : {args.stacked}")
    print(f"Depth output  : {output}")
    print(f"Operator      : {args.operator}")
    print(f"Blur sigma    : {args.blur_sigma}")
    print(f"Smooth sigma  : {args.smooth_sigma}")
    print(f"Blur depth σ  : {args.blur_depth}")
    print(f"Downsample    : {args.downsample}  (3-D plot)")
    print()

    # --- Load data ---
    images, distances_nm = load_images(folder)
    h, w = images[0].shape[:2]

    print("\nLoading stacked image...")
    stacked_bgr = load_stacked(args.stacked, target_shape=(h, w))

    # --- Compute depth map ---
    print("\nComputing focus measures...")
    depth_nm = compute_depth_map(
        images, distances_nm,
        operator=args.operator,
        blur_sigma=args.blur_sigma,
        smooth_sigma=args.smooth_sigma,
    )

    # --- All pixels kept (no mask) ---
    keep_mask = np.ones((h, w), dtype=bool)

    # Diagnostic: report the actual depth range computed
    d_min, d_max = float(depth_nm.min()), float(depth_nm.max())
    unique = len(np.unique(depth_nm))
    print(f"Depth range: {d_min:.0f} – {d_max:.0f} nm  ({unique} distinct values)")
    if unique <= 1:
        print("WARNING: depth map is completely flat — all pixels assigned the same "
              "distance. The focus stack may have too few frames or the focus measure "
              "scores are uniform. Try a different --operator or check your input images.",
              file=sys.stderr)
    # --- Gaussian blur depth map within keep region ---
    if args.blur_depth > 0:
        print(f"\nSmoothing depth map (sigma={args.blur_depth})...")
        depth_nm = smooth_depth(depth_nm, keep_mask, sigma=args.blur_depth)

    # --- Save PNG ---
    if not args.no_save:
        save_depth_map(depth_nm, keep_mask, output, use_colormap=args.colormap)

    # --- 3-D viewer ---
    if not args.no_plot:
        print("\nOpening 3-D viewer  (press T to toggle colors, close window to exit)...")
        show_3d(depth_nm, keep_mask, stacked_bgr, downsample=args.downsample)


if __name__ == "__main__":
    main()