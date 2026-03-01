"""
focus_overlay.py - Generate a focus heatmap overlay on an image.

Usage:
    python focus_overlay.py <image_path> [--output <output_path>] [--kernel-size 7]
                            [--window-size 64] [--stride 16] [--alpha 0.6]
                            [--colormap jet]
"""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np


def compute_focus_score(patch: np.ndarray, kernel_size: int) -> float:
    """Compute focus score for a single patch using the same mechanism as analyze_focus."""
    if len(patch.shape) == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(patch, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
    abs_laplacian = np.absolute(laplacian)

    variance = np.var(abs_laplacian)
    percentile_90 = np.percentile(abs_laplacian, 90)
    return (variance + percentile_90) / 2


def generate_focus_map(
    image: np.ndarray,
    kernel_size: int = 7,
    window_size: int = 64,
    stride: int = 16,
) -> np.ndarray:
    """
    Slide a window over the image, compute a focus score per window,
    and accumulate into a float score map of the same spatial size as image.

    Returns a 2D float32 array of unnormalized focus scores.
    """
    height, width = image.shape[:2]
    score_map = np.zeros((height, width), dtype=np.float64)
    weight_map = np.zeros((height, width), dtype=np.float64)

    half = window_size // 2

    y_centers = range(half, height - half + 1, stride)
    x_centers = range(half, width - half + 1, stride)

    for y in y_centers:
        for x in x_centers:
            y0, y1 = y - half, y + half
            x0, x1 = x - half, x + half
            patch = image[y0:y1, x0:x1]
            score = compute_focus_score(patch, kernel_size)
            score_map[y0:y1, x0:x1] += score
            weight_map[y0:y1, x0:x1] += 1.0

    # Avoid division by zero at image borders that were never covered
    mask = weight_map > 0
    score_map[mask] /= weight_map[mask]

    return score_map.astype(np.float32)


def apply_focus_overlay(
    image: np.ndarray,
    score_map: np.ndarray,
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """
    Normalise the score map, apply a colormap, and blend it with the original image.

    Returns a BGR uint8 composite image.
    """
    smoothed = score_map
    if smooth_sigma > 0:
        # Blur in float space before normalising so the gradient is truly continuous
        ksize = int(smooth_sigma * 6) | 1  # nearest odd number >= 6*sigma
        smoothed = cv2.GaussianBlur(score_map, (ksize, ksize), smooth_sigma)

    # Normalise to 0-255
    min_val, max_val = smoothed.min(), smoothed.max()
    if not math.isclose(float(min_val), float(max_val)):
        norm = ((smoothed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(smoothed, dtype=np.uint8)

    heatmap = cv2.applyColorMap(norm, colormap)

    if len(image.shape) == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    overlay = cv2.addWeighted(base, 1.0 - alpha, heatmap, alpha, 0)
    return overlay


def add_colorbar(
    image: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    bar_width: int = 40,
    label_low: str = "Soft",
    label_high: str = "Sharp",
    side: str = "right",
) -> np.ndarray:
    """Append a vertical colorbar legend on the right side of the image."""
    h = image.shape[0]
    gradient = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    bar = cv2.applyColorMap(np.repeat(gradient, bar_width, axis=1), colormap)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    pad = 4

    cv2.putText(bar, label_high, (2, 14), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(bar, label_low, (2, h - pad), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return np.hstack([bar, image]) if side == "left" else np.hstack([image, bar])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a focus heatmap overlay on an image.")
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument("--output", default=None, help="Output path (default: <input>_focus_overlay.<ext>).")
    parser.add_argument("--kernel-size", type=int, default=7, help="Laplacian kernel size (odd, default 7).")
    parser.add_argument("--window-size", type=int, default=64, help="Sliding window size in pixels (default 64).")
    parser.add_argument("--stride", type=int, default=16, help="Stride of the sliding window (default 16).")
    parser.add_argument("--alpha", type=float, default=0.6, help="Heatmap blend alpha, 0=image only, 1=heatmap only (default 0.6).")
    parser.add_argument("--colorbar-side", default="right", choices=["left", "right"], help="Which side to place the colorbar (default: right).")
    parser.add_argument("--smooth-sigma", type=float, default=30.0, help="Gaussian sigma for smoothing the heatmap into a gradient (default 30, 0 to disable).")
    parser.add_argument(
        "--colormap",
        default="jet",
        choices=["jet", "hot", "inferno", "plasma", "viridis", "turbo"],
        help="Colormap name (default: jet).",
    )
    args = parser.parse_args()

    colormap_map: dict[str, int] = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo": cv2.COLORMAP_TURBO,
    }
    colormap = colormap_map[args.colormap]

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Window: {args.window_size}px, stride: {args.stride}px, kernel: {args.kernel_size}")

    print("Computing focus map...")
    score_map = generate_focus_map(
        image,
        kernel_size=args.kernel_size,
        window_size=args.window_size,
        stride=args.stride,
    )

    print(f"Score range: {score_map.min():.2f} - {score_map.max():.2f}")

    overlay = apply_focus_overlay(image, score_map, alpha=args.alpha, colormap=colormap, smooth_sigma=args.smooth_sigma)
    overlay = add_colorbar(overlay, colormap=colormap, side=args.colorbar_side)

    if args.output:
        out_path = args.output
    else:
        dot = args.image.rfind(".")
        if dot != -1:
            out_path = args.image[:dot] + "_focus_overlay" + args.image[dot:]
        else:
            out_path = args.image + "_focus_overlay.png"

    cv2.imwrite(out_path, overlay)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()