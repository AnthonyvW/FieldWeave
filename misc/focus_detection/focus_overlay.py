"""
focus_overlay.py - Generate a focus heatmap overlay on an image or folder of images.

Uses the Tenengrad focus measure based on squared Sobel gradient magnitude.

Single image:
    python focus_overlay.py <image_path> [--output <output_path>]

Folder (produces a GIF):
    python focus_overlay.py <folder_path> [--output <output_path>] [--gif-duration 5.0]
                            [--side-by-side]

Common options:
    --kernel-size 3 --radius 8 --threshold 0 --alpha 0.6
    --half-resolution --box-blur --colormap jet
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from focus_detection import (
    FocusScores,
    apply_focus_overlay,
    add_colorbar,
    build_frame,
    generate_focus_map,
    normalize_score_map,
)


IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)
ANIMATION_EXTENSIONS: frozenset[str] = frozenset(
    {".webp", ".gif"}
)


def make_animation(
    frames_bgr: list[np.ndarray],
    out_path: str,
    duration_s: float,
    fmt: str = "webp",
    scale: float = 1.0,
    gif_colors: int = 256,
) -> None:
    """
    Encode a list of BGR uint8 frames into an animated WebP or GIF.

    Requires Pillow. Frame duration is distributed evenly across all frames.

    Args:
        frames_bgr: BGR uint8 frames from OpenCV.
        out_path: Output file path. Extension should match fmt.
        duration_s: Total animation duration in seconds.
        fmt: "webp" (default, smaller, full colour) or "gif" (256-colour palette,
            wider compatibility).
        scale: Resize factor applied before encoding (default 1.0 = full size).
            0.5 halves each dimension, reducing file size by ~4x.
        gif_colors: Number of palette colours for GIF output (2-256, default 256).
            Lower values reduce file size at the cost of colour accuracy.
    """
    try:
        from PIL import Image as PilImage
    except ImportError:
        raise ImportError(
            "Pillow is required for animation output. Install it with: pip install Pillow"
        )

    frame_duration_ms = int(duration_s * 1000 / len(frames_bgr))

    # Determine the canonical frame size — use the most common (h, w) among all
    # frames so a single outlier doesn't force everything else to resize.
    # WebP requires all frames to be identical in size; GIF has the same constraint
    # in practice with Pillow's encoder.
    from collections import Counter
    size_counts = Counter((f.shape[0], f.shape[1]) for f in frames_bgr)
    canonical_h, canonical_w = size_counts.most_common(1)[0][0]
    if scale != 1.0:
        canonical_w = max(1, int(canonical_w * scale))
        canonical_h = max(1, int(canonical_h * scale))

    n_resized = sum(1 for f in frames_bgr if (f.shape[0], f.shape[1]) != size_counts.most_common(1)[0][0])
    if n_resized:
        print(f"  Note: {n_resized} frame(s) had non-standard dimensions and will be resized to {canonical_w}x{canonical_h}")

    pil_frames: list[PilImage.Image] = []
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Apply scale and/or conform to canonical size in one resize if needed
        target_w = max(1, int(rgb.shape[1] * scale)) if scale != 1.0 else rgb.shape[1]
        target_h = max(1, int(rgb.shape[0] * scale)) if scale != 1.0 else rgb.shape[0]
        if target_w != canonical_w or target_h != canonical_h:
            target_w, target_h = canonical_w, canonical_h
        if target_w != rgb.shape[1] or target_h != rgb.shape[0]:
            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        pil_frame = PilImage.fromarray(rgb)
        if fmt == "gif":
            # GIF is limited to 256 colours — build a global palette from all
            # frames so the LZW compressor can exploit inter-frame similarity
            pil_frame = pil_frame.quantize(
                colors=gif_colors,
                method=PilImage.Quantize.MEDIANCUT,
                dither=PilImage.Dither.NONE,
            )
        pil_frames.append(pil_frame)

    save_kwargs: dict = {
        "save_all": True,
        "append_images": pil_frames[1:],
        "duration": frame_duration_ms,
        "loop": 0,
    }

    if fmt == "webp":
        # lossless=False with quality=80 gives excellent visual quality at a
        # fraction of lossless size; method=4 balances encode speed vs compression
        save_kwargs.update({"lossless": False, "quality": 80, "method": 4})
    elif fmt == "gif":
        save_kwargs["optimize"] = True

    pil_frames[0].save(out_path, **save_kwargs)


# Encode MP4 at this fixed frame rate regardless of how many source frames there
# are. Each source frame is repeated as needed to fill its share of the duration.
# At low fps (e.g. 2fps for a 10-frame/5s stack) H.264 treats every frame as a
# keyframe with no inter-frame compression, producing huge files and stuttery
# playback. At 24fps with repeated frames the inter-frame delta is zero and
# compresses to near-nothing, giving smooth playback and small files.
_MP4_FPS: int = 24


def make_video(
    frames_bgr: list[np.ndarray],
    out_path: str,
    duration_s: float,
    fmt: str = "mp4",
    scale: float = 1.0,
    crf: int = 28,
) -> None:
    """
    Encode a list of BGR uint8 frames into an MP4 (H.264) or WebM (VP9) video
    using imageio-ffmpeg.

    Encodes at a fixed 24fps, repeating each source frame for its proportional
    share of the total duration. This gives smooth playback and lets inter-frame
    compression reduce repeated frames to near-zero bytes, keeping file sizes
    small even for large high-resolution images.

    imageio-ffmpeg bundles its own FFmpeg binary, avoiding the OpenCV/H.264
    licensing conflict and DLL issues on Windows.

    Requires: pip install imageio imageio-ffmpeg

    Args:
        frames_bgr: BGR uint8 frames from OpenCV.
        out_path: Output file path (.mp4 or .webm).
        duration_s: Total video duration in seconds.
        fmt: "mp4" (H.264, broad compatibility) or "webm" (VP9, Google Slides).
        scale: Resize factor applied before encoding (default 1.0 = full size).
        crf: Constant rate factor controlling quality vs file size.
            H.264 range 0-51 (default 28). VP9 range 0-63 (default 28).
            Lower values give higher quality and larger files.
    """
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio and imageio-ffmpeg are required for video output. "
            "Install with: pip install imageio imageio-ffmpeg"
        )

    from collections import Counter
    size_counts = Counter((f.shape[0], f.shape[1]) for f in frames_bgr)
    canonical_h, canonical_w = size_counts.most_common(1)[0][0]
    if scale != 1.0:
        canonical_w = max(2, int(canonical_w * scale))
        canonical_h = max(2, int(canonical_h * scale))
    # Both H.264 and VP9 require dimensions divisible by 2
    canonical_w += canonical_w % 2
    canonical_h += canonical_h % 2

    n_resized = sum(1 for f in frames_bgr if (f.shape[0], f.shape[1]) != size_counts.most_common(1)[0][0])
    if n_resized:
        print(f"  Note: {n_resized} frame(s) had non-standard dimensions and will be resized to {canonical_w}x{canonical_h}")

    # Compute how many video frames each source frame occupies at _MP4_FPS.
    # We distribute the total frame budget across source frames as evenly as
    # possible using integer rounding, so the total always sums to exactly
    # round(duration_s * _MP4_FPS) video frames.
    total_video_frames = round(duration_s * _MP4_FPS)
    n = len(frames_bgr)
    repeat_counts = [
        round((i + 1) * total_video_frames / n) - round(i * total_video_frames / n)
        for i in range(n)
    ]
    print(f"  {_MP4_FPS}fps, {total_video_frames} total video frames ({repeat_counts[0]} repeats per source frame)")

    if fmt == "webm":
        codec = "libvpx-vp9"
        pixelformat = "yuv420p"
        # VP9 uses -b:v 0 alongside -crf to enable constant-quality mode
        extra_params = ["-b:v", "0", "-crf", str(crf)]
    else:
        codec = "libx264"
        pixelformat = "yuv420p"
        extra_params = ["-crf", str(crf), "-preset", "slow"]

    with imageio.get_writer(
        out_path,
        format="ffmpeg",
        mode="I",
        fps=_MP4_FPS,
        codec=codec,
        pixelformat=pixelformat,
        output_params=extra_params,
    ) as writer:
        for bgr, repeat in zip(frames_bgr, repeat_counts):
            if bgr.shape[1] != canonical_w or bgr.shape[0] != canonical_h:
                bgr = cv2.resize(bgr, (canonical_w, canonical_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for _ in range(repeat):
                writer.append_data(rgb)


def make_mp4(
    frames_bgr: list[np.ndarray],
    out_path: str,
    duration_s: float,
    scale: float = 1.0,
    crf: int = 28,
) -> None:
    """Convenience wrapper around make_video for MP4 output."""
    make_video(frames_bgr, out_path, duration_s, fmt="mp4", scale=scale, crf=crf)


def process_single(
    image_path: str,
    args: argparse.Namespace,
    colormap: int,
) -> None:
    """Process a single image file and write the overlay output."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    flags = []
    if args.half_resolution:
        flags.append("half-res")
    if args.box_blur:
        flags.append("box-blur")
    flags_str = f" [{', '.join(flags)}]" if flags else ""

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Sobel kernel: {args.kernel_size}, radius: {args.radius}, threshold: {args.threshold}{flags_str}")

    t_total_start = time.perf_counter()

    print("Computing focus map...")
    t_start = time.perf_counter()
    frame, scores = build_frame(
        image,
        colormap=colormap,
        alpha=args.alpha,
        colorbar_side=args.colorbar_side,
        kernel_size=args.kernel_size,
        radius=args.radius,
        threshold=args.threshold,
        half_resolution=args.half_resolution,
        box_blur=args.box_blur,
        side_by_side=args.side_by_side,
        verbose=True,
    )
    t_focus = time.perf_counter() - t_start
    print(f"  Done in {t_focus:.2f}s")

    print()
    print("Focus scores:")
    print(f"  Whole image: {scores.whole:.4f}")
    print(f"  Center:      {scores.center:.4f}")
    print(f"  Peak:        {scores.peak:.4f}")
    print()

    if args.output:
        out_path = args.output
    else:
        dot = image_path.rfind(".")
        suffix = "_focus_overlay_side_by_side" if args.side_by_side else "_focus_overlay"
        out_path = (image_path[:dot] + suffix + image_path[dot:]) if dot != -1 else image_path + suffix + ".png"

    print("Writing output...")
    t_start = time.perf_counter()
    cv2.imwrite(out_path, frame)
    t_write = time.perf_counter() - t_start

    t_total = time.perf_counter() - t_total_start
    print(f"Saved: {out_path}")
    print()
    print("Timing summary:")
    print(f"  Focus map:   {t_focus:.2f}s")
    print(f"  Write:       {t_write:.3f}s")
    print(f"  Total:       {t_total:.2f}s")


def process_folder(
    folder_path: str,
    args: argparse.Namespace,
    colormap: int,
) -> None:
    """Process all images in a folder alphabetically and write an animated WebP or GIF."""
    paths = sorted(
        p for p in Path(folder_path).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.suffix.lower() not in ANIMATION_EXTENSIONS
    )

    if not paths:
        raise ValueError(f"No supported image files found in: {folder_path}")

    ext = {"gif": "gif", "webp": "webp", "mp4": "mp4", "webm": "webm"}[args.format]

    print(f"Found {len(paths)} images in {folder_path}")
    print(f"Output format: {args.format.upper()}, duration: {args.gif_duration}s ({args.gif_duration / len(paths) * 1000:.0f}ms per frame)")
    if args.scale != 1.0:
        print(f"Output scale: {args.scale}x")
    if args.format == "gif" and args.gif_colors < 256:
        print(f"GIF palette: {args.gif_colors} colours")
    print()

    # Two-pass approach for global normalisation:
    # Pass 1 — compute raw (unnormalised) score maps for all frames
    # Pass 2 — derive a global percentile ceiling, normalise all maps consistently,
    #           then render frames so brightness is comparable across the animation
    images_and_raw: list[tuple[np.ndarray, Path, np.ndarray]] = []

    t_total_start = time.perf_counter()

    print("Pass 1/2 — computing focus maps...")
    for i, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            print(f"  [{i + 1}/{len(paths)}] Skipping (could not read): {path.name}")
            continue

        print(f"  [{i + 1}/{len(paths)}] {path.name} ({image.shape[1]}x{image.shape[0]})")
        raw_map = generate_focus_map(
            image,
            kernel_size=args.kernel_size,
            radius=args.radius,
            threshold=args.threshold,
            half_resolution=args.half_resolution,
            box_blur=args.box_blur,
            verbose=False,
            normalize=False,
        )
        images_and_raw.append((image, path, raw_map))

    if not images_and_raw:
        raise ValueError("No images could be processed.")

    # Compute a global ceiling from the distribution of per-frame maxima.
    # Using the 95th percentile of maxima rather than the absolute maximum
    # prevents one unusually sharp frame from compressing all others toward zero.
    per_frame_maxima = np.array([raw.max() for _, _, raw in images_and_raw])
    global_ceiling = float(np.percentile(per_frame_maxima, 95))
    print()
    print(f"Global normalisation ceiling (95th pct of frame maxima): {global_ceiling:.2f}")
    print(f"  Per-frame max range: {per_frame_maxima.min():.2f} – {per_frame_maxima.max():.2f}")
    print()

    frames: list[np.ndarray] = []
    all_scores: list[tuple[str, FocusScores]] = []

    # For export-max: accumulate the per-pixel maximum raw score across all frames
    # that share the most common resolution. Frames with different sizes are skipped
    # for the max map (consistent with how make_animation handles size mismatches).
    from collections import Counter as _Counter
    size_counts = _Counter((img.shape[0], img.shape[1]) for img, _, _ in images_and_raw)
    canonical_h, canonical_w = size_counts.most_common(1)[0][0]
    max_raw_map: np.ndarray | None = None

    print("Pass 2/2 — rendering frames...")
    for i, (image, path, raw_map) in enumerate(images_and_raw):
        print(f"  [{i + 1}/{len(images_and_raw)}] {path.name}")
        norm_map = normalize_score_map(raw_map, ceiling=global_ceiling)
        frame, scores = build_frame(
            image,
            colormap=colormap,
            alpha=args.alpha,
            colorbar_side=args.colorbar_side,
            kernel_size=args.kernel_size,
            radius=args.radius,
            threshold=args.threshold,
            half_resolution=args.half_resolution,
            box_blur=args.box_blur,
            side_by_side=args.side_by_side,
            verbose=False,
            score_map=norm_map,
        )
        print(f"    Scores — whole: {scores.whole:.4f}  center: {scores.center:.4f}  peak: {scores.peak:.4f}")

        frames.append(frame)
        all_scores.append((path.name, scores))

        if args.export_max and image.shape[0] == canonical_h and image.shape[1] == canonical_w:
            max_raw_map = raw_map if max_raw_map is None else np.maximum(max_raw_map, raw_map)

    if args.output:
        out_path = args.output
    else:
        side_suffix = "_side_by_side" if args.side_by_side else ""
        out_path = str(Path(folder_path) / f"{Path(folder_path).name}_focus_overlay{side_suffix}.{ext}")

    print()
    print(f"Encoding {args.format.upper()} with {len(frames)} frames...")
    t_start = time.perf_counter()
    if args.format in ("mp4", "webm"):
        make_video(frames, out_path, duration_s=args.gif_duration, fmt=args.format, scale=args.scale, crf=args.mp4_crf)
    else:
        make_animation(
            frames,
            out_path,
            duration_s=args.gif_duration,
            fmt=args.format,
            scale=args.scale,
            gif_colors=args.gif_colors,
        )
    t_encode = time.perf_counter() - t_start

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    t_total = time.perf_counter() - t_total_start
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")

    if args.export_max and max_raw_map is not None:
        # Normalise the max map with the same global ceiling for brightness consistency
        max_norm_map = normalize_score_map(max_raw_map, ceiling=global_ceiling)
        # Use the first canonical-sized image as the base for the overlay
        base_image = next(
            img for img, _, _ in images_and_raw
            if img.shape[0] == canonical_h and img.shape[1] == canonical_w
        )
        max_overlay = apply_focus_overlay(base_image, max_norm_map, alpha=args.alpha, colormap=colormap)
        max_overlay = add_colorbar(max_overlay, colormap=colormap, side=args.colorbar_side)
        max_out_path = str(Path(folder_path) / f"{Path(folder_path).name}_focus_max.jpg")
        cv2.imwrite(max_out_path, max_overlay)
        print(f"Saved max overlay: {max_out_path}")

    print()
    print("Timing summary:")
    print(f"  Processing:  {t_total - t_encode:.2f}s")
    print(f"  Encode:      {t_encode:.2f}s")
    print(f"  Total:       {t_total:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a focus heatmap overlay on an image or folder of images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to an image file or a folder of images.")
    parser.add_argument("--output", default=None, help="Output path. For folders defaults to <folder>/<folder>_focus_overlay.<ext> where ext matches --format.")
    parser.add_argument("--kernel-size", type=int, default=3, help="Sobel kernel size (1, 3, 5, or 7 — default 3).")
    parser.add_argument("--radius", type=float, default=8.0, help="Blur radius for spreading focus signal (default 8, 0 to disable).")
    parser.add_argument("--threshold", type=float, default=0.0, help="Suppress gradient magnitudes below this value before blurring (default 0 = disabled).")
    parser.add_argument("--half-resolution", action="store_true", help="Process at half resolution for ~4x speedup, then upscale result.")
    parser.add_argument("--box-blur", action="store_true", help="Use a box blur instead of Gaussian blur. Faster with visually similar results.")
    parser.add_argument("--side-by-side", action="store_true", help="Show original image and focus overlay side by side.")
    parser.add_argument("--export-max", action="store_true", help="Also export a single image showing the per-pixel maximum focus score across all frames. Saved as <folder>_focus_max.jpg alongside the animation.")
    parser.add_argument("--gif-duration", type=float, default=5.0, help="Total animation duration in seconds when input is a folder (default 5.0).")
    parser.add_argument("--format", default="webp", choices=["webp", "gif", "mp4", "webm"], help="Animation format for folder input: webp (default), gif (256-colour palette), mp4 (H.264), or webm (VP9 — recommended for Google Slides).")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor applied before encoding the animation (default 1.0). 0.5 halves each dimension, reducing file size ~4x.")
    parser.add_argument("--gif-colors", type=int, default=256, help="Number of palette colours for GIF output (2-256, default 256). Lower values reduce file size.")
    parser.add_argument("--mp4-crf", type=int, default=28, help="Constant rate factor for MP4/WebM output. H.264 range 0-51, VP9 range 0-63 (default 28). Lower values give higher quality and larger files.")
    parser.add_argument("--alpha", type=float, default=0.6, help="Heatmap blend alpha, 0=image only, 1=heatmap only (default 0.6).")
    parser.add_argument("--colorbar-side", default="right", choices=["left", "right"], help="Which side to place the colorbar (default: right).")
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

    if os.path.isdir(args.input):
        process_folder(args.input, args, colormap)
    else:
        process_single(args.input, args, colormap)


if __name__ == "__main__":
    main()