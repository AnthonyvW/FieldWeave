#!/usr/bin/env python3
"""
Image Stitching Script
Collects one JPEG image from each subfolder of the input directory,
orders them by Y position parsed from filenames (format: x..._y...),
then stitches sequential neighbor pairs into a single panorama named
after the root folder.
"""

from __future__ import annotations

import os
import re
import argparse

import cv2
import numpy as np


def parse_y_position(filename: str) -> int:
    match = re.search(r'_y(\d+)', filename)
    return int(match.group(1)) if match else 0


def collect_images(root_folder: str) -> list[tuple[cv2.Mat, str]]:
    valid_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG'}

    subfolders = [
        entry.path
        for entry in os.scandir(root_folder)
        if entry.is_dir()
    ]

    if not subfolders:
        raise ValueError(f"No subfolders found in {root_folder}")

    entries: list[tuple[int, str, str]] = []

    for subfolder in subfolders:
        jpeg_files = sorted([
            f for f in os.listdir(subfolder)
            if os.path.splitext(f)[1] in valid_extensions
        ])

        if not jpeg_files:
            print(f"  Warning: No JPEG found in {os.path.basename(subfolder)}, skipping...")
            continue

        if len(jpeg_files) > 1:
            print(f"  Warning: Multiple JPEGs in {os.path.basename(subfolder)}, using first: {jpeg_files[0]}")

        filename = jpeg_files[0]
        entries.append((parse_y_position(filename), subfolder, filename))

    entries.sort(key=lambda e: e[0])

    print(f"Found {len(entries)} image(s), ordered by Y position:")
    images = []
    for y_pos, subfolder, filename in entries:
        img_path = os.path.join(subfolder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: Could not load {img_path}, skipping...")
            continue
        label = f"{os.path.basename(subfolder)}/{filename}"
        print(f"  + y={y_pos:>12}  {label}")
        images.append((img, label))

    return images


def match_offset(a: cv2.Mat, b: cv2.Mat) -> tuple[float, int, int]:
    """Return (confidence, y_offset, x_offset) for where the top of `b` lands in `a`."""
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    h_b = gray_b.shape[0]
    template = gray_b[:h_b // 2, :]

    result = cv2.matchTemplate(gray_a, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    return float(max_val), int(max_loc[1]), int(max_loc[0])


def find_offset(a: cv2.Mat, b: cv2.Mat, detect_direction: bool = False) -> tuple[tuple[int, int] | None, str]:
    """Find where the top-left of `b` lands in `a`'s coordinate space.

    Returns (y_offset, x_offset) or (None, error). When detect_direction is
    True, also checks the reverse ordering and returns negative y to signal
    that the image sequence should be reversed.
    """
    conf_ab, y_ab, x_ab = match_offset(a, b)

    if detect_direction:
        conf_ba, y_ba, x_ba = match_offset(b, a)
        print(f"    Match confidence a→b: {conf_ab:.3f} (y={y_ab}px x={x_ab}px)  b→a: {conf_ba:.3f} (y={y_ba}px x={x_ba}px)")
        if conf_ab < 0.1 and conf_ba < 0.1:
            return None, f"Template match confidence too low (a→b: {conf_ab:.3f}, b→a: {conf_ba:.3f})"
        if y_ba > y_ab:
            return (-y_ba, x_ba), ""
    else:
        print(f"    Match confidence: {conf_ab:.3f} (y={y_ab}px x={x_ab}px)")
        if conf_ab < 0.1:
            return None, f"Template match confidence too low ({conf_ab:.3f})"

    return (y_ab, x_ab), ""


def place_image(canvas: cv2.Mat, image: cv2.Mat, y_offset: int, x_offset: int, prev_end_y: int) -> None:
    """Place image onto canvas at (y_offset, x_offset), seaming at the overlap midpoint."""
    h_i, w_i = image.shape[:2]
    seam_y = (y_offset + prev_end_y) // 2
    src_seam = seam_y - y_offset

    x_end = x_offset + w_i
    canvas_w = canvas.shape[1]

    # Clamp to canvas width in case the x offset pushes past the edge
    src_x_end = min(w_i, canvas_w - x_offset)

    if x_offset >= 0:
        canvas[seam_y:y_offset + h_i, x_offset:x_offset + src_x_end] = image[src_seam:, :src_x_end]
    else:
        # Negative x offset: image extends to the left of the canvas origin
        src_x_start = -x_offset
        canvas[seam_y:y_offset + h_i, :x_end] = image[src_seam:, src_x_start:src_x_start + x_end]


def stitch_images(images: list[tuple[cv2.Mat, str]]) -> cv2.Mat | None:
    print(f"\nStitching {len(images)} images sequentially by Y position...")

    offsets, error = find_offset(images[0][0], images[1][0], detect_direction=True)
    if offsets is None:
        print(f"  Failed to determine scan direction: {error}")
        return None

    if offsets[0] < 0:
        print(f"  Detected bottom-to-top scan order (y={offsets[0]}px), reversing image sequence")
        images = list(reversed(images))
        offsets, error = find_offset(images[0][0], images[1][0], detect_direction=True)
        if offsets is None:
            print(f"  Failed after reversal: {error}")
            return None

    y_off, x_off = offsets
    print(f"  Merging '{images[0][1]}' + '{images[1][1]}'")
    print(f"    Vertical offset: {y_off}px  Horizontal offset: {x_off}px")

    h_a, w_a = images[0][0].shape[:2]
    h_b, w_b = images[1][0].shape[:2]

    cum_x = x_off
    canvas_h = max(y_off + h_b, h_a)
    canvas_w = max(w_a, cum_x + w_b) if cum_x >= 0 else max(w_a, w_b - cum_x)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h_a, :w_a] = images[0][0]
    place_image(canvas, images[1][0], y_off, cum_x, h_a)

    cum_y = y_off
    prev_end_y = y_off + h_b

    for i in range(2, len(images)):
        img, label = images[i]
        prev_label = images[i - 1][1]
        print(f"  Merging '{prev_label}' + '{label}'...")

        offsets, error = find_offset(images[i - 1][0], img)
        if offsets is None:
            print(f"  Failed at step {i}: {error}")
            return None

        y_off, x_off = offsets
        print(f"    Vertical offset: {y_off}px  Horizontal offset: {x_off}px")

        cum_y += y_off
        cum_x += x_off
        h_i, w_i = img.shape[:2]
        new_h = max(cum_y + h_i, canvas.shape[0])
        new_w = max(cum_x + w_i, canvas.shape[1]) if cum_x >= 0 else max(canvas.shape[1], w_i - cum_x)

        if new_h > canvas.shape[0] or new_w > canvas.shape[1]:
            expanded = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            expanded[:canvas.shape[0], :canvas.shape[1]] = canvas
            canvas = expanded

        place_image(canvas, img, cum_y, cum_x, prev_end_y)
        prev_end_y = cum_y + h_i

    print("Stitching successful!")
    return canvas


def crop_black_borders(image: cv2.Mat) -> cv2.Mat:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y:y + h, x:x + w]

    return image


def _rotate_image(image: cv2.Mat, degrees: int) -> cv2.Mat:
    """Rotate image by 0, 90, 180, or 270 degrees clockwise."""
    if degrees == 0:
        return image
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"degrees must be 0, 90, 180, or 270; got {degrees}")


def _find_scalebar_rect(image: cv2.Mat) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) of the scale bar bounding box, or None if not found.

    Looks for the most elongated horizontal contour, which is assumed to be
    the scale bar. The image passed in should already be oriented so the bar
    is expected to run horizontally.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_h, img_w = image.shape[:2]
    best: tuple[int, int, int, int] | None = None
    best_score = 0.0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < img_w * 0.1 or h == 0:
            continue
        aspect = w / h
        if aspect < 4:
            continue
        score = float(w) * aspect
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    return best


def _text_mass_above_vs_below(image: cv2.Mat, bar_y: int, bar_h: int) -> tuple[float, float]:
    """Return (mass_above, mass_below) of text-like blobs relative to the bar.

    Small dark blobs consistent with printed digits are counted by total area
    on each side of the bar's vertical midpoint.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape

    bar_mid = bar_y + bar_h // 2

    mask = np.ones_like(gray, dtype=np.uint8)
    mask[max(0, bar_y - 4):bar_y + bar_h + 4, :] = 0

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, binary, mask=mask)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mass_above = 0.0
    mass_below = 0.0
    max_blob = img_h * img_w * 0.01

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 10 or area > max_blob:
            continue
        cy = y + h // 2
        if cy < bar_mid:
            mass_above += area
        else:
            mass_below += area

    return float(mass_above), float(mass_below)


def detect_orientation(image: cv2.Mat) -> int:
    """Return the clockwise rotation in degrees (0, 90, 180, 270) needed to
    orient the panorama with the scale bar horizontal and numbers above it.

    The aspect ratio constrains candidates to two 180°-apart options, then
    the text-above-bar heuristic selects between them.
    """
    h, w = image.shape[:2]
    candidates = [0, 180] if w >= h else [90, 270]

    print("  Detecting orientation...")

    best_rotation = candidates[0]
    best_ratio = -1.0

    for rot in candidates:
        rotated = _rotate_image(image, rot)
        bar = _find_scalebar_rect(rotated)
        if bar is None:
            print(f"    {rot}°: scale bar not found")
            continue
        bx, by, bw, bh = bar
        above, below = _text_mass_above_vs_below(rotated, by, bh)
        total = above + below
        ratio = above / total if total > 0 else 0.0
        print(f"    {rot}°: bar at y={by} h={bh} w={bw} | text above={above:.0f} below={below:.0f} ratio={ratio:.3f}")
        if ratio > best_ratio:
            best_ratio = ratio
            best_rotation = rot

    print(f"  Selected rotation: {best_rotation}° (text-above ratio={best_ratio:.3f})")
    return best_rotation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stitch one JPEG per subfolder into a single panorama named after the root folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python siftstitch.py /path/to/root
  python siftstitch.py /path/to/root --no-crop
  python siftstitch.py /path/to/root --no-rotate
        """,
    )
    parser.add_argument("input_folder", type=str,
                        help="Root folder containing subfolders, each with one JPEG image")
    parser.add_argument("--no-crop", action="store_true",
                        help="Skip automatic cropping of black borders")
    parser.add_argument("--no-rotate", action="store_true",
                        help="Skip automatic orientation correction")

    args = parser.parse_args()

    input_folder = os.path.abspath(args.input_folder)
    if not os.path.isdir(input_folder):
        print(f"Error: {input_folder} is not a valid directory")
        return 1

    try:
        images = collect_images(input_folder)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if len(images) < 2:
        print("Error: Need at least 2 images to stitch")
        return 1

    result = stitch_images(images)
    if result is None:
        print("\nTips:")
        print("  - Ensure images have sufficient overlap (30-50%)")
        print("  - Check that Y positions in filenames reflect correct scan order")
        return 1

    if not args.no_crop:
        print("Cropping black borders...")
        result = crop_black_borders(result)

    if not args.no_rotate:
        rotation = detect_orientation(result)
        if rotation != 0:
            print(f"Rotating panorama {rotation}° clockwise...")
            result = _rotate_image(result, rotation)
        else:
            print("Orientation correct, no rotation needed.")

    root_name = os.path.basename(input_folder)
    output_path = os.path.join(input_folder, f"{root_name}.jpg")
    cv2.imwrite(output_path, result)
    print(f"\nPanorama saved to: {output_path}")
    print(f"Output size: {result.shape[1]}x{result.shape[0]} pixels")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())