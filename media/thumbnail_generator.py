from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"}


def create_thumbnail(
    input_path: Path,
    output_path: Path | None = None,
    max_width: int = 320,
    quality: int = 60,
) -> Path:
    """Create a compressed, low-resolution thumbnail from an image.

    Args:
        input_path: Path to the source image.
        output_path: Destination path. Defaults to <stem>_thumb.jpg next to the source.
        max_width: Maximum width in pixels; height scales proportionally.
        quality: JPEG compression quality (1-95, lower = smaller file).

    Returns:
        The path where the thumbnail was saved.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_thumb.jpg")
    output_path = Path(output_path)

    with Image.open(input_path) as img:
        # Convert to RGB so we can always save as JPEG (handles RGBA/P modes)
        if img.mode != "RGB":
            img = img.convert("RGB")

        orig_width, orig_height = img.size
        if orig_width > max_width:
            scale = max_width / orig_width
            new_size = (max_width, max(1, round(orig_height * scale)))
        else:
            new_size = (orig_width, orig_height)

        thumb = img.resize(new_size, Image.LANCZOS)
        thumb.save(output_path, format="JPEG", quality=quality, optimize=True)

    print(f"  OK  {input_path.name} -> {output_path.name} ({new_size[0]}x{new_size[1]})")
    return output_path


def process_folder(
    input_folder: Path,
    output_folder: Path | None = None,
    max_width: int = 320,
    quality: int = 60,
) -> tuple[int, int]:
    """Convert all images in a folder to thumbnails.

    Args:
        input_folder: Directory containing source images.
        output_folder: Directory for thumbnails. Defaults to <input_folder>/thumbnails.
        max_width: Maximum width in pixels; height scales proportionally.
        quality: JPEG compression quality (1-95).

    Returns:
        A (succeeded, failed) count tuple.
    """
    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_folder}")

    if output_folder is None:
        output_folder = input_folder / "thumbnails"
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    candidates = [
        p for p in sorted(input_folder.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not candidates:
        print(f"No supported image files found in {input_folder}")
        return 0, 0

    print(f"Processing {len(candidates)} image(s) from {input_folder}")
    print(f"Output folder: {output_folder}\n")

    succeeded = 0
    failed = 0

    for image_path in candidates:
        output_path = output_folder / f"{image_path.stem}_thumb.jpg"
        try:
            create_thumbnail(image_path, output_path, max_width, quality)
            succeeded += 1
        except (UnidentifiedImageError, OSError) as exc:
            print(f"  FAIL {image_path.name}: {exc}")
            failed += 1

    print(f"\nDone. {succeeded} succeeded, {failed} failed.")
    return succeeded, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate compressed thumbnails from an image or a folder of images."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a source image or a folder of images.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help=(
            "Output path. For a single image: destination file "
            "(default: <stem>_thumb.jpg alongside input). "
            "For a folder: destination directory "
            "(default: <input_folder>/thumbnails/)."
        ),
    )
    parser.add_argument(
        "-w", "--max-width",
        type=int,
        default=320,
        help="Maximum width in pixels (default: 320). Height scales proportionally.",
    )
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=60,
        help="JPEG quality 1-95 (default: 60). Lower = smaller file.",
    )
    args = parser.parse_args()

    if not 1 <= args.quality <= 95:
        parser.error("--quality must be between 1 and 95.")

    if args.input.is_dir():
        process_folder(
            input_folder=args.input,
            output_folder=args.output,
            max_width=args.max_width,
            quality=args.quality,
        )
    else:
        create_thumbnail(
            input_path=args.input,
            output_path=args.output,
            max_width=args.max_width,
            quality=args.quality,
        )


if __name__ == "__main__":
    main()