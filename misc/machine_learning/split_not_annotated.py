from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".avif"}


def split_images(source: Path, dest: Path, batch_size: int, copy: bool) -> None:
    images = sorted(
        f for f in source.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"No images found in '{source}'.")
        sys.exit(1)

    total = len(images)
    num_batches = (total + batch_size - 1) // batch_size
    pad = len(str(num_batches))

    print(f"Found {total} image(s). Splitting into {num_batches} folder(s) of up to {batch_size} each.")

    dest.mkdir(parents=True, exist_ok=True)
    action = shutil.copy2 if copy else shutil.move

    for batch_index in range(num_batches):
        batch = images[batch_index * batch_size : (batch_index + 1) * batch_size]
        folder_name = f"batch_{str(batch_index + 1).zfill(pad)}"
        folder_path = dest / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        for image in batch:
            action(str(image), str(folder_path / image.name))

        print(f"  {folder_name}/ <- {len(batch)} file(s)")

    verb = "Copied" if copy else "Moved"
    print(f"\n{verb} {total} image(s) into {num_batches} folder(s) under '{dest}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a folder of images into subfolders of N images each."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the folder containing images.",
    )
    parser.add_argument(
        "batch_size",
        type=int,
        help="Maximum number of images per subfolder.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination folder for the subfolders (default: same as source).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them (default: move).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        print(f"Error: '{source}' is not a directory.")
        sys.exit(1)

    if args.batch_size < 1:
        print("Error: batch_size must be at least 1.")
        sys.exit(1)

    dest = (args.dest or source).resolve()

    split_images(source, dest, args.batch_size, args.copy)


if __name__ == "__main__":
    main()