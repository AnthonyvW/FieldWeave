from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}


def rename_tiles(source_dir: str | Path, output_subdir: str = "tiles") -> None:
    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")

    images = sorted(
        p for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"No images found in {source}")
        return

    dest = source / output_subdir
    dest.mkdir(exist_ok=True)

    pad = len(str(len(images) - 1))

    for i, img in enumerate(images):
        dest_path = dest / f"tile_{str(i).zfill(pad)}.jpg"
        shutil.copy2(img, dest_path)
        print(f"  {img.name} -> {dest_path.name}")

    print(f"\nCopied {len(images)} images to {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy images into a child folder named tile_{i}.jpg, sorted by filename."
    )
    parser.add_argument("source_dir", help="Path to the folder containing images")
    parser.add_argument(
        "--subdir",
        default="tiles",
        help="Name of the child folder to create (default: tiles)",
    )
    args = parser.parse_args()

    rename_tiles(args.source_dir, args.subdir)