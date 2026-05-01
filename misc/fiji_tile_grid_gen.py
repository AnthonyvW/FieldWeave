from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

_TILE_RE = re.compile(r"x(\d+)_y(\d+)", re.IGNORECASE)


def tile_sort_key(path: Path) -> tuple[int, int]:
    m = _TILE_RE.search(path.stem)
    if m:
        # Sort by y descending (highest y = bottom of sample comes last),
        # then x ascending (left-to-right within each row)
        return -int(m.group(2)), int(m.group(1))
    return (0, 0)


def rename_tiles(source_dir: str | Path, output_subdir: str = "tiles") -> None:
    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")

    images = sorted(
        (p for p in source.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=tile_sort_key,
    )

    if not images:
        print(f"No images found in {source}")
        return

    dest = source / output_subdir
    dest.mkdir(exist_ok=True)

    pad = len(str(len(images) - 1))

    xs: set[int] = set()
    ys: set[int] = set()
    for img in images:
        m = _TILE_RE.search(img.stem)
        if m:
            xs.add(int(m.group(1)))
            ys.add(int(m.group(2)))

    for i, img in enumerate(images):
        dest_path = dest / f"tile_{str(i+1).zfill(pad)}.jpg"
        shutil.copy2(img, dest_path)
        print(f"  {img.name} -> {dest_path.name}")

    grid_info = ""
    if xs and ys:
        grid_cols = len(xs)
        grid_rows = len(ys)
        grid_info = f"  Grid dimensions: {grid_cols} columns x {grid_rows} rows ({grid_cols * grid_rows} tiles)"

    print(f"\nCopied {len(images)} images to {dest}")
    if grid_info:
        print(grid_info)


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