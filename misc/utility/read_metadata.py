"""Read back metadata written by BaseCamera to TIFF, JPEG, and PNG images.

Usage:
    python read_image_metadata.py path/to/image.tiff
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

# Per-strip/tile file layout tags, not camera metadata; can run to
# thousands of entries for a single image and drown out everything else.
NOISY_TAGS = {"StripOffsets", "StripByteCounts", "TileOffsets", "TileByteCounts"}
MAX_ARRAY_LEN = 10


def truncate_for_display(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: truncate_for_display(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and len(value) > MAX_ARRAY_LEN:
        return f"<{len(value)} entries, first {MAX_ARRAY_LEN}: {list(value[:MAX_ARRAY_LEN])}>"
    return value


def decode_user_comment(raw: bytes) -> Any:
    # BaseCamera writes this as metadata_json.encode('utf-16'), with none of
    # the EXIF 2.3 charset-designation prefix a strict reader expects, so it
    # has to be decoded the same nonstandard way it was written.
    try:
        text = raw.decode("utf-16")
    except UnicodeDecodeError:
        return raw

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def read_exif_metadata(filepath: Path) -> dict[str, Any] | None:
    try:
        image = Image.open(filepath)
    except (OSError, ValueError):
        return None

    with image:
        exif = image.getexif()
        if not exif:
            return None

        base_tags = {tag.value: tag.name for tag in ExifTags.Base}
        result: dict[str, Any] = {}

        for tag_id, value in exif.items():
            name = base_tags.get(tag_id, tag_id)
            if name in NOISY_TAGS:
                continue
            if name == "UserComment" and isinstance(value, bytes):
                value = decode_user_comment(value)
            result[name] = value

        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        for tag_id, value in exif_ifd.items():
            name = base_tags.get(tag_id, tag_id)
            if name in NOISY_TAGS:
                continue
            if name == "UserComment" and isinstance(value, bytes):
                value = decode_user_comment(value)
            result[name] = value

        return result


def read_png_metadata(filepath: Path) -> dict[str, Any] | None:
    try:
        image = Image.open(filepath)
    except (OSError, ValueError):
        return None

    with image:
        if not image.info:
            return None

        result: dict[str, Any] = {}
        for key, value in image.info.items():
            if key == "Metadata" and isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            result[key] = value

        return result


def read_metadata(filepath: Path) -> dict[str, Any] | None:
    ext = filepath.suffix.lower()
    if ext in (".tif", ".tiff", ".jpg", ".jpeg"):
        return read_exif_metadata(filepath)
    if ext == ".png":
        return read_png_metadata(filepath)
    return None


def main() -> bool:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filepath", type=Path)
    args = parser.parse_args()

    if not args.filepath.exists():
        print(f"File not found: {args.filepath}")
        return False

    metadata = read_metadata(args.filepath)
    if metadata is None:
        print(f"No readable metadata found in {args.filepath}")
        return False

    print(json.dumps(truncate_for_display(metadata), indent=2, default=str))
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)