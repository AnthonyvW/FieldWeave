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
from PIL.PngImagePlugin import PngInfo

# EXIF 2.3 UserComment charset-designation codes (first 8 bytes of the tag).
_USER_COMMENT_CHARSETS = {
    b"UNICODE\x00": "utf-16-le",
    b"ASCII\x00\x00\x00": "ascii",
}

# Per-strip/tile file layout tags, not camera metadata; can run to
# thousands of entries for a single image and drown out everything else.
NOISY_TAGS = {"StripOffsets", "StripByteCounts", "TileOffsets", "TileByteCounts"}
MAX_ARRAY_LEN = 10

# Camera settings with no standard EXIF tag are written by BaseCamera to this
# private, unassigned tag block, one JSON-encoded value per field. Must stay
# in sync with _PRIVATE_CAMERA_FIELDS in camera/cameras/base_camera.py.
_PRIVATE_CAMERA_FIELDS = (
    "preview_resolution",
    "still_resolution",
    "file_format",
    "codec",
    "rotate",
    "hflip",
    "vflip",
    "exposure",
    "contrast",
    "saturation",
    "hue",
    "temp",
    "tint",
    "level_range_low",
    "level_range_high",
    "dfc_enable",
    "dfc_quantity",
)
_PRIVATE_TAG_BASE = 60100
_PRIVATE_TAG_NAMES: dict[int, str] = {
    _PRIVATE_TAG_BASE + i: name for i, name in enumerate(_PRIVATE_CAMERA_FIELDS)
}


def truncate_for_display(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: truncate_for_display(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and len(value) > MAX_ARRAY_LEN:
        return f"<{len(value)} entries, first {MAX_ARRAY_LEN}: {list(value[:MAX_ARRAY_LEN])}>"
    return value


def decode_user_comment(raw: bytes) -> Any:
    text: str | None = None

    for prefix, encoding in _USER_COMMENT_CHARSETS.items():
        if raw.startswith(prefix):
            try:
                text = raw[len(prefix):].decode(encoding)
            except UnicodeDecodeError:
                text = None
            break

    if text is None:
        # Fall back to how BaseCamera wrote this before it added the
        # charset-designation prefix (plain metadata_json.encode('utf-16')).
        try:
            text = raw.decode("utf-16")
        except UnicodeDecodeError:
            return raw

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _resolve_tag(tag_id: int, value: Any, base_tags: dict[int, str]) -> tuple[Any, Any]:
    """Map a raw (tag_id, value) pair to (name, value), decoding private tags."""
    if tag_id in _PRIVATE_TAG_NAMES:
        name = _PRIVATE_TAG_NAMES[tag_id]
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        return name, value

    name = base_tags.get(tag_id, tag_id)
    if name == "UserComment" and isinstance(value, bytes):
        value = decode_user_comment(value)
    return name, value


# Tags Pillow derives from the array actually being saved (dimensions, bit
# depth, layout). Re-applying stored values for these could conflict with
# the real output image, so they're excluded when building output EXIF.
STRUCTURAL_TAGS = {
    "ImageWidth",
    "ImageLength",
    "BitsPerSample",
    "Compression",
    "PhotometricInterpretation",
    "SamplesPerPixel",
    "RowsPerStrip",
    "PlanarConfiguration",
}

# These tags store the byte offset of a nested IFD within the *source*
# file, not real data. Carrying an offset over verbatim breaks
# Exif.tobytes(): it tries to dereference the offset via self.fp, which
# doesn't exist on a freshly built Exif object.
IFD_POINTER_TAGS = {"ExifOffset", "GPSInfo", "ExifInteroperabilityOffset", "SubIFDs"}


def extract_dpi(metadata: dict[str, Any]) -> float | None:
    """Pull the DPI value out of a metadata dict.

    TIFF/JPEG store it as XResolution/YResolution; PNG surfaces it directly
    as a 'dpi' pair (from the pHYs chunk). FieldWeave always writes the same
    value to X and Y, so either side of the pair is authoritative.
    """
    dpi_pair = metadata.get("dpi")
    if dpi_pair is not None:
        try:
            return float(dpi_pair[0])
        except (TypeError, ValueError, IndexError):
            return None

    value = metadata.get("XResolution", metadata.get("YResolution"))
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_exif_bytes(metadata: dict[str, Any]) -> bytes | None:
    """Encode a dict produced by read_metadata() back into EXIF bytes for Image.save(exif=...)."""
    exif = Image.Exif()
    base_tag_ids = {tag.name: tag.value for tag in ExifTags.Base}
    private_tag_ids = {name: tag_id for tag_id, name in _PRIVATE_TAG_NAMES.items()}

    for name, value in metadata.items():
        if name in STRUCTURAL_TAGS or name in NOISY_TAGS or name in IFD_POINTER_TAGS:
            continue
        if name in private_tag_ids:
            exif[private_tag_ids[name]] = value if isinstance(value, str) else json.dumps(value)
        elif name == "UserComment":
            payload = value if isinstance(value, str) else json.dumps(value)
            exif[base_tag_ids["UserComment"]] = b"UNICODE\x00" + payload.encode("utf-16-le")
        elif name in base_tag_ids:
            exif[base_tag_ids[name]] = value

    try:
        return exif.tobytes()
    except (ValueError, TypeError):
        return None


def build_png_info(metadata: dict[str, Any]) -> PngInfo:
    """Encode a dict produced by read_metadata() back into PNG text chunks for Image.save(pnginfo=...).

    PNG metadata is flat (dotted keys like "Camera.model", string values)
    rather than tag-numbered like EXIF, so no tag lookup is needed here.
    'dpi' is excluded since that's carried via the save() dpi= kwarg instead.
    """
    info = PngInfo()
    for name, value in metadata.items():
        if name == "dpi":
            continue
        info.add_text(name, value if isinstance(value, str) else json.dumps(value))

    return info


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
            name, value = _resolve_tag(tag_id, value, base_tags)
            if name in NOISY_TAGS:
                continue
            result[name] = value

        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        for tag_id, value in exif_ifd.items():
            name, value = _resolve_tag(tag_id, value, base_tags)
            if name in NOISY_TAGS:
                continue
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