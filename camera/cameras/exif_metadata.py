"""
TIFF/JPEG/PNG metadata writing for BaseCamera.save_image().

Distributes camera settings across the closest standard EXIF tag where one 
exists, our own private tag block (_PRIVATE_CAMERA_TAGS) for
camera settings with no standard equivalent, and JSON in UserComment as a
last resort for anything neither covers. PNG has no EXIF IFD, so it gets one
text chunk per field instead.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any
import json

from PIL import Image, ExifTags
from PIL.Image import Exif
from PIL import PngImagePlugin
from PIL.TiffImagePlugin import IFDRational

from common.logger import debug

# EXIF 2.3 requires an 8-byte charset-designation code ahead of the actual
# UserComment text; without it, strict readers like Windows Explorer and
# GIMP reject or garble the tag.
_EXIF_USER_COMMENT_UNICODE_PREFIX = b"UNICODE\x00"

# Computed once; ExifTags.Base doesn't change at runtime.
_BASE_EXIF_TAGS: dict[str, int] = {tag.name: tag.value for tag in ExifTags.Base}


# TIFF/EXIF has no standard tag for these camera-specific settings, so each
# gets its own tag in this unassigned private block instead of being folded
# into UserComment. IDs are fixed once written; append new names to the end
# rather than reordering, or older files' tags will be misread.
# Keep in sync with the mirror copy of this list in read_metadata.py.
_PRIVATE_CAMERA_FIELDS: tuple[str, ...] = (
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
_PRIVATE_CAMERA_TAGS: dict[str, int] = {
    name: _PRIVATE_TAG_BASE + i for i, name in enumerate(_PRIVATE_CAMERA_FIELDS)
}


def _encode_exif_user_comment(text: str) -> bytes:
    return _EXIF_USER_COMMENT_UNICODE_PREFIX + text.encode("utf-16-le")


def _software_string() -> str:
    # Deferred import to avoid a circular import with app_context.
    from common.app_context import get_app_context
    return f"FieldWeave - v{get_app_context().settings.version}"


def _apply_standard_exposure_tags(ifd: Any, camera_meta: dict[str, Any]) -> set[str]:
    """Write camera settings that map onto real EXIF tags. Returns the keys consumed."""
    consumed: set[str] = set()

    if "exposure_time_us" in camera_meta:
        ifd[_BASE_EXIF_TAGS["ExposureTime"]] = (int(camera_meta["exposure_time_us"]), 1_000_000)
        consumed.add("exposure_time_us")

    if "gain" in camera_meta:
        ifd[_BASE_EXIF_TAGS["ISOSpeedRatings"]] = int(camera_meta["gain"])
        consumed.add("gain")

    if "auto_exposure" in camera_meta:
        ifd[_BASE_EXIF_TAGS["ExposureMode"]] = 0 if camera_meta["auto_exposure"] else 1
        consumed.add("auto_exposure")

    return consumed


def _apply_extended_standard_tags(ifd: Any, camera_meta: dict[str, Any]) -> set[str]:
    """Write settings onto real but less common EXIF tags. BodySerialNumber is
    an exact match; BrightnessValue and Gamma are a pragmatic reuse of the
    closest standard numeric tag (same reasoning as ISOSpeedRatings for gain)
    rather than a spec-intended fit. Returns the keys consumed.
    """
    consumed: set[str] = set()

    if "serial" in camera_meta:
        ifd[_BASE_EXIF_TAGS["BodySerialNumber"]] = str(camera_meta["serial"])
        consumed.add("serial")

    if "brightness" in camera_meta:
        ifd[_BASE_EXIF_TAGS["BrightnessValue"]] = (int(camera_meta["brightness"]), 1)
        consumed.add("brightness")

    if "gamma" in camera_meta:
        # Camera gamma settings are stored as percent (100 == 1.0 gamma).
        ifd[_BASE_EXIF_TAGS["Gamma"]] = (int(camera_meta["gamma"]), 100)
        consumed.add("gamma")

    return consumed


_DIRECTION_TAG_NAMES = {
    "contrast_direction": "Contrast",
    "saturation_direction": "Saturation",
}


def _apply_direction_tags(ifd: Any, camera_meta: dict[str, Any]) -> set[str]:
    """Write the standard Contrast/Saturation tags from the Normal/Low/High
    codes the settings classes compute (relative to that camera's factory
    default). This is a coarse summary for tools that only know the standard
    vocabulary; the exact contrast/saturation values are written separately
    to their own private tags, not consumed here.
    """
    consumed: set[str] = set()

    for key, tag_name in _DIRECTION_TAG_NAMES.items():
        if key in camera_meta:
            ifd[_BASE_EXIF_TAGS[tag_name]] = int(camera_meta[key])
            consumed.add(key)

    return consumed


def _apply_private_camera_tags(ifd: Any, camera_meta: dict[str, Any]) -> set[str]:
    """Write camera settings with no standard EXIF equivalent to our own private
    tag IDs so each still lands in its own field. Values are JSON-encoded so
    numbers, bools, and nested dicts (e.g. level_range_low) round-trip cleanly.
    """
    consumed: set[str] = set()

    for name, tag_id in _PRIVATE_CAMERA_TAGS.items():
        if name in camera_meta:
            ifd[tag_id] = json.dumps(camera_meta[name])
            consumed.add(name)

    return consumed


def _apply_resolution_tags(ifd: Any, additional_meta: dict[str, Any]) -> set[str]:
    """Map a DPI hint onto the standard XResolution/YResolution tags. Returns keys consumed."""
    if "DPI" not in additional_meta:
        return set()

    dpi = additional_meta["DPI"]
    # A plain (num, denom) tuple here gets read by Pillow's TIFF/JPEG writer as
    # two separate rationals instead of one, so it must be an IFDRational.
    ifd[_BASE_EXIF_TAGS["XResolution"]] = IFDRational(dpi)
    ifd[_BASE_EXIF_TAGS["YResolution"]] = IFDRational(dpi)
    ifd[_BASE_EXIF_TAGS["ResolutionUnit"]] = 2  # inches
    return {"DPI"}


def _build_image_description(additional_meta: dict[str, Any]) -> tuple[str | None, set[str]]:
    parts = []
    consumed: set[str] = set()

    if "description" in additional_meta:
        parts.append(str(additional_meta["description"]))
        consumed.add("description")
    if "sample_id" in additional_meta:
        parts.append(f"Sample: {additional_meta['sample_id']}")
        consumed.add("sample_id")

    return (" | ".join(parts) if parts else None), consumed


def _leftover_metadata_json(
    camera_meta: dict[str, Any],
    additional_meta: dict[str, Any],
    consumed_camera: set[str],
    consumed_additional: set[str],
) -> str | None:
    """JSON for whatever camera/additional fields have no dedicated tag, kept in UserComment."""
    leftover: dict[str, Any] = {}

    remaining_camera = {k: v for k, v in camera_meta.items() if k not in consumed_camera}
    remaining_additional = {k: v for k, v in additional_meta.items() if k not in consumed_additional}

    if remaining_camera:
        leftover["camera"] = remaining_camera
    if remaining_additional:
        leftover["additional"] = remaining_additional

    return json.dumps(leftover, indent=2) if leftover else None


def save_tiff_with_metadata(
    pil_image: Image.Image,
    filepath: Path,
    metadata: dict[str, Any]
):
    """Save TIFF with metadata spread across standard EXIF tags where one
    exists, and our own private tags (see _PRIVATE_CAMERA_TAGS) where it
    doesn't. Anything neither covers is kept as JSON in UserComment.

    Pillow's TIFF writer does not serialize nested EXIF sub-IFDs
    (exif.get_ifd(ExifTags.IFD.Exif)) the way it does for JPEG, so every
    tag below is set on the top-level exif object instead.
    """
    exif = Exif()
    exif[_BASE_EXIF_TAGS['Software']] = _software_string()

    timestamp = metadata.get("timestamp", datetime.now().isoformat())
    exif_timestamp = datetime.fromisoformat(timestamp).strftime("%Y:%m:%d %H:%M:%S")
    exif[_BASE_EXIF_TAGS['DateTime']] = exif_timestamp
    exif[_BASE_EXIF_TAGS['DateTimeOriginal']] = exif_timestamp
    exif[_BASE_EXIF_TAGS['DateTimeDigitized']] = exif_timestamp

    camera_meta = metadata.get("camera", {})
    additional_meta = metadata.get("additional", {})
    consumed_camera: set[str] = set()
    consumed_additional: set[str] = set()

    if "model" in camera_meta:
        exif[_BASE_EXIF_TAGS['Model']] = str(camera_meta["model"])
        consumed_camera.add("model")

    consumed_camera |= _apply_standard_exposure_tags(exif, camera_meta)
    consumed_camera |= _apply_extended_standard_tags(exif, camera_meta)
    consumed_camera |= _apply_direction_tags(exif, camera_meta)
    consumed_camera |= _apply_private_camera_tags(exif, camera_meta)
    consumed_additional |= _apply_resolution_tags(exif, additional_meta)

    description, description_consumed = _build_image_description(additional_meta)
    if description:
        exif[_BASE_EXIF_TAGS['ImageDescription']] = description
    consumed_additional |= description_consumed

    leftover_json = _leftover_metadata_json(camera_meta, additional_meta, consumed_camera, consumed_additional)
    if leftover_json:
        exif[_BASE_EXIF_TAGS['UserComment']] = _encode_exif_user_comment(leftover_json)

    pil_image.save(filepath, format='TIFF', exif=exif, compression='tiff_deflate')
    debug(f"TIFF with EXIF metadata saved to {filepath}")


def save_jpeg_with_metadata(
    pil_image: Image.Image,
    filepath: Path,
    metadata: dict[str, Any]
):
    """Save JPEG with metadata spread across standard EXIF tags where one
    exists, and our own private tags (see _PRIVATE_CAMERA_TAGS) where it
    doesn't. Anything neither covers is kept as JSON in UserComment.
    """
    exif = Exif()
    exif[_BASE_EXIF_TAGS['Software']] = _software_string()

    timestamp = metadata.get("timestamp", datetime.now().isoformat())
    exif_timestamp = datetime.fromisoformat(timestamp).strftime("%Y:%m:%d %H:%M:%S")
    exif[_BASE_EXIF_TAGS['DateTime']] = exif_timestamp

    camera_meta = metadata.get("camera", {})
    additional_meta = metadata.get("additional", {})
    consumed_camera: set[str] = set()
    consumed_additional: set[str] = set()

    if "model" in camera_meta:
        exif[_BASE_EXIF_TAGS['Model']] = str(camera_meta["model"])
        consumed_camera.add("model")

    consumed_additional |= _apply_resolution_tags(exif, additional_meta)

    description, description_consumed = _build_image_description(additional_meta)
    if description:
        exif[_BASE_EXIF_TAGS['ImageDescription']] = description
    consumed_additional |= description_consumed

    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    exif_ifd[_BASE_EXIF_TAGS['DateTimeOriginal']] = exif_timestamp
    exif_ifd[_BASE_EXIF_TAGS['DateTimeDigitized']] = exif_timestamp
    consumed_camera |= _apply_standard_exposure_tags(exif_ifd, camera_meta)
    consumed_camera |= _apply_extended_standard_tags(exif_ifd, camera_meta)
    consumed_camera |= _apply_direction_tags(exif_ifd, camera_meta)
    consumed_camera |= _apply_private_camera_tags(exif_ifd, camera_meta)

    leftover_json = _leftover_metadata_json(camera_meta, additional_meta, consumed_camera, consumed_additional)
    if leftover_json:
        exif_ifd[_BASE_EXIF_TAGS['UserComment']] = _encode_exif_user_comment(leftover_json)

    pil_image.save(filepath, format='JPEG', exif=exif, quality=95)
    debug(f"JPEG with EXIF metadata saved to {filepath}")


def save_png_with_metadata(
    pil_image: Image.Image,
    filepath: Path,
    metadata: dict[str, Any]
):
    """Save PNG with one text chunk per field. PNG text chunks are cheap and
    arbitrary, so every camera and additional field gets its own readable
    chunk instead of being folded into a single JSON blob.
    """
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Software", _software_string())

    timestamp = metadata.get("timestamp", datetime.now().isoformat())
    pnginfo.add_text("DateTimeOriginal", timestamp)

    camera_meta = metadata.get("camera", {})
    for key, value in camera_meta.items():
        pnginfo.add_text(f"Camera.{key}", json.dumps(value) if isinstance(value, (dict, list)) else str(value))

    additional_meta = metadata.get("additional", {})
    save_kwargs: dict[str, Any] = {}

    if "DPI" in additional_meta:
        dpi = additional_meta["DPI"]
        save_kwargs["dpi"] = (dpi, dpi)

    for key, value in additional_meta.items():
        if key == "DPI":
            continue
        pnginfo.add_text(f"Additional.{key}", json.dumps(value) if isinstance(value, (dict, list)) else str(value))

    pil_image.save(filepath, format='PNG', pnginfo=pnginfo, **save_kwargs)
    debug(f"PNG metadata saved to {filepath}")