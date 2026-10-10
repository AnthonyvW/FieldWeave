from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from UI.widgets.measurements.measurement_data import (
    MeasurementData,
    data_from_dict,
    data_to_dict,
)
from UI.widgets.measurements.measurement_kind import MeasurementKindRegistry, Point2D
from UI.widgets.measurements.measurement_meta import DEFAULT_META, MeasurementMeta
from UI.widgets.measurements.units import MeasurementUnit
from common.fieldweaveConfig import FIELDWEAVE_VERSION

# Marks a file as one FieldWeave wrote, stored first under HEADER_KEY so it
# is the first thing a reader (or a person opening the file) sees.
HEADER_KEY = "fieldweave"
FILE_TYPE = "measurements"

# Bumped whenever the document shape changes in a way old readers can't
# already tolerate (deserialize_measurements already drops unknown meta
# keys and skips unknown/malformed entries, so most additions — a new
# kind, a new MeasurementMeta field, the optional per-entry "data" — don't
# need a bump at all).
FORMAT_VERSION = 1


class MeasurementLike(Protocol):
    """What serialize_measurements needs from each entry — Measurement itself satisfies this structurally, without measurement_io needing to import it (and risk a cycle back from measurement_overlay.py)."""

    kind: str
    points: tuple[Point2D, ...]
    meta: MeasurementMeta
    data: MeasurementData


@dataclass
class DeserializeResult:
    """Entries as plain (kind, points, meta, data) tuples — the caller wraps each into its own Measurement type — plus any warnings for entries that were skipped or fell back to defaults rather than crashing the load."""

    entries: list[tuple[str, tuple[Point2D, ...], MeasurementMeta, MeasurementData]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # The image scale the file itself recorded, for formats that carry one —
    # the caller decides whether to apply it.
    dpi: float | None = None


@dataclass(frozen=True)
class FormatContext:
    """What a foreign format needs to know about the image the measurements sit on: its pixel size (for converting to and from its own coordinates) and its DPI."""

    image_dims: tuple[int, int] | None
    dpi: float | None


@dataclass
class ExportResult:
    """The document to write, or None when nothing could be exported (then *warnings* says why)."""

    document: dict | None
    warnings: list[str] = field(default_factory=list)


def _meta_to_dict(meta: MeasurementMeta) -> dict:
    data = meta._asdict()
    # Units serialize by their display value; None stays None. area_unit
    # is the same kind of optional-unit override as unit (feature 13).
    data["unit"] = meta.unit.value if meta.unit is not None else None
    data["area_unit"] = meta.area_unit.value if meta.area_unit is not None else None
    return data


def _meta_from_dict(data: dict) -> MeasurementMeta:
    fields = {k: v for k, v in data.items() if k in MeasurementMeta._fields}
    for unit_field in ("unit", "area_unit"):
        if unit_field in fields:
            value = fields[unit_field]
            fields[unit_field] = MeasurementUnit(value) if value is not None else None
    return MeasurementMeta(**fields)


def _measurement_to_dict(m: MeasurementLike) -> dict:
    entry = {"kind": m.kind, "points": [list(p) for p in m.points], "meta": _meta_to_dict(m.meta)}
    if not m.data.is_empty:
        entry["data"] = data_to_dict(m.data)
    return entry


def file_header() -> dict:
    return {"type": FILE_TYPE, "app_version": FIELDWEAVE_VERSION, "format_version": FORMAT_VERSION}


def is_fieldweave_document(data: object) -> bool:
    """True for a file FieldWeave wrote — recognized by its header, or, for files from before the header existed, by a top-level "measurements" list."""
    if not isinstance(data, dict):
        return False
    header = data.get(HEADER_KEY)
    if isinstance(header, dict) and header.get("type") == FILE_TYPE:
        return True
    return isinstance(data.get("measurements"), list)


def serialize_measurements(measurements: list[MeasurementLike]) -> dict:
    """A plain, JSON-serializable document of *measurements* — see deserialize_measurements for the read side."""
    return {
        HEADER_KEY: file_header(),
        "measurements": [_measurement_to_dict(m) for m in measurements],
    }


def deserialize_measurements(data: dict, registry: MeasurementKindRegistry) -> DeserializeResult:
    """
    Parse a document produced by ``serialize_measurements``, validating
    each entry against *registry* rather than trusting the file — a
    hand-edited file, or one written by a newer build with kinds this
    one doesn't know about yet, shouldn't be able to crash the load;
    its bad entries are just skipped and reported instead.
    """
    entries = data.get("measurements")
    if not isinstance(entries, list):
        return DeserializeResult(warnings=["Malformed file: 'measurements' is not a list."])

    result = DeserializeResult()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            result.warnings.append(f"Entry {i}: not an object, skipped.")
            continue
        kind = entry.get("kind")
        kind_entry = registry.get(kind)
        if kind_entry is None:
            result.warnings.append(f"Entry {i}: unknown measurement kind {kind!r}, skipped.")
            continue
        raw_points = entry.get("points")
        if not isinstance(raw_points, list):
            result.warnings.append(f"Entry {i} ({kind}): missing or malformed points, skipped.")
            continue
        try:
            points = tuple((float(p[0]), float(p[1])) for p in raw_points)
        except (TypeError, ValueError, IndexError):
            result.warnings.append(f"Entry {i} ({kind}): malformed point values, skipped.")
            continue
        required = kind_entry.required_points
        if required is not None and len(points) != required:
            result.warnings.append(f"Entry {i} ({kind}): expected {required} points, got {len(points)}, skipped.")
            continue

        try:
            meta = _meta_from_dict(entry.get("meta") or {})
        except (TypeError, ValueError) as exc:
            result.warnings.append(f"Entry {i} ({kind}): malformed meta ({exc}), using defaults.")
            meta = DEFAULT_META

        data = data_from_dict(entry.get("data"), len(points), f"Entry {i} ({kind})", result.warnings)

        result.entries.append((kind, points, meta, data))

    return result


def save_measurements_to_file(path: str | Path, measurements: list[MeasurementLike]) -> None:
    document = serialize_measurements(measurements)
    Path(path).write_text(json.dumps(document, indent=2), encoding="utf-8")


def load_measurements_from_file(path: str | Path, registry: MeasurementKindRegistry) -> DeserializeResult:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return DeserializeResult(warnings=[f"Could not read {path}: {exc}"])
    if not isinstance(document, dict):
        return DeserializeResult(warnings=[f"{path} does not contain a measurement document."])
    return deserialize_measurements(document, registry)
