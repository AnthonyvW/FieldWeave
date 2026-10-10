from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import datetime
from statistics import median

from UI.widgets.measurements.measurement_data import POINT_COLOR, SEGMENT_BREAK, MeasurementData, break_indices
from UI.widgets.measurements.measurement_io import DeserializeResult, ExportResult, FormatContext, MeasurementLike
from UI.widgets.measurements.measurement_kind import MeasurementKindRegistry, Point2D
from UI.widgets.measurements.measurement_meta import DEFAULT_META
from UI.widgets.measurements.units import MeasurementUnit

# Dendroelevator describes points in a Leaflet "simple" map where the
# image's longest side is 1.0 and y runs upward, so a pixel (x, y) sits at
# lng = x / longest, lat = -y / longest. Its ppm (pixels per mm) is measured
# in that same longest-side space, which is why it only converts straight
# to a DPI when the image is the same resolution the file was made on.
_MM_PER_INCH = 25.4
_SCALE_TOLERANCE = 0.01
_WIDTH_DECIMALS = 5

LINE_KIND = "Arbitrary Line"
POINT_KIND = "Point"

# Dendroelevator marks every tenth year's ring red and the rest blue.
_DECADE_COLOR = "#ff1c22"
_YEAR_COLOR = "#1c7bff"
_DEFAULT_ANNOTATION_COLOR = "#ff1c22"
_DEFAULT_VIEW = {
    "brightness": "100",
    "contrast": "100",
    "sharpness": "0",
    "emboss": "0",
    "saturate": "100",
    "edgeDetect": "0",
    "invert": False,
}
_DEFAULT_PITH = {
    "yearEstimate": None,
    "growthRate": None,
    "pithLatLng": None,
    "toPithRadius": None,
    "estimatedRadiiArray": None,
}

_POINT_KEYS = {"skip", "year", "break", "start", "latLng", "earlywood"}
_ANNOTATION_KEYS = {"text", "year", "color", "latLng"}
_TOP_LEVEL_KEYS = {
    "SaveDate", "year", "forwardDirection", "subAnnual", "earlywood", "index", "points", "annotations", "ppm", "ptWidths",
}
_ATTR_PREFIX = "dendro."
_POINT_EXTRA_PREFIX = "dendro.extra."


def sniff(doc: object) -> bool:
    return isinstance(doc, dict) and isinstance(doc.get("points"), list) and "ppm" in doc


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, bool)) or _is_number(value)


def _ring_lengths(xy: list[tuple[float, float]], years: list[int | None], breaks: frozenset[int]) -> dict[int, float]:
    """
    Width of each ring, keyed the way Dendroelevator's ptWidths is: the
    length walked since the previous year point, excluding the jump into a
    point that starts a new run — so a ring crossing a core break sums the
    measured stretches on either side and skips the gap between them. It is
    filed under year + 1, the ring whose far edge the year point marks.
    """
    lengths: dict[int, float] = {}
    walked = 0.0
    for i, point in enumerate(xy):
        if i and i not in breaks:
            walked += math.hypot(point[0] - xy[i - 1][0], point[1] - xy[i - 1][1])
        year = years[i]
        if year is not None:
            lengths[year + 1] = walked
            walked = 0.0
    return lengths


# ----------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------


def _parse_lat_lng(raw: object) -> tuple[float, float] | None:
    try:
        lat, lng = float(raw["lat"]), float(raw["lng"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return None
    return (lat, lng) if math.isfinite(lat) and math.isfinite(lng) else None


def _point_attrs(raw: dict, index: int) -> dict:
    attrs: dict = {}
    year = raw.get("year")
    if _is_number(year):
        attrs[f"{_ATTR_PREFIX}year"] = int(year)
        attrs[POINT_COLOR] = _DECADE_COLOR if int(year) % 10 == 0 else _YEAR_COLOR
    if isinstance(raw.get("earlywood"), bool):
        attrs[f"{_ATTR_PREFIX}earlywood"] = raw["earlywood"]
    for flag in ("start", "break", "skip"):
        if raw.get(flag) is True:
            attrs[f"{_ATTR_PREFIX}{flag}"] = True
    if raw.get("start") is True and index > 0:
        attrs[SEGMENT_BREAK] = True
    for key, value in raw.items():
        if key not in _POINT_KEYS and _is_scalar(value):
            attrs[f"{_POINT_EXTRA_PREFIX}{key}"] = value
    return attrs


def _fitted_scale_warning(
    doc: dict, xy_units: list[tuple[float, float]], years: list[int | None], breaks: frozenset[int], scale: int
) -> str | None:
    """Compare the longest-side length implied by the file's own ring widths against the loaded image's, since a mismatch means the wrong image (or a different resolution)."""
    widths = _tw_series(doc)
    ppm = doc.get("ppm")
    if widths is None or not _is_number(ppm) or ppm <= 0:
        return None
    implied = [
        widths[year] * ppm / length
        for year, length in _ring_lengths(xy_units, years, breaks).items()
        if year in widths and length > 0
    ]
    if not implied:
        return None
    fitted = median(implied)
    if abs(fitted - scale) / scale <= _SCALE_TOLERANCE:
        return None
    return (
        f"The ring widths in this file imply an image {fitted:,.0f} px on its longest side, but the loaded image is "
        f"{scale:,} px — this may not be the image it was measured on."
    )


def _tw_series(doc: dict) -> dict[int, float] | None:
    tw = (doc.get("ptWidths") or {}).get("tw") if isinstance(doc.get("ptWidths"), dict) else None
    if not isinstance(tw, dict):
        return None
    xs, ys = tw.get("x"), tw.get("y")
    if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys):
        return None
    if not all(_is_number(v) for v in xs + ys):
        return None
    return {int(x): float(y) for x, y in zip(xs, ys)}


def _sample_name(doc: dict) -> str:
    tw = (doc.get("ptWidths") or {}).get("tw") if isinstance(doc.get("ptWidths"), dict) else None
    name = tw.get("name") if isinstance(tw, dict) else None
    if not isinstance(name, str):
        return ""
    return name[: -len("_tw")] if name.endswith("_tw") else name


def _extras(doc: dict) -> dict:
    extras = {k: v for k, v in doc.items() if k not in _TOP_LEVEL_KEYS}
    ptwidths = doc.get("ptWidths")
    if isinstance(ptwidths, dict):
        other = {k: v for k, v in ptwidths.items() if k != "tw"}
        if other:
            extras["ptWidths_other"] = other
    return extras


def load(doc: dict, ctx: FormatContext, registry: MeasurementKindRegistry) -> DeserializeResult:
    result = DeserializeResult()
    if ctx.image_dims is None:
        result.warnings.append("Open the image these measurements were made on before importing a Dendroelevator file.")
        return result
    if LINE_KIND not in registry or POINT_KIND not in registry:
        result.warnings.append("This build has no line or point measurement kind to import Dendroelevator data into.")
        return result

    width, height = ctx.image_dims
    scale = max(width, height)
    ppm = doc.get("ppm")
    if _is_number(ppm) and ppm > 0:
        result.dpi = ppm * _MM_PER_INCH
    else:
        result.warnings.append("Missing or invalid ppm: the image scale was not imported.")

    def to_fraction(lat: float, lng: float) -> Point2D:
        return lng * scale / width, -lat * scale / height

    points: list[Point2D] = []
    xy_units: list[tuple[float, float]] = []
    point_attrs: list[dict] = []
    years: list[int | None] = []
    for i, raw in enumerate(doc["points"]):
        lat_lng = _parse_lat_lng(raw.get("latLng")) if isinstance(raw, dict) else None
        if lat_lng is None:
            result.warnings.append(f"Point {i}: missing or malformed latLng, skipped.")
            continue
        attrs = _point_attrs(raw, len(points))
        points.append(to_fraction(*lat_lng))
        xy_units.append((lat_lng[1], -lat_lng[0]))
        point_attrs.append(attrs)
        year = attrs.get(f"{_ATTR_PREFIX}year")
        years.append(year if isinstance(year, int) else None)

    if points:
        name = _sample_name(doc)
        attrs = {f"{_ATTR_PREFIX}name": name, f"{_ATTR_PREFIX}extras": json.dumps(_extras(doc))}
        for source, key in (
            ("year", "series_year"), ("forwardDirection", "forward_direction"), ("subAnnual", "sub_annual"),
            ("earlywood", "earlywood"), ("index", "index"),
        ):
            if _is_scalar(doc.get(source)):
                attrs[f"{_ATTR_PREFIX}{key}"] = doc[source]
        if result.dpi is not None:
            attrs[f"{_ATTR_PREFIX}ppm"] = ppm
        meta = DEFAULT_META._replace(
            title=name, unit=MeasurementUnit.MM, point_label_template=f"{{{_ATTR_PREFIX}year}}"
        )
        result.entries.append((LINE_KIND, tuple(points), meta, MeasurementData(attrs, tuple(point_attrs))))

        breaks = break_indices(MeasurementData({}, tuple(point_attrs)), len(points))
        mismatch = _fitted_scale_warning(doc, xy_units, years, breaks, scale)
        if mismatch is not None:
            result.warnings.append(mismatch)
    else:
        result.warnings.append("The file has no usable points.")

    _load_annotations(doc, to_fraction, result)
    return result


def _annotation_order(item: tuple[str, object]) -> tuple[bool, int, str]:
    """Numeric keys in numeric order (Dendroelevator numbers them), anything else after."""
    key = str(item[0])
    return (not key.isdigit(), int(key) if key.isdigit() else 0, key)


def _load_annotations(
    doc: dict, to_fraction: Callable[[float, float], Point2D], result: DeserializeResult
) -> None:
    annotations = doc.get("annotations")
    if not isinstance(annotations, dict):
        return
    for key, raw in sorted(annotations.items(), key=_annotation_order):
        lat_lng = _parse_lat_lng(raw.get("latLng")) if isinstance(raw, dict) else None
        if lat_lng is None:
            result.warnings.append(f"Annotation {key}: missing or malformed latLng, skipped.")
            continue
        text = raw.get("text")
        color = raw.get("color")
        year = raw.get("year")
        attrs: dict = {}
        if _is_number(year):
            attrs[f"{_ATTR_PREFIX}year"] = int(year)
        extras = {k: v for k, v in raw.items() if k not in _ANNOTATION_KEYS}
        if extras:
            attrs[f"{_ATTR_PREFIX}annotation_extras"] = json.dumps(extras)
        color_hex = color if isinstance(color, str) else ""
        # A marker whose year is its label (tinted like the marker) and whose
        # note text appears on hover, which is how Dendroelevator shows it.
        meta = DEFAULT_META._replace(
            title=str(int(year)) if _is_number(year) else "",
            description=text if isinstance(text, str) else "",
            line_color=color_hex,
            tag_background_color=color_hex,
        )
        result.entries.append((POINT_KIND, (to_fraction(*lat_lng),), meta, MeasurementData(attrs, ())))


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------


def _bool_attr(attrs: dict, key: str, default: bool) -> bool:
    value = attrs.get(f"{_ATTR_PREFIX}{key}")
    return value if isinstance(value, bool) else default


def _json_attr(attrs: dict, key: str) -> dict:
    raw = attrs.get(f"{_ATTR_PREFIX}{key}")
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _has_years(measurement: MeasurementLike) -> bool:
    return measurement.kind == LINE_KIND and any(f"{_ATTR_PREFIX}year" in a for a in measurement.data.point_attrs)


def _save_date() -> dict:
    now = datetime.now()
    return {"day": now.day, "hour": now.hour, "year": now.year, "month": now.month, "minute": now.minute}


def save(measurements: list[MeasurementLike], ctx: FormatContext) -> ExportResult:
    result = ExportResult(None)
    if ctx.image_dims is None:
        result.warnings.append("No image is open, so there is no pixel size to export against.")
        return result
    rings = [m for m in measurements if _has_years(m)]
    if not rings:
        result.warnings.append(
            "Nothing to export: Dendroelevator files hold a line of ring-boundary points that carry years, and none is placed."
        )
        return result
    if len(rings) > 1:
        result.warnings.append("Several year-labelled lines are placed; only the first was exported.")
    ring = rings[0]
    attrs = ring.data.attrs

    ppm = ctx.dpi / _MM_PER_INCH if ctx.dpi else None
    recorded_ppm = attrs.get(f"{_ATTR_PREFIX}ppm")
    if ppm is None and _is_number(recorded_ppm) and recorded_ppm > 0:
        ppm = float(recorded_ppm)
        result.warnings.append("The image has no DPI set, so the ppm recorded in the original file was reused.")
    if ppm is None:
        result.warnings.append("Set the image's DPI first: Dendroelevator files record the scale as ppm.")
        return result

    width, height = ctx.image_dims
    scale = max(width, height)
    breaks = break_indices(ring.data, len(ring.points))

    points = []
    xy_px: list[tuple[float, float]] = []
    years: list[int | None] = []
    for i, (fx, fy) in enumerate(ring.points):
        point_attrs = ring.data.point_attrs[i] if i < len(ring.data.point_attrs) else {}
        x_px, y_px = fx * width, fy * height
        xy_px.append((x_px, y_px))
        raw: dict = {
            "skip": point_attrs.get(f"{_ATTR_PREFIX}skip") is True,
            "break": point_attrs.get(f"{_ATTR_PREFIX}break") is True,
            "start": point_attrs.get(f"{_ATTR_PREFIX}start") is True or i in breaks,
            "latLng": {"lat": -y_px / scale, "lng": x_px / scale},
        }
        year = point_attrs.get(f"{_ATTR_PREFIX}year")
        if isinstance(year, int) and not isinstance(year, bool):
            raw["year"] = year
        years.append(raw.get("year"))
        if isinstance(point_attrs.get(f"{_ATTR_PREFIX}earlywood"), bool):
            raw["earlywood"] = point_attrs[f"{_ATTR_PREFIX}earlywood"]
        for key, value in point_attrs.items():
            if key.startswith(_POINT_EXTRA_PREFIX):
                raw[key[len(_POINT_EXTRA_PREFIX):]] = value
        points.append(raw)

    lengths_mm = {year: length / ppm for year, length in _ring_lengths(xy_px, years, breaks).items()}
    tw_years = sorted(lengths_mm)
    name = attrs.get(f"{_ATTR_PREFIX}name") or ring.meta.title or "FieldWeave measurement"
    series_year = attrs.get(f"{_ATTR_PREFIX}series_year")
    known_years = [y for y in years if y is not None]

    ptwidths = {
        "tw": {
            "x": tw_years,
            "y": [round(lengths_mm[y], _WIDTH_DECIMALS) for y in tw_years],
            "name": f"{name}_tw",
        },
    }
    extras = _json_attr(attrs, "extras")
    ptwidths.update(extras.pop("ptWidths_other", None) or {})

    document: dict = {
        "SaveDate": _save_date(),
        "year": int(series_year) if _is_number(series_year) else (min(known_years) - 1 if known_years else None),
        "forwardDirection": _bool_attr(attrs, "forward_direction", False),
        "subAnnual": _bool_attr(attrs, "sub_annual", False),
        "earlywood": _bool_attr(attrs, "earlywood", True),
        "index": int(attrs["dendro.index"]) if _is_number(attrs.get("dendro.index")) else 0,
        "points": points,
        "annotations": _save_annotations(measurements, width, height, scale, result),
        "ppm": ppm,
        "ptWidths": ptwidths,
        "ellipses": [],
        "currentView": dict(_DEFAULT_VIEW),
        "pithEstimate": dict(_DEFAULT_PITH),
    }
    document.update(extras)
    result.document = document
    return result


def _save_annotations(
    measurements: list[MeasurementLike], width: int, height: int, scale: int, result: ExportResult
) -> dict:
    annotations: dict = {}
    skipped = 0
    for m in measurements:
        if m.kind != POINT_KIND or not m.points:
            continue
        year = m.data.attrs.get(f"{_ATTR_PREFIX}year")
        if not _is_number(year):
            skipped += 1
            continue
        fx, fy = m.points[0]
        entry = {
            "code": [],
            "text": m.meta.description,
            "year": int(year),
            "color": m.meta.line_color or m.meta.tag_background_color or _DEFAULT_ANNOTATION_COLOR,
            "latLng": {"lat": -fy * height / scale, "lng": fx * width / scale},
            "description": [],
        }
        entry.update(_json_attr(m.data.attrs, "annotation_extras"))
        annotations[str(len(annotations))] = entry
    if skipped:
        result.warnings.append(f"{skipped} point(s) without a year were not exported as annotations.")
    return annotations
