from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import datetime
from statistics import median

from UI.widgets.measurements.measurement_data import SEGMENT_BREAK, MeasurementData, break_indices
from UI.widgets.measurements.measurement_io import DeserializeResult, ExportResult, FormatContext, MeasurementLike
from UI.widgets.measurements.measurement_kind import MeasurementKindRegistry, Point2D
from UI.widgets.measurements.measurement_meta import DEFAULT_META
from UI.widgets.measurements.units import MeasurementUnit

# Dendroelevator describes points in a Leaflet "simple" map with y running
# upward, so a pixel (x, y) from the image's top-left corner sits at
# lng = x / S, lat = -y / S for the file's own unit scale S. S is not the
# loaded image's size: in the one file studied it is exactly 2**17 while the
# image is narrower, so S is recovered from the file itself (see _fit_scale)
# and the image only supplies the fractions the points are placed at. The
# file's ppm (pixels per mm) is in those same pixels, so it only converts
# straight to a DPI when the loaded image has the resolution the file was
# made on.
#
# Those map pixels are not quite image pixels. The viewer's WebGL layer lays
# its tiles out every 256 px, but each DeepZoom tile (254 px plus a 1 px
# overlap each side) adds only 254 new image pixels, so the map runs 256/254
# larger than the image at every zoom level. The ring measuring plugin takes
# both the points and ppm from that map, so both are scaled by this ratio
# (UMN-LATIS/elevator: Leaflet.TileLayer.GL.js, imageHandler_dendro.php).
_TILE_RATIO = 254 / 256
_MM_PER_INCH = 25.4
_POWER_OF_TWO_SNAP = 0.01
_BOUNDS_TOLERANCE = 0.001
_WIDTH_DECIMALS = 5

LINE_KIND = "Arbitrary Line"
POINT_KIND = "Point"

# Dendroelevator draws every tenth year's ring red and the rest blue.
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
_RING_INDEX = f"{_ATTR_PREFIX}ring_index"
_RING_YEAR = f"{_ATTR_PREFIX}ring_year"
_POINT_INDEX = f"{_ATTR_PREFIX}index"
_YEAR = f"{_ATTR_PREFIX}year"


def sniff(doc: object) -> bool:
    return isinstance(doc, dict) and isinstance(doc.get("points"), list) and "ppm" in doc


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, bool)) or _is_number(value)


def _path_length(xy: list[tuple[float, float]], breaks: frozenset[int]) -> float:
    """Length walked along *xy*, leaving out the jump into any point that starts a new run."""
    return sum(
        math.hypot(xy[i][0] - xy[i - 1][0], xy[i][1] - xy[i - 1][1]) for i in range(1, len(xy)) if i not in breaks
    )


def _ring_chains(years: list[int | None], forward: bool) -> list[tuple[int, int, int | None]]:
    """
    The rings, as (first point index, last point index, ring year): each ring
    is the stretch of the measuring path between two consecutive year points.
    A ring's year is that of the year point it starts from along the path, so
    the stretch ending at year point Y is ring Y + 1 when measuring toward
    older wood (the file studied, where ptWidths confirms it) and Y - 1 when
    measuring the other way. The stretch before the first year point is the
    ring that point closes off; anything after the last one is a ring with no
    year.
    """
    step = -1 if forward else 1
    chains: list[tuple[int, int, int | None]] = []
    start = 0
    for index, year in enumerate(years):
        if year is None:
            continue
        if index > start:
            chains.append((start, index, year + step))
        start = index
    if len(years) - 1 > start:
        chains.append((start, len(years) - 1, None))
    return chains


# ----------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------


def _parse_lat_lng(raw: object) -> tuple[float, float] | None:
    try:
        lat, lng = float(raw["lat"]), float(raw["lng"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return None
    return (lat, lng) if math.isfinite(lat) and math.isfinite(lng) else None


def _point_attrs(raw: dict, file_index: int, run_index: int) -> dict:
    attrs: dict = {_POINT_INDEX: file_index}
    year = raw.get("year")
    if _is_number(year):
        attrs[_YEAR] = int(year)
    if isinstance(raw.get("earlywood"), bool):
        attrs[f"{_ATTR_PREFIX}earlywood"] = raw["earlywood"]
    for flag in ("start", "break", "skip"):
        if raw.get(flag) is True:
            attrs[f"{_ATTR_PREFIX}{flag}"] = True
    if raw.get("start") is True and run_index > 0:
        attrs[SEGMENT_BREAK] = True
    for key, value in raw.items():
        if key not in _POINT_KEYS and _is_scalar(value):
            attrs[f"{_POINT_EXTRA_PREFIX}{key}"] = value
    return attrs


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, math.ceil(math.log2(max(1, value))))


def _snap_to_power_of_two(value: float) -> float:
    nearest = 2.0 ** round(math.log2(value))
    return nearest if abs(value - nearest) / nearest <= _POWER_OF_TWO_SNAP else value


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


def _fit_scale(
    doc: dict,
    xy_units: list[tuple[float, float]],
    chains: list[tuple[int, int, int | None]],
    breaks: frozenset[int],
) -> float | None:
    """
    The file's unit scale in pixels, recovered from its own numbers: a ring's
    width in mm times ppm is its length in pixels, and the same ring measured
    in the file's lat/lng units gives pixels per unit. Snapped to a power of
    two when within 1% of one, since the file's rounding would otherwise leave
    it a few parts per million off.
    """
    widths = _tw_series(doc)
    ppm = doc.get("ppm")
    if widths is None or not _is_number(ppm) or ppm <= 0:
        return None
    implied = []
    for first, last, ring_year in chains:
        if ring_year not in widths:
            continue
        local_breaks = frozenset(b - first for b in breaks if first < b <= last)
        length = _path_length(xy_units[first:last + 1], local_breaks)
        if length > 0:
            implied.append(widths[ring_year] * ppm / length)
    return _snap_to_power_of_two(median(implied)) if implied else None


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


def _sample_attrs(doc: dict, scale: float, ppm: object, has_dpi: bool) -> dict:
    """What describes the sample as a whole; every ring carries a copy, so export can read it from whichever rings are left."""
    attrs: dict = {
        f"{_ATTR_PREFIX}name": _sample_name(doc),
        f"{_ATTR_PREFIX}scale": scale,
        f"{_ATTR_PREFIX}extras": json.dumps(_extras(doc)),
    }
    for source, key in (
        ("year", "series_year"), ("forwardDirection", "forward_direction"), ("subAnnual", "sub_annual"),
        ("earlywood", "earlywood"), ("index", "index"),
    ):
        if _is_scalar(doc.get(source)):
            attrs[f"{_ATTR_PREFIX}{key}"] = doc[source]
    if has_dpi:
        attrs[f"{_ATTR_PREFIX}ppm"] = ppm
    return attrs


def load(doc: dict, ctx: FormatContext, registry: MeasurementKindRegistry) -> DeserializeResult:
    result = DeserializeResult()
    if ctx.image_dims is None:
        result.warnings.append("Open the image these measurements were made on before importing a Dendroelevator file.")
        return result
    if LINE_KIND not in registry or POINT_KIND not in registry:
        result.warnings.append("This build has no line or point measurement kind to import Dendroelevator data into.")
        return result

    width, height = ctx.image_dims
    ppm = doc.get("ppm")
    has_dpi = _is_number(ppm) and ppm > 0
    if has_dpi:
        result.dpi = ppm * _MM_PER_INCH * _TILE_RATIO
    else:
        result.warnings.append("Missing or invalid ppm: the image scale was not imported.")

    lat_lngs: list[tuple[float, float]] = []
    raws: list[dict] = []
    file_indices: list[int] = []
    for i, raw in enumerate(doc["points"]):
        lat_lng = _parse_lat_lng(raw.get("latLng")) if isinstance(raw, dict) else None
        if lat_lng is None:
            result.warnings.append(f"Point {i}: missing or malformed latLng, skipped.")
            continue
        lat_lngs.append(lat_lng)
        raws.append(raw)
        file_indices.append(i)

    years = [int(r["year"]) if _is_number(r.get("year")) else None for r in raws]
    forward = doc.get("forwardDirection") is True
    chains = _ring_chains(years, forward)
    start_flags = frozenset(i for i, r in enumerate(raws) if i > 0 and r.get("start") is True)
    xy_units = [(lng, -lat) for lat, lng in lat_lngs]

    scale = _fit_scale(doc, xy_units, chains, start_flags)
    if scale is None:
        scale = float(_next_power_of_two(max(width, height)))
        result.warnings.append(
            f"The file has no ring widths to recover its scale from, so {scale:,.0f} px was assumed."
        )

    pixels_per_unit = scale * _TILE_RATIO

    def to_fraction(lat: float, lng: float) -> Point2D:
        return lng * pixels_per_unit / width, -lat * pixels_per_unit / height

    sample_attrs = _sample_attrs(doc, scale, ppm, has_dpi)
    for ring_index, (first, last, ring_year) in enumerate(chains):
        point_attrs = tuple(
            _point_attrs(raws[i], file_indices[i], i - first) for i in range(first, last + 1)
        )
        attrs = dict(sample_attrs)
        attrs[_RING_INDEX] = ring_index
        if ring_year is not None:
            attrs[_RING_YEAR] = ring_year
        color = ""
        if ring_year is not None:
            color = _DECADE_COLOR if ring_year % 10 == 0 else _YEAR_COLOR
        meta = DEFAULT_META._replace(
            title=str(ring_year) if ring_year is not None else "",
            unit=MeasurementUnit.MM,
            line_color=color,
            tag_background_color=color,
        )
        points = tuple(to_fraction(*lat_lngs[i]) for i in range(first, last + 1))
        result.entries.append((LINE_KIND, points, meta, MeasurementData(attrs, point_attrs)))

    if not chains:
        result.warnings.append("The file has no ring to import: it needs at least two usable points.")

    _load_annotations(doc, to_fraction, result)

    outside = sum(
        1
        for _, entry_points, _, _ in result.entries
        for fx, fy in entry_points
        if not (-_BOUNDS_TOLERANCE <= fx <= 1 + _BOUNDS_TOLERANCE and -_BOUNDS_TOLERANCE <= fy <= 1 + _BOUNDS_TOLERANCE)
    )
    if outside:
        result.warnings.append(
            f"{outside} point(s) fall outside the loaded {width:,} x {height:,} px image — it may not be the image "
            "these were measured on, or may be cropped differently (for example without an attached calibration strip)."
        )
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
            attrs[_YEAR] = int(year)
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


def _is_ring(measurement: MeasurementLike) -> bool:
    index = measurement.data.attrs.get(_RING_INDEX)
    return measurement.kind == LINE_KIND and isinstance(index, int) and not isinstance(index, bool)


def _save_date() -> dict:
    now = datetime.now()
    return {"day": now.day, "hour": now.hour, "year": now.year, "month": now.month, "minute": now.minute}


def save(measurements: list[MeasurementLike], ctx: FormatContext) -> ExportResult:
    result = ExportResult(None)
    if ctx.image_dims is None:
        result.warnings.append("No image is open, so there is no pixel size to export against.")
        return result
    rings = sorted((m for m in measurements if _is_ring(m)), key=lambda m: m.data.attrs[_RING_INDEX])
    if not rings:
        result.warnings.append(
            "Nothing to export: Dendroelevator files hold the ring lines made by importing one, and none is placed."
        )
        return result
    attrs = rings[0].data.attrs

    ppm = ctx.dpi / _MM_PER_INCH if ctx.dpi else None
    recorded_ppm = attrs.get(f"{_ATTR_PREFIX}ppm")
    if ppm is None and _is_number(recorded_ppm) and recorded_ppm > 0:
        ppm = float(recorded_ppm) * _TILE_RATIO
        result.warnings.append("The image has no DPI set, so the ppm recorded in the original file was reused.")
    if ppm is None:
        result.warnings.append("Set the image's DPI first: Dendroelevator files record the scale as ppm.")
        return result

    width, height = ctx.image_dims
    recorded_scale = attrs.get(f"{_ATTR_PREFIX}scale")
    scale = float(recorded_scale) if _is_number(recorded_scale) and recorded_scale > 0 else float(
        _next_power_of_two(max(width, height))
    )
    pixels_per_unit = scale * _TILE_RATIO

    # Neighbouring rings share the year point between them, so each file point
    # is collected once, from the first ring that holds it.
    collected: dict[int, dict] = {}
    lengths_mm: dict[int, float] = {}
    unindexed = 0
    for ring in rings:
        xy_px = [(fx * width, fy * height) for fx, fy in ring.points]
        for i, (x_px, y_px) in enumerate(xy_px):
            point_attrs = ring.data.point_attrs[i] if i < len(ring.data.point_attrs) else {}
            file_index = point_attrs.get(_POINT_INDEX)
            if not isinstance(file_index, int):
                unindexed += 1
                continue
            if file_index in collected:
                continue
            raw: dict = {
                "skip": point_attrs.get(f"{_ATTR_PREFIX}skip") is True,
                "break": point_attrs.get(f"{_ATTR_PREFIX}break") is True,
                "start": point_attrs.get(f"{_ATTR_PREFIX}start") is True or point_attrs.get(SEGMENT_BREAK) is True,
                "latLng": {"lat": -y_px / pixels_per_unit, "lng": x_px / pixels_per_unit},
            }
            year = point_attrs.get(_YEAR)
            if isinstance(year, int) and not isinstance(year, bool):
                raw["year"] = year
            if isinstance(point_attrs.get(f"{_ATTR_PREFIX}earlywood"), bool):
                raw["earlywood"] = point_attrs[f"{_ATTR_PREFIX}earlywood"]
            for key, value in point_attrs.items():
                if key.startswith(_POINT_EXTRA_PREFIX):
                    raw[key[len(_POINT_EXTRA_PREFIX):]] = value
            collected[file_index] = raw
        ring_year = ring.data.attrs.get(_RING_YEAR)
        if isinstance(ring_year, int) and not isinstance(ring_year, bool):
            breaks = break_indices(ring.data, len(xy_px))
            lengths_mm[ring_year] = lengths_mm.get(ring_year, 0.0) + _path_length(xy_px, breaks) / ppm
    if unindexed:
        result.warnings.append(f"{unindexed} ring point(s) had no position in the original file and were left out.")

    points = [collected[i] for i in sorted(collected)]
    years = [p["year"] for p in points if "year" in p]
    tw_years = sorted(lengths_mm)
    name = attrs.get(f"{_ATTR_PREFIX}name") or "FieldWeave measurement"
    series_year = attrs.get(f"{_ATTR_PREFIX}series_year")

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
        "year": int(series_year) if _is_number(series_year) else (min(years) - 1 if years else None),
        "forwardDirection": _bool_attr(attrs, "forward_direction", False),
        "subAnnual": _bool_attr(attrs, "sub_annual", False),
        "earlywood": _bool_attr(attrs, "earlywood", True),
        "index": int(attrs[f"{_ATTR_PREFIX}index"]) if _is_number(attrs.get(f"{_ATTR_PREFIX}index")) else 0,
        "points": points,
        "annotations": _save_annotations(measurements, width, height, pixels_per_unit, result),
        "ppm": ppm / _TILE_RATIO,
        "ptWidths": ptwidths,
        "ellipses": [],
        "currentView": dict(_DEFAULT_VIEW),
        "pithEstimate": dict(_DEFAULT_PITH),
    }
    document.update(extras)
    result.document = document
    return result


def _save_annotations(
    measurements: list[MeasurementLike], width: int, height: int, scale: float, result: ExportResult
) -> dict:
    annotations: dict = {}
    skipped = 0
    for m in measurements:
        if m.kind != POINT_KIND or not m.points:
            continue
        year = m.data.attrs.get(_YEAR)
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
