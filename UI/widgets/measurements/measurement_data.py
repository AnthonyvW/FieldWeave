from __future__ import annotations

import math
import re
from typing import NamedTuple

AttrValue = str | int | float | bool | None
Attrs = dict[str, AttrValue]
Point = tuple[float, float]


class MeasurementData(NamedTuple):
    """
    Arbitrary, source-defined data attached to a measurement — kept apart
    from MeasurementMeta (which is purely FieldWeave's own presentation
    settings) so an importer can carry values FieldWeave doesn't interpret
    (a ring's year, a break flag) through to a later export unchanged.

    Keys are namespaced by their source (e.g. ``"dendro.year"``) so
    different importers don't collide. ``point_attrs`` is either empty
    (no per-point data) or exactly one dict per point, index-aligned with
    the measurement's points. Instances are treated as immutable: build a
    new one rather than mutating the dicts in place.
    """

    attrs: Attrs
    point_attrs: tuple[Attrs, ...]

    @property
    def is_empty(self) -> bool:
        return not self.attrs and not any(self.point_attrs)


EMPTY_DATA = MeasurementData({}, ())

# The one per-point attribute FieldWeave itself interprets: true on a point
# means the line does not connect the previous point to it, so a polyline
# splits into separate runs there.
SEGMENT_BREAK = "fw.segment_break"


def break_indices(data: MeasurementData, point_count: int) -> frozenset[int]:
    """Indices (never 0) of the points that start a new run."""
    return frozenset(
        i for i, attrs in enumerate(data.point_attrs[:point_count]) if i > 0 and attrs.get(SEGMENT_BREAK) is True
    )


def segment_runs(points: tuple[Point, ...], data: MeasurementData) -> list[tuple[Point, ...]]:
    """*points* split into the separately-connected runs *data* describes — a single run holding every point when nothing is flagged."""
    breaks = break_indices(data, len(points))
    if not breaks:
        return [points]
    runs: list[tuple[Point, ...]] = []
    start = 0
    for i in sorted(breaks):
        runs.append(points[start:i])
        start = i
    runs.append(points[start:])
    return runs


def drop_point(data: MeasurementData, index: int) -> MeasurementData:
    """*data* with the per-point entry at *index* removed, to follow a point being deleted from its measurement."""
    if not data.point_attrs:
        return data
    remaining = data.point_attrs[:index] + data.point_attrs[index + 1:]
    return MeasurementData(data.attrs, remaining)


def _is_attr_value(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    return value is None or isinstance(value, (str, int, bool))


def _clean_attrs(raw: object, where: str, warnings: list[str]) -> Attrs:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        warnings.append(f"{where}: attributes are not an object, ignored.")
        return {}
    cleaned: Attrs = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not _is_attr_value(value):
            warnings.append(f"{where}: attribute {key!r} has an unsupported value, dropped.")
            continue
        cleaned[key] = value
    return cleaned


def data_to_dict(data: MeasurementData) -> dict:
    return {"attrs": dict(data.attrs), "point_attrs": [dict(a) for a in data.point_attrs]}


def data_from_dict(raw: object, point_count: int, where: str, warnings: list[str]) -> MeasurementData:
    """
    Parse the ``"data"`` member of a serialized measurement, never raising:
    a malformed value is dropped (and reported in *warnings*) rather than
    failing the whole entry, and ``point_attrs`` of the wrong length is
    discarded since index-aligned data that no longer lines up with its
    points would silently describe the wrong ones.
    """
    if raw is None:
        return EMPTY_DATA
    if not isinstance(raw, dict):
        warnings.append(f"{where}: 'data' is not an object, ignored.")
        return EMPTY_DATA

    attrs = _clean_attrs(raw.get("attrs"), f"{where} data", warnings)

    raw_point_attrs = raw.get("point_attrs")
    point_attrs: tuple[Attrs, ...] = ()
    if raw_point_attrs:
        if not isinstance(raw_point_attrs, list) or len(raw_point_attrs) != point_count:
            warnings.append(f"{where}: per-point data does not match its {point_count} points, ignored.")
        else:
            point_attrs = tuple(
                _clean_attrs(entry, f"{where} point {i}", warnings) for i, entry in enumerate(raw_point_attrs)
            )
    return MeasurementData(attrs, point_attrs)


_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def _attr_text(value: AttrValue) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def render_point_label(template: str, index: int, data: MeasurementData) -> str:
    """
    *template* with each ``{key}`` replaced by that point's attribute of the
    same name, and ``{index}`` by its 1-based position. A missing attribute
    renders as nothing, so a point lacking the data simply gets no text
    (the caller skips a label that comes out blank). Not ``str.format``:
    attribute keys contain dots (``dendro.year``), which format would read
    as attribute access.
    """
    attrs = data.point_attrs[index] if index < len(data.point_attrs) else {}

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        return str(index + 1) if key == "index" else _attr_text(attrs.get(key))

    return _PLACEHOLDER.sub(substitute, template).strip()
