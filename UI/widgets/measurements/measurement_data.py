from __future__ import annotations

import math
from typing import NamedTuple

AttrValue = str | int | float | bool | None
Attrs = dict[str, AttrValue]


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
