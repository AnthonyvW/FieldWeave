from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from UI.widgets.measurements.measurement_data import EMPTY_DATA, MeasurementData, drop_point
from UI.widgets.measurements.measurement_io import (
    deserialize_measurements,
    load_measurements_from_file,
    save_measurements_to_file,
    serialize_measurements,
)
from UI.widgets.measurements.measurement_kind import DEFAULT_REGISTRY
from UI.widgets.measurements.measurement_meta import DEFAULT_META
from UI.widgets.preview_overlay.measurement_overlay import Measurement

POINTS = ((0.1, 0.2), (0.3, 0.2), (0.5, 0.2))


def _line(data: MeasurementData = EMPTY_DATA) -> Measurement:
    return Measurement("Arbitrary Line", POINTS, DEFAULT_META, data)


def _ring_data() -> MeasurementData:
    return MeasurementData(
        {"dendro.sample_index": 92, "dendro.forward": False},
        (
            {"dendro.start": True},
            {"dendro.year": 2018, "dendro.earlywood": True},
            {"dendro.year": 2017, "dendro.earlywood": True},
        ),
    )


def test_data_round_trips_through_a_document():
    doc = serialize_measurements([_line(_ring_data())])
    result = deserialize_measurements(json.loads(json.dumps(doc)), DEFAULT_REGISTRY)

    assert result.warnings == []
    (_, points, _, data), = result.entries
    assert points == POINTS
    assert data == _ring_data()


def test_data_round_trips_through_a_file(tmp_path):
    path = tmp_path / "m.json"
    save_measurements_to_file(path, [_line(_ring_data())])

    result = load_measurements_from_file(path, DEFAULT_REGISTRY)

    assert result.warnings == []
    assert result.entries[0][3] == _ring_data()


def test_empty_data_is_not_written():
    doc = serialize_measurements([_line()])

    assert "data" not in doc["measurements"][0]


def test_entry_without_data_loads_as_empty():
    doc = {"measurements": [{"kind": "Arbitrary Line", "points": [list(p) for p in POINTS]}]}

    result = deserialize_measurements(doc, DEFAULT_REGISTRY)

    assert result.warnings == []
    assert result.entries[0][3] == EMPTY_DATA


def test_mismatched_point_attrs_are_dropped_but_the_measurement_loads():
    doc = serialize_measurements([_line(_ring_data())])
    doc["measurements"][0]["data"]["point_attrs"].pop()

    result = deserialize_measurements(doc, DEFAULT_REGISTRY)

    assert len(result.entries) == 1
    assert result.entries[0][3].point_attrs == ()
    assert result.entries[0][3].attrs == _ring_data().attrs
    assert any("does not match" in w for w in result.warnings)


def test_unsupported_attr_values_are_dropped_with_a_warning():
    doc = serialize_measurements([_line()])
    doc["measurements"][0]["data"] = {
        "attrs": {"ok": 1, "nested": {"a": 1}, "listy": [1], "inf": float("inf")},
        "point_attrs": [],
    }

    result = deserialize_measurements(doc, DEFAULT_REGISTRY)

    assert result.entries[0][3].attrs == {"ok": 1}
    assert len(result.warnings) == 3


def test_malformed_data_does_not_fail_the_entry():
    doc = serialize_measurements([_line()])
    doc["measurements"][0]["data"] = "nonsense"

    result = deserialize_measurements(doc, DEFAULT_REGISTRY)

    assert len(result.entries) == 1
    assert result.entries[0][3] == EMPTY_DATA
    assert result.warnings


def test_drop_point_keeps_per_point_data_aligned():
    trimmed = drop_point(_ring_data(), 1)

    assert trimmed.point_attrs == ({"dendro.start": True}, {"dendro.year": 2017, "dendro.earlywood": True})
    assert trimmed.attrs == _ring_data().attrs


def test_drop_point_without_per_point_data_is_a_no_op():
    data = MeasurementData({"a": 1}, ())

    assert drop_point(data, 0) is data


def test_deleting_a_count_point_drops_its_per_point_data():
    from PySide6.QtWidgets import QApplication

    from UI.widgets.preview_overlay.measurement_overlay import MeasurementOverlay

    QApplication.instance() or QApplication([])
    overlay = MeasurementOverlay()
    overlay.measurements.append(Measurement("Count", POINTS, DEFAULT_META, _ring_data()))
    overlay._near_index = 0
    overlay._near_point_index = 1

    assert overlay.delete_hovered_count_point()

    remaining = overlay.measurements[0]
    assert remaining.points == (POINTS[0], POINTS[2])
    assert remaining.data.point_attrs == (
        {"dendro.start": True},
        {"dendro.year": 2017, "dendro.earlywood": True},
    )
    assert remaining.data.attrs == _ring_data().attrs
