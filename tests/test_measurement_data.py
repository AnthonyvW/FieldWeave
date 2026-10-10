from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF

from UI.widgets.measurements.measurement_data import (
    EMPTY_DATA,
    SEGMENT_BREAK,
    MeasurementData,
    break_indices,
    drop_point,
    render_point_label,
    segment_runs,
)
from UI.widgets.measurements.measurement_io import (
    deserialize_measurements,
    load_measurements_from_file,
    save_measurements_to_file,
    serialize_measurements,
)
from UI.widgets.measurements.measurement_kind import DEFAULT_REGISTRY
from UI.widgets.measurements.measurement_meta import DEFAULT_META
from UI.widgets.measurements.units import MeasurementUnit
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


def _broken_line_data() -> MeasurementData:
    return MeasurementData({}, ({}, {}, {SEGMENT_BREAK: True}, {}))


BROKEN_POINTS = ((0.05, 0.5), (0.25, 0.5), (0.75, 0.5), (0.95, 0.5))


def _overlay():
    from PySide6.QtWidgets import QApplication

    from UI.widgets.preview_overlay.measurement_overlay import MeasurementOverlay

    QApplication.instance() or QApplication([])
    return MeasurementOverlay()


def test_segment_runs_split_at_flagged_points():
    runs = segment_runs(BROKEN_POINTS, _broken_line_data())

    assert runs == [BROKEN_POINTS[:2], BROKEN_POINTS[2:]]


def test_segment_runs_without_breaks_is_one_run():
    assert segment_runs(BROKEN_POINTS, EMPTY_DATA) == [BROKEN_POINTS]


def test_break_on_the_first_point_and_non_true_values_are_ignored():
    data = MeasurementData({}, ({SEGMENT_BREAK: True}, {SEGMENT_BREAK: False}, {SEGMENT_BREAK: 1}, {}))

    assert break_indices(data, 4) == frozenset()


def test_line_length_excludes_the_gap_between_runs():
    overlay = _overlay()
    overlay.set_live_dpi(25.4)

    whole = overlay._length_suffix(BROKEN_POINTS, (1000, 100), MeasurementUnit.PX)
    broken = overlay._length_suffix(BROKEN_POINTS, (1000, 100), MeasurementUnit.PX, 2, _broken_line_data())

    assert whole == "900.00 px"
    assert broken == "400.00 px"


def test_hit_test_ignores_the_gap_between_runs():
    from PySide6.QtCore import QPoint, QRect

    from UI.widgets.preview_overlay.coordinate_space import CoordinateSpace

    class _Frame(CoordinateSpace):
        def __init__(self) -> None:
            pass

        def current_frame_dims(self):
            return (1000, 100)

    overlay = _overlay()
    overlay._zoom_handler = _Frame()
    overlay._screen_point = lambda p, rect, widget_rect: p
    overlay._to_point = lambda rect, p: QPointF(p[0] * 1000, p[1] * 100)
    overlay.measurements.append(Measurement("Arbitrary Line", BROKEN_POINTS, DEFAULT_META, _broken_line_data()))
    rect = QRect(0, 0, 1000, 100)

    assert overlay._hit_test_proximity(QPoint(150, 50), rect, rect) == 0
    assert overlay._hit_test_proximity(QPoint(500, 50), rect, rect) is None


def test_draw_leaves_the_gap_undrawn():
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QImage, QPainter

    overlay = _overlay()
    image = QImage(1000, 100, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    overlay._draw_measurement(
        painter, QRect(0, 0, 1000, 100), "Arbitrary Line", BROKEN_POINTS, 1.0, 1.0, 1.0, None,
        data=_broken_line_data(),
    )
    painter.end()

    assert image.pixelColor(150, 50).alpha() > 0
    assert image.pixelColor(850, 50).alpha() > 0
    assert image.pixelColor(500, 50).alpha() == 0


def test_point_label_fills_attributes_and_index():
    data = MeasurementData({}, ({}, {"dendro.year": 2018}, {"dendro.year": 2017.0, "x": True}))

    assert render_point_label("{dendro.year}", 1, data) == "2018"
    assert render_point_label("{index}: {dendro.year}", 2, data) == "3: 2017"
    assert render_point_label("{x}", 2, data) == "true"


def test_point_label_is_blank_when_the_attribute_is_missing():
    data = MeasurementData({}, ({}, {"dendro.year": 2018}))

    assert render_point_label("{dendro.year}", 0, data) == ""
    assert render_point_label("{dendro.year}", 5, data) == ""
    assert render_point_label("{dendro.year}", 0, EMPTY_DATA) == ""


def test_point_label_template_round_trips_in_meta():
    meta = DEFAULT_META._replace(point_label_template="{dendro.year}")
    doc = serialize_measurements([Measurement("Arbitrary Line", POINTS, meta)])

    result = deserialize_measurements(json.loads(json.dumps(doc)), DEFAULT_REGISTRY)

    assert result.entries[0][2].point_label_template == "{dendro.year}"


def _draw_labels(kind: str, template: str, data: MeasurementData):
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QImage, QPainter

    overlay = _overlay()
    meta = DEFAULT_META._replace(point_label_template=template, tag_text_color="#ff0000")
    measurement = Measurement(kind, POINTS, meta, data)
    image = QImage(1000, 100, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    overlay._draw_measurement_label(painter, QRect(0, 0, 1000, 100), 0, measurement, 1.0, 1.0, None)
    painter.end()
    return image


def _has_ink(image, x_center: int) -> bool:
    return any(
        image.pixelColor(x, y).alpha() > 0 for x in range(x_center, x_center + 50) for y in range(0, 60)
    )


def test_line_draws_labels_only_for_points_that_have_the_data():
    data = MeasurementData({}, ({}, {"dendro.year": 2018}, {}))

    image = _draw_labels("Arbitrary Line", "{dendro.year}", data)

    assert not _has_ink(image, 100)
    assert _has_ink(image, 300)
    assert not _has_ink(image, 500)


def test_line_without_a_template_draws_no_point_labels():
    data = MeasurementData({}, ({}, {"dendro.year": 2018}, {}))

    image = _draw_labels("Arbitrary Line", "", data)

    assert not any(_has_ink(image, x) for x in (100, 300, 500))


def test_count_falls_back_to_numbers_without_a_template_and_uses_it_with_one():
    data = MeasurementData({}, ({}, {"dendro.year": 2018}, {}))

    numbered = _draw_labels("Count", "", data)
    templated = _draw_labels("Count", "{dendro.year}", data)

    assert all(_has_ink(numbered, x) for x in (100, 300, 500))
    assert not _has_ink(templated, 100)
    assert _has_ink(templated, 300)


def test_customize_menu_shows_and_edits_the_point_label_template():
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication

    from UI.widgets.preview_overlay.measurement_customize_menu import MeasurementCustomizeMenu

    QApplication.instance() or QApplication([])
    menu = MeasurementCustomizeMenu()
    menu.open_for(0, "Arbitrary Line", DEFAULT_META._replace(point_label_template="{index}"), QPoint(0, 0))

    assert menu._point_label_edit.text() == "{index}"
    assert not menu._point_label_edit.isHidden()

    menu._point_label_edit.setText("{dendro.year}")

    assert menu._current_meta().point_label_template == "{dendro.year}"

    menu.open_for(0, "Radius Circle", DEFAULT_META, QPoint(0, 0))

    assert menu._point_label_edit.isHidden()
