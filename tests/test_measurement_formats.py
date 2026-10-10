from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from UI.widgets.measurements import dendroelevator_io
from UI.widgets.measurements.measurement_data import SEGMENT_BREAK, MeasurementData
from UI.widgets.measurements.measurement_formats import (
    DENDROELEVATOR,
    FIELDWEAVE,
    detect_format,
    format_for_filter,
    read_measurement_file,
    write_measurement_file,
)
from UI.widgets.measurements.measurement_io import FormatContext, serialize_measurements
from UI.widgets.measurements.measurement_kind import DEFAULT_REGISTRY
from UI.widgets.measurements.measurement_meta import DEFAULT_META
from UI.widgets.preview_overlay.measurement_overlay import Measurement
from common.fieldweaveConfig import FIELDWEAVE_VERSION

WIDTH, HEIGHT = 10000, 400
SCALE = max(WIDTH, HEIGHT)
PPM = 100.0
DPI = PPM * 25.4
RATIO = 254 / 256
CTX = FormatContext((WIDTH, HEIGHT), DPI)

# (x px, y px, extra fields) — laid out so every expected width is a round
# number: 2018 and 2017 are plain 1.0 and 2.0 mm, and 2016 crosses a core
# break whose two measured stretches (1.0 mm each) sum to 2.0 mm with the
# jumps into each `start` point left out.
_PIXELS = [
    (9000, 100, {"start": True}),
    (8900, 100, {"year": 2018, "earlywood": True}),
    (8700, 100, {"year": 2017, "earlywood": True}),
    (8000, 100, {"start": True}),
    (7900, 100, {"break": True}),
    (7500, 100, {"start": True}),
    (7400, 100, {"year": 2016, "earlywood": True}),
]


def _lat_lng(x: float, y: float, side: float = SCALE) -> dict:
    return {"lat": -y / (side * RATIO), "lng": x / (side * RATIO)}


def _dendro_doc() -> dict:
    points = []
    for x, y, extra in _PIXELS:
        point = {"skip": False, "break": False, "start": False, "latLng": _lat_lng(x, y)}
        point.update(extra)
        points.append(point)
    return {
        "SaveDate": {"day": 13, "hour": 9, "year": 2024, "month": 8, "minute": 49},
        "year": 2015,
        "forwardDirection": False,
        "subAnnual": False,
        "earlywood": True,
        "index": 92,
        "points": points,
        "annotations": {
            "0": {
                "code": ["x"],
                "text": "Marker ring",
                "year": 2017,
                "color": "#00ff00",
                "latLng": _lat_lng(8800, 300),
                "description": [],
                "calculatedYear": 2017,
                "yearAdjustment": 0,
            }
        },
        "ppm": PPM / RATIO,
        "ptWidths": {"tw": {"x": [2017, 2018, 2019], "y": [2.0, 2.0, 1.0], "name": "My Sample_tw"}},
        "ellipses": [],
        "currentView": {"brightness": "120", "invert": True},
        "pithEstimate": {"yearEstimate": 1990},
        "futureField": {"nested": [1, 2]},
    }


def _import(doc: dict | None = None, ctx: FormatContext = CTX):
    return dendroelevator_io.load(doc or _dendro_doc(), ctx, DEFAULT_REGISTRY)


def _as_measurements(result) -> list[Measurement]:
    return [Measurement(*entry) for entry in result.entries]


def _rings(result):
    return [e for e in result.entries if e[0] == "Arbitrary Line"]


def test_import_makes_one_ordinary_line_per_ring():
    result = _import()

    assert [e[0] for e in result.entries] == ["Arbitrary Line"] * 3 + ["Point"]
    assert all(kind in DEFAULT_REGISTRY for kind, *_ in result.entries)
    assert [e[3].attrs["dendro.ring_year"] for e in _rings(result)] == [2019, 2018, 2017]
    assert result.dpi == pytest.approx(DPI)
    assert result.warnings == []


def test_each_ring_runs_between_its_two_year_points_and_is_titled_with_its_year():
    first, second, third = _rings(_import())

    flat = lambda points: [c for point in points for c in point]
    assert flat(first[1]) == pytest.approx([9000 / WIDTH, 100 / HEIGHT, 8900 / WIDTH, 100 / HEIGHT])
    assert flat(second[1]) == pytest.approx([8900 / WIDTH, 100 / HEIGHT, 8700 / WIDTH, 100 / HEIGHT])
    assert [m[2].title for m in (first, second, third)] == ["2019", "2018", "2017"]
    assert first[2].unit.value == "mm"


def test_a_ring_crossing_a_core_break_keeps_its_runs_apart():
    ring = _rings(_import())[2]

    flagged = [i for i, attrs in enumerate(ring[3].point_attrs) if attrs.get(SEGMENT_BREAK)]
    assert len(ring[1]) == 5
    assert flagged == [1, 3]
    assert ring[3].point_attrs[2]["dendro.break"] is True
    assert ring[3].point_attrs[0]["dendro.year"] == 2017


def test_rings_carry_the_sample_wide_data():
    attrs = _rings(_import())[1][3].attrs

    assert attrs["dendro.index"] == 92
    assert attrs["dendro.forward_direction"] is False
    assert attrs["dendro.name"] == "My Sample"


def test_decade_rings_are_red_and_the_rest_blue():
    doc = _dendro_doc()
    doc["points"][2]["year"] = 2019

    first, second, third = _rings(_import(doc))

    assert first[3].attrs["dendro.ring_year"] == 2019
    assert (first[2].line_color, first[2].tag_background_color) == ("#1c7bff", "#1c7bff")
    assert second[3].attrs["dendro.ring_year"] == 2020
    assert (second[2].line_color, second[2].tag_background_color) == ("#ff1c22", "#ff1c22")


def test_measuring_the_other_way_shifts_each_rings_year_down():
    doc = _dendro_doc()
    doc["forwardDirection"] = True

    ring_years = [e[3].attrs["dendro.ring_year"] for e in _rings(_import(doc))]

    assert ring_years == [2017, 2016, 2015]


def test_import_maps_annotations_to_points_labelled_with_their_year():
    kind, points, meta, data = _import().entries[3]

    assert kind == "Point"
    assert points[0] == pytest.approx((8800 / WIDTH, 300 / HEIGHT))
    assert meta.title == "2017"
    assert meta.description == "Marker ring"
    assert meta.line_color == "#00ff00"
    assert meta.tag_background_color == "#00ff00"
    assert data.attrs["dendro.year"] == 2017


def test_import_needs_the_image_size():
    result = _import(ctx=FormatContext(None, None))

    assert result.entries == []
    assert result.warnings


def test_map_pixels_run_a_tile_ratio_larger_than_image_pixels():
    result = _import()

    assert result.dpi == pytest.approx(DPI)
    first_ring_end = result.entries[0][1][1]
    assert first_ring_end[0] * WIDTH == pytest.approx(8900)


def test_import_takes_its_scale_from_the_file_not_the_image_size():
    result = _import(ctx=FormatContext((12000, HEIGHT), DPI))

    kind, points, meta, data = result.entries[0]
    assert data.attrs["dendro.scale"] == pytest.approx(SCALE)
    assert points[1] == pytest.approx((8900 / 12000, 100 / HEIGHT))
    assert result.warnings == []


def test_import_snaps_a_scale_near_a_power_of_two_to_it():
    doc = _dendro_doc()
    side = 8192
    for item in [p["latLng"] for p in doc["points"]] + [doc["annotations"]["0"]["latLng"]]:
        item["lat"] *= SCALE / side
        item["lng"] *= SCALE / side
    doc["ppm"] = PPM / RATIO * 1.0004

    result = _import(doc, FormatContext((12000, HEIGHT), DPI))

    assert result.entries[0][3].attrs["dendro.scale"] == 8192


def test_import_into_an_image_narrower_than_the_files_scale_keeps_pixel_positions():
    side = 131072
    doc = _dendro_doc()
    for point, (x, y, _) in zip(doc["points"], _PIXELS):
        point["latLng"] = _lat_lng(x, y, side)
    doc["annotations"] = {}

    result = _import(doc, FormatContext((119747, 5810), DPI))

    assert result.entries[0][3].attrs["dendro.scale"] == side
    assert result.entries[0][1][1] == pytest.approx((8900 / 119747, 100 / 5810))


def test_import_without_ring_widths_assumes_the_next_power_of_two():
    doc = _dendro_doc()
    del doc["ptWidths"]

    result = _import(doc, FormatContext((12000, HEIGHT), DPI))

    assert result.entries[0][3].attrs["dendro.scale"] == 16384
    assert any("assumed" in w for w in result.warnings)


def test_import_warns_when_points_fall_outside_the_image():
    result = _import(ctx=FormatContext((5000, HEIGHT), DPI))

    assert any("outside the loaded" in w for w in result.warnings)


def test_import_skips_points_without_coordinates_and_keeps_each_rings_data_aligned():
    doc = _dendro_doc()
    del doc["points"][2]["latLng"]

    result = _import(doc)

    assert any("Point 2" in w for w in result.warnings)
    for kind, points, meta, data in _rings(result):
        assert len(points) == len(data.point_attrs)


def test_points_after_the_last_year_become_a_ring_without_a_year():
    doc = _dendro_doc()
    doc["points"].append({"skip": False, "break": False, "start": False, "latLng": _lat_lng(7300, 100)})

    rings = _rings(_import(doc))

    assert len(rings) == 4
    assert "dendro.ring_year" not in rings[3][3].attrs
    assert rings[3][2].title == ""


def test_round_trip_reproduces_the_file():
    original = _dendro_doc()
    measurements = _as_measurements(_import(original))

    exported = dendroelevator_io.save(measurements, CTX)

    assert exported.warnings == []
    out = exported.document
    assert out["ppm"] == pytest.approx(PPM / RATIO)
    assert out["ptWidths"]["tw"]["x"] == [2017, 2018, 2019]
    assert out["ptWidths"]["tw"]["y"] == pytest.approx([2.0, 2.0, 1.0])
    assert out["ptWidths"]["tw"]["name"] == "My Sample_tw"
    assert [(p["start"], p["break"], p.get("year")) for p in out["points"]] == [
        (p["start"], p["break"], p.get("year")) for p in original["points"]
    ]
    for got, want in zip(out["points"], original["points"]):
        assert got["latLng"]["lat"] == pytest.approx(want["latLng"]["lat"])
        assert got["latLng"]["lng"] == pytest.approx(want["latLng"]["lng"])
    for key in ("year", "forwardDirection", "subAnnual", "earlywood", "index", "currentView", "pithEstimate", "futureField"):
        assert out[key] == original[key]
    annotation = out["annotations"]["0"]
    assert annotation["text"] == "Marker ring"
    assert annotation["color"] == "#00ff00"
    assert annotation["year"] == 2017
    assert annotation["code"] == ["x"]
    assert annotation["calculatedYear"] == 2017


def test_export_recomputes_widths_after_a_point_moves():
    measurements = _as_measurements(_import())
    ring = measurements[1]
    moved = list(ring.points)
    moved[1] = (8600 / WIDTH, 100 / HEIGHT)
    measurements[1] = ring._replace(points=tuple(moved))

    out = dendroelevator_io.save(measurements, CTX).document

    widths = dict(zip(out["ptWidths"]["tw"]["x"], out["ptWidths"]["tw"]["y"]))
    assert widths[2018] == pytest.approx(3.0)
    assert widths[2017] == pytest.approx(2.0)


def test_export_keeps_the_points_of_the_rings_that_remain():
    measurements = _as_measurements(_import())
    del measurements[1]

    out = dendroelevator_io.save(measurements, CTX).document

    assert [p.get("year") for p in out["points"]] == [None, 2018, 2017, None, None, None, 2016]
    assert out["ptWidths"]["tw"]["x"] == [2017, 2019]


def test_export_marks_a_break_made_in_fieldweave_as_a_start():
    attrs = {"dendro.ring_index": 0, "dendro.ring_year": 2018}
    point_attrs = tuple({"dendro.index": i} for i in range(4))
    point_attrs = point_attrs[:2] + ({"dendro.index": 2, SEGMENT_BREAK: True},) + point_attrs[3:]
    ring = Measurement(
        "Arbitrary Line", ((0.1, 0.5), (0.2, 0.5), (0.4, 0.5), (0.5, 0.5)), DEFAULT_META, MeasurementData(attrs, point_attrs)
    )

    out = dendroelevator_io.save([ring], CTX).document

    assert [p["start"] for p in out["points"]] == [False, False, True, False]


def test_export_without_ring_lines_writes_nothing():
    plain = Measurement("Arbitrary Line", ((0.1, 0.5), (0.2, 0.5)), DEFAULT_META)

    result = dendroelevator_io.save([plain], CTX)

    assert result.document is None
    assert result.warnings


def test_export_needs_a_dpi_unless_the_file_recorded_one():
    measurements = _as_measurements(_import())
    no_dpi = FormatContext((WIDTH, HEIGHT), None)

    reused = dendroelevator_io.save(measurements, no_dpi)

    assert reused.document["ppm"] == pytest.approx(PPM / RATIO)
    assert any("reused" in w for w in reused.warnings)

    stripped = [
        m._replace(data=MeasurementData({k: v for k, v in m.data.attrs.items() if k != "dendro.ppm"}, m.data.point_attrs))
        if m.kind == "Arbitrary Line" else m
        for m in measurements
    ]

    assert dendroelevator_io.save(stripped, no_dpi).document is None


def test_export_skips_points_without_a_year():
    measurements = _as_measurements(_import())
    measurements.append(Measurement("Point", ((0.5, 0.5),), DEFAULT_META._replace(title="note")))

    result = dendroelevator_io.save(measurements, CTX)

    assert len(result.document["annotations"]) == 1
    assert any("without a year" in w for w in result.warnings)


def test_fieldweave_files_start_with_a_header_naming_the_app_version():
    document = serialize_measurements([])

    assert list(document)[0] == "fieldweave"
    assert document["fieldweave"]["type"] == "measurements"
    assert document["fieldweave"]["app_version"] == FIELDWEAVE_VERSION
    assert list(json.loads(json.dumps(document)))[0] == "fieldweave"


def test_format_detection():
    assert detect_format(serialize_measurements([])) is FIELDWEAVE
    assert detect_format({"measurements": []}) is FIELDWEAVE
    assert detect_format({"fieldweave": {"type": "measurements"}, "measurements": []}) is FIELDWEAVE
    assert detect_format(_dendro_doc()) is DENDROELEVATOR
    assert detect_format({"something": "else"}) is None
    assert detect_format([1, 2]) is None


def test_save_dialog_filters_pick_the_format():
    assert format_for_filter(DENDROELEVATOR.file_filter) is DENDROELEVATOR
    assert format_for_filter(FIELDWEAVE.file_filter) is FIELDWEAVE
    assert format_for_filter("anything else") is FIELDWEAVE


def test_read_handles_a_byte_order_mark_and_each_format(tmp_path):
    dendro = tmp_path / "d.json"
    dendro.write_bytes(b"\xef\xbb\xbf" + json.dumps(_dendro_doc()).encode())
    native = tmp_path / "f.json"
    native.write_bytes(b"\xef\xbb\xbf" + json.dumps(serialize_measurements(_as_measurements(_import()))).encode())

    from_dendro = read_measurement_file(dendro, DEFAULT_REGISTRY, CTX)
    from_native = read_measurement_file(native, DEFAULT_REGISTRY, CTX)

    assert [e[0] for e in from_dendro.entries] == ["Arbitrary Line"] * 3 + ["Point"]
    assert [e[0] for e in from_native.entries] == ["Arbitrary Line"] * 3 + ["Point"]
    assert [e[3] for e in from_native.entries] == [e[3] for e in from_dendro.entries]


def test_read_reports_unreadable_and_unrecognized_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    other = tmp_path / "other.json"
    other.write_text('{"hello": 1}')

    assert read_measurement_file(bad, DEFAULT_REGISTRY, CTX).warnings
    assert "not a recognized" in read_measurement_file(other, DEFAULT_REGISTRY, CTX).warnings[0]
    assert read_measurement_file(tmp_path / "missing.json", DEFAULT_REGISTRY, CTX).warnings


def test_write_measurement_file_round_trips_through_disk(tmp_path):
    measurements = _as_measurements(_import())
    path = tmp_path / "out.json"

    result = write_measurement_file(path, DENDROELEVATOR, measurements, CTX)
    reread = read_measurement_file(path, DEFAULT_REGISTRY, CTX)

    assert result.document is not None
    assert reread.warnings == []
    flat = lambda points: [c for point in points for c in point]
    assert flat(reread.entries[0][1]) == pytest.approx(flat(_import().entries[0][1]))


def test_write_measurement_file_leaves_no_file_when_nothing_can_be_exported(tmp_path):
    path = tmp_path / "out.json"

    result = write_measurement_file(path, DENDROELEVATOR, [], CTX)

    assert result.document is None
    assert not path.exists()


def _controller():
    from PySide6.QtWidgets import QApplication

    from UI.widgets.preview_overlay.coordinate_space import IdentityCoordinateSpace
    from UI.widgets.preview_overlay.measurement_overlay import MeasurementOverlay, MeasurementOverlayController

    QApplication.instance() or QApplication([])
    overlay = MeasurementOverlay()
    overlay._zoom_handler = IdentityCoordinateSpace((WIDTH, HEIGHT))
    overlay.set_live_dpi(DPI)
    return MeasurementOverlayController(overlay, lambda: None), overlay


def test_controller_import_replaces_current_measurements(tmp_path):
    controller, overlay = _controller()
    overlay.measurements.append(Measurement("Point", ((0.5, 0.5),)))
    path = tmp_path / "d.json"
    path.write_text(json.dumps(_dendro_doc()))

    result = controller.import_measurements_from_file(str(path))

    assert [m.kind for m in overlay.measurements] == ["Arbitrary Line"] * 3 + ["Point"]
    assert result.warnings == []


def test_controller_import_of_a_bad_file_keeps_current_measurements(tmp_path):
    controller, overlay = _controller()
    overlay.measurements.append(Measurement("Point", ((0.5, 0.5),)))
    path = tmp_path / "bad.json"
    path.write_text('{"hello": 1}')

    result = controller.import_measurements_from_file(str(path))

    assert [m.kind for m in overlay.measurements] == ["Point"]
    assert result.warnings


def test_controller_exports_in_the_chosen_format(tmp_path):
    controller, overlay = _controller()
    overlay.measurements.extend(_as_measurements(_import()))
    native = tmp_path / "n.json"
    foreign = tmp_path / "d.json"

    controller.export_measurements_to_file(str(native))
    controller.export_measurements_to_file(str(foreign), DENDROELEVATOR)

    assert list(json.loads(native.read_text()))[0] == "fieldweave"
    assert "ptWidths" in json.loads(foreign.read_text())


SAMPLE = os.environ.get("FIELDWEAVE_DENDRO_SAMPLE")


@pytest.mark.skipif(not SAMPLE, reason="set FIELDWEAVE_DENDRO_SAMPLE to a real Dendroelevator export to run this")
def test_real_dendroelevator_file_round_trips():
    original = json.loads(Path(SAMPLE).read_text(encoding="utf-8-sig"))
    ctx = FormatContext((119747, 5810), original["ppm"] * 25.4 * (254 / 256))

    result = dendroelevator_io.load(copy.deepcopy(original), ctx, DEFAULT_REGISTRY)
    assert result.warnings == []
    assert result.entries[0][3].attrs["dendro.scale"] == 131072
    out = dendroelevator_io.save(_as_measurements(result), ctx).document

    assert len(out["points"]) == len(original["points"])
    for got, want in zip(out["points"], original["points"]):
        assert (got["start"], got["break"], got["skip"], got.get("year")) == (
            want["start"], want["break"], want["skip"], want.get("year")
        )
        assert got["latLng"]["lat"] == pytest.approx(want["latLng"]["lat"], abs=1e-12)
        assert got["latLng"]["lng"] == pytest.approx(want["latLng"]["lng"], abs=1e-12)
    assert out["ptWidths"]["tw"]["x"] == original["ptWidths"]["tw"]["x"]
    assert out["ptWidths"]["tw"]["y"] == pytest.approx(original["ptWidths"]["tw"]["y"], abs=2e-5)
    assert out["ptWidths"]["tw"]["name"] == original["ptWidths"]["tw"]["name"]
    for key in ("year", "forwardDirection", "subAnnual", "earlywood", "index", "currentView", "pithEstimate", "ellipses"):
        assert out[key] == original[key]
    assert out["annotations"]["0"]["text"] == original["annotations"]["0"]["text"]
    assert out["annotations"]["0"]["latLng"]["lng"] == pytest.approx(original["annotations"]["0"]["latLng"]["lng"])
