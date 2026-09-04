from __future__ import annotations

from typing import NamedTuple

from UI.widgets.measurements.units import MeasurementUnit

# Doesn't resolve to anything yet — it exists so many measurements can
# later share one MeasurementStyle by id instead of each serializing its
# own copy of the same style.
DEFAULT_STYLE_ID = "default"


class MeasurementMeta(NamedTuple):
    """
    User-facing customization for a single placed measurement: its
    title, a free-form description, an optional unit override (falls
    back to MeasurementsWidget's global unit when None), and appearance
    overrides for its tag and line — each empty/zero value means "use
    the overlay's default" (see measurement_style.py). Colors are hex
    strings rather than QColor so this stays a plain, directly
    serializable tuple. Kept apart from Measurement's points so geometry
    and customization each serialize independently.

    outline_enabled turns the outline pass off entirely rather than
    just zeroing outline_thickness — a thickness of 0.0 already means
    "use the overlay's default" (see above), so it can't double as
    "disabled" too without losing that meaning; disabling needs its own
    flag instead. Placed last, after every pre-existing field, rather
    than grouped with the other outline_* fields above — MeasurementMeta
    values may get built positionally elsewhere (e.g. loading a saved
    project), and inserting a new field in the middle would silently
    shift every field after it into the wrong slot. New fields belong
    at the end.
    """

    title: str = ""
    description: str = ""
    unit: MeasurementUnit | None = None
    style_id: str = DEFAULT_STYLE_ID
    tag_background_color: str = ""
    tag_text_color: str = ""
    always_show_description: bool = False
    line_color: str = ""
    line_thickness: float = 0.0
    outline_color: str = ""
    outline_thickness: float = 0.0
    line_dash_style: str = "solid"
    line_start_cap: str = "curved"
    line_end_cap: str = "curved"
    outline_enabled: bool = True
    decimal_places: int = 2
    hidden: bool = False
    opacity: float = 1.0
    tag_background_transparent: bool = False
    midpoint_style: str = "none"
    show_area: bool = False
    area_unit: MeasurementUnit | None = None
    tag_offset_x: float = 0.0
    tag_offset_y: float = 0.0
    # Multiplier on the arrow/bracket caps' base size (see
    # UI.widgets.measurements.lines.arrow_dims) — one shared "size" knob
    # for both, rather than a separate control per cap style.
    cap_size_scale: float = 1.0
    # Circle/ellipse/"Radius Arc" kinds: draw a small marker at the
    # shape's own center at all times, not just while hovered/dragged
    # like its other anchor points.
    always_show_center: bool = False
    # "3 Point Angle"/"4 Point Angle" only, default off: each leg's own
    # length alongside the angle value.
    show_leg_lengths: bool = False
    # The secondary "indicator" line shared by every measurement that has
    # one — the parallel/perpendicular dimension connectors and the angle
    # indicator's dashed guide + curve. On by default (so an angle shows
    # its indicator and a parallel shows its connectors without extra
    # clicks); fully customizable — color, opacity, dash style, on/off.
    indicator_enabled: bool = True
    indicator_color: str = ""
    indicator_opacity: float = 1.0
    indicator_dash_style: str = "dash"
    # Enclosed shapes (circle/ellipse/rectangle/polygon/annulus): a
    # translucent interior fill. Empty means no fill; fill_opacity scales
    # the fill's alpha independently of the whole-measurement opacity.
    fill_color: str = ""
    fill_opacity: float = 0.3
    # Indices (into a kind's extra_measures list) of secondary tags the
    # user has dismissed by hovering one and pressing Delete — see
    # MeasurementOverlay._draw_extra_measure_labels.
    hidden_extra: tuple[int, ...] = ()


DEFAULT_META = MeasurementMeta()