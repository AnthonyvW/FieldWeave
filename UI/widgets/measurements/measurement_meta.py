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
    # Per-secondary-tag drag offsets, as (extra index, dx, dy) triples in
    # fraction space — the multi-tag analogue of tag_offset_x/y.
    extra_offsets: tuple[tuple[int, float, float], ...] = ()
    # Tag typography and box width. Empty family / zero size mean "use the
    # overlay default"; zero tag_width means the box auto-sizes to its text.
    font_family: str = ""
    font_size: float = 0.0
    tag_width: float = 0.0
    # Deprecated: a text annotation's box transparency was folded into the
    # shared "opacity" field (which now fades the whole annotation) and its
    # transparent-background toggle. Kept only so older saved projects still
    # deserialize positionally — see the field-ordering note above.
    text_transparency: float = 0.0
    # "text"/"scalebar" annotation: the padding around the text / the scale
    # bar's inset from its anchoring corner.
    text_margin: float = 4.0
    # "scalebar" annotation: its real-world length (in meta.unit), bar
    # thickness (px), whether it's pinned to the preview (True) or the
    # image (False), a corner preset ("lower_left"/"lower_right"/
    # "upper_left"/"upper_right"/"custom"), and whether it draws a
    # background panel behind the bar and label.
    scalebar_length: float = 0.0
    scalebar_thickness: float = 4.0
    scalebar_anchor_preview: bool = True
    scalebar_position: str = "lower_left"
    scalebar_show_bg: bool = True
    # Padding inside the scale bar's background panel (between the bar/label
    # and the panel edge), separate from text_margin, which is the panel's
    # inset from its anchoring corner.
    scalebar_padding: float = 4.0
    # "text" annotation: draw the text bold. (Its content — meta.title — may
    # contain newlines for a multi-line annotation.)
    font_bold: bool = False
    # "point" category: the placed marker's shape — "dot" (filled, the
    # original look), "circle" (hollow ring), "square", "diamond",
    # "cross", "x", or "triangle" — see MeasurementOverlay._draw_point_marker.
    point_style: str = "dot"
    # "count" category: hide every point's number at once (they're drawn
    # as real text, not tags — see MeasurementOverlay._draw_count_numbers).
    # A specific point is instead removed via hover+Delete — see
    # MeasurementOverlay.delete_hovered_count_point.
    count_hide_numbers: bool = False


DEFAULT_META = MeasurementMeta()