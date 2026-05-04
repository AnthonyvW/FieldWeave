from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from machine_vision.algorithms.camera_calibration import CameraCalibration

# All Position coordinates are stored internally in nanometers.
# 1 mm = 1_000_000 nm
_NM_PER_MM = 1_000_000
_NM_PER_TICK = 10_000


@dataclass
class Position:
    x: int  # nanometers
    y: int  # nanometers
    z: int  # nanometers

    def to_gcode(self) -> str:
        """Convert position to G-code coordinates (millimetres, 6 decimal places)."""
        return (
            f"X{self.x / _NM_PER_MM:.6f}"
            f" Y{self.y / _NM_PER_MM:.6f}"
            f" Z{self.z / _NM_PER_MM:.6f}"
        )

    @classmethod
    def from_mm(cls, x: float, y: float, z: float) -> Position:
        """Construct a Position from millimetre values."""
        return cls(
            x=round(x * _NM_PER_MM),
            y=round(y * _NM_PER_MM),
            z=round(z * _NM_PER_MM),
        )

    def to_mm(self) -> tuple[float, float, float]:
        """Return (x, y, z) in millimetres."""
        return (
            self.x / _NM_PER_MM,
            self.y / _NM_PER_MM,
            self.z / _NM_PER_MM,
        )


class FocusScore(Enum):
    GOOD = "GOOD"
    MODERATE = "MODERATE"
    POOR = "POOR"


def pixels_to_stage_delta(
    calibration: CameraCalibration,
    sensor_pixel_x: float,
    sensor_pixel_y: float,
    sensor_w: int,
    sensor_h: int,
) -> tuple[int, int]:
    """Convert a sensor pixel coordinate to a stage delta in nanometres.

    Returns the (dx_nm, dy_nm) move required to bring the given pixel to the
    image centre, accounting for any difference between sensor resolution and
    the calibration image resolution.
    """
    cal_w = float(calibration.image_width)
    cal_h = float(calibration.image_height)
    cal_x = sensor_pixel_x * (cal_w / sensor_w)
    cal_y = sensor_pixel_y * (cal_h / sensor_h)
    dx_ticks, dy_ticks = calibration.pixel_to_world_delta(cal_x, cal_y)
    return int(round(dx_ticks * _NM_PER_TICK)), int(round(dy_ticks * _NM_PER_TICK))


def distance_to_stage_delta(
    calibration: CameraCalibration,
    distance_nm: int,
    axis: str,
    sensor_w: int,
    sensor_h: int,
) -> tuple[int, int]:
    """Convert a physical distance in nanometres to a stage delta.

    Uses the calibration scale factor derived from a full-edge pixel mapping so
    that the returned delta (dx_nm, dy_nm) represents the requested physical
    distance along the given axis (``'x'`` or ``'y'``).
    """
    cal_w = float(calibration.image_width)
    cal_h = float(calibration.image_height)
    if axis == "x":
        ref_x = sensor_w * (cal_w / sensor_w)
        ref_y = (sensor_h / 2.0) * (cal_h / sensor_h)
    else:
        ref_x = (sensor_w / 2.0) * (cal_w / sensor_w)
        ref_y = sensor_h * (cal_h / sensor_h)
    edge_ticks_x, edge_ticks_y = calibration.pixel_to_world_delta(ref_x, ref_y)
    full_frame_nm_x = abs(int(round(edge_ticks_x * _NM_PER_TICK))) * 2
    full_frame_nm_y = abs(int(round(edge_ticks_y * _NM_PER_TICK))) * 2
    if axis == "x":
        scale = full_frame_nm_x / sensor_w if sensor_w > 0 else 1.0
        return int(round(distance_nm * scale / full_frame_nm_x * full_frame_nm_x)), 0
    scale = full_frame_nm_y / sensor_h if sensor_h > 0 else 1.0
    return 0, int(round(distance_nm * scale / full_frame_nm_y * full_frame_nm_y))


def fraction_to_stage_delta(
    calibration: CameraCalibration,
    fraction: float,
    axis: str,
    sensor_w: int,
    sensor_h: int,
) -> tuple[int, int]:
    """Convert a fraction of the camera frame to a stage delta in nanometres.

    ``fraction=1.0`` corresponds to the full frame width (for ``axis='x'``) or
    full frame height (for ``axis='y'``).  Returns (dx_nm, dy_nm).
    """
    cal_w = float(calibration.image_width)
    cal_h = float(calibration.image_height)
    if axis == "x":
        ref_x = sensor_w * (cal_w / sensor_w)
        ref_y = (sensor_h / 2.0) * (cal_h / sensor_h)
        ticks_x, ticks_y = calibration.pixel_to_world_delta(ref_x, ref_y)
        full_nm = abs(int(round(ticks_x * _NM_PER_TICK))) * 2
        return int(round(full_nm * fraction)), 0
    ref_x = (sensor_w / 2.0) * (cal_w / sensor_w)
    ref_y = sensor_h * (cal_h / sensor_h)
    ticks_x, ticks_y = calibration.pixel_to_world_delta(ref_x, ref_y)
    full_nm = abs(int(round(ticks_y * _NM_PER_TICK))) * 2
    return 0, int(round(full_nm * fraction))