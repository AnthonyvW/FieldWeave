from __future__ import annotations

from enum import Enum

_MM_PER_INCH = 25.4
_MM_PER_M = 1000.0
_MM_PER_FT = 304.8


class MeasurementUnit(Enum):
    PX = "px"
    UM = "\u00b5m"
    MM = "mm"
    CM = "cm"
    M = "m"
    IN = "in"
    FT = "ft"


def pixels_to_unit(pixels: float, dpi: float, unit: MeasurementUnit) -> float:
    """Convert a pixel length to *unit*, given the source's DPI (pixels per inch). Pixels are DPI-independent, so PX returns the raw pixel count unchanged."""
    if unit is MeasurementUnit.PX:
        return pixels
    mm = (pixels / dpi) * _MM_PER_INCH
    if unit is MeasurementUnit.UM:
        return mm * 1000
    if unit is MeasurementUnit.CM:
        return mm / 10
    if unit is MeasurementUnit.M:
        return mm / _MM_PER_M
    if unit is MeasurementUnit.IN:
        return mm / _MM_PER_INCH
    if unit is MeasurementUnit.FT:
        return mm / _MM_PER_FT
    return mm


def format_length(pixels: float, dpi: float, unit: MeasurementUnit, decimals: int = 2) -> str:
    return f"{pixels_to_unit(pixels, dpi, unit):.{max(0, decimals)}f} {unit.value}"


def area_pixels_to_unit(area_pixels: float, dpi: float, unit: MeasurementUnit) -> float:
    """Convert an area given in square pixels to square *unit*, given the source's DPI."""
    per_pixel = pixels_to_unit(1.0, dpi, unit)
    return area_pixels * per_pixel * per_pixel


def format_area(area_pixels: float, dpi: float, unit: MeasurementUnit, decimals: int = 2) -> str:
    return f"{area_pixels_to_unit(area_pixels, dpi, unit):.{max(0, decimals)}f} {unit.value}²"


def unit_to_mm(value: float, unit: MeasurementUnit) -> float:
    """Inverse of pixels_to_unit's scaling — convert a value given in *unit* to millimeters."""
    if unit is MeasurementUnit.UM:
        return value / 1000
    if unit is MeasurementUnit.CM:
        return value * 10
    if unit is MeasurementUnit.M:
        return value * _MM_PER_M
    if unit is MeasurementUnit.IN:
        return value * _MM_PER_INCH
    if unit is MeasurementUnit.FT:
        return value * _MM_PER_FT
    return value


def unit_to_px(value: float, unit: MeasurementUnit, dpi: float) -> float:
    """Convert a real-world length in *unit* to pixels at *dpi* — the inverse of pixels_to_unit. Pixels pass straight through."""
    if unit is MeasurementUnit.PX:
        return value
    return unit_to_mm(value, unit) * dpi / _MM_PER_INCH


def dpi_from_measurement(pixel_length: float, value: float, unit: MeasurementUnit) -> float | None:
    """Derive a DPI from a measured pixel length and the real-world length the user says it represents. None if either isn't a usable positive number."""
    if pixel_length <= 0:
        return None
    mm = unit_to_mm(value, unit)
    if mm <= 0:
        return None
    return pixel_length * _MM_PER_INCH / mm