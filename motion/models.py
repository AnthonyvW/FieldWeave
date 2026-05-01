from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# All Position coordinates are stored internally in nanometers.
# 1 mm = 1_000_000 nm
_NM_PER_MM = 1_000_000


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