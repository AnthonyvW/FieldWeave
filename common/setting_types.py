from __future__ import annotations

from dataclasses import dataclass

from typing import NamedTuple
from enum import Enum

class FileFormat(str, Enum):
    PNG = 'png'
    TIFF = 'tiff'
    JPEG = 'jpeg'


class RGBALevel(NamedTuple):
    r: int
    g: int
    b: int
    a: int
    
    def validate(self) -> None:
        for name, value in [('r', self.r), ('g', self.g), ('b', self.b), ('a', self.a)]:
            if not (0 <= value <= 255):
                raise ValueError(f"RGBALevel.{name} must be in range [0, 255], got {value}")


class SettingType(str, Enum):
    BOOL = "bool"
    RANGE = "range"
    DROPDOWN = "dropdown"
    RGBA_LEVEL = "rgba_level"
    BUTTON = "button"
    FILE_PICKER_BUTTON = "file_picker_button"
    NUMBER_PICKER = "number_picker"


@dataclass
class SettingMetadata:
    name: str
    display_name: str
    setting_type: SettingType
    description: str = ""
    min_value: int | None = None
    max_value: int | None = None
    choices: list[str] | None = None
    group: str = "General"
    runtime_changeable: bool = True
    # When set, this field is greyed out (and, for live-value fields, polled from
    # hardware) while the named boolean setting equals *controlled_when*.
    #
    # controlled_when=True  (default): grey out while the controller is ON.
    #   Example: exposure_time / gain are greyed while auto_exposure is True.
    # controlled_when=False: grey out while the controller is OFF.
    #   Example: exposure target is greyed while auto_exposure is False.
    controlled_by: str | None = None
    controlled_when: bool = True