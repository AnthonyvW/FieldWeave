"""
vision_algorithm.py

Base class for self-contained vision algorithm implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from machine_vision.machine_vision_config import MachineVisionSettings


class VisionAlgorithm(ABC):
    """
    Base class for a self-contained vision algorithm.

    Each subclass holds a reference to the shared ``MachineVisionSettings``
    and reads from it directly in ``process()``.  Subclasses have no
    dependency on Qt and can be constructed and tested independently.

    ``reset`` clears any accumulated per-session state (EMA values, hysteresis
    counters, etc.).  The default implementation is a no-op for stateless
    algorithms.
    """

    def __init__(self, settings: MachineVisionSettings) -> None:
        self._settings = settings

    @abstractmethod
    def process(self, *args, **kwargs): ...

    def reset(self) -> None:
        pass