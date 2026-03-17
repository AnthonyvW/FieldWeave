from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QPainter


class Overlay(ABC):
    """
    Base class for all camera preview overlays.

    Two optional frame hooks are provided for subclasses that need to
    accumulate state across frames:

    - ``update_full`` — called with the full camera-resolution array before
      scaling. Use this for anything that needs accurate measurements, such as
      focus scoring or sub-pixel feature tracking.
    - ``update_scaled`` — called with the display-resolution array after
      scaling. Use this for anything that only needs what's on screen, such as
      motion detection or histogram computation.

    Both default to no-ops, so stateless overlays only need to implement
    ``draw``.
    """

    def __init__(self) -> None:
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def update_full(self, frame: np.ndarray) -> None:
        """
        Called once per frame with the full-resolution RGB array (H×W×3, uint8)
        before any scaling is applied.
        """

    def update_scaled(self, frame: np.ndarray) -> None:
        """
        Called once per frame with the display-resolution RGB array (H×W×3, uint8)
        after the image has been scaled to fit the label.
        """

    @abstractmethod
    def draw(self, painter: QPainter, rect: QRect) -> None:
        """Called during paintEvent when this overlay is enabled."""