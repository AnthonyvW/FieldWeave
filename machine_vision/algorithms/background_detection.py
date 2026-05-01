"""
background_detection.py

Detects whether the camera is looking at the bare black-plastic background.

The background is achromatic, appearing as dark grey under normal lighting
and light grey under bright lighting.  Detection is based on two statistics
of the grayscale value channel:

- Median value <= val_median_max: the surface is dark-to-mid grey.
- Standard deviation of value <= val_std_max: the surface is uniform —
  it has no texture or markings that would create local brightness variation.

These two constraints together are highly specific to the plastic surface and
reject wood, paper, and other materials that share its approximate brightness
but not its uniformity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from machine_vision.machine_vision_config import MachineVisionSettings


@dataclass
class BackgroundDetectionResult:
    """Result of a single background detection pass."""

    is_background: bool
    """True when the frame is classified as bare background."""

    val_median: float
    """Median HSV value (brightness) across the frame [0-255]."""

    val_std: float
    """Standard deviation of HSV value across the frame [0-255]."""

    elapsed_ms: float
    """Wall-clock time taken for this detection pass in milliseconds."""


class BackgroundDetection:
    """
    Classifies whether the entire frame shows the bare black-plastic background.

    Holds a reference to the shared ``MachineVisionSettings`` so parameter
    changes take effect on the next ``process()`` call without any explicit
    fan-out.
    """

    def __init__(self, settings: MachineVisionSettings) -> None:
        self._settings = settings

    def reset(self) -> None:
        pass

    def process(self, frame_bytes: bytes, width: int, height: int) -> BackgroundDetectionResult:
        t0 = time.perf_counter()
        bg = self._settings.background

        arr = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
        is_bg, val_median, val_std = is_background_frame(
            arr,
            val_median_max=bg.val_median_max,
            val_std_max=bg.val_std_max,
            scale=bg.scale,
        )

        return BackgroundDetectionResult(
            is_background=is_bg,
            val_median=val_median,
            val_std=val_std,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )


def is_background_frame(
    img_rgb: np.ndarray,
    val_median_max: int,
    val_std_max: float,
    scale: int = 4,
) -> tuple[bool, float, float]:
    """
    Return whether an RGB frame shows the bare black-plastic background.

    The frame is classified as background when the median HSV value is <=
    val_median_max and the standard deviation of HSV value is <= val_std_max.
    The median constraint rejects bright surfaces; the std constraint rejects
    textured or non-uniform surfaces such as wood grain or printed paper.

    Parameters
    ----------
    img_rgb:
        RGB image, shape (H, W, 3), dtype uint8.
    val_median_max:
        Maximum median HSV value (brightness) [0-255].  Corresponds to
        0.5 in GIMP's normalised [0-1] scale, i.e. 128 in uint8.
    val_std_max:
        Maximum standard deviation of HSV value [0-255].  Corresponds to
        0.1 in GIMP's normalised scale, i.e. ~25 in uint8.
    scale:
        Downsample factor >= 1 applied before computing statistics.

    Returns
    -------
    is_background, val_median, val_std
    """
    if scale > 1:
        proc = cv2.resize(img_rgb, (0, 0), fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_AREA)
    else:
        proc = img_rgb

    val = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)[:, :, 2].astype(np.float32)
    val_median = float(np.median(val))
    val_std = float(np.std(val))

    return val_median <= val_median_max and val_std <= val_std_max, val_median, val_std
