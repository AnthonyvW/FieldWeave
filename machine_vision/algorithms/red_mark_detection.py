"""
red_mark_detection.py

Red registration-mark detection algorithm.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from machine_vision.machine_vision_config import MachineVisionSettings


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RedMarkDetectionResult:
    """
    Result of a single red-mark detection pass.

    All coordinate values are in full source-image pixel space (not
    downsampled), matching the frame dimensions in ``source_width`` /
    ``source_height``.  The overlay reprojects them into display space at
    paint time.
    """

    valid_centers: list[tuple[float, float]]
    """Centroids of accepted red-mark blobs, (x, y) in source pixels."""

    filtered_centers: list[tuple[float, float]]
    """Centroids of blobs that were rejected by area, aspect-ratio, or
    position filters, kept for diagnostic display."""

    valid_mask: np.ndarray
    """
    uint8 pixel mask of accepted blob footprints (255 = blob, 0 = background),
    shape (source_height, source_width).  Used by the overlay to paint actual
    blob pixels rather than just centroid markers.
    """

    filtered_mask: np.ndarray
    """uint8 pixel mask of rejected blob footprints, same shape as valid_mask."""

    mean_x: float | None
    """Mean X of valid centroid coordinates, or None when no valid blobs."""

    mean_y: float | None
    """Mean Y of valid centroid coordinates, or None when no valid blobs."""

    stabilized_x: float | None
    """
    EMA-smoothed mean X of valid mark centroids; used for the vertical
    reference line.  ``None`` when no valid marks are present.
    """

    stabilized_y: float | None
    """
    EMA-smoothed mean Y of valid mark centroids; used for the horizontal
    reference line.  ``None`` when no valid marks are present.
    """

    image_center_x: float | None
    """Horizontal midpoint of the source frame (source_width / 2)."""

    image_center_y: float | None
    """Vertical midpoint of the source frame (source_height / 2)."""

    line_orientation: str
    """``'vertical'`` or ``'horizontal'`` — which axis the marks indicate."""

    elapsed_ms: float
    """Wall-clock time taken for this detection pass in milliseconds."""

    source_width: int
    source_height: int


# ---------------------------------------------------------------------------
# RedMarkDetection
# ---------------------------------------------------------------------------

class RedMarkDetection:
    """
    Detects red registration marks with EMA smoothing and orientation hysteresis.

    Holds a reference to the shared ``MachineVisionSettings`` so parameter
    changes take effect on the next ``process()`` call without any explicit
    fan-out.
    """

    def __init__(self, settings: MachineVisionSettings) -> None:
        self._settings = settings
        self._smoothed_x: float | None = None
        self._smoothed_y: float | None = None
        self._cluster_frames: int = 0

    def reset(self) -> None:
        """Reset EMA and orientation hysteresis state for a new detection session."""
        self._smoothed_x = None
        self._smoothed_y = None
        self._cluster_frames = 0

    def process(self, frame_bytes: bytes, width: int, height: int) -> RedMarkDetectionResult:
        t0 = time.perf_counter()
        rm = self._settings.red_mark

        arr = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3)).copy()
        height, width = arr.shape[:2]

        valid, filtered, valid_mask, filtered_mask, mean_x, mean_y = detect_red_marks(
            arr,
            open_kernel_size=rm.open_kernel_size,
            min_area=rm.min_area,
            scale=rm.scale,
            max_aspect_ratio=rm.max_aspect_ratio,
            min_area_fraction=rm.min_area_fraction,
            hue_low=rm.hue_low,
            hue_high=rm.hue_high,
            sat_min=rm.sat_min,
            val_min=rm.val_min,
        )

        self._smoothed_x = smooth_value(
            self._smoothed_x, mean_x,
            rm.smoothing_alpha, rm.deadband_px, rm.max_step_px, rm.jump_threshold_px,
        )
        self._smoothed_y = smooth_value(
            self._smoothed_y, mean_y,
            rm.smoothing_alpha, rm.deadband_px, rm.max_step_px, rm.jump_threshold_px,
        )

        orientation, self._cluster_frames = line_orientation(
            valid, width, rm.side_cluster_fraction, rm.side_cluster_margin, self._cluster_frames,
        )

        return RedMarkDetectionResult(
            valid_centers=valid,
            filtered_centers=filtered,
            valid_mask=valid_mask,
            filtered_mask=filtered_mask,
            mean_x=mean_x,
            mean_y=mean_y,
            stabilized_x=self._smoothed_x,
            stabilized_y=self._smoothed_y,
            image_center_x=float(width) / 2.0,
            image_center_y=float(height) / 2.0,
            line_orientation=orientation,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            source_width=width,
            source_height=height,
        )


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def detect_red_marks(
    img_rgb: np.ndarray,
    open_kernel_size: int,
    min_area: int,
    max_aspect_ratio: float,
    min_area_fraction: float,
    hue_low: int,
    hue_high: int,
    sat_min: int,
    val_min: int,
    scale: int = 1,
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    np.ndarray,
    np.ndarray,
    float | None,
    float | None,
]:
    """
    Detect red blobs in an RGB frame.

    Pipeline:
      1. Optionally downsample by 1/scale for speed.
      2. Convert RGB -> HSV, isolate red with two inRange masks (handles
         hue wrap-around at 0/180), morphological opening to kill noise.
      3. Connected-component analysis.
      4. Absolute area (>= min_area) and aspect-ratio filter.
      5. Relative area filter: discard blobs < min_area_fraction of the
         largest survivor.
      6. IQR outlier removal on X coordinates (only when > 3 blobs survive).
      7. Scale centroids and masks back to full-resolution coordinates.

    Pixel masks are built via an O(pixels) LUT indexed by connected-component
    label id so the overlay can paint actual blob footprints.

    Parameters
    ----------
    img_rgb:
        RGB image, shape (H, W, 3), dtype uint8.
    open_kernel_size:
        Side length of the elliptical MORPH_OPEN kernel.
    min_area:
        Minimum blob pixel area in full-resolution pixels.  Scaled
        proportionally before processing when scale > 1.
    max_aspect_ratio:
        Maximum bounding-box aspect ratio.
    min_area_fraction:
        Minimum blob area as a fraction of the largest surviving blob.
    hue_low, hue_high, sat_min, val_min:
        HSV thresholds.  Red occupies [hue_low, 180] | [0, hue_high].
    scale:
        Downsample factor.  The image is resized to 1/scale before
        processing and all results are projected back to full resolution.
        Must be >= 1.  Default is 1 (no downsampling).

    Returns
    -------
    valid_centers, filtered_centers, valid_mask, filtered_mask, mean_x, mean_y
    """
    orig_h, orig_w = img_rgb.shape[:2]

    if scale > 1:
        proc = cv2.resize(img_rgb, (0, 0), fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_AREA)
        scaled_min_area = max(1, int(min_area / (scale * scale)))
    else:
        proc = img_rgb
        scaled_min_area = min_area

    img_hsv = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

    lower1 = np.array([hue_low,  sat_min, val_min], dtype=np.uint8)
    upper1 = np.array([180,      255,     255],     dtype=np.uint8)
    lower2 = np.array([0,        sat_min, val_min], dtype=np.uint8)
    upper2 = np.array([hue_high, 255,     255],     dtype=np.uint8)

    red_isolated = cv2.bitwise_or(
        cv2.inRange(img_hsv, lower1, upper1),
        cv2.inRange(img_hsv, lower2, upper2),
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size)
    )
    opened = cv2.morphologyEx(red_isolated, cv2.MORPH_OPEN, open_kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        opened, connectivity=8
    )

    empty_mask = np.zeros_like(opened)

    if num_labels <= 1:
        return [], [], empty_mask, empty_mask, None, None

    areas = stats[1:, cv2.CC_STAT_AREA]
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    heights_cc = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    aspect_ratios = np.maximum(widths, heights_cc) / np.maximum(np.minimum(widths, heights_cc), 1)

    keep = (areas >= scaled_min_area) & (aspect_ratios <= max_aspect_ratio)
    if keep.any():
        max_kept_area = float(areas[keep].max())
        keep &= areas >= min_area_fraction * max_kept_area

    kept_indices = np.where(keep)[0] + 1
    if kept_indices.size > 3:
        cx = centroids[kept_indices, 0]
        q1, q3 = float(np.percentile(cx, 25)), float(np.percentile(cx, 75))
        iqr = q3 - q1
        inlier = (cx >= q1 - 1.5 * iqr) & (cx <= q3 + 1.5 * iqr)
        keep[kept_indices[~inlier] - 1] = False
        kept_indices = kept_indices[inlier]

    rejected_indices = np.where(~keep)[0] + 1

    valid_centroids = centroids[kept_indices]
    valid_centers: list[tuple[float, float]] = [
        (float(c[0]), float(c[1])) for c in valid_centroids
    ]
    filtered_centers: list[tuple[float, float]] = [
        (float(centroids[i, 0]), float(centroids[i, 1])) for i in rejected_indices
    ]

    lut = np.zeros(num_labels, dtype=np.uint8)
    lut[kept_indices] = 255
    valid_mask = lut[labels]

    lut[:] = 0
    lut[rejected_indices] = 255
    filtered_mask = lut[labels]

    if scale > 1:
        valid_centers = [(c[0] * scale, c[1] * scale) for c in valid_centers]
        filtered_centers = [(c[0] * scale, c[1] * scale) for c in filtered_centers]
        valid_mask = cv2.resize(valid_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        filtered_mask = cv2.resize(filtered_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mean_x: float | None = float(np.mean([c[0] for c in valid_centers])) if valid_centers else None
        mean_y: float | None = float(np.mean([c[1] for c in valid_centers])) if valid_centers else None
    else:
        mean_x = float(np.mean(valid_centroids[:, 0])) if kept_indices.size > 0 else None
        mean_y = float(np.mean(valid_centroids[:, 1])) if kept_indices.size > 0 else None

    return valid_centers, filtered_centers, valid_mask, filtered_mask, mean_x, mean_y


def smooth_value(
    prev: float | None,
    raw: float | None,
    alpha: float,
    deadband: float,
    max_step: float,
    jump_threshold: float,
) -> float | None:
    """
    Single-axis EMA smoother with jump detection, deadband, and slew-rate limit.

    - If ``raw`` is None (no detection this frame), hold ``prev`` unchanged.
    - If ``prev`` is None (first detection), return ``raw`` directly.
    - If |raw - prev| > jump_threshold, jump directly to ``raw``.
    - If |raw - prev| <= deadband, hold ``prev`` (ignore jitter).
    - Otherwise clamp the step to max_step and apply the EMA weight ``alpha``.
    """
    if raw is None:
        return prev
    if prev is None:
        return raw
    delta = raw - prev
    if abs(delta) > jump_threshold:
        return raw
    if abs(delta) <= deadband:
        return prev
    if max_step > 0:
        delta = float(np.clip(delta, -max_step, max_step))
    return prev + float(np.clip(alpha, 0.0, 1.0)) * delta


def line_orientation(
    centers: list[tuple[float, float]],
    img_width: int,
    side_cluster_fraction: float,
    side_cluster_margin: float,
    cluster_frames: int,
) -> tuple[str, int]:
    """
    Determine whether the reference line should be ``'vertical'`` or
    ``'horizontal'`` based on how the accepted mark centroids are distributed.

    Returns ``'horizontal'`` when most marks cluster within
    ``side_cluster_margin`` of one horizontal edge for at least two
    consecutive frames (hysteresis), otherwise ``'vertical'``.
    """
    if not centers or img_width <= 0:
        return "vertical", 0

    xs = np.array([c[0] for c in centers], dtype=np.float32)
    margin = float(np.clip(side_cluster_margin, 0.0, 0.45))
    left_edge  = img_width * margin
    right_edge = img_width * (1.0 - margin)

    frac = max(
        float(np.mean(xs <= left_edge)),
        float(np.mean(xs >= right_edge)),
    )

    if frac >= side_cluster_fraction:
        cluster_frames += 1
    else:
        cluster_frames = max(0, cluster_frames - 1)

    orientation = "horizontal" if cluster_frames >= 2 else "vertical"
    return orientation, cluster_frames