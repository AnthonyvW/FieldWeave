"""
focus_detection.py - Focus detection and single-frame visualisation.

Core pipeline
-------------
  FocusMethod           -- literal type alias: "tenengrad" | "laplacian"
  generate_focus_map    -- compute a per-pixel Tenengrad focus score map
  generate_focus_map_laplacian
                        -- compute a per-pixel Laplacian-variance focus score map
  normalize_score_map   -- normalise a raw score map to [0, 1]
  FocusScores           -- whole / center / peak summary scores
  compute_focus_scores  -- derive FocusScores from a normalised map

Single-frame visualisation
--------------------------
  apply_focus_overlay   -- blend a colourised focus heatmap onto the source image
  add_colorbar          -- append a vertical legend bar to an image
  build_frame           -- end-to-end: image → composited BGR frame + FocusScores
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

# Literal type for focus detection method selection.
FocusMethod = str  # "tenengrad" | "laplacian"
FOCUS_METHOD_TENENGRAD = "tenengrad"
FOCUS_METHOD_LAPLACIAN = "laplacian"


@dataclass(frozen=True)
class FocusRegion:
    """
    Axis-aligned region of interest for focus analysis.

    All four margins are fractional values in [0, 1) representing the
    proportion of the image dimension to exclude from each edge.  The
    values are derived from ``FocusRegionSettings`` percentages divided
    by 100.  When passed to a focus-map function the input image is cropped
    to this rectangle before any processing, so only the region pixels are
    ever touched.  The result is embedded back into a full-frame zeros array
    at the correct position before returning.
    """
    left: float = 0.0
    right: float = 0.0
    top: float = 0.0
    bottom: float = 0.0

    def pixel_bounds(self, h: int, w: int) -> tuple[int, int, int, int]:
        """
        Return (x0, y0, x1, y1) pixel coordinates of the active rectangle
        for an image of shape (h, w).  The rectangle is half-open: rows
        [y0, y1) and columns [x0, x1) are inside the region.
        """
        x0 = int(round(w * self.left))
        x1 = int(round(w * (1.0 - self.right)))
        y0 = int(round(h * self.top))
        y1 = int(round(h * (1.0 - self.bottom)))
        # Clamp to valid range and ensure at least 1 pixel.
        x0 = max(0, min(x0, w - 1))
        x1 = max(x0 + 1, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(y0 + 1, min(y1, h))
        return x0, y0, x1, y1


def generate_focus_map(
    image: np.ndarray,
    kernel_size: int = 3,
    radius: float = 8.0,
    threshold: float = 0.0,
    half_resolution: bool = False,
    box_blur: bool = False,
    verbose: bool = True,
    normalize: bool = True,
    focus_region: FocusRegion | None = None,
) -> np.ndarray:
    """
    Compute a per-pixel focus map using the Tenengrad focus measure.

    Based on 'Autofocusing Algorithm Selection in Computer Microscopy' by Sun et al.

    Pipeline:
      1. If focus_region is set, crop the image to the region bounds so all
         subsequent steps only process the smaller area.
      2. Grayscale conversion
      3. Optional 2x downscale (half_resolution=True) — all processing runs at
         quarter pixel count, score map is upscaled back to crop size before
         embedding into the full-frame output.
      4. Horizontal and vertical Sobel gradients, squared and summed
         to get gradient magnitude squared
      5. Zero out values below threshold to suppress soft/textureless regions,
         sharpening contrast between in-focus and out-of-focus areas
      6. Box blur or Gaussian blur to spread remaining sharp signal smoothly.
         Box blur is significantly faster with visually similar results for
         this use case.
      7. sqrt to bring back to gradient magnitude units and compress dynamic range
      8. Normalise to [0, 1]
      9. Embed the crop result into a full-frame zeros array at the correct
         position (no-op when no region is set).

    Returns a 2D float32 array at the original image resolution.
    Normalised to [0, 1] when normalize=True (default). When normalize=False,
    returns raw gradient magnitude units suitable for cross-frame normalisation.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    full_h, full_w = image.shape[:2]

    # Crop to the region of interest before any processing.
    if focus_region is not None:
        t0 = time.perf_counter()
        x0, y0, x1, y1 = focus_region.pixel_bounds(full_h, full_w)
        image = image[y0:y1, x0:x1]
        log(f"    Region crop:        {(time.perf_counter() - t0) * 1000:.1f}ms")
    else:
        x0, y0 = 0, 0

    t0 = time.perf_counter()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray_f = gray.astype(np.float32)
    log(f"    Grayscale convert:  {(time.perf_counter() - t0) * 1000:.1f}ms")

    if half_resolution:
        t0 = time.perf_counter()
        gray_f = cv2.resize(gray_f, (gray_f.shape[1] // 2, gray_f.shape[0] // 2), interpolation=cv2.INTER_AREA)
        log(f"    Downscale:          {(time.perf_counter() - t0) * 1000:.1f}ms")

    t0 = time.perf_counter()
    sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=kernel_size)
    log(f"    Sobel:              {(time.perf_counter() - t0) * 1000:.1f}ms")

    t0 = time.perf_counter()
    magnitude = sobel_x ** 2 + sobel_y ** 2
    log(f"    Magnitude squared:  {(time.perf_counter() - t0) * 1000:.1f}ms")

    t0 = time.perf_counter()
    if threshold > 0:
        magnitude[magnitude < threshold] = 0.0
    log(f"    Threshold:          {(time.perf_counter() - t0) * 1000:.1f}ms")

    t0 = time.perf_counter()
    if radius > 0:
        # When running at half resolution the spatial radius halves too, so
        # scale it down to keep the effective blur radius consistent with
        # what the caller specified in full-resolution pixel units.
        effective_radius = radius / 2 if half_resolution else radius
        blur_window = int(effective_radius * 4) + 1
        if blur_window % 2 == 0:
            blur_window += 1
        if box_blur:
            magnitude = cv2.blur(magnitude, (blur_window, blur_window), borderType=cv2.BORDER_REFLECT)
            log(f"    Box blur:           {(time.perf_counter() - t0) * 1000:.1f}ms")
        else:
            magnitude = cv2.GaussianBlur(
                magnitude,
                (blur_window, blur_window),
                effective_radius,
                borderType=cv2.BORDER_REFLECT,
            )
            log(f"    Gaussian blur:      {(time.perf_counter() - t0) * 1000:.1f}ms")

    t0 = time.perf_counter()
    result = np.sqrt(magnitude)
    log(f"    sqrt:               {(time.perf_counter() - t0) * 1000:.1f}ms")

    if half_resolution:
        t0 = time.perf_counter()
        # Upscale back to crop size, not full-frame size.
        crop_h, crop_w = image.shape[:2]
        result = cv2.resize(result, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        log(f"    Upscale:            {(time.perf_counter() - t0) * 1000:.1f}ms")

    if normalize:
        t0 = time.perf_counter()
        result = normalize_score_map(result)
        log(f"    Normalise:          {(time.perf_counter() - t0) * 1000:.1f}ms")

    # Embed the crop result back into a full-frame array.
    if focus_region is not None:
        t0 = time.perf_counter()
        full = np.zeros((full_h, full_w), dtype=np.float32)
        full[y0:y0 + result.shape[0], x0:x0 + result.shape[1]] = result
        result = full
        log(f"    Embed:              {(time.perf_counter() - t0) * 1000:.1f}ms")

    return result.astype(np.float32)


def generate_focus_map_laplacian(
    image: np.ndarray,
    window_size: int = 15,
    radius: float = 8.0,
    threshold: float = 0.0,
    half_resolution: bool = False,
    box_blur: bool = False,
    verbose: bool = True,
    normalize: bool = True,
    focus_region: FocusRegion | None = None,
) -> np.ndarray:
    """
    Compute a per-pixel focus map using local Laplacian variance.

    The classical Laplacian-variance focus measure collapses an entire image to a
    single scalar (variance of the Laplacian).  To produce a spatially useful
    heatmap we instead compute a *local* variance over a sliding window, giving
    each pixel a score that reflects how much high-frequency detail exists in its
    neighbourhood.

    Pipeline:
      1. If focus_region is set, crop the image to the region bounds so all
         subsequent steps only process the smaller area.
      2. Grayscale conversion
      3. Optional 2x downscale (half_resolution=True)
      4. Laplacian filter to extract second-derivative (edge) response
      5. Compute E[x²] and E[x]² over a local window using box filtering
         → local variance = E[x²] − E[x]²
      6. Zero out values below threshold (suppress flat/textureless regions)
      7. Box blur or Gaussian blur to spread the signal smoothly (same
         semantics as the Tenengrad radius parameter)
      8. sqrt to compress dynamic range
      9. Normalise to [0, 1] (when normalize=True)
      10. Embed the crop result into a full-frame zeros array at the correct
          position (no-op when no region is set).

    The window_size parameter (in full-resolution pixels) controls the spatial
    extent of the local variance computation — larger values integrate more
    context but reduce spatial resolution of the map.

    Returns a 2D float32 array at the original image resolution.
    When normalize=False, returns raw variance units for cross-frame normalisation.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    full_h, full_w = image.shape[:2]

    # Crop to the region of interest before any processing.
    if focus_region is not None:
        t0 = time.perf_counter()
        x0, y0, x1, y1 = focus_region.pixel_bounds(full_h, full_w)
        image = image[y0:y1, x0:x1]
        log(f"    Region crop:        {(time.perf_counter() - t0) * 1000:.1f}ms")
    else:
        x0, y0 = 0, 0

    t0 = time.perf_counter()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray_f = gray.astype(np.float32)
    log(f"    Grayscale convert:  {(time.perf_counter() - t0) * 1000:.1f}ms")

    if half_resolution:
        t0 = time.perf_counter()
        gray_f = cv2.resize(
            gray_f, (gray_f.shape[1] // 2, gray_f.shape[0] // 2),
            interpolation=cv2.INTER_AREA,
        )
        log(f"    Downscale:          {(time.perf_counter() - t0) * 1000:.1f}ms")

    t0 = time.perf_counter()
    lap = cv2.Laplacian(gray_f, cv2.CV_32F)
    log(f"    Laplacian:          {(time.perf_counter() - t0) * 1000:.1f}ms")

    # Local variance = E[lap²] - E[lap]²
    # Scale window to half-resolution if needed so spatial extent stays consistent.
    t0 = time.perf_counter()
    effective_win = max(3, window_size // 2 if half_resolution else window_size)
    if effective_win % 2 == 0:
        effective_win += 1
    ksize = (effective_win, effective_win)
    mean_lap  = cv2.boxFilter(lap,        cv2.CV_32F, ksize, normalize=True)
    mean_lap2 = cv2.boxFilter(lap ** 2,   cv2.CV_32F, ksize, normalize=True)
    local_var = np.maximum(mean_lap2 - mean_lap ** 2, 0.0)
    log(f"    Local variance:     {(time.perf_counter() - t0) * 1000:.1f}ms")

    t0 = time.perf_counter()
    if threshold > 0:
        local_var[local_var < threshold] = 0.0
    log(f"    Threshold:          {(time.perf_counter() - t0) * 1000:.1f}ms")

    t0 = time.perf_counter()
    if radius > 0:
        effective_radius = radius / 2 if half_resolution else radius
        blur_window = int(effective_radius * 4) + 1
        if blur_window % 2 == 0:
            blur_window += 1
        if box_blur:
            local_var = cv2.blur(
                local_var, (blur_window, blur_window), borderType=cv2.BORDER_REFLECT,
            )
            log(f"    Box blur:           {(time.perf_counter() - t0) * 1000:.1f}ms")
        else:
            local_var = cv2.GaussianBlur(
                local_var,
                (blur_window, blur_window),
                effective_radius,
                borderType=cv2.BORDER_REFLECT,
            )
            log(f"    Gaussian blur:      {(time.perf_counter() - t0) * 1000:.1f}ms")

    t0 = time.perf_counter()
    result = np.sqrt(local_var)
    log(f"    sqrt:               {(time.perf_counter() - t0) * 1000:.1f}ms")

    if half_resolution:
        t0 = time.perf_counter()
        # Upscale back to crop size, not full-frame size.
        crop_h, crop_w = image.shape[:2]
        result = cv2.resize(
            result, (crop_w, crop_h),
            interpolation=cv2.INTER_LINEAR,
        )
        log(f"    Upscale:            {(time.perf_counter() - t0) * 1000:.1f}ms")

    if normalize:
        t0 = time.perf_counter()
        result = normalize_score_map(result)
        log(f"    Normalise:          {(time.perf_counter() - t0) * 1000:.1f}ms")

    # Embed the crop result back into a full-frame array.
    if focus_region is not None:
        t0 = time.perf_counter()
        full = np.zeros((full_h, full_w), dtype=np.float32)
        full[y0:y0 + result.shape[0], x0:x0 + result.shape[1]] = result
        result = full
        log(f"    Embed:              {(time.perf_counter() - t0) * 1000:.1f}ms")

    return result.astype(np.float32)


def normalize_score_map(
    score_map: np.ndarray,
    ceiling: float | None = None,
) -> np.ndarray:
    """
    Normalise a raw score map to [0, 1].

    Args:
        score_map: Raw float32 gradient magnitude array from generate_focus_map.
        ceiling: Value to treat as 1.0. If None, uses the map's own maximum
            (per-frame normalisation). Pass a pre-computed global percentile
            value for cross-frame consistent brightness.

    Returns:
        float32 array clipped to [0, 1].
    """
    if ceiling is None:
        ceiling = float(score_map.max())
    if ceiling > 0:
        return np.clip(score_map / ceiling, 0.0, 1.0).astype(np.float32)
    return np.zeros_like(score_map, dtype=np.float32)


@dataclass
class FocusScores:
    """Focus scores derived from a normalised [0, 1] score map."""
    whole: float
    """Mean score across the active region (or entire image if no region is set)."""
    center: float
    """Mean score within the central region (inner quarter of area by default)."""
    peak: float
    """Mean score of the brightest pixels (top percentile by default)."""


def compute_focus_scores(
    score_map: np.ndarray,
    center_fraction: float = 0.5,
    peak_percentile: float = 99.0,
    focus_region: FocusRegion | None = None,
) -> FocusScores:
    """
    Derive whole-image, center-region, and peak focus scores from a score map.

    When *focus_region* is provided the ``whole`` score is computed only over
    the active rectangle so that the masked-out border pixels (which are zero)
    do not drag the mean down.  The ``center`` and ``peak`` scores always
    operate within the same active rectangle.

    Args:
        score_map: 2D float32 array normalised to [0, 1].
        center_fraction: Linear fraction of each axis defining the center crop.
            0.5 means the inner 50% of width and height (25% of total area).
        peak_percentile: Pixels at or above this percentile are averaged for
            the peak score. 99.0 means the top 1% of pixels.
        focus_region: Optional region of interest.  When given, all scores are
            computed within the active rectangle only.

    Returns:
        FocusScores with whole, center, and peak values in [0, 1].
    """
    h, w = score_map.shape

    if focus_region is not None:
        rx0, ry0, rx1, ry1 = focus_region.pixel_bounds(h, w)
    else:
        rx0, ry0, rx1, ry1 = 0, 0, w, h

    active = score_map[ry0:ry1, rx0:rx1]
    whole = float(active.mean())

    # Center crop is relative to the active region, not the full frame.
    ah = ry1 - ry0
    aw = rx1 - rx0
    cy0 = ry0 + int(ah * (1.0 - center_fraction) / 2)
    cy1 = ry0 + int(ah * (1.0 + center_fraction) / 2)
    cx0 = rx0 + int(aw * (1.0 - center_fraction) / 2)
    cx1 = rx0 + int(aw * (1.0 + center_fraction) / 2)
    center = float(score_map[cy0:cy1, cx0:cx1].mean())

    threshold = float(np.percentile(active, peak_percentile))
    peak_pixels = active[active >= threshold]
    peak = float(peak_pixels.mean()) if peak_pixels.size > 0 else 0.0

    return FocusScores(whole=whole, center=center, peak=peak)


def apply_focus_overlay(
    image: np.ndarray,
    score_map: np.ndarray,
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """
    Apply a colormap to the score map and blend it with the original image.

    score_map is expected to be normalised to [0, 1] by generate_focus_map.
    smooth_sigma can optionally apply a final Gaussian blur before colourising,
    but contrast stretching is already handled upstream.

    Returns a BGR uint8 composite image.
    """
    smoothed = score_map
    if smooth_sigma > 0:
        ksize = int(smooth_sigma * 6) | 1
        smoothed = cv2.GaussianBlur(score_map, (ksize, ksize), smooth_sigma)

    norm = (np.clip(smoothed, 0.0, 1.0) * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(norm, colormap)

    if len(image.shape) == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    overlay = cv2.addWeighted(base, 1.0 - alpha, heatmap, alpha, 0)
    return overlay


def add_colorbar(
    image: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    bar_width: int = 40,
    label_low: str = "Soft",
    label_high: str = "Sharp",
    side: str = "right",
) -> np.ndarray:
    """Append a vertical colorbar legend on the right (or left) side of the image."""
    h = image.shape[0]
    gradient = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    bar = cv2.applyColorMap(np.repeat(gradient, bar_width, axis=1), colormap)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    pad = 4

    cv2.putText(bar, label_high, (2, 14), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(bar, label_low, (2, h - pad), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return np.hstack([bar, image]) if side == "left" else np.hstack([image, bar])


def build_frame(
    image: np.ndarray,
    colormap: int,
    alpha: float,
    colorbar_side: str,
    kernel_size: int,
    radius: float,
    threshold: float,
    half_resolution: bool,
    box_blur: bool,
    side_by_side: bool,
    verbose: bool = True,
    score_map: np.ndarray | None = None,
    method: FocusMethod = FOCUS_METHOD_TENENGRAD,
    laplacian_window: int = 15,
    focus_region: FocusRegion | None = None,
) -> tuple[np.ndarray, FocusScores]:
    """
    Process a single image into a composited BGR frame and its focus scores.

    If score_map is provided it is used directly (already normalised), skipping
    focus map computation. This is used by multi-frame callers after global
    normalisation.

    method selects the focus measure: FOCUS_METHOD_TENENGRAD (Sobel-based) or
    FOCUS_METHOD_LAPLACIAN (local Laplacian variance).  laplacian_window is only
    used when method is FOCUS_METHOD_LAPLACIAN.

    focus_region, when provided, restricts scoring to the defined rectangle.
    Pixels outside the region are zeroed in the score map so they appear dark
    in the heatmap, making the active area visually obvious.

    Returns the composited BGR frame and its FocusScores.
    """
    if score_map is None:
        if method == FOCUS_METHOD_LAPLACIAN:
            score_map = generate_focus_map_laplacian(
                image,
                window_size=laplacian_window,
                radius=radius,
                threshold=threshold,
                half_resolution=half_resolution,
                box_blur=box_blur,
                verbose=verbose,
                focus_region=focus_region,
            )
        else:
            score_map = generate_focus_map(
                image,
                kernel_size=kernel_size,
                radius=radius,
                threshold=threshold,
                half_resolution=half_resolution,
                box_blur=box_blur,
                verbose=verbose,
                focus_region=focus_region,
            )
    scores = compute_focus_scores(score_map, focus_region=focus_region)
    overlay = apply_focus_overlay(image, score_map, alpha=alpha, colormap=colormap)
    overlay = add_colorbar(overlay, colormap=colormap, side=colorbar_side)

    if side_by_side:
        original = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Ensure both halves are the same height before hstacking
        h = min(original.shape[0], overlay.shape[0])
        frame = np.hstack([
            cv2.resize(original, (original.shape[1], h)),
            cv2.resize(overlay, (overlay.shape[1], h)),
        ])
    else:
        frame = overlay

    return frame, scores