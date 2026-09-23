"""
camera_calibration.py

Camera-to-stage spatial calibration for vision-guided movement.

This module provides the pure calibration maths and image-processing helpers
that map between image pixel coordinates and physical stage coordinates (in
nanometres, matching ``motion.models.Position``).  It has no dependency on any printer controller,
GUI framework, or Qt; all I/O and motion control remains the caller's
responsibility.

Typical workflow
----------------
1.  Capture a base image at the current stage position.
2.  Move the stage a known distance in +X; capture again.
3.  Return to base; move a known distance in +Y; capture again.
4.  Call ``build_calibration`` with the three edge maps and the known move
    distances to obtain a ``CameraCalibration`` instance.
5.  Call ``pixel_to_world_delta`` on any subsequent frame to convert a
    pixel coordinate into a stage-coordinate delta (in nanometres).
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING
from dataclasses import dataclass
from collections.abc import Callable

import cv2
import numpy as np

from machine_vision.algorithms.vision_algorithm import VisionAlgorithm

if TYPE_CHECKING:
    from machine_vision.machine_vision_config import MachineVisionSettings as _MachineVisionSettings


# ---------------------------------------------------------------------------
# Y-axis orientation
# ---------------------------------------------------------------------------

CameraYAxisOrientation = Literal["horizontal", "vertical"]
"""
Which image axis the world Y axis is primarily aligned with.

``"vertical"``   — a +Y stage move shifts the image mostly up or down.
``"horizontal"`` — a +Y stage move shifts the image mostly left or right.

Derived from ``CameraCalibration.M_est`` and never persisted; recomputed
whenever a calibration is loaded or newly built.
"""


def derive_y_axis_orientation(M_est: np.ndarray) -> CameraYAxisOrientation:
    """
    Determine which camera axis the world Y axis primarily aligns with.

    Applies ``M_est`` to a unit world-Y vector ``[0, 1]`` to obtain the
    pixel displacement produced by a +Y stage move.  Whichever pixel
    component (X = horizontal, Y = vertical) has the larger absolute
    magnitude is the primary axis.

    Parameters
    ----------
    M_est:
        2×2 camera-to-stage mapping matrix from ``CameraCalibration``.

    Returns
    -------
    ``"horizontal"`` if the world Y axis maps primarily along the image
    X axis, ``"vertical"`` otherwise.
    """
    world_y = np.array([[0.0], [1.0]], dtype=np.float64)
    pixel_delta = M_est @ world_y
    dpx = abs(float(pixel_delta[0, 0]))
    dpy = abs(float(pixel_delta[1, 0]))
    return "horizontal" if dpx > dpy else "vertical"


# ---------------------------------------------------------------------------
# Calibration state
# ---------------------------------------------------------------------------

@dataclass
class CameraCalibration:
    """
    Immutable snapshot of a completed camera-to-stage calibration.

    All stage coordinates and distances are in nanometres, matching
    ``motion.models.Position``.

    Attributes
    ----------
    M_est:
        2×2 matrix that maps a world delta (in nm) to a pixel delta.
        ``pixel_delta = M_est @ world_delta``
    M_inv:
        Inverse of ``M_est``.  Maps a pixel delta to a world delta.
        ``world_delta = M_inv @ pixel_delta``
    ref_x, ref_y, ref_z:
        Stage position (in nm) where calibration was performed.  Used as
        the origin for absolute vision-guided moves.
    image_width, image_height:
        Resolution of the images used during calibration.
    move_x_nm, move_y_nm:
        Calibration move distances (in nm) that were used to build M_est.
    dpi:
        Estimated camera resolution in dots-per-inch, derived from M_est.
        ``None`` if the calculation failed.
    """

    M_est: np.ndarray          # shape (2, 2), dtype float64
    M_inv: np.ndarray          # shape (2, 2), dtype float64

    ref_x: int
    ref_y: int
    ref_z: int

    image_width: int
    image_height: int

    move_x_nm: int
    move_y_nm: int

    dpi: float | None = None

    y_axis_orientation: CameraYAxisOrientation = "vertical"
    """
    Which image axis the world Y axis is primarily aligned with.
    Derived from ``M_est``; not persisted.  Recomputed whenever a
    calibration is loaded or built.
    """

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def pixel_to_world_delta(
        self,
        pixel_x: float,
        pixel_y: float,
        image_center_x: float | None = None,
        image_center_y: float | None = None,
    ) -> tuple[float, float]:
        """
        Convert an image pixel coordinate to the stage delta required to
        move that point under the camera centre.

        ``pixel_x`` and ``pixel_y`` are full-resolution camera coordinates
        measured from the image top-left origin.  Internally the image centre
        is subtracted to express the click as a delta from centre.

        ``M_inv`` maps that pixel delta to the stage move that *produced* the
        observed shift during calibration.  The result is negated to get the
        move that *cancels* the offset — i.e. brings the clicked point to
        centre.  This single negation is correct for all machines because the
        calibration matrix already encodes any machine-specific axis inversion
        via the measured phase-correlation signs.

        Parameters
        ----------
        pixel_x, pixel_y:
            Target pixel coordinates in full-resolution camera space
            (origin top-left).
        image_center_x, image_center_y:
            Override the image centre.  Defaults to half the image dimensions
            recorded at calibration time.

        Returns
        -------
        (dx_nm, dy_nm):
            Stage delta in nanometres.  Add to the current stage
            position to move the clicked point under the camera centre.
        """
        cx = image_center_x if image_center_x is not None else self.image_width / 2.0
        cy = image_center_y if image_center_y is not None else self.image_height / 2.0

        pixel_delta = np.array([[pixel_x - cx], [pixel_y - cy]], dtype=np.float64)
        world_delta = self.M_inv @ pixel_delta

        # M_inv maps a pixel delta to the stage move that *produced* that shift.
        # We want the move that *cancels* it — i.e. brings the clicked point to
        # centre — so we negate once.
        dx_nm = -float(world_delta[0, 0])
        dy_nm = -float(world_delta[1, 0])
        return dx_nm, dy_nm


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------

CALIBRATION_WORKING_LONG_SIDE = 2560

# An empty or featureless edge map makes cv2.phaseCorrelate return exactly
# half the image size with a near-zero response, which would otherwise be
# stored as a wildly wrong calibration.
MIN_PHASE_CORRELATION_RESPONSE = 0.05


def rgb_to_gray(arr: np.ndarray) -> np.ndarray:
    """
    Convert an RGB (or already-greyscale) uint8 array to greyscale.

    Parameters
    ----------
    arr:
        Shape (H, W) or (H, W, 3), dtype uint8.

    Returns
    -------
    Greyscale array of shape (H, W), dtype uint8.
    """
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def downscale_to_working_size(gray_u8: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Shrink a greyscale image so its long side is at most
    ``CALIBRATION_WORKING_LONG_SIDE`` pixels.

    Canny's thresholds are absolute gradient values, and a higher-resolution
    sensor viewing the same optics spreads each edge over more pixels, so
    per-pixel gradients shrink and most edges fall below threshold.  Working
    at a fixed resolution keeps edge detection independent of the camera.

    Returns
    -------
    (image, scale_x, scale_y):
        The (possibly) downscaled image and the per-axis factors that convert
        a shift measured in it back to original-resolution pixels.  They
        differ slightly when the image cannot be shrunk by an exact ratio,
        because each output dimension is rounded to a whole pixel.
    """
    h, w = gray_u8.shape[:2]
    long_side = max(h, w)
    if long_side <= CALIBRATION_WORKING_LONG_SIDE:
        return gray_u8, 1.0, 1.0
    factor = CALIBRATION_WORKING_LONG_SIDE / long_side
    small = cv2.resize(
        gray_u8,
        (round(w * factor), round(h * factor)),
        interpolation=cv2.INTER_AREA,
    )
    return small, w / small.shape[1], h / small.shape[0]


def compute_edge_map(gray_u8: np.ndarray) -> np.ndarray:
    """
    Produce a zero-mean, unit-variance Canny edge map suitable for phase
    correlation.

    A mild Gaussian blur is applied first to suppress sensor noise without
    destroying structural edges.  The resulting float32 array is
    mean-subtracted and divided by its standard deviation (plus a small
    epsilon to prevent division by zero).

    Parameters
    ----------
    gray_u8:
        Greyscale image, shape (H, W), dtype uint8.

    Returns
    -------
    Normalised edge map, shape (H, W), dtype float32.
    """
    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 180).astype(np.float32)
    edges -= edges.mean()
    edges /= (edges.std() + 1e-6)
    return edges


def phase_correlation_shift(
    img_a: np.ndarray,
    img_b: np.ndarray,
) -> tuple[float, float, float]:
    """
    Estimate the translational shift between two float32 images using phase
    correlation.

    A Hanning window is applied before the FFT to reduce spectral leakage at
    the image boundaries.

    Parameters
    ----------
    img_a, img_b:
        Registered float32 images of identical shape (H, W).

    Returns
    -------
    (dx, dy, response):
        ``dx`` and ``dy`` are the sub-pixel shift of ``img_b`` relative to
        ``img_a`` (positive dx means ``img_b`` is shifted right).
        ``response`` is the peak correlation value; higher is more reliable.
    """
    h, w = img_a.shape[:2]
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(img_a, img_b, window)
    return float(dx), float(dy), float(response)


# ---------------------------------------------------------------------------
# Calibration builder
# ---------------------------------------------------------------------------

def build_calibration(
    edges_base: np.ndarray,
    edges_x: np.ndarray,
    edges_y: np.ndarray,
    move_x_nm: int,
    move_y_nm: int,
    ref_x: int,
    ref_y: int,
    ref_z: int,
    image_width: int,
    image_height: int,
    pixel_scale: tuple[float, float] = (1.0, 1.0),
) -> CameraCalibration:
    """
    Compute the pixel-to-world calibration matrix from three edge maps.

    The caller must supply edge maps captured at three stage positions:

    * ``edges_base`` — at the reference (origin) position.
    * ``edges_x``   — after moving ``move_x_nm`` in +X from base.
    * ``edges_y``   — after moving ``move_y_nm`` in +Y from base
      (the stage must have returned to base before this capture).

    Phase correlation between base↔x and base↔y gives the pixel shift
    produced by each known world move.  These two observations directly fill
    the columns of the 2×2 mapping matrix ``M_est``:

    .. code-block:: text

        M_est @ [move_x_nm, 0        ]ᵀ = [dpx_x, dpy_x]ᵀ
        M_est @ [0,         move_y_nm]ᵀ = [dpx_y, dpy_y]ᵀ

    Parameters
    ----------
    edges_base, edges_x, edges_y:
        Float32 normalised edge maps of identical shape, produced by
        ``compute_edge_map``.
    move_x_nm, move_y_nm:
        Calibration move distances in nanometres.
    ref_x, ref_y, ref_z:
        Stage position in nanometres at the time ``edges_base`` was captured.
    image_width, image_height:
        Pixel dimensions of the original (full-resolution) frames.
    pixel_scale:
        ``(scale_x, scale_y)`` converting a shift measured in the edge maps
        to original-resolution pixels, as returned by
        ``downscale_to_working_size``.

    Returns
    -------
    ``CameraCalibration`` instance ready for ``pixel_to_world_delta`` calls.

    Raises
    ------
    ValueError
        If either shift could not be measured reliably, or the world matrix
        is singular (moves were collinear or too small to produce measurable
        pixel shifts).
    """
    dpx_x, dpy_x, resp_x = phase_correlation_shift(edges_base, edges_x)
    dpx_y, dpy_y, resp_y = phase_correlation_shift(edges_base, edges_y)

    for axis, resp in (("X", resp_x), ("Y", resp_y)):
        if resp < MIN_PHASE_CORRELATION_RESPONSE:
            raise ValueError(
                f"Calibration failed: the {axis} move could not be matched "
                f"(phase correlation response {resp:.3f} < "
                f"{MIN_PHASE_CORRELATION_RESPONSE}).  Make sure the sample is "
                "in focus and has visible texture, and that the move keeps "
                "most of the field of view overlapping."
            )

    scale_x, scale_y = pixel_scale
    dpx_x *= scale_x
    dpy_x *= scale_y
    dpx_y *= scale_x
    dpy_y *= scale_y

    # Build 2×2 world and pixel matrices; solve M_est = pixel_mat @ world_inv.
    world_mat = np.array(
        [[move_x_nm, 0.0],
         [0.0,       move_y_nm]],
        dtype=np.float64,
    ).T  # columns are world vectors

    pixel_mat = np.array(
        [[dpx_x, dpx_y],
         [dpy_x, dpy_y]],
        dtype=np.float64,
    )

    try:
        M_est: np.ndarray = pixel_mat @ np.linalg.inv(world_mat)
        M_inv: np.ndarray = np.linalg.inv(M_est)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Calibration failed: world matrix is singular.  "
            "Ensure the X and Y moves produced distinct pixel shifts."
        ) from exc

    return CameraCalibration(
        M_est=M_est,
        M_inv=M_inv,
        ref_x=ref_x,
        ref_y=ref_y,
        ref_z=ref_z,
        image_width=image_width,
        image_height=image_height,
        move_x_nm=move_x_nm,
        move_y_nm=move_y_nm,
        y_axis_orientation=derive_y_axis_orientation(M_est),
    )

# ---------------------------------------------------------------------------
# CalibrationBuild — settings-aware algorithm class
# ---------------------------------------------------------------------------


class CalibrationBuild(VisionAlgorithm):
    """
    Builds a ``CameraCalibration`` from three captured frames.

    On success, writes the result back into
    ``settings.camera_calibration.calibration`` and calls ``save_settings``
    so the calibration is persisted without any involvement from the manager.
    """

    def __init__(self, settings: _MachineVisionSettings, save_settings: Callable[[], None]) -> None:
        super().__init__(settings)
        self._save_settings = save_settings

    def process(
        self,
        base_bytes: bytes, base_width: int, base_height: int,
        x_bytes: bytes,    x_width: int,    x_height: int,
        y_bytes: bytes,    y_width: int,    y_height: int,
        move_x_nm: int, move_y_nm: int,
        ref_x: int, ref_y: int, ref_z: int,
    ) -> CameraCalibration:
        def _to_edge(fb: bytes, w: int, h: int) -> tuple[np.ndarray, tuple[float, float]]:
            arr = np.frombuffer(fb, dtype=np.uint8).reshape((h, w, 3))
            small, scale_x, scale_y = downscale_to_working_size(rgb_to_gray(arr))
            return compute_edge_map(small), (scale_x, scale_y)

        edges_base, scale = _to_edge(base_bytes, base_width, base_height)
        edges_x, _ = _to_edge(x_bytes, x_width, x_height)
        edges_y, _ = _to_edge(y_bytes, y_width, y_height)

        calibration = build_calibration(
            edges_base=edges_base,
            edges_x=edges_x,
            edges_y=edges_y,
            move_x_nm=move_x_nm,
            move_y_nm=move_y_nm,
            ref_x=ref_x,
            ref_y=ref_y,
            ref_z=ref_z,
            image_width=base_width,
            image_height=base_height,
            pixel_scale=scale,
        )
        self._settings.camera_calibration.calibration = calibration
        self._save_settings()
        return calibration
