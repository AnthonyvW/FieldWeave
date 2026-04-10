"""
camera_calibration.py

Camera-to-stage spatial calibration for vision-guided movement.

This module provides the pure calibration maths and image-processing helpers
that map between image pixel coordinates and physical stage coordinates (in
0.01 mm "tick" units).  It has no dependency on any printer controller,
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
    pixel coordinate into a stage-coordinate delta (in ticks).

Serialisation
-------------
``CameraCalibration.to_dict`` / ``CameraCalibration.from_dict`` round-trip
through plain Python dicts so the caller can persist the result via whatever
config system is in use (e.g. ``MachineVisionSettingsManager``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Calibration state
# ---------------------------------------------------------------------------

@dataclass
class CameraCalibration:
    """
    Immutable snapshot of a completed camera-to-stage calibration.

    All coordinates use "tick" units (0.01 mm) to match the rest of the
    printer controller.

    Attributes
    ----------
    M_est:
        2×2 matrix that maps a world delta (in ticks) to a pixel delta.
        ``pixel_delta = M_est @ world_delta``
    M_inv:
        Inverse of ``M_est``.  Maps a pixel delta to a world delta.
        ``world_delta = M_inv @ pixel_delta``
    ref_x, ref_y, ref_z:
        Stage position (in ticks) where calibration was performed.  Used as
        the origin for absolute vision-guided moves.
    image_width, image_height:
        Resolution of the images used during calibration.
    move_x_ticks, move_y_ticks:
        Calibration move distances (in ticks) that were used to build M_est.
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

    move_x_ticks: int
    move_y_ticks: int

    dpi: float | None = None

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
        (dx_ticks, dy_ticks):
            Stage delta in tick units (0.01 mm).  Add to the current stage
            position to move the clicked point under the camera centre.
        """
        cx = image_center_x if image_center_x is not None else self.image_width / 2.0
        cy = image_center_y if image_center_y is not None else self.image_height / 2.0

        pixel_delta = np.array([[pixel_x - cx], [pixel_y - cy]], dtype=np.float64)
        world_delta = self.M_inv @ pixel_delta

        # M_inv maps a pixel delta to the stage move that *produced* that shift.
        # We want the move that *cancels* it — i.e. brings the clicked point to
        # centre — so we negate once.
        dx_ticks = -float(world_delta[0, 0])
        dy_ticks = -float(world_delta[1, 0])
        return dx_ticks, dy_ticks

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for YAML / JSON storage.

        All values are converted to native Python scalars (``float``, ``int``)
        so that YAML serialisers that reject numpy types (``np.float64`` etc.)
        do not fail.
        """
        def _to_float_list(arr: np.ndarray) -> list[list[float]]:
            return [[float(v) for v in row] for row in arr]

        return {
            "M_est": _to_float_list(self.M_est),
            "M_inv": _to_float_list(self.M_inv),
            "ref_pos_x": self.ref_x,
            "ref_pos_y": self.ref_y,
            "ref_pos_z": self.ref_z,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "move_x_ticks": self.move_x_ticks,
            "move_y_ticks": self.move_y_ticks,
            "dpi": float(self.dpi) if self.dpi is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraCalibration:
        """
        Deserialise from a plain dict (inverse of ``to_dict``).

        Raises ``KeyError`` if required keys are missing and ``ValueError``
        if the stored matrices are not 2×2.
        """
        M_est = np.array(d["M_est"], dtype=np.float64)
        M_inv = np.array(d["M_inv"], dtype=np.float64)

        if M_est.shape != (2, 2) or M_inv.shape != (2, 2):
            raise ValueError("Calibration matrices must be 2×2")

        return cls(
            M_est=M_est,
            M_inv=M_inv,
            ref_x=int(d["ref_pos_x"]),
            ref_y=int(d["ref_pos_y"]),
            ref_z=int(d["ref_pos_z"]),
            image_width=int(d["image_width"]),
            image_height=int(d["image_height"]),
            move_x_ticks=int(d.get("move_x_ticks", 100)),
            move_y_ticks=int(d.get("move_y_ticks", 100)),
            dpi=d.get("dpi"),
        )


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------

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
# DPI calculation
# ---------------------------------------------------------------------------

def compute_dpi_from_matrix(M_est: np.ndarray) -> float | None:
    """
    Derive an approximate camera DPI from a calibration matrix.

    The matrix ``M_est`` maps stage deltas in tick units (0.01 mm) to pixel
    deltas.  The diagonal entries give pixels-per-tick for each axis.  This
    function averages the two diagonal magnitudes, converts to pixels-per-mm,
    then to DPI (pixels per inch, 1 in = 25.4 mm).

    Parameters
    ----------
    M_est:
        2×2 calibration matrix, dtype float64.

    Returns
    -------
    DPI as a float, or ``None`` if the calculation raises an exception.
    """
    try:
        px_per_tick_x = abs(M_est[0, 0])
        px_per_tick_y = abs(M_est[1, 1])
        px_per_tick_avg = (px_per_tick_x + px_per_tick_y) / 2.0
        px_per_mm = px_per_tick_avg * 100.0   # 1 tick = 0.01 mm
        return px_per_mm * 25.4
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Calibration builder
# ---------------------------------------------------------------------------

def build_calibration(
    edges_base: np.ndarray,
    edges_x: np.ndarray,
    edges_y: np.ndarray,
    move_x_ticks: int,
    move_y_ticks: int,
    ref_x: int,
    ref_y: int,
    ref_z: int,
    image_width: int,
    image_height: int,
) -> CameraCalibration:
    """
    Compute the pixel-to-world calibration matrix from three edge maps.

    The caller must supply edge maps captured at three stage positions:

    * ``edges_base`` — at the reference (origin) position.
    * ``edges_x``   — after moving ``move_x_ticks`` in +X from base.
    * ``edges_y``   — after moving ``move_y_ticks`` in +Y from base
      (the stage must have returned to base before this capture).

    Phase correlation between base↔x and base↔y gives the pixel shift
    produced by each known world move.  These two observations directly fill
    the columns of the 2×2 mapping matrix ``M_est``:

    .. code-block:: text

        M_est @ [move_x_ticks, 0     ]ᵀ = [dpx_x, dpy_x]ᵀ
        M_est @ [0,      move_y_ticks]ᵀ = [dpx_y, dpy_y]ᵀ

    Parameters
    ----------
    edges_base, edges_x, edges_y:
        Float32 normalised edge maps of identical shape, produced by
        ``compute_edge_map``.
    move_x_ticks, move_y_ticks:
        Calibration move distances in tick units (0.01 mm).
    ref_x, ref_y, ref_z:
        Stage position in ticks at the time ``edges_base`` was captured.
    image_width, image_height:
        Pixel dimensions of the images used for calibration.

    Returns
    -------
    ``CameraCalibration`` instance ready for ``pixel_to_world_delta`` calls.

    Raises
    ------
    ValueError
        If the world matrix is singular (moves were collinear or too small
        to produce measurable pixel shifts).
    """
    dpx_x, dpy_x, _resp_x = phase_correlation_shift(edges_base, edges_x)
    dpx_y, dpy_y, _resp_y = phase_correlation_shift(edges_base, edges_y)

    # Build 2×2 world and pixel matrices; solve M_est = pixel_mat @ world_inv.
    world_mat = np.array(
        [[move_x_ticks, 0.0],
         [0.0,          move_y_ticks]],
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

    dpi = compute_dpi_from_matrix(M_est)

    return CameraCalibration(
        M_est=M_est,
        M_inv=M_inv,
        ref_x=ref_x,
        ref_y=ref_y,
        ref_z=ref_z,
        image_width=image_width,
        image_height=image_height,
        move_x_ticks=move_x_ticks,
        move_y_ticks=move_y_ticks,
        dpi=dpi,
    )