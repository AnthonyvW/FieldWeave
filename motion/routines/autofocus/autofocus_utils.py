"""
Shared helpers for autofocus routines.

Kept in a private module so the three routine files stay focused on their
own logic without duplicating the move-and-wait and scoring patterns.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from motion.motion_controller_manager import MotionControllerManager
from motion.models import Position
from common.logger import error

_NM_PER_MM = 1_000_000


def quantize(z_nm: int, step_nm: int) -> int:
    """Snap *z_nm* down to the nearest multiple of *step_nm*."""
    return (z_nm // step_nm) * step_nm


def move_z_and_wait(
    motion: MotionControllerManager,
    z_nm: int,
    z_floor_nm: int = 0,
) -> None:
    """
    Issue an absolute Z move and block until the axis reaches the target.

    The position is clamped to *z_floor_nm* before issuing the move.
    ``move_to_position`` blocks until the move is complete, so this returns
    as soon as the stage has settled.
    """
    target = max(z_nm, z_floor_nm)
    current = motion.get_position()
    motion.move_to_position(Position(x=current.x, y=current.y, z=target), wait=True)


def capture_still_frame(
    camera_manager,
    timeout_s: float = 10.0,
) -> np.ndarray | None:
    """
    Trigger a still capture and block until the frame is available.

    Calls ``camera_manager.capture_still`` with an ``on_complete`` callback,
    then waits on a threading event.  Returns the captured numpy array, or
    None if the capture timed out or failed.

    Parameters
    ----------
    camera_manager:
        The :class:`CameraManager` instance from the app context.
    timeout_s:
        Maximum seconds to wait for the frame to arrive.
    """
    done = threading.Event()
    result: list[np.ndarray | None] = [None]

    def _on_complete(success: bool, frame: np.ndarray | None) -> None:
        if success and frame is not None:
            result[0] = frame
        else:
            error(
                f"capture_still_frame: callback called with success={success}, "
                f"frame={'None' if frame is None else frame.shape}"
            )
        done.set()

    success = camera_manager.capture_still(on_complete=_on_complete)
    if not success:
        error("capture_still_frame: capture_still() returned False — camera not ready")
        return None

    if not done.wait(timeout=timeout_s):
        error(f"capture_still_frame: timed out waiting for frame after {timeout_s:.1f} s")
        return None

    return result[0]


def score_still_frame(
    camera_manager,
    mv,
    settle_s: float = 0.4,
    log_prefix: str = "",
) -> float:
    """
    Capture a still frame and return its peak focus score.

    Parameters
    ----------
    camera_manager:
        The :class:`CameraManager` instance from the app context.
    mv:
        The machine-vision manager from the app context.
    settle_s:
        Seconds to wait before capturing.
    log_prefix:
        Prepended to error messages, e.g. ``"[Autofocus]"``.
    """
    if settle_s > 0:
        time.sleep(settle_s)
    frame = capture_still_frame(camera_manager)
    if frame is None:
        error(f"{log_prefix} score_still_frame: no frame from capture_still_frame")
        return float("-inf")
    try:
        future = mv.request_focus_analysis_guaranteed(frame)
        return float(future.result(timeout=10.0).scores.peak)
    except Exception as exc:
        error(f"{log_prefix} score_still_frame: focus analysis failed: {exc!r}")
        return float("-inf")


def score_preview_frame(
    camera_manager,
    mv,
    settle_s: float = 0.4,
    log_prefix: str = "",
) -> float:
    """
    Score the current preview frame for focus.

    Parameters
    ----------
    camera_manager:
        The :class:`CameraManager` instance from the app context.
    mv:
        The machine-vision manager from the app context.
    settle_s:
        Seconds to wait before grabbing the frame.
    log_prefix:
        Prepended to error messages, e.g. ``"[Autofocus]"``.
    """
    if settle_s > 0:
        time.sleep(settle_s)
    frame = camera_manager.copy_current_frame_to_numpy()
    if frame is None:
        error(f"{log_prefix} score_preview_frame: no current frame available")
        return float("-inf")
    try:
        future = mv.request_focus_analysis_guaranteed(frame)
        return float(future.result(timeout=10.0).scores.peak)
    except Exception as exc:
        error(f"{log_prefix} score_preview_frame: focus analysis failed: {exc!r}")
        return float("-inf")