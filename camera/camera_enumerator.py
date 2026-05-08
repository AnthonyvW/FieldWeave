"""
Camera enumeration system with plugin architecture.
Supports multiple camera types through enumerator plugins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from common.logger import error, exception, debug


class CameraType(Enum):
    """Supported camera types"""
    AMSCOPE = "amscope"
    GENERIC_USB = "generic_usb"


@dataclass
class CameraInfo:
    """
    Information about an available camera.
    Lightweight object returned by enumeration before camera instantiation.
    """
    camera_type: CameraType
    device_id: str
    display_name: str
    model: str | None = None
    manufacturer: str | None = None
    serial_number: str | None = None
    max_resolution: tuple[int, int] | None = None
    metadata: dict[str, Any] | None = None

    def __str__(self) -> str:
        parts = [self.display_name]
        if self.model:
            parts.append(f"({self.model})")
        if self.serial_number:
            parts.append(f"SN:{self.serial_number}")
        return " ".join(parts)

    def __repr__(self) -> str:
        return f"CameraInfo({self.camera_type.value}, {self.display_name})"


class CameraEnumerator(ABC):
    """
    Base class for camera enumerators.
    Each camera type implements this to provide enumeration capability.
    """

    @abstractmethod
    def enumerate(self) -> list[CameraInfo]:
        pass

    @abstractmethod
    def get_camera_type(self) -> CameraType:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


class AmscopeEnumerator(CameraEnumerator):
    """Enumerator for Amscope cameras"""

    def __init__(self):
        self._sdk = None

    def get_camera_type(self) -> CameraType:
        return CameraType.AMSCOPE

    def is_available(self) -> bool:
        if self._sdk is not None:
            return True
        from camera.cameras.amscope_camera import _get_amcam  # noqa: PLC0415
        self._sdk = _get_amcam()
        if self._sdk is None:
            error("Failed to load Amscope SDK")
        return self._sdk is not None

    def enumerate(self) -> list[CameraInfo]:
        if not self.is_available():
            error("Amscope SDK not available, cannot enumerate cameras")
            return []

        cameras = []
        try:
            device_list = self._sdk.Amcam.EnumV2()
            debug(f"Amscope enumerator found {len(device_list)} camera(s)")

            for idx, device in enumerate(device_list):
                try:
                    model_name = device.model.name if device.model else "Unknown"
                    max_res = None
                    if device.model and device.model.res and len(device.model.res) > 0:
                        max_res = (device.model.res[0].width, device.model.res[0].height)

                    cameras.append(CameraInfo(
                        camera_type=CameraType.AMSCOPE,
                        device_id=device.id,
                        display_name=device.displayname or f"Amscope Camera {idx}",
                        model=model_name,
                        manufacturer="Amscope",
                        serial_number=None,
                        max_resolution=max_res,
                        metadata={
                            "device_index": idx,
                            "model_info": device.model,
                        },
                    ))
                except Exception as e:
                    exception(f"Error processing Amscope device {idx}: {e}")
                    continue

        except Exception as e:
            exception(f"Error enumerating Amscope cameras: {e}")

        return cameras


def _backend_from_index(index: int) -> int:
    """Recover the backend constant from a cv2_enumerate_cameras-encoded index.

    cv2_enumerate_cameras encodes the backend base offset into the index
    (e.g. DSHOW camera 0 -> index 700, MSMF camera 0 -> index 1400).
    VideoCapture expects a 0-based local index when a backend is specified,
    so callers must subtract the base before opening.
    """
    import cv2  # noqa: PLC0415
    known: list[tuple[int, int]] = [
        (cv2.CAP_MSMF,  cv2.CAP_MSMF),
        (cv2.CAP_DSHOW, cv2.CAP_DSHOW),
        (cv2.CAP_VFW,   cv2.CAP_VFW),
        (cv2.CAP_V4L2,  cv2.CAP_V4L2),
    ]
    for base, backend in known:
        if base <= index < base + 100:
            return backend
    return cv2.CAP_ANY


class GenericUSBEnumerator(CameraEnumerator):
    """Enumerator for generic USB/webcam cameras.

    Uses cv2_enumerate_cameras for reliable device discovery with real names,
    VID/PID, and correct backend-encoded indices. Falls back to blind index
    probing (0–7) if the library is not installed.

    Each physical device (identified by VID+PID) appears once. When the same
    device is available on multiple backends, DSHOW is preferred since it
    honours property writes more reliably on Windows.

    Virtual cameras and capture devices without a VID/PID (screen capture,
    VR headsets, etc.) are excluded.
    """

    def get_camera_type(self) -> CameraType:
        return CameraType.GENERIC_USB

    def is_available(self) -> bool:
        try:
            import cv2  # noqa: PLC0415
            return True
        except ImportError:
            return False

    def enumerate(self) -> list[CameraInfo]:
        import cv2  # noqa: PLC0415

        try:
            from cv2_enumerate_cameras import enumerate_cameras  # noqa: PLC0415
            return self._enumerate_with_library(cv2, enumerate_cameras)
        except ImportError:
            debug("cv2_enumerate_cameras not available, falling back to index probe")
            return self._enumerate_by_index(cv2)

    def _enumerate_with_library(self, cv2, enumerate_cameras) -> list[CameraInfo]:
        import os  # noqa: PLC0415
        import sys  # noqa: PLC0415

        # cv2_enumerate_cameras can spam stderr with driver warnings on Windows
        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        try:
            raw = list(enumerate_cameras(cv2.CAP_ANY))
        finally:
            sys.stderr.close()
            sys.stderr = original_stderr

        # Drop virtual/capture devices that have no VID/PID
        raw = [c for c in raw if c.vid is not None and c.pid is not None]

        # Deduplicate: same physical device can appear multiple times per backend
        seen_keys: set[tuple] = set()
        deduped = []
        for c in raw:
            key = (c.vid, c.pid, c.index)
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(c)

        # One entry per unique physical device; prefer MSMF since it honours
        # CAP_PROP_FOURCC when set before the first read, which is required for
        # codec selection (e.g. MJPG for full frame rate). DSHOW silently ignores
        # the fourcc on most drivers.
        best: dict[tuple, Any] = {}
        for c in deduped:
            key = (c.vid, c.pid)
            existing = best.get(key)
            backend = _backend_from_index(c.index)
            if existing is None or (backend == cv2.CAP_MSMF and _backend_from_index(existing.index) != cv2.CAP_MSMF):
                best[key] = c

        cameras: list[CameraInfo] = []
        for (vid, pid), c in best.items():
            backend = _backend_from_index(c.index)
            local_index = c.index - backend

            cap = cv2.VideoCapture(local_index, backend)
            if not cap.isOpened():
                cap.release()
                debug(f"Could not open USB camera {c.name!r} at local index {local_index}")
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            max_res = (width, height) if width > 0 and height > 0 else None
            display_name = c.name or f"USB Camera (VID={vid:#06x} PID={pid:#06x})"

            cameras.append(CameraInfo(
                camera_type=CameraType.GENERIC_USB,
                device_id=str(local_index),
                display_name=display_name,
                model=display_name,
                manufacturer=None,
                serial_number=None,
                max_resolution=max_res,
                metadata={
                    "local_index": local_index,
                    "backend": backend,
                    "vid": vid,
                    "pid": pid,
                },
            ))

        debug(f"Generic USB enumerator found {len(cameras)} camera(s)")
        return cameras

    def _enumerate_by_index(self, cv2) -> list[CameraInfo]:
        cameras: list[CameraInfo] = []

        for index in range(8):
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            max_res = (width, height) if width > 0 and height > 0 else None

            cameras.append(CameraInfo(
                camera_type=CameraType.GENERIC_USB,
                device_id=str(index),
                display_name=f"USB Camera {index}",
                model=f"USB Camera {index}",
                manufacturer=None,
                serial_number=None,
                max_resolution=max_res,
                metadata={
                    "local_index": index,
                    "backend": cv2.CAP_MSMF,
                    "vid": None,
                    "pid": None,
                },
            ))

        debug(f"Generic USB enumerator (index probe) found {len(cameras)} camera(s)")
        return cameras