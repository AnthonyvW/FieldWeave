from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from camera.settings.camera_settings import (
    CameraSettings,
    SettingMetadata,
    SettingType,
    FileFormat,
)
from common.logger import info, error, debug, warning

if TYPE_CHECKING:
    from camera.cameras.base_camera import BaseCamera, CameraResolution

import cv2


# Codecs to probe in preference order. MJPG is first because it removes the
# USB bandwidth bottleneck that causes low frame rates. YUY2 is raw and wide
# compatible. NV12 and H264 are included for cameras that support them.
_CANDIDATE_CODECS: list[str] = ["MJPG", "YUY2", "NV12", "H264"]


def _probe_supported_codecs(cap: cv2.VideoCapture) -> list[str]:
    """Determine which codecs the camera driver actually accepts.

    For each candidate fourcc, sets CAP_PROP_FOURCC, grabs a frame to flush
    the driver's format negotiation, then reads back the active fourcc. A codec
    is considered supported only when the readback matches the request.
    The original codec is restored afterward.
    """
    saved_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    saved_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    saved_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    supported: list[str] = []

    for name in _CANDIDATE_CODECS:
        fourcc = cv2.VideoWriter.fourcc(*name)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.grab()
        actual = int(cap.get(cv2.CAP_PROP_FOURCC))
        if actual == fourcc:
            supported.append(name)
            debug(f"USB codec supported: {name}")
        else:
            actual_str = "".join(chr((actual >> 8 * i) & 0xFF) for i in range(4)).rstrip("\x00")
            debug(f"USB codec {name} not accepted (got {actual_str!r})")

    # Restore original codec and dimensions
    cap.set(cv2.CAP_PROP_FOURCC, saved_fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, saved_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, saved_h)
    cap.grab()

    return supported


# Maps each settings field name to its cv2 property constant.
_PROP_MAP: dict[str, int] = {
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast":   cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "hue":        cv2.CAP_PROP_HUE,
    "gain":       cv2.CAP_PROP_GAIN,
    "gamma":      cv2.CAP_PROP_GAMMA,
    "exposure":   cv2.CAP_PROP_EXPOSURE,
}


def _probe_prop_ranges(cap: cv2.VideoCapture) -> dict[str, tuple[float, float, float] | None]:
    """Probe which image-control properties the camera actually supports and find
    their true min/max/step via exponential-walk + bisect.

    Auto WB is disabled before probing so white-balance drift doesn't affect
    readbacks, and restored afterward. Each property is restored to its original
    value when done.

    Returns a dict keyed by field name (matching ``_PROP_MAP``).
    ``None`` means the property exists in OpenCV but is not writable on this device.
    """
    cap.grab()
    saved_auto_wb = cap.get(cv2.CAP_PROP_AUTO_WB)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.grab()

    saved: dict[str, float] = {
        name: cap.get(prop_id) for name, prop_id in _PROP_MAP.items()
    }
    ranges: dict[str, tuple[float, float, float] | None] = {}

    for name, prop_id in _PROP_MAP.items():
        current = saved[name]

        writable = False
        for delta in (1.0, -1.0):
            cap.set(prop_id, current + delta)
            cap.grab()
            if cap.get(prop_id) != current:
                writable = True
                cap.set(prop_id, current)
                cap.grab()
                break

        if not writable:
            ranges[name] = None
            debug(f"USB prop {name!r} is not writable on this camera")
            continue

        step = 1.0
        for candidate in (1.0, 2.0, 5.0, 10.0, 100.0):
            cap.set(prop_id, current + candidate)
            cap.grab()
            actual = cap.get(prop_id)
            if actual != current:
                step = abs(actual - current)
                cap.set(prop_id, current)
                cap.grab()
                break

        def find_bound(
            direction: float,
            _current: float = current,
            _step: float = step,
            _prop_id: int = prop_id,
        ) -> float:
            last_accepted = _current
            increment = _step
            while increment < 1_000_000:
                target = _current + direction * increment
                cap.set(_prop_id, target)
                cap.grab()
                actual = cap.get(_prop_id)
                if actual == last_accepted:
                    lo = last_accepted
                    hi = target
                    for _ in range(40):
                        mid = lo + direction * abs(hi - lo) / 2.0
                        cap.set(_prop_id, mid)
                        cap.grab()
                        got = cap.get(_prop_id)
                        if got != lo:
                            lo = got
                        else:
                            hi = mid
                        if abs(hi - lo) <= _step * 0.5:
                            break
                    cap.set(_prop_id, _current)
                    cap.grab()
                    return lo
                last_accepted = actual
                increment *= 2
            cap.set(_prop_id, _current)
            cap.grab()
            return last_accepted

        prop_min = find_bound(-1.0)
        prop_max = find_bound(1.0)
        cap.set(prop_id, current)
        ranges[name] = (prop_min, prop_max, step)
        debug(f"USB prop {name!r}: min={prop_min}, max={prop_max}, step={step}")

    cap.set(cv2.CAP_PROP_AUTO_WB, saved_auto_wb)
    return ranges


@dataclass
class UsbCameraSettings(CameraSettings):
    version: str = "0"
    auto_exposure: bool = True
    exposure: int = -6
    exposure_time: int = 0
    preview_resolution: str = ""
    still_resolution: str = ""
    contrast: int = 0
    hue: int = 0
    saturation: int = 64
    brightness: int = 0
    gain: int = 0
    gamma: int = 100
    fformat: FileFormat = FileFormat.TIFF

    # Populated on first open, persisted so slow probes never repeat.
    # Keys match field names in _PROP_MAP; None means unsupported on this device.
    # Stored as list[float] triples rather than tuples since YAML/JSON round-trips
    # don't preserve tuple types.
    cached_resolutions: list[str] = field(default_factory=list)
    cached_prop_ranges: dict[str, list[float] | None] = field(default_factory=dict)
    cached_codecs: list[str] = field(default_factory=list)
    codec: str = "MJPG"

    _camera: BaseCamera | None = field(default=None, repr=False, compare=False, init=False)
    _ui_update_callback: Any | None = field(default=None, repr=False, compare=False, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._camera = None
        self._ui_update_callback = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_cap(self) -> cv2.VideoCapture | None:
        if self._camera is None or not hasattr(self._camera, "_cap"):
            return None
        cap: cv2.VideoCapture = self._camera._cap
        return cap if cap is not None and cap.isOpened() else None

    def _prop_range(self, name: str) -> tuple[float, float, float] | None:
        entry = self.cached_prop_ranges.get(name)
        if entry is None:
            return None
        return (entry[0], entry[1], entry[2])

    def _prop_supported(self, name: str) -> bool:
        return self.cached_prop_ranges.get(name) is not None

    def _clamp(self, name: str, value: int) -> int:
        r = self._prop_range(name)
        if r is None:
            return value
        return int(max(r[0], min(r[1], value)))

    def _set_prop(self, name: str, prop_id: int, value: float) -> None:
        if self.cached_prop_ranges and not self._prop_supported(name):
            return
        cap = self._get_cap()
        if cap is not None:
            cap.set(prop_id, value)
            cap.grab()

    # ------------------------------------------------------------------
    # Metadata — only expose controls the camera actually supports
    # ------------------------------------------------------------------

    def get_metadata(self) -> list[SettingMetadata]:
        resolutions = self.get_preview_resolutions()
        resolution_choices = [f"{r.width}x{r.height}" for r in resolutions]

        probed = bool(self.cached_prop_ranges)

        def _range_args(name: str, fallback_min: int, fallback_max: int) -> dict[str, Any]:
            r = self._prop_range(name)
            if r is not None:
                return {"min_value": int(r[0]), "max_value": int(r[1])}
            return {"min_value": fallback_min, "max_value": fallback_max}

        entries: list[SettingMetadata] = [
            SettingMetadata(
                name="preview_resolution",
                display_name="Preview Resolution",
                setting_type=SettingType.DROPDOWN,
                description="Camera preview resolution",
                choices=resolution_choices,
                group="Capture",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="codec",
                display_name="Codec",
                setting_type=SettingType.DROPDOWN,
                description="Capture codec. MJPG greatly increases frame rate at higher resolutions by reducing USB bandwidth. Requires camera restart.",
                choices=self.cached_codecs if self.cached_codecs else _CANDIDATE_CODECS,
                group="Capture",
                runtime_changeable=False,
            ),
            SettingMetadata(
                name="fformat",
                display_name="File Format",
                setting_type=SettingType.DROPDOWN,
                description="Default file format for saved images",
                choices=self._file_formats,
                group="Capture",
                runtime_changeable=True,
            ),
            SettingMetadata(
                name="auto_exposure",
                display_name="Auto Exposure",
                setting_type=SettingType.BOOL,
                description="Enable automatic exposure control",
                group="Exposure",
                runtime_changeable=True,
            ),
        ]

        if not probed or self._prop_supported("exposure"):
            entries.append(SettingMetadata(
                name="exposure",
                display_name="Exposure",
                setting_type=SettingType.RANGE,
                description="Manual exposure value (log2 seconds, camera-specific)",
                group="Exposure",
                runtime_changeable=True,
                controlled_by="auto_exposure",
                **_range_args("exposure", -13, 0),
            ))

        if not probed or self._prop_supported("brightness"):
            entries.append(SettingMetadata(
                name="brightness",
                display_name="Brightness",
                setting_type=SettingType.RANGE,
                description="Image brightness",
                group="Image",
                runtime_changeable=True,
                **_range_args("brightness", -64, 64),
            ))

        if not probed or self._prop_supported("contrast"):
            entries.append(SettingMetadata(
                name="contrast",
                display_name="Contrast",
                setting_type=SettingType.RANGE,
                description="Image contrast",
                group="Image",
                runtime_changeable=True,
                **_range_args("contrast", 0, 95),
            ))

        if not probed or self._prop_supported("saturation"):
            entries.append(SettingMetadata(
                name="saturation",
                display_name="Saturation",
                setting_type=SettingType.RANGE,
                description="Image saturation",
                group="Image",
                runtime_changeable=True,
                **_range_args("saturation", 0, 100),
            ))

        if not probed or self._prop_supported("hue"):
            entries.append(SettingMetadata(
                name="hue",
                display_name="Hue",
                setting_type=SettingType.RANGE,
                description="Image hue",
                group="Image",
                runtime_changeable=True,
                **_range_args("hue", -180, 180),
            ))

        if not probed or self._prop_supported("gain"):
            entries.append(SettingMetadata(
                name="gain",
                display_name="Gain",
                setting_type=SettingType.RANGE,
                description="Image gain",
                group="Image",
                runtime_changeable=True,
                **_range_args("gain", 0, 100),
            ))

        if not probed or self._prop_supported("gamma"):
            entries.append(SettingMetadata(
                name="gamma",
                display_name="Gamma",
                setting_type=SettingType.RANGE,
                description="Image gamma",
                group="Image",
                runtime_changeable=True,
                **_range_args("gamma", 100, 500),
            ))

        return entries

    # ------------------------------------------------------------------
    # Probe — runs once on a background thread, called from UsbCamera
    # ------------------------------------------------------------------

    def probe_and_cache(self) -> None:
        """Run both the resolution and property-range probes, populating the cache.

        Intentionally slow — must be called from a background thread. After this
        returns the camera class saves settings to disk so subsequent opens skip
        both probes entirely.
        """
        cap = self._get_cap()
        if cap is None:
            warning("Cannot probe: camera not open")
            return

        self._probe_codecs(cap)
        self._probe_resolutions(cap)
        self._probe_prop_ranges(cap)
        debug("USB camera probe complete")

    def _probe_codecs(self, cap: cv2.VideoCapture) -> None:
        supported = _probe_supported_codecs(cap)
        self.cached_codecs = supported

        if not supported:
            debug("No candidate codecs accepted; leaving codec unchanged")
            return

        # If the saved codec is supported keep it, otherwise prefer MJPG then
        # fall back to the first accepted codec.
        if self.codec in supported:
            preferred = self.codec
        elif "MJPG" in supported:
            preferred = "MJPG"
        else:
            preferred = supported[0]

        self.set_codec(preferred, cap=cap)

    def set_codec(self, value: str, index: int | None = None, cap: cv2.VideoCapture | None = None) -> bool:
        """Set the capture codec.

        The codec must be applied to the VideoCapture before streaming starts.
        Changing it while streaming requires a camera restart (runtime_changeable=False
        in metadata), which the settings UI handles by saving and restarting.

        Args:
            value: Fourcc string, e.g. ``"MJPG"`` or ``"YUY2"``.
            cap:   VideoCapture to apply to immediately. When None the value is
                   stored only; ``UsbCamera.open`` applies it at next open.
        """
        choices = self.cached_codecs if self.cached_codecs else _CANDIDATE_CODECS
        if value not in choices:
            error(f"Codec {value!r} not in supported list {choices}")
            return False

        self.codec = value

        target_cap = cap if cap is not None else self._get_cap()
        if target_cap is None:
            return True

        fourcc = cv2.VideoWriter.fourcc(*value)
        target_cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        target_cap.grab()
        actual = int(target_cap.get(cv2.CAP_PROP_FOURCC))
        if actual != fourcc:
            actual_str = "".join(chr((actual >> 8 * i) & 0xFF) for i in range(4)).rstrip("\x00")
            warning(f"Codec {value!r} set but driver reported {actual_str!r}")
            return False

        debug(f"Codec set to {value}")
        return True

    def _probe_resolutions(self, cap: cv2.VideoCapture) -> None:
        from camera.cameras.base_camera import CameraResolution  # pylint: disable=import-outside-toplevel

        candidates = [
            (3840, 2160),
            (2560, 1440),
            (1920, 1080),
            (1600, 1200),
            (1280, 960),
            (1280, 720),
            (1024, 768),
            (800, 600),
            (640, 480),
            (320, 240),
        ]

        saved_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        saved_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        found: list[CameraResolution] = []
        seen: set[tuple[int, int]] = set()

        for w, h in candidates:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (actual_w, actual_h) not in seen:
                seen.add((actual_w, actual_h))
                found.append(CameraResolution(width=actual_w, height=actual_h))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, saved_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, saved_h)

        found.sort(key=lambda r: r.width * r.height, reverse=True)
        self.cached_resolutions = [f"{r.width}x{r.height}" for r in found]
        debug(f"Probed {len(found)} USB camera resolutions")

    def _probe_prop_ranges(self, cap: cv2.VideoCapture) -> None:
        raw = _probe_prop_ranges(cap)
        self.cached_prop_ranges = {
            name: list(r) if r is not None else None
            for name, r in raw.items()
        }

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def get_preview_resolutions(self) -> list[CameraResolution]:
        from camera.cameras.base_camera import CameraResolution  # pylint: disable=import-outside-toplevel

        if self.cached_resolutions:
            result = []
            for entry in self.cached_resolutions:
                parts = entry.split("x")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    result.append(CameraResolution(width=int(parts[0]), height=int(parts[1])))
            if result:
                return result

        cap = self._get_cap()
        if cap is None:
            return []
        self._probe_resolutions(cap)
        return self.get_preview_resolutions()

    def get_current_preview_resolution(self) -> tuple[int, int, int]:
        cap = self._get_cap()
        if cap is None:
            return (0, 0, 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (0, w, h)

    def get_still_resolutions(self) -> list[CameraResolution]:
        return []

    # ------------------------------------------------------------------
    # Resolution setters
    # ------------------------------------------------------------------

    def set_preview_resolution(self, value: str, index: int | None = None) -> bool:
        resolutions = self.get_preview_resolutions()
        choices = [f"{r.width}x{r.height}" for r in resolutions]

        if index is not None:
            if not (0 <= index < len(choices)):
                error(f"Invalid preview resolution index: {index}")
                return False
            value = choices[index]

        if value not in choices:
            error(f"Preview resolution {value!r} not available")
            return False

        cap = self._get_cap()
        if cap is None:
            self.preview_resolution = value
            return True

        w, h = (int(x) for x in value.split("x"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        actual = f"{actual_w}x{actual_h}"
        if actual != value:
            warning(f"Requested resolution {value} but camera reported {actual}")

        self.preview_resolution = actual
        debug(f"Preview resolution set to {actual}")
        return True

    def set_still_resolution(self, value: str, index: int | None = None) -> bool:
        return True

    # ------------------------------------------------------------------
    # Exposure
    # ------------------------------------------------------------------

    def get_exposure_time(self) -> int:
        return self.exposure_time

    def set_exposure_time(self, time_us: int) -> bool:
        self.exposure_time = time_us
        return True

    def set_auto_exposure(self, enabled: bool) -> None:
        self.auto_exposure = enabled
        cap = self._get_cap()
        if cap is None:
            return
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if enabled else 1)
        if not enabled:
            self._set_prop("exposure", cv2.CAP_PROP_EXPOSURE, self.exposure)

    def set_exposure(self, value: int) -> None:
        self.exposure = self._clamp("exposure", value)
        if self.auto_exposure:
            return
        self._set_prop("exposure", cv2.CAP_PROP_EXPOSURE, self.exposure)

    # ------------------------------------------------------------------
    # Image controls
    # ------------------------------------------------------------------

    def set_contrast(self, value: int) -> None:
        self.contrast = self._clamp("contrast", value)
        self._set_prop("contrast", cv2.CAP_PROP_CONTRAST, self.contrast)

    def set_hue(self, value: int) -> None:
        self.hue = self._clamp("hue", value)
        self._set_prop("hue", cv2.CAP_PROP_HUE, self.hue)

    def set_gain(self, value: int) -> None:
        self.gain = self._clamp("gain", value)
        self._set_prop("gain", cv2.CAP_PROP_GAIN, self.gain)

    def set_saturation(self, value: int) -> None:
        self.saturation = self._clamp("saturation", value)
        self._set_prop("saturation", cv2.CAP_PROP_SATURATION, self.saturation)

    def set_brightness(self, value: int) -> None:
        self.brightness = self._clamp("brightness", value)
        self._set_prop("brightness", cv2.CAP_PROP_BRIGHTNESS, self.brightness)

    def set_gamma(self, value: int) -> None:
        self.gamma = self._clamp("gamma", value)
        self._set_prop("gamma", cv2.CAP_PROP_GAMMA, self.gamma)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def get_camera_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}

        if self._camera is not None:
            metadata['model'] = self._camera.model

        metadata['exposure_time_us'] = self.get_exposure_time()
        metadata['auto_exposure'] = self.auto_exposure
        metadata['exposure'] = self.exposure
        metadata['brightness'] = self.brightness
        metadata['contrast'] = self.contrast
        metadata['saturation'] = self.saturation
        metadata['hue'] = self.hue
        metadata['gain'] = self.gain
        metadata['gamma'] = self.gamma
        metadata['codec'] = self.codec
        metadata['preview_resolution'] = self.preview_resolution
        metadata['file_format'] = self.fformat.value

        return metadata

    def supports_still_capture(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # apply / refresh
    # ------------------------------------------------------------------

    def apply_to_camera(self, camera: BaseCamera) -> None:
        self._camera = camera
        info(f"Applying settings to USB camera {camera.model}")

        # Codec must be set before resolution and before any frame is grabbed.
        # MSMF honours CAP_PROP_FOURCC only when set before the first read.
        if self.codec:
            cap = self._get_cap()
            if cap is not None:
                fourcc = cv2.VideoWriter.fourcc(*self.codec)
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                debug(f"Set codec to {self.codec} before first grab")

        # Resolution is applied next, also before any grab.
        if self.preview_resolution:
            cap = self._get_cap()
            if cap is not None:
                w, h = (int(x) for x in self.preview_resolution.split("x"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.preview_resolution = f"{actual_w}x{actual_h}"
                debug(f"Set resolution to {self.preview_resolution}")

        self.set_auto_exposure(self.auto_exposure)
        self.set_exposure(self.exposure)
        self.set_brightness(self.brightness)
        self.set_contrast(self.contrast)
        self.set_saturation(self.saturation)
        self.set_hue(self.hue)
        self.set_gain(self.gain)
        self.set_gamma(self.gamma)

        debug("Applied settings to USB camera")

    def refresh_from_camera(self, camera: BaseCamera) -> None:
        self._camera = camera
        info(f"Refreshing USB camera settings from {camera.model}")

        cap = self._get_cap()
        if cap is None:
            error("USB camera capture not available for refresh")
            return

        auto_exp_val = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        self.auto_exposure = auto_exp_val >= 2

        if self._prop_supported("exposure"):
            self.exposure = int(cap.get(cv2.CAP_PROP_EXPOSURE))
        if self._prop_supported("brightness"):
            self.brightness = int(cap.get(cv2.CAP_PROP_BRIGHTNESS))
        if self._prop_supported("contrast"):
            self.contrast = int(cap.get(cv2.CAP_PROP_CONTRAST))
        if self._prop_supported("saturation"):
            self.saturation = int(cap.get(cv2.CAP_PROP_SATURATION))
        if self._prop_supported("hue"):
            self.hue = int(cap.get(cv2.CAP_PROP_HUE))
        if self._prop_supported("gain"):
            self.gain = int(cap.get(cv2.CAP_PROP_GAIN))
        if self._prop_supported("gamma"):
            self.gamma = int(cap.get(cv2.CAP_PROP_GAMMA))

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.preview_resolution = f"{w}x{h}"

        active_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        active_codec = "".join(chr((active_fourcc >> 8 * i) & 0xFF) for i in range(4)).rstrip("\x00")
        if active_codec in (self.cached_codecs or _CANDIDATE_CODECS):
            self.codec = active_codec

        info("Refreshed USB camera settings")