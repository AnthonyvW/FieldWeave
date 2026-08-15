#!/usr/bin/env python3
"""
Camera Detection Script
Detects all available cameras using OpenCV and displays their capabilities.
"""

from __future__ import annotations

import argparse
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import sys
import os

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

try:
    from cv2_enumerate_cameras import enumerate_cameras
except ImportError:
    print("Error: cv2_enumerate_cameras is not installed")
    print("Install it with: pip install opencv-camera-enumeration")
    sys.exit(1)


def get_camera_properties(cap: cv2.VideoCapture, camera_info: dict) -> dict[str, float | str | int]:
    """Get detailed properties of a camera."""
    vid = camera_info.get('vid')
    pid = camera_info.get('pid')
    
    properties = {
        'Index': camera_info['index'],
        'Name': camera_info.get('name', 'Unknown'),
        'Path': camera_info.get('path', 'Unknown'),
        'VID': f"0x{vid:04X}" if vid is not None else "N/A",
        'PID': f"0x{pid:04X}" if pid is not None else "N/A",
        'Backend': camera_info.get('backend_name', 'Unknown'),
        'Width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        'Height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        'FPS': cap.get(cv2.CAP_PROP_FPS),
        'Codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
        'Brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
        'Contrast': cap.get(cv2.CAP_PROP_CONTRAST),
        'Saturation': cap.get(cv2.CAP_PROP_SATURATION),
        'Hue': cap.get(cv2.CAP_PROP_HUE),
        'Gain': cap.get(cv2.CAP_PROP_GAIN),
        'Exposure': cap.get(cv2.CAP_PROP_EXPOSURE),
        'Auto Exposure': cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        'Auto Focus': cap.get(cv2.CAP_PROP_AUTOFOCUS),
        'Auto WB': cap.get(cv2.CAP_PROP_AUTO_WB),
        'White Balance (K)': cap.get(cv2.CAP_PROP_TEMPERATURE),
        'Buffer Size': cap.get(cv2.CAP_PROP_BUFFERSIZE),
    }
    
    # Convert FOURCC code to readable format
    fourcc = properties['Codec']
    if fourcc > 0:
        properties['Codec_String'] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    else:
        properties['Codec_String'] = 'Unknown'
    
    return properties


def get_property_ranges(cap: cv2.VideoCapture) -> dict[str, tuple[float, float, float] | None]:
    """Probe min/max/step ranges for adjustable camera properties.

    Steps outward from the current value doubling each iteration. The boundary
    is detected when the readback stops advancing (i.e. equals the previous
    accepted value), meaning the hardware has clamped. A bisect then finds the
    exact edge between the last accepted step and the first clamped one.

    Auto WB is disabled before probing and restored afterward.

    Returns a dict of name -> (min, max, step) or None if not writable.
    """
    adjustable_props = {
        'Brightness':    cv2.CAP_PROP_BRIGHTNESS,
        'Contrast':      cv2.CAP_PROP_CONTRAST,
        'Saturation':    cv2.CAP_PROP_SATURATION,
        'Hue':           cv2.CAP_PROP_HUE,
        'Gain':          cv2.CAP_PROP_GAIN,
        'Exposure':      cv2.CAP_PROP_EXPOSURE,
        'Sharpness':     cv2.CAP_PROP_SHARPNESS,
        'Gamma':         cv2.CAP_PROP_GAMMA,
        'White Balance': cv2.CAP_PROP_TEMPERATURE,
        'Backlight':     cv2.CAP_PROP_BACKLIGHT,
        'Auto Exposure': cv2.CAP_PROP_AUTO_EXPOSURE,
        'Auto Focus':    cv2.CAP_PROP_AUTOFOCUS,
        'Pan':           cv2.CAP_PROP_PAN,
        'Tilt':          cv2.CAP_PROP_TILT,
        'Zoom':          cv2.CAP_PROP_ZOOM,
        'Focus':         cv2.CAP_PROP_FOCUS,
        'Iris':          cv2.CAP_PROP_IRIS,
    }

    cap.grab()
    saved_auto_wb = cap.get(cv2.CAP_PROP_AUTO_WB)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.grab()

    saved = {name: cap.get(prop_id) for name, prop_id in adjustable_props.items()}
    ranges: dict[str, tuple[float, float, float] | None] = {}

    for name, prop_id in adjustable_props.items():
        current = saved[name]

        # Confirm writability.
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
            continue

        # Find step size: smallest positive delta the camera accepts.
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

        def find_bound(direction: float) -> float:
            last_accepted = current
            increment = step
            # Phase 1: exponential walk until readback stops advancing.
            while increment < 1_000_000:
                target = current + direction * increment
                cap.set(prop_id, target)
                cap.grab()
                actual = cap.get(prop_id)
                if actual == last_accepted:
                    # Readback stopped moving — boundary is between last_accepted
                    # and current target. Bisect to find the exact edge.
                    lo = last_accepted
                    hi = target
                    for _ in range(40):
                        mid = lo + direction * abs(hi - lo) / 2.0
                        cap.set(prop_id, mid)
                        cap.grab()
                        got = cap.get(prop_id)
                        if got != lo:
                            lo = got
                        else:
                            hi = mid
                        if abs(hi - lo) <= step * 0.5:
                            break
                    cap.set(prop_id, current)
                    cap.grab()
                    return lo
                last_accepted = actual
                increment *= 2
            cap.set(prop_id, current)
            cap.grab()
            return last_accepted

        prop_min = find_bound(-1.0)
        prop_max = find_bound(1.0)
        cap.set(prop_id, current)
        ranges[name] = (prop_min, prop_max, step)

    cap.set(cv2.CAP_PROP_AUTO_WB, saved_auto_wb)
    return ranges


def test_white_balance(cap: cv2.VideoCapture) -> None:
    """Attempt to disable auto white balance and set a manual value.

    Reports the result of each step so it is clear whether the camera
    accepts writes via this backend.
    """
    auto_wb_id = cv2.CAP_PROP_AUTO_WB
    wb_id = cv2.CAP_PROP_TEMPERATURE

    auto_before = cap.get(auto_wb_id)
    wb_before = cap.get(wb_id)
    print(f"  Auto WB before : {auto_before}")
    print(f"  WB value before: {wb_before}")

    print("  Setting auto WB to 0 (disabled)...")
    cap.set(auto_wb_id, 0)
    cap.grab()
    auto_after = cap.get(auto_wb_id)
    print(f"  Auto WB after  : {auto_after}")

    if auto_after != 0:
        print("  WARNING: Camera did not accept auto WB disable -- manual WB may be ignored.")

    target = 4000.0
    print(f"  Setting WB to {target}...")
    cap.set(wb_id, target)
    cap.grab()
    wb_after = cap.get(wb_id)
    print(f"  WB value after : {wb_after}")

    if wb_after == wb_before:
        print("  RESULT: WB value unchanged -- camera is not accepting writes on this backend.")
    elif wb_after == target:
        print(f"  RESULT: WB set successfully to {wb_after}.")
    else:
        print(f"  RESULT: WB changed but clamped to {wb_after} (camera accepted the write).")

    print("  Restoring original values...")
    cap.set(wb_id, wb_before)
    cap.set(auto_wb_id, auto_before)


def open_dshow_settings(index: int) -> None:
    """Open the DirectShow driver property page for the camera at the given local index."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open camera with DSHOW backend to show settings page.")
        return
    cap.set(cv2.CAP_PROP_SETTINGS, 1)
    cap.release()


def test_resolutions(cap: cv2.VideoCapture) -> list[tuple[int, int]]:
    """Test common resolutions to see which ones are supported."""
    common_resolutions = [
        (320, 240),
        (640, 480),
        (800, 600),
        (1024, 768),
        (1280, 720),
        (1280, 1024),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160),
    ]
    
    supported = []
    
    for width, height in common_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width == width and actual_height == height:
            supported.append((width, height))
    
    return supported


def get_backend_name(backend: int) -> str:
    """Convert backend ID to name."""
    backend_names = {
        cv2.CAP_ANY: "CAP_ANY",
        cv2.CAP_VFW: "CAP_VFW",
        cv2.CAP_V4L: "CAP_V4L",
        cv2.CAP_V4L2: "CAP_V4L2",
        cv2.CAP_FIREWIRE: "CAP_FIREWIRE",
        cv2.CAP_FIREWARE: "CAP_FIREWARE",
        cv2.CAP_IEEE1394: "CAP_IEEE1394",
        cv2.CAP_DC1394: "CAP_DC1394",
        cv2.CAP_CMU1394: "CAP_CMU1394",
        cv2.CAP_DSHOW: "CAP_DSHOW",
        cv2.CAP_PVAPI: "CAP_PVAPI",
        cv2.CAP_OPENNI: "CAP_OPENNI",
        cv2.CAP_OPENNI_ASUS: "CAP_OPENNI_ASUS",
        cv2.CAP_ANDROID: "CAP_ANDROID",
        cv2.CAP_XIAPI: "CAP_XIAPI",
        cv2.CAP_AVFOUNDATION: "CAP_AVFOUNDATION",
        cv2.CAP_GIGANETIX: "CAP_GIGANETIX",
        cv2.CAP_MSMF: "CAP_MSMF",
        cv2.CAP_WINRT: "CAP_WINRT",
        cv2.CAP_INTELPERC: "CAP_INTELPERC",
        cv2.CAP_OPENNI2: "CAP_OPENNI2",
        cv2.CAP_OPENNI2_ASUS: "CAP_OPENNI2_ASUS",
        cv2.CAP_GPHOTO2: "CAP_GPHOTO2",
        cv2.CAP_GSTREAMER: "CAP_GSTREAMER",
        cv2.CAP_FFMPEG: "CAP_FFMPEG",
        cv2.CAP_IMAGES: "CAP_IMAGES",
        cv2.CAP_ARAVIS: "CAP_ARAVIS",
        cv2.CAP_OPENCV_MJPEG: "CAP_OPENCV_MJPEG",
        cv2.CAP_INTEL_MFX: "CAP_INTEL_MFX",
        cv2.CAP_XINE: "CAP_XINE",
    }
    return backend_names.get(backend, f"Unknown ({backend})")


def backend_from_index(index: int) -> tuple[int, str]:
    """Derive the real backend constant from an encoded camera index.

    cv2_enumerate_cameras encodes the backend into the index by adding the
    backend base offset (CAP_DSHOW=700, CAP_MSMF=1400, etc.) and always
    reports camera_info.backend=0.  We recover the true backend from the
    index range so we can open the capture with the correct API.
    """
    known_backends: list[tuple[int, str]] = [
        (cv2.CAP_MSMF,  "CAP_MSMF"),
        (cv2.CAP_DSHOW, "CAP_DSHOW"),
        (cv2.CAP_VFW,   "CAP_VFW"),
        (cv2.CAP_V4L2,  "CAP_V4L2"),
    ]
    for base, name in known_backends:
        if base <= index < base + 100:
            return base, name
    return cv2.CAP_ANY, "CAP_ANY"


def detect_cameras(backend: int = cv2.CAP_ANY) -> list[dict]:
    """Detect available cameras using cv2_enumerate_cameras."""
    cameras = []
    
    # Suppress OpenCV error messages temporarily
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    try:
        for camera_info in enumerate_cameras(backend):
            derived_backend, derived_name = backend_from_index(camera_info.index)
            cameras.append({
                'index': camera_info.index,
                'name': camera_info.name,
                'path': camera_info.path,
                'vid': camera_info.vid,
                'pid': camera_info.pid,
                'backend': derived_backend,
                'backend_name': derived_name,
            })
    finally:
        # Restore stderr
        sys.stderr.close()
        sys.stderr = original_stderr
    
    return cameras


def main() -> None:
    """Main function to detect and display camera information."""
    parser = argparse.ArgumentParser(description='USB camera detection and control')
    parser.add_argument('--gui', action='store_true', help='Launch live feed GUI with property sliders after probing')
    args = parser.parse_args()

    print("=" * 80)
    print("OpenCV Camera Detection Script")
    print("=" * 80)
    print(f"OpenCV Version: {cv2.__version__}")
    print()

    print("Scanning for cameras...")
    all_cameras = detect_cameras()
    
    # Filter out cameras with None VID/PID (virtual cameras, VR headsets, etc.)
    cameras = [cam for cam in all_cameras if cam['vid'] is not None and cam['pid'] is not None]
    
    if not cameras:
        print("No physical cameras detected!")
        sys.exit(1)
    
    print(f"Found {len(cameras)} camera instance(s)")
    print()
    
    # Deduplicate by (vid, pid, index) — same physical device can appear
    # multiple times if enumerate_cameras returns duplicate entries.
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for cam in cameras:
        key = (cam['vid'], cam['pid'], cam['index'])
        if key not in seen:
            seen.add(key)
            deduped.append(cam)
    cameras = deduped

    # Group by (vid, pid) to count unique physical devices.
    vid_pid_groups: dict[tuple, list[dict]] = {}
    for cam in cameras:
        key = (cam['vid'], cam['pid'])
        if key not in vid_pid_groups:
            vid_pid_groups[key] = []
        vid_pid_groups[key].append(cam)

    print(f"Unique physical cameras: {len(vid_pid_groups)}")
    for (vid, pid), cam_list in vid_pid_groups.items():
        vid_str = f"0x{vid:04X}"
        pid_str = f"0x{pid:04X}"
        print(f"  VID: {vid_str}, PID: {pid_str} - {cam_list[0]['name']}")
        if len(cam_list) > 1:
            backends = ', '.join([c['backend_name'] for c in cam_list])
            print(f"    Available on {len(cam_list)} backend(s): {backends}")
    print()

    # One entry per unique physical camera, preferring DSHOW over other backends.
    unique_cameras: dict[tuple, dict] = {}
    for cam in cameras:
        key = (cam['vid'], cam['pid'])
        existing = unique_cameras.get(key)
        if existing is None or (cam['backend'] == cv2.CAP_DSHOW and existing['backend'] != cv2.CAP_DSHOW):
            unique_cameras[key] = cam

    # Compute the device-local index for each camera by subtracting the backend
    # base offset.  cv2_enumerate_cameras encodes the backend into the index
    # (e.g. DSHOW camera 0 -> index 700, MSMF camera 0 -> index 1400), but
    # VideoCapture expects a 0-based device number when a backend is specified.
    for cam in unique_cameras.values():
        cam['local_index'] = cam['index'] - cam['backend']
    
    # Get detailed information for each unique camera
    for (vid, pid), camera_info in unique_cameras.items():
        print("=" * 80)
        print(f"Camera: {camera_info['name']}")
        print("=" * 80)
        
        cap = cv2.VideoCapture(camera_info['local_index'], camera_info['backend'])
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_info['index']}")
            continue
        
        # Get camera properties
        props = get_camera_properties(cap, camera_info)
        
        print(f"Name: {props['Name']}")
        print(f"Path: {props['Path']}")
        print(f"VID: {props['VID']}")
        print(f"PID: {props['PID']}")
        print(f"Backend: {props['Backend']}")
        print(f"Resolution: {int(props['Width'])}x{int(props['Height'])}")
        print(f"FPS: {props['FPS']}")
        print(f"Codec: {props['Codec_String']} (FOURCC: {props['Codec']})")
        print(f"Brightness: {props['Brightness']}")
        print(f"Contrast: {props['Contrast']}")
        print(f"Saturation: {props['Saturation']}")
        print(f"Hue: {props['Hue']}")
        print(f"Gain: {props['Gain']}")
        print(f"Exposure: {props['Exposure']}")
        print(f"Auto Exposure: {props['Auto Exposure']}")
        print(f"Auto Focus: {props['Auto Focus']}")
        print(f"Auto WB: {props['Auto WB']}")
        print(f"White Balance (K): {props['White Balance (K)']}")
        print(f"Buffer Size: {props['Buffer Size']}")
        print()

        print("Testing white balance control...")
        test_white_balance(cap)
        print()

        print("Probing property ranges...")
        prop_ranges = get_property_ranges(cap)
        supported = {k: v for k, v in prop_ranges.items() if v is not None}
        unsupported = [k for k, v in prop_ranges.items() if v is None]

        if supported:
            name_w = max(len(k) for k in supported)
            print(f"  {'Property':<{name_w}}  {'Min':>10}  {'Max':>10}  {'Step':>6}")
            print(f"  {'-'*name_w}  {'':->10}  {'':->10}  {'':->6}")
            for pname, (lo, hi, step) in supported.items():
                print(f"  {pname:<{name_w}}  {lo:>10.1f}  {hi:>10.1f}  {step:>6.1f}")
        else:
            print("  No adjustable properties detected.")

        if unsupported:
            print(f"  Read-only/unsupported: {', '.join(unsupported)}")
        print()

        # Test supported resolutions
        print("Testing supported resolutions...")
        supported_resolutions = test_resolutions(cap)

        if supported_resolutions:
            print("Supported resolutions:")
            for width, height in supported_resolutions:
                print(f"  - {width}x{height}")
        else:
            print("No standard resolutions detected")

        cap.release()
        print()

        if args.gui:
            print(f"Launching GUI for {camera_info['name']}...")
            run_gui(camera_info, prop_ranges, supported_resolutions)
    
    print("=" * 80)
    print("Camera detection complete!")
    print("=" * 80)


def run_gui(camera_info: dict, prop_ranges: dict[str, tuple[float, float, float] | None], supported_resolutions: list[tuple[int, int]]) -> None:
    """Launch a tkinter GUI showing the live camera feed with property sliders."""
    if Image is None or ImageTk is None:
        print("GUI requires Pillow: pip install Pillow")
        return

    local_index = camera_info['local_index']
    backend     = camera_info['backend']

    cap = cv2.VideoCapture(local_index, backend)
    if not cap.isOpened():
        print(f"GUI: could not open camera {local_index}")
        return

    cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    prop_ids = {
        'Brightness':    cv2.CAP_PROP_BRIGHTNESS,
        'Contrast':      cv2.CAP_PROP_CONTRAST,
        'Saturation':    cv2.CAP_PROP_SATURATION,
        'Hue':           cv2.CAP_PROP_HUE,
        'Gain':          cv2.CAP_PROP_GAIN,
        'Exposure':      cv2.CAP_PROP_EXPOSURE,
        'Sharpness':     cv2.CAP_PROP_SHARPNESS,
        'Gamma':         cv2.CAP_PROP_GAMMA,
        'White Balance': cv2.CAP_PROP_TEMPERATURE,
        'Backlight':     cv2.CAP_PROP_BACKLIGHT,
        'Auto Exposure': cv2.CAP_PROP_AUTO_EXPOSURE,
        'Auto Focus':    cv2.CAP_PROP_AUTOFOCUS,
        'Pan':           cv2.CAP_PROP_PAN,
        'Tilt':          cv2.CAP_PROP_TILT,
        'Zoom':          cv2.CAP_PROP_ZOOM,
        'Focus':         cv2.CAP_PROP_FOCUS,
        'Iris':          cv2.CAP_PROP_IRIS,
    }

    supported = {k: v for k, v in prop_ranges.items() if v is not None}

    root = tk.Tk()
    root.title(f"Camera: {camera_info['name']}")
    root.resizable(True, True)

    # Left: live feed
    feed_frame = tk.Frame(root)
    feed_frame.pack(side=tk.LEFT, padx=8, pady=8)

    feed_label = tk.Label(feed_frame)
    feed_label.pack()

    status_var = tk.StringVar(value="Running")
    tk.Label(feed_frame, textvariable=status_var, anchor='w').pack(fill=tk.X)

    # Right: scrollable sliders
    ctrl_outer = tk.Frame(root)
    ctrl_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

    canvas = tk.Canvas(ctrl_outer, width=360)
    scrollbar = ttk.Scrollbar(ctrl_outer, orient=tk.VERTICAL, command=canvas.yview)
    ctrl_frame = tk.Frame(canvas)

    ctrl_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.create_window((0, 0), window=ctrl_frame, anchor='nw')
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Codec dropdown — must be applied before resolution/FPS to take effect
    codec_row = tk.Frame(ctrl_frame)
    codec_row.pack(fill=tk.X, pady=4, padx=4)
    tk.Label(codec_row, text='Codec', width=16, anchor='w').pack(side=tk.LEFT)

    codec_options = ['MJPG', 'YUY2', 'NV12', 'H264']
    current_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    current_codec = ''.join(chr((current_fourcc >> 8 * i) & 0xFF) for i in range(4)).strip()
    if current_codec not in codec_options:
        codec_options.insert(0, current_codec)
    codec_var = tk.StringVar(value=current_codec)

    ttk.Combobox(codec_row, textvariable=codec_var, values=codec_options, width=12, state='readonly').pack(side=tk.LEFT, padx=4)

    # Resolution dropdown
    res_row = tk.Frame(ctrl_frame)
    res_row.pack(fill=tk.X, pady=4, padx=4)
    tk.Label(res_row, text='Resolution', width=16, anchor='w').pack(side=tk.LEFT)

    res_options = [f"{w}x{h}" for w, h in supported_resolutions]
    current_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    current_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    current_res = f"{current_w}x{current_h}"
    if current_res not in res_options:
        res_options.insert(0, current_res)

    res_var = tk.StringVar(value=current_res)

    # Mutable reference so closures can rebind the capture after reopen.
    cap_ref: list[cv2.VideoCapture] = [cap]

    # Pause flag lets apply_format safely release/reopen without racing the capture thread.
    paused = threading.Event()

    def apply_format(*_: object) -> None:
        """Release and reopen the capture with the selected codec/resolution.

        DSHOW ignores CAP_PROP_FOURCC on reopen for many drivers. MSMF (the
        Windows Media Foundation backend) honours the FOURCC set before the
        first read, so we reopen with CAP_MSMF for format changes. The local
        index is the same since both backends use 0-based device numbering.
        """
        codec = codec_var.get()
        val = res_var.get()
        try:
            w, h = (int(x) for x in val.split('x'))
        except ValueError:
            return
        fourcc = cv2.VideoWriter.fourcc(*codec.ljust(4)[:4])

        paused.set()
        time.sleep(0.1)  # Let the capture thread finish any in-flight read.
        cap_ref[0].release()
        new_cap = cv2.VideoCapture(local_index, cv2.CAP_MSMF)
        new_cap.set(cv2.CAP_PROP_FOURCC,       fourcc)
        new_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        new_cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap_ref[0] = new_cap
        with frame_queue.mutex:
            frame_queue.queue.clear()
        paused.clear()

        actual_w = int(new_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(new_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = new_cap.get(cv2.CAP_PROP_FPS)

        res_var.trace_remove('write', res_trace_id[0])
        res_var.set(f"{actual_w}x{actual_h}")
        res_trace_id[0] = res_var.trace_add('write', apply_format)

        codec_var.trace_remove('write', codec_trace_id[0])
        codec_trace_id[0] = codec_var.trace_add('write', apply_format)
        requested_codec_var[0] = codec
        status_var.set(f"{actual_w}x{actual_h}  {actual_fps:.4g} fps  [{codec}]")
        capture_times.clear()
        display_times.clear()
        last_frame_hash[0] = b''

    res_trace_id: list[str] = [res_var.trace_add('write', apply_format)]
    codec_trace_id: list[str] = [codec_var.trace_add('write', apply_format)]

    ttk.Combobox(res_row, textvariable=res_var, values=res_options, width=12, state='readonly').pack(side=tk.LEFT, padx=4)

    ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6, padx=4)

    slider_vars: dict[str, tk.DoubleVar] = {}

    def on_slider(name: str, prop_id: int, var: tk.DoubleVar, label_var: tk.StringVar) -> None:
        val = round(var.get())
        cap_ref[0].set(prop_id, val)
        actual = cap_ref[0].get(prop_id)
        label_var.set(f"{actual:.0f}")

    for name, (lo, hi, step) in supported.items():
        prop_id = prop_ids.get(name)
        if prop_id is None:
            continue

        current = cap_ref[0].get(prop_id)

        row = tk.Frame(ctrl_frame)
        row.pack(fill=tk.X, pady=3, padx=4)

        tk.Label(row, text=name, width=16, anchor='w').pack(side=tk.LEFT)

        label_var = tk.StringVar(value=f"{current:.0f}")
        tk.Label(row, textvariable=label_var, width=6, anchor='e').pack(side=tk.RIGHT)

        var = tk.DoubleVar(value=current)
        slider_vars[name] = var

        tk.Scale(
            row,
            variable=var,
            from_=lo,
            to=hi,
            resolution=step,
            orient=tk.HORIZONTAL,
            length=180,
            showvalue=False,
            command=lambda _, n=name, pid=prop_id, v=var, lv=label_var: on_slider(n, pid, v, lv),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(row, text=f"{lo:.0f}", fg='gray', width=5, anchor='e').pack(side=tk.LEFT)
        tk.Label(row, text=f"{hi:.0f}", fg='gray', width=5, anchor='w').pack(side=tk.RIGHT)

    import queue
    import time

    running = threading.Event()
    running.set()
    capture_times: list[float] = []
    display_times: list[float] = []
    frame_queue: queue.Queue = queue.Queue(maxsize=2)
    last_frame_hash: list[bytes] = [b'']
    requested_codec_var: list[str] = [current_codec]

    def capture_thread() -> None:
        while running.is_set():
            if paused.is_set():
                time.sleep(0.01)
                continue
            ret, frame = cap_ref[0].read()
            if not ret:
                continue
            frame_hash = frame[::32, ::32].tobytes()
            if frame_hash == last_frame_hash[0]:
                continue
            last_frame_hash[0] = frame_hash
            now = time.monotonic()
            capture_times.append(now)
            while len(capture_times) > 1 and capture_times[-1] - capture_times[0] > 2.0:
                capture_times.pop(0)
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)

    def update_feed() -> None:
        if not running.is_set():
            return
        try:
            frame = frame_queue.get_nowait()
        except queue.Empty:
            root.after(8, update_feed)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        max_w, max_h = 640, 480
        scale = min(max_w / w, max_h / h, 1.0)
        disp = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
        img = ImageTk.PhotoImage(Image.fromarray(disp))
        feed_label.configure(image=img)
        feed_label.image = img

        now = time.monotonic()
        display_times.append(now)
        while len(display_times) > 1 and display_times[-1] - display_times[0] > 2.0:
            display_times.pop(0)

        cap_fps = (len(capture_times) - 1) / (capture_times[-1] - capture_times[0]) if len(capture_times) > 1 and capture_times[-1] > capture_times[0] else 0.0
        disp_fps = (len(display_times) - 1) / (display_times[-1] - display_times[0]) if len(display_times) > 1 and display_times[-1] > display_times[0] else 0.0
        codec_label = requested_codec_var[0] if requested_codec_var[0] else '...'
        status_var.set(f"{w}x{h}  capture: {cap_fps:.1f} fps  display: {disp_fps:.1f} fps  [{codec_label}]")

        root.after(8, update_feed)

    def on_close() -> None:
        running.clear()
        root.after(100, lambda: (cap_ref[0].release(), root.destroy()))

    threading.Thread(target=capture_thread, daemon=True).start()
    root.protocol('WM_DELETE_WINDOW', on_close)
    root.after(8, update_feed)
    root.mainloop()

if __name__ == "__main__":
    main()