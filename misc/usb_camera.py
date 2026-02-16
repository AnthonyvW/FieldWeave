#!/usr/bin/env python3
"""
Camera Detection Script
Detects all available cameras using OpenCV and displays their capabilities.
"""

from __future__ import annotations

import cv2
import sys
import os

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
    }
    
    # Convert FOURCC code to readable format
    fourcc = properties['Codec']
    if fourcc > 0:
        properties['Codec_String'] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    else:
        properties['Codec_String'] = 'Unknown'
    
    return properties


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


def detect_cameras(backend: int = cv2.CAP_ANY) -> list[dict]:
    """Detect available cameras using cv2_enumerate_cameras."""
    cameras = []
    
    # Suppress OpenCV error messages temporarily
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    try:
        for camera_info in enumerate_cameras(backend):
            cameras.append({
                'index': camera_info.index,
                'name': camera_info.name,
                'path': camera_info.path,
                'vid': camera_info.vid,
                'pid': camera_info.pid,
                'backend': camera_info.backend,
                'backend_name': get_backend_name(camera_info.backend),
            })
    finally:
        # Restore stderr
        sys.stderr.close()
        sys.stderr = original_stderr
    
    return cameras


def main() -> None:
    """Main function to detect and display camera information."""
    print("=" * 80)
    print("OpenCV Camera Detection Script")
    print("=" * 80)
    print(f"OpenCV Version: {cv2.__version__}")
    print()
    
    # Detect available cameras
    print("Scanning for cameras...")
    all_cameras = detect_cameras()
    
    # Filter out cameras with None VID/PID (virtual cameras, VR headsets, etc.)
    cameras = [cam for cam in all_cameras if cam['vid'] is not None and cam['pid'] is not None]
    
    if not cameras:
        print("No physical cameras detected!")
        sys.exit(1)
    
    print(f"Found {len(cameras)} camera instance(s)")
    print()
    
    # Group cameras by VID/PID to identify duplicates across backends
    vid_pid_groups = {}
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
    
    # Process only one instance per unique VID/PID combination
    # Prefer the first instance found for each unique camera
    unique_cameras = {}
    for cam in cameras:
        key = (cam['vid'], cam['pid'])
        if key not in unique_cameras:
            unique_cameras[key] = cam
    
    # Get detailed information for each unique camera
    for (vid, pid), camera_info in unique_cameras.items():
        print("=" * 80)
        print(f"Camera: {camera_info['name']}")
        print("=" * 80)
        
        cap = cv2.VideoCapture(camera_info['index'], camera_info['backend'])
        
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
    
    print("=" * 80)
    print("Camera detection complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()