"""
Utility for loading the Amscope SDK dynamically.

This module handles:
- Extracting the SDK from zip if needed (including SDK_{date}.zip wrappers)
- Platform-specific DLL/SO path configuration
- Dynamic module import with correct __file__ override
"""

from __future__ import annotations

import os
import sys
import platform
import zipfile
import shutil
import importlib.util
from pathlib import Path

from common.logger import get_logger


def _find_amcamsdk_zip(directory: Path) -> Path | None:
    for f in directory.iterdir():
        if f.is_file() and f.name.lower().startswith("amcamsdk") and f.suffix.lower() == ".zip":
            return f
    return None


def _find_sdk_wrapper_zip(directory: Path) -> Path | None:
    for f in directory.iterdir():
        if f.is_file() and f.name.lower().startswith("sdk_") and f.suffix.lower() == ".zip":
            return f
    return None


def _flatten_single_subdir(directory: Path) -> None:
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        tmp = subdirs[0]
        for item in tmp.iterdir():
            shutil.move(str(item), directory)
        tmp.rmdir()


def _extract_amcamsdk_zip(zip_path: Path, dest: Path) -> bool:
    get_logger().info(f"Extracting AmScope SDK from {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    _flatten_single_subdir(dest)
    return True


def _ensure_sdk(sdk_base_dir: Path, official_dir: Path) -> tuple[Path, Path] | None:
    sdk_py = official_dir / "python" / "amcam.py"

    if sdk_py.is_file():
        return official_dir, sdk_py

    sdk_base_dir.mkdir(parents=True, exist_ok=True)

    amcamsdk_zip = _find_amcamsdk_zip(sdk_base_dir)

    if amcamsdk_zip is None:
        wrapper_zip = _find_sdk_wrapper_zip(sdk_base_dir)
        if wrapper_zip is None:
            get_logger().error(
                f"No AmScope SDK zip found in {sdk_base_dir}. "
                f"Expected amcamsdk*.zip or SDK_{{date}}.zip containing amcamsdk*.zip."
            )
            return None

        get_logger().info(f"Found SDK wrapper zip {wrapper_zip.name}, searching for amcamsdk zip inside...")
        with zipfile.ZipFile(wrapper_zip, "r") as zf:
            inner_names = [n for n in zf.namelist() if Path(n).name.lower().startswith("amcamsdk") and n.lower().endswith(".zip")]
            if not inner_names:
                get_logger().error(f"No amcamsdk*.zip found inside {wrapper_zip.name}")
                return None
            inner_name = inner_names[0]
            extracted_inner = sdk_base_dir / Path(inner_name).name
            with zf.open(inner_name) as src, open(extracted_inner, "wb") as dst:
                shutil.copyfileobj(src, dst)
        amcamsdk_zip = extracted_inner

    _extract_amcamsdk_zip(amcamsdk_zip, official_dir)

    if not sdk_py.is_file():
        get_logger().error(f"Extracted SDK does not contain python/amcam.py. Expected at: {sdk_py}")
        return None

    get_logger().info(f"AmScope SDK ready at {official_dir}")
    return official_dir, sdk_py


def _get_dll_directory(sdk_root: Path) -> Path | None:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        dll_dir = sdk_root / "win" / "x64"

    elif system == "linux":
        arch_map = {
            "x86_64": "x64",
            "amd64": "x64",
            "i386": "x86",
            "i686": "x86",
            "arm64": "arm64",
            "aarch64": "arm64",
            "armv7l": "armhf",
            "armv6l": "armel",
        }
        subarch = arch_map.get(machine)
        if not subarch:
            get_logger().error(f"Unsupported Linux architecture: {machine}. Supported: {', '.join(arch_map.keys())}")
            return None
        dll_dir = sdk_root / "linux" / subarch

    elif system == "darwin":
        dll_dir = sdk_root / "mac"

    else:
        get_logger().error(f"Unsupported operating system: {system}")
        return None

    if not dll_dir.exists():
        get_logger().error(f"Platform library directory not found: {dll_dir}. System: {system}, Architecture: {machine}")
        return None

    return dll_dir


def _configure_library_path(dll_dir: Path) -> None:
    system = platform.system().lower()
    dll_dir_str = str(dll_dir)

    if system == "windows":
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dll_dir_str)
        else:
            os.environ["PATH"] = dll_dir_str + os.pathsep + os.environ.get("PATH", "")
    else:
        env_var = "DYLD_LIBRARY_PATH" if system == "darwin" else "LD_LIBRARY_PATH"
        current_path = os.environ.get(env_var, "")
        os.environ[env_var] = dll_dir_str + os.pathsep + current_path


def _load_module(sdk_py: Path, dll_dir: Path):
    spec = importlib.util.spec_from_file_location("amcam", sdk_py)
    amcam_module = importlib.util.module_from_spec(spec)
    amcam_module.__file__ = str(dll_dir / "amcam.py")
    sys.modules["amcam"] = amcam_module
    spec.loader.exec_module(amcam_module)
    return amcam_module


def load_amscope_sdk(sdk_base_dir: Path | None = None):
    """
    Load the Amscope SDK.

    Args:
        sdk_base_dir: Optional base directory for SDK files.
                      If None, auto-detects from project structure.

    Returns:
        The loaded amcam module, or None if loading failed.

    Example:
        >>> amcam = load_amscope_sdk()
        >>> cameras = amcam.Amcam.EnumV2()
    """
    if sdk_base_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        sdk_base_dir = project_root / "3rd_party_imports"

    sdk_base_dir = Path(sdk_base_dir)
    official_dir = sdk_base_dir / "official_amscope"

    result = _ensure_sdk(sdk_base_dir, official_dir)
    if result is None:
        return None

    sdk_root, sdk_py = result

    dll_dir = _get_dll_directory(sdk_root)
    if dll_dir is None:
        return None

    _configure_library_path(dll_dir)
    return _load_module(sdk_py, dll_dir)


if __name__ == "__main__":
    amcam = load_amscope_sdk()
    if amcam is None:
        print("Failed to load amcam SDK")
        sys.exit(1)

    print(f"Successfully loaded amcam SDK")
    print(f"Module location: {amcam.__file__}")

    cameras = amcam.Amcam.EnumV2()
    print(f"Found {len(cameras)} camera(s)")
    for i, cam in enumerate(cameras):
        print(f"  {i+1}. {cam.displayname}")