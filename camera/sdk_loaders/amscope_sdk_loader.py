"""
Utility for loading the Amscope SDK dynamically.

This module handles:
- Extracting the SDK from zip if needed
- Platform-specific DLL/SO path configuration
- Dynamic module import with correct __file__ override
"""

import os
import sys
import platform
import zipfile
import shutil
import importlib.util
from pathlib import Path
from typing import Optional

from logger import get_logger

class AmscopeSdkLoader:
    """
    Loader for the Amscope camera SDK.
    
    Handles automatic extraction from zip, platform detection,
    and dynamic module loading.
    """
    
    def __init__(self, sdk_base_dir: Optional[Path] = None):
        """
        Initialize the SDK loader.
        
        Args:
            sdk_base_dir: Optional base directory for SDK files.
                         If None, uses project_root/3rd_party_imports
        """
        if sdk_base_dir is None:
            # Auto-detect project root (2 levels up from this file)
            project_root = Path(__file__).resolve().parent.parent.parent
            sdk_base_dir = project_root / "3rd_party_imports"
        
        self.sdk_base_dir = Path(sdk_base_dir)
        self.official_dir = self.sdk_base_dir / "official_amscope"
        self.amcam_module = None
        
    def load(self):
        """
        Load the Amscope SDK.
        
        Returns:
            The loaded amcam module
            
        Raises:
            RuntimeError: If SDK cannot be found or loaded
        """
        # Ensure SDK is extracted
        sdk_root, sdk_py = self._ensure_sdk()
        
        # Get platform-specific DLL directory
        dll_dir = self._get_dll_directory(sdk_root)
        
        # Configure library search path
        self._configure_library_path(dll_dir)
        
        # Load the module
        self.amcam_module = self._load_module(sdk_py, dll_dir)
        
        return self.amcam_module
    
    def _ensure_sdk(self) -> tuple[Path, Path]:
        """
        Ensure the AmScope SDK is available under:
            sdk_base_dir / "official_amscope"
        If not, extract the first amcamsdk*.zip in sdk_base_dir.
        
        Returns:
            Tuple of (sdk_root_dir, sdk_py_path)
            
        Raises:
            RuntimeError: If SDK cannot be found or extracted
        """
        sdk_py = self.official_dir / "python" / "amcam.py"
        
        # Already extracted?
        if sdk_py.is_file():
            return self.official_dir, sdk_py
        
        # Ensure base directory exists
        self.sdk_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Look for a zip starting with "amcamsdk"
        for f in self.sdk_base_dir.iterdir():
            if (f.is_file() and 
                f.name.lower().startswith("amcamsdk") and 
                f.suffix.lower() == ".zip"):
                
                get_logger().info(f"Extracting AmScope SDK from {f.name}...")
                with zipfile.ZipFile(f, "r") as zf:
                    zf.extractall(self.official_dir)
                break
        else:
            raise RuntimeError(
                f"No AmScope SDK zip found in {self.sdk_base_dir}\n"
                f"Expected a file named amcamsdk*.zip"
            )
        
        # Handle case where zip contains a single subdirectory
        if not sdk_py.is_file():
            subdirs = [d for d in self.official_dir.iterdir() if d.is_dir()]
            if len(subdirs) == 1:
                nested_sdk_py = subdirs[0] / "python" / "amcam.py"
                if nested_sdk_py.is_file():
                    # Move contents up one level
                    tmp = subdirs[0]
                    for item in tmp.iterdir():
                        shutil.move(str(item), self.official_dir)
                    tmp.rmdir()
        
        # Verify extraction succeeded
        if not sdk_py.is_file():
            raise RuntimeError(
                f"Extracted SDK does not contain python/amcam.py\n"
                f"Expected at: {sdk_py}"
            )
        
        get_logger().info(f"AmScope SDK ready at {self.official_dir}")
        return self.official_dir, sdk_py
    
    def _get_dll_directory(self, sdk_root: Path) -> Path:
        """
        Determine platform-specific DLL/SO directory.
        
        Args:
            sdk_root: Root directory of the SDK
            
        Returns:
            Path to the directory containing platform libraries
            
        Raises:
            RuntimeError: If platform is not supported
        """
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == 'windows':
            dll_dir = sdk_root / 'win' / 'x64'
            
        elif system == 'linux':
            arch_map = {
                'x86_64': 'x64',
                'amd64': 'x64',
                'i386': 'x86',
                'i686': 'x86',
                'arm64': 'arm64',
                'aarch64': 'arm64',
                'armv7l': 'armhf',
                'armv6l': 'armel'
            }
            subarch = arch_map.get(machine)
            if not subarch:
                raise RuntimeError(
                    f"Unsupported Linux architecture: {machine}\n"
                    f"Supported: {', '.join(arch_map.keys())}"
                )
            dll_dir = sdk_root / 'linux' / subarch
            
        elif system == 'darwin':
            dll_dir = sdk_root / 'mac'
            
        else:
            raise RuntimeError(f"Unsupported operating system: {system}")
        
        if not dll_dir.exists():
            raise RuntimeError(
                f"Platform library directory not found: {dll_dir}\n"
                f"System: {system}, Architecture: {machine}"
            )
        
        return dll_dir
    
    def _configure_library_path(self, dll_dir: Path):
        """
        Configure library search paths for the current platform.
        
        Args:
            dll_dir: Directory containing platform libraries
        """
        system = platform.system().lower()
        dll_dir_str = str(dll_dir)
        
        if system == 'windows':
            # Windows: Use add_dll_directory if available (Python 3.8+)
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(dll_dir_str)
            else:
                # Fallback for older Python versions
                os.environ['PATH'] = dll_dir_str + os.pathsep + os.environ.get('PATH', '')
                
        else:
            # Linux/macOS: Set LD_LIBRARY_PATH or DYLD_LIBRARY_PATH
            if system == 'darwin':
                env_var = 'DYLD_LIBRARY_PATH'
            else:
                env_var = 'LD_LIBRARY_PATH'
            
            current_path = os.environ.get(env_var, '')
            os.environ[env_var] = dll_dir_str + os.pathsep + current_path
    
    def _load_module(self, sdk_py: Path, dll_dir: Path):
        """
        Dynamically load the amcam module.
        
        Args:
            sdk_py: Path to amcam.py
            dll_dir: Directory containing platform libraries
            
        Returns:
            The loaded amcam module
        """
        # Create module spec
        spec = importlib.util.spec_from_file_location("amcam", sdk_py)
        amcam_module = importlib.util.module_from_spec(spec)
        
        # Override __file__ to trick the SDK's LoadLibrary logic
        # The SDK uses __file__ to find the DLL, so we point it to the DLL directory
        amcam_module.__file__ = str(dll_dir / 'amcam.py')
        
        # Register in sys.modules before execution
        sys.modules["amcam"] = amcam_module
        
        # Execute the module
        spec.loader.exec_module(amcam_module)
        
        return amcam_module


def load_amscope_sdk(sdk_base_dir: Optional[Path] = None):
    """
    Convenience function to load the Amscope SDK.
    
    Args:
        sdk_base_dir: Optional base directory for SDK files.
                     If None, auto-detects from project structure.
    
    Returns:
        The loaded amcam module
        
    Example:
        >>> amcam = load_amscope_sdk()
        >>> cameras = amcam.Amcam.EnumV2()
    """
    loader = AmscopeSdkLoader(sdk_base_dir)
    return loader.load()


if __name__ == "__main__":
    # Test the loader
    try:
        amcam = load_amscope_sdk()
        print(f"Successfully loaded amcam SDK")
        print(f"Module location: {amcam.__file__}")
        
        # Try to enumerate cameras
        cameras = amcam.Amcam.EnumV2()
        print(f"Found {len(cameras)} camera(s)")
        for i, cam in enumerate(cameras):
            print(f"  {i+1}. {cam.displayname}")
            
    except Exception as e:
        print(f"Error loading SDK: {e}")
        sys.exit(1)
