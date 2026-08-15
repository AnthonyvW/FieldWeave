#!/usr/bin/env python3
"""
Batch focus stacking script.
Processes all subfolders in a given directory, running focus-stack on the JPEG images in each.
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("Usage: python focus_stack_batch.py <input_folder>")
        sys.exit(1)
    
    input_folder = Path(sys.argv[1])
    
    if not input_folder.exists():
        print(f"Error: Folder '{input_folder}' does not exist")
        sys.exit(1)
    
    if not input_folder.is_dir():
        print(f"Error: '{input_folder}' is not a directory")
        sys.exit(1)
    
    # Get the focus-stack executable path relative to this script
    script_dir = Path(__file__).parent
    focus_stack_exe = script_dir / "../../focus-stack/focus-stack.exe"
    focus_stack_exe = focus_stack_exe.resolve()
    
    if not focus_stack_exe.exists():
        print(f"Error: focus-stack executable not found at '{focus_stack_exe}'")
        sys.exit(1)
    
    # Process each subfolder
    subfolders = [d for d in input_folder.iterdir() if d.is_dir()]
    
    if not subfolders:
        print(f"No subfolders found in '{input_folder}'")
        sys.exit(0)
    
    print(f"Found {len(subfolders)} subfolder(s) to process")
    
    for subfolder in sorted(subfolders):
        # Check if there are any JPEG files in this subfolder
        jpeg_files = sorted(subfolder.glob("*.jpeg"))
        
        if not jpeg_files:
            print(f"Skipping '{subfolder.name}': no JPEG files found")
            continue
        
        # Construct output path (use absolute path)
        output_file = (input_folder / f"{subfolder.name}.jpeg").resolve()
        
        # Build the command with expanded file list (use absolute paths)
        cmd = [str(focus_stack_exe)]
        cmd.extend([str(f.resolve()) for f in jpeg_files])
        cmd.append(f"--output={str(output_file)}")
        
        print(f"\nProcessing '{subfolder.name}' ({len(jpeg_files)} images)...")
        print(f"Output: {output_file}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Success: {subfolder.name}")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error processing '{subfolder.name}':")
            print(f"Exit code: {e.returncode}")
            if e.stderr:
                print(f"Error output: {e.stderr}")
    
    print("\nBatch processing complete!")


if __name__ == "__main__":
    main()