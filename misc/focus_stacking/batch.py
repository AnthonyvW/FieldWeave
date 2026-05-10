from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FOCUS_STACKER_PATH = r"focusweave"


def run_focus_stacking(input_folder: Path, output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    subfolders = [
        entry for entry in input_folder.iterdir()
        if entry.is_dir() and entry.name != "output"
    ]

    if not subfolders:
        print(f"No subfolders found in {input_folder}")
        return

    for subfolder in subfolders:
        output_path = output_folder / f"{subfolder.name}.jpg"
        cmd = [
            FOCUS_STACKER_PATH,
            str(subfolder),
            "--output", str(output_path),
            "--cull",
            "--keep-size",
            "--workers", "0",
        ]

        print(f"Processing: {subfolder.name}")
        subprocess.run(cmd, check=True)

    print("Done.")


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])

    if not input_folder.is_dir():
        print(f"Error: input folder '{input_folder}' does not exist or is not a directory")
        sys.exit(1)

    run_focus_stacking(input_folder, output_folder)


if __name__ == "__main__":
    main()