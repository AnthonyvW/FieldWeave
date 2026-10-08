from __future__ import annotations

from pathlib import Path


def unique_path(path: Path) -> Path:
    """Return *path* if nothing exists there, else the first free ``<stem>_<n><suffix>`` beside it."""
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
