"""Loading of the FieldWeave changelog.

Each release's notes live in their own markdown file in this folder,
named after the version they describe (e.g. ``1.2.md``), so they can be
pasted directly from a GitHub release.
"""

from __future__ import annotations

from pathlib import Path

from common.logger import warning

CHANGELOG_DIR = Path(__file__).resolve().parent


def _version_key(path: Path) -> tuple[int, ...]:
    return tuple(int(chunk) if chunk.isdigit() else 0 for chunk in path.stem.split("."))


def load_changelog() -> str | None:
    """Read every version's markdown file and combine them, newest first."""
    if not CHANGELOG_DIR.is_dir():
        return None

    files = sorted(CHANGELOG_DIR.glob("*.md"), key=_version_key, reverse=True)
    if not files:
        return None

    sections: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            warning(f"Failed to read changelog file {path}: {exc}")
            continue
        sections.append(f"# FieldWeave v{path.stem}\n\n{text}")

    if not sections:
        return None

    return "\n\n---\n\n".join(sections)