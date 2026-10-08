"""
Background update checker/installer for FieldWeave.

The Updater checks GitHub Releases for a newer version than the one
currently running, and - if the user accepts - checks out that release's
tag and reinstalls dependencies. All git/network work runs on a background
thread and only ever mutates plain attributes on itself (guarded by a
lock). It never touches widgets or emits Qt signals. A polling QTimer on
the main thread (UI/widgets/update_notifier.py) reads this state and
drives all dialogs and notifications.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError

from common.fieldweaveConfig import FIELDWEAVE_VERSION
from common.logger import info, warning, error

_ERROR_MESSAGE_LIMIT = 500
_GITHUB_OWNER = "AnthonyvW"
_GITHUB_REPO = "FieldWeave"
_RELEASES_URL = f"https://api.github.com/repos/{_GITHUB_OWNER}/{_GITHUB_REPO}/releases?per_page=100"
_GITHUB_REQUEST_TIMEOUT = 10


@dataclass(frozen=True)
class ReleaseInfo:
    tag: str
    version: str
    title: str
    notes: str
    prerelease: bool


class UpdateStatus:
    IDLE = "idle"
    CHECKING = "checking"
    UPDATE_AVAILABLE = "update_available"
    UP_TO_DATE = "up_to_date"
    CHECK_FAILED = "check_failed"
    UPDATING = "updating"
    UPDATE_COMPLETE = "update_complete"
    UPDATE_FAILED = "update_failed"


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a dotted version string ("1.2.0", "v1.2.0-beta") into a comparable tuple of ints."""
    parts = []
    for chunk in version.strip().lstrip("vV").split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


class Updater:
    def __init__(self, repo_dir: Path | None = None, include_prereleases: bool = False) -> None:
        self._repo_dir = repo_dir or Path.cwd()
        self._include_prereleases = include_prereleases
        self._lock = threading.Lock()

        self._status: str = UpdateStatus.IDLE
        self._latest_version: str = ""
        self._release_tag: str = ""
        self._release_title: str = ""
        self._release_notes: str = ""
        self._error_message: str = ""
        self._releases: list[ReleaseInfo] = []
        self._target_tag: str = ""

        self._check_thread: threading.Thread | None = None
        self._update_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # State snapshot — safe to read from the main thread at any time
    # ------------------------------------------------------------------

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def latest_version(self) -> str:
        with self._lock:
            return self._latest_version

    @property
    def release_title(self) -> str:
        with self._lock:
            return self._release_title

    @property
    def release_notes(self) -> str:
        with self._lock:
            return self._release_notes

    @property
    def error_message(self) -> str:
        with self._lock:
            return self._error_message

    @property
    def available_releases(self) -> list[ReleaseInfo]:
        """All published releases from the last successful check, newest first."""
        with self._lock:
            return list(self._releases)

    @property
    def target_version(self) -> str:
        """Version of the release most recently requested via start_update()."""
        with self._lock:
            return self._target_tag.lstrip("vV")

    @property
    def include_prereleases(self) -> bool:
        with self._lock:
            return self._include_prereleases

    def set_include_prereleases(self, include: bool) -> None:
        with self._lock:
            self._include_prereleases = include

    def can_install(self) -> bool:
        return self._is_git_checkout()

    def is_busy(self) -> bool:
        with self._lock:
            return self._status in (UpdateStatus.CHECKING, UpdateStatus.UPDATING)

    # ------------------------------------------------------------------
    # Public control — called from the main thread
    # ------------------------------------------------------------------

    def start_check(self) -> bool:
        """Kick off a background release check. Returns False if a check/update is already running."""
        if self.is_busy():
            return False

        self._set_status(UpdateStatus.CHECKING)
        self._check_thread = threading.Thread(target=self._run_check, daemon=True)
        self._check_thread.start()
        return True

    def start_update(self, tag: str | None = None) -> bool:
        """
        Kick off a background checkout of a release, then a dependency install.
        Defaults to the latest release from the last check; pass a tag to
        install a specific release. Returns False if already busy.
        """
        if self.is_busy():
            return False

        if not self._is_git_checkout():
            self._fail_update("Not running from a git checkout - update unavailable")
            return False

        if tag is None:
            with self._lock:
                tag = self._release_tag
        if not tag:
            self._fail_update("No release found to update to - run a check first")
            return False

        with self._lock:
            self._target_tag = tag
        self._set_status(UpdateStatus.UPDATING)
        self._update_thread = threading.Thread(target=self._run_update, args=(tag,), daemon=True)
        self._update_thread.start()
        return True

    def reset(self) -> None:
        """Return to IDLE so a fresh check can be started (e.g. after a dismissed prompt)."""
        self._set_status(UpdateStatus.IDLE)

    # ------------------------------------------------------------------
    # Background thread work
    # ------------------------------------------------------------------

    def _run_check(self) -> None:
        try:
            raw_releases = self._fetch_releases()
        except (URLError, HTTPError, ValueError, OSError) as exc:
            self._fail_check(f"Could not reach GitHub - {exc}")
            return

        releases = sorted(
            (
                ReleaseInfo(
                    tag=raw["tag_name"],
                    version=raw["tag_name"].lstrip("vV"),
                    title=raw.get("name") or raw["tag_name"],
                    notes=raw.get("body") or "",
                    prerelease=bool(raw.get("prerelease")),
                )
                for raw in raw_releases
                if raw.get("tag_name") and not raw.get("draft")
            ),
            key=lambda r: (_parse_version(r.version), not r.prerelease),
            reverse=True,
        )

        include_prereleases = self.include_prereleases
        latest = next((r for r in releases if include_prereleases or not r.prerelease), None)
        if latest is None:
            self._fail_check("No published releases found")
            return

        tag = latest.tag
        latest_version = latest.version

        with self._lock:
            self._releases = releases
            self._latest_version = latest_version
            self._release_tag = tag
            self._release_title = latest.title
            self._release_notes = latest.notes
            self._status = (
                UpdateStatus.UPDATE_AVAILABLE
                if _parse_version(latest_version) > _parse_version(FIELDWEAVE_VERSION)
                else UpdateStatus.UP_TO_DATE
            )

        info(f"Update check complete: latest release is {latest_version}, running {FIELDWEAVE_VERSION}")

    def _run_update(self, tag: str) -> None:
        success, detail = self._git_checkout_release(tag)
        if not success:
            self._fail_update(f"Could not switch to release {tag} - {detail}")
            return

        success, detail = self._install_requirements()
        if not success:
            self._fail_update(f"Failed to install dependencies - {detail}")
            return

        with self._lock:
            self._status = UpdateStatus.UPDATE_COMPLETE

        info(f"Updated to release {tag} - restart required")

    # ------------------------------------------------------------------
    # GitHub API
    # ------------------------------------------------------------------

    def _fetch_releases(self) -> list[dict]:
        request = urllib.request.Request(
            _RELEASES_URL,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": f"FieldWeave-Updater/{FIELDWEAVE_VERSION}",
            },
        )
        with urllib.request.urlopen(request, timeout=_GITHUB_REQUEST_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))

    # ------------------------------------------------------------------
    # Git / pip helpers. These shell out to external tools, so failures
    # (missing binary, no network, timeout) are expected and handled by
    # returning a (success, detail) pair rather than propagating.
    # ------------------------------------------------------------------

    def _is_git_checkout(self) -> bool:
        return (self._repo_dir / ".git").exists()

    def _run_git(self, args: list[str], timeout: int, log_errors: bool = True) -> tuple[bool, str, str]:
        """Returns (success, stdout, stderr). On launch failure, stderr holds the reason."""
        command = " ".join(["git", *args])

        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self._repo_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            return False, "", "git is not installed or not on PATH"
        except subprocess.TimeoutExpired:
            return False, "", f"'{command}' timed out after {timeout}s"
        except OSError as e:
            return False, "", f"Failed to run '{command}': {e}"

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0 and log_errors:
            error(f"'{command}' exited {result.returncode}\nstdout: {stdout}\nstderr: {stderr}")

        return result.returncode == 0, stdout, stderr

    def _git_checkout_release(self, tag: str) -> tuple[bool, str]:
        success, _, stderr = self._run_git(["fetch", "--tags"], timeout=30)
        if not success:
            return False, self._summarize(stderr)

        success, stdout, stderr = self._run_git(["checkout", tag], timeout=30)
        if not success:
            return False, self._summarize(stderr or stdout)

        return True, ""

    def _install_requirements(self) -> tuple[bool, str]:
        requirements = self._repo_dir / "requirements.txt"
        if not requirements.exists():
            warning("requirements.txt not found - skipping dependency install")
            return True, ""

        pip_exe = Path(sys.executable).with_name("pip.exe" if sys.platform == "win32" else "pip")
        if not pip_exe.exists():
            return False, f"pip executable not found at {pip_exe}"

        try:
            result = subprocess.run(
                [str(pip_exe), "install", "-r", str(requirements)],
                cwd=self._repo_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return False, "pip install timed out after 300s"
        except OSError as e:
            return False, f"Failed to run pip: {e}"

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "pip install failed"
            error(f"pip install exited {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}")
            return False, self._summarize(detail)

        return True, ""

    def _summarize(self, detail: str) -> str:
        """Trim long git/pip output down to something short enough for a popup; full text still goes to the log."""
        detail = detail.strip()
        if len(detail) <= _ERROR_MESSAGE_LIMIT:
            return detail
        return detail[:_ERROR_MESSAGE_LIMIT].rstrip() + "... (see log for full output)"

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _fail_check(self, message: str) -> None:
        warning(f"Update check failed: {message}")
        with self._lock:
            self._status = UpdateStatus.CHECK_FAILED
            self._error_message = message

    def _fail_update(self, message: str) -> None:
        error(f"Update failed: {message}")
        with self._lock:
            self._status = UpdateStatus.UPDATE_FAILED
            self._error_message = message

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status
            self._error_message = ""


def relaunch() -> None:
    """
    Replace the current process image with a fresh launch of the app.

    Works both for `python main.py` (sys.argv already starts with the script
    path, sys.executable is the interpreter) and a frozen executable
    (sys.executable *is* the app, so sys.argv[0] must be dropped to avoid
    passing it twice) - so this doesn't need to change when the venv/git
    workflow is replaced by GitHub-release executables.
    """
    if getattr(sys, "frozen", False):
        args = [sys.executable, *sys.argv[1:]]
    else:
        args = [sys.executable, *sys.argv]

    os.execv(sys.executable, args)
