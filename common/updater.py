"""
Background update checker/installer for FieldWeave.

The Updater runs git/pip work on a background thread and only ever mutates
plain attributes on itself (guarded by a lock). It never touches widgets or
emits Qt signals. A polling QTimer on the main thread (UI/widgets/update_notifier.py)
reads this state and drives all dialogs and notifications.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

from common.logger import info, warning, error

_ERROR_MESSAGE_LIMIT = 500
_NO_UPSTREAM_MARKER = "no upstream configured"


class UpdateStatus:
    IDLE = "idle"
    CHECKING = "checking"
    UPDATE_AVAILABLE = "update_available"
    UP_TO_DATE = "up_to_date"
    CHECK_FAILED = "check_failed"
    UPDATING = "updating"
    UPDATE_COMPLETE = "update_complete"
    UPDATE_FAILED = "update_failed"
    NO_UPSTREAM = "no_upstream"


class Updater:
    def __init__(self, repo_dir: Path | None = None) -> None:
        self._repo_dir = repo_dir or Path.cwd()
        self._lock = threading.Lock()

        self._status: str = UpdateStatus.IDLE
        self._local_commit: str = ""
        self._remote_commit: str = ""
        self._commits_behind: int = 0
        self._error_message: str = ""

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
    def commits_behind(self) -> int:
        with self._lock:
            return self._commits_behind

    @property
    def error_message(self) -> str:
        with self._lock:
            return self._error_message

    def is_busy(self) -> bool:
        with self._lock:
            return self._status in (UpdateStatus.CHECKING, UpdateStatus.UPDATING)

    # ------------------------------------------------------------------
    # Public control — called from the main thread
    # ------------------------------------------------------------------

    def start_check(self) -> bool:
        """Kick off a background update check. Returns False if a check/update is already running."""
        if self.is_busy():
            return False

        if not self._is_git_checkout():
            self._fail_check("Not running from a git checkout - update checking unavailable")
            return False

        self._set_status(UpdateStatus.CHECKING)
        self._check_thread = threading.Thread(target=self._run_check, daemon=True)
        self._check_thread.start()
        return True

    def start_update(self) -> bool:
        """Kick off a background git pull + dependency install. Returns False if already busy."""
        if self.is_busy():
            return False

        if not self._is_git_checkout():
            self._fail_update("Not running from a git checkout - update unavailable")
            return False

        self._set_status(UpdateStatus.UPDATING)
        self._update_thread = threading.Thread(target=self._run_update, daemon=True)
        self._update_thread.start()
        return True

    def reset(self) -> None:
        """Return to IDLE so a fresh check can be started (e.g. after a dismissed prompt)."""
        self._set_status(UpdateStatus.IDLE)

    # ------------------------------------------------------------------
    # Background thread work
    # ------------------------------------------------------------------

    def _run_check(self) -> None:
        success, detail = self._git_fetch()
        if not success:
            self._fail_check(f"Could not reach git remote - {detail}")
            return

        local = self._git_rev_parse("HEAD")
        if not local:
            self._fail_check("Could not determine local commit")
            return

        success, remote, stderr = self._run_git(["rev-parse", "@{u}"], timeout=10, log_errors=False)
        if not success:
            if _NO_UPSTREAM_MARKER in stderr.lower():
                info("Current branch has no upstream - skipping update check")
                with self._lock:
                    self._status = UpdateStatus.NO_UPSTREAM
                return
            warning(f"git rev-parse @{{u}} failed: {stderr}")
            self._fail_check("Could not determine remote commit - is the current branch tracking a remote?")
            return

        behind = self._git_commits_behind(local, remote)

        with self._lock:
            self._local_commit = local
            self._remote_commit = remote
            self._commits_behind = behind
            self._status = UpdateStatus.UPDATE_AVAILABLE if behind > 0 else UpdateStatus.UP_TO_DATE

        info(f"Update check complete: {behind} commit(s) behind")

    def _run_update(self) -> None:
        success, detail = self._git_pull()
        if not success:
            self._fail_update(f"git pull failed - {detail}")
            return

        success, detail = self._install_requirements()
        if not success:
            self._fail_update(f"Failed to install dependencies - {detail}")
            return

        with self._lock:
            self._status = UpdateStatus.UPDATE_COMPLETE

        info("Update applied successfully - restart required")

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

    def _git_fetch(self) -> tuple[bool, str]:
        success, _, stderr = self._run_git(["fetch"], timeout=30)
        return success, self._summarize(stderr) if not success else ""

    def _git_rev_parse(self, ref: str) -> str:
        success, stdout, stderr = self._run_git(["rev-parse", ref], timeout=10)
        if not success:
            warning(f"git rev-parse {ref} failed: {stderr}")
            return ""
        return stdout

    def _git_commits_behind(self, local: str, remote: str) -> int:
        if local == remote:
            return 0
        success, stdout, stderr = self._run_git(["rev-list", "--count", f"{local}..{remote}"], timeout=10)
        if not success or not stdout.isdigit():
            warning(f"git rev-list failed: {stderr}")
            return 0
        return int(stdout)

    def _git_pull(self) -> tuple[bool, str]:
        success, stdout, stderr = self._run_git(["pull"], timeout=60)
        if success:
            return True, ""
        return False, self._summarize(stderr or stdout)

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