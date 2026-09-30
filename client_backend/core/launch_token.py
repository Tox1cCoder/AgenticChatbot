"""The per-launch token that proves a request comes from this machine's user.

Every sidecar start writes a fresh random token to a file only the signed-in user
can read. The Streamlit frontend (``demo.py``) reads it and sends it in the
``X-Kani-Client`` header; the sandbox account, other local accounts and web pages
cannot read the file, so they cannot send it.

The file lives in the profile root, which sits in the user's own profile folder
and is closed to other accounts. It is not the sandbox runtime folder in
ProgramData: the sandbox account is granted read access there. On Windows the
file's permissions are also replaced with a single grant to the current user, so
it stays private even when ``CLIENT_PROFILE_ROOT`` points somewhere shared.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
from pathlib import Path

from client_backend.core.config import client_settings

__all__ = [
    "HEADER_NAME",
    "LaunchTokenError",
    "current_launch_token",
    "issue_launch_token",
    "launch_token_matches",
    "launch_token_path",
    "read_launch_token",
    "revoke_launch_token",
]

HEADER_NAME = "X-Kani-Client"
_FILENAME = ".client_launch_token"

_current_token: bytes | None = None


class LaunchTokenError(RuntimeError):
    """The launch token file could not be written privately."""


def launch_token_path() -> Path:
    """Where this profile's launch token is written. Shared by sidecar and frontend."""

    return Path(client_settings.profile_root) / _FILENAME


def issue_launch_token() -> str:
    """Create this launch's token, write it owner-only, and start accepting it.

    The previous launch's token stops working: the in-memory value is replaced and
    the file is overwritten atomically, so a reader sees the old or the new token,
    never a partial one.
    """

    global _current_token
    token = secrets.token_urlsafe(32)
    _write_owner_only(launch_token_path(), token)
    _current_token = token.encode("ascii")
    return token


def revoke_launch_token() -> None:
    """Stop accepting the current token and remove its file if it is still ours."""

    global _current_token
    issued, _current_token = _current_token, None
    if issued is None:
        return
    path = launch_token_path()
    try:
        if path.read_bytes().strip() == issued:
            path.unlink()
    except OSError:
        # Already gone, or replaced by a newer launch: nothing of ours to remove.
        pass


def current_launch_token() -> bytes | None:
    """The token this process accepts, or None when none was issued."""

    return _current_token


def launch_token_matches(presented: str) -> bool:
    """Constant-time comparison of a presented header value with the issued token."""

    issued = _current_token
    if issued is None:
        return False
    # Bytes, not str: compare_digest raises TypeError on non-ASCII str, which would
    # turn a junk header into a 500 instead of a 401.
    return secrets.compare_digest(presented.strip().encode("utf-8"), issued)


def read_launch_token() -> str | None:
    """Read the token the running sidecar issued, for a client of that sidecar."""

    try:
        token = launch_token_path().read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return token or None


def _write_owner_only(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f"{path.name}.{secrets.token_hex(6)}.tmp")
    # O_EXCL with 0o600: on POSIX there is no moment when the umask leaves the
    # staged file readable by others. On Windows the mode only sets the read-only
    # bit, so the ACL is replaced before anything is written.
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        try:
            if sys.platform == "win32":
                _restrict_to_current_user(staging)
            os.write(descriptor, token.encode("ascii"))
        finally:
            os.close(descriptor)
        os.replace(staging, path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise


def _restrict_to_current_user(path: Path) -> None:
    """Replace the file's inherited permissions with one grant to the current user."""

    username = os.environ.get("USERNAME", "")
    if not username:
        raise LaunchTokenError(f"Cannot restrict {path}: USERNAME is not set")
    domain = os.environ.get("USERDOMAIN", "")
    principal = f"{domain}\\{username}" if domain else username
    completed = subprocess.run(
        ["icacls", str(path), "/inheritance:r", "/grant:r", f"{principal}:F"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise LaunchTokenError(f"Restricting {path} to {principal} failed: {detail}")
