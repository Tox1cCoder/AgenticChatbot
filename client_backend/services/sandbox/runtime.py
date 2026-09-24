"""What the sandbox account may reach, and the copy of Desktop Commander it runs.

The account can change files only under the workspace roots it is granted,
and can read and run -- not change -- the runtime folder holding its node and
Desktop Commander. None of this needs administrator rights.

The runtime cannot live in the user's profile: node reads the attributes of
every folder on the way to its script, and profile folders are closed to other
accounts. It lives in ProgramData instead, in a folder with a random name that
this process creates exclusively and locks down, because any account may
create folders there and one made in advance would be its maker's to rewrite.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
from pathlib import Path

from client_backend.core.config import client_settings
from client_backend.services.desktop_commander_policy import PACKAGE, PINNED_VERSION
from client_backend.services.sandbox.account import SANDBOX_USERNAME

__all__ = [
    "PINNED_VERSION",
    "SandboxRuntimeError",
    "desktop_commander_entry",
    "ensure_runtime_root",
    "ensure_workspace_access",
    "grant_workspace_access",
    "install_desktop_commander",
    "node_executable",
]

_SYSTEM_SID = "*S-1-5-18"
_ADMINISTRATORS_SID = "*S-1-5-32-544"


class SandboxRuntimeError(RuntimeError):
    """Preparing a folder or program for the sandbox account failed."""


def grant_workspace_access(folder: Path, principal: str = SANDBOX_USERNAME) -> None:
    """Let the account create and change anything under ``folder``."""

    _icacls(folder, "/grant", f"{principal}:(OI)(CI)M")


def ensure_workspace_access(folder: Path, principal: str = SANDBOX_USERNAME) -> None:
    """Grant workspace access unless ``folder`` already has it.

    Granting again is not free: Windows stamps the permission onto every file
    below the folder anew, which on a large project is slow on every start.
    """

    completed = subprocess.run(
        ["icacls", str(folder)], capture_output=True, text=True, check=False
    )
    # icacls prints DOMAIN\name:(flags); an inherited grant from a granted
    # parent counts too.
    marker = f"\\{principal}:".lower()
    for line in completed.stdout.lower().splitlines():
        if marker in line and "(oi)(ci)" in line and ("(m)" in line or "(f)" in line):
            return
    grant_workspace_access(folder, principal)


def ensure_runtime_root(principal: str = SANDBOX_USERNAME) -> Path:
    """The runtime folder, created and locked down on first use."""

    record = _runtime_record_path()
    if record.is_file():
        recorded = Path(json.loads(record.read_text(encoding="utf-8"))["path"])
        if recorded.is_dir():
            return recorded

    folder = Path(os.environ["PROGRAMDATA"]) / f"KaniDesktop-runtime-{secrets.token_hex(8)}"
    os.mkdir(folder)  # exclusive: it cannot already belong to anyone else
    user = f"{os.environ['USERDOMAIN']}\\{os.environ['USERNAME']}"
    _icacls(
        folder,
        "/inheritance:r",
        "/grant:r",
        f"{_SYSTEM_SID}:(OI)(CI)F",
        f"{_ADMINISTRATORS_SID}:(OI)(CI)F",
        f"{user}:(OI)(CI)F",
        f"{principal}:(OI)(CI)RX",
    )
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"path": str(folder)}), encoding="utf-8")
    return folder


def desktop_commander_entry(root: Path, version: str = PINNED_VERSION) -> Path:
    return root / f"desktop-commander-{version}" / "node_modules" / PACKAGE / "dist" / "index.js"


def _runtime_record_path() -> Path:
    return Path(client_settings.profile_root) / "sandbox" / "runtime.json"


def install_desktop_commander(version: str = PINNED_VERSION) -> Path:
    """Install the pinned build once, for the account to read and run.

    ``--ignore-scripts`` skips the package's install hooks: one sends the
    vendor's install telemetry regardless of the telemetry setting, another
    downloads Chromium for PDF rendering. Search still works, because
    ripgrep's binary ships in its platform package.
    """

    root = ensure_runtime_root()
    entry = desktop_commander_entry(root, version)
    if entry.is_file():
        return entry
    target = root / f"desktop-commander-{version}"
    target.mkdir(parents=True, exist_ok=True)
    npm = shutil.which("npm") or "npm"
    completed = subprocess.run(
        [
            npm,
            "install",
            "--prefix",
            str(target),
            "--ignore-scripts",
            "--no-audit",
            "--no-fund",
            f"{PACKAGE}@{version}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
        raise SandboxRuntimeError(f"Installing Desktop Commander {version} failed: {detail}")
    return entry


def node_executable() -> Path:
    """A node the account can run.

    A node under the user's own profile (fnm, nvm, a per-user installer) is
    closed to the account, so it is copied into the runtime folder.
    """

    found = shutil.which("node")
    if not found:
        raise SandboxRuntimeError("Node.js is not installed or not on PATH")
    node = Path(found).resolve()
    if not node.is_relative_to(Path.home().resolve()):
        return node
    copy = ensure_runtime_root() / "node" / node.name
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(node, copy)
    return copy


def _icacls(folder: Path, *arguments: str) -> None:
    completed = subprocess.run(
        ["icacls", str(folder), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise SandboxRuntimeError(f"Granting access to {folder} failed: {detail}")
