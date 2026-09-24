"""Launching Desktop Commander as the sandbox account.

Once the account is set up, Desktop Commander always runs as it, from the
pinned runtime install, with the configured workspace roots granted. The
configured command (``npx ...@latest`` or similar) is not used then: the
account could not run the user's npx cache, and the pinned install is the one
this sidecar has vetted.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import client_backend
from client_backend.core.config import client_settings
from client_backend.services.desktop_commander_policy import NO_ONBOARDING_FLAG
from client_backend.services.sandbox.account import load_credentials
from client_backend.services.sandbox.path_resolution import grant_path_resolution
from client_backend.services.sandbox.runtime import (
    SandboxRuntimeError,
    ensure_workspace_access,
    install_desktop_commander,
    node_executable,
)
from client_backend.services.sandbox.workspace import ensure_managed_workspace

__all__ = [
    "SandboxLaunch",
    "SandboxRuntimeError",
    "prepare_sandbox_launch",
    "sandbox_is_set_up",
]

# Workspace files belong to the user, not the sandbox account, and git refuses
# to work in a repository someone else owns ("dubious ownership").
_GIT_SAFE_DIRECTORY = {
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "safe.directory",
    "GIT_CONFIG_VALUE_0": "*",
}
_LAUNCHER_MODULE = "client_backend.services.sandbox.launcher"
# Where ``python -m client_backend...`` resolves: the source or bundle root.
_PACKAGE_ROOT = Path(client_backend.__file__).resolve().parent.parent


@dataclass(frozen=True)
class SandboxLaunch:
    node: Path
    desktop_commander: Path
    cwd: Path

    def mcp_entry(self, env: Mapping[str, str]) -> dict[str, Any]:
        """The MCP server entry that starts Desktop Commander through the launcher."""

        args = ["-m", _LAUNCHER_MODULE, "--cwd", str(self.cwd)]
        for name, value in {**env, **_GIT_SAFE_DIRECTORY}.items():
            args += ["--env", f"{name}={value}"]
        args += ["--", str(self.node), str(self.desktop_commander), NO_ONBOARDING_FLAG]
        return {
            "transport": "stdio",
            "command": sys.executable,
            "args": args,
            "cwd": str(_PACKAGE_ROOT),
        }


def sandbox_is_set_up() -> bool:
    return load_credentials() is not None


def prepare_sandbox_launch() -> SandboxLaunch:
    """Install the runtime and grant the workspaces. Blocking: run it in a thread.

    The first grant on a large folder can take a while, because Windows
    stamps the new permission onto every file below it.
    """

    entry = install_desktop_commander()
    node = node_executable()
    roots = [Path(root) for root in client_settings.workspace_roots]
    if not roots:
        # No workspace configured: give the account a clean one it may write,
        # under the profile, rather than the read-only runtime folder, where it
        # can create nothing. A configured root is used as-is (and still
        # refused by ensure_workspace_access if it holds a .git or environment).
        roots = [ensure_managed_workspace()]
    for root in roots:
        ensure_workspace_access(root)
        # Without this, PowerShell cannot start in a root under the user's
        # profile and silently starts at the drive root instead.
        grant_path_resolution(root)
    cwd = roots[0]
    return SandboxLaunch(node=node, desktop_commander=entry, cwd=cwd)
