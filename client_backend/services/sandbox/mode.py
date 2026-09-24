"""The sandbox mode in effect on this device, and how to change it at runtime.

``CLIENT_SANDBOX_MODE`` in the environment is the default. A device override,
set from the app, is written to the profile and read live, so flipping the
switch takes effect on the next Desktop Commander launch without restarting the
sidecar. A missing or unreadable override falls back to the configured default:
a corrupt file must never leave Desktop Commander stuck in a mode nobody chose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast, get_args

from client_backend.core.config import client_settings

SandboxMode = Literal["off", "workspace"]
_MODES = frozenset(get_args(SandboxMode))

__all__ = ["SandboxMode", "read_sandbox_mode", "sandbox_mode_path", "write_sandbox_mode"]


def sandbox_mode_path() -> Path:
    return Path(client_settings.profile_root) / "sandbox" / "mode.json"


def read_sandbox_mode() -> SandboxMode:
    """The device override if one is set and valid, else the configured default."""

    try:
        stored = json.loads(sandbox_mode_path().read_text(encoding="utf-8")).get("mode")
    except (OSError, ValueError, AttributeError):
        stored = None
    if stored in _MODES:
        return cast(SandboxMode, stored)
    return client_settings.sandbox_mode


def write_sandbox_mode(mode: str) -> SandboxMode:
    """Persist the device override, refusing any value that is not a real mode."""

    if mode not in _MODES:
        raise ValueError(f"sandbox mode must be one of {sorted(_MODES)}, not {mode!r}")
    path = sandbox_mode_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mode": mode}), encoding="utf-8")
    return cast(SandboxMode, mode)
