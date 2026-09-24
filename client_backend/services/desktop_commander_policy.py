"""Launch hardening and tool policy for the Desktop Commander MCP server.

Desktop Commander runs shell commands and edits files with the user's full
privileges, on behalf of a remote model. So the sidecar:

- pins the build it launches, switches off the vendor's telemetry and
  onboarding, and sets a non-interactive environment;
- hides the tools that let a model loosen the server's own limits or read
  other clients' history;
- marks every tool not known to be read-only as a mutation, which routes it
  through human approval on the server;
- refuses file tools on credential locations, because the read tools are not
  approval-gated and a prompt injection could otherwise read a private key.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from client_backend.core.config import client_settings

PACKAGE = "@wonderwhy-er/desktop-commander"
# Bumped deliberately, never followed: each build runs with the user's rights.
PINNED_VERSION = "0.2.51"

TELEMETRY_ENV = "DESKTOP_COMMANDER_DISABLE_TELEMETRY"
# Onboarding can open a welcome page and, behind a vendor-served feature flag,
# add vendor text to tool results the model reads.
NO_ONBOARDING_FLAG = "--no-onboarding"

# ``set_config_value`` rewrites blockedCommands and allowedDirectories in a
# config file shared by every Desktop Commander client of this OS user. The
# history and stats tools expose those other clients' calls, and the feedback
# tool opens the vendor's form.
_HIDDEN_TOOLS = frozenset(
    {
        "set_config_value",
        "get_recent_tool_calls",
        "get_usage_stats",
        "give_feedback_to_desktop_commander",
    }
)

# Anything else -- including tools a later release adds -- is a mutation.
_READ_ONLY_TOOLS = frozenset(
    {
        "get_config",
        "get_file_info",
        "get_more_search_results",
        "get_prompts",
        "list_directory",
        "list_processes",
        "list_searches",
        "list_sessions",
        "read_file",
        "read_multiple_files",
        "read_process_output",
        "start_search",
        "stop_search",
    }
)

_PACKAGE_SPEC = re.compile(rf"^{re.escape(PACKAGE)}(?:@(?P<version>.+))?$")
_EXACT_VERSION = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")

# Commands run with no one at the keyboard: a prompt waits until the deadline.
# git fails at once instead of asking for credentials; npx installs instead of
# asking "Ok to proceed?" (the command itself was already approved).
_NON_INTERACTIVE_ENV = {"GIT_TERMINAL_PROMPT": "0", "npm_config_yes": "true"}

# The arguments through which each file tool names the paths it touches.
_PATH_ARGUMENTS = {
    "read_file": ("path",),
    "read_multiple_files": ("paths",),
    "write_file": ("path",),
    "write_pdf": ("path", "outputPath"),
    "create_directory": ("path",),
    "list_directory": ("path",),
    "move_file": ("source", "destination"),
    "start_search": ("path",),
    "get_file_info": ("path",),
    "edit_block": ("file_path",),
}

_SECRET_FILE_NAME = re.compile(
    r"^(\.env(\..+)?|id_(rsa|dsa|ecdsa|ed25519)(\..+)?|.+\.(pem|key|pfx|p12|keystore|jks))$",
    re.IGNORECASE,
)
_TEMPLATE_SUFFIXES = (".example", ".sample", ".template")


class SensitivePathError(PermissionError):
    """Desktop Commander was asked to touch a credential location."""


def is_desktop_commander(command: str | None, args: Sequence[str]) -> bool:
    """Whether a launch runs Desktop Commander, through npx, node, or its binary."""

    return any("desktop-commander" in str(token).lower() for token in (command or "", *args))


def harden_launch(
    args: Sequence[str],
    env: Mapping[str, str] | None,
) -> tuple[list[str], dict[str, str]]:
    """Return launch arguments and environment with the pin and quiet flags applied.

    An exact version the user chose, and a telemetry setting they chose, are
    kept: the goal is that nothing changes underneath them, not to override
    their decisions.
    """

    hardened_args = [_pinned(arg) for arg in args]
    if NO_ONBOARDING_FLAG not in hardened_args:
        hardened_args.append(NO_ONBOARDING_FLAG)
    hardened_env = dict(env or {})
    hardened_env.setdefault(TELEMETRY_ENV, "1")
    for name, value in _NON_INTERACTIVE_ENV.items():
        hardened_env.setdefault(name, value)
    return hardened_args, hardened_env


def refuse_sensitive_paths(tool_name: str, arguments: Mapping[str, Any]) -> None:
    """Raise ``SensitivePathError`` when a file tool names a credential location.

    A guardrail, not a boundary: a shell command can still reach these files,
    but ``start_process`` needs approval and shows the command, while these
    read tools do not. Links are followed, so a junction inside a project that
    points into ``~/.ssh`` is refused like ``~/.ssh`` itself.
    """

    keys = _PATH_ARGUMENTS.get(tool_name)
    if keys is None or (tool_name == "read_file" and _is_web_url(arguments)):
        return
    folders = _credential_folders()
    for raw in _path_values(arguments, keys):
        path = _resolve(raw)
        if any(_is_within(path, folder) for folder in folders):
            raise SensitivePathError(_refusal(raw, "it is inside a credential folder"))
        if _SECRET_FILE_NAME.match(path.name) and not path.name.lower().endswith(
            _TEMPLATE_SUFFIXES
        ):
            raise SensitivePathError(_refusal(raw, "it is a secrets or key file"))

    if tool_name == "start_search" and arguments.get("includeHidden"):
        root = _resolve(arguments.get("path") or ".")
        if any(_is_within(folder, root) for folder in folders):
            raise SensitivePathError(
                _refusal(
                    str(arguments.get("path")),
                    "a search that includes hidden files would reach credential folders "
                    "under it; search without includeHidden or a narrower folder",
                )
            )


def _credential_folders() -> list[Path]:
    home = Path.home()
    roaming = Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming")
    local = Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
    folders = [
        home / ".ssh",
        home / ".aws",
        home / ".azure",
        home / ".gnupg",
        home / ".kube",
        home / ".docker",
        home / ".config" / "gcloud",
        home / ".config" / "gh",
        home / ".claude",
        home / ".claude-server-commander",
        home / ".git-credentials",
        home / ".npmrc",
        home / ".pypirc",
        home / ".netrc",
        local / "Google" / "Chrome" / "User Data",
        local / "Microsoft" / "Edge" / "User Data",
        local / "BraveSoftware" / "Brave-Browser" / "User Data",
        roaming / "Mozilla" / "Firefox" / "Profiles",
        roaming / "Microsoft" / "Credentials",
        roaming / "Microsoft" / "Protect",
        local / "Microsoft" / "Credentials",
        Path(client_settings.profile_root),
    ]
    return [_resolve(folder) for folder in folders]


def _path_values(arguments: Mapping[str, Any], keys: Sequence[str]) -> Iterator[str]:
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            yield value
        elif isinstance(value, list):
            yield from (item for item in value if isinstance(item, str) and item.strip())


def _resolve(raw: str | Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(raw)))
    return Path(expanded).resolve(strict=False)


def _is_within(path: Path, folder: Path) -> bool:
    candidate, parent = os.path.normcase(str(path)), os.path.normcase(str(folder))
    return candidate == parent or candidate.startswith(parent.rstrip(os.sep) + os.sep)


def _is_web_url(arguments: Mapping[str, Any]) -> bool:
    if not arguments.get("isUrl"):
        return False
    return urlsplit(str(arguments.get("path") or "")).scheme.lower() in {"http", "https"}


def _refusal(raw: str, reason: str) -> str:
    return (
        f"Desktop Commander may not use {raw!r}: {reason}. Credential locations are "
        "closed to the assistant; ask the user for the specific non-secret detail instead."
    )


def is_hidden_tool(tool_name: str) -> bool:
    return tool_name in _HIDDEN_TOOLS


def is_mutating_tool(tool_name: str) -> bool:
    return tool_name not in _READ_ONLY_TOOLS


def _pinned(arg: str) -> str:
    match = _PACKAGE_SPEC.match(arg)
    if match is None:
        return arg
    version = match.group("version")
    if version and _EXACT_VERSION.match(version):
        return arg
    return f"{PACKAGE}@{PINNED_VERSION}"
