"""Launch hardening and tool policy for the Desktop Commander MCP server.

Desktop Commander runs shell commands and edits files with the user's full
privileges, on behalf of a remote model. So the sidecar pins the build it
launches, switches off the vendor's telemetry and onboarding, hides the tools
that let a model loosen the server's own limits or read other clients'
history, and marks every tool not known to be read-only as a mutation, which
routes it through human approval on the server.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

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
    return hardened_args, hardened_env


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
