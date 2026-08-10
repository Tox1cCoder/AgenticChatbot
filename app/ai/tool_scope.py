"""Tool scope resolution.

Defines the execution scope used when binding and searching tools for a given
request. Two scopes are supported:

- ``DEFAULT``: the normal mode; both server MCP tools and device-scoped client
  tools are eligible for binding, tool_search, and execution.
- ``CLIENT_ONLY``: the model and tool_search operate only against the active
  device's client-local catalog. Server MCP tools are suppressed.

``CLIENT_ONLY`` remains restrictive even when a request has no concrete
``device_id``. In that case there are no eligible client tools, but silently
falling back to ``DEFAULT`` would expose server tools against the caller's
explicit scope request.
"""

from __future__ import annotations

from enum import Enum


class ToolScope(str, Enum):
    """Execution scope for tool binding, search, and execution."""

    DEFAULT = "default"
    CLIENT_ONLY = "client_only"


def resolve_tool_scope(
    *,
    device_id: str | None = None,
    tool_scope: str | ToolScope | None = None,
) -> ToolScope:
    """Resolve a scope hint into a concrete ``ToolScope``.

    Accepts either a ``ToolScope`` instance or a string value. Unknown string
    values resolve to ``DEFAULT``. An explicit ``CLIENT_ONLY`` hint remains
    client-only when no ``device_id`` is available, yielding an empty client
    catalog while continuing to suppress server tools.
    """
    if isinstance(tool_scope, ToolScope):
        candidate = tool_scope
    elif isinstance(tool_scope, str):
        try:
            candidate = ToolScope(tool_scope.strip().lower())
        except ValueError:
            candidate = ToolScope.DEFAULT
    else:
        candidate = ToolScope.DEFAULT

    return candidate


def is_client_only_scope(
    *,
    device_id: str | None = None,
    tool_scope: str | ToolScope | None = None,
) -> bool:
    """Return ``True`` when the effective scope is ``CLIENT_ONLY``."""
    return resolve_tool_scope(device_id=device_id, tool_scope=tool_scope) is ToolScope.CLIENT_ONLY
