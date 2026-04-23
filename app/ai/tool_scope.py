"""Tool scope resolution.

Defines the execution scope used when binding and searching tools for a given
request. Two scopes are supported:

- ``DEFAULT``: the normal mode; both server MCP tools and device-scoped client
  tools are eligible for binding, tool_search, and execution.
- ``CLIENT_ONLY``: the model and tool_search operate only against the active
  device's client-local catalog. Server MCP tools are suppressed.

``CLIENT_ONLY`` only has meaning when the request carries a concrete
``device_id`` — without one there are no client tools to scope to, so the
helpers fall back to ``DEFAULT`` in that case.
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
    values resolve to ``DEFAULT``. A ``CLIENT_ONLY`` hint is downgraded to
    ``DEFAULT`` when no ``device_id`` is available, since there would be no
    client catalog to scope to.
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

    if candidate is ToolScope.CLIENT_ONLY and not device_id:
        return ToolScope.DEFAULT
    return candidate


def is_client_only_scope(
    *,
    device_id: str | None = None,
    tool_scope: str | ToolScope | None = None,
) -> bool:
    """Return ``True`` when the effective scope is ``CLIENT_ONLY``."""
    return resolve_tool_scope(device_id=device_id, tool_scope=tool_scope) is ToolScope.CLIENT_ONLY
