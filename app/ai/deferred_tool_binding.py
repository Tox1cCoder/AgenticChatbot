"""
Deferred tool binding helpers for agents.

This module provides utilities for agents to build their tool list
when mcp_tool_search is enabled. Instead of binding all MCP tools,
agents bind:
- The tool_search tool itself
- Pinned MCP tools (configured in settings)
- Currently loaded deferred tools for the conversation

This follows the Claude-style pattern for reducing tool schema token bloat.
"""

import copy
from typing import Any

from langchain_core.tools import BaseTool

from ..core.config import settings
from .deferred_tool_state import get_deferred_tool_state
from .mcp_integration import MCPManager
from .tool_search_tool import create_tool_search_tool


def _tool_with_call_name(tool: BaseTool, call_name: str | None) -> BaseTool:
    """Return a copy of ``tool`` bound under the public ``call_name`` alias.

    For ambiguous same-name server tools, the execution map must contain the
    exact public alias (e.g. ``brave__search``) returned by tool_search, not the
    raw MCP tool name (``search``). When ``call_name`` matches the tool's name
    (non-ambiguous case) the original tool is returned unchanged.
    """
    if not call_name or getattr(tool, "name", None) == call_name:
        return tool

    metadata: dict[str, Any] = {
        **(getattr(tool, "metadata", {}) or {}),
        "aliased_from_tool_name": getattr(tool, "name", ""),
        "call_name": call_name,
    }
    updates = {"name": call_name, "metadata": metadata}

    model_copy = getattr(tool, "model_copy", None)
    if callable(model_copy):
        try:
            return model_copy(update=updates)
        except Exception:
            pass

    legacy_copy = getattr(tool, "copy", None)
    if callable(legacy_copy):
        try:
            return legacy_copy(update=updates)
        except Exception:
            pass

    alias = copy.copy(tool)
    try:
        alias.name = call_name
    except Exception:
        object.__setattr__(alias, "name", call_name)
    try:
        alias.metadata = metadata
    except Exception:
        object.__setattr__(alias, "metadata", metadata)
    return alias


_WIDGET_PINNED_AGENT_KEYS = {"chat", "rag", "search"}
_WIDGET_PINNED_SPECS = (
    "widgets::widget_create",
    "widgets::widget_update",
    "widgets::widget_get_state",
)
_SEARCH_AGENT_PINNED_SPECS = (
    "time::get_current_time",
    "tavily::tavily_search",
)


def _get_pinned_specs(agent_key: str | None) -> list[str]:
    pinned_specs = list(settings.mcp_tool_search_pinned_tools or [])
    if agent_key == "search":
        for spec in _SEARCH_AGENT_PINNED_SPECS:
            if spec not in pinned_specs:
                pinned_specs.append(spec)
    if agent_key in _WIDGET_PINNED_AGENT_KEYS:
        for spec in _WIDGET_PINNED_SPECS:
            if spec not in pinned_specs:
                pinned_specs.append(spec)
    return pinned_specs


def get_pinned_tools(
    mcp_manager: MCPManager,
    all_tools: list[BaseTool],
    agent_key: str | None = None,
) -> list[BaseTool]:
    """
    Get the pinned MCP tools that should always be bound.

    Pinned tools are configured in settings.mcp_tool_search_pinned_tools
    and can be specified as:
    - "tool_name" - matches any tool with that name
    - "server_name::tool_name" - matches specific server's tool

    Args:
        mcp_manager: The MCP manager instance
        all_tools: All available tools from MCP

    Returns:
        List of BaseTool objects for pinned tools
    """
    pinned_specs = _get_pinned_specs(agent_key)
    max_pinned = settings.mcp_tool_search_max_pinned_tools

    if not pinned_specs:
        return []

    pinned_tools: list[BaseTool] = []

    for spec in pinned_specs[:max_pinned]:  # Enforce max
        if "::" in spec:
            # Server-qualified: "server_name::tool_name"
            server_name, tool_name = spec.split("::", 1)
            for tool in all_tools:
                if tool.name == tool_name:
                    tool_server = mcp_manager.get_server_for_tool(tool)
                    if tool_server == server_name:
                        pinned_tools.append(tool)
                        break
        else:
            # Just tool name
            for tool in all_tools:
                if tool.name == spec:
                    pinned_tools.append(tool)
                    break

    return pinned_tools


def get_deferred_tools_for_binding(
    conversation_id: str,
    agent_key: str,
    mcp_manager: MCPManager,
    all_tools: list[BaseTool],
) -> list[BaseTool]:
    """
    Get the currently loaded deferred tools for a conversation.

    These are tools that were discovered via tool_search and loaded
    for use in this conversation.

    IMPORTANT: This function retrieves tools from the MCP manager's full tool
    index, NOT from the filtered all_tools list. This ensures that tools found
    by tool_search can be properly bound even if they weren't in the agent's
    initial filtered tool list.

    Args:
        conversation_id: The conversation ID
        agent_key: The agent key (e.g., "chat", "search", "rag")
        mcp_manager: The MCP manager instance
        all_tools: All available tools from MCP (may be deduplicated/filtered)

    Returns:
        List of BaseTool objects for loaded deferred tools
    """
    import logging

    logger = logging.getLogger(__name__)

    state = get_deferred_tool_state()
    loaded_tools = state.get_loaded(conversation_id, agent_key)

    if not loaded_tools:
        return []

    deferred_tools: list[BaseTool] = []

    for loaded in loaded_tools:
        tool = None

        # PRIMARY: Use MCP manager's _tool_index which contains ALL tools
        # This is critical - it gives us access to tools that may have been
        # filtered out of the agent's initial tool list
        if hasattr(mcp_manager, "_tool_index"):
            candidates = mcp_manager._tool_index.get(loaded.tool_name, [])
            for t in candidates:
                if mcp_manager.get_server_for_tool(t) == loaded.server_name:
                    tool = t
                    break

        # SECONDARY: Check _server_tools directly (another way to access full tool set)
        if tool is None and hasattr(mcp_manager, "_server_tools"):
            server_tools = mcp_manager._server_tools.get(loaded.server_name, [])
            for t in server_tools:
                if t.name == loaded.tool_name:
                    tool = t
                    break

        # TERTIARY: Fall back to all_tools if above methods fail
        # Note: This may miss tools that were filtered from all_tools
        if tool is None:
            for t in all_tools:
                if (
                    t.name == loaded.tool_name
                    and mcp_manager.get_server_for_tool(t) == loaded.server_name
                ):
                    tool = t
                    break

        if tool:
            # Bind under the public call_name alias so ambiguous same-name tools
            # are callable under the exact name tool_search returned this turn.
            deferred_tools.append(_tool_with_call_name(tool, loaded.call_name))
        else:
            # Log warning when tool can't be found - this helps debug loading issues
            logger.warning(
                "Deferred tool '%s' from server '%s' could not be found in MCP manager. "
                "The tool may have been removed or the server disconnected.",
                loaded.tool_name,
                loaded.server_name,
            )

    return deferred_tools


def build_deferred_tool_list(
    conversation_id: str | None,
    agent_key: str,
    mcp_manager: MCPManager | None,
    all_mcp_tools: list[BaseTool],
    internal_tools: list[BaseTool] | None = None,
    allowlist: list[str] | None = None,
) -> list[BaseTool]:
    """
    Build the complete tool list for an agent with deferred loading enabled.

    The returned list includes:
    1. Internal tools (e.g., write_todos, search_documents)
    2. tool_search tool (for discovering MCP tools)
    3. Pinned MCP tools (always bound, up to max_pinned)
    4. Loaded deferred tools for this conversation

    Args:
        conversation_id: Current conversation ID (None if no conversation context)
        agent_key: The agent key (e.g., "chat", "search", "rag")
        mcp_manager: The MCP manager instance
        all_mcp_tools: All available tools from MCP
        internal_tools: Non-MCP internal tools to include
        allowlist: Optional allowlist for filtering (applies to tool_search)

    Returns:
        List of BaseTool objects to bind to the model
    """
    result_tools: list[BaseTool] = []
    seen_names: set[str] = set()

    # 1. Add internal tools first
    if internal_tools:
        for tool in internal_tools:
            if tool.name not in seen_names:
                result_tools.append(tool)
                seen_names.add(tool.name)

    # 2. Add tool_search tool
    tool_search = create_tool_search_tool(allowlist=allowlist)
    if tool_search.name not in seen_names:
        result_tools.append(tool_search)
        seen_names.add(tool_search.name)

    # 3. Add pinned MCP tools
    if mcp_manager and all_mcp_tools:
        pinned = get_pinned_tools(mcp_manager, all_mcp_tools, agent_key=agent_key)
        for tool in pinned:
            if tool.name not in seen_names:
                result_tools.append(tool)
                seen_names.add(tool.name)

    # 4. Add loaded deferred tools (if we have a conversation context)
    if conversation_id and mcp_manager and all_mcp_tools:
        deferred = get_deferred_tools_for_binding(
            conversation_id, agent_key, mcp_manager, all_mcp_tools
        )
        for tool in deferred:
            if tool.name not in seen_names:
                result_tools.append(tool)
                seen_names.add(tool.name)

    return result_tools


def should_use_deferred_loading(agent_key: str) -> bool:
    """
    Check if deferred loading should be used for an agent.

    Currently this is controlled by the global mcp_tool_search_enabled flag.
    In the future, this could support per-agent toggles.

    Args:
        agent_key: The agent key (e.g., "chat", "search", "rag")

    Returns:
        True if deferred loading should be used
    """
    return settings.mcp_tool_search_enabled
