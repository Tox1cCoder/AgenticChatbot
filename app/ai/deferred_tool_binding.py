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

from typing import List, Optional, Set

from langchain_core.tools import BaseTool

from ..core.config import settings
from .mcp_integration import MCPManager
from .deferred_tool_state import get_deferred_tool_state
from .tool_search_tool import create_tool_search_tool


def get_pinned_tools(
    mcp_manager: MCPManager,
    all_tools: List[BaseTool],
) -> List[BaseTool]:
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
    pinned_specs = settings.mcp_tool_search_pinned_tools or []
    max_pinned = settings.mcp_tool_search_max_pinned_tools

    if not pinned_specs:
        return []

    pinned_tools: List[BaseTool] = []

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
    all_tools: List[BaseTool],
) -> List[BaseTool]:
    """
    Get the currently loaded deferred tools for a conversation.

    These are tools that were discovered via tool_search and loaded
    for use in this conversation.

    Args:
        conversation_id: The conversation ID
        agent_key: The agent key (e.g., "chat", "search", "rag")
        mcp_manager: The MCP manager instance
        all_tools: All available tools from MCP (may be deduplicated)

    Returns:
        List of BaseTool objects for loaded deferred tools
    """

    state = get_deferred_tool_state()
    loaded_tools = state.get_loaded(conversation_id, agent_key)

    if not loaded_tools:
        return []

    deferred_tools: List[BaseTool] = []

    for loaded in loaded_tools:
        # Use MCP manager's get_tool_by_name with server_name to handle collisions correctly
        tool = None

        # First try direct lookup from _tool_index (handles collision correctly)
        if hasattr(mcp_manager, "_tool_index"):
            candidates = mcp_manager._tool_index.get(loaded.tool_name, [])
            for t in candidates:
                if mcp_manager.get_server_for_tool(t) == loaded.server_name:
                    tool = t
                    break

        # Fallback: search through all_tools (may not find the right server version)
        if tool is None:
            for t in all_tools:
                if t.name == loaded.tool_name:
                    if mcp_manager.get_server_for_tool(t) == loaded.server_name:
                        tool = t
                        break

        if tool:
            deferred_tools.append(tool)

    return deferred_tools


def build_deferred_tool_list(
    conversation_id: Optional[str],
    agent_key: str,
    mcp_manager: Optional[MCPManager],
    all_mcp_tools: List[BaseTool],
    internal_tools: Optional[List[BaseTool]] = None,
    allowlist: Optional[List[str]] = None,
) -> List[BaseTool]:
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
    result_tools: List[BaseTool] = []
    seen_names: Set[str] = set()

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
        pinned = get_pinned_tools(mcp_manager, all_mcp_tools)
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
