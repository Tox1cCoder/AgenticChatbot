"""
Tool Search Tool - Claude-style deferred MCP tool discovery.

This module provides unified tool search across BOTH server-side MCP tools
AND client device tools. The search behavior is consistent regardless of
tool origin - the only difference is the tool list available on each device.

Key features:
- Searches server MCP tools (from McpToolCatalog)
- Searches client device tools (from ClientToolCatalog) when a device is connected
- Results include origin information (server_mcp, client_mcp, client_native)
- Autoloading works for both server and client tools
"""

import json
import logging
import re
import time
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..core.config import settings
from .client_tool_catalog import ClientToolReference, get_client_tool_catalog
from .deferred_tool_state import get_deferred_tool_state
from .mcp_registry import get_global_mcp_manager
from .mcp_tool_catalog import (
    ToolReference,
    get_tool_catalog,
)
from .tool_context import get_tool_context

logger = logging.getLogger(__name__)

_TOOL_SEARCH_QUERY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "computer": ("local", "device", "client"),
    "machine": ("local", "device", "client"),
    "pc": ("local", "device", "client"),
    "local": ("device", "client"),
    "device": ("local", "client"),
    "file": ("files", "filesystem", "path"),
    "files": ("file", "filesystem", "directory", "folder", "path"),
    "folder": ("directory", "filesystem", "path"),
    "directory": ("folder", "filesystem", "path"),
    "edit": ("write", "update", "modify"),
    "write": ("edit", "save", "create"),
    "read": ("open", "view", "inspect"),
    "search": ("find", "lookup"),
    "find": ("search", "lookup"),
    "shell": ("terminal", "command", "execute", "run"),
    "terminal": ("shell", "command", "execute", "run"),
    "command": ("shell", "terminal", "execute", "run"),
    "run": ("execute", "shell", "command"),
}


def _expand_tool_search_query(query: str | None) -> str | None:
    """
    Expand a natural-language tool query with a few capability aliases.

    This keeps tool_search flexible when users or agents describe local-device
    work in different words, such as "computer" vs "device" or "edit" vs
    "write", without hard-coding specific tool names.
    """
    if not query or not query.strip():
        return query

    tokens = [tok.lower() for tok in re.split(r"[^a-zA-Z0-9_]+", query) if tok]
    if not tokens:
        return query

    expanded_tokens: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        if token not in seen:
            expanded_tokens.append(token)
            seen.add(token)

        for alias in _TOOL_SEARCH_QUERY_SYNONYMS.get(token, ()):
            if alias not in seen:
                expanded_tokens.append(alias)
                seen.add(alias)

    return " ".join(expanded_tokens)


class ToolSearchInput(BaseModel):
    """Input schema for the tool_search tool."""

    query: str | None = Field(
        default=None,
        description=(
            "Describe the job, target, and environment in natural language. "
            "Prefer specific phrases like 'read local text file', "
            "'run shell command', or 'web search current news'. "
            "Leave empty to list all available tools."
        ),
    )
    top_k: int | None = Field(
        default=None,
        description=(
            "Maximum number of tools to return. Defaults to 5. "
            "Use a larger value to see more options."
        ),
    )
    server_name: str | None = Field(
        default=None,
        description=(
            "Filter results to a specific MCP server. "
            "Use this to disambiguate when multiple servers provide tools with the same name."
        ),
    )


class ToolSearchResult(BaseModel):
    """Individual tool result from search."""

    tool_name: str = Field(description="The exact name to use when calling this tool")
    description: str = Field(description="What the tool does")
    arg_hints: str = Field(description="Summary of arguments (* = required)")
    is_loaded: bool = Field(
        default=False, description="True if this tool was autoloaded and is ready for immediate use"
    )


class ToolSearchOutput(BaseModel):
    """Output schema for the tool_search tool."""

    query: str | None = Field(description="The search query used")
    results: list[ToolSearchResult] = Field(description="List of matching tools")
    loaded_count: int = Field(
        description="Number of tools that were autoloaded and ready for immediate use"
    )
    more_available: bool = Field(description="True if more results exist beyond the returned list")


async def _execute_tool_search(
    query: str | None = None,
    top_k: int | None = None,
    server_name: str | None = None,
    allowlist: list[str] | None = None,
) -> dict[str, Any]:
    """
    Core implementation of tool search logic.

    Searches BOTH server MCP tools AND client device tools (if a device is connected).
    Results are merged and ranked by relevance.

    Args:
        query: Search query (None for list all)
        top_k: Max results to return
        server_name: Optional server filter
        allowlist: Optional per-agent allowlist filter

    Returns:
        Dict with search results and metadata
    """
    start_time = time.time()

    # Apply config defaults and limits
    default_top_k = settings.mcp_tool_search_default_top_k
    max_top_k = settings.mcp_tool_search_max_top_k
    autoload_top_k = settings.mcp_tool_search_autoload_top_k

    effective_top_k = top_k if top_k is not None else default_top_k
    effective_top_k = max(1, min(effective_top_k, max_top_k))
    expanded_query = _expand_tool_search_query(query)

    # Get tool context for conversation-scoped loading
    ctx = get_tool_context()
    conversation_id = ctx.conversation_id
    agent_key = ctx.agent_key
    device_id = ctx.device_id
    user_id = ctx.user_id

    # Log query if enabled
    if settings.mcp_tool_search_log_queries:
        logger.info(
            "tool_search: query=%r expanded=%r top_k=%d server=%s conversation=%s agent=%s device=%s",
            query,
            expanded_query,
            effective_top_k,
            server_name,
            conversation_id,
            agent_key,
            device_id[:8] if device_id else None,
        )

    # Get the MCP manager and catalog for SERVER tools
    unavailable_servers: list[str] = []
    server_results = []
    catalog = None
    try:
        mcp_manager = await get_global_mcp_manager()
        catalog = await get_tool_catalog(mcp_manager)

        # Search server tools
        server_results = catalog.search(
            query=expanded_query,
            top_k=effective_top_k * 2,  # Request extra for merging
            server_name=server_name,
            allowlist=allowlist,
        )
    except Exception as e:
        logger.error("Failed to get server tool catalog: %s", e)
        unavailable_servers.append("all_server")

    # Get CLIENT tools if device is connected
    client_results = []
    client_catalog = None

    if device_id and user_id:
        try:
            client_catalog = get_client_tool_catalog(device_id, user_id)
            if client_catalog.tool_count > 0:
                client_results = client_catalog.search(
                    query=expanded_query,
                    top_k=effective_top_k * 2,  # Request extra for merging
                    server_name=server_name,
                    allowlist=allowlist,
                )
                logger.debug(
                    "tool_search: found %d client tools from device %s",
                    len(client_results),
                    device_id[:8],
                )
        except Exception as e:
            logger.warning("Failed to search client tool catalog: %s", e)

    # Merge and rank results from both sources
    # Returns both public (for model) and internal (for autoloading) versions
    public_results, internal_results = _merge_search_results(
        server_results=server_results,
        client_results=client_results,
        query=expanded_query,
        top_k=effective_top_k + 1,  # Request one extra to detect truncation
    )

    # Check if truncated
    truncated = len(public_results) > effective_top_k
    if truncated:
        public_results = public_results[:effective_top_k]
        internal_results = internal_results[:effective_top_k]

    # Determine which tools to autoload (using internal results with server_name)
    autoload_server_refs: list[ToolReference] = []
    autoload_client_refs: list[ClientToolReference] = []
    autoloaded_tool_names: set[str] = set()

    for internal in internal_results[:autoload_top_k]:
        is_client = internal.get("is_client_tool", False)
        tool_name = internal.get("tool_name", "")
        srv_name = internal.get("server_name", "")

        if is_client:
            # Client tool
            autoload_client_refs.append(
                ClientToolReference(
                    tool_name=tool_name,
                    server_name=srv_name,
                    device_id=internal.get("device_id", device_id or ""),
                )
            )
            autoloaded_tool_names.add(tool_name)
        else:
            # Server tool - skip autoloading ambiguous tools unless server_name specified
            if catalog and catalog.is_ambiguous(tool_name) and not server_name:
                logger.debug(
                    "Skipping autoload for ambiguous tool '%s' (multiple servers)",
                    tool_name,
                )
                continue
            autoload_server_refs.append(ToolReference(tool_name=tool_name, server_name=srv_name))
            autoloaded_tool_names.add(tool_name)

    # Autoload tools into deferred state
    loaded_count = 0
    if conversation_id and (autoload_server_refs or autoload_client_refs):
        state = get_deferred_tool_state()

        # Autoload server tools
        if autoload_server_refs:
            loaded_refs = state.autoload(
                conversation_id=conversation_id,
                agent_key=agent_key,
                references=autoload_server_refs,
            )
            loaded_count += len(loaded_refs)

        # Autoload client tools
        if autoload_client_refs:
            loaded_client_refs = state.autoload_client_tools(
                conversation_id=conversation_id,
                agent_key=agent_key,
                references=autoload_client_refs,
                device_id=device_id,
            )
            loaded_count += len(loaded_client_refs)

    # Mark which tools in the public results are loaded
    for result in public_results:
        if result["tool_name"] in autoloaded_tool_names:
            result["is_loaded"] = True

    latency_ms = (time.time() - start_time) * 1000

    # Log search completion with metrics (internal logging only)
    logger.debug(
        "tool_search completed: latency=%.1fms results=%d (server=%d, client=%d) "
        "loaded=%d truncated=%s conversation=%s agent=%s",
        latency_ms,
        len(public_results),
        len(server_results),
        len(client_results),
        loaded_count,
        truncated,
        conversation_id,
        agent_key,
    )

    # Return clean results for model consumption
    # NO internal metadata like server_name, origin, generation, etc.
    return {
        "query": query,
        "results": public_results,
        "loaded_count": loaded_count,
        "more_available": truncated,
    }


def _merge_search_results(
    server_results: list,
    client_results: list,
    query: str | None,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Merge search results from server and client catalogs.

    Results are interleaved by relevance score (implicit from ordering)
    and then truncated to top_k.

    Args:
        server_results: Results from McpToolCatalog.search()
        client_results: Results from ClientToolCatalog.search()
        query: The original search query
        top_k: Maximum results to return

    Returns:
        Tuple of (public_results, internal_results):
        - public_results: Clean results for model consumption (no internal metadata)
        - internal_results: Full results with server_name etc. for autoloading
    """
    # Convert to internal result dicts with scoring
    merged: list[tuple[float, int, dict, dict]] = []  # (score, source, public, internal)

    # Process server results (already sorted by relevance)
    for idx, desc in enumerate(server_results):
        public_dict = desc.to_search_result()
        internal_dict = desc._to_internal_result()
        # Use position as implicit score (lower position = higher score)
        score = 1000 - idx
        merged.append((score, 0, public_dict, internal_dict))  # 0 = server (prefer on tie)

    # Process client results
    for idx, desc in enumerate(client_results):
        public_dict = desc.to_search_result()
        internal_dict = desc._to_internal_result()
        score = 1000 - idx
        merged.append((score, 1, public_dict, internal_dict))  # 1 = client

    # Sort by score descending, then by source (server first on tie)
    merged.sort(key=lambda x: (-x[0], x[1]))

    # Deduplicate by tool_name (prefer server if same name exists)
    seen_names: set[str] = set()
    public_results: list[dict] = []
    internal_results: list[dict] = []

    for _, _, public, internal in merged:
        name = public["tool_name"]
        if name not in seen_names:
            seen_names.add(name)
            public_results.append(public)
            internal_results.append(internal)
        if len(public_results) >= top_k:
            break

    return public_results, internal_results


@tool(args_schema=ToolSearchInput)
async def tool_search(
    query: str | None = None,
    top_k: int | None = None,
    server_name: str | None = None,
) -> str:
    """
    Search for available tools to accomplish a task.

    Use this tool to discover what tools are available before attempting to
    call them. This is most useful when the exact tool is unclear, when
    several tools might fit, or when you need a capability that is not
    already obvious from the bound tool list.

    When deferred MCP loading is enabled, do not guess tool names first.
    Use tool_search with a specific query to find and autoload the tool you need,
    then call the discovered tool by name.

    After searching:
    - Read each result's description and arg_hints before choosing a tool
    - Prefer task-based queries that include the action and target
    - Refine the query and search again if the results are weak or ambiguous
    - The top results are automatically loaded; any result with is_loaded=true
      is ready to call immediately

    Examples:
    - Search for web tools: tool_search(query="search the web")
    - Search for local file tools: tool_search(query="read local text file")
    - Search for computer interaction tools: tool_search(query="run shell command")
    - List all tools: tool_search()
    - Filter by server: tool_search(query="search", server_name="tavily")
    """
    result = await _execute_tool_search(
        query=query,
        top_k=top_k,
        server_name=server_name,
    )

    # Format as JSON string for tool output
    return json.dumps(result, indent=2)


def create_tool_search_tool(allowlist: list[str] | None = None):
    """
    Create a tool_search tool with a specific allowlist baked in.

    This is used to create per-agent versions of tool_search that
    respect the agent's tool allowlist.

    Args:
        allowlist: List of allowed tool names or server names

    Returns:
        A LangChain tool configured for tool search
    """

    @tool("tool_search", args_schema=ToolSearchInput)
    async def tool_search_impl(
        query: str | None = None,
        top_k: int | None = None,
        server_name: str | None = None,
    ) -> str:
        """
        Search for available tools to accomplish a task.

        Use this tool to discover what tools are available before attempting to
        call them. This is most useful when the exact tool is unclear, when
        several tools might fit, or when you need a capability that is not
        already obvious from the bound tool list.

        When deferred MCP loading is enabled, do not guess tool names first.
        Use tool_search with a specific query to find and autoload the tool you need,
        then call the discovered tool by name.

        After searching, inspect the descriptions and arg_hints, refine the
        query if needed, and call the best matching tool by name. Results with
        is_loaded=true are ready to use immediately.
        """
        result = await _execute_tool_search(
            query=query,
            top_k=top_k,
            server_name=server_name,
            allowlist=allowlist,
        )
        return json.dumps(result, indent=2)

    return tool_search_impl


# Export the default tool
__all__ = [
    "tool_search",
    "create_tool_search_tool",
    "ToolSearchInput",
    "ToolSearchOutput",
]
