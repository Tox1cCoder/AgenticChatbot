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


def _search_results_refer_to_same_capability(
    public_a: dict[str, Any],
    internal_a: dict[str, Any],
    public_b: dict[str, Any],
    internal_b: dict[str, Any],
) -> bool:
    """
    Return True when two search results represent the same underlying tool.

    Uses `call_name` (the invokable alias) for deduplication when available,
    falling back to `tool_name`. Ambiguous same-name tools from different
    servers get distinct call_names (e.g. tavily__search vs brave__search),
    so they are NOT collapsed.

    A client-side tool and a server-side tool may refer to the same underlying
    MCP capability even though the client version has a `client__...` prefix.
    In that case we detect duplicates via the shared qualified tool id.
    """
    # Use call_name for dedup — aliases differ for ambiguous same-name tools
    call_name_a = internal_a.get("call_name") or public_a.get("tool_name")
    call_name_b = internal_b.get("call_name") or public_b.get("tool_name")
    if call_name_a == call_name_b:
        return True

    # Cross-origin dedup: server tool vs client tool with same underlying capability
    if bool(internal_a.get("is_client_tool")) == bool(internal_b.get("is_client_tool")):
        return False

    qualified_a = str(internal_a.get("qualified_tool_id") or "").strip().lower()
    qualified_b = str(internal_b.get("qualified_tool_id") or "").strip().lower()
    return bool(qualified_a and qualified_b and qualified_a == qualified_b)


def _prefer_search_result_candidate(
    candidate_internal: dict[str, Any],
    existing_internal: dict[str, Any],
) -> bool:
    """
    Decide whether a duplicate candidate should replace the existing result.

    When the same underlying capability exists both on the connected client
    device and on the server, prefer the client-scoped variant so agents are
    nudged toward interacting with the user's actual device when available.
    """
    candidate_is_client = bool(candidate_internal.get("is_client_tool"))
    existing_is_client = bool(existing_internal.get("is_client_tool"))
    return candidate_is_client and not existing_is_client


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

    # Determine search mode based on query and server_name
    # - query present -> discovery mode
    # - query absent and no server_name -> global inventory mode (server summaries)
    # - query absent and server_name present -> per-server inventory mode
    is_inventory_mode = not query or not str(query).strip()
    is_per_server_inventory = is_inventory_mode and bool(server_name)

    # Apply mode-appropriate top_k defaults and limits
    if is_inventory_mode:
        default_top_k = settings.mcp_tool_search_inventory_default_top_k
        max_top_k = settings.mcp_tool_search_inventory_max_top_k
    else:
        default_top_k = settings.mcp_tool_search_default_top_k
        max_top_k = settings.mcp_tool_search_max_top_k
    autoload_top_k = settings.mcp_tool_search_autoload_top_k

    effective_top_k = top_k if top_k is not None else default_top_k
    effective_top_k = max(1, min(effective_top_k, max_top_k))
    # Get tool context for conversation-scoped loading
    ctx = get_tool_context()
    conversation_id = ctx.conversation_id
    agent_key = ctx.agent_key
    device_id = ctx.device_id
    user_id = ctx.user_id

    # Log query if enabled (gated on mcp_tool_search_log_queries)
    if settings.mcp_tool_search_log_queries:
        mode_label = (
            "inventory" if is_inventory_mode and not is_per_server_inventory
            else "per_server_inventory" if is_per_server_inventory
            else "discovery"
        )
        logger.info(
            "tool_search: mode=%s query=%r top_k=%d server=%s conversation=%s agent=%s device=%s",
            mode_label,
            query,
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

        # Global inventory mode: return server summaries without searching tools
        if is_inventory_mode and not is_per_server_inventory:
            inventory = catalog.get_server_inventory(allowlist=allowlist)
            latency_ms = (time.time() - start_time) * 1000
            if settings.mcp_tool_search_log_queries:
                logger.info(
                    "tool_search inventory mode: latency=%.1fms servers=%d",
                    latency_ms,
                    len(inventory),
                )
            return {
                "query": None,
                "mode": "inventory",
                "inventory": inventory,
                "results": [],
                "loaded_count": 0,
                "more_available": False,
            }

        # Search server tools (discovery or per-server inventory mode)
        # Use search_scored() when available so we have real scores for autoload gating
        if query and hasattr(catalog, "search_scored"):
            server_results = catalog.search_scored(
                query=query,
                top_k=effective_top_k * 2,
                server_name=server_name,
                allowlist=allowlist,
            )
        else:
            server_results = catalog.search(
                query=query,
                top_k=effective_top_k * 2,
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
                    query=query,
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
        query=query,
        top_k=effective_top_k + 1,  # Request one extra to detect truncation
    )

    # Check if truncated
    truncated = len(public_results) > effective_top_k
    if truncated:
        public_results = public_results[:effective_top_k]
        internal_results = internal_results[:effective_top_k]

    # Determine which tools to autoload (gated on autoload relevance threshold)
    autoload_min_score = settings.mcp_tool_search_autoload_min_relevance_score
    autoload_server_refs: list[ToolReference] = []
    autoload_client_refs: list[ClientToolReference] = []

    for internal in internal_results[:autoload_top_k]:
        is_client = internal.get("is_client_tool", False)
        tool_name = internal.get("tool_name", "")
        srv_name = internal.get("server_name", "")
        tool_score = float(internal.get("_score", 0.0))

        # Gate autoloading on the stricter autoload threshold
        if tool_score < autoload_min_score:
            if settings.mcp_tool_search_log_queries:
                logger.debug(
                    "tool_search: skipping autoload for '%s' (score=%.2f < threshold=%.2f)",
                    tool_name,
                    tool_score,
                    autoload_min_score,
                )
            continue

        if is_client:
            # Client tool
            autoload_client_refs.append(
                ClientToolReference(
                    tool_name=tool_name,
                    server_name=srv_name,
                    device_id=internal.get("device_id", device_id or ""),
                    session_id=internal.get("session_id", ""),
                    catalog_version=int(internal.get("catalog_version") or 0),
                    tool_instance_id=internal.get("tool_instance_id", ""),
                )
            )
        else:
            # Server tool - skip autoloading ambiguous tools without an alias
            # (they can still be autoloaded once they have a call_name alias)
            call_name = internal.get("call_name") or tool_name
            if catalog and catalog.is_ambiguous(tool_name) and call_name == tool_name and not server_name:
                logger.debug(
                    "Skipping autoload for ambiguous tool '%s' (multiple servers, no alias)",
                    tool_name,
                )
                continue
            autoload_server_refs.append(
                ToolReference(tool_name=tool_name, server_name=srv_name, call_name=call_name)
            )

    # Autoload tools into deferred state
    # is_loaded is derived from ACTUAL successful loads, not preselected candidates
    actually_loaded_names: set[str] = set()
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
            for ref in loaded_refs:
                # Track by call_name (the invokable alias, or tool_name if unambiguous)
                actually_loaded_names.add(getattr(ref, "call_name", None) or ref.tool_name)

        # Autoload client tools
        if autoload_client_refs:
            loaded_client_refs = state.autoload_client_tools(
                conversation_id=conversation_id,
                agent_key=agent_key,
                references=autoload_client_refs,
                device_id=device_id,
                session_id=client_catalog.session_id if client_catalog is not None else None,
                user_id=user_id,
            )
            loaded_count += len(loaded_client_refs)
            for ref in loaded_client_refs:
                actually_loaded_names.add(ref.tool_name)

    # Mark which tools in the public results are loaded (truthful: from actual loads)
    # Public tool_name is the call_name (alias for ambiguous tools, raw name otherwise)
    for result in public_results:
        if result["tool_name"] in actually_loaded_names:
            result["is_loaded"] = True
        elif result.get("call_name") in actually_loaded_names:
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
    mode = "per_server_inventory" if is_per_server_inventory else "discovery"
    return {
        "query": query,
        "mode": mode,
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

    Accepts either plain descriptor lists (from search()) or scored tuples
    (descriptor, score) from search_scored(). Results are interleaved by
    relevance score, duplicate capabilities are collapsed, and the final list
    is truncated to top_k.

    Args:
        server_results: Results from McpToolCatalog.search() or search_scored()
        client_results: Results from ClientToolCatalog.search() or search_scored()
        query: The original search query
        top_k: Maximum results to return

    Returns:
        Tuple of (public_results, internal_results):
        - public_results: Clean results for model consumption (no internal metadata)
        - internal_results: Full results with server_name, score, etc. for autoloading
    """
    # Convert to internal result dicts with scoring
    merged: list[tuple[float, int, dict, dict]] = []  # (score, source, public, internal)

    # Process server results (already sorted by relevance)
    for idx, item in enumerate(server_results):
        # Support both (descriptor, score) tuples and plain descriptors
        if isinstance(item, tuple):
            desc, real_score = item
        else:
            desc = item
            real_score = float(1000 - idx)
        public_dict = desc.to_search_result()
        internal_dict = desc._to_internal_result()
        internal_dict["_score"] = real_score
        merged.append((real_score, 0, public_dict, internal_dict))  # 0 = server (prefer on tie)

    # Process client results
    for idx, item in enumerate(client_results):
        if isinstance(item, tuple):
            desc, real_score = item
        else:
            desc = item
            real_score = float(1000 - idx)
        public_dict = desc.to_search_result()
        internal_dict = desc._to_internal_result()
        internal_dict["_score"] = real_score
        merged.append((real_score, 1, public_dict, internal_dict))  # 1 = client

    # Sort by score descending, then by source (server first on tie)
    merged.sort(key=lambda x: (-x[0], x[1]))

    chosen_results: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for _, _, public, internal in merged:
        duplicate_index = None
        for idx, (existing_public, existing_internal) in enumerate(chosen_results):
            if _search_results_refer_to_same_capability(
                public,
                internal,
                existing_public,
                existing_internal,
            ):
                duplicate_index = idx
                break

        if duplicate_index is None:
            chosen_results.append((public, internal))
            continue

        existing_public, existing_internal = chosen_results[duplicate_index]
        if _prefer_search_result_candidate(internal, existing_internal):
            chosen_results[duplicate_index] = (public, internal)

    chosen_results = chosen_results[:top_k]
    public_results = [public for public, _ in chosen_results]
    internal_results = [internal for _, internal in chosen_results]
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
