"""
Tool Search Tool - Claude-style deferred MCP tool discovery.

This module provides unified tool search across BOTH server-side MCP tools
AND client device tools. The search behavior is consistent regardless of
tool origin - the only difference is the tool list available on each device.

Key features:
- Searches server MCP tools (from McpToolCatalog)
- Searches client device tools (from ClientToolCatalog) when a device is connected
- Results include origin information (server_mcp, client_mcp)
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
from .tool_scope import is_client_only_scope
from .tool_search_scoring import build_query_tokens

logger = logging.getLogger(__name__)
_ALLOWLIST_UNSET = object()


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
            "Leave empty to list all available servers, or combine an empty query "
            "with server_name to browse one server's tools."
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
            "Hard-scope results to a specific MCP server. "
            "Use this when the user explicitly names an integration and you want "
            "to inspect or search only that server's tools. Use the exact "
            "server_name returned by inventory results; do not guess variants."
        ),
    )


class ToolSearchResult(BaseModel):
    """Individual tool result from search."""

    tool_name: str = Field(description="The exact name to use when calling this tool")
    description: str = Field(description="Compact purpose of the tool")
    arg_hints: str = Field(description="Summary of arguments (* = required)")
    is_loaded: bool = Field(
        default=False, description="True if this tool was autoloaded and is ready for immediate use"
    )
    confidence: str = Field(
        default="low", description="Match confidence band: high, medium, or low"
    )
    match_reasons: list[str] = Field(
        default_factory=list, description="At most two short reasons this tool matched"
    )


class RecommendedTool(BaseModel):
    """Response-level recommendation pointing at the single best loaded tool."""

    tool_name: str = Field(description="The exact name to call")
    confidence: str = Field(description="Match confidence band: high, medium, or low")
    is_loaded: bool = Field(description="True when the recommended tool is callable this turn")


class ToolSearchOutput(BaseModel):
    """Output schema for the tool_search tool."""

    query: str | None = Field(description="The search query used")
    mode: str = Field(description="Search mode: discovery, inventory, or per_server_inventory")
    resolved_server_name: str | None = Field(
        default=None,
        description=(
            "Canonical server name chosen by tool_search when it can resolve a "
            "named integration or fuzzy server identifier."
        ),
    )
    inventory: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Server inventory summaries returned in inventory mode. Each item includes "
            "server_name, description, and tool_count."
        ),
    )
    recommended_tool: RecommendedTool | None = Field(
        default=None,
        description=(
            "The single high-confidence loaded tool to call next, or null when the "
            "search should be refined."
        ),
    )
    results: list[ToolSearchResult] = Field(description="List of matching tools")
    requires_refinement: bool = Field(
        description="True when no loaded high-confidence recommendation is available"
    )
    next_action: str = Field(
        description="call_recommended_tool, refine_search, or inspect_inventory"
    )
    loaded_count: int = Field(
        description="Number of tools that were autoloaded and ready for immediate use"
    )
    more_available: bool = Field(description="True if more results exist beyond the returned list")


async def _execute_tool_search(
    query: str | None = None,
    top_k: int | None = None,
    server_name: str | None = None,
    allowlist: list[str] | None = None,
    server_allowlist: list[str] | None | object = _ALLOWLIST_UNSET,
    client_allowlist: list[str] | None | object = _ALLOWLIST_UNSET,
) -> dict[str, Any]:
    """
    Core implementation of tool search logic.

    Searches BOTH server MCP tools AND client device tools (if a device is connected).
    Results are merged and ranked by relevance.

    Args:
        query: Search query (None for list all)
        top_k: Max results to return
        server_name: Optional server filter
        allowlist: Optional per-agent allowlist filter applied to both origins
        server_allowlist: Optional server-only allowlist; ``None`` allows all
        client_allowlist: Optional client-only allowlist; ``None`` allows all

    Returns:
        Dict with search results and metadata
    """
    start_time = time.time()
    effective_server_allowlist = (
        allowlist if server_allowlist is _ALLOWLIST_UNSET else server_allowlist
    )
    effective_client_allowlist = (
        allowlist if client_allowlist is _ALLOWLIST_UNSET else client_allowlist
    )
    # Catalogs treat an empty allowlist as "no filter". For explicit per-origin
    # scoping an empty list must mean "no tools from this origin".
    if server_allowlist is not _ALLOWLIST_UNSET and effective_server_allowlist == []:
        effective_server_allowlist = ["__custom_agent_no_server_tools__"]
    if client_allowlist is not _ALLOWLIST_UNSET and effective_client_allowlist == []:
        effective_client_allowlist = ["__custom_agent_no_client_tools__"]
    # Get tool context for conversation-scoped loading
    ctx = get_tool_context()
    conversation_id = ctx.conversation_id
    agent_key = ctx.agent_key
    device_id = ctx.device_id
    user_id = ctx.user_id
    client_only_scope = is_client_only_scope(
        device_id=device_id,
        tool_scope=getattr(ctx, "tool_scope", None),
    )
    resolved_server_name: str | None = None

    # Get the MCP manager and catalog for SERVER tools
    unavailable_servers: list[str] = []
    server_results = []
    catalog = None
    is_inventory_mode = not query or not str(query).strip()
    is_per_server_inventory = is_inventory_mode and bool(server_name)
    if is_inventory_mode:
        default_top_k = settings.mcp_tool_search_inventory_default_top_k
        max_top_k = settings.mcp_tool_search_inventory_max_top_k
    else:
        default_top_k = settings.mcp_tool_search_default_top_k
        max_top_k = settings.mcp_tool_search_max_top_k
    autoload_top_k = settings.mcp_tool_search_autoload_top_k
    effective_top_k = top_k if top_k is not None else default_top_k
    effective_top_k = max(1, min(effective_top_k, max_top_k))

    if not client_only_scope:
        try:
            mcp_manager = await get_global_mcp_manager()
            catalog = await get_tool_catalog(mcp_manager)

            # Resolve server identifiers before deciding between discovery and
            # inventory modes so broad integration queries like "Canva" or
            # plausible variants like "excel-server" can use the correct server
            # scope without hard-coded aliases.
            if server_name and hasattr(catalog, "resolve_server_name"):
                resolved_server_name = catalog.resolve_server_name(server_name)
                if resolved_server_name:
                    server_name = resolved_server_name
            elif query and hasattr(catalog, "resolve_server_name"):
                resolved_server_name = catalog.resolve_server_name(query)
                if resolved_server_name:
                    query_tokens = build_query_tokens(query)[1]
                    server_name = resolved_server_name
                    if len(query_tokens) <= 1:
                        query = None
                        is_inventory_mode = True
                        is_per_server_inventory = True
                        default_top_k = settings.mcp_tool_search_inventory_default_top_k
                        max_top_k = settings.mcp_tool_search_inventory_max_top_k
                        effective_top_k = top_k if top_k is not None else default_top_k
                        effective_top_k = max(1, min(effective_top_k, max_top_k))

            # Log query if enabled (gated on mcp_tool_search_log_queries)
            if settings.mcp_tool_search_log_queries:
                mode_label = (
                    "inventory"
                    if is_inventory_mode and not is_per_server_inventory
                    else "per_server_inventory"
                    if is_per_server_inventory
                    else "discovery"
                )
                logger.info(
                    "tool_search: mode=%s query=%r top_k=%d server=%s "
                    "resolved_server=%s conversation=%s agent=%s device=%s",
                    mode_label,
                    query,
                    effective_top_k,
                    server_name,
                    resolved_server_name,
                    conversation_id,
                    agent_key,
                    device_id[:8] if device_id else None,
                )

            # Global inventory mode: return server summaries without searching tools
            if is_inventory_mode and not is_per_server_inventory:
                inventory = catalog.get_server_inventory(allowlist=effective_server_allowlist)
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
                    "resolved_server_name": resolved_server_name,
                    "inventory": inventory,
                    "recommended_tool": None,
                    "results": [],
                    "requires_refinement": False,
                    "next_action": "inspect_inventory",
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
                    allowlist=effective_server_allowlist,
                )
            else:
                server_results = catalog.search(
                    query=query,
                    top_k=effective_top_k * 2,
                    server_name=server_name,
                    allowlist=effective_server_allowlist,
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
            if client_only_scope and server_name and hasattr(client_catalog, "resolve_server_name"):
                resolved_client_server_name = client_catalog.resolve_server_name(server_name)
                if resolved_client_server_name:
                    server_name = resolved_client_server_name
            if client_catalog.tool_count > 0:
                if client_only_scope and is_inventory_mode and not is_per_server_inventory:
                    inventory = client_catalog.get_server_inventory(
                        allowlist=effective_client_allowlist
                    )
                    return {
                        "query": None,
                        "mode": "inventory",
                        "resolved_server_name": resolved_server_name,
                        "inventory": inventory,
                        "recommended_tool": None,
                        "results": [],
                        "requires_refinement": False,
                        "next_action": "inspect_inventory",
                        "loaded_count": 0,
                        "more_available": False,
                    }
                if query and hasattr(client_catalog, "search_scored"):
                    client_results = client_catalog.search_scored(
                        query=query,
                        top_k=effective_top_k * 2,  # Request extra for merging
                        server_name=server_name,
                        allowlist=effective_client_allowlist,
                    )
                else:
                    client_results = client_catalog.search(
                        query=query,
                        top_k=effective_top_k * 2,  # Request extra for merging
                        server_name=server_name,
                        allowlist=effective_client_allowlist,
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
        server_results=[] if client_only_scope else server_results,
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

        # Gate autoloading on the stricter autoload threshold (secondary safety
        # floor for backward compatibility with unscored/legacy results).
        if tool_score < autoload_min_score:
            if settings.mcp_tool_search_log_queries:
                logger.debug(
                    "tool_search: skipping autoload for '%s' (score=%.2f < threshold=%.2f)",
                    tool_name,
                    tool_score,
                    autoload_min_score,
                )
            continue

        # Scored results carry explicit eligibility: only the single
        # high-confidence recommended candidate is eligible, so we no longer
        # broadly autoload the first N candidates. Legacy/plain results have no
        # eligibility flag (None) and fall back to the score floor above.
        if internal.get("_autoload_eligible") is False:
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
            if (
                catalog
                and catalog.is_ambiguous(tool_name)
                and call_name == tool_name
                and not server_name
            ):
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
        if (
            result["tool_name"] in actually_loaded_names
            or result.get("call_name") in actually_loaded_names
        ):
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
    recommended_tool, requires_refinement, next_action = _build_recommendation(
        public_results,
        internal_results,
    )
    result = {
        "query": query,
        "mode": mode,
        "recommended_tool": recommended_tool,
        "results": public_results,
        "requires_refinement": requires_refinement,
        "next_action": next_action,
        "loaded_count": loaded_count,
        "more_available": truncated,
    }
    if resolved_server_name:
        result["resolved_server_name"] = resolved_server_name
    if settings.mcp_tool_search_debug_scores:
        result["debug_scores"] = [
            {
                "tool_name": public.get("tool_name"),
                "score": internal.get("_score"),
                "confidence": internal.get("_confidence"),
                "reasons": internal.get("_match_reasons", []),
            }
            for public, internal in zip(public_results, internal_results, strict=False)
        ]
    return result


def _public_result_from_scored(item: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build (public, internal) dicts from a ToolSearchScore.

    The public dict uses the compact capability purpose (not a truncated raw MCP
    prompt) and carries confidence + match reasons. The internal dict carries the
    numeric score and autoload eligibility for the autoload pass and debug output.
    """
    score_meta = item if hasattr(item, "tool") else None
    tool = score_meta.tool if score_meta is not None else item

    public = tool.to_search_result()
    if score_meta is not None:
        public["description"] = score_meta.profile.purpose[
            : settings.mcp_tool_search_description_max_chars
        ]
        public["confidence"] = score_meta.confidence
        public["match_reasons"] = score_meta.match_reasons[
            : settings.mcp_tool_search_match_reasons_max
        ]

    internal = tool._to_internal_result()
    if score_meta is not None:
        internal["_score"] = score_meta.score
        internal["_confidence"] = score_meta.confidence
        internal["_autoload_eligible"] = score_meta.autoload_eligible
        internal["_match_reasons"] = list(score_meta.match_reasons)
        if settings.mcp_tool_search_debug_scores:
            internal["_debug_score"] = {
                "score": score_meta.score,
                "capabilities": sorted(score_meta.profile.capabilities),
            }
    return public, internal


def _search_item_to_dicts(item: Any, idx: int) -> tuple[float, dict[str, Any], dict[str, Any]]:
    """Normalize a merge input into (score, public_dict, internal_dict).

    Accepts scored objects (``.tool``), legacy ``(descriptor, score)`` tuples, and
    plain descriptors. Tuple/plain paths are temporary compatibility for older
    fakes and external callers during the scored-contract migration.
    """
    if hasattr(item, "tool"):
        public_dict, internal_dict = _public_result_from_scored(item)
        real_score = float(item.score)
    elif isinstance(item, tuple) and len(item) == 2:
        desc, real_score = item
        real_score = float(real_score)
        public_dict = desc.to_search_result()
        internal_dict = desc._to_internal_result()
    else:
        desc = item
        real_score = float(1000 - idx)
        public_dict = desc.to_search_result()
        internal_dict = desc._to_internal_result()
    internal_dict["_score"] = real_score
    return real_score, public_dict, internal_dict


def _build_recommendation(
    public_results: list[dict[str, Any]],
    internal_results: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, bool, str]:
    """Decide the single recommended tool and the response-level next action.

    A recommendation is only emitted when the top public result is loaded
    (callable this turn) and high confidence, or medium confidence as the lone
    result. Otherwise the model is told to refine the search.
    """
    if not public_results or not internal_results:
        return None, True, "refine_search"

    top_public = public_results[0]
    top_internal = internal_results[0]
    confidence = str(top_public.get("confidence") or top_internal.get("_confidence") or "low")
    is_loaded = bool(top_public.get("is_loaded"))

    if confidence == "high" and is_loaded:
        return (
            {
                "tool_name": top_public["tool_name"],
                "confidence": confidence,
                "is_loaded": is_loaded,
            },
            False,
            "call_recommended_tool",
        )
    if confidence == "medium" and is_loaded and len(public_results) == 1:
        return (
            {
                "tool_name": top_public["tool_name"],
                "confidence": confidence,
                "is_loaded": is_loaded,
            },
            False,
            "call_recommended_tool",
        )
    return None, True, "refine_search"


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
        real_score, public_dict, internal_dict = _search_item_to_dicts(item, idx)
        merged.append((real_score, 0, public_dict, internal_dict))  # 0 = server (prefer on tie)

    # Process client results
    for idx, item in enumerate(client_results):
        real_score, public_dict, internal_dict = _search_item_to_dicts(item, idx)
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
    If a suitable specialized tool is already bound, use it directly.
    Use tool_search when the needed capability is missing, ambiguous,
    or not currently bound.

    For named integrations:
    - If you do not yet know the exact server identifier, call tool_search()
      first and inspect the enabled servers and their descriptions
    - Then call tool_search(server_name="...") to browse that server's tools
    - Use the exact server_name returned by tool_search(); do not invent or
      modify server identifiers
    - For a specific task inside that integration, use
      tool_search(query="...", server_name="...")
    - If the scoped search returns no suitable tool, retry with an unscoped
      capability query rather than guessing

    After searching:
    - Read each result's description and arg_hints before choosing a tool
    - If resolved_server_name is present, reuse that exact server_name in later calls
    - Prefer task-based queries that include the action and target
    - Refine the query and search again if the results are weak or ambiguous
    - High-confidence eligible results are automatically loaded; any result
      with is_loaded=true is ready to call immediately

    Examples:
    - Search for web tools: tool_search(query="search the web")
    - Search for local file tools: tool_search(query="read local text file")
    - Search for computer interaction tools: tool_search(query="run shell command")
    - List all servers: tool_search()
    - Browse one integration's tools: tool_search(server_name="target_server")
    - Search within one integration: tool_search(query="specific task", server_name="target_server")
    """
    result = await _execute_tool_search(
        query=query,
        top_k=top_k,
        server_name=server_name,
    )

    # Format as JSON string for tool output
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def create_tool_search_tool(
    allowlist: list[str] | None = None,
    *,
    server_allowlist: list[str] | None | object = _ALLOWLIST_UNSET,
    client_allowlist: list[str] | None | object = _ALLOWLIST_UNSET,
):
    """
    Create a tool_search tool with a specific allowlist baked in.

    This is used to create per-agent versions of tool_search that
    respect the agent's tool allowlist.

    Args:
        allowlist: List of allowed tool names or server names
        server_allowlist: Optional server-only allowlist; ``None`` allows all
        client_allowlist: Optional client-only allowlist; ``None`` allows all

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
        If a suitable specialized tool is already bound, use it directly.
        Use tool_search when the needed capability is missing, ambiguous,
        or not currently bound.

        For named integrations, identify the exact server first with
        tool_search(), then inspect that server with tool_search(server_name="...")
        before narrowing to tool_search(query="...", server_name="...").
        Use the exact server_name returned by tool_search(); do not invent
        variants.
        If the scoped search has no suitable tool, retry with an unscoped query.

        After searching, inspect the descriptions and arg_hints, refine the
        query if needed, and call the best matching tool by name. If
        resolved_server_name is present, reuse that exact server_name in later
        calls. High-confidence eligible results with is_loaded=true are ready
        to use immediately.
        """
        result = await _execute_tool_search(
            query=query,
            top_k=top_k,
            server_name=server_name,
            allowlist=allowlist,
            server_allowlist=server_allowlist,
            client_allowlist=client_allowlist,
        )
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    return tool_search_impl


def create_tool_search_tool_for_custom_agent(spec: Any):
    """Restricted tool_search for a custom agent.

    Discovery includes the backend MCP server catalog and only the exact client
    tool instances selected by the agent, so sidecar-local tools from another
    session cannot be surfaced as substitutes.
    """
    return create_tool_search_tool(
        server_allowlist=spec.server_tool_search_allowlist(),
        client_allowlist=spec.client_tool_search_allowlist(),
    )


# Export the default tool
__all__ = [
    "tool_search",
    "create_tool_search_tool",
    "create_tool_search_tool_for_custom_agent",
    "ToolSearchInput",
    "ToolSearchOutput",
]
