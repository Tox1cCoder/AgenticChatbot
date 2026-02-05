"""
Tool Search Tool - Claude-style deferred MCP tool discovery.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..core.config import settings
from .mcp_integration import get_global_mcp_manager
from .mcp_registry import get_mcp_tools_generation
from .mcp_tool_catalog import (
    ToolReference,
    get_tool_catalog,
)
from .deferred_tool_state import get_deferred_tool_state
from .tool_context import get_tool_context

logger = logging.getLogger(__name__)


class ToolSearchInput(BaseModel):
    """Input schema for the tool_search tool."""

    query: Optional[str] = Field(
        default=None,
        description=(
            "Natural language description of what you need the tool to do. "
            "Leave empty to list all available tools."
        ),
    )
    top_k: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of tools to return. Defaults to 5. "
            "Use a larger value to see more options."
        ),
    )
    server_name: Optional[str] = Field(
        default=None,
        description=(
            "Filter results to a specific MCP server. "
            "Use this to disambiguate when multiple servers provide tools with the same name."
        ),
    )


class ToolSearchResult(BaseModel):
    """Individual tool result from search."""

    tool_name: str = Field(description="The name to use when calling this tool")
    server_name: str = Field(description="The MCP server providing this tool")
    display_name: str = Field(description="Human-readable name with server prefix")
    description: str = Field(description="What the tool does")
    arg_hints: str = Field(description="Summary of arguments (* = required)")
    call_as: str = Field(description="The exact name to use in a tool call")


class ToolSearchOutput(BaseModel):
    """Output schema for the tool_search tool."""

    query: Optional[str] = Field(description="The search query used")
    top_k: int = Field(description="The top_k value used")
    server_filter: Optional[str] = Field(description="Server filter applied, if any")
    results: List[ToolSearchResult] = Field(description="List of matching tools")
    autoloaded: List[Dict[str, str]] = Field(
        description="Tools that were automatically loaded for immediate use"
    )
    generation: int = Field(description="MCP tools generation (for cache tracking)")
    latency_ms: float = Field(description="Search latency in milliseconds")
    truncated: bool = Field(description="True if more results exist beyond top_k")
    total_available: int = Field(
        description="Total tools available (optionally filtered by server)"
    )
    unavailable_servers: List[str] = Field(
        default_factory=list, description="Servers that could not be queried (errors)"
    )


async def _execute_tool_search(
    query: Optional[str] = None,
    top_k: Optional[int] = None,
    server_name: Optional[str] = None,
    allowlist: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Core implementation of tool search logic.

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

    # Get tool context for conversation-scoped loading
    ctx = get_tool_context()
    conversation_id = ctx.conversation_id
    agent_key = ctx.agent_key

    # Log query if enabled
    if settings.mcp_tool_search_log_queries:
        logger.info(
            "tool_search: query=%r top_k=%d server=%s conversation=%s agent=%s",
            query,
            effective_top_k,
            server_name,
            conversation_id,
            agent_key,
        )

    # Get the MCP manager and catalog
    unavailable_servers: List[str] = []
    try:
        mcp_manager = await get_global_mcp_manager()
        catalog = await get_tool_catalog(mcp_manager)
    except Exception as e:
        logger.error("Failed to get tool catalog: %s", e)
        return {
            "query": query,
            "top_k": effective_top_k,
            "server_filter": server_name,
            "results": [],
            "autoloaded": [],
            "generation": get_mcp_tools_generation(),
            "latency_ms": (time.time() - start_time) * 1000,
            "truncated": False,
            "total_available": 0,
            "unavailable_servers": ["all"],
            "error": str(e),
        }

    # Search the catalog
    results = catalog.search(
        query=query,
        top_k=effective_top_k + 1,  # Request one extra to detect truncation
        server_name=server_name,
        allowlist=allowlist,
    )

    # Check if truncated
    truncated = len(results) > effective_top_k
    if truncated:
        results = results[:effective_top_k]

    # Get total count for "list all" queries
    total_available = len(catalog.list_all(allowlist=allowlist))
    if server_name:
        total_available = len(
            [
                t
                for t in catalog.list_all(allowlist=allowlist)
                if t.server_name == server_name
            ]
        )

    # Determine which tools to autoload
    autoload_candidates: List[ToolReference] = []
    for desc in results[:autoload_top_k]:
        # Skip autoloading ambiguous tools unless server_name is specified
        if catalog.is_ambiguous(desc.tool_name) and not server_name:
            logger.debug(
                "Skipping autoload for ambiguous tool '%s' (multiple servers)",
                desc.tool_name,
            )
            continue
        autoload_candidates.append(
            ToolReference(tool_name=desc.tool_name, server_name=desc.server_name)
        )

    # Autoload tools into deferred state
    autoloaded: List[Dict[str, str]] = []
    if autoload_candidates and conversation_id:
        state = get_deferred_tool_state()
        loaded_refs = state.autoload(
            conversation_id=conversation_id,
            agent_key=agent_key,
            references=autoload_candidates,
        )
        autoloaded = [
            {"tool_name": ref.tool_name, "server_name": ref.server_name}
            for ref in loaded_refs
        ]

    # Format results
    formatted_results = [desc.to_search_result() for desc in results]

    latency_ms = (time.time() - start_time) * 1000

    # Log search completion with metrics
    logger.info(
        "tool_search completed: latency=%.1fms results=%d autoloaded=%d "
        "truncated=%s conversation=%s agent=%s",
        latency_ms,
        len(results),
        len(autoloaded),
        truncated,
        conversation_id,
        agent_key,
    )

    return {
        "query": query,
        "top_k": effective_top_k,
        "server_filter": server_name,
        "results": formatted_results,
        "autoloaded": autoloaded,
        "generation": get_mcp_tools_generation(),
        "latency_ms": round(latency_ms, 2),
        "truncated": truncated,
        "total_available": total_available,
        "unavailable_servers": unavailable_servers,
    }


@tool(args_schema=ToolSearchInput)
async def tool_search(
    query: Optional[str] = None,
    top_k: Optional[int] = None,
    server_name: Optional[str] = None,
) -> str:
    """
    Search for available tools to accomplish a task.

    Use this tool to discover what tools are available before attempting to
    call them. The search will return a list of matching tools with their
    names, descriptions, and argument hints.

    After searching, you can call the discovered tools directly by name.
    The top results are automatically loaded and ready to use.

    Examples:
    - Search for web tools: tool_search(query="search the web")
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


def create_tool_search_tool(allowlist: Optional[List[str]] = None):
    """
    Create a tool_search tool with a specific allowlist baked in.

    This is used to create per-agent versions of tool_search that
    respect the agent's tool allowlist.

    Args:
        allowlist: List of allowed tool names or server names

    Returns:
        A LangChain tool configured for tool search
    """

    @tool(args_schema=ToolSearchInput, name="tool_search")
    async def tool_search_impl(
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        server_name: Optional[str] = None,
    ) -> str:
        """
        Search for available tools to accomplish a task.

        Use this tool to discover what tools are available before attempting to
        call them. The search will return a list of matching tools with their
        names, descriptions, and argument hints.

        After searching, you can call the discovered tools directly by name.
        The top results are automatically loaded and ready to use.
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
