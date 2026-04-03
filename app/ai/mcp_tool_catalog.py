"""
MCP Tool Catalog - Tool discovery and search for deferred loading.

This module provides a searchable catalog of SERVER-SIDE MCP tools that supports:
- Caching keyed by MCP tools generation (invalidated on config changes)
- Lightweight search ranking (name match + description token overlap)
- Per-agent allowlist filtering
- Tool name collision detection across servers

Client device tools are indexed separately in client_tool_catalog.py and are
merged with server tool results in tool_search_tool.py for unified search.

The tool_search system provides consistent deferred loading for BOTH server
and client tools - the only difference is the tool list available.
"""

import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .mcp_registry import get_mcp_tools_generation
from .text_normalization import tokenize_text

logger = logging.getLogger(__name__)

# Import client tool prefix for reference
CLIENT_TOOL_PREFIX = "client__"

# Tool origin constants (duplicated here to avoid circular imports)
TOOL_ORIGIN_SERVER_MCP = "server_mcp"


@dataclass
class ToolDescriptor:
    """
    Describes a single MCP tool with metadata for search and loading.

    This class is used for SERVER-SIDE MCP tools. Client device tools
    use ClientToolDescriptor from client_tool_catalog.py.
    """

    tool_name: str
    server_name: str
    description: str
    arg_names: list[str]
    required_arg_names: list[str]
    schema_fingerprint: str
    args_schema: dict[str, Any] = field(default_factory=dict)
    origin: str = field(default=TOOL_ORIGIN_SERVER_MCP)  # Always server_mcp for this catalog

    @property
    def arg_hints(self) -> str:
        """Compact string summarizing tool arguments."""
        if not self.arg_names:
            return "(no arguments)"

        hints = []
        for arg in self.arg_names:
            if arg in self.required_arg_names:
                hints.append(f"{arg}*")
            else:
                hints.append(arg)
        return f"({', '.join(hints)})"

    def to_search_result(self) -> dict[str, Any]:
        """
        Convert to a search result dict for tool_search output.

        IMPORTANT: Only expose information the model needs to USE the tool.
        Do NOT expose internal details like server_name, origin, etc.
        that could lead to the model guessing tool names.
        """
        return {
            "tool_name": self.tool_name,
            "description": self.description[:200] if self.description else "",
            "arg_hints": self.arg_hints,
            "is_loaded": False,  # Will be updated by tool_search
        }

    def _to_internal_result(self) -> dict[str, Any]:
        """
        Internal result with full metadata for autoloading logic.
        NOT exposed to the model.
        """
        return {
            "tool_name": self.tool_name,
            "server_name": self.server_name,
            "qualified_tool_id": f"{self.server_name}::{self.tool_name}",
            "description": self.description[:200] if self.description else "",
            "arg_hints": self.arg_hints,
            "origin": self.origin,
            "is_client_tool": False,
        }

    def is_server_tool(self) -> bool:
        """Check if this is a server-side tool (always True for ToolDescriptor)."""
        return True


@dataclass
class ToolReference:
    """Minimal reference to a tool for loading purposes."""

    tool_name: str
    server_name: str


def compute_schema_fingerprint(args_schema: dict[str, Any], description: str) -> str:
    """
    Compute a stable hash of a tool's schema for cache invalidation.

    The fingerprint changes when:
    - The args_schema structure changes
    - The description changes
    """
    # Normalize the schema to a canonical JSON string
    schema_str = json.dumps(args_schema, sort_keys=True, default=str)
    combined = f"{description}|{schema_str}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def extract_arg_info(args_schema: dict[str, Any]) -> tuple[list[str], list[str]]:
    """
    Extract argument names and required argument names from a JSON schema.

    Returns:
        Tuple of (all_arg_names, required_arg_names)
    """
    properties = args_schema.get("properties", {})
    required = set(args_schema.get("required", []))

    all_args = list(properties.keys())
    required_args = [arg for arg in all_args if arg in required]

    return all_args, required_args


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for search matching.

    Splits on non-alphanumeric characters and lowercases.
    """
    return tokenize_text(text)


class McpToolCatalog:
    """
    Searchable catalog of MCP tools with generation-based caching.

    The catalog automatically refreshes when the MCP tools generation changes
    (e.g., when servers are enabled/disabled or config is modified).
    """

    def __init__(self, mcp_manager: Any):
        """
        Initialize the catalog.

        Args:
            mcp_manager: The MCPManager instance to pull tools from
        """
        self._mcp_manager = mcp_manager
        self._cached_generation: int = -1
        self._tools: list[ToolDescriptor] = []
        self._tools_by_name: dict[str, list[ToolDescriptor]] = {}
        self._tools_by_server: dict[str, list[ToolDescriptor]] = {}
        self._colliding_names: set[str] = set()

        # Inverted index for search: token -> set of tool indices
        self._token_index: dict[str, set[int]] = {}

        # Document frequency for BM25-style scoring
        self._doc_freq: Counter = Counter()
        self._total_docs: int = 0

    async def refresh_if_needed(self) -> bool:
        """
        Refresh the catalog if the MCP tools generation has changed.

        Returns:
            True if the catalog was refreshed, False if cache was valid
        """
        current_gen = get_mcp_tools_generation()
        if current_gen == self._cached_generation:
            return False

        await self._rebuild_catalog()
        self._cached_generation = current_gen
        return True

    async def _rebuild_catalog(self) -> None:
        """
        Rebuild the entire catalog from the MCP manager.

        IMPORTANT: This only indexes SERVER-SIDE MCP tools. Any tools with names
        starting with CLIENT_TOOL_PREFIX are explicitly filtered out to maintain
        clean separation between server and client tool namespaces.
        """
        start_time = time.time()

        # Clear existing data
        self._tools.clear()
        self._tools_by_name.clear()
        self._tools_by_server.clear()
        self._colliding_names.clear()
        self._token_index.clear()
        self._doc_freq.clear()

        # Fetch all tools from MCP manager
        try:
            tools_info = await self._mcp_manager.get_all_tools_info()
        except Exception as e:
            logger.error("Failed to fetch tools from MCP manager: %s", e)
            tools_info = []

        # Track filtered tools for logging
        filtered_count = 0

        # Build descriptors (only for server-side tools)
        for tool_info in tools_info:
            tool_name = tool_info.get("name", "")
            server_name = tool_info.get("server_name", "unknown")
            description = tool_info.get("description", "")
            args_schema = tool_info.get("args_schema", {}) or {}

            # CRITICAL: Filter out any client tools to maintain clean separation
            # Client tools should never appear in MCP manager, but this is defense-in-depth
            if tool_name.startswith(CLIENT_TOOL_PREFIX):
                logger.warning(
                    "Filtering client tool '%s' from server MCP catalog - "
                    "client tools should not be in MCP manager",
                    tool_name,
                )
                filtered_count += 1
                continue

            arg_names, required_args = extract_arg_info(args_schema)
            fingerprint = compute_schema_fingerprint(args_schema, description)

            descriptor = ToolDescriptor(
                tool_name=tool_name,
                server_name=server_name,
                description=description,
                arg_names=arg_names,
                required_arg_names=required_args,
                schema_fingerprint=fingerprint,
                args_schema=args_schema,
                origin=TOOL_ORIGIN_SERVER_MCP,
            )

            idx = len(self._tools)
            self._tools.append(descriptor)

            # Index by name
            if tool_name not in self._tools_by_name:
                self._tools_by_name[tool_name] = []
            self._tools_by_name[tool_name].append(descriptor)

            # Index by server
            if server_name not in self._tools_by_server:
                self._tools_by_server[server_name] = []
            self._tools_by_server[server_name].append(descriptor)

            # Build search index
            self._index_tool(idx, descriptor)

        # Detect collisions (tool names exposed by multiple servers)
        for tool_name, descriptors in self._tools_by_name.items():
            if len(descriptors) > 1:
                servers = {d.server_name for d in descriptors}
                if len(servers) > 1:
                    self._colliding_names.add(tool_name)

        self._total_docs = len(self._tools)

        elapsed = time.time() - start_time
        log_msg = (
            "Rebuilt MCP tool catalog: %d server tools from %d servers in %.2fms (collisions: %d)"
        )
        if filtered_count:
            log_msg += f" [filtered {filtered_count} client tools]"
        logger.info(
            log_msg,
            len(self._tools),
            len(self._tools_by_server),
            elapsed * 1000,
            len(self._colliding_names),
        )

        if self._colliding_names:
            logger.warning(
                "Tool name collisions detected across servers: %s",
                sorted(self._colliding_names),
            )

    def _index_tool(self, idx: int, descriptor: ToolDescriptor) -> None:
        """Add a tool to the search index."""
        # Combine searchable text
        searchable = " ".join(
            [
                descriptor.tool_name,
                descriptor.server_name,
                descriptor.description,
                " ".join(descriptor.arg_names),
            ]
        )

        tokens = tokenize(searchable)
        unique_tokens = set(tokens)

        for token in unique_tokens:
            if token not in self._token_index:
                self._token_index[token] = set()
            self._token_index[token].add(idx)
            self._doc_freq[token] += 1

    def get_colliding_names(self) -> set[str]:
        """Return tool names that are exposed by multiple servers."""
        return self._colliding_names.copy()

    def is_ambiguous(self, tool_name: str) -> bool:
        """Check if a tool name is ambiguous (exposed by multiple servers)."""
        return tool_name in self._colliding_names

    def get_servers_for_tool(self, tool_name: str) -> list[str]:
        """Return all server names that expose a given tool name."""
        descriptors = self._tools_by_name.get(tool_name, [])
        return list({d.server_name for d in descriptors})

    def search(
        self,
        query: str | None = None,
        top_k: int = 5,
        server_name: str | None = None,
        allowlist: list[str] | None = None,
    ) -> list[ToolDescriptor]:
        """
        Search for tools matching a query.

        Args:
            query: Natural language search query (None or empty for "list all")
            top_k: Maximum number of results to return
            server_name: Optional server filter
            allowlist: Optional list of allowed tool names or server names

        Returns:
            List of ToolDescriptor objects, ranked by relevance
        """
        # Start with all tools or server-filtered tools
        candidates = self._tools_by_server.get(server_name, []) if server_name else self._tools

        # Apply allowlist filtering
        if allowlist:
            allowlist_set = set(allowlist)
            candidates = [
                t
                for t in candidates
                if t.tool_name in allowlist_set or t.server_name in allowlist_set
            ]

        if not candidates:
            return []

        # If no query, return first top_k (stable order)
        if not query or not query.strip():
            return candidates[:top_k]

        # Score and rank candidates
        scored = self._rank_candidates(query, candidates)

        # Return top_k results
        return [t for t, _ in scored[:top_k]]

    def _rank_candidates(
        self,
        query: str,
        candidates: list[ToolDescriptor],
    ) -> list[tuple[ToolDescriptor, float]]:
        """
        Rank candidates by relevance to query.

        Scoring factors:
        - Exact name match (highest boost)
        - Name prefix match
        - Token overlap with description and args
        """
        query_lower = query.lower().strip()
        query_tokens = set(tokenize(query))

        scored: list[tuple[ToolDescriptor, float]] = []

        for tool in candidates:
            score = 0.0
            tool_name_lower = tool.tool_name.lower()

            # Exact name match (highest priority)
            if tool_name_lower == query_lower:
                score += 100.0
            # Name prefix match
            elif tool_name_lower.startswith(query_lower):
                score += 50.0
            # Query is substring of name
            elif query_lower in tool_name_lower:
                score += 30.0

            # Token overlap scoring (simple TF approach)
            tool_tokens = set(
                tokenize(f"{tool.tool_name} {tool.description} {' '.join(tool.arg_names)}")
            )

            overlap = query_tokens & tool_tokens
            if overlap:
                # Weight by inverse document frequency
                for token in overlap:
                    df = self._doc_freq.get(token, 1)
                    idf = 1.0 / (1.0 + df / max(self._total_docs, 1))
                    score += idf * 10.0

            # Small boost for tools with descriptions
            if tool.description:
                score += 0.1

            scored.append((tool, score))

        # Sort by score descending, then by name for stability
        scored.sort(key=lambda x: (-x[1], x[0].tool_name))

        return scored

    def filter_by_allowlist(
        self,
        tools: list[ToolDescriptor],
        allowlist: list[str],
    ) -> list[ToolDescriptor]:
        """
        Filter tools by an allowlist.

        Allowlist can contain:
        - Tool names (e.g., "tavily_search")
        - Server names (e.g., "tavily") - matches all tools from that server
        """
        if not allowlist:
            return tools

        allowlist_set = set(allowlist)

        return [t for t in tools if t.tool_name in allowlist_set or t.server_name in allowlist_set]

    def get_tool(
        self,
        tool_name: str,
        server_name: str | None = None,
    ) -> ToolDescriptor | None:
        """
        Get a specific tool by name, optionally scoped to a server.

        Args:
            tool_name: The tool name to look up
            server_name: Optional server to scope the lookup

        Returns:
            ToolDescriptor or None if not found
        """
        descriptors = self._tools_by_name.get(tool_name, [])

        if not descriptors:
            return None

        if server_name:
            for d in descriptors:
                if d.server_name == server_name:
                    return d
            return None

        # Return first match (may be ambiguous)
        return descriptors[0]

    def list_all(self, allowlist: list[str] | None = None) -> list[ToolDescriptor]:
        """
        List all tools in the catalog.

        Args:
            allowlist: Optional filter by tool names or server names

        Returns:
            List of all ToolDescriptor objects (filtered if allowlist provided)
        """
        if allowlist:
            return self.filter_by_allowlist(self._tools, allowlist)
        return list(self._tools)

    @property
    def tool_count(self) -> int:
        """Return the total number of tools in the catalog."""
        return len(self._tools)

    @property
    def server_count(self) -> int:
        """Return the number of servers with tools."""
        return len(self._tools_by_server)


# Module-level singleton for the catalog
_catalog_instance: McpToolCatalog | None = None


async def get_tool_catalog(mcp_manager: Any) -> McpToolCatalog:
    """
    Get or create the global tool catalog instance.

    Args:
        mcp_manager: The MCPManager to use for tool loading

    Returns:
        The McpToolCatalog singleton
    """
    global _catalog_instance

    if _catalog_instance is None:
        _catalog_instance = McpToolCatalog(mcp_manager)

    await _catalog_instance.refresh_if_needed()
    return _catalog_instance


def reset_tool_catalog() -> None:
    """Reset the global catalog instance (for testing)."""
    global _catalog_instance
    _catalog_instance = None
