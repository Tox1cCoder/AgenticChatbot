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
from .text_normalization import sanitize_identifier, tokenize_text
from .tool_search_scoring import (
    ToolSearchScore,
    build_query_tokens,
    rank_tool_candidates,
    score_tool,
)

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
    # Deterministic invokable alias for ambiguous same-name tools.
    # None for non-ambiguous tools (call_name == tool_name in that case).
    call_name: str | None = field(default=None)

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

    def get_call_name(self) -> str:
        """Return the invokable name for this tool.

        For non-ambiguous tools this is equal to tool_name.
        For ambiguous same-name tools this is the deterministic alias
        '{sanitized_server}__{tool_name}'.
        """
        return self.call_name if self.call_name is not None else self.tool_name

    def to_search_result(self) -> dict[str, Any]:
        """
        Convert to a search result dict for tool_search output.

        Exposes call_name so the model always knows the exact invokable name.
        For non-ambiguous tools call_name equals tool_name.
        For ambiguous same-name tools call_name is the deterministic alias.
        Exposes source_server only when the tool is ambiguous (call_name differs
        from tool_name), so the model can disambiguate.
        """
        effective_call_name = self.get_call_name()
        result: dict[str, Any] = {
            "tool_name": effective_call_name,
            "description": self.description[:200] if self.description else "",
            "arg_hints": self.arg_hints,
            "is_loaded": False,  # Will be updated by tool_search
        }
        # Expose source_server only for ambiguous tools so the model can reason
        # about which server provides what capability
        if self.call_name is not None:
            result["source_server"] = self.server_name
        return result

    def _to_internal_result(self) -> dict[str, Any]:
        """
        Internal result with full metadata for autoloading logic.
        NOT exposed to the model.
        """
        effective_call_name = self.get_call_name()
        return {
            "tool_name": self.tool_name,
            "call_name": effective_call_name,
            "server_name": self.server_name,
            "qualified_tool_id": f"{self.server_name}::{self.tool_name}",
            "description": self.description[:200] if self.description else "",
            "arg_hints": self.arg_hints,
            "origin": self.origin,
            "is_client_tool": False,
        }



@dataclass
class ToolReference:
    """Minimal reference to a tool for loading purposes."""

    tool_name: str
    server_name: str
    # The invokable name used as the deferred-state storage key.
    # Equals tool_name for non-ambiguous tools; alias for ambiguous ones.
    call_name: str | None = None

    def get_call_name(self) -> str:
        """Return the stable call key for deferred state storage."""
        return self.call_name if self.call_name is not None else self.tool_name


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


class McpToolCatalog:
    """
    Searchable catalog of MCP tools with generation-based caching.

    The catalog automatically refreshes when the MCP tools generation changes
    (e.g., when servers are enabled/disabled or config is modified).
    """

    def __init__(self, mcp_manager: Any):
        self._mcp_manager = mcp_manager
        self._cached_generation: int = -1
        self._tools: list[ToolDescriptor] = []
        self._tools_by_name: dict[str, list[ToolDescriptor]] = {}
        self._tools_by_server: dict[str, list[ToolDescriptor]] = {}
        self._colliding_names: set[str] = set()
        # Case-insensitive server name lookup: {lower_name: canonical_name}
        self._server_name_lower_map: dict[str, str] = {}
        self._server_descriptions: dict[str, str] = {}

    async def refresh_if_needed(self) -> bool:
        """Rebuild when the MCP tools generation changed; True if it rebuilt."""
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

        # Fetch BEFORE clearing, and never await between clear and fill: the
        # catalog is a shared singleton, so a search issued during the fetch
        # used to see an empty catalog, and two overlapping rebuilds both
        # appended into the same lists and duplicated every descriptor.
        try:
            tools_info = await self._mcp_manager.get_all_tools_info()
        except Exception as e:
            logger.error("Failed to fetch tools from MCP manager: %s", e)
            tools_info = []

        self._tools.clear()
        self._tools_by_name.clear()
        self._tools_by_server.clear()
        self._colliding_names.clear()
        self._server_name_lower_map.clear()
        self._server_descriptions.clear()

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

            self._tools.append(descriptor)
            self._tools_by_name.setdefault(tool_name, []).append(descriptor)
            self._tools_by_server.setdefault(server_name, []).append(descriptor)

        # Build case-insensitive server name lookup
        for sname in self._tools_by_server:
            self._server_name_lower_map[sname.lower()] = sname

        # Cache configured server descriptions so inventory mode can expose
        # short capability summaries without forcing the model to guess from
        # opaque server identifiers alone.
        for sname, info in self._mcp_manager.get_servers_status().items():
            description = str((info or {}).get("description") or "").strip()
            if description:
                self._server_descriptions[sname] = description

        # Detect collisions (tool names exposed by multiple servers)
        for tool_name, descriptors in self._tools_by_name.items():
            if len(descriptors) > 1:
                servers = {d.server_name for d in descriptors}
                if len(servers) > 1:
                    self._colliding_names.add(tool_name)
                    # Assign deterministic aliases to all descriptors with this name
                    for descriptor in descriptors:
                        sanitized = sanitize_identifier(descriptor.server_name)
                        descriptor.call_name = f"{sanitized}__{descriptor.tool_name}"
                        logger.debug(
                            "Ambiguous tool '%s' from server '%s' aliased to '%s'",
                            tool_name,
                            descriptor.server_name,
                            descriptor.call_name,
                        )

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

    def is_ambiguous(self, tool_name: str) -> bool:
        """Check if a tool name is ambiguous (exposed by multiple servers)."""
        return tool_name in self._colliding_names

    def resolve_server_name(self, server_name: str) -> str | None:
        """Resolve a server name case-insensitively to its canonical form.

        Returns the canonical server name if found, or None if no server
        with that name (case-insensitive) exists in the catalog.
        """
        canonical = self._server_name_lower_map.get(server_name.lower())
        if canonical is not None:
            return canonical

        from ..core.config import settings as _settings

        query_lower, query_tokens = build_query_tokens(server_name)
        if not query_tokens:
            return None

        profiles: list[tuple[str, str, list[str]]] = []
        doc_freq: Counter = Counter()

        for candidate_name, descriptors in self._tools_by_server.items():
            candidate_tokens = set(tokenize_text(candidate_name))
            has_name_signal = (
                candidate_name.lower() == query_lower
                or candidate_name.lower().startswith(query_lower)
                or query_lower.startswith(candidate_name.lower())
                or bool(query_tokens & candidate_tokens)
            )
            if not has_name_signal:
                continue

            description = self._server_descriptions.get(candidate_name, "").strip()
            example_tools = [descriptor.tool_name for descriptor in descriptors[:3]]
            profiles.append((candidate_name, description, example_tools))

            searchable = " ".join([candidate_name, description, " ".join(example_tools)])
            for token in set(tokenize_text(searchable)):
                doc_freq[token] += 1

        if not profiles:
            return None

        total_profiles = len(profiles)
        scored: list[tuple[str, float]] = []
        for candidate_name, description, example_tools in profiles:
            score = score_tool(
                tool_name=candidate_name,
                description=description,
                arg_names=example_tools,
                query_lower=query_lower,
                query_tokens=query_tokens,
                doc_freq=doc_freq,
                total_docs=total_profiles,
            )
            scored.append((candidate_name, score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        top_name, top_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0

        if top_score < _settings.mcp_tool_search_autoload_min_relevance_score:
            return None
        if (
            second_score
            and (top_score - second_score) < _settings.mcp_tool_search_min_relevance_score
        ):
            return None

        logger.debug(
            "tool_search: server_name=%r fuzzy-resolved to %r (score=%.2f, second=%.2f)",
            server_name,
            top_name,
            top_score,
            second_score,
        )
        return top_name

    def get_server_inventory(self, allowlist: list[str] | None = None) -> list[dict]:
        """Return server-level inventory summaries (name + tool count).

        Used by inventory mode (tool_search with no query and no server_name).
        Returns a token-cheap summary: server_name, short description, and
        tool_count per server.
        """
        summaries = []
        for sname, descriptors in self._tools_by_server.items():
            if allowlist:
                allowlist_set = set(allowlist)
                visible = [
                    d for d in descriptors if self._descriptor_matches_allowlist(d, allowlist_set)
                ]
                if not visible:
                    continue
                tool_count = len(visible)
            else:
                tool_count = len(descriptors)
            description = self._server_descriptions.get(sname, "").strip()
            if not description and descriptors:
                example_tools = ", ".join(descriptor.tool_name for descriptor in descriptors[:3])
                description = f"Tools: {example_tools}"
            summaries.append(
                {
                    "server_name": sname,
                    "description": description,
                    "tool_count": tool_count,
                }
            )
        return summaries

    @staticmethod
    def _descriptor_matches_allowlist(
        descriptor: ToolDescriptor,
        allowlist_set: set[str],
    ) -> bool:
        qualified_id = f"{descriptor.server_name}::{descriptor.tool_name}"
        return (
            descriptor.tool_name in allowlist_set
            or descriptor.server_name in allowlist_set
            or qualified_id in allowlist_set
            or descriptor.get_call_name() in allowlist_set
        )

    def search(
        self,
        query: str | None = None,
        top_k: int = 5,
        server_name: str | None = None,
        allowlist: list[str] | None = None,
    ) -> list[ToolDescriptor]:
        """Rank tools by relevance; an empty ``query`` lists in stable order.

        ``allowlist`` entries may be tool names, server names, qualified ids,
        or call names.
        """
        # Canonicalize server_name case-insensitively
        canonical_server = None
        if server_name:
            canonical_server = self.resolve_server_name(server_name)
            if canonical_server is None:
                logger.debug(
                    "tool_search: server_name=%r not found in catalog "
                    "(case-insensitive lookup failed)",
                    server_name,
                )
                return []
            if canonical_server != server_name:
                logger.debug(
                    "tool_search: server_name=%r canonicalized to %r",
                    server_name,
                    canonical_server,
                )

        # Start with all tools or server-filtered tools
        candidates = (
            self._tools_by_server.get(canonical_server, []) if canonical_server else self._tools
        )

        # Apply allowlist filtering
        if allowlist:
            allowlist_set = set(allowlist)
            candidates = [
                t for t in candidates if self._descriptor_matches_allowlist(t, allowlist_set)
            ]

        if not candidates:
            return []

        # If no query, return first top_k (stable order)
        if not query or not query.strip():
            return candidates[:top_k]

        # Score and rank candidates via the intent-aware scorer, unwrapping tools.
        return [item.tool for item in self.search_scored(query, top_k, server_name, allowlist)]

    def search_scored(
        self,
        query: str,
        top_k: int = 5,
        server_name: str | None = None,
        allowlist: list[str] | None = None,
    ) -> list[ToolSearchScore]:
        """Like search(), but returns ToolSearchScore objects.

        Each result carries the descriptor (``.tool``), the numeric score,
        confidence band, match reasons, capability profile, and autoload
        eligibility, so tool_search can build compact model-facing results and
        gate autoloading on a single high-confidence recommendation.
        """
        # Canonicalize server_name case-insensitively
        canonical_server = self.resolve_server_name(server_name) if server_name else None
        if server_name and canonical_server is None:
            return []

        candidates = (
            self._tools_by_server.get(canonical_server, []) if canonical_server else self._tools
        )

        if allowlist:
            allowlist_set = set(allowlist)
            candidates = [
                tool
                for tool in candidates
                if self._descriptor_matches_allowlist(tool, allowlist_set)
            ]

        if not query or not query.strip():
            return []

        return rank_tool_candidates(query=query, candidates=candidates)[:top_k]

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

        return [t for t in tools if self._descriptor_matches_allowlist(t, allowlist_set)]

    def list_all(self, allowlist: list[str] | None = None) -> list[ToolDescriptor]:
        if allowlist:
            return self.filter_by_allowlist(self._tools, allowlist)
        return list(self._tools)

    @property
    def tool_count(self) -> int:
        return len(self._tools)


_catalog_instance: McpToolCatalog | None = None


async def get_tool_catalog(mcp_manager: Any) -> McpToolCatalog:
    """The process-wide catalog. ``mcp_manager`` is only used on first creation."""
    global _catalog_instance

    if _catalog_instance is None:
        _catalog_instance = McpToolCatalog(mcp_manager)

    await _catalog_instance.refresh_if_needed()
    return _catalog_instance
