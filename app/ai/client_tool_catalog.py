"""
Client Tool Catalog - Searchable catalog of client device tools.

This module provides a searchable catalog for tools available on connected
client devices. It mirrors the API of McpToolCatalog to enable unified
tool_search functionality across server and client tools.

Key features:
- Device-scoped: Only includes tools from the currently connected device
- Lightweight search: Same ranking approach as server tools
- Consistent interface: Same ToolDescriptor/ToolReference types
"""

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.services.client_device_service import ClientDeviceService, DeviceSession

from .client_runtime_tools import (
    CLIENT_TOOL_PREFIX,
    TOOL_ORIGIN_CLIENT_MCP,
    TOOL_ORIGIN_CLIENT_NATIVE,
)
from .text_normalization import sanitize_identifier, tokenize_text
from .tool_search_scoring import build_query_tokens, rank_and_filter, score_tool

logger = logging.getLogger(__name__)


@dataclass
class ClientToolDescriptor:
    """
    Describes a single client device tool with metadata for search and loading.

    This is the client tool equivalent of ToolDescriptor from mcp_tool_catalog.
    """

    tool_name: str  # The exposed name (e.g., client__shell_execute)
    server_name: str  # "native" for native tools, or the local MCP server name
    description: str
    arg_names: list[str]
    required_arg_names: list[str]
    qualified_tool_id: str  # e.g., "native::shell_execute" or "pylance::get_docs"
    origin: str  # TOOL_ORIGIN_CLIENT_MCP or TOOL_ORIGIN_CLIENT_NATIVE
    device_id: str  # The device this tool belongs to
    session_id: str = ""
    catalog_version: int = 0
    tool_instance_id: str = ""
    args_schema: dict[str, Any] = field(default_factory=dict)

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
        Do NOT expose internal details like server_name, origin, device_id, etc.
        that could lead to the model guessing tool names or revealing architecture.
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
            "qualified_tool_id": self.qualified_tool_id,
            "description": self.description[:200] if self.description else "",
            "arg_hints": self.arg_hints,
            "origin": self.origin,
            "device_id": self.device_id,
            "session_id": self.session_id,
            "catalog_version": self.catalog_version,
            "tool_instance_id": self.tool_instance_id,
            "is_client_tool": True,
        }

    def is_client_tool(self) -> bool:
        """Always True for ClientToolDescriptor."""
        return True


@dataclass
class ClientToolReference:
    """Minimal reference to a client tool for loading purposes."""

    tool_name: str
    server_name: str
    device_id: str
    session_id: str = ""
    catalog_version: int = 0
    tool_instance_id: str = ""

    def is_client_tool(self) -> bool:
        """Always True for ClientToolReference."""
        return True


def _extract_arg_info(args_schema: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Extract argument names and required argument names from a JSON schema."""
    properties = args_schema.get("properties", {})
    required = set(args_schema.get("required", []))

    all_args = list(properties.keys())
    required_args = [arg for arg in all_args if arg in required]

    return all_args, required_args


def _tokenize(text: str) -> list[str]:
    """Tokenize text for search matching."""
    return tokenize_text(text)


class ClientToolCatalog:
    """
    Searchable catalog of tools from a connected client device.

    This catalog is built from the tool catalog synced by a connected device
    and provides the same search interface as McpToolCatalog.
    """

    def __init__(self, device_id: str, user_id: str):
        """
        Initialize the catalog for a specific device.

        Args:
            device_id: The device UUID string
            user_id: The user UUID string
        """
        self._device_id = device_id
        self._user_id = user_id
        self._tools: list[ClientToolDescriptor] = []
        self._tools_by_name: dict[str, ClientToolDescriptor] = {}
        self._tools_by_server: dict[str, list[ClientToolDescriptor]] = {}
        # Case-insensitive server name lookup: {lower_name: canonical_name}
        self._server_name_lower_map: dict[str, str] = {}
        self._token_index: dict[str, set[int]] = {}
        self._doc_freq: Counter = Counter()
        self._total_docs: int = 0
        self._session_id: str = ""
        self._catalog_version: int = -1
        self._last_refresh: float = 0

    def refresh_from_session(self, session: DeviceSession | None = None) -> bool:
        """
        Refresh the catalog from the device session.

        Args:
            session: Optional DeviceSession. If not provided, will look up from registry.

        Returns:
            True if the catalog was refreshed, False if no refresh needed
        """
        from uuid import UUID

        if session is None:
            try:
                device_uuid = UUID(self._device_id)
                session = ClientDeviceService.lookup_active_session(device_uuid)
            except Exception:
                logger.warning("Invalid device_id for client tool catalog: %s", self._device_id)
                return False

        if session is None:
            # Device not connected - clear catalog
            if self._tools:
                self._clear()
            return False

        if str(session.user_id) != str(self._user_id):
            logger.warning(
                "Ignoring client tool catalog refresh for device %s because it is bound to user %s, "
                "not user %s",
                self._device_id,
                session.user_id,
                self._user_id,
            )
            if self._tools:
                self._clear()
            return False

        # Check if catalog version changed
        if session.tool_catalog_version == self._catalog_version:
            return False

        self._rebuild_from_catalog(
            session.tool_catalog,
            session.tool_catalog_version,
            session.session_id,
        )
        return True

    def _clear(self) -> None:
        """Clear all catalog data."""
        self._tools.clear()
        self._tools_by_name.clear()
        self._tools_by_server.clear()
        self._server_name_lower_map.clear()
        self._token_index.clear()
        self._doc_freq.clear()
        self._total_docs = 0
        self._session_id = ""
        self._catalog_version = -1

    def _rebuild_from_catalog(self, catalog: dict[str, Any], version: int, session_id: str) -> None:
        """Rebuild the catalog from a device's tool catalog."""
        start_time = time.time()

        self._clear()
        self._session_id = session_id
        self._catalog_version = version
        self._last_refresh = time.time()

        raw_tools = catalog.get("tools", []) if isinstance(catalog, dict) else []
        if not isinstance(raw_tools, list):
            return

        for raw_entry in raw_tools:
            if not isinstance(raw_entry, dict):
                continue

            # Extract tool info
            name = str(raw_entry.get("name") or "").strip()
            qualified_id = str(raw_entry.get("qualified_id") or "").strip()
            if not name or not qualified_id:
                continue

            origin = str(raw_entry.get("origin") or "native").strip().lower()
            server_name = str(raw_entry.get("server_name") or "native").strip()
            description = str(raw_entry.get("description") or "").strip()
            input_schema = raw_entry.get("input_schema", {}) or {}

            # Build the exposed name (with client__ prefix)
            if origin == "mcp" and server_name:
                exposed_name = f"{CLIENT_TOOL_PREFIX}{server_name}__{name}".lower()
                tool_origin = TOOL_ORIGIN_CLIENT_MCP
            else:
                exposed_name = f"{CLIENT_TOOL_PREFIX}{name}".lower()
                tool_origin = TOOL_ORIGIN_CLIENT_NATIVE

            # Sanitize exposed name
            exposed_name = sanitize_identifier(exposed_name)

            arg_names, required_args = _extract_arg_info(input_schema)

            descriptor = ClientToolDescriptor(
                tool_name=exposed_name,
                server_name=server_name,
                description=description,
                arg_names=arg_names,
                required_arg_names=required_args,
                qualified_tool_id=qualified_id,
                origin=tool_origin,
                device_id=self._device_id,
                session_id=self._session_id,
                catalog_version=self._catalog_version,
                tool_instance_id=str(raw_entry.get("tool_instance_id") or ""),
                args_schema=input_schema,
            )

            idx = len(self._tools)
            self._tools.append(descriptor)

            # Index by exposed name
            self._tools_by_name[exposed_name] = descriptor

            # Index by server
            if server_name not in self._tools_by_server:
                self._tools_by_server[server_name] = []
            self._tools_by_server[server_name].append(descriptor)

            # Build search index
            self._index_tool(idx, descriptor)

        self._total_docs = len(self._tools)

        # Build case-insensitive server name lookup
        for sname in self._tools_by_server:
            self._server_name_lower_map[sname.lower()] = sname

        elapsed = time.time() - start_time
        logger.info(
            "Rebuilt client tool catalog for device %s: %d tools in %.2fms",
            self._device_id[:8],
            len(self._tools),
            elapsed * 1000,
        )

    def _index_tool(self, idx: int, descriptor: ClientToolDescriptor) -> None:
        """Add a tool to the search index."""
        searchable = " ".join(
            [
                descriptor.tool_name,
                descriptor.server_name,
                descriptor.description,
                " ".join(descriptor.arg_names),
            ]
        )

        tokens = _tokenize(searchable)
        unique_tokens = set(tokens)

        for token in unique_tokens:
            if token not in self._token_index:
                self._token_index[token] = set()
            self._token_index[token].add(idx)
            self._doc_freq[token] += 1

    def resolve_server_name(self, server_name: str) -> str | None:
        """Resolve a server name case-insensitively to its canonical form."""
        canonical = self._server_name_lower_map.get(server_name.lower())
        if canonical is not None:
            return canonical

        from app.core.config import settings as _settings

        query_lower, query_tokens = build_query_tokens(server_name)
        if not query_tokens:
            return None

        profiles: list[tuple[str, str, list[str]]] = []
        doc_freq: Counter = Counter()

        for candidate_name, descriptors in self._tools_by_server.items():
            candidate_tokens = set(_tokenize(candidate_name))
            has_name_signal = (
                candidate_name.lower() == query_lower
                or candidate_name.lower().startswith(query_lower)
                or query_lower.startswith(candidate_name.lower())
                or bool(query_tokens & candidate_tokens)
            )
            if not has_name_signal:
                continue

            description = ""
            example_tools = [descriptor.tool_name for descriptor in descriptors[:3]]
            profiles.append((candidate_name, description, example_tools))

            searchable = " ".join([candidate_name, " ".join(example_tools)])
            for token in set(_tokenize(searchable)):
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
        if second_score and (top_score - second_score) < _settings.mcp_tool_search_min_relevance_score:
            return None

        logger.debug(
            "client tool search: server_name=%r fuzzy-resolved to %r (score=%.2f, second=%.2f)",
            server_name,
            top_name,
            top_score,
            second_score,
        )
        return top_name

    def search(
        self,
        query: str | None = None,
        top_k: int = 5,
        server_name: str | None = None,
        allowlist: list[str] | None = None,
    ) -> list[ClientToolDescriptor]:
        """
        Search for tools matching a query.

        Args:
            query: Natural language search query (None for "list all")
            top_k: Maximum results to return
            server_name: Optional server filter
            allowlist: Optional allowlist filter

        Returns:
            List of ClientToolDescriptor objects, ranked by relevance
        """
        # Canonicalize server_name case-insensitively
        canonical_server = None
        if server_name:
            canonical_server = self.resolve_server_name(server_name)
            if canonical_server is None:
                logger.debug(
                    "client tool search: server_name=%r not found (case-insensitive lookup failed)",
                    server_name,
                )
                return []

        # Start with all tools or server-filtered tools
        candidates = self._tools_by_server.get(canonical_server, []) if canonical_server else self._tools

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

        # If no query, return first top_k
        if not query or not query.strip():
            return candidates[:top_k]

        # Score and rank
        scored = self._rank_candidates(query, candidates)
        return [t for t, _ in scored[:top_k]]

    def _rank_candidates(
        self,
        query: str,
        candidates: list[ClientToolDescriptor],
    ) -> list[tuple[ClientToolDescriptor, float]]:
        """Rank candidates by relevance to query using shared scoring logic."""
        from app.core.config import settings as _settings

        query_lower, query_tokens = build_query_tokens(query)
        min_score = _settings.mcp_tool_search_min_relevance_score

        scored: list[tuple[ClientToolDescriptor, float]] = []
        for tool in candidates:
            s = score_tool(
                tool_name=tool.tool_name,
                description=tool.description,
                arg_names=tool.arg_names,
                query_lower=query_lower,
                query_tokens=query_tokens,
                doc_freq=self._doc_freq,
                total_docs=self._total_docs,
            )
            scored.append((tool, s))

        return rank_and_filter(scored, min_relevance_score=min_score)

    def get_tool(
        self,
        tool_name: str,
        server_name: str | None = None,
    ) -> ClientToolDescriptor | None:
        """Get a specific tool by name."""
        descriptor = self._tools_by_name.get(tool_name)
        if descriptor and server_name and descriptor.server_name != server_name:
            return None
        return descriptor

    def list_all(self, allowlist: list[str] | None = None) -> list[ClientToolDescriptor]:
        """List all tools in the catalog."""
        if allowlist:
            allowlist_set = set(allowlist)
            return [
                t
                for t in self._tools
                if t.tool_name in allowlist_set or t.server_name in allowlist_set
            ]
        return list(self._tools)

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def catalog_version(self) -> int:
        return self._catalog_version

    @property
    def session_id(self) -> str:
        return self._session_id


# Module-level cache for client catalogs
_client_catalogs: dict[str, ClientToolCatalog] = {}


def get_client_tool_catalog(device_id: str, user_id: str) -> ClientToolCatalog:
    """
    Get or create a client tool catalog for a device.

    Args:
        device_id: The device UUID string
        user_id: The user UUID string

    Returns:
        ClientToolCatalog instance (may be empty if device not connected)
    """
    cache_key = f"{user_id}:{device_id}"

    if cache_key not in _client_catalogs:
        _client_catalogs[cache_key] = ClientToolCatalog(device_id, user_id)

    catalog = _client_catalogs[cache_key]
    catalog.refresh_from_session()
    return catalog


def clear_client_tool_catalog(device_id: str, user_id: str) -> None:
    """Clear a cached client tool catalog."""
    cache_key = f"{user_id}:{device_id}"
    _client_catalogs.pop(cache_key, None)


def reset_all_client_catalogs() -> None:
    """Clear all cached client tool catalogs (for testing)."""
    _client_catalogs.clear()
