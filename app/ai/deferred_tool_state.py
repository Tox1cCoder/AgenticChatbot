"""
Deferred Tool State - Per-conversation tracking of loaded MCP tools.

This module manages deferred tool loading for both server MCP tools and
client device tools. Tools are loaded on-demand via tool_search and then
remain available for subsequent calls in the same conversation.

Key features:
- LRU eviction when capacity exceeded
- TTL-based expiration
- Generation tracking for stale tool detection
- Separate tracking for server vs client tools
"""

import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING
from uuid import UUID

from ..core.config import settings
from .mcp_registry import get_mcp_tools_generation
from .mcp_tool_catalog import ToolReference

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class LoadedTool:
    """
    Represents a tool that has been loaded for a conversation.

    Attributes:
        tool_name: The name of the tool
        server_name: The server that provides the tool
        loaded_at: Timestamp when the tool was loaded
        last_used: Timestamp when the tool was last accessed (for LRU)
        generation: The MCP tools generation when loaded
    """

    tool_name: str
    server_name: str
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    generation: int = 0

    def touch(self) -> None:
        """Update the last_used timestamp."""
        self.last_used = time.time()

    def is_expired(self, ttl_minutes: int) -> bool:
        """Check if the tool has exceeded its TTL."""
        if ttl_minutes <= 0:
            return False
        age_minutes = (time.time() - self.loaded_at) / 60.0
        return age_minutes > ttl_minutes

    def is_stale(self, current_generation: int) -> bool:
        """Check if the tool was loaded from an outdated generation."""
        return self.generation != current_generation

    def to_reference(self) -> ToolReference:
        """Convert to a ToolReference."""
        return ToolReference(tool_name=self.tool_name, server_name=self.server_name)


@dataclass
class LoadedClientTool:
    """
    Represents a client device tool that has been loaded for a conversation.

    Similar to LoadedTool but includes device binding information.

    Attributes:
        tool_name: The name of the tool (with client__ prefix)
        server_name: The local MCP server name (or "native")
        device_id: The device this tool belongs to
        loaded_at: Timestamp when the tool was loaded
        last_used: Timestamp when the tool was last accessed (for LRU)
        catalog_version: The client catalog version when loaded (session-scoped)
        tool_instance_id: Opaque capability identifier for dispatch validation
    """

    tool_name: str
    server_name: str
    device_id: str
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    catalog_version: int = 0
    tool_instance_id: str = ""

    def touch(self) -> None:
        """Update the last_used timestamp."""
        self.last_used = time.time()

    def is_expired(self, ttl_minutes: int) -> bool:
        """Check if the tool has exceeded its TTL."""
        if ttl_minutes <= 0:
            return False
        age_minutes = (time.time() - self.loaded_at) / 60.0
        return age_minutes > ttl_minutes

    def is_client_tool(self) -> bool:
        """Always True for LoadedClientTool."""
        return True


@dataclass
class ClientToolScope:
    """
    Set of client tools loaded for a specific execution scope.

    Keyed by (conversation_id, agent_key, device_id, session_id) so that
    two sidecars connected to the same conversation get independent pools
    and independent LRU budgets.
    """

    loaded: dict[str, LoadedClientTool] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def add(
        self,
        tool_name: str,
        server_name: str,
        device_id: str,
        catalog_version: int,
        max_tools: int,
        tool_instance_id: str = "",
    ) -> LoadedClientTool | None:
        """Add or replace a client tool, evicting LRU when at capacity."""
        if tool_name in self.loaded:
            existing = self.loaded[tool_name]
            existing.server_name = server_name
            existing.device_id = device_id
            existing.catalog_version = catalog_version
            existing.tool_instance_id = tool_instance_id
            existing.touch()
            return existing

        while len(self.loaded) >= max_tools:
            evicted = self._evict_lru()
            if evicted:
                logger.debug("Evicted LRU client tool '%s'", evicted)
            else:
                return None

        loaded_tool = LoadedClientTool(
            tool_name=tool_name,
            server_name=server_name,
            device_id=device_id,
            catalog_version=catalog_version,
            tool_instance_id=tool_instance_id,
        )
        self.loaded[tool_name] = loaded_tool
        return loaded_tool

    def get(self, tool_name: str) -> LoadedClientTool | None:
        tool = self.loaded.get(tool_name)
        if tool:
            tool.touch()
        return tool

    def cleanup_expired(self, ttl_minutes: int) -> int:
        to_remove = [name for name, tool in self.loaded.items() if tool.is_expired(ttl_minutes)]
        for name in to_remove:
            del self.loaded[name]
        return len(to_remove)

    def _evict_lru(self) -> str | None:
        if not self.loaded:
            return None
        lru_name = min(self.loaded.keys(), key=lambda k: self.loaded[k].last_used)
        self.loaded.pop(lru_name)
        return lru_name

    def list_tools(self) -> list[LoadedClientTool]:
        return list(self.loaded.values())

    def __len__(self) -> int:
        return len(self.loaded)

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self.loaded


@dataclass
class ConversationToolSet:
    """
    Set of server MCP tools loaded for a specific (conversation_id, agent_key) pair.

    Note: Client device tools are tracked separately in ClientToolScope
    instances keyed by execution scope (conversation_id, agent_key,
    device_id, session_id).
    """

    loaded: dict[str, LoadedTool] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def add(
        self,
        tool_name: str,
        server_name: str,
        generation: int,
        max_tools: int,
        call_name: str | None = None,
    ) -> LoadedTool | None:
        # Use call_name as the stable storage key; fall back to tool_name
        storage_key = call_name if call_name is not None else tool_name
        if storage_key in self.loaded:
            existing = self.loaded[storage_key]
            if existing.server_name != server_name:
                logger.debug(
                    "Replacing tool '%s' binding: %s -> %s",
                    storage_key,
                    existing.server_name,
                    server_name,
                )
            existing.server_name = server_name
            existing.generation = generation
            existing.touch()
            return existing

        while len(self.loaded) >= max_tools:
            evicted = self._evict_lru()
            if evicted:
                logger.debug(
                    "Evicted LRU tool '%s' from %s to make room",
                    evicted.tool_name,
                    evicted.server_name,
                )
            else:
                logger.warning("Could not evict tool to make room for '%s'", storage_key)
                return None

        loaded_tool = LoadedTool(
            tool_name=tool_name,
            server_name=server_name,
            generation=generation,
        )
        self.loaded[storage_key] = loaded_tool
        return loaded_tool

    def get(self, call_name: str) -> LoadedTool | None:
        tool = self.loaded.get(call_name)
        if tool:
            tool.touch()
            return tool
        return None

    def remove(self, call_name: str) -> LoadedTool | None:
        return self.loaded.pop(call_name, None)

    def _evict_lru(self) -> LoadedTool | None:
        if not self.loaded:
            return None
        lru_name = min(self.loaded.keys(), key=lambda k: self.loaded[k].last_used)
        return self.loaded.pop(lru_name)

    def cleanup_expired(self, ttl_minutes: int, current_generation: int) -> int:
        to_remove = []
        for name, tool in self.loaded.items():
            if tool.is_expired(ttl_minutes):
                to_remove.append(name)
                logger.debug("Tool '%s' expired (TTL)", name)
            elif tool.is_stale(current_generation):
                to_remove.append(name)
                logger.debug("Tool '%s' stale (generation mismatch)", name)
        for name in to_remove:
            del self.loaded[name]
        return len(to_remove)

    def list_tools(self) -> list[ToolReference]:
        return [tool.to_reference() for tool in self.loaded.values()]

    def list_all_tool_names(self) -> list[str]:
        return list(self.loaded.keys())

    def __len__(self) -> int:
        return len(self.loaded)

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self.loaded


class DeferredToolState:
    """
    Global state manager for deferred tool loading across all conversations.

    Thread-safe for concurrent access.
    """

    def __init__(self):
        # Map from (conversation_id, agent_key) -> ConversationToolSet (server tools only)
        self._conversation_tools: dict[tuple[str, str], ConversationToolSet] = {}
        # Map from (conversation_id, agent_key, device_id, session_id) -> ClientToolScope
        self._client_tool_scopes: dict[tuple[str, str, str, str], ClientToolScope] = {}
        self._lock = Lock()

    def _get_key(
        self,
        conversation_id: str | None,
        agent_key: str | None,
    ) -> tuple[str, str]:
        """Generate a lookup key from conversation_id and agent_key."""
        return (conversation_id or "", agent_key or "default")

    def _get_client_key(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        device_id: str | None,
        session_id: str | None,
    ) -> tuple[str, str, str, str]:
        """Generate a lookup key for client tool scopes."""
        return (
            conversation_id or "",
            agent_key or "default",
            device_id or "",
            session_id or "",
        )

    @staticmethod
    def _resolve_active_session_id(device_id: str | None) -> str | None:
        if not device_id:
            return None

        from app.services.client_device_service import ClientDeviceService

        try:
            session = ClientDeviceService.lookup_active_session(UUID(str(device_id)))
        except Exception:
            return None
        return session.session_id if session is not None else None

    def autoload(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        references: list[ToolReference],
        max_tools: int | None = None,
    ) -> list[ToolReference]:
        """
        Load tools for a conversation, respecting capacity limits.

        This is called by tool_search to load the top-N search results.

        Args:
            conversation_id: The conversation to load tools for
            agent_key: The agent key (e.g., "chat", "rag")
            references: List of ToolReferences to load
            max_tools: Override for max tools (uses config default if None)

        Returns:
            List of ToolReferences that were actually loaded
        """
        if max_tools is None:
            max_tools = settings.mcp_tool_search_max_loaded_tools_per_conversation

        generation = get_mcp_tools_generation()
        key = self._get_key(conversation_id, agent_key)
        loaded: list[ToolReference] = []

        with self._lock:
            if key not in self._conversation_tools:
                self._conversation_tools[key] = ConversationToolSet()

            tool_set = self._conversation_tools[key]

            # Clean up expired tools first
            ttl = settings.mcp_tool_search_loaded_tools_ttl_minutes
            tool_set.cleanup_expired(ttl, generation)

            # Load each reference, keyed by call_name for stable identity
            for ref in references:
                call_name = getattr(ref, "call_name", None)
                result = tool_set.add(
                    tool_name=ref.tool_name,
                    server_name=ref.server_name,
                    generation=generation,
                    max_tools=max_tools,
                    call_name=call_name,
                )
                if result:
                    loaded.append(ref)

        return loaded

    def autoload_client_tools(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        references: list,  # List of ClientToolReference
        device_id: str | None,
        session_id: str | None = None,
        user_id: str | None = None,
        max_tools: int | None = None,
    ) -> list:
        """
        Load client device tools for a conversation, respecting capacity limits.

        Client tools are stored in a separate scope keyed by
        (conversation_id, agent_key, device_id, session_id) so that
        two sidecars in the same conversation get independent pools.
        """
        from .client_tool_catalog import ClientToolReference, get_client_tool_catalog

        if max_tools is None:
            max_tools = settings.mcp_tool_search_max_loaded_tools_per_conversation

        if not device_id:
            return []

        effective_session_id = session_id
        if not effective_session_id:
            effective_session_id = next(
                (str(ref.session_id) for ref in references if getattr(ref, "session_id", None)),
                None,
            )
        if not effective_session_id:
            effective_session_id = self._resolve_active_session_id(device_id)

        client_key = self._get_client_key(
            conversation_id,
            agent_key,
            device_id,
            effective_session_id,
        )
        loaded: list[ClientToolReference] = []

        catalog_version = 0
        if device_id:
            try:
                catalog = get_client_tool_catalog(device_id, user_id or "")
                catalog_version = catalog.catalog_version
            except Exception:
                pass

        with self._lock:
            if client_key not in self._client_tool_scopes:
                self._client_tool_scopes[client_key] = ClientToolScope()

            scope = self._client_tool_scopes[client_key]
            ttl = settings.mcp_tool_search_loaded_tools_ttl_minutes
            scope.cleanup_expired(ttl)

            for ref in references:
                ref_catalog_version = getattr(ref, "catalog_version", None)
                result = scope.add(
                    tool_name=ref.tool_name,
                    server_name=ref.server_name,
                    device_id=ref.device_id or device_id or "",
                    catalog_version=(
                        int(ref_catalog_version)
                        if ref_catalog_version is not None
                        else catalog_version
                    ),
                    max_tools=max_tools,
                    tool_instance_id=getattr(ref, "tool_instance_id", ""),
                )
                if result:
                    loaded.append(ref)

        return loaded

    def get_loaded(
        self,
        conversation_id: str | None,
        agent_key: str | None,
    ) -> list[ToolReference]:
        """
        Get all loaded tools for a conversation.

        This is called when binding tools to the model to include
        previously loaded deferred tools.

        Args:
            conversation_id: The conversation to get tools for
            agent_key: The agent key

        Returns:
            List of ToolReferences for loaded tools
        """
        key = self._get_key(conversation_id, agent_key)
        generation = get_mcp_tools_generation()
        ttl = settings.mcp_tool_search_loaded_tools_ttl_minutes

        with self._lock:
            tool_set = self._conversation_tools.get(key)
            if not tool_set:
                return []

            # Clean up expired tools
            tool_set.cleanup_expired(ttl, generation)

            return tool_set.list_tools()

    def get_loaded_client_tools(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        device_id: str | None = None,
        session_id: str | None = None,
    ) -> list[LoadedClientTool]:
        """
        Get all loaded client tools for a conversation and execution scope.

        When device_id and session_id are provided, returns tools from
        that specific execution scope only.
        """
        ttl = settings.mcp_tool_search_loaded_tools_ttl_minutes

        with self._lock:
            if device_id and not session_id:
                session_id = self._resolve_active_session_id(device_id)
            if device_id and session_id:
                client_key = self._get_client_key(
                    conversation_id,
                    agent_key,
                    device_id,
                    session_id,
                )
                scope = self._client_tool_scopes.get(client_key)
                if not scope:
                    return []
                scope.cleanup_expired(ttl)
                return scope.list_tools()

            conv_id = conversation_id or ""
            agent = agent_key or "default"
            result: list[LoadedClientTool] = []
            for key, scope in list(self._client_tool_scopes.items()):
                if key[0] == conv_id and key[1] == agent:
                    if device_id and key[2] != str(device_id):
                        continue
                    scope.cleanup_expired(ttl)
                    result.extend(scope.list_tools())
            return result

    def get_all_loaded_tool_names(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        device_id: str | None = None,
        session_id: str | None = None,
    ) -> list[str]:
        """Get names of all loaded tools (server + client) for a conversation."""
        key = self._get_key(conversation_id, agent_key)
        generation = get_mcp_tools_generation()
        ttl = settings.mcp_tool_search_loaded_tools_ttl_minutes

        with self._lock:
            names: list[str] = []
            tool_set = self._conversation_tools.get(key)
            if tool_set:
                tool_set.cleanup_expired(ttl, generation)
                names.extend(tool_set.list_all_tool_names())

            conv_id = conversation_id or ""
            agent = agent_key or "default"
            if device_id and not session_id:
                session_id = self._resolve_active_session_id(device_id)
            for ckey, scope in list(self._client_tool_scopes.items()):
                if ckey[0] == conv_id and ckey[1] == agent:
                    if device_id and ckey[2] != str(device_id):
                        continue
                    if session_id and ckey[3] != str(session_id):
                        continue
                    scope.cleanup_expired(ttl)
                    names.extend(name for name in scope.loaded)
            return names

    def is_loaded(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        tool_name: str,
    ) -> bool:
        """
        Check if a specific tool is loaded for a conversation.

        Args:
            conversation_id: The conversation to check
            agent_key: The agent key
            tool_name: The tool name to check

        Returns:
            True if the tool is loaded, False otherwise
        """
        key = self._get_key(conversation_id, agent_key)

        with self._lock:
            tool_set = self._conversation_tools.get(key)
            if not tool_set:
                return False
            return tool_name in tool_set

    def mark_tool_used(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        tool_name: str,
    ) -> bool:
        """
        Mark a tool as recently used, updating its LRU timestamp.

        This should be called when a tool is actually executed to ensure
        frequently-used tools are not evicted.

        Args:
            conversation_id: The conversation
            agent_key: The agent key
            tool_name: The tool name that was executed

        Returns:
            True if the tool was found and marked, False otherwise
        """
        key = self._get_key(conversation_id, agent_key)

        with self._lock:
            tool_set = self._conversation_tools.get(key)
            if not tool_set:
                return False
            tool = tool_set.get(tool_name)  # get() calls touch()
            return tool is not None

    def get_server_for_loaded_tool(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        tool_name: str,
    ) -> str | None:
        """
        Get the server name for a loaded tool.

        Args:
            conversation_id: The conversation
            agent_key: The agent key
            tool_name: The tool name

        Returns:
            Server name if tool is loaded, None otherwise
        """
        key = self._get_key(conversation_id, agent_key)

        with self._lock:
            tool_set = self._conversation_tools.get(key)
            if not tool_set:
                return None
            tool = tool_set.get(tool_name)
            return tool.server_name if tool else None

    def clear_conversation(
        self,
        conversation_id: str | None,
        agent_key: str | None = None,
    ) -> int:
        """Clear loaded tools for a conversation (both server and client scopes)."""
        cleared = 0
        with self._lock:
            if agent_key:
                key = self._get_key(conversation_id, agent_key)
                if key in self._conversation_tools:
                    del self._conversation_tools[key]
                    cleared = 1
                client_keys = [
                    k
                    for k in self._client_tool_scopes
                    if k[0] == (conversation_id or "") and k[1] == (agent_key or "default")
                ]
                for k in client_keys:
                    del self._client_tool_scopes[k]
                    cleared += 1
            else:
                conv_id = conversation_id or ""
                keys_to_remove = [k for k in self._conversation_tools if k[0] == conv_id]
                for key in keys_to_remove:
                    del self._conversation_tools[key]
                cleared = len(keys_to_remove)
                client_keys = [k for k in self._client_tool_scopes if k[0] == conv_id]
                for k in client_keys:
                    del self._client_tool_scopes[k]
                    cleared += 1

        if cleared:
            logger.debug(
                "Cleared %d tool set(s) for conversation=%s agent=%s",
                cleared,
                conversation_id,
                agent_key,
            )
        return cleared

    def clear_all(self) -> int:
        """Clear all loaded tools (for testing/reset)."""
        with self._lock:
            count = len(self._conversation_tools) + len(self._client_tool_scopes)
            self._conversation_tools.clear()
            self._client_tool_scopes.clear()
        return count

    def get_stats(self) -> dict:
        """Get statistics about the deferred tool state."""
        with self._lock:
            server_tools = sum(len(ts) for ts in self._conversation_tools.values())
            client_tools = sum(len(s) for s in self._client_tool_scopes.values())
            return {
                "conversation_count": len(self._conversation_tools),
                "client_scope_count": len(self._client_tool_scopes),
                "total_server_tools": server_tools,
                "total_client_tools": client_tools,
                "total_loaded_tools": server_tools + client_tools,
            }


# Module-level singleton
_state_instance: DeferredToolState | None = None


def get_deferred_tool_state() -> DeferredToolState:
    """
    Get the global DeferredToolState singleton.

    Returns:
        The DeferredToolState instance
    """
    global _state_instance
    if _state_instance is None:
        _state_instance = DeferredToolState()
    return _state_instance


def reset_deferred_tool_state() -> None:
    """Reset the global state (for testing)."""
    global _state_instance
    if _state_instance:
        _state_instance.clear_all()
    _state_instance = None
