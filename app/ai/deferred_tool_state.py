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
        catalog_version: The client catalog version when loaded
    """

    tool_name: str
    server_name: str
    device_id: str
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    catalog_version: int = 0

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
class ConversationToolSet:
    """
    Set of tools loaded for a specific (conversation_id, agent_key) pair.

    Manages:
    - The set of currently loaded server MCP tools
    - The set of currently loaded client device tools
    - LRU ordering for eviction
    - Capacity limits
    """

    # Loaded server tools keyed by tool_name
    # Only one server per tool_name at a time (replacement semantics)
    loaded: dict[str, LoadedTool] = field(default_factory=dict)

    # Loaded client tools keyed by tool_name
    loaded_client: dict[str, LoadedClientTool] = field(default_factory=dict)

    # When this set was created
    created_at: float = field(default_factory=time.time)

    def add(
        self,
        tool_name: str,
        server_name: str,
        generation: int,
        max_tools: int,
    ) -> LoadedTool | None:
        """
        Add or replace a tool in the loaded set.

        If the tool_name already exists, it's replaced with the new server.
        If capacity is exceeded, the least recently used tool is evicted.

        Args:
            tool_name: Name of the tool to load
            server_name: Server providing the tool
            generation: Current MCP tools generation
            max_tools: Maximum number of tools allowed

        Returns:
            The LoadedTool if added successfully, None if eviction failed
        """
        # If tool already exists, replace it (update server binding)
        if tool_name in self.loaded:
            existing = self.loaded[tool_name]
            if existing.server_name != server_name:
                logger.debug(
                    "Replacing tool '%s' binding: %s -> %s",
                    tool_name,
                    existing.server_name,
                    server_name,
                )
            existing.server_name = server_name
            existing.generation = generation
            existing.touch()
            return existing

        # Check capacity and evict if needed
        while len(self.loaded) >= max_tools:
            evicted = self._evict_lru()
            if evicted:
                logger.debug(
                    "Evicted LRU tool '%s' from %s to make room",
                    evicted.tool_name,
                    evicted.server_name,
                )
            else:
                # Couldn't evict (shouldn't happen)
                logger.warning("Could not evict tool to make room for '%s'", tool_name)
                return None

        # Add new tool
        loaded_tool = LoadedTool(
            tool_name=tool_name,
            server_name=server_name,
            generation=generation,
        )
        self.loaded[tool_name] = loaded_tool
        return loaded_tool

    def add_client_tool(
        self,
        tool_name: str,
        server_name: str,
        device_id: str,
        catalog_version: int,
        max_tools: int,
    ) -> LoadedClientTool | None:
        """
        Add or replace a client tool in the loaded set.

        Args:
            tool_name: Name of the tool to load (with client__ prefix)
            server_name: Local MCP server name (or "native")
            device_id: Device the tool belongs to
            catalog_version: Client catalog version when loaded
            max_tools: Maximum number of tools allowed

        Returns:
            The LoadedClientTool if added successfully, None if eviction failed
        """
        # If tool already exists, update it
        if tool_name in self.loaded_client:
            existing = self.loaded_client[tool_name]
            existing.server_name = server_name
            existing.device_id = device_id
            existing.catalog_version = catalog_version
            existing.touch()
            return existing

        # Check capacity and evict if needed (evict from both pools)
        total_count = len(self.loaded) + len(self.loaded_client)
        while total_count >= max_tools:
            evicted = self._evict_lru_any()
            if evicted:
                logger.debug(
                    "Evicted LRU tool '%s' to make room",
                    evicted,
                )
                total_count -= 1
            else:
                logger.warning("Could not evict tool to make room for '%s'", tool_name)
                return None

        # Add new client tool
        loaded_tool = LoadedClientTool(
            tool_name=tool_name,
            server_name=server_name,
            device_id=device_id,
            catalog_version=catalog_version,
        )
        self.loaded_client[tool_name] = loaded_tool
        return loaded_tool

    def get(self, tool_name: str) -> LoadedTool | LoadedClientTool | None:
        """
        Get a loaded tool by name, updating its LRU timestamp.

        Checks both server and client tool pools.

        Args:
            tool_name: Name of the tool to get

        Returns:
            LoadedTool or LoadedClientTool if found, None otherwise
        """
        # Check server tools first
        tool = self.loaded.get(tool_name)
        if tool:
            tool.touch()
            return tool

        # Check client tools
        client_tool = self.loaded_client.get(tool_name)
        if client_tool:
            client_tool.touch()
            return client_tool

        return None

    def get_client_tool(self, tool_name: str) -> LoadedClientTool | None:
        """
        Get a loaded client tool by name, updating its LRU timestamp.

        Args:
            tool_name: Name of the tool to get

        Returns:
            LoadedClientTool if found, None otherwise
        """
        tool = self.loaded_client.get(tool_name)
        if tool:
            tool.touch()
        return tool

    def remove(self, tool_name: str) -> LoadedTool | LoadedClientTool | None:
        """
        Remove a tool from the loaded set.

        Checks both server and client tool pools.

        Args:
            tool_name: Name of the tool to remove

        Returns:
            The removed tool, or None if not found
        """
        if tool_name in self.loaded:
            return self.loaded.pop(tool_name)
        if tool_name in self.loaded_client:
            return self.loaded_client.pop(tool_name)
        return None

    def _evict_lru(self) -> LoadedTool | None:
        """Evict and return the least recently used server tool."""
        if not self.loaded:
            return None

        # Find LRU tool
        lru_name = min(self.loaded.keys(), key=lambda k: self.loaded[k].last_used)
        return self.loaded.pop(lru_name)

    def _evict_lru_any(self) -> str | None:
        """Evict the least recently used tool from either pool. Returns tool name."""
        candidates: list[tuple[str, float, str]] = []  # (name, last_used, pool)

        for name, tool in self.loaded.items():
            candidates.append((name, tool.last_used, "server"))
        for name, tool in self.loaded_client.items():
            candidates.append((name, tool.last_used, "client"))

        if not candidates:
            return None

        # Find LRU across both pools
        lru = min(candidates, key=lambda x: x[1])
        name, _, pool = lru

        if pool == "server":
            self.loaded.pop(name)
        else:
            self.loaded_client.pop(name)

        return name

    def cleanup_expired(self, ttl_minutes: int, current_generation: int) -> int:
        """
        Remove expired and stale tools.

        Args:
            ttl_minutes: TTL in minutes (0 = no TTL)
            current_generation: Current MCP tools generation

        Returns:
            Number of tools removed
        """
        removed = 0

        # Clean up server tools
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
        removed += len(to_remove)

        # Clean up client tools (only TTL-based, no generation for client tools)
        to_remove_client = []
        for name, tool in self.loaded_client.items():
            if tool.is_expired(ttl_minutes):
                to_remove_client.append(name)
                logger.debug("Client tool '%s' expired (TTL)", name)

        for name in to_remove_client:
            del self.loaded_client[name]
        removed += len(to_remove_client)

        return removed

    def list_tools(self) -> list[ToolReference]:
        """Return list of all loaded server tool references."""
        return [tool.to_reference() for tool in self.loaded.values()]

    def list_client_tools(self) -> list[LoadedClientTool]:
        """Return list of all loaded client tools."""
        return list(self.loaded_client.values())

    def list_all_tool_names(self) -> list[str]:
        """Return list of all loaded tool names (server + client)."""
        return list(self.loaded.keys()) + list(self.loaded_client.keys())

    def __len__(self) -> int:
        return len(self.loaded) + len(self.loaded_client)

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self.loaded or tool_name in self.loaded_client


class DeferredToolState:
    """
    Global state manager for deferred tool loading across all conversations.

    Thread-safe for concurrent access.
    """

    def __init__(self):
        # Map from (conversation_id, agent_key) -> ConversationToolSet
        self._conversation_tools: dict[tuple[str, str], ConversationToolSet] = {}
        self._lock = Lock()

    def _get_key(
        self,
        conversation_id: str | None,
        agent_key: str | None,
    ) -> tuple[str, str]:
        """Generate a lookup key from conversation_id and agent_key."""
        return (conversation_id or "", agent_key or "default")

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

            # Load each reference
            for ref in references:
                result = tool_set.add(
                    tool_name=ref.tool_name,
                    server_name=ref.server_name,
                    generation=generation,
                    max_tools=max_tools,
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
        user_id: str | None = None,
        max_tools: int | None = None,
    ) -> list:
        """
        Load client device tools for a conversation, respecting capacity limits.

        This is called by tool_search to load client tools from search results.

        Args:
            conversation_id: The conversation to load tools for
            agent_key: The agent key (e.g., "chat", "rag")
            references: List of ClientToolReference to load
            device_id: The device these tools belong to
            max_tools: Override for max tools (uses config default if None)

        Returns:
            List of ClientToolReference that were actually loaded
        """
        from .client_tool_catalog import ClientToolReference, get_client_tool_catalog

        if max_tools is None:
            max_tools = settings.mcp_tool_search_max_loaded_tools_per_conversation

        key = self._get_key(conversation_id, agent_key)
        loaded: list[ClientToolReference] = []

        # Get current catalog version for the device
        catalog_version = 0
        if device_id:
            try:
                catalog = get_client_tool_catalog(device_id, user_id or "")
                catalog_version = catalog.catalog_version
            except Exception:
                pass

        generation = get_mcp_tools_generation()

        with self._lock:
            if key not in self._conversation_tools:
                self._conversation_tools[key] = ConversationToolSet()

            tool_set = self._conversation_tools[key]

            # Clean up expired tools first
            ttl = settings.mcp_tool_search_loaded_tools_ttl_minutes
            tool_set.cleanup_expired(ttl, generation)

            # Load each client tool reference
            for ref in references:
                result = tool_set.add_client_tool(
                    tool_name=ref.tool_name,
                    server_name=ref.server_name,
                    device_id=ref.device_id or device_id or "",
                    catalog_version=catalog_version,
                    max_tools=max_tools,
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
    ) -> list[LoadedClientTool]:
        """
        Get all loaded client tools for a conversation.

        Args:
            conversation_id: The conversation to get tools for
            agent_key: The agent key

        Returns:
            List of LoadedClientTool for loaded client tools
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

            return tool_set.list_client_tools()

    def get_all_loaded_tool_names(
        self,
        conversation_id: str | None,
        agent_key: str | None,
    ) -> list[str]:
        """
        Get names of all loaded tools (server + client) for a conversation.

        Args:
            conversation_id: The conversation to get tools for
            agent_key: The agent key

        Returns:
            List of tool names (strings)
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

            return tool_set.list_all_tool_names()

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
        """
        Clear loaded tools for a conversation.

        Args:
            conversation_id: The conversation to clear
            agent_key: Optional agent key (clears all agents if None)

        Returns:
            Number of tool sets cleared
        """
        cleared = 0

        with self._lock:
            if agent_key:
                # Clear specific agent
                key = self._get_key(conversation_id, agent_key)
                if key in self._conversation_tools:
                    del self._conversation_tools[key]
                    cleared = 1
            else:
                # Clear all agents for conversation
                conv_id = conversation_id or ""
                keys_to_remove = [k for k in self._conversation_tools if k[0] == conv_id]
                for key in keys_to_remove:
                    del self._conversation_tools[key]
                cleared = len(keys_to_remove)

        if cleared:
            logger.debug(
                "Cleared %d tool set(s) for conversation=%s agent=%s",
                cleared,
                conversation_id,
                agent_key,
            )

        return cleared

    def clear_all(self) -> int:
        """
        Clear all loaded tools (for testing/reset).

        Returns:
            Number of conversations cleared
        """
        with self._lock:
            count = len(self._conversation_tools)
            self._conversation_tools.clear()
        return count

    def get_stats(self) -> dict:
        """
        Get statistics about the deferred tool state.

        Returns:
            Dict with stats about loaded tools
        """
        with self._lock:
            total_tools = sum(len(ts) for ts in self._conversation_tools.values())
            return {
                "conversation_count": len(self._conversation_tools),
                "total_loaded_tools": total_tools,
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
