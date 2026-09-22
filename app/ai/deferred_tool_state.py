"""
Deferred Tool State - per-conversation tracking of tools loaded via tool_search.

Tools are discovered on demand by ``tool_search`` and then stay bound for the
rest of the conversation instead of being re-discovered every turn.

Two scopes are tracked:

``(conversation_id, agent_key)``
    Server MCP tools, keyed by the public *call name* so that an ambiguous
    same-name tool stays addressable under its deterministic alias.

``(conversation_id, agent_key, device_id, session_id)``
    Client device tools. Two sidecars attached to one conversation therefore
    get independent pools, independent LRU budgets, and cannot see each other.

Three bounds keep a long-lived process from growing without limit, and all
three are enforced here rather than merely documented:

* per-scope tool count - LRU eviction at
  ``mcp_tool_search_max_loaded_tools_per_conversation``
* per-tool idle TTL - ``mcp_tool_search_loaded_tools_ttl_minutes``, measured
  from last *use* so a tool still in use is not dropped mid-conversation
* process-wide scope count - idle scopes are swept periodically and the least
  recently active scope is evicted at ``mcp_tool_search_max_tracked_scopes``

Staleness is tracked per origin: server tools carry the MCP tools generation,
client tools carry the sidecar catalog version. Server entries are re-checked
on every read, client entries on the write and restore paths - a read of client
tools does not pay for a catalog lookup because binding intersects the loaded
names against the device's live tools anyway.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING
from uuid import UUID

from ..core.config import settings
from .mcp_registry import get_mcp_tools_generation
from .mcp_tool_catalog import ToolReference

if TYPE_CHECKING:
    from .client_tool_catalog import ClientToolReference

logger = logging.getLogger(__name__)

#: Identifies this process. Both the MCP tools generation and the client
#: catalog version are process-local counters that restart at zero, so a
#: snapshot restored into a *different* process cannot be validated against
#: them - comparing anyway would drop tools that are still perfectly live.
RUNTIME_ID = uuid.uuid4().hex

#: Lower bound between two full sweeps of the scope tables. Sweeping is done
#: opportunistically on the write path: a Celery beat task cannot reach this
#: state because it lives in the API process's memory.
_SWEEP_MIN_INTERVAL_SECONDS = 60.0


def _now() -> float:
    """Monotonic clock for TTL and LRU.

    Wall-clock time is not used: a system clock adjustment would otherwise
    expire every loaded tool at once, or keep expired ones forever.
    """
    return time.monotonic()


def _idle_minutes(since: float) -> float:
    return (_now() - since) / 60.0


def _is_idle(since: float, ttl_minutes: int) -> bool:
    if ttl_minutes <= 0:
        return False
    return _idle_minutes(since) > ttl_minutes


@dataclass
class LoadedTool:
    """A server MCP tool loaded into a conversation by ``tool_search``.

    Attributes:
        tool_name: Raw tool name as the MCP server exposes it.
        server_name: Server that provides the tool.
        call_name: Public invokable name. Equals ``tool_name`` for
            unambiguous tools, and the deterministic alias (``brave__search``)
            for same-name tools served by several servers.
        generation: MCP tools generation this tool was loaded from.
        loaded_at: When the tool was first loaded (diagnostics only).
        last_used: When the tool was last bound or executed; drives both LRU
            eviction and idle expiry.
    """

    tool_name: str
    server_name: str
    call_name: str = ""
    generation: int = 0
    loaded_at: float = field(default_factory=_now)
    last_used: float = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.call_name:
            self.call_name = self.tool_name

    def touch(self) -> None:
        """Reset the idle timer; called on every bind and every execution."""
        self.last_used = _now()

    def is_idle(self, ttl_minutes: int) -> bool:
        """True when the tool has gone unused for longer than the TTL."""
        return _is_idle(self.last_used, ttl_minutes)

    def is_stale(self, current_generation: int) -> bool:
        """True when the MCP tool set has changed since this tool was loaded."""
        return self.generation != current_generation

    def to_reference(self) -> ToolReference:
        """Convert to a ToolReference carrying the public call name."""
        return ToolReference(
            tool_name=self.tool_name,
            server_name=self.server_name,
            call_name=self.call_name,
        )


@dataclass
class LoadedClientTool:
    """A client device tool loaded into a conversation by ``tool_search``.

    Attributes:
        tool_name: Exposed name, always carrying the ``client__`` prefix.
        server_name: Local MCP server name on the device, or ``"native"``.
        device_id: Device the tool belongs to.
        catalog_version: Sidecar catalog version this tool was loaded from.
            Session-scoped, so it is only comparable within one scope.
        tool_instance_id: Opaque capability identifier minted by the sidecar;
            rotates whenever the catalog changes.
        loaded_at: When the tool was first loaded (diagnostics only).
        last_used: Drives LRU eviction and idle expiry, as for server tools.
    """

    tool_name: str
    server_name: str
    device_id: str
    catalog_version: int = 0
    tool_instance_id: str = ""
    loaded_at: float = field(default_factory=_now)
    last_used: float = field(default_factory=_now)

    def touch(self) -> None:
        """Reset the idle timer; called on every bind and every execution."""
        self.last_used = _now()

    def is_idle(self, ttl_minutes: int) -> bool:
        """True when the tool has gone unused for longer than the TTL."""
        return _is_idle(self.last_used, ttl_minutes)

    def is_stale(self, current_catalog_version: int) -> bool:
        """True when the sidecar re-published its catalog since the load.

        A non-positive version on either side means "unknown" - the sidecar
        had not reported one yet - and never counts as stale, so an
        unreachable catalog cannot silently unload a working tool.
        """
        if current_catalog_version <= 0 or self.catalog_version <= 0:
            return False
        return self.catalog_version != current_catalog_version


@dataclass
class _ToolScope:
    """Shared bookkeeping for one pool of loaded tools.

    ``last_active`` tracks reads as well as writes so that the process-wide
    scope sweep can tell an abandoned conversation from a quiet one.
    """

    created_at: float = field(default_factory=_now)
    last_active: float = field(default_factory=_now)

    def touch_scope(self) -> None:
        self.last_active = _now()

    def is_abandoned(self, ttl_minutes: int) -> bool:
        """True when the scope holds nothing and nobody has touched it."""
        if len(self):  # type: ignore[arg-type]
            return False
        if ttl_minutes <= 0:
            return True
        return _is_idle(self.last_active, ttl_minutes)


@dataclass
class ClientToolScope(_ToolScope):
    """Client tools loaded for one (conversation, agent, device, session)."""

    loaded: dict[str, LoadedClientTool] = field(default_factory=dict)

    def add(
        self,
        tool_name: str,
        server_name: str,
        device_id: str,
        catalog_version: int,
        max_tools: int,
        tool_instance_id: str = "",
    ) -> LoadedClientTool | None:
        """Add or refresh a client tool, evicting the LRU entry when full."""
        self.touch_scope()
        existing = self.loaded.get(tool_name)
        if existing is not None:
            existing.server_name = server_name
            existing.device_id = device_id
            existing.catalog_version = catalog_version
            existing.tool_instance_id = tool_instance_id
            existing.touch()
            return existing

        while len(self.loaded) >= max_tools:
            evicted = self._evict_lru()
            if evicted is None:
                logger.warning(
                    "Client tool '%s' not loaded: scope capacity is %d",
                    tool_name,
                    max_tools,
                )
                return None
            logger.debug("Evicted LRU client tool '%s'", evicted.tool_name)

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
        self.touch_scope()
        tool = self.loaded.get(tool_name)
        if tool:
            tool.touch()
        return tool

    def cleanup(self, ttl_minutes: int, current_catalog_version: int = 0) -> int:
        """Drop idle tools and tools left behind by a catalog re-publish."""
        to_remove: list[str] = []
        for name, tool in self.loaded.items():
            if tool.is_idle(ttl_minutes):
                to_remove.append(name)
                logger.debug("Client tool '%s' expired (idle TTL)", name)
            elif tool.is_stale(current_catalog_version):
                to_remove.append(name)
                logger.debug(
                    "Client tool '%s' stale (catalog %d -> %d)",
                    name,
                    tool.catalog_version,
                    current_catalog_version,
                )
        for name in to_remove:
            del self.loaded[name]
        return len(to_remove)

    def _evict_lru(self) -> LoadedClientTool | None:
        if not self.loaded:
            return None
        lru_name = min(self.loaded, key=lambda k: self.loaded[k].last_used)
        return self.loaded.pop(lru_name)

    def list_tools(self) -> list[LoadedClientTool]:
        self.touch_scope()
        return list(self.loaded.values())

    def __len__(self) -> int:
        return len(self.loaded)

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self.loaded


@dataclass
class ConversationToolSet(_ToolScope):
    """Server MCP tools loaded for one (conversation, agent) pair.

    Client device tools are *not* stored here; they live in per-device
    :class:`ClientToolScope` instances so one sidecar cannot see another's.
    """

    loaded: dict[str, LoadedTool] = field(default_factory=dict)

    def add(
        self,
        tool_name: str,
        server_name: str,
        generation: int,
        max_tools: int,
        call_name: str | None = None,
    ) -> LoadedTool | None:
        """Add or refresh a server tool, evicting the LRU entry when full."""
        self.touch_scope()
        storage_key = call_name or tool_name

        existing = self.loaded.get(storage_key)
        if existing is not None:
            if existing.server_name != server_name:
                logger.debug(
                    "Replacing tool '%s' binding: %s -> %s",
                    storage_key,
                    existing.server_name,
                    server_name,
                )
            existing.tool_name = tool_name
            existing.server_name = server_name
            existing.generation = generation
            existing.call_name = storage_key
            existing.touch()
            return existing

        while len(self.loaded) >= max_tools:
            evicted = self._evict_lru()
            if evicted is None:
                logger.warning(
                    "Tool '%s' not loaded: scope capacity is %d",
                    storage_key,
                    max_tools,
                )
                return None
            logger.debug(
                "Evicted LRU tool '%s' from %s to make room",
                evicted.call_name,
                evicted.server_name,
            )

        loaded_tool = LoadedTool(
            tool_name=tool_name,
            server_name=server_name,
            call_name=storage_key,
            generation=generation,
        )
        self.loaded[storage_key] = loaded_tool
        return loaded_tool

    def get(self, call_name: str) -> LoadedTool | None:
        self.touch_scope()
        tool = self.loaded.get(call_name)
        if tool:
            tool.touch()
        return tool

    def cleanup(self, ttl_minutes: int, current_generation: int) -> int:
        """Drop idle tools and tools left behind by an MCP reload."""
        to_remove: list[str] = []
        for name, tool in self.loaded.items():
            if tool.is_idle(ttl_minutes):
                to_remove.append(name)
                logger.debug("Tool '%s' expired (idle TTL)", name)
            elif tool.is_stale(current_generation):
                to_remove.append(name)
                logger.debug("Tool '%s' stale (generation mismatch)", name)
        for name in to_remove:
            del self.loaded[name]
        return len(to_remove)

    def _evict_lru(self) -> LoadedTool | None:
        if not self.loaded:
            return None
        lru_name = min(self.loaded, key=lambda k: self.loaded[k].last_used)
        return self.loaded.pop(lru_name)

    def list_tools(self) -> list[ToolReference]:
        self.touch_scope()
        return [tool.to_reference() for tool in self.loaded.values()]

    def list_all_tool_names(self) -> list[str]:
        self.touch_scope()
        return list(self.loaded)

    def __len__(self) -> int:
        return len(self.loaded)

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self.loaded


class DeferredToolState:
    """Process-wide registry of deferred tools, safe for concurrent access."""

    def __init__(self) -> None:
        self._conversation_tools: dict[tuple[str, str], ConversationToolSet] = {}
        self._client_tool_scopes: dict[tuple[str, str, str, str], ClientToolScope] = {}
        self._lock = Lock()
        self._last_sweep = _now()

    # ------------------------------------------------------------------
    # Keys and session resolution
    # ------------------------------------------------------------------

    def _get_key(
        self,
        conversation_id: str | None,
        agent_key: str | None,
    ) -> tuple[str, str]:
        """Scope key for server tools."""
        return (conversation_id or "", agent_key or "default")

    def _get_client_key(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        device_id: str | None,
        session_id: str | None,
    ) -> tuple[str, str, str, str]:
        """Scope key for client tools."""
        return (
            conversation_id or "",
            agent_key or "default",
            device_id or "",
            session_id or "",
        )

    @staticmethod
    def _resolve_active_session_id(
        device_id: str | None,
        user_id: str | None = None,
    ) -> str | None:
        """Resolve the runtime session a device is currently serving.

        When ``user_id`` is known the ownership-validated resolver is used and
        is the *only* path taken: it honours the client-runtime bridge flag and
        rejects a device that belongs to someone else. Falling back to the
        device-only lookup in that case would re-open exactly the cross-user
        hole the caller's own validation just closed.

        The unvalidated device-only lookup remains for callers that have no
        user in hand (snapshot restore for an internal scope, tests).
        """
        if not device_id:
            return None

        if user_id:
            try:
                from .client_runtime_tools import get_active_client_runtime_session

                session = get_active_client_runtime_session(
                    user_id=user_id,
                    device_id=device_id,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Validated session lookup failed for device=%s: %s",
                    device_id,
                    exc,
                )
                return None
            return session.session_id if session is not None else None

        from app.services.client_device_service import ClientDeviceService

        try:
            session = ClientDeviceService.lookup_active_session(UUID(str(device_id)))
        except Exception as exc:
            logger.debug("Session lookup failed for device=%s: %s", device_id, exc)
            return None
        return session.session_id if session is not None else None

    @staticmethod
    def _current_catalog_version(device_id: str | None, user_id: str | None) -> int:
        """Current sidecar catalog version, or 0 when it cannot be determined."""
        if not device_id:
            return 0
        from .client_tool_catalog import get_client_tool_catalog

        try:
            return int(get_client_tool_catalog(device_id, user_id or "").catalog_version)
        except Exception as exc:
            logger.debug("Client catalog version unavailable for %s: %s", device_id, exc)
            return 0

    # ------------------------------------------------------------------
    # Capacity maintenance
    # ------------------------------------------------------------------

    def _maybe_sweep_locked(self) -> None:
        """Rate-limited sweep of abandoned scopes. Caller holds the lock.

        Without this, every conversation ever seen would keep an entry alive
        for the lifetime of the process even after its tools expired.
        """
        now = _now()
        if now - self._last_sweep < _SWEEP_MIN_INTERVAL_SECONDS:
            return
        self._last_sweep = now

        ttl = settings.mcp_tool_search_loaded_tools_ttl_minutes
        generation = get_mcp_tools_generation()
        dropped = 0

        for key, tool_set in list(self._conversation_tools.items()):
            tool_set.cleanup(ttl, generation)
            if tool_set.is_abandoned(ttl):
                del self._conversation_tools[key]
                dropped += 1

        for client_key, scope in list(self._client_tool_scopes.items()):
            scope.cleanup(ttl)
            if scope.is_abandoned(ttl):
                del self._client_tool_scopes[client_key]
                dropped += 1

        if dropped:
            logger.debug("Swept %d abandoned deferred tool scope(s)", dropped)

    def _enforce_scope_cap_locked(self) -> None:
        """Bound the number of tracked scopes. Caller holds the lock.

        The sweep only reclaims scopes that have gone quiet; this is the hard
        ceiling that holds under sustained concurrency.
        """
        max_scopes = max(1, settings.mcp_tool_search_max_tracked_scopes)

        while len(self._conversation_tools) > max_scopes:
            key = min(
                self._conversation_tools,
                key=lambda k: self._conversation_tools[k].last_active,
            )
            del self._conversation_tools[key]
            logger.warning(
                "Deferred tool scope cap (%d) reached; dropped server scope %s",
                max_scopes,
                key,
            )

        while len(self._client_tool_scopes) > max_scopes:
            client_key = min(
                self._client_tool_scopes,
                key=lambda k: self._client_tool_scopes[k].last_active,
            )
            del self._client_tool_scopes[client_key]
            logger.warning(
                "Deferred tool scope cap (%d) reached; dropped client scope %s",
                max_scopes,
                client_key,
            )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def autoload(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        references: list[ToolReference],
        max_tools: int | None = None,
    ) -> list[ToolReference]:
        """Load server MCP tools for a conversation, respecting capacity.

        Called by ``tool_search`` with the results it recommends.

        Returns:
            The subset of ``references`` that was actually loaded.
        """
        if max_tools is None:
            max_tools = settings.mcp_tool_search_max_loaded_tools_per_conversation

        generation = get_mcp_tools_generation()
        key = self._get_key(conversation_id, agent_key)
        loaded: list[ToolReference] = []

        with self._lock:
            self._maybe_sweep_locked()

            tool_set = self._conversation_tools.get(key)
            if tool_set is None:
                tool_set = ConversationToolSet()
                self._conversation_tools[key] = tool_set
                self._enforce_scope_cap_locked()

            tool_set.cleanup(settings.mcp_tool_search_loaded_tools_ttl_minutes, generation)

            for ref in references:
                result = tool_set.add(
                    tool_name=ref.tool_name,
                    server_name=ref.server_name,
                    generation=generation,
                    max_tools=max_tools,
                    call_name=getattr(ref, "call_name", None),
                )
                if result:
                    loaded.append(ref)

        return loaded

    def autoload_client_tools(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        references: list["ClientToolReference"],
        device_id: str | None,
        session_id: str | None = None,
        user_id: str | None = None,
        max_tools: int | None = None,
    ) -> list["ClientToolReference"]:
        """Load client device tools into their own (device, session) scope.

        Two sidecars on one conversation therefore never share a pool, and
        neither can evict the other's tools.
        """
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
            effective_session_id = self._resolve_active_session_id(device_id, user_id)

        client_key = self._get_client_key(
            conversation_id,
            agent_key,
            device_id,
            effective_session_id,
        )
        catalog_version = self._current_catalog_version(device_id, user_id)
        loaded: list[ClientToolReference] = []

        with self._lock:
            self._maybe_sweep_locked()

            scope = self._client_tool_scopes.get(client_key)
            if scope is None:
                scope = ClientToolScope()
                self._client_tool_scopes[client_key] = scope
                self._enforce_scope_cap_locked()

            scope.cleanup(
                settings.mcp_tool_search_loaded_tools_ttl_minutes,
                catalog_version,
            )

            for ref in references:
                ref_catalog_version = getattr(ref, "catalog_version", None)
                result = scope.add(
                    tool_name=ref.tool_name,
                    server_name=ref.server_name,
                    device_id=getattr(ref, "device_id", None) or device_id,
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

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_loaded(
        self,
        conversation_id: str | None,
        agent_key: str | None,
    ) -> list[ToolReference]:
        """Server tools still loaded for a conversation, for model binding."""
        key = self._get_key(conversation_id, agent_key)
        generation = get_mcp_tools_generation()
        ttl = settings.mcp_tool_search_loaded_tools_ttl_minutes

        with self._lock:
            tool_set = self._conversation_tools.get(key)
            if not tool_set:
                return []
            tool_set.cleanup(ttl, generation)
            return tool_set.list_tools()

    def get_loaded_client_tools(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        device_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> list[LoadedClientTool]:
        """Client tools still loaded for one execution scope.

        Visibility is strictly bound to a single (device_id, session_id) pair.
        When ``session_id`` is omitted it is resolved from the device's active
        runtime session - ownership-validated when ``user_id`` is supplied. If
        no full scope can be established the lookup returns nothing: there is
        no cross-device fallback, so one client's tools can never surface on a
        turn belonging to another client, or to none.
        """
        ttl = settings.mcp_tool_search_loaded_tools_ttl_minutes

        if device_id and not session_id:
            session_id = self._resolve_active_session_id(device_id, user_id)
        if not device_id or not session_id:
            return []

        client_key = self._get_client_key(conversation_id, agent_key, device_id, session_id)

        with self._lock:
            scope = self._client_tool_scopes.get(client_key)
            if not scope:
                return []
            scope.cleanup(ttl)
            return scope.list_tools()

    def get_all_loaded_tool_names(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        device_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> list[str]:
        """Names of every loaded tool (server first, then client).

        Client names follow the same strict scope as
        :meth:`get_loaded_client_tools`: without a resolvable
        (device_id, session_id) pair only server names are returned.
        """
        key = self._get_key(conversation_id, agent_key)
        generation = get_mcp_tools_generation()
        ttl = settings.mcp_tool_search_loaded_tools_ttl_minutes

        if device_id and not session_id:
            session_id = self._resolve_active_session_id(device_id, user_id)

        with self._lock:
            names: list[str] = []
            tool_set = self._conversation_tools.get(key)
            if tool_set:
                tool_set.cleanup(ttl, generation)
                names.extend(tool_set.list_all_tool_names())

            if device_id and session_id:
                client_key = self._get_client_key(
                    conversation_id,
                    agent_key,
                    device_id,
                    session_id,
                )
                scope = self._client_tool_scopes.get(client_key)
                if scope:
                    scope.cleanup(ttl)
                    names.extend(scope.loaded)
            return names

    def mark_tool_used(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        tool_name: str,
    ) -> bool:
        """Reset a tool's idle timer after it executes.

        ``tool_name`` is the public call name. Returns False when the tool is
        not loaded for this conversation.
        """
        key = self._get_key(conversation_id, agent_key)

        with self._lock:
            tool_set = self._conversation_tools.get(key)
            if not tool_set:
                return False
            return tool_set.get(tool_name) is not None

    def get_server_for_loaded_tool(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        tool_name: str,
    ) -> str | None:
        """Server backing a loaded tool, keyed by public call name."""
        key = self._get_key(conversation_id, agent_key)

        with self._lock:
            tool_set = self._conversation_tools.get(key)
            if not tool_set:
                return None
            tool = tool_set.get(tool_name)
            return tool.server_name if tool else None

    def get_raw_tool_name_for_loaded_tool(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        tool_name: str,
    ) -> str | None:
        """Raw MCP tool name behind a public call name.

        Used during same-turn recovery to look the tool up on the live MCP
        manager, which knows it under its raw name rather than the alias.
        """
        key = self._get_key(conversation_id, agent_key)

        with self._lock:
            tool_set = self._conversation_tools.get(key)
            if not tool_set:
                return None
            tool = tool_set.get(tool_name)
            return tool.tool_name if tool else None

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

    def snapshot(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        device_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, object]:
        """Serialize the loaded tools for checkpoint persistence.

        ``runtime_id`` records which process produced the snapshot. The MCP
        generation and the catalog version are process-local counters that
        restart at zero, so :meth:`restore` can only compare them against a
        snapshot this same process wrote.
        """
        server_key = self._get_key(conversation_id, agent_key)
        generation = get_mcp_tools_generation()
        ttl = settings.mcp_tool_search_loaded_tools_ttl_minutes

        if device_id and not session_id:
            session_id = self._resolve_active_session_id(device_id, user_id)

        with self._lock:
            server_tools: list[dict[str, object]] = []
            tool_set = self._conversation_tools.get(server_key)
            if tool_set:
                tool_set.cleanup(ttl, generation)
                for tool in tool_set.loaded.values():
                    server_tools.append(
                        {
                            "tool_name": tool.tool_name,
                            "server_name": tool.server_name,
                            "call_name": tool.call_name,
                            "generation": tool.generation,
                        }
                    )

            # Client tools are serialized only for the turn's own
            # (device, session) scope; a turn without a connected client
            # snapshots no client tools (FR-1).
            client_tools: list[dict[str, object]] = []
            if device_id and session_id:
                client_key = self._get_client_key(
                    conversation_id,
                    agent_key,
                    device_id,
                    session_id,
                )
                scope = self._client_tool_scopes.get(client_key)
                if scope:
                    scope.cleanup(ttl)
                    for client_tool in scope.list_tools():
                        client_tools.append(
                            {
                                "tool_name": client_tool.tool_name,
                                "server_name": client_tool.server_name,
                                "device_id": client_tool.device_id,
                                "session_id": str(session_id),
                                "catalog_version": client_tool.catalog_version,
                                "tool_instance_id": client_tool.tool_instance_id,
                            }
                        )

        return {
            "runtime_id": RUNTIME_ID,
            "server_tools": server_tools,
            "client_tools": client_tools,
        }

    def restore(
        self,
        conversation_id: str | None,
        agent_key: str | None,
        snapshot: dict[str, object] | None,
        device_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, int]:
        """Restore a checkpoint snapshot into the live in-memory state.

        Entries the snapshot itself marks as outdated are dropped rather than
        re-stamped as fresh: a snapshot written by *this* process is checked
        against the current MCP generation and client catalog version, and
        client entries from a superseded runtime session are discarded because
        their capability identifiers have since rotated. A snapshot from
        another process cannot be checked that way and is restored as-is;
        binding tolerates a tool that is no longer there.
        """
        if not isinstance(snapshot, dict):
            return {"server_tools": 0, "client_tools": 0}

        same_runtime = snapshot.get("runtime_id") == RUNTIME_ID
        generation = get_mcp_tools_generation()

        server_refs: list[ToolReference] = []
        for entry in snapshot.get("server_tools") or []:
            if not isinstance(entry, dict):
                continue
            tool_name = str(entry.get("tool_name") or "").strip()
            server_name = str(entry.get("server_name") or "").strip()
            if not tool_name or not server_name:
                continue
            if same_runtime and int(entry.get("generation") or 0) != generation:
                logger.debug(
                    "Dropping stale snapshot tool '%s' (generation %s != %d)",
                    tool_name,
                    entry.get("generation"),
                    generation,
                )
                continue
            call_name_value = entry.get("call_name")
            server_refs.append(
                ToolReference(
                    tool_name=tool_name,
                    server_name=server_name,
                    call_name=str(call_name_value).strip() if call_name_value else None,
                )
            )

        restored_server = self.autoload(
            conversation_id=conversation_id,
            agent_key=agent_key,
            references=server_refs,
        )

        restored_client: list[ClientToolReference] = []
        if device_id:
            restored_client = self._restore_client_tools(
                conversation_id=conversation_id,
                agent_key=agent_key,
                entries=snapshot.get("client_tools") or [],
                device_id=device_id,
                session_id=session_id,
                user_id=user_id,
                same_runtime=same_runtime,
            )

        return {
            "server_tools": len(restored_server),
            "client_tools": len(restored_client),
        }

    def _restore_client_tools(
        self,
        *,
        conversation_id: str | None,
        agent_key: str | None,
        entries: object,
        device_id: str,
        session_id: str | None,
        user_id: str | None,
        same_runtime: bool,
    ) -> list["ClientToolReference"]:
        """Restore the client half of a snapshot into the turn's own scope.

        Entries belonging to another device are dropped, as are entries from a
        superseded runtime session: the sidecar mints a fresh
        ``tool_instance_id`` on every reconnect, so carrying the old one over
        would put an uninvokable capability into the new session's budget.
        """
        from .client_tool_catalog import ClientToolReference

        if not isinstance(entries, list):
            return []

        effective_session_id = session_id or self._resolve_active_session_id(device_id, user_id)
        catalog_version = (
            self._current_catalog_version(device_id, user_id) if same_runtime else 0
        )

        client_refs: list[ClientToolReference] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            tool_name = str(entry.get("tool_name") or "").strip()
            server_name = str(entry.get("server_name") or "").strip()
            if not tool_name or not server_name:
                continue

            entry_device_id = str(entry.get("device_id") or device_id).strip()
            if entry_device_id != str(device_id):
                continue

            entry_session_id = str(entry.get("session_id") or "").strip()
            superseded = (
                bool(effective_session_id)
                and bool(entry_session_id)
                and entry_session_id != effective_session_id
            )
            if superseded:
                logger.debug(
                    "Dropping snapshot client tool '%s' from superseded session %s",
                    tool_name,
                    entry_session_id,
                )
                continue

            entry_catalog_version = int(entry.get("catalog_version") or 0)
            if (
                catalog_version > 0
                and entry_catalog_version > 0
                and entry_catalog_version != catalog_version
            ):
                logger.debug(
                    "Dropping snapshot client tool '%s' (catalog %d != %d)",
                    tool_name,
                    entry_catalog_version,
                    catalog_version,
                )
                continue

            client_refs.append(
                ClientToolReference(
                    tool_name=tool_name,
                    server_name=server_name,
                    device_id=entry_device_id,
                    session_id=entry_session_id or (effective_session_id or ""),
                    catalog_version=entry_catalog_version,
                    tool_instance_id=str(entry.get("tool_instance_id") or ""),
                )
            )

        if not client_refs:
            return []

        return self.autoload_client_tools(
            conversation_id=conversation_id,
            agent_key=agent_key,
            references=client_refs,
            device_id=device_id,
            session_id=effective_session_id,
            user_id=user_id,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear_conversation(
        self,
        conversation_id: str | None,
        agent_key: str | None = None,
    ) -> int:
        """Drop every scope for a conversation, or for one of its agents.

        Called when a conversation is deleted so its tool state dies with it
        instead of waiting out the TTL.

        Returns:
            Number of scopes removed (server and client combined).
        """
        conv_id = conversation_id or ""
        cleared = 0

        with self._lock:
            if agent_key:
                key = self._get_key(conversation_id, agent_key)
                if self._conversation_tools.pop(key, None) is not None:
                    cleared += 1
                client_keys = [
                    k
                    for k in self._client_tool_scopes
                    if k[0] == conv_id and k[1] == (agent_key or "default")
                ]
            else:
                server_keys = [k for k in self._conversation_tools if k[0] == conv_id]
                for key in server_keys:
                    del self._conversation_tools[key]
                cleared += len(server_keys)
                client_keys = [k for k in self._client_tool_scopes if k[0] == conv_id]

            for client_key in client_keys:
                del self._client_tool_scopes[client_key]
            cleared += len(client_keys)

        if cleared:
            logger.debug(
                "Cleared %d tool scope(s) for conversation=%s agent=%s",
                cleared,
                conversation_id,
                agent_key,
            )
        return cleared

    def clear_all(self) -> int:
        """Drop every scope. Used by tests and by a full MCP reset."""
        with self._lock:
            count = len(self._conversation_tools) + len(self._client_tool_scopes)
            self._conversation_tools.clear()
            self._client_tool_scopes.clear()
            self._last_sweep = _now()
        return count

    def get_stats(self) -> dict[str, int]:
        """Counters for diagnostics and capacity monitoring."""
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


_state_instance: DeferredToolState | None = None
_state_lock = Lock()


def get_deferred_tool_state() -> DeferredToolState:
    """Return the process-wide DeferredToolState singleton."""
    global _state_instance
    if _state_instance is None:
        with _state_lock:
            if _state_instance is None:
                _state_instance = DeferredToolState()
    return _state_instance


def reset_deferred_tool_state() -> None:
    """Drop the singleton and everything it holds (tests)."""
    global _state_instance
    with _state_lock:
        if _state_instance is not None:
            _state_instance.clear_all()
        _state_instance = None
