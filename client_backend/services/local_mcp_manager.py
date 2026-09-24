"""Device-scoped local MCP runtime backed exclusively by schema-v2 storage.

Each enabled server runs as one long-lived session for as long as the manager
is initialized. A server keeps state between calls -- Desktop Commander tracks
the processes it started so a later call can read or stop them -- and a
per-call session would discard that state along with the server process.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import anyio
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED

from app.core.mcp_adapter_utils import (
    build_mcp_server_entry,
    clean_mcp_tool_name,
    sanitize_mcp_schema,
)
from client_backend.core.logging import get_logger
from client_backend.core.security import generate_device_identifier
from client_backend.schemas.mcp_config import MCPProfileScope
from client_backend.services.desktop_commander_policy import (
    harden_launch,
    is_desktop_commander,
    is_hidden_tool,
    is_mutating_tool,
    refuse_sensitive_paths,
)
from client_backend.services.mcp_config_migration import prepare_mcp_config_store
from client_backend.services.mcp_config_store import EffectiveMCPServer, MCPConfigStore
from client_backend.services.upstream_auth import get_upstream_auth_service

logger = get_logger(__name__)


@dataclass
class MCPTool:
    """A sanitized MCP tool definition."""

    name: str
    description: str
    server_name: str
    input_schema: dict[str, Any]
    # Mutations are gated for human approval by the server.
    mutation: bool = False

    @property
    def qualified_id(self) -> str:
        return f"{self.server_name}::{self.name}"


class MCPServerRuntime:
    """Tracks the runtime state of one effective MCP server."""

    def __init__(self, config: EffectiveMCPServer):
        self.config = config
        self.tools: list[MCPTool] = []
        self.started_at: datetime | None = None
        self.error_message: str | None = None
        self._running = False

    def mark_running(self, tools: list[MCPTool]) -> None:
        self.tools = tools
        self.started_at = datetime.now(timezone.utc)
        self.error_message = None
        self._running = True

    def mark_error(self, message: str) -> None:
        self.tools = []
        self.started_at = None
        self.error_message = message
        self._running = False

    def mark_stopped(self) -> None:
        self.tools = []
        self.started_at = None
        self.error_message = None
        self._running = False

    def is_running(self) -> bool:
        return self._running


def _terminal_exception_message(exc: BaseException) -> str:
    """Return the useful leaf cause hidden by task groups and wrappers."""

    current: BaseException = exc
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        grouped = getattr(current, "exceptions", None)
        if isinstance(grouped, tuple) and grouped:
            current = grouped[0]
            continue
        cause = current.__cause__ or current.__context__
        if cause is None:
            break
        current = cause
    message = str(current).strip()
    return message or current.__class__.__name__


# Sending on a closed transport fails before the request leaves this process,
# so the server never saw it and a fresh session may carry it instead.
_UNSENT_REQUEST_ERRORS = (anyio.ClosedResourceError, anyio.BrokenResourceError)

_SESSION_STOP_TIMEOUT_SECONDS = 10.0


class _ServerSession:
    """One long-lived MCP session, held open by the task that opened it.

    anyio requires a transport's cancel scope to be exited by the task that
    entered it, so a dedicated task owns the ``async with`` and callers only
    send requests through the tools it loaded. Closing the session ends the
    server process and, through its kill-on-close job, everything it started.
    """

    def __init__(self, client: MultiServerMCPClient, server_name: str):
        self._client = client
        self._server_name = server_name
        self._stop_requested = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._opened = False
        self.tools: dict[str, BaseTool] = {}
        self.alive = False

    async def start(self) -> list[BaseTool]:
        """Open the session and return the tools the server advertises."""

        ready: asyncio.Future[list[BaseTool]] = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(
            self._hold_open(ready),
            name=f"mcp-session:{self._server_name}",
        )
        self._task.add_done_callback(self._log_unexpected_exit)
        try:
            loaded = await ready
        except BaseException:
            await self.stop()
            raise
        self.alive = True
        return loaded

    async def stop(self) -> None:
        """Close the session, cancelling it if the server will not exit."""

        self.alive = False
        self._stop_requested.set()
        task = self._task
        if task is None:
            return
        done, _ = await asyncio.wait({task}, timeout=_SESSION_STOP_TIMEOUT_SECONDS)
        if not done:
            logger.warning(
                "MCP server '%s' did not close within %.0fs; cancelling its session",
                self._server_name,
                _SESSION_STOP_TIMEOUT_SECONDS,
            )
            task.cancel()
            await asyncio.wait({task})

    async def _hold_open(self, ready: asyncio.Future[list[BaseTool]]) -> None:
        try:
            async with self._client.session(self._server_name) as session:
                ready.set_result(list(await load_mcp_tools(session)))
                self._opened = True
                await self._stop_requested.wait()
        except BaseException as exc:
            if not ready.done():
                if isinstance(exc, asyncio.CancelledError):
                    ready.cancel()
                else:
                    ready.set_exception(exc)
            raise
        finally:
            self.alive = False

    def _log_unexpected_exit(self, task: asyncio.Task[None]) -> None:
        # A failure before the session opened is raised to ``start``'s caller.
        if task.cancelled() or task.exception() is None or not self._opened:
            return
        logger.warning(
            "MCP server '%s' session ended with an error: %s",
            self._server_name,
            _terminal_exception_message(task.exception()),
        )


class LocalMCPManager:
    """Runs MCP servers for exactly one authenticated user/device scope."""

    def __init__(self, *, store: MCPConfigStore):
        self.store = store
        self.scope = store.scope
        self.servers: dict[str, MCPServerRuntime] = {}
        self._client: MultiServerMCPClient | None = None
        self._sessions: dict[str, _ServerSession] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Two overlapping initializations would each start every server and
        # the loser's processes would outlive the manager that forgot them.
        self._init_lock = asyncio.Lock()
        self._initialized = False

    @property
    def config_path(self):
        """Return the canonical v2 profile path for diagnostics."""

        return self.store.profile_path

    async def initialize(self, server_names: set[str] | None = None) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            await self._initialize_locked(server_names)

    async def _initialize_locked(self, server_names: set[str] | None) -> None:
        configs = [
            config
            for config in self.store.list_effective_servers()
            if config.enabled and (server_names is None or config.name in server_names)
        ]
        self.servers = {config.name: MCPServerRuntime(config) for config in configs}
        server_config = self._build_server_config(configs)
        logger.info(
            "Initializing MCP manager for user=%s device=%s with %d enabled servers",
            self.scope.user_id,
            self.scope.device_identifier,
            len(configs),
        )

        if not server_config:
            self._initialized = True
            return

        try:
            self._client = MultiServerMCPClient(server_config)
        except Exception as exc:
            message = _terminal_exception_message(exc)
            logger.error("Failed to initialize MCP client: %s", message, exc_info=True)
            for runtime in self.servers.values():
                runtime.mark_error(message)
            self._initialized = True
            return

        for config in configs:
            await self._start_server(config.name)

        self._initialized = True
        logger.info(
            "MCP manager initialized with %d configured servers and %d tools",
            len(self.servers),
            len(self.get_all_tools()),
        )

    @staticmethod
    def _build_server_config(
        configs: list[EffectiveMCPServer],
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for config in configs:
            args, env = config.args, config.env
            if is_desktop_commander(config.command, args):
                args, env = harden_launch(args, env)
            entry = build_mcp_server_entry(
                transport=config.transport,
                command=config.command,
                args=args,
                cwd=config.cwd,
                env=env,
                url=config.url,
                headers=config.headers,
            )
            if entry is not None:
                result[config.name] = entry
        return result

    async def _start_server(self, server_name: str) -> None:
        """Start one server during initialization, recording rather than raising a failure."""

        try:
            await self._live_session(server_name)
        except Exception as exc:
            logger.error(
                "Failed to start MCP server '%s': %s",
                server_name,
                _terminal_exception_message(exc),
                exc_info=True,
            )

    async def _live_session(self, server_name: str) -> _ServerSession:
        """Return the server's open session, starting a new one if it has died."""

        lock = self._session_locks.setdefault(server_name, asyncio.Lock())
        async with lock:
            session = self._sessions.get(server_name)
            if session is not None and session.alive:
                return session
            if session is not None:
                await session.stop()
                self._sessions.pop(server_name, None)
            return await self._open_session(server_name)

    async def _open_session(self, server_name: str) -> _ServerSession:
        runtime = self.servers[server_name]
        if self._client is None:
            raise RuntimeError(runtime.error_message or "MCP client is not initialized")

        session = _ServerSession(self._client, server_name)
        try:
            loaded_tools = await session.start()
        except Exception as exc:
            message = _terminal_exception_message(exc)
            runtime.mark_error(message)
            raise RuntimeError(f"MCP server '{server_name}' failed to start: {message}") from exc

        desktop_commander = is_desktop_commander(runtime.config.command, runtime.config.args)
        named_tools = {
            clean_mcp_tool_name(str(getattr(tool, "name", "") or ""), server_name=server_name): tool
            for tool in loaded_tools
        }
        session.tools = {
            name: tool
            for name, tool in named_tools.items()
            if not (desktop_commander and is_hidden_tool(name))
        }
        records = self._tool_records_from_loaded_tools(
            server_name,
            list(session.tools.values()),
            desktop_commander=desktop_commander,
        )
        runtime.mark_running(records)
        self._sessions[server_name] = session
        logger.info("Started MCP server '%s' with %d tools", server_name, len(records))
        return session

    @staticmethod
    def _tool_records_from_loaded_tools(
        server_name: str,
        loaded_tools: list[Any],
        *,
        desktop_commander: bool = False,
    ) -> list[MCPTool]:
        records = []
        for tool in loaded_tools:
            name = clean_mcp_tool_name(
                str(getattr(tool, "name", "") or "unknown"),
                server_name=server_name,
            )
            records.append(
                MCPTool(
                    name=name,
                    description=str(getattr(tool, "description", "") or ""),
                    server_name=server_name,
                    input_schema=sanitize_mcp_schema(getattr(tool, "args_schema", None)),
                    mutation=desktop_commander and is_mutating_tool(name),
                )
            )
        return records

    async def shutdown(self) -> None:
        sessions = list(self._sessions.values())
        self._sessions.clear()
        await asyncio.gather(*(session.stop() for session in sessions))
        for runtime in self.servers.values():
            runtime.mark_stopped()
        self.servers.clear()
        self._client = None
        self._initialized = False

    def get_all_tools(self) -> list[MCPTool]:
        return [
            tool
            for runtime in self.servers.values()
            if runtime.is_running()
            for tool in runtime.tools
        ]

    def get_tools_by_server(self, server_name: str) -> list[MCPTool]:
        runtime = self.servers.get(server_name)
        return list(runtime.tools) if runtime and runtime.is_running() else []

    def get_tool_catalog(self) -> dict[str, Any]:
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "origin": "mcp",
                "server_name": tool.server_name,
                "qualified_id": tool.qualified_id,
                "input_schema": tool.input_schema,
                "mutation": tool.mutation,
            }
            for tool in self.get_all_tools()
        ]
        return {
            "tools": tools,
            "server_count": len(self.servers),
            "active_servers": [
                name for name, runtime in self.servers.items() if runtime.is_running()
            ],
        }

    async def call_tool(
        self,
        qualified_tool_id: str,
        arguments: dict[str, Any],
        timeout: float = 30.0,
    ) -> Any:
        if not self._initialized:
            await self.initialize()
        if "::" not in qualified_tool_id:
            raise ValueError(f"Invalid qualified tool ID: {qualified_tool_id}")

        server_name, tool_name = qualified_tool_id.split("::", 1)
        runtime = self.servers.get(server_name)
        if runtime is None:
            raise ValueError(f"MCP server not found: {server_name}")
        if is_desktop_commander(runtime.config.command, runtime.config.args):
            refuse_sensitive_paths(tool_name, arguments)

        session = await self._live_session(server_name)
        try:
            return await self._invoke(session, qualified_tool_id, tool_name, arguments, timeout)
        except _UNSENT_REQUEST_ERRORS:
            # The server died while idle and this request never reached it, so
            # a fresh server can carry it without repeating any effect.
            logger.info("MCP server '%s' had exited; restarting it for this call", server_name)
            session = await self._live_session(server_name)
            return await self._invoke(session, qualified_tool_id, tool_name, arguments, timeout)

    @staticmethod
    async def _invoke(
        session: _ServerSession,
        qualified_tool_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float,
    ) -> Any:
        tool = session.tools.get(tool_name)
        if tool is None:
            raise ValueError(f"MCP tool not found: {qualified_tool_id}")
        try:
            return await asyncio.wait_for(tool.ainvoke(arguments), timeout=timeout)
        except McpError as exc:
            # The connection closed after the request was sent: it may have run,
            # so it is reported rather than repeated, and the next call restarts.
            if exc.error.code == CONNECTION_CLOSED:
                session.alive = False
            raise
        except _UNSENT_REQUEST_ERRORS:
            session.alive = False
            raise


def resolve_current_mcp_scope() -> MCPProfileScope:
    """Resolve the authenticated account and stable local installation."""

    user_id = get_upstream_auth_service().get_current_user_id()
    if not user_id:
        raise RuntimeError("MCP scope requires an authenticated user")
    device_identifier = generate_device_identifier()
    if not device_identifier:
        raise RuntimeError("MCP scope requires a device identifier")
    return MCPProfileScope(
        user_id=user_id,
        device_identifier=device_identifier,
    )


_mcp_managers: dict[MCPProfileScope, LocalMCPManager] = {}


def get_mcp_manager(scope: MCPProfileScope | None = None) -> LocalMCPManager:
    """Get the manager owned by one user/device scope."""

    resolved_scope = scope or resolve_current_mcp_scope()
    manager = _mcp_managers.get(resolved_scope)
    if manager is None:
        store, _ = prepare_mcp_config_store(resolved_scope)
        manager = LocalMCPManager(store=store)
        _mcp_managers[resolved_scope] = manager
    return manager


async def shutdown_mcp_manager(scope: MCPProfileScope | None = None) -> None:
    """Shut down one scoped manager, or all managers during process teardown."""

    if scope is not None:
        manager = _mcp_managers.pop(scope, None)
        if manager is not None:
            await manager.shutdown()
        return

    managers = list(_mcp_managers.values())
    _mcp_managers.clear()
    for manager in managers:
        await manager.shutdown()
