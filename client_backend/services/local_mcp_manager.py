"""Device-scoped local MCP runtime backed exclusively by schema-v2 storage."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from app.core.mcp_adapter_utils import (
    build_mcp_server_entry,
    clean_mcp_tool_name,
    sanitize_mcp_schema,
)
from client_backend.core.logging import get_logger
from client_backend.core.security import generate_device_identifier
from client_backend.schemas.mcp_config import MCPProfileScope
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


class LocalMCPManager:
    """Runs MCP servers for exactly one authenticated user/device scope."""

    def __init__(self, *, store: MCPConfigStore):
        self.store = store
        self.scope = store.scope
        self.servers: dict[str, MCPServerRuntime] = {}
        self._client: MultiServerMCPClient | None = None
        self._initialized = False

    @property
    def config_path(self):
        """Return the canonical v2 profile path for diagnostics."""

        return self.store.profile_path

    async def initialize(self, server_names: set[str] | None = None) -> None:
        if self._initialized:
            return

        configs = [
            config
            for config in self.store.list_effective_servers()
            if config.enabled
            and (server_names is None or config.name in server_names)
        ]
        self.servers = {
            config.name: MCPServerRuntime(config)
            for config in configs
        }
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
            await self._load_server_tools(config.name)

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
            entry = build_mcp_server_entry(
                transport=config.transport,
                command=config.command,
                args=config.args,
                cwd=config.cwd,
                env=config.env,
                url=config.url,
                headers=config.headers,
            )
            if entry is not None:
                result[config.name] = entry
        return result

    async def _load_server_tools(self, server_name: str) -> list[MCPTool]:
        runtime = self.servers.get(server_name)
        if runtime is None or self._client is None:
            return []
        try:
            async with self._client.session(server_name) as session:
                loaded_tools = list(await load_mcp_tools(session))
            records = self._tool_records_from_loaded_tools(server_name, loaded_tools)
            runtime.mark_running(records)
            logger.info("Loaded %d MCP tools from '%s'", len(records), server_name)
            return records
        except Exception as exc:
            message = _terminal_exception_message(exc)
            runtime.mark_error(message)
            logger.error(
                "Failed to load MCP server '%s': %s",
                server_name,
                message,
                exc_info=True,
            )
            return []

    @staticmethod
    def _tool_records_from_loaded_tools(
        server_name: str,
        loaded_tools: list[Any],
    ) -> list[MCPTool]:
        return [
            MCPTool(
                name=clean_mcp_tool_name(
                    str(getattr(tool, "name", "") or "unknown"),
                    server_name=server_name,
                ),
                description=str(getattr(tool, "description", "") or ""),
                server_name=server_name,
                input_schema=sanitize_mcp_schema(getattr(tool, "args_schema", None)),
            )
            for tool in loaded_tools
        ]

    async def shutdown(self) -> None:
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
    ) -> dict[str, Any]:
        if not self._initialized:
            await self.initialize()
        if "::" not in qualified_tool_id:
            raise ValueError(f"Invalid qualified tool ID: {qualified_tool_id}")

        server_name, _ = qualified_tool_id.split("::", 1)
        runtime = self.servers.get(server_name)
        if runtime is None:
            raise ValueError(f"MCP server not found: {server_name}")
        if not runtime.is_running():
            raise RuntimeError(
                runtime.error_message or f"MCP server {server_name} is not running"
            )

        try:
            return await self._call_tool_once(
                server_name,
                qualified_tool_id,
                arguments,
                timeout,
            )
        except Exception:
            await self._load_server_tools(server_name)
            raise

    async def _call_tool_once(
        self,
        server_name: str,
        qualified_tool_id: str,
        arguments: dict[str, Any],
        timeout: float,
    ) -> Any:
        if self._client is None:
            raise RuntimeError("MCP client is not initialized")
        _, tool_name = qualified_tool_id.split("::", 1)
        async with self._client.session(server_name) as session:
            loaded_tools = list(await load_mcp_tools(session))
            runtime = self.servers.get(server_name)
            if runtime is not None:
                runtime.mark_running(
                    self._tool_records_from_loaded_tools(server_name, loaded_tools)
                )
            for tool in loaded_tools:
                loaded_name = clean_mcp_tool_name(
                    str(getattr(tool, "name", "") or ""),
                    server_name=server_name,
                )
                if loaded_name == tool_name:
                    return await asyncio.wait_for(
                        tool.ainvoke(arguments),
                        timeout=timeout,
                    )
        raise ValueError(f"MCP tool not found: {qualified_tool_id}")


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
        manager = LocalMCPManager(store=MCPConfigStore(resolved_scope))
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
