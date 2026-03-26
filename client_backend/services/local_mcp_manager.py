"""
Local MCP (Model Context Protocol) manager for the client backend.

Uses a real MCP client implementation so local stdio and HTTP/SSE servers can
be discovered and invoked using the same protocol behavior as the server.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import BaseModel as PydanticBaseModel

from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger
from client_backend.core.paths import get_profile_subdir, normalize_path
from client_backend.services.upstream_auth import get_upstream_auth_service

logger = get_logger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""

    name: str
    transport: str = "stdio"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class MCPTool:
    """A sanitized MCP tool definition."""

    name: str
    description: str
    server_name: str
    input_schema: dict[str, Any]

    @property
    def qualified_id(self) -> str:
        """Get fully qualified tool ID."""
        return f"{self.server_name}::{self.name}"


class MCPServerRuntime:
    """Tracks the runtime state of a configured MCP server."""

    def __init__(self, config: MCPServerConfig):
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


class LocalMCPManager:
    """
    Manager for local MCP servers.

    Handles loading configuration, establishing MCP sessions, and exposing
    sanitized tool metadata for the runtime bridge.
    """

    def __init__(self, config_path: Path | None = None):
        self._explicit_config_path = normalize_path(config_path) if config_path else None
        self.config_path = self._resolve_config_path()
        self.servers: dict[str, MCPServerRuntime] = {}
        self._client: MultiServerMCPClient | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the MCP manager by loading configuration and opening sessions.
        """
        resolved_config_path = self._resolve_config_path()
        if self._initialized and resolved_config_path == self.config_path:
            return
        if self._initialized and resolved_config_path != self.config_path:
            logger.info(
                "MCP config path changed from %s to %s; reloading manager",
                self.config_path,
                resolved_config_path,
            )
            await self.shutdown()

        self.config_path = resolved_config_path
        self._ensure_default_config_exists()
        logger.info("Initializing MCP manager from: %s", self.config_path)

        configs = await self._load_config()
        self.servers = {config.name: MCPServerRuntime(config) for config in configs}

        server_config = self._build_server_config(configs)
        if not server_config:
            self._initialized = True
            logger.info("MCP manager initialized with 0 active servers")
            return

        try:
            self._client = MultiServerMCPClient(server_config)
        except Exception as exc:
            logger.error("Failed to initialize MCP client: %s", exc, exc_info=True)
            for runtime in self.servers.values():
                runtime.mark_error(str(exc))
            self._initialized = True
            return

        for config in configs:
            await self._load_server_tools(config.name)

        self._initialized = True
        logger.info(
            "MCP manager initialized with %d configured servers and %d discovered tools",
            len(self.servers),
            len(self.get_all_tools()),
        )

    async def _load_config(self) -> list[MCPServerConfig]:
        """
        Load MCP configuration from JSON file.

        Returns:
            List of MCPServerConfig objects.
        """
        try:
            config_data = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to load MCP config: %s", exc)
            return []

        servers: list[MCPServerConfig] = []
        mcp_servers = config_data.get("mcpServers") or config_data.get("mcp_servers") or {}
        if not isinstance(mcp_servers, dict):
            logger.warning("Invalid MCP config structure in %s", self.config_path)
            return []

        config_dir = self.config_path.parent

        for name, server_data in mcp_servers.items():
            if not isinstance(server_data, dict):
                logger.warning("Skipping MCP server %s: invalid configuration", name)
                continue

            if server_data.get("enabled", True) is False:
                logger.debug("Skipping disabled MCP server %s", name)
                continue

            transport = self._normalize_transport(server_data.get("transport"))
            env = {
                str(key): self._expand_env_placeholders(str(value))
                for key, value in (server_data.get("env") or {}).items()
            }

            if transport == "stdio":
                command = server_data.get("command")
                if not command:
                    logger.warning("Skipping MCP server %s: no command specified", name)
                    continue

                args = server_data.get("args") or []
                cwd = server_data.get("cwd")

                resolved_command = self._resolve_config_relative_value(
                    self._expand_env_placeholders(str(command)),
                    config_dir=config_dir,
                )
                resolved_args = [
                    self._resolve_config_relative_value(
                        self._expand_env_placeholders(str(arg)),
                        config_dir=config_dir,
                    )
                    for arg in args
                ]
                resolved_cwd = None
                if cwd:
                    resolved_cwd = str(
                        normalize_path(
                            self._expand_env_placeholders(str(cwd)),
                            base_dir=config_dir,
                        )
                    )

                servers.append(
                    MCPServerConfig(
                        name=name,
                        transport=transport,
                        command=resolved_command,
                        args=resolved_args,
                        env=env,
                        cwd=resolved_cwd,
                    )
                )
                continue

            if transport in {"streamable_http", "sse"}:
                url = str(server_data.get("url") or "").strip()
                if not url:
                    logger.warning("Skipping MCP server %s: no url specified", name)
                    continue

                headers = {
                    str(key): self._expand_env_placeholders(str(value))
                    for key, value in (server_data.get("headers") or {}).items()
                }
                servers.append(
                    MCPServerConfig(
                        name=name,
                        transport=transport,
                        url=url,
                        headers=headers,
                    )
                )
                continue

            logger.warning(
                "Skipping MCP server %s: unsupported transport '%s'",
                name,
                transport,
            )

        logger.info("Loaded %d MCP server configurations", len(servers))
        return servers

    def _build_server_config(
        self,
        configs: list[MCPServerConfig],
    ) -> dict[str, dict[str, Any]]:
        server_config: dict[str, dict[str, Any]] = {}

        for config in configs:
            if config.transport == "stdio":
                entry: dict[str, Any] = {
                    "transport": "stdio",
                    "command": config.command,
                    "args": list(config.args),
                }
                if config.cwd:
                    entry["cwd"] = config.cwd
                if config.env:
                    entry["env"] = config.env
            elif config.transport in {"streamable_http", "sse"}:
                entry = {
                    "transport": config.transport,
                    "url": config.url,
                }
                if config.headers:
                    entry["headers"] = config.headers
            else:
                continue

            server_config[config.name] = entry

        return server_config

    async def _load_server_tools(self, server_name: str) -> list[MCPTool]:
        runtime = self.servers.get(server_name)
        if runtime is None or self._client is None:
            return []

        try:
            async with self._client.session(server_name) as session:
                loaded_tools = list(await load_mcp_tools(session))
                loaded_tools = self._clean_loaded_tools(loaded_tools)

            tool_records = self._tool_records_from_loaded_tools(server_name, loaded_tools)
            runtime.mark_running(tool_records)
            logger.info(
                "Loaded %d MCP tools from server '%s'",
                len(tool_records),
                server_name,
            )
            return tool_records
        except Exception as exc:
            runtime.mark_error(str(exc))
            logger.error(
                "Failed to load tools from MCP server '%s': %s",
                server_name,
                exc,
                exc_info=True,
            )
            return []

    def _resolve_config_path(self) -> Path:
        if self._explicit_config_path is not None:
            return self._explicit_config_path

        if client_settings.mcp_config_path:
            return normalize_path(client_settings.mcp_config_path)

        auth_service = get_upstream_auth_service()
        current_user_id = auth_service.get_current_user_id()
        if current_user_id:
            return get_profile_subdir(current_user_id, "mcp") / "mcp_config.json"

        return client_settings.get_profile_path("default", "mcp_config.json")

    def _ensure_default_config_exists(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            return

        self.config_path.write_text(
            json.dumps({"mcpServers": {}}, indent=2),
            encoding="utf-8",
        )
        logger.info("Created default MCP config at %s", self.config_path)

    @staticmethod
    def _expand_env_placeholders(value: str) -> str:
        """Expand `${VAR}` placeholders explicitly against the current environment."""

        def replace(match: re.Match[str]) -> str:
            env_name = match.group(1)
            return os.environ.get(env_name, "")

        return re.sub(r"\$\{([^}]+)\}", replace, value)

    @staticmethod
    def _resolve_config_relative_value(value: str, *, config_dir: Path) -> str:
        """
        Resolve relative path-like config values against the MCP config directory.

        Non-path command names such as `python` or `npx` are left unchanged.
        """
        if not value:
            return value

        candidate = Path(value)
        if candidate.is_absolute():
            return str(normalize_path(candidate))

        explicit_config_relative = value.startswith(("./", ".\\", "../", "..\\"))
        looks_path_like = (
            explicit_config_relative or "\\" in value or (config_dir / candidate).exists()
        )
        if not looks_path_like:
            return value

        config_relative_path = normalize_path(candidate, base_dir=config_dir)
        if explicit_config_relative or config_relative_path.exists():
            return str(config_relative_path)

        fallback_path = normalize_path(candidate)
        if fallback_path.exists():
            return str(fallback_path)

        return str(config_relative_path)

    @staticmethod
    def _normalize_transport(value: Any) -> str:
        transport = str(value or "stdio").strip().lower()
        if transport == "http":
            return "streamable_http"
        return transport

    @staticmethod
    def _clean_tool_name(tool_name: str) -> str:
        if ":" in tool_name:
            return tool_name.split(":", 1)[-1]
        return tool_name

    def _tool_records_from_loaded_tools(
        self,
        server_name: str,
        loaded_tools: list[Any],
    ) -> list[MCPTool]:
        records: list[MCPTool] = []
        for tool in loaded_tools:
            tool_name = self._clean_tool_name(getattr(tool, "name", "") or "unknown")
            records.append(
                MCPTool(
                    name=tool_name,
                    description=str(getattr(tool, "description", "") or ""),
                    server_name=server_name,
                    input_schema=self._serialize_args_schema(getattr(tool, "args_schema", None)),
                )
            )
        return records

    def _filter_schema_recursively(self, schema: Any) -> Any:
        unsupported_keys = {"$schema", "additionalProperties"}

        if isinstance(schema, dict):
            filtered = {}
            for key, value in schema.items():
                if key in unsupported_keys or value is None:
                    continue

                if key in {
                    "properties",
                    "items",
                    "anyOf",
                    "allOf",
                    "oneOf",
                    "definitions",
                } or isinstance(value, dict):
                    filtered_value = self._filter_schema_recursively(value)
                    if filtered_value:
                        filtered[key] = filtered_value
                elif isinstance(value, list):
                    filtered[key] = [
                        self._filter_schema_recursively(item) for item in value if item is not None
                    ]
                else:
                    filtered[key] = value

            if filtered.get("type") == "array" and "items" not in filtered:
                filtered["items"] = {"type": "string"}

            return filtered

        if isinstance(schema, list):
            return [self._filter_schema_recursively(item) for item in schema if item is not None]

        return schema

    def _remove_non_string_enums(self, schema: Any) -> Any:
        if isinstance(schema, dict):
            cleaned: dict[str, Any] = {}
            for key, value in schema.items():
                if (
                    key == "enum"
                    and isinstance(value, list)
                    and any(not isinstance(item, str) for item in value)
                ):
                    continue
                cleaned[key] = self._remove_non_string_enums(value)
            return cleaned

        if isinstance(schema, list):
            return [self._remove_non_string_enums(item) for item in schema]

        return schema

    def _clean_loaded_tools(self, tools: list[BaseTool]) -> list[BaseTool]:
        cleaned_tools: list[BaseTool] = []

        for tool in tools:
            args_schema = getattr(tool, "args_schema", None)

            if args_schema is None:
                tool.args_schema = {"type": "object", "properties": {}}
                cleaned_tools.append(tool)
                continue

            if isinstance(args_schema, dict):
                filtered = self._filter_schema_recursively(args_schema)
                filtered = self._remove_non_string_enums(filtered)
                if not filtered.get("properties"):
                    filtered["properties"] = {}
                if not filtered.get("type"):
                    filtered["type"] = "object"
                tool.args_schema = filtered
                cleaned_tools.append(tool)
                continue

            if isinstance(args_schema, type) and issubclass(args_schema, PydanticBaseModel):
                original_schema_method = args_schema.model_json_schema

                def filtered_schema_method(
                    *args,
                    original_schema_method=original_schema_method,
                    **kwargs,
                ):
                    schema = original_schema_method(*args, **kwargs)
                    if isinstance(schema, dict):
                        schema = self._filter_schema_recursively(schema)
                        schema = self._remove_non_string_enums(schema)
                        if not schema.get("properties"):
                            schema["properties"] = {}
                        if not schema.get("type"):
                            schema["type"] = "object"
                    return schema

                args_schema.model_json_schema = staticmethod(filtered_schema_method)

            cleaned_tools.append(tool)

        return cleaned_tools

    def _serialize_args_schema(self, schema: Any) -> dict[str, Any]:
        if not schema:
            return {}

        result_schema = {}

        if isinstance(schema, dict):
            result_schema = schema
        elif (
            (
                isinstance(schema, type)
                and issubclass(schema, PydanticBaseModel)
                and hasattr(schema, "model_json_schema")
            )
            or isinstance(schema, PydanticBaseModel)
            and hasattr(schema, "model_json_schema")
        ):
            result_schema = schema.model_json_schema()

        if not result_schema:
            for attr_name in ("model_json_schema", "json_schema", "schema"):
                exporter = getattr(schema, attr_name, None)
                if not callable(exporter):
                    continue
                try:
                    result_schema = exporter()
                    break
                except TypeError:
                    try:
                        result_schema = exporter(by_alias=True)
                        break
                    except Exception:
                        continue
                except Exception:
                    continue

        if isinstance(result_schema, dict):
            result_schema = self._filter_schema_recursively(result_schema)
            result_schema = self._remove_non_string_enums(result_schema)

        return result_schema if isinstance(result_schema, dict) else {}

    async def shutdown(self) -> None:
        """Clear loaded MCP state."""
        logger.info("Shutting down MCP manager...")

        for runtime in self.servers.values():
            runtime.mark_stopped()

        self.servers.clear()
        self._client = None
        self._initialized = False
        logger.info("MCP manager shutdown complete")

    def get_all_tools(self) -> list[MCPTool]:
        """Get all tools from all running MCP servers."""
        all_tools: list[MCPTool] = []
        for runtime in self.servers.values():
            if runtime.is_running():
                all_tools.extend(runtime.tools)
        return all_tools

    def get_tools_by_server(self, server_name: str) -> list[MCPTool]:
        """Get tools from a specific server."""
        runtime = self.servers.get(server_name)
        if runtime and runtime.is_running():
            return list(runtime.tools)
        return []

    def get_tool_catalog(self) -> dict[str, Any]:
        """
        Generate a sanitized tool catalog for syncing to the server.

        Returns:
            Tool catalog dictionary.
        """
        tools = []

        for tool in self.get_all_tools():
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "origin": "mcp",
                    "server_name": tool.server_name,
                    "qualified_id": tool.qualified_id,
                    "input_schema": tool.input_schema,
                }
            )

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
        timeout: int = 30,
    ) -> dict[str, Any]:
        """
        Call a tool by its qualified ID.

        Args:
            qualified_tool_id: Fully qualified tool ID (server_name::tool_name).
            arguments: Tool arguments.
            timeout: Call timeout in seconds.

        Returns:
            Tool result.

        Raises:
            ValueError: If tool not found.
            RuntimeError: If call fails.
        """
        if not self._initialized:
            await self.initialize()

        if "::" not in qualified_tool_id:
            raise ValueError(f"Invalid qualified tool ID: {qualified_tool_id}")

        server_name, _tool_name = qualified_tool_id.split("::", 1)
        runtime = self.servers.get(server_name)
        if not runtime:
            raise ValueError(f"MCP server not found: {server_name}")

        if not runtime.is_running():
            raise RuntimeError(runtime.error_message or f"MCP server {server_name} is not running")

        try:
            return await self._call_tool_once(server_name, qualified_tool_id, arguments, timeout)
        except Exception:
            await self._load_server_tools(server_name)
            raise

    async def _call_tool_once(
        self,
        server_name: str,
        qualified_tool_id: str,
        arguments: dict[str, Any],
        timeout: int,
    ) -> Any:
        if self._client is None:
            raise RuntimeError("MCP client is not initialized")

        _runtime_name, tool_name = qualified_tool_id.split("::", 1)

        async with self._client.session(server_name) as session:
            loaded_tools = list(await load_mcp_tools(session))
            loaded_tools = self._clean_loaded_tools(loaded_tools)
            runtime = self.servers.get(server_name)
            if runtime is not None:
                runtime.mark_running(
                    self._tool_records_from_loaded_tools(server_name, loaded_tools)
                )
            for tool in loaded_tools:
                if self._clean_tool_name(getattr(tool, "name", "") or "") != tool_name:
                    continue
                return await asyncio.wait_for(tool.ainvoke(arguments), timeout=timeout)

        raise ValueError(f"MCP tool not found: {qualified_tool_id}")


# Global singleton
_mcp_manager: LocalMCPManager | None = None


def get_mcp_manager() -> LocalMCPManager:
    """Get the global MCP manager."""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = LocalMCPManager()
    return _mcp_manager


async def shutdown_mcp_manager() -> None:
    """Shutdown the global MCP manager."""
    global _mcp_manager
    if _mcp_manager:
        await _mcp_manager.shutdown()
        _mcp_manager = None
