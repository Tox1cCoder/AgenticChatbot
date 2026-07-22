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
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from app.core.mcp_adapter_utils import (
    build_mcp_server_entry,
    clean_mcp_tool_name,
    normalize_mcp_transport,
    sanitize_mcp_schema,
)
from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger
from client_backend.core.paths import get_profile_subdir, normalize_path
from client_backend.services.upstream_auth import get_upstream_auth_service

logger = get_logger(__name__)

_CONFIG_RELATIVE_SUFFIXES = {
    ".bat",
    ".cjs",
    ".cmd",
    ".exe",
    ".js",
    ".json",
    ".mjs",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}


def canonicalize_mcp_config_document(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Merge legacy MCP keys into the canonical camel-case document shape."""
    if not isinstance(payload, dict):
        return {"mcpServers": {}}

    legacy = payload.get("mcp_servers")
    canonical = payload.get("mcpServers")
    divergent_names: set[str] = set()
    merged: dict[str, Any] = {}
    if isinstance(legacy, dict):
        merged.update(legacy)
    if isinstance(canonical, dict):
        if isinstance(legacy, dict):
            divergent_names = {
                str(name)
                for name in legacy.keys() & canonical.keys()
                if legacy[name] != canonical[name]
            }
        merged.update(canonical)

    normalized = {key: value for key, value in payload.items() if key != "mcp_servers"}
    bundled_names = normalized.get("_sample_chatbot_bundled_servers")
    if divergent_names and isinstance(bundled_names, list):
        normalized["_sample_chatbot_bundled_servers"] = [
            name for name in bundled_names if str(name) not in divergent_names
        ]
    normalized["mcpServers"] = merged
    return normalized


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
        config_data = canonicalize_mcp_config_document(config_data)
        mcp_servers = config_data["mcpServers"]

        config_dir = self.config_path.parent
        bundled_seed = config_data.get("_sample_chatbot_seed") == "bundled-defaults-v1"
        bundled_server_names = config_data.get("_sample_chatbot_bundled_servers") or []
        if not isinstance(bundled_server_names, list):
            bundled_server_names = []
        bundled_server_names = {str(name) for name in bundled_server_names}
        bundled_root = Path(__file__).resolve().parents[2]

        for name, server_data in mcp_servers.items():
            if not isinstance(server_data, dict):
                logger.warning("Skipping MCP server %s: invalid configuration", name)
                continue

            if server_data.get("enabled", True) is False:
                logger.debug("Skipping disabled MCP server %s", name)
                continue

            transport = normalize_mcp_transport(server_data.get("transport"))
            env = {
                str(key): self._expand_env_placeholders(str(value))
                for key, value in (server_data.get("env") or {}).items()
            }

            if transport == "stdio":
                bundled_server = bundled_seed and name in bundled_server_names
                command = server_data.get("command")
                if not command:
                    logger.warning("Skipping MCP server %s: no command specified", name)
                    continue

                args = server_data.get("args") or []
                cwd = server_data.get("cwd")

                resolved_command = self._resolve_config_relative_value(
                    self._expand_env_placeholders(str(command)),
                    config_dir=bundled_root if bundled_server else config_dir,
                    preserve_bare_command=True,
                )
                if bundled_server and str(command).lower() in {"python", "python3"}:
                    resolved_command = sys.executable
                resolved_args = [
                    self._resolve_config_relative_value(
                        self._expand_env_placeholders(str(arg)),
                        config_dir=bundled_root if bundled_server else config_dir,
                    )
                    for arg in args
                ]
                resolved_cwd = None
                if cwd:
                    resolved_cwd = str(
                        normalize_path(
                            self._expand_env_placeholders(str(cwd)),
                            base_dir=bundled_root if bundled_server else config_dir,
                        )
                    )
                else:
                    resolved_cwd = self._default_stdio_cwd()

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
                server_config[config.name] = entry

        return server_config

    async def _load_server_tools(self, server_name: str) -> list[MCPTool]:
        runtime = self.servers.get(server_name)
        if runtime is None or self._client is None:
            return []

        try:
            async with self._client.session(server_name) as session:
                loaded_tools = list(await load_mcp_tools(session))

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
        seed_payload = self._load_repo_seed_config()

        existing_payload: dict[str, Any] | None = None
        if self.config_path.exists():
            try:
                existing_payload = json.loads(self.config_path.read_text(encoding="utf-8"))
            except Exception:
                existing_payload = None

            if self._has_configured_servers(existing_payload):
                canonical = canonicalize_mcp_config_document(existing_payload)
                if canonical != existing_payload:
                    self.config_path.write_text(
                        json.dumps(canonical, indent=2),
                        encoding="utf-8",
                    )
                return

            if seed_payload is None:
                return

        payload_to_write = seed_payload or {"mcpServers": {}}
        self.config_path.write_text(
            json.dumps(payload_to_write, indent=2),
            encoding="utf-8",
        )
        if seed_payload is not None:
            logger.info("Seeded local MCP config at %s from repo config", self.config_path)
        else:
            logger.info("Created default MCP config at %s", self.config_path)

    @staticmethod
    def _has_configured_servers(payload: dict[str, Any] | None) -> bool:
        return bool(canonicalize_mcp_config_document(payload)["mcpServers"])

    def _load_repo_seed_config(self) -> dict[str, Any] | None:
        """
        In source/dev mode, seed the local sidecar config from the repo's
        canonical MCP config so the sidecar starts with the same integrations
        as the Streamlit/server runtime unless the user chose an explicit file.
        """
        if self._explicit_config_path is not None or client_settings.mcp_config_path:
            return None
        if str(client_settings.environment).strip().lower() == "production":
            return None

        candidate_paths = [
            Path.cwd() / "app" / "ai" / "mcp_config.json",
            Path(__file__).resolve().parents[2] / "app" / "ai" / "mcp_config.json",
        ]
        resolved_target = self.config_path.resolve()

        for candidate in candidate_paths:
            try:
                resolved_candidate = candidate.resolve()
            except Exception:
                continue
            if resolved_candidate == resolved_target or not resolved_candidate.exists():
                continue
            try:
                payload = json.loads(resolved_candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if self._has_configured_servers(payload):
                payload = canonicalize_mcp_config_document(payload)
                payload["_sample_chatbot_seed"] = "bundled-defaults-v1"
                servers = payload["mcpServers"]
                payload["_sample_chatbot_bundled_servers"] = sorted(servers)
                return payload
        return None

    @staticmethod
    def _default_stdio_cwd() -> str | None:
        workspace_roots = list(client_settings.workspace_roots or [])
        if not workspace_roots:
            return None

        first_root = str(workspace_roots[0]).strip()
        if not first_root:
            return None
        return str(normalize_path(first_root))

    @staticmethod
    def _expand_env_placeholders(value: str) -> str:
        """Expand `${VAR}` placeholders explicitly against the current environment."""

        def replace(match: re.Match[str]) -> str:
            env_name = match.group(1)
            return os.environ.get(env_name, "")

        return re.sub(r"\$\{([^}]+)\}", replace, value)

    @staticmethod
    def _resolve_config_relative_value(
        value: str,
        *,
        config_dir: Path,
        preserve_bare_command: bool = False,
    ) -> str:
        """
        Resolve relative path-like config values against the MCP config directory.

        Non-path command names such as `python` or `npx` are left unchanged.
        """
        if not value:
            return value
        if value.startswith(("-", "@")) or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
            return value

        candidate = Path(value)
        if candidate.is_absolute():
            return str(normalize_path(candidate))

        explicit_config_relative = value.startswith(("./", ".\\", "../", "..\\"))
        contains_separator = "/" in value or "\\" in value
        if preserve_bare_command and not explicit_config_relative and not contains_separator:
            return value
        looks_path_like = (
            explicit_config_relative
            or contains_separator
            or candidate.suffix.lower() in _CONFIG_RELATIVE_SUFFIXES
            or (config_dir / candidate).exists()
        )
        if not looks_path_like:
            return value

        config_relative_path = normalize_path(candidate, base_dir=config_dir)
        return str(config_relative_path)

    def _tool_records_from_loaded_tools(
        self,
        server_name: str,
        loaded_tools: list[Any],
    ) -> list[MCPTool]:
        records: list[MCPTool] = []
        for tool in loaded_tools:
            tool_name = clean_mcp_tool_name(
                str(getattr(tool, "name", "") or "unknown"),
                server_name=server_name,
            )
            records.append(
                MCPTool(
                    name=tool_name,
                    description=str(getattr(tool, "description", "") or ""),
                    server_name=server_name,
                    input_schema=sanitize_mcp_schema(getattr(tool, "args_schema", None)),
                )
            )
        return records

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
        timeout: float = 30.0,
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
        timeout: float,
    ) -> Any:
        if self._client is None:
            raise RuntimeError("MCP client is not initialized")

        _runtime_name, tool_name = qualified_tool_id.split("::", 1)

        async with self._client.session(server_name) as session:
            loaded_tools = list(await load_mcp_tools(session))
            runtime = self.servers.get(server_name)
            if runtime is not None:
                runtime.mark_running(
                    self._tool_records_from_loaded_tools(server_name, loaded_tools)
                )
            for tool in loaded_tools:
                if (
                    clean_mcp_tool_name(
                        str(getattr(tool, "name", "") or ""),
                        server_name=server_name,
                    )
                    != tool_name
                ):
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
