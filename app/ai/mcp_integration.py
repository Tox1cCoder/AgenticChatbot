import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anyio import BrokenResourceError, ClosedResourceError
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from app.core.config import settings
from app.core.exceptions.mcp import (
    AmbiguousToolNameError,
    ServerConfigurationError,
    ServerNotFoundError,
    ToolNotFoundError,
)
from app.core.mcp_adapter_utils import (
    build_mcp_server_entry,
    clone_mcp_tool,
    normalize_mcp_transport,
    sanitize_mcp_schema,
)

from .utils import get_error_recovery_hint


def compute_catalog_version(descriptors: list[dict[str, Any]]) -> str:
    """Deterministic content hash of a set of sanitized tool descriptors.

    Stable across workers/processes and independent of load order: descriptors are
    canonicalized to ``(server_name, name, sorted-json args_schema)`` triples, sorted,
    then hashed. Two workers observing the same catalog produce the same version, so
    the value can be compared for cross-worker consistency (see FR-CAP-011). Returns a
    ``sha256:<hex>`` string.
    """
    canonical = sorted(
        (
            str(d.get("server_name") or ""),
            str(d.get("name") or ""),
            json.dumps(d.get("args_schema") or {}, sort_keys=True, default=str),
        )
        for d in descriptors
    )
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

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

logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)

# The only ``tool_origin`` this manager owns. Anything else (client_mcp,
# client_skill, internal) belongs to a different catalog and must never be
# reported as backend MCP provenance.
_SERVER_MCP_ORIGIN = "server_mcp"


class MCPManager:
    """Manages MCP server connections and tool loading"""

    DEFAULT_SERVERS = {"calculator", "tavily", "time", "widgets", "brave_image_search"}

    def __init__(self, config_path: str | None = None):
        """
        Initialize MCP Manager

        Args:
            config_path: Path to MCP configuration JSON file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config: dict[str, Any] = {}
        self.client: MultiServerMCPClient | None = None
        self._tools: list[BaseTool] = []
        self._session_contexts: dict[str, Any] = {}
        self._server_tools: dict[str, list[BaseTool]] = {}
        self._tool_index: dict[str, list[BaseTool]] = {}

    async def _run_session_owner(
        self,
        *,
        server_name: str,
        session_context: Any,
        ready: asyncio.Future,
        close_requested: asyncio.Event,
    ) -> None:
        session = None
        entered = False
        try:
            session = await session_context.__aenter__()
            entered = True
            if not ready.done():
                ready.set_result(session)
            await close_requested.wait()
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                logger.warning("MCP session owner for %s failed: %s", server_name, exc)
        finally:
            if entered:
                try:
                    await session_context.__aexit__(None, None, None)
                    logger.debug("Closed session for server: %s", server_name)
                except Exception as exc:
                    logger.warning("Error closing session for %s: %s", server_name, exc)

    def _get_default_config_path(self) -> str:
        return str(Path(__file__).parent / "mcp_config.json")

    def _load_config(self) -> dict[str, Any]:
        try:
            with open(self.config_path) as f:
                config = json.load(f)
                return config
        except FileNotFoundError:
            logger.warning("MCP config file not found at %s", self.config_path)
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse MCP config JSON: {e}")
            return {}

    def _ensure_config_loaded(self) -> None:
        if not self.config:
            self.config = self._load_config()
        servers = self.config.get("servers")
        if not isinstance(servers, dict):
            self.config["servers"] = {}

    def _build_server_config(self) -> dict[str, dict[str, Any]]:
        servers = self.config.get("servers", {})
        server_config: dict[str, dict[str, Any]] = {}
        default_config = Path(self._get_default_config_path()).resolve()
        configured_path = Path(self.config_path).resolve()
        uses_bundled_config = configured_path == default_config
        script_base = (
            Path(__file__).resolve().parents[2] if uses_bundled_config else configured_path.parent
        )

        def resolve_path_like(value: str, *, preserve_bare_command: bool = False) -> str:
            if value.startswith(("-", "@")) or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
                return value
            path = Path(value)
            if path.is_absolute():
                return str(path)
            explicit_relative = value.startswith(("./", ".\\", "../", "..\\"))
            contains_separator = "/" in value or "\\" in value
            if preserve_bare_command and not explicit_relative and not contains_separator:
                return value
            if (
                explicit_relative
                or contains_separator
                or path.suffix.lower() in _CONFIG_RELATIVE_SUFFIXES
                or (script_base / path).exists()
            ):
                return str((script_base / path).resolve())
            return value

        for server_name, server_info in servers.items():
            if not self._server_enabled(server_info):
                continue

            transport = normalize_mcp_transport(server_info.get("transport"))

            if transport == "stdio":
                # Resolve scripts against the selected bundled/custom base,
                # independent of the process working directory.
                abs_args = [resolve_path_like(str(arg)) for arg in server_info.get("args", [])]

                command = str(server_info.get("command") or "python")
                if uses_bundled_config and command.lower() in {"python", "python3"}:
                    command = sys.executable
                else:
                    command = resolve_path_like(command, preserve_bare_command=True)
                cwd = server_info.get("cwd")
                resolved_cwd = (
                    str((script_base / cwd).resolve())
                    if cwd and not Path(cwd).is_absolute()
                    else cwd
                )

                entry = build_mcp_server_entry(
                    transport=transport,
                    command=command,
                    args=abs_args,
                    cwd=resolved_cwd,
                    env=server_info.get("env"),
                )
            elif transport in {"streamable_http", "sse"}:
                entry = build_mcp_server_entry(
                    transport=transport,
                    url=server_info.get("url", ""),
                    headers=server_info.get("headers"),
                )
            else:
                entry = None

            if entry is None:
                continue
            server_config[server_name] = entry

        return server_config

    def _get_enabled_server_names(self) -> list[str]:
        """Return names of all enabled servers from configuration."""
        self._ensure_config_loaded()
        servers = self.config.get("servers", {})
        return [name for name, cfg in servers.items() if self._server_enabled(cfg)]

    @staticmethod
    def _server_enabled(server_info: dict[str, Any]) -> bool:
        return bool(
            server_info.get(
                "enabledByDefault",
                server_info.get("enabled", True),
            )
        )

    async def initialize(self) -> None:
        if settings.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
        if settings.brave_search_api_key:
            os.environ["BRAVE_SEARCH_API_KEY"] = settings.brave_search_api_key

        self.config = self._load_config()
        server_config = self._build_server_config()

        if not server_config:
            logger.warning("No MCP servers configured. MCP tools unavailable.")
            return

        try:
            self.client = MultiServerMCPClient(server_config)
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            self.client = None

    async def _ensure_client_ready(self) -> bool:
        if self.client:
            return True
        await self.initialize()
        return bool(self.client)

    async def get_tools(self) -> list[BaseTool]:
        if not await self._ensure_client_ready():
            return []

        enabled_servers = self._get_enabled_server_names()
        missing_servers = [name for name in enabled_servers if name not in self._server_tools]

        for server_name in missing_servers:
            try:
                await self.get_server_tools(server_name)
            except ServerNotFoundError:
                pass
            except Exception as exc:
                logger.error(
                    "Unexpected error loading tools for server '%s': %s",
                    server_name,
                    exc,
                    exc_info=True,
                )

        if not missing_servers and self._tools:
            return self._tools

        combined_tools: list[BaseTool] = []
        for server_name in enabled_servers:
            server_tools = self._server_tools.get(server_name, [])
            combined_tools.extend(server_tools)

        self._tools = combined_tools
        return self._tools

    async def get_server_tools(self, server_name: str) -> list[BaseTool]:
        if not await self._ensure_client_ready():
            return []

        self._ensure_config_loaded()
        server_cfg = self.config.get("servers", {}).get(server_name)
        if not server_cfg or not self._server_enabled(server_cfg):
            raise ServerNotFoundError(server_name)

        if server_name in self._server_tools:
            return self._server_tools[server_name]

        session_info: dict[str, Any] | None = None
        try:
            session_context = self.client.session(server_name)
            loop = asyncio.get_running_loop()
            ready: asyncio.Future = loop.create_future()
            close_requested = asyncio.Event()
            owner_task = asyncio.create_task(
                self._run_session_owner(
                    server_name=server_name,
                    session_context=session_context,
                    ready=ready,
                    close_requested=close_requested,
                )
            )
            session = await ready
            session_info = {
                "close_requested": close_requested,
                "owner_task": owner_task,
                "session": session,
            }

            loaded_tools = list(await load_mcp_tools(session))
            cleaned_tools = [clone_mcp_tool(tool, server_name=server_name) for tool in loaded_tools]

            self._session_contexts[server_name] = session_info
            self._index_server_tools(server_name, cleaned_tools)

            logger.debug(
                "Loaded %d tools from server '%s' (session active)",
                len(cleaned_tools),
                server_name,
            )
            return cleaned_tools

        except ValueError as exc:
            raise ServerNotFoundError(server_name) from exc
        except Exception as exc:
            logger.error(
                "Failed to load tools from server '%s': %s",
                server_name,
                exc,
                exc_info=True,
            )
            if session_info is not None:
                await self._close_session_context(server_name, session_info)
            return []

    async def _close_session_context(self, server_name: str, session_info: Any) -> None:
        """Ask the task that opened an MCP session to close it."""
        if isinstance(session_info, dict) and "close_requested" in session_info:
            close_requested = session_info["close_requested"]
            owner_task = session_info["owner_task"]
            close_requested.set()
            try:
                await owner_task
            except Exception as exc:
                logger.warning("Error closing session for %s: %s", server_name, exc)
            return

        context = session_info["context"] if isinstance(session_info, dict) else session_info
        try:
            await context.__aexit__(None, None, None)
            logger.debug("Closed session for server: %s", server_name)
        except Exception as exc:
            logger.warning("Error closing session for %s: %s", server_name, exc)

    def _index_server_tools(self, server_name: str, tools: Iterable[BaseTool]) -> None:
        """Store tools for a server and update lookup indexes."""
        tool_list = list(tools)
        self._server_tools[server_name] = tool_list

        for tool in tool_list:
            indexed_tools = self._tool_index.setdefault(tool.name, [])
            if not any(existing is tool for existing in indexed_tools):
                indexed_tools.append(tool)

    def get_tool_args_schema(self, tool: BaseTool) -> dict[str, Any]:
        """Public helper to expose argument schema for a tool."""
        return sanitize_mcp_schema(getattr(tool, "args_schema", None))

    def get_server_for_tool(self, tool: BaseTool) -> str | None:
        """Return the BACKEND server that provided ``tool``, from stamped provenance.

        Reads the application-owned ``metadata["server_name"]`` that
        :func:`clone_mcp_tool` stamps at load time (FR-MCP-003), so a tool this
        manager instance never personally indexed — a clone, a rebuilt binding,
        a tool handed back by another layer — still resolves correctly. Tools
        without metadata (older adapters, test doubles) fall back to an identity
        scan of the per-server index.

        The stamp is only honoured for ``tool_origin == "server_mcp"``. Device-local
        client tools carry their OWN ``server_name`` (the MCP server on the user's
        device); attributing that to the backend catalog would let a locally-named
        server shadow a backend one and slip through a base agent's server allowlist
        (``base_agent._filter_tools_by_allowlist``) — a scope-broadening fail-open
        across the device boundary.
        """
        metadata = getattr(tool, "metadata", None)
        if isinstance(metadata, dict):
            origin = str(metadata.get("tool_origin") or "").strip()
            if origin and origin != _SERVER_MCP_ORIGIN:
                return None
            stamped = metadata.get("server_name")
            if isinstance(stamped, str) and stamped:
                return stamped

        for server_name, tools in self._server_tools.items():
            if any(existing is tool for existing in tools):
                return server_name
        return None

    async def get_servers_for_tool_name(self, tool_name: str) -> list[str]:
        """Return all server names that expose a tool with the given name."""
        await self.get_tools()
        servers = []
        for tool in self._tool_index.get(tool_name, []):
            server_name = self.get_server_for_tool(tool)
            if server_name and server_name not in servers:
                servers.append(server_name)
        return servers

    async def cleanup(self) -> None:
        """Cleanup MCP client resources and properly close all sessions"""
        for server_name, session_info in list(self._session_contexts.items()):
            await self._close_session_context(server_name, session_info)

        self._session_contexts.clear()
        self._tools = []
        self._server_tools.clear()
        self._tool_index.clear()

        if self.client:
            self.client = None
            logger.debug("MCP client cleaned up")

    def add_server(self, server_name: str, server_config: dict[str, Any]) -> None:
        self._ensure_config_loaded()

        if "servers" not in self.config:
            self.config["servers"] = {}

        if server_name in self.DEFAULT_SERVERS:
            raise ServerConfigurationError(
                f"Server '{server_name}' is reserved and managed by the system"
            )

        if server_name in self.config["servers"]:
            raise ServerConfigurationError(f"Server '{server_name}' already exists")

        self.config["servers"][server_name] = server_config
        self.save_config()

        # Notify registry of configuration change
        self._notify_registry_change()

    async def remove_server(self, server_name: str) -> None:
        self._ensure_config_loaded()
        if server_name not in self.config.get("servers", {}):
            raise ServerNotFoundError(server_name)
        if server_name in self.DEFAULT_SERVERS:
            raise ServerConfigurationError(f"Cannot remove core server '{server_name}'")

        # Properly close session context if it exists
        if server_name in self._session_contexts:
            await self._close_session_context(server_name, self._session_contexts[server_name])
            del self._session_contexts[server_name]

        # Remove cached tools
        removed_tools = self._server_tools.pop(server_name, [])
        for tool in removed_tools:
            indexed = self._tool_index.get(tool.name)
            if indexed:
                self._tool_index[tool.name] = [
                    existing for existing in indexed if existing is not tool
                ]
                if not self._tool_index[tool.name]:
                    del self._tool_index[tool.name]
        if removed_tools:
            self._tools = [tool for tool in self._tools if tool not in removed_tools]

        del self.config["servers"][server_name]
        self.save_config()

        # Notify registry of configuration change
        self._notify_registry_change()

    async def enable_server(self, server_name: str) -> None:
        self._ensure_config_loaded()
        if server_name not in self.config.get("servers", {}):
            raise ServerNotFoundError(server_name)

        self.config["servers"][server_name]["enabledByDefault"] = True
        self.save_config()

        # Notify registry of configuration change
        self._notify_registry_change()

    async def disable_server(self, server_name: str) -> None:
        self._ensure_config_loaded()
        if server_name not in self.config.get("servers", {}):
            raise ServerNotFoundError(server_name)

        # Properly close session context if it exists
        if server_name in self._session_contexts:
            await self._close_session_context(server_name, self._session_contexts[server_name])
            del self._session_contexts[server_name]

        # Remove tools from caches
        removed_tools = self._server_tools.pop(server_name, [])
        for tool in removed_tools:
            indexed = self._tool_index.get(tool.name)
            if indexed:
                self._tool_index[tool.name] = [
                    existing for existing in indexed if existing is not tool
                ]
                if not self._tool_index[tool.name]:
                    del self._tool_index[tool.name]
        if removed_tools:
            self._tools = [tool for tool in self._tools if tool not in removed_tools]

        self.config["servers"][server_name]["enabledByDefault"] = False
        self.save_config()

        # Notify registry of configuration change
        self._notify_registry_change()

    def save_config(self) -> None:
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    async def reload_tools(self) -> None:
        """
        Refresh all tools from all enabled servers
        Clears cache and reinitializes client
        """
        await self.cleanup()
        await self.initialize()
        await self.get_tools()

        # Notify registry of configuration change
        self._notify_registry_change()

    async def list_tool_descriptors(
        self, server_name: str | None = None
    ) -> list[dict[str, Any]]:
        """Return sanitized tool descriptors, scoped to one server when given.

        A scoped request loads ONLY that server's tools (via ``get_server_tools``)
        instead of the whole catalog, and reads each tool's owning server from the
        per-server index — never guessed from a bare tool name. An unknown or
        disabled scoped server raises ``ServerNotFoundError``.

        Args:
            server_name: When provided, restrict the load and result to that one
                server. When ``None``, return descriptors for every enabled server.

        Returns:
            List of dicts with ``name``, ``description``, ``args_schema``, ``server_name``.
        """
        if server_name is not None:
            await self.get_server_tools(server_name)
            server_items: list[tuple[str, list[BaseTool]]] = [
                (server_name, self._server_tools.get(server_name, []))
            ]
        else:
            await self.get_tools()
            server_items = list(self._server_tools.items())

        descriptors: list[dict[str, Any]] = []
        for sname, tools in server_items:
            for tool in tools:
                args_schema = sanitize_mcp_schema(getattr(tool, "args_schema", None))
                descriptors.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "args_schema": args_schema,
                        "server_name": sname,
                    }
                )
        return descriptors

    async def get_all_tools_info(self) -> list[dict[str, Any]]:
        """Information about all available tools across every enabled server.

        Thin back-compat wrapper over :meth:`list_tool_descriptors`.
        """
        return await self.list_tool_descriptors(None)

    @staticmethod
    def _is_session_error(error: Exception) -> bool:
        """Check if an error indicates a dead/closed MCP session."""
        return isinstance(error, (ClosedResourceError, BrokenResourceError))

    async def reconnect_server(self, server_name: str) -> list[BaseTool]:
        """
        Close and re-establish the session for *server_name*, returning fresh tools.

        This is the recovery path when a ``ClosedResourceError`` (or similar)
        is raised during tool execution – the underlying MCP server process
        likely crashed.
        """
        logger.info("Reconnecting MCP server '%s' after session error", server_name)

        # 1. Tear down old session context
        old = self._session_contexts.pop(server_name, None)
        if old:
            await self._close_session_context(server_name, old)

        # 2. Drop cached tools so get_server_tools re-creates everything
        removed_tools = self._server_tools.pop(server_name, [])
        for tool in removed_tools:
            indexed = self._tool_index.get(tool.name)
            if indexed:
                self._tool_index[tool.name] = [t for t in indexed if t is not tool]
                if not self._tool_index[tool.name]:
                    del self._tool_index[tool.name]
        self._tools = [t for t in self._tools if t not in removed_tools]

        # 3. Re-create client entry if needed (config unchanged)
        if not self.client:
            await self.initialize()

        # 4. Load fresh tools via a new session
        fresh_tools = await self.get_server_tools(server_name)

        # Rebuild the combined tools list
        combined: list[BaseTool] = []
        for sn in self._get_enabled_server_names():
            combined.extend(self._server_tools.get(sn, []))
        self._tools = combined

        logger.info(
            "Reconnected MCP server '%s' – %d tools available",
            server_name,
            len(fresh_tools),
        )
        return fresh_tools

    async def reconnect_and_get_tool(
        self, tool_name: str, server_name: str | None = None
    ) -> BaseTool | None:
        """
        Reconnect whichever server owns *tool_name* and return a fresh tool.

        ``server_name`` pins the reconnect to a known owner; without it the
        owner is discovered from the per-server index and then from stamped
        provenance. Returns ``None`` when the server cannot be determined or the
        tool no longer appears after reconnect.
        """
        if not server_name:
            for sname, tools in self._server_tools.items():
                if any(t.name == tool_name for t in tools):
                    server_name = sname
                    break

        if not server_name:
            # Fallback: read stamped provenance off the stale index
            for tool in self._tool_index.get(tool_name, []):
                server_name = self.get_server_for_tool(tool)
                if server_name:
                    break

        if not server_name:
            logger.warning("Cannot reconnect for tool '%s': server unknown", tool_name)
            return None

        await self.reconnect_server(server_name)
        for tool in self._server_tools.get(server_name, []):
            if tool.name == tool_name:
                return tool
        return None

    async def get_tool_by_name(
        self, tool_name: str, server_name: str | None = None
    ) -> BaseTool | None:
        """
        Get a specific tool by name

        Duplicate bare names across servers are legal, so an unqualified lookup
        that matches more than one server is an error rather than a silent
        first-indexed-wins pick (FR-MCP-003/FR-CAP-010).

        Args:
            tool_name: Name of the tool to retrieve
            server_name: Optional server to restrict the lookup

        Returns:
            BaseTool instance or None if not found

        Raises:
            AmbiguousToolNameError: If several servers expose ``tool_name`` and
                no ``server_name`` was supplied.
        """
        await self.get_tools()

        if server_name:
            for tool in self._server_tools.get(server_name, []):
                if tool.name == tool_name:
                    return tool
            return None

        candidates = self._tool_index.get(tool_name, [])
        if not candidates:
            return None

        owners: list[str] = []
        for tool in candidates:
            owner = self.get_server_for_tool(tool)
            if owner and owner not in owners:
                owners.append(owner)
        if len(owners) > 1:
            raise AmbiguousToolNameError(tool_name, owners)
        return candidates[0]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        server_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute a tool for testing purposes

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            server_name: Owning server, required when the bare name is exposed
                by more than one server

        Returns:
            Dict with execution result and metadata

        Raises:
            ToolNotFoundError: If tool doesn't exist
            AmbiguousToolNameError: If the unqualified name maps to several servers
            ToolExecutionError: If execution fails
        """
        tool = await self.get_tool_by_name(tool_name, server_name=server_name)
        if not tool:
            raise ToolNotFoundError(tool_name)

        # Determine server name
        server_name = server_name or self.get_server_for_tool(tool) or "unknown"

        start_time = time.time()
        try:
            # Execute tool using ainvoke for async support
            result = await tool.ainvoke(arguments)
            execution_time = time.time() - start_time

            return {
                "success": True,
                "result": result,
                "error": None,
                "execution_time": execution_time,
                "tool_name": tool_name,
                "server_name": server_name,
            }
        except (ClosedResourceError, BrokenResourceError) as session_err:
            # MCP session died – try to reconnect once and retry
            logger.warning(
                "Session error executing tool '%s' on server '%s': %s. Attempting reconnect…",
                tool_name,
                server_name,
                session_err,
            )
            try:
                fresh_tool = await self.reconnect_and_get_tool(
                    tool_name,
                    server_name=None if server_name == "unknown" else server_name,
                )
                if fresh_tool:
                    result = await fresh_tool.ainvoke(arguments)
                    execution_time = time.time() - start_time
                    return {
                        "success": True,
                        "result": result,
                        "error": None,
                        "execution_time": execution_time,
                        "tool_name": tool_name,
                        "server_name": server_name,
                    }
            except Exception as retry_err:
                logger.error(
                    "Retry after reconnect also failed for '%s': %s",
                    tool_name,
                    retry_err,
                )
                # Fall through to the normal error handling below
                session_err = retry_err  # use retry error for reporting

            execution_time = time.time() - start_time
            e = session_err  # noqa: F841 – reuse variable for shared path

            recovery_hint = get_error_recovery_hint(e, tool_name, arguments)
            error_category = self._categorize_error(e)

            logger.error(
                f"Tool execution failed for {tool_name} with args {arguments}: {e}",
                exc_info=True,
            )

            return {
                "success": False,
                "result": None,
                "error": f"{type(e).__name__}: {str(e)}",
                "error_category": error_category,
                "error_hint": recovery_hint,
                "execution_time": execution_time,
                "tool_name": tool_name,
                "server_name": server_name,
            }
        except Exception as e:
            execution_time = time.time() - start_time

            # Get error recovery hint
            recovery_hint = get_error_recovery_hint(e, tool_name, arguments)

            # Categorize error type
            error_category = self._categorize_error(e)

            # Log detailed error with full traceback
            logger.error(
                f"Tool execution failed for {tool_name} with args {arguments}: {e}",
                exc_info=True,
            )

            return {
                "success": False,
                "result": None,
                "error": f"{type(e).__name__}: {str(e)}",
                "error_category": error_category,
                "error_hint": recovery_hint,
                "execution_time": execution_time,
                "tool_name": tool_name,
                "server_name": server_name,
            }

    def _categorize_error(self, error: Exception) -> str:
        """Categorize error for structured error handling."""
        error_msg = str(error).lower()

        if isinstance(error, (ClosedResourceError, BrokenResourceError)):
            return "session_error"
        elif isinstance(error, TypeError):
            return "argument_error"
        elif isinstance(error, ValueError):
            return "value_error"
        elif isinstance(error, KeyError):
            return "missing_key"
        elif (
            "connection" in error_msg
            or "network" in error_msg
            or "timeout" in error_msg
            or "closedresource" in error_msg
        ):
            return "network_error"
        elif "permission" in error_msg or "unauthorized" in error_msg:
            return "permission_error"
        elif "not found" in error_msg:
            return "not_found"
        else:
            return "unknown_error"

    def get_servers_status(self) -> dict[str, dict[str, Any]]:
        """
        Get status of all configured servers

        Returns:
            Dict mapping server names to their status info
        """
        self._ensure_config_loaded()
        status = {}
        servers = self.config.get("servers", {})

        for server_name, server_info in servers.items():
            enabled = self._server_enabled(server_info)
            tool_count = len(self._server_tools.get(server_name, []))

            status[server_name] = {
                "enabled": enabled,
                "tool_count": tool_count,
                "transport": server_info.get("transport", "unknown"),
                "description": server_info.get("description", ""),
            }

        return status

    def get_server_info(self, server_name: str) -> dict[str, Any]:
        """
        Get detailed information about a specific server

        Args:
            server_name: Name of the server

        Returns:
            Dict with server details

        Raises:
            ServerNotFoundError: If server doesn't exist
        """
        self._ensure_config_loaded()
        if server_name not in self.config.get("servers", {}):
            raise ServerNotFoundError(server_name)

        server_config = self.config["servers"][server_name]
        tool_count = len(self._server_tools.get(server_name, []))

        return {
            "name": server_name,
            "enabled": self._server_enabled(server_config),
            "tool_count": tool_count,
            "config": server_config,
            "description": server_config.get("description", ""),
        }

    def _notify_registry_change(self) -> None:
        """
        Notify the MCP Registry that configuration has changed.

        This increments the tools generation version so that agents
        know to refresh their tool caches.
        """
        try:
            from .mcp_registry import MCPRegistry

            MCPRegistry.notify_server_change()
        except ImportError:
            # Registry not available, ignore
            pass
