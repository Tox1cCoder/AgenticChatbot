import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterable

from app.core.config import settings
from app.core.exceptions.mcp import (
    ServerNotFoundError,
    ToolNotFoundError,
    ServerConfigurationError,
)

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import BaseTool
from pydantic import BaseModel as PydanticBaseModel


logger = logging.getLogger(__name__)


class MCPManager:
    """Manages MCP server connections and tool loading"""

    DEFAULT_SERVERS = {"calculator", "tavily", "time"}

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MCP Manager

        Args:
            config_path: Path to MCP configuration JSON file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config: Dict[str, Any] = {}
        self.client: Optional[MultiServerMCPClient] = None
        self._tools: List[BaseTool] = []
        self._sessions: Dict[str, Any] = {}
        self._server_tools: Dict[str, List[BaseTool]] = {}
        self._tool_index: Dict[str, List[BaseTool]] = {}
        self._tool_server_map: Dict[int, str] = {}

    def _get_default_config_path(self) -> str:
        return str(Path(__file__).parent / "mcp_config.json")

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r") as f:
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
        mcp_servers = self.config.get("mcp_servers")
        if not isinstance(mcp_servers, dict):
            self.config["mcp_servers"] = {}

    def _build_server_config(self) -> Dict[str, Dict[str, Any]]:
        mcp_servers = self.config.get("mcp_servers", {})
        server_config = {}

        for server_name, server_info in mcp_servers.items():
            if not server_info.get("enabled", True):
                continue

            transport = server_info.get("transport", "stdio")

            if transport == "stdio":
                # Convert relative paths to absolute
                args = server_info.get("args", [])
                abs_args = []
                for arg in args:
                    if arg.endswith(".py") and not os.path.isabs(arg):
                        # Make path absolute relative to project root
                        abs_path = os.path.abspath(arg)
                        abs_args.append(abs_path)
                    else:
                        abs_args.append(arg)

                server_config[server_name] = {
                    "transport": transport,
                    "command": server_info.get("command", "python"),
                    "args": abs_args,
                }

                # Pass environment variables to subprocess if specified
                if "env" in server_info:
                    server_config[server_name]["env"] = server_info["env"]
            elif transport in ["streamable_http", "sse"]:
                server_config[server_name] = {
                    "transport": transport,
                    "url": server_info.get("url", ""),
                }
                if "headers" in server_info:
                    server_config[server_name]["headers"] = server_info["headers"]
            else:
                logger.warning(f"Unknown transport type for {server_name}: {transport}")
                continue

        return server_config

    def _get_enabled_server_names(self) -> List[str]:
        """Return names of all enabled servers from configuration."""
        self._ensure_config_loaded()
        servers = self.config.get("mcp_servers", {})
        return [
            name
            for name, cfg in servers.items()
            if cfg.get("enabled", True)
        ]

    async def initialize(self) -> None:
        if settings.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = settings.tavily_api_key

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
        if not self.client:
            logger.warning("MCP client is unavailable.")
            return False
        return True

    async def get_tools(self) -> List[BaseTool]:
        if not await self._ensure_client_ready():
            return []

        enabled_servers = self._get_enabled_server_names()
        missing_servers = [
            name for name in enabled_servers if name not in self._server_tools
        ]

        for server_name in missing_servers:
            try:
                await self.get_server_tools(server_name)
            except ServerNotFoundError as exc:
                logger.warning("Configured server '%s' not found: %s", server_name, exc)
            except Exception as exc:
                logger.error(
                    "Unexpected error loading tools for server '%s': %s",
                    server_name,
                    exc,
                    exc_info=True,
                )

        if not missing_servers and self._tools:
            return self._tools

        combined_tools: List[BaseTool] = []
        for server_name in enabled_servers:
              server_tools = self._server_tools.get(server_name, [])
              combined_tools.extend(server_tools)

        self._tools = combined_tools
        return self._tools

    async def get_server_tools(self, server_name: str) -> List[BaseTool]:
        if not await self._ensure_client_ready():
            return []

        self._ensure_config_loaded()
        server_cfg = self.config.get("mcp_servers", {}).get(server_name)
        if not server_cfg or not server_cfg.get("enabled", True):
            raise ServerNotFoundError(server_name)

        if server_name in self._server_tools:
            logger.debug("Returning cached tools for server '%s'", server_name)
            return self._server_tools[server_name]

        try:
            session_context = self.client.session(server_name)
        except ValueError as exc:
            raise ServerNotFoundError(server_name) from exc

        try:
            session = await session_context.__aenter__()
            tools = list(await load_mcp_tools(session))

            self._sessions[server_name] = {
                "context": session_context,
                "session": session,
            }
            self._index_server_tools(server_name, tools)

            logger.info(
                "Loaded %d tools from server '%s' (session kept open)",
                len(tools),
                server_name,
            )
            return tools
        except Exception as exc:
            logger.error(
                "Failed to load tools from server '%s': %s",
                server_name,
                exc,
                exc_info=True,
            )
            await session_context.__aexit__(*([None] * 3))
            
            return []

    def _index_server_tools(self, server_name: str, tools: Iterable[BaseTool]) -> None:
        """Store tools for a server and update lookup indexes."""
        tool_list = list(tools)
        self._server_tools[server_name] = tool_list

        for tool in tool_list:
            self._tool_server_map[id(tool)] = server_name
            indexed_tools = self._tool_index.setdefault(tool.name, [])
            if not any(existing is tool for existing in indexed_tools):
                indexed_tools.append(tool)

    def _serialize_args_schema(self, schema: Any) -> Dict[str, Any]:
        """Normalize a tool args schema into a serializable dictionary."""
        if not schema:
            return {}

        if isinstance(schema, dict):
            return schema

        if PydanticBaseModel is not None:
            if isinstance(schema, type) and issubclass(schema, PydanticBaseModel):
                if hasattr(schema, "model_json_schema"):
                    return schema.model_json_schema()
            if isinstance(schema, PydanticBaseModel):
                if hasattr(schema, "model_json_schema"):
                    return schema.model_json_schema()

        # Generic callable schema exporters
        for attr_name in ("model_json_schema", "json_schema", "schema"):
            exporter = getattr(schema, attr_name, None)
            if callable(exporter):
                try:
                    return exporter()
                except TypeError:
                    try:
                        return exporter(by_alias=True)
                    except Exception:
                        continue
                except Exception:
                    continue
        return {}

    def get_tool_args_schema(self, tool: BaseTool) -> Dict[str, Any]:
        """Public helper to expose argument schema for a tool."""
        return self._serialize_args_schema(getattr(tool, "args_schema", None))

    def get_server_for_tool(self, tool: BaseTool) -> Optional[str]:
        """Return the server name that provided the given tool, if known."""
        return self._tool_server_map.get(id(tool))

    async def get_servers_for_tool_name(self, tool_name: str) -> List[str]:
        """Return all server names that expose a tool with the given name."""
        await self.get_tools()
        servers = []
        for tool in self._tool_index.get(tool_name, []):
            server_name = self._tool_server_map.get(id(tool))
            if server_name:
                servers.append(server_name)
        return servers

    async def cleanup(self) -> None:
        """Cleanup MCP client resources and close all active sessions"""
        # Close all active sessions
        for server_name, session_info in self._sessions.items():
            try:
                context = session_info["context"]
                await context.__aexit__(None, None, None)
                logger.debug(f"Closed session for {server_name}")
            except Exception as e:
                logger.error(f"Error closing session for {server_name}: {e}")

        self._sessions.clear()
        self._tools = []
        self._server_tools.clear()
        self._tool_index.clear()
        self._tool_server_map.clear()

        if self.client:
            self.client = None
            logger.info("MCP client cleaned up")

    def add_server(self, server_name: str, server_config: Dict[str, Any]) -> None:
        self._ensure_config_loaded()

        if "mcp_servers" not in self.config:
            self.config["mcp_servers"] = {}

        if server_name in self.DEFAULT_SERVERS:
            raise ServerConfigurationError(
                f"Server '{server_name}' is reserved and managed by the system"
            )

        if server_name in self.config["mcp_servers"]:
            raise ServerConfigurationError(
                f"Server '{server_name}' already exists"
            )

        self.config["mcp_servers"][server_name] = server_config
        self.save_config()

    async def remove_server(self, server_name: str) -> None:
        self._ensure_config_loaded()
        if server_name not in self.config.get("mcp_servers", {}):
            raise ServerNotFoundError(server_name)
        if server_name in self.DEFAULT_SERVERS:
            raise ServerConfigurationError(
                f"Cannot remove core server '{server_name}'"
            )

        # Cleanup session if active
        if server_name in self._sessions:
            try:
                context = self._sessions[server_name]["context"]
                await context.__aexit__(None, None, None)
                del self._sessions[server_name]
                logger.debug(f"Closed session for {server_name} during removal")
            except Exception as e:
                logger.error(f"Error cleaning up session for {server_name}: {e}")

        removed_tools = self._server_tools.pop(server_name, [])
        for tool in removed_tools:
            self._tool_server_map.pop(id(tool), None)
            indexed = self._tool_index.get(tool.name)
            if indexed:
                self._tool_index[tool.name] = [
                    existing for existing in indexed if existing is not tool
                ]
                if not self._tool_index[tool.name]:
                    del self._tool_index[tool.name]
        if removed_tools:
            self._tools = [tool for tool in self._tools if tool not in removed_tools]

        del self.config["mcp_servers"][server_name]
        self.save_config()

    async def enable_server(self, server_name: str) -> None:
        self._ensure_config_loaded()
        if server_name not in self.config.get("mcp_servers", {}):
            raise ServerNotFoundError(server_name)

        self.config["mcp_servers"][server_name]["enabled"] = True
        self.save_config()

    async def disable_server(self, server_name: str) -> None:
        self._ensure_config_loaded()
        if server_name not in self.config.get("mcp_servers", {}):
            raise ServerNotFoundError(server_name)

        if server_name in self._sessions:
            try:
                context = self._sessions[server_name]["context"]
                await context.__aexit__(None, None, None)
                del self._sessions[server_name]
                logger.debug(f"Closed session for {server_name} during disable")
            except Exception as e:
                logger.error(f"Error closing session for {server_name}: {e}")

        # Remove tools from caches
        removed_tools = self._server_tools.pop(server_name, [])
        for tool in removed_tools:
            self._tool_server_map.pop(id(tool), None)
            indexed = self._tool_index.get(tool.name)
            if indexed:
                self._tool_index[tool.name] = [
                    existing for existing in indexed if existing is not tool
                ]
                if not self._tool_index[tool.name]:
                    del self._tool_index[tool.name]
        if removed_tools:
            self._tools = [tool for tool in self._tools if tool not in removed_tools]

        self.config["mcp_servers"][server_name]["enabled"] = False
        self.save_config()

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
        logger.debug(
            "Reloaded %d MCP tools from %d servers",
            len(self._tools),
            len(self._server_tools),
        )

    async def get_all_tools_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all available tools

        Returns:
            List of dicts with tool metadata (name, description, args_schema, server_name)
        """
        await self.get_tools()
        tools_info: List[Dict[str, Any]] = []

        for server_name, tools in self._server_tools.items():
            for tool in tools:
                args_schema = self._serialize_args_schema(
                    getattr(tool, "args_schema", None)
                )
                tools_info.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "args_schema": args_schema,
                        "server_name": server_name,
                    }
                )

        return tools_info

    async def get_tool_by_name(
        self, tool_name: str, server_name: Optional[str] = None
    ) -> Optional[BaseTool]:
        """
        Get a specific tool by name

        Args:
            tool_name: Name of the tool to retrieve
            server_name: Optional server to restrict the lookup

        Returns:
            BaseTool instance or None if not found
        """
        await self.get_tools()

        if server_name:
            for tool in self._server_tools.get(server_name, []):
                if tool.name == tool_name:
                    return tool
            return None

        candidates = self._tool_index.get(tool_name, [])
        return candidates[0] if candidates else None

    async def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool for testing purposes

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Dict with execution result and metadata

        Raises:
            ToolNotFoundError: If tool doesn't exist
            ToolExecutionError: If execution fails
        """
        tool = await self.get_tool_by_name(tool_name)
        if not tool:
            raise ToolNotFoundError(tool_name)

        # Determine server name
        server_name = self.get_server_for_tool(tool) or "unknown"

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
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "execution_time": execution_time,
                "tool_name": tool_name,
                "server_name": server_name,
            }

    def get_servers_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all configured servers

        Returns:
            Dict mapping server names to their status info
        """
        self._ensure_config_loaded()
        status = {}
        mcp_servers = self.config.get("mcp_servers", {})

        for server_name, server_info in mcp_servers.items():
            enabled = server_info.get("enabled", True)
            tool_count = len(self._server_tools.get(server_name, []))

            status[server_name] = {
                "enabled": enabled,
                "tool_count": tool_count,
                "transport": server_info.get("transport", "unknown"),
                "description": server_info.get("description", ""),
            }

        return status

    def get_server_info(self, server_name: str) -> Dict[str, Any]:
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
        if server_name not in self.config.get("mcp_servers", {}):
            raise ServerNotFoundError(server_name)

        server_config = self.config["mcp_servers"][server_name]
        tool_count = len(self._server_tools.get(server_name, []))

        return {
            "name": server_name,
            "enabled": server_config.get("enabled", True),
            "tool_count": tool_count,
            "config": server_config,
            "description": server_config.get("description", ""),
        }
