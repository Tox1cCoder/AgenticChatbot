import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterable, TYPE_CHECKING

from anyio import ClosedResourceError, BrokenResourceError

from app.core.config import settings
from app.core.exceptions.mcp import (
    ServerNotFoundError,
    ToolNotFoundError,
    ServerConfigurationError,
)
from .utils import get_error_recovery_hint

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import BaseTool
from pydantic import BaseModel as PydanticBaseModel

if TYPE_CHECKING:
    from .mcp_registry import MCPRegistry

logger = logging.getLogger(__name__)

logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)


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
        self._session_contexts: Dict[str, Any] = {}  # Store context managers
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

                # Pass current working directory if specified
                if "cwd" in server_info:
                    server_config[server_name]["cwd"] = server_info["cwd"]

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
                continue

        return server_config

    def _get_enabled_server_names(self) -> List[str]:
        """Return names of all enabled servers from configuration."""
        self._ensure_config_loaded()
        servers = self.config.get("mcp_servers", {})
        return [name for name, cfg in servers.items() if cfg.get("enabled", True)]

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
        return bool(self.client)

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
            return self._server_tools[server_name]

        try:
            # Create and enter session context
            session_context = self.client.session(server_name)
            session = await session_context.__aenter__()

            tools = list(await load_mcp_tools(session))

            for tool in tools:
                if hasattr(tool, "name") and ":" in tool.name:
                    # Remove the prefix before the colon
                    tool.name = tool.name.split(":", 1)[-1]

            cleaned_tools = self._clean_tool_schemas(tools)

            # Store context and session for proper cleanup
            self._session_contexts[server_name] = {
                "context": session_context,
                "session": session,
            }
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

    def _filter_schema_recursively(self, schema: Any) -> Any:
        unsupported_keys = {"$schema", "additionalProperties"}

        if isinstance(schema, dict):
            filtered = {}
            for key, value in schema.items():
                if key in unsupported_keys:
                    continue

                # Skip None values
                if value is None:
                    continue

                # Recursively filter nested structures
                if key in (
                    "properties",
                    "items",
                    "anyOf",
                    "allOf",
                    "oneOf",
                    "definitions",
                ):
                    filtered_value = self._filter_schema_recursively(value)
                    # Only include if not empty after filtering
                    if filtered_value:
                        filtered[key] = filtered_value
                elif isinstance(value, dict):
                    filtered_value = self._filter_schema_recursively(value)
                    if filtered_value:  # Only include non-empty dicts
                        filtered[key] = filtered_value
                elif isinstance(value, list):
                    filtered[key] = [
                        self._filter_schema_recursively(item)
                        for item in value
                        if item is not None
                    ]
                else:
                    filtered[key] = value
            return filtered
        elif isinstance(schema, list):
            return [
                self._filter_schema_recursively(item)
                for item in schema
                if item is not None
            ]
        else:
            return schema

    def _remove_non_string_enums(self, schema: Any) -> Any:
        if isinstance(schema, dict):
            cleaned: Dict[str, Any] = {}
            for key, value in schema.items():
                if key == "enum" and isinstance(value, list):
                    if any(not isinstance(item, str) for item in value):
                        continue
                cleaned[key] = self._remove_non_string_enums(value)
            return cleaned
        if isinstance(schema, list):
            return [self._remove_non_string_enums(item) for item in schema]
        return schema

    def _clean_tool_schemas(self, tools: List[BaseTool]) -> List[BaseTool]:

        cleaned_tools = []
        for tool in tools:
            # Check if tool has an args_schema
            tool_had_no_schema = (
                not hasattr(tool, "args_schema") or tool.args_schema is None
            )
            if tool_had_no_schema:
                # Add empty properties schema for tools without args_schema
                # Only set to dict for tools that originally had None
                tool.args_schema = {"type": "object", "properties": {}}
                cleaned_tools.append(tool)
                continue

            args_schema = tool.args_schema

            # Some MCP adapters expose JSON-schema dicts directly
            # Sanitize those in-place (schema-only)
            if isinstance(args_schema, dict):
                filtered = self._filter_schema_recursively(args_schema)
                tool.args_schema = self._remove_non_string_enums(filtered)

                # Ensure OpenAI-compatible schema: must have type and properties
                if not tool.args_schema.get("properties"):
                    tool.args_schema["properties"] = {}
                if not tool.args_schema.get("type"):
                    tool.args_schema["type"] = "object"

                cleaned_tools.append(tool)
                continue

            if isinstance(args_schema, type) and issubclass(
                args_schema, PydanticBaseModel
            ):
                # Get the original schema
                original_schema_method = args_schema.model_json_schema

                # Create a wrapper that filters unsupported keys
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

    def _serialize_args_schema(self, schema: Any) -> Dict[str, Any]:
        """Normalize a tool args schema into a serializable dictionary."""
        if not schema:
            return {}

        result_schema = {}

        if isinstance(schema, dict):
            result_schema = schema
        elif PydanticBaseModel is not None:
            if isinstance(schema, type) and issubclass(schema, PydanticBaseModel):
                if hasattr(schema, "model_json_schema"):
                    result_schema = schema.model_json_schema()
            elif isinstance(schema, PydanticBaseModel):
                if hasattr(schema, "model_json_schema"):
                    result_schema = schema.model_json_schema()

        if not result_schema:
            for attr_name in ("model_json_schema", "json_schema", "schema"):
                exporter = getattr(schema, attr_name, None)
                if callable(exporter):
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

        unsupported_keys = {"$schema", "additionalProperties"}
        if result_schema and isinstance(result_schema, dict):
            result_schema = self._filter_schema_recursively(result_schema)

        return result_schema

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
        """Cleanup MCP client resources and properly close all sessions"""
        # Properly close all session contexts in this task
        for server_name, session_info in list(self._session_contexts.items()):
            try:
                context = session_info["context"]
                await asyncio.shield(context.__aexit__(None, None, None))
                logger.debug(f"Closed session for server: {server_name}")
            except Exception as e:
                logger.warning(f"Error closing session for {server_name}: {e}")

        self._session_contexts.clear()
        self._tools = []
        self._server_tools.clear()
        self._tool_index.clear()
        self._tool_server_map.clear()

        if self.client:
            self.client = None
            logger.debug("MCP client cleaned up")

    def add_server(self, server_name: str, server_config: Dict[str, Any]) -> None:
        self._ensure_config_loaded()

        if "mcp_servers" not in self.config:
            self.config["mcp_servers"] = {}

        if server_name in self.DEFAULT_SERVERS:
            raise ServerConfigurationError(
                f"Server '{server_name}' is reserved and managed by the system"
            )

        if server_name in self.config["mcp_servers"]:
            raise ServerConfigurationError(f"Server '{server_name}' already exists")

        self.config["mcp_servers"][server_name] = server_config
        self.save_config()

        # Notify registry of configuration change
        self._notify_registry_change()

    async def remove_server(self, server_name: str) -> None:
        self._ensure_config_loaded()
        if server_name not in self.config.get("mcp_servers", {}):
            raise ServerNotFoundError(server_name)
        if server_name in self.DEFAULT_SERVERS:
            raise ServerConfigurationError(f"Cannot remove core server '{server_name}'")

        # Properly close session context if it exists
        if server_name in self._session_contexts:
            try:
                context = self._session_contexts[server_name]["context"]
                await context.__aexit__(None, None, None)
                logger.debug(f"Closed session for server: {server_name}")
            except Exception as e:
                logger.warning(f"Error closing session for {server_name}: {e}")
            del self._session_contexts[server_name]

        # Remove cached tools
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

        # Notify registry of configuration change
        self._notify_registry_change()

    async def enable_server(self, server_name: str) -> None:
        self._ensure_config_loaded()
        if server_name not in self.config.get("mcp_servers", {}):
            raise ServerNotFoundError(server_name)

        self.config["mcp_servers"][server_name]["enabled"] = True
        self.save_config()

        # Notify registry of configuration change
        self._notify_registry_change()

    async def disable_server(self, server_name: str) -> None:
        self._ensure_config_loaded()
        if server_name not in self.config.get("mcp_servers", {}):
            raise ServerNotFoundError(server_name)

        # Properly close session context if it exists
        if server_name in self._session_contexts:
            try:
                context = self._session_contexts[server_name]["context"]
                await asyncio.shield(context.__aexit__(None, None, None))
                logger.debug(f"Closed session for server: {server_name}")
            except RuntimeError as e:
                if "different task" in str(e) or "already running" in str(e):
                    logger.warning(
                        f"Cannot close session for {server_name} from different task"
                    )
                else:
                    logger.warning(f"Error closing session for {server_name}: {e}")
            except Exception as e:
                logger.warning(f"Error closing session for {server_name}: {e}")
            del self._session_contexts[server_name]

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

    @staticmethod
    def _is_session_error(error: Exception) -> bool:
        """Check if an error indicates a dead/closed MCP session."""
        return isinstance(error, (ClosedResourceError, BrokenResourceError))

    async def reconnect_server(self, server_name: str) -> List[BaseTool]:
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
            try:
                await asyncio.shield(old["context"].__aexit__(None, None, None))
            except Exception:
                pass  # best-effort cleanup

        # 2. Drop cached tools so get_server_tools re-creates everything
        removed_tools = self._server_tools.pop(server_name, [])
        for tool in removed_tools:
            self._tool_server_map.pop(id(tool), None)
            indexed = self._tool_index.get(tool.name)
            if indexed:
                self._tool_index[tool.name] = [
                    t for t in indexed if t is not tool
                ]
                if not self._tool_index[tool.name]:
                    del self._tool_index[tool.name]
        self._tools = [t for t in self._tools if t not in removed_tools]

        # 3. Re-create client entry if needed (config unchanged)
        if not self.client:
            await self.initialize()

        # 4. Load fresh tools via a new session
        fresh_tools = await self.get_server_tools(server_name)

        # Rebuild the combined tools list
        combined: List[BaseTool] = []
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
        self, tool_name: str
    ) -> Optional[BaseTool]:
        """
        Reconnect whichever server owns *tool_name* and return a fresh tool.

        Returns ``None`` when the server cannot be determined or the tool no
        longer appears after reconnect.
        """
        # Determine which server provided this tool
        server_name: Optional[str] = None
        for sname, tools in self._server_tools.items():
            if any(t.name == tool_name for t in tools):
                server_name = sname
                break

        if not server_name:
            # Fallback: look at tool_server_map via the stale index
            for tool in self._tool_index.get(tool_name, []):
                server_name = self._tool_server_map.get(id(tool))
                if server_name:
                    break

        if not server_name:
            logger.warning(
                "Cannot reconnect for tool '%s': server unknown", tool_name
            )
            return None

        await self.reconnect_server(server_name)
        return self._tool_index.get(tool_name, [None])[0]

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
        except (ClosedResourceError, BrokenResourceError) as session_err:
            # MCP session died – try to reconnect once and retry
            logger.warning(
                "Session error executing tool '%s' on server '%s': %s. "
                "Attempting reconnect…",
                tool_name,
                server_name,
                session_err,
            )
            try:
                fresh_tool = await self.reconnect_and_get_tool(tool_name)
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


# =============================================================================
# Module-level functions - delegate to MCPRegistry for unified instance management
# =============================================================================


async def get_global_mcp_manager() -> MCPManager:
    """
    Get or create a singleton MCPManager instance.

    This function delegates to MCPRegistry to ensure a single shared instance
    is used by both agents and the DI container.

    Returns:
        MCPManager: The global MCP manager instance with tools pre-loaded.
    """
    from .mcp_registry import MCPRegistry

    return await MCPRegistry.get_manager_async()


async def reset_global_mcp_manager() -> None:
    """
    Reset the global MCP manager.

    Delegates to MCPRegistry for unified cleanup.
    """
    from .mcp_registry import MCPRegistry

    await MCPRegistry.reset()
    logger.debug("Global MCP manager reset via registry")
