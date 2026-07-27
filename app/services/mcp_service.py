"""Service layer for MCP (Model Context Protocol) operations."""

import logging
from typing import Any

from app.ai.mcp_integration import MCPManager, compute_catalog_version
from app.core.exceptions.mcp import (
    ServerConfigurationError,
    ToolNotFoundError,
)

logger = logging.getLogger(__name__)


class MCPService:
    """Business logic for MCP server and tool management"""

    def __init__(self, mcp_manager: MCPManager):
        """
        Initialize MCP Service

        Args:
            mcp_manager: MCPManager instance for MCP operations
        """
        self.mcp_manager = mcp_manager

    async def list_servers(self) -> dict[str, Any]:
        """
        List all configured MCP servers with status

        Returns:
            Dict with servers list and summary statistics
        """
        servers_status = self.mcp_manager.get_servers_status()
        configured_servers = self.mcp_manager.config.get("servers", {})

        servers_list = []
        enabled_count = 0

        for server_name, status in servers_status.items():
            if status["enabled"]:
                enabled_count += 1

            # Get full config for this server
            server_config = configured_servers.get(server_name, {})

            servers_list.append(
                {
                    "name": server_name,
                    "enabled": status["enabled"],
                    "tool_count": status["tool_count"],
                    "transport": status["transport"],
                    "description": status.get("description", ""),
                    "config": server_config,
                }
            )

        return {
            "servers": servers_list,
            "total_count": len(servers_list),
            "enabled_count": enabled_count,
        }

    async def get_server_details(self, server_name: str) -> dict[str, Any]:
        """
        Get detailed information about a specific server

        Args:
            server_name: Name of the server

        Returns:
            Dict with server details

        Raises:
            ServerNotFoundError: If server doesn't exist
        """
        return self.mcp_manager.get_server_info(server_name)

    async def add_server(self, server_config: dict[str, Any]) -> dict[str, str]:
        """
        Add a new MCP server

        Args:
            server_config: Server configuration dict

        Returns:
            Dict with success message

        Raises:
            ServerConfigurationError: If validation fails or server already exists
        """
        # Validate required fields
        server_name = server_config.get("name")
        if not server_name:
            raise ServerConfigurationError("Server name is required")

        transport = server_config.get("transport")
        if transport not in ["stdio", "http", "sse", "streamable_http"]:
            raise ServerConfigurationError(f"Invalid transport type: {transport}")

        # Validate transport-specific fields
        if transport == "stdio":
            if not server_config.get("command"):
                raise ServerConfigurationError("Command is required for stdio transport")
        elif transport in ["http", "sse", "streamable_http"] and not server_config.get("url"):
            raise ServerConfigurationError("URL is required for HTTP transport")

        config_copy = dict(server_config)
        name = config_copy.pop("name")

        self.mcp_manager.add_server(name, config_copy)

        # Reload tools to include new server
        await self.mcp_manager.reload_tools()

        logger.debug("Added server: %s", name)
        return {"message": f"Server '{name}' added successfully"}

    async def add_server_from_url(self, url_config: dict[str, Any]) -> dict[str, str]:
        """
        Add a new MCP server from a URL

        Args:
            url_config: Dict containing:
                - url: str - The URL string (npx command or HTTP URL)
                - name: Optional[str] - Custom server name
                - description: Optional[str] - Server description
                - enabled: bool - Whether to enable the server

        Returns:
            Dict with success message including generated server name

        Raises:
            ServerConfigurationError: If URL parsing or validation fails
        """
        from app.utils.mcp_url_parser import (
            generate_server_name_from_url,
            parse_mcp_url,
        )

        url = url_config.get("url", "").strip()
        if not url:
            raise ServerConfigurationError("URL is required")

        try:
            # Parse URL to get server configuration
            parsed_config = parse_mcp_url(url)

            # Generate or use provided server name
            server_name = url_config.get("name", "").strip()
            if not server_name:
                server_name = generate_server_name_from_url(url)

            # Build complete server config
            server_config = {
                "name": server_name,
                "enabled": url_config.get("enabled", True),
                **parsed_config,
            }

            # Add description if provided
            if url_config.get("description"):
                server_config["description"] = url_config["description"]

            # Use the existing add_server method
            await self.add_server(server_config)

            logger.info("Added server from URL: %s -> %s", url, server_name)
            return {"message": f"Server '{server_name}' added successfully from URL"}

        except ServerConfigurationError:
            # Re-raise configuration errors as-is
            raise
        except Exception as e:
            # Wrap other exceptions
            logger.error("Failed to add server from URL: %s", str(e))
            raise ServerConfigurationError(
                detail=f"Failed to parse URL: {str(e)}", error_code="URL_PARSING_ERROR"
            ) from e

    async def remove_server(self, server_name: str) -> dict[str, str]:
        """
        Remove an MCP server

        Args:
            server_name: Name of the server to remove

        Returns:
            Dict with success message

        Raises:
            ServerNotFoundError: If server doesn't exist
        """
        await self.mcp_manager.remove_server(server_name)

        # Reload tools to reflect removal
        await self.mcp_manager.reload_tools()

        logger.debug("Removed server: %s", server_name)
        return {"message": f"Server '{server_name}' removed successfully"}

    async def toggle_server(self, server_name: str, enabled: bool) -> dict[str, str]:
        """
        Enable or disable an MCP server

        Args:
            server_name: Name of the server
            enabled: True to enable, False to disable

        Returns:
            Dict with success message

        Raises:
            ServerNotFoundError: If server doesn't exist
        """
        if enabled:
            await self.mcp_manager.enable_server(server_name)
        else:
            await self.mcp_manager.disable_server(server_name)

        # Reload tools to reflect changes
        await self.mcp_manager.reload_tools()

        action = "enabled" if enabled else "disabled"
        logger.debug("%s server: %s", action.title(), server_name)
        return {"message": f"Server '{server_name}' {action} successfully"}

    # ===== Tool Management Methods =====

    async def list_tools(self, server_name: str | None = None) -> dict[str, Any]:
        """
        List all available tools, or tools from exactly one server when scoped.

        Scoping is performed in the catalog operation itself
        (``MCPManager.list_tool_descriptors``), NOT as a response-layer filter over
        the full catalog. A scoped request for an unknown/disabled server raises
        ``ServerNotFoundError`` (mapped to 404 at the API boundary).

        Args:
            server_name: Optional server name to scope to.

        Returns:
            Dict with the tools list, counts, the applied ``scope``, and a
            deterministic ``catalog_version``.
        """
        descriptors = await self.mcp_manager.list_tool_descriptors(server_name)

        if server_name is not None:
            # A known scoped server reports servers_count == 1 even when it
            # currently exposes zero tools (per the MCP endpoint contract).
            scope: dict[str, Any] = {"kind": "server", "serverName": server_name}
            servers_count = 1
        else:
            scope = {"kind": "all"}
            servers_count = len({t["server_name"] for t in descriptors})

        return {
            "tools": descriptors,
            "total_count": len(descriptors),
            "servers_count": servers_count,
            "catalog_version": compute_catalog_version(descriptors),
            "scope": scope,
        }

    async def get_tool_info(
        self, tool_name: str, server_name: str | None = None
    ) -> dict[str, Any]:
        """
        Get detailed information about a specific tool

        Args:
            tool_name: Name of the tool
            server_name: Owning server; required when several servers expose
                the same bare name

        Returns:
            Dict with tool information

        Raises:
            ToolNotFoundError: If tool doesn't exist
            AmbiguousToolNameError: If the unqualified name maps to several servers
        """
        tool = await self.mcp_manager.get_tool_by_name(tool_name, server_name=server_name)
        if not tool:
            raise ToolNotFoundError(tool_name)

        # Find server name and schema using manager utilities
        resolved_server = server_name or self.mcp_manager.get_server_for_tool(tool) or "unknown"
        args_schema = self.mcp_manager.get_tool_args_schema(tool)

        return {
            "name": tool.name,
            "description": tool.description or "",
            "args_schema": args_schema,
            "server_name": resolved_server,
        }

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
            server_name: Owning server; required when several servers expose
                the same bare name

        Returns:
            Dict with execution result and metadata

        Raises:
            ToolNotFoundError: If tool doesn't exist
            AmbiguousToolNameError: If the unqualified name maps to several servers
            ToolExecutionError: If execution fails
        """
        # Validate that tool exists (and that the bare name is unambiguous)
        tool = await self.mcp_manager.get_tool_by_name(tool_name, server_name=server_name)
        if not tool:
            raise ToolNotFoundError(tool_name)

        # Validate arguments against schema (basic validation)
        args_schema_def = getattr(tool, "args_schema", None)
        if args_schema_def and callable(args_schema_def):
            try:
                # Let Pydantic validate the arguments
                args_schema_def(**arguments)
            except Exception as e:
                logger.warning(f"Argument validation warning for {tool_name}: {e}")
                # Continue anyway - let the tool handle invalid args

        # Execute tool
        result = await self.mcp_manager.execute_tool(
            tool_name, arguments, server_name=server_name
        )

        if not result["success"]:
            logger.warning("Tool execution failed: %s - %s", tool_name, result["error"])

        return result
