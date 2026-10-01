"""Service layer for MCP (Model Context Protocol) operations."""

import logging
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from app.ai.mcp_integration import MCPManager, compute_catalog_version
from app.core.exceptions import AuthorizationException
from app.core.exceptions.mcp import (
    ServerConfigurationError,
    ToolNotFoundError,
)
from app.core.mcp_adapter_utils import normalize_mcp_transport

logger = logging.getLogger(__name__)


_REDACTED = "***"


def _redact_url(url: str) -> str:
    """Mask query values and a userinfo password; parameter names stay visible.

    A query part with no ``=`` is masked whole, since it may be the key itself.
    A URL that cannot be parsed is masked whole.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return _REDACTED
    netloc = parts.netloc
    if parts.password is not None:
        userinfo, host = netloc.rsplit("@", 1)
        netloc = f"{userinfo.split(':', 1)[0]}:{_REDACTED}@{host}"
    query = "&".join(
        f"{segment.split('=', 1)[0]}={_REDACTED}" if "=" in segment else _REDACTED
        for segment in parts.query.split("&")
        if segment
    )
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def redact_server_config(config: dict[str, Any]) -> dict[str, Any]:
    """A server config safe to return to a client.

    ``env`` and ``headers`` hold API keys and bearer tokens, and an HTTP
    server's ``url`` can carry one in its query string. Every signed-in user
    can list servers, so their values are masked; the names stay, so a client
    can still see which variables and parameters a server is configured with.
    """
    redacted = dict(config)
    for field in ("env", "headers"):
        values = redacted.get(field)
        if isinstance(values, dict):
            redacted[field] = dict.fromkeys(values, _REDACTED)
    url = redacted.get("url")
    if isinstance(url, str) and url:
        redacted["url"] = _redact_url(url)
    return redacted


class MCPService:
    def __init__(self, mcp_manager: MCPManager, allow_api_stdio: bool = False):
        self.mcp_manager = mcp_manager
        self.allow_api_stdio = allow_api_stdio

    def _require_api_transport(self, transport: str | None) -> None:
        if normalize_mcp_transport(transport) == "stdio" and not self.allow_api_stdio:
            raise AuthorizationException(
                detail="Configuring or enabling stdio MCP servers through the API is disabled",
                error_code="API_STDIO_DISABLED",
            )

    async def list_servers(self) -> dict[str, Any]:
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
                    "config": redact_server_config(server_config),
                }
            )

        return {
            "servers": servers_list,
            "total_count": len(servers_list),
            "enabled_count": enabled_count,
        }

    async def get_server_details(self, server_name: str) -> dict[str, Any]:
        details = dict(self.mcp_manager.get_server_info(server_name))
        details["config"] = redact_server_config(details["config"])
        return details

    async def add_server(self, server_config: dict[str, Any]) -> dict[str, str]:
        server_name = server_config.get("name")
        if not server_name:
            raise ServerConfigurationError("Server name is required")

        transport = server_config.get("transport")
        self._require_api_transport(transport)
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
        """Add a server from an npx command or an HTTP URL."""
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

            # Not the URL: hosted MCP servers take their API key in the query.
            logger.info("Added server from URL as %s", server_name)
            return {"message": f"Server '{server_name}' added successfully from URL"}

        except (ServerConfigurationError, AuthorizationException):
            # Preserve configuration and authorization responses from the shared add path.
            raise
        except Exception as e:
            # Wrap other exceptions
            logger.error("Failed to add server from URL: %s", type(e).__name__)
            raise ServerConfigurationError(
                detail=f"Failed to parse URL: {str(e)}", error_code="URL_PARSING_ERROR"
            ) from e

    async def remove_server(self, server_name: str) -> dict[str, str]:
        await self.mcp_manager.remove_server(server_name)

        # Reload tools to reflect removal
        await self.mcp_manager.reload_tools()

        logger.debug("Removed server: %s", server_name)
        return {"message": f"Server '{server_name}' removed successfully"}

    async def toggle_server(self, server_name: str, enabled: bool) -> dict[str, str]:
        if enabled:
            config = self.mcp_manager.get_server_info(server_name)["config"]
            self._require_api_transport(config.get("transport"))
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

    async def get_tool_info(self, tool_name: str, server_name: str | None = None) -> dict[str, Any]:
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
                logger.warning("Argument validation warning for %s: %s", tool_name, e)
                # Continue anyway - let the tool handle invalid args

        # Execute tool
        result = await self.mcp_manager.execute_tool(tool_name, arguments, server_name=server_name)

        if not result["success"]:
            logger.warning("Tool execution failed: %s - %s", tool_name, result["error"])

        return result
