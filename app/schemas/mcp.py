from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.utils.case_conversion import to_camel_case as to_camel


class MCPServerConfig(BaseModel):
    """Schema for MCP server configuration."""

    name: str = Field(..., description="Unique name for the MCP server")
    transport: str = Field(
        ...,
        description="Transport type: 'stdio' for command-based or 'http' for HTTP-based",
    )
    command: str | None = Field(
        None,
        description="Command to execute for stdio transport (e.g., 'node', 'python')",
    )
    args: list[str] | None = Field(None, description="Arguments for the command in stdio transport")
    env: dict[str, str] | None = Field(
        None, description="Environment variables for stdio transport"
    )
    url: str | None = Field(None, description="Base URL for HTTP transport")
    headers: dict[str, str] | None = Field(
        None, description="Headers for HTTP transport authentication"
    )
    enabled: bool = Field(default=True, description="Whether the server is enabled")
    description: str | None = Field(None, description="Human-readable description of the server")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MCPServerInfo(BaseModel):
    """Schema for MCP server information response."""

    name: str = Field(..., description="Server name")
    transport: str = Field(..., description="Transport type (stdio/http)")
    enabled: bool = Field(..., description="Whether the server is enabled")
    description: str | None = Field(None, description="Server description")
    tool_count: int = Field(..., description="Number of tools provided by this server")
    config: dict[str, Any] = Field(..., description="Full server configuration")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MCPServerListResponse(BaseModel):
    """Schema for list of MCP servers response."""

    servers: list[MCPServerInfo] = Field(..., description="List of configured MCP servers")
    total_count: int = Field(..., description="Total number of servers")
    enabled_count: int = Field(..., description="Number of enabled servers")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MCPToolInfo(BaseModel):
    """Schema for MCP tool information."""

    name: str = Field(..., description="Tool name")
    description: str | None = Field(None, description="Tool description")
    args_schema: dict[str, Any] = Field(..., description="JSON Schema for tool arguments")
    server_name: str = Field(..., description="Name of the server providing this tool")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MCPToolListResponse(BaseModel):
    """Schema for list of MCP tools response."""

    tools: list[MCPToolInfo] = Field(..., description="List of available tools")
    total_count: int = Field(..., description="Total number of tools")
    servers_count: int = Field(..., description="Number of servers providing tools")
    scope: dict[str, Any] | None = Field(
        None,
        description="Applied scope, e.g. {'kind': 'server', 'serverName': ...} or {'kind': 'all'}",
    )
    catalog_version: str | None = Field(
        None,
        description="Deterministic content hash of the returned catalog (sha256:...)",
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MCPToolExecuteRequest(BaseModel):
    """Schema for tool execution request."""

    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MCPToolExecuteResponse(BaseModel):
    """Schema for tool execution response."""

    success: bool = Field(..., description="Whether the tool execution succeeded")
    result: Any | None = Field(None, description="Tool execution result")
    error: str | None = Field(None, description="Error message if execution failed")
    execution_time: float = Field(..., description="Execution time in seconds")
    tool_name: str = Field(..., description="Name of the executed tool")
    server_name: str = Field(..., description="Name of the server that executed the tool")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MCPServerURLConfig(BaseModel):
    """Schema for adding MCP server from URL."""

    url: str = Field(..., description="The URL string (npx command or HTTP URL)")
    name: str | None = Field(
        None,
        description="Optional custom server name (auto-generated if not provided)",
    )
    description: str | None = Field(None, description="Optional description")
    enabled: bool = Field(default=True, description="Whether to enable the server")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MCPOperationResponse(BaseModel):
    """Schema for MCP operation responses (add/remove/toggle server)."""

    message: str = Field(..., description="Operation result message")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
