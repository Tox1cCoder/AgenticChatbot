from __future__ import annotations

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from app.utils.case_conversion import to_camel_case as to_camel


class MCPServerConfig(BaseModel):
    """Schema for MCP server configuration."""

    name: str = Field(..., description="Unique name for the MCP server")
    transport: str = Field(
        ...,
        description="Transport type: 'stdio' for command-based or 'http' for HTTP-based",
    )
    command: Optional[str] = Field(
        None,
        description="Command to execute for stdio transport (e.g., 'node', 'python')",
    )
    args: Optional[List[str]] = Field(
        None, description="Arguments for the command in stdio transport"
    )
    env: Optional[Dict[str, str]] = Field(
        None, description="Environment variables for stdio transport"
    )
    url: Optional[str] = Field(None, description="Base URL for HTTP transport")
    headers: Optional[Dict[str, str]] = Field(
        None, description="Headers for HTTP transport authentication"
    )
    enabled: bool = Field(default=True, description="Whether the server is enabled")
    description: Optional[str] = Field(
        None, description="Human-readable description of the server"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MCPServerInfo(BaseModel):
    """Schema for MCP server information response."""

    name: str = Field(..., description="Server name")
    transport: str = Field(..., description="Transport type (stdio/http)")
    enabled: bool = Field(..., description="Whether the server is enabled")
    description: Optional[str] = Field(None, description="Server description")
    tool_count: int = Field(..., description="Number of tools provided by this server")
    config: Dict[str, Any] = Field(..., description="Full server configuration")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MCPServerListResponse(BaseModel):
    """Schema for list of MCP servers response."""

    servers: List[MCPServerInfo] = Field(
        ..., description="List of configured MCP servers"
    )
    total_count: int = Field(..., description="Total number of servers")
    enabled_count: int = Field(..., description="Number of enabled servers")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MCPToolInfo(BaseModel):
    """Schema for MCP tool information."""

    name: str = Field(..., description="Tool name")
    description: Optional[str] = Field(None, description="Tool description")
    args_schema: Dict[str, Any] = Field(
        ..., description="JSON Schema for tool arguments"
    )
    server_name: str = Field(..., description="Name of the server providing this tool")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MCPToolListResponse(BaseModel):
    """Schema for list of MCP tools response."""

    tools: List[MCPToolInfo] = Field(..., description="List of available tools")
    total_count: int = Field(..., description="Total number of tools")
    servers_count: int = Field(..., description="Number of servers providing tools")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MCPToolExecuteRequest(BaseModel):
    """Schema for tool execution request."""

    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MCPToolExecuteResponse(BaseModel):
    """Schema for tool execution response."""

    success: bool = Field(..., description="Whether the tool execution succeeded")
    result: Optional[Any] = Field(None, description="Tool execution result")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time: float = Field(..., description="Execution time in seconds")
    tool_name: str = Field(..., description="Name of the executed tool")
    server_name: str = Field(
        ..., description="Name of the server that executed the tool"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MCPOperationResponse(BaseModel):
    """Schema for MCP operation responses (add/remove/toggle server)."""

    message: str = Field(..., description="Operation result message")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
