"""
MCP-related schemas for the client backend.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MCPServerStatus(str, Enum):
    """Status of an MCP server connection."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


class MCPServerType(str, Enum):
    """Type of MCP server transport."""

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"
    HTTP = "http"
    SSE = "sse"


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    name: str
    type: MCPServerType = MCPServerType.STDIO
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None  # For HTTP/SSE servers
    enabled: bool = True


class MCPServerState(BaseModel):
    """Runtime state of an MCP server."""

    name: str
    config: MCPServerConfig
    status: MCPServerStatus
    started_at: datetime | None = None
    error_message: str | None = None
    tools: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)


class MCPConfigFile(BaseModel):
    """Schema for the MCP configuration file."""

    version: str = "1.0"
    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class MCPToolSchema(BaseModel):
    """Schema for an MCP tool."""

    name: str
    description: str
    server_name: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPServerListResponse(BaseModel):
    """Response for listing MCP servers."""

    servers: list[MCPServerState]


class MCPServerStartRequest(BaseModel):
    """Request to start an MCP server."""

    server_name: str


class MCPServerStopRequest(BaseModel):
    """Request to stop an MCP server."""

    server_name: str


class MCPToolCallRequest(BaseModel):
    """Request to call an MCP tool."""

    server_name: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class MCPToolCallResponse(BaseModel):
    """Response from an MCP tool call."""

    success: bool
    result: Any = None
    error: str | None = None
    execution_time_ms: int
