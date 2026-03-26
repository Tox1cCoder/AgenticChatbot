"""
Runtime-related schemas for the client backend.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RuntimeStatus(str, Enum):
    """Status of the client runtime."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class HealthStatus(str, Enum):
    """Health status for the client backend."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint."""

    status: HealthStatus
    version: str
    uptime_seconds: float
    server_connected: bool
    device_id: str | None = None
    device_identifier: str | None = None
    checks: dict[str, bool] = Field(default_factory=dict)


class DeviceInfo(BaseModel):
    """Information about the local device."""

    device_id: str | None = None
    device_identifier: str
    device_name: str
    platform: str
    app_version: str
    runtime_version: str


class RuntimeState(BaseModel):
    """Current state of the runtime connection."""

    status: RuntimeStatus
    device_info: DeviceInfo | None = None
    server_url: str
    connected_at: datetime | None = None
    last_heartbeat: datetime | None = None
    session_id: str | None = None
    error_message: str | None = None


class ToolCatalogEntry(BaseModel):
    """Entry in the local tool catalog."""

    name: str
    description: str
    origin: str = Field(description="Tool origin: 'native', 'mcp'")
    server_name: str | None = Field(default=None, description="MCP server name if origin is 'mcp'")
    qualified_id: str = Field(description="Fully qualified tool identifier")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for tool input"
    )


class ToolCatalog(BaseModel):
    """Catalog of locally available tools."""

    tools: list[ToolCatalogEntry]
    generated_at: datetime
    version: str


class ToolDispatchRequest(BaseModel):
    """Request to execute a local tool."""

    request_id: str
    tool_name: str
    qualified_tool_id: str
    arguments: dict[str, Any]
    timeout_seconds: int = 30


class ToolDispatchResult(BaseModel):
    """Result of a local tool execution."""

    request_id: str
    success: bool
    result: Any = None
    error: str | None = None
    execution_time_ms: int
    truncated: bool = False
