import logging
from typing import Optional
from fastapi import APIRouter, status, Query

from app.core.dependency_injection import AppAutoInjector
from app.services.mcp_service import MCPService
from app.schemas.mcp import (
    MCPServerConfig,
    MCPServerInfo,
    MCPServerListResponse,
    MCPToolInfo,
    MCPToolListResponse,
    MCPToolExecuteRequest,
    MCPToolExecuteResponse,
    MCPOperationResponse,
)
from app.schemas.responses import ApiResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["mcp"])


# ===== Server Endpoints =====


@router.get("/servers")
@AppAutoInjector.auto_inject()
async def list_servers(
    mcp_service: MCPService,
) -> ApiResponse[MCPServerListResponse]:
    """List all configured MCP servers with status"""
    result = await mcp_service.list_servers()
    response_data = MCPServerListResponse(**result)
    return ApiResponse(
        success=True, message="MCP servers retrieved successfully", data=response_data
    )


@router.get("/servers/{server_name}")
@AppAutoInjector.auto_inject()
async def get_server_details(
    server_name: str,
    mcp_service: MCPService,
) -> ApiResponse[MCPServerInfo]:
    """Get details about a specific MCP server"""
    result = await mcp_service.get_server_details(server_name)
    response_data = MCPServerInfo(
        name=result["name"],
        transport=result["config"]["transport"],
        enabled=result["enabled"],
        description=result.get("description", ""),
        tool_count=result["tool_count"],
        config=result["config"],
    )
    return ApiResponse(
        success=True,
        message=f"Server '{server_name}' details retrieved successfully",
        data=response_data,
    )


@router.post("/servers", status_code=status.HTTP_201_CREATED)
@AppAutoInjector.auto_inject()
async def add_server(
    server_config: MCPServerConfig,
    mcp_service: MCPService,
) -> ApiResponse[MCPOperationResponse]:
    """Add a new MCP server"""
    config_dict = server_config.model_dump(exclude_none=True)
    result = await mcp_service.add_server(config_dict)
    response_data = MCPOperationResponse(message=result["message"])
    return ApiResponse(success=True, message=result["message"], data=response_data)


@router.delete("/servers/{server_name}")
@AppAutoInjector.auto_inject()
async def remove_server(
    server_name: str,
    mcp_service: MCPService,
) -> ApiResponse[MCPOperationResponse]:
    """Remove an MCP server"""
    result = await mcp_service.remove_server(server_name)
    response_data = MCPOperationResponse(message=result["message"])
    return ApiResponse(success=True, message=result["message"], data=response_data)


@router.patch("/servers/{server_name}/toggle")
@AppAutoInjector.auto_inject()
async def toggle_server(
    server_name: str,
    mcp_service: MCPService,
    enabled: bool = Query(..., description="True to enable, False to disable"),
) -> ApiResponse[MCPOperationResponse]:
    """Enable or disable an MCP server"""
    result = await mcp_service.toggle_server(server_name, enabled)
    response_data = MCPOperationResponse(message=result["message"])
    return ApiResponse(success=True, message=result["message"], data=response_data)


@router.get("/tools")
@AppAutoInjector.auto_inject()
async def list_tools(
    mcp_service: MCPService,
    server_name: Optional[str] = Query(None, description="Filter by server name"),
) -> ApiResponse[MCPToolListResponse]:
    """List all available MCP tools"""
    result = await mcp_service.list_tools(server_name)
    response_data = MCPToolListResponse(**result)
    return ApiResponse(
        success=True, message="MCP tools retrieved successfully", data=response_data
    )


@router.get("/tools/{tool_name}")
@AppAutoInjector.auto_inject()
async def get_tool_details(
    tool_name: str,
    mcp_service: MCPService,
) -> ApiResponse[MCPToolInfo]:
    """Get detailed information about a specific tool"""
    result = await mcp_service.get_tool_info(tool_name)
    response_data = MCPToolInfo(**result)
    return ApiResponse(
        success=True,
        message=f"Tool '{tool_name}' details retrieved successfully",
        data=response_data,
    )


@router.post("/tools/{tool_name}/execute")
@AppAutoInjector.auto_inject()
async def execute_tool(
    tool_name: str,
    request: MCPToolExecuteRequest,
    mcp_service: MCPService,
) -> ApiResponse[MCPToolExecuteResponse]:
    """Execute a tool with provided arguments for testing"""
    result = await mcp_service.execute_tool(tool_name, request.arguments)
    response_data = MCPToolExecuteResponse(**result)

    if result["success"]:
        return ApiResponse(
            success=True,
            message=f"Tool '{tool_name}' executed successfully",
            data=response_data,
        )
    else:
        return ApiResponse(
            success=False,
            message=f"Tool '{tool_name}' execution failed: {result['error']}",
            data=response_data,
        )
