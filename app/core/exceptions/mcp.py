from fastapi import status
from .http import CustomHTTPException


class MCPException(CustomHTTPException):
    """Base exception for MCP-related errors"""

    def __init__(
        self,
        detail: str = "MCP operation failed",
        error_code: str = "MCP_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(status_code=status_code, detail=detail, error_code=error_code)


class ServerNotFoundError(MCPException):
    """Raised when an MCP server is not found"""

    def __init__(self, server_name: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MCP server '{server_name}' not found",
            error_code="SERVER_NOT_FOUND",
        )


class ToolNotFoundError(MCPException):
    """Raised when an MCP tool is not found"""

    def __init__(self, tool_name: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MCP tool '{tool_name}' not found",
            error_code="TOOL_NOT_FOUND",
        )


class ToolExecutionError(MCPException):
    """Raised when MCP tool execution fails"""

    def __init__(self, tool_name: str, error_message: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool '{tool_name}' execution failed: {error_message}",
            error_code="TOOL_EXECUTION_ERROR",
        )


class ServerConfigurationError(MCPException):
    """Raised when MCP server configuration is invalid"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="SERVER_CONFIGURATION_ERROR",
        )
