"""
Response wrapper schemas for consistent API responses
DEPRECATED: Use individual response classes from app.schemas.responses instead
"""

# Re-export from the new location for backwards compatibility
from .responses import ApiResponse, ErrorResponse, SuccessResponse

__all__ = ["ApiResponse", "ErrorResponse", "SuccessResponse"]
