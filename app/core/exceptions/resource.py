"""
Resource-related exception classes
"""

from fastapi import status
from .http import CustomHTTPException


class ResourceNotFoundException(CustomHTTPException):
    """Resource not found exceptions"""

    def __init__(
        self, detail: str = "Resource not found", error_code: str = "NOT_FOUND"
    ):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND, detail=detail, error_code=error_code
        )
