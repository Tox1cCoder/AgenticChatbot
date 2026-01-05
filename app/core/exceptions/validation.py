"""
Validation-related exception classes
"""

from fastapi import status
from .http import CustomHTTPException


class ValidationException(CustomHTTPException):
    """Validation-related exceptions"""

    def __init__(
        self, detail: str = "Validation failed", error_code: str = "VALIDATION_ERROR"
    ):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code=error_code,
        )


class FileValidationError(CustomHTTPException):
    """Exception raised when file validation fails"""

    def __init__(
        self,
        detail: str = "File validation failed",
        error_code: str = "FILE_VALIDATION_ERROR",
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code=error_code,
        )
