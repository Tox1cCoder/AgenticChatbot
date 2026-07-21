"""
Validation-related exception classes
"""

from fastapi import status

from .http import CustomHTTPException


class ValidationException(CustomHTTPException):
    """Validation-related exceptions"""

    def __init__(self, detail: str = "Validation failed", error_code: str = "VALIDATION_ERROR"):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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


class DuplicateDocumentFilenameError(CustomHTTPException):
    """Raised when a document filename already exists in the same conversation."""

    def __init__(
        self,
        detail: str = "A document with this filename already exists in this conversation.",
        error_code: str = "DUPLICATE_FILENAME",
    ):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code=error_code,
        )
