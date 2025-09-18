"""
Custom exception classes for categorizing application errors
"""

from fastapi import HTTPException, status


class CustomHTTPException(HTTPException):
    """Base custom HTTP exception"""

    def __init__(self, status_code: int, detail: str, error_code: str = None):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code


class AuthenticationException(CustomHTTPException):
    """Authentication-related exceptions"""

    def __init__(
        self, detail: str = "Authentication failed", error_code: str = "AUTH_FAILED"
    ):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code=error_code,
        )
        self.headers = {"WWW-Authenticate": "Bearer"}


class AuthorizationException(CustomHTTPException):
    """Authorization-related exceptions"""

    def __init__(
        self, detail: str = "Access denied", error_code: str = "ACCESS_DENIED"
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN, detail=detail, error_code=error_code
        )


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


class ResourceNotFoundException(CustomHTTPException):
    """Resource not found exceptions"""

    def __init__(
        self, detail: str = "Resource not found", error_code: str = "NOT_FOUND"
    ):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND, detail=detail, error_code=error_code
        )


class TokenExpiredException(AuthenticationException):
    """Token expired exception"""

    def __init__(
        self, detail: str = "Token has expired", error_code: str = "TOKEN_EXPIRED"
    ):
        super().__init__(detail=detail, error_code=error_code)
