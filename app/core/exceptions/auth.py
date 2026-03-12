"""
Authentication-related exception classes
"""

from fastapi import status

from .http import CustomHTTPException


class AuthenticationException(CustomHTTPException):
    """Authentication-related exceptions"""

    def __init__(self, detail: str = "Unauthenticated", error_code: str = "unauthenticated"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code=error_code,
        )
        self.headers = {"WWW-Authenticate": "Bearer"}


class AuthorizationException(CustomHTTPException):
    """Authorization-related exceptions"""

    def __init__(self, detail: str = "Access denied", error_code: str = "ACCESS_DENIED"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN, detail=detail, error_code=error_code
        )


class TokenExpiredException(AuthenticationException):
    """Token expired exception"""

    def __init__(self, detail: str = "Token has expired", error_code: str = "TOKEN_EXPIRED"):
        super().__init__(detail=detail, error_code=error_code)
