"""
Exception package - Backward compatibility module

Re-exports all exception classes from categorized modules to maintain backward compatibility.
"""

# Import all exceptions from categorized modules
from .auth import AuthenticationException, AuthorizationException, TokenExpiredException
from .validation import ValidationException, FileValidationError
from .resource import ResourceNotFoundException
from .http import CustomHTTPException, DocumentProcessingError

# Export all exceptions for backward compatibility
__all__ = [
    "CustomHTTPException",
    "AuthenticationException",
    "AuthorizationException",
    "TokenExpiredException",
    "ValidationException",
    "FileValidationError",
    "ResourceNotFoundException",
    "DocumentProcessingError",
]
