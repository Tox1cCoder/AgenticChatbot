"""
HTTP and general exception classes
"""

from fastapi import HTTPException, status


class CustomHTTPException(HTTPException):
    """Base custom HTTP exception"""

    def __init__(self, status_code: int, detail: str, error_code: str = None):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code


class DocumentProcessingError(CustomHTTPException):
    """Exception raised when document processing fails"""

    def __init__(
        self,
        detail: str = "Document processing failed",
        error_code: str = "DOCUMENT_PROCESSING_ERROR",
    ):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code=error_code,
        )
