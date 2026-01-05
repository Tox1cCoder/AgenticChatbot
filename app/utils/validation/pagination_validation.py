"""
Error handling for pagination requests
"""

from fastapi import HTTPException, status


class PaginationError(HTTPException):
    """Custom exception for pagination errors"""

    def __init__(self, detail: str = "Invalid pagination parameters"):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


class InvalidPageError(PaginationError):
    """Exception for invalid page numbers"""

    def __init__(self, page: int):
        super().__init__(f"Invalid page number: {page}. Page must be >= 1")


class InvalidLimitError(PaginationError):
    """Exception for invalid limit values"""

    def __init__(self, limit: int):
        super().__init__(f"Invalid limit: {limit}. Limit must be between 1 and 100")


def validate_pagination_params(page: int, limit: int) -> None:
    """Validate pagination parameters and raise exceptions if invalid"""
    if page < 1:
        raise InvalidPageError(page)

    if limit < 1 or limit > 100:
        raise InvalidLimitError(limit)
