"""Project exception classes mapped to the API error contract.

Matches the convention already set by ``custom_agent.py``: 403 when the
resource exists but belongs to someone else, 404 only when it is genuinely
missing or soft-deleted.
"""

from fastapi import status

from .http import CustomHTTPException


class ProjectValidationError(CustomHTTPException):
    """400 — invalid name, instructions, or agent id list."""

    def __init__(
        self,
        detail: str = "Project validation failed",
        error_code: str = "PROJECT_VALIDATION_FAILED",
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST, detail=detail, error_code=error_code
        )


class ProjectForbiddenError(CustomHTTPException):
    """403 — the user does not own the project, conversation, or agent."""

    def __init__(
        self,
        detail: str = "Access denied to this project resource",
        error_code: str = "PROJECT_FORBIDDEN",
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN, detail=detail, error_code=error_code
        )


class ProjectNotFoundError(CustomHTTPException):
    """404 — the project is missing or soft-deleted."""

    def __init__(
        self,
        detail: str = "Project not found",
        error_code: str = "PROJECT_NOT_FOUND",
    ):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND, detail=detail, error_code=error_code
        )


class ProjectConversationNotFoundError(CustomHTTPException):
    """404 — the conversation is not a member of this project."""

    def __init__(
        self,
        detail: str = "Conversation is not in this project",
        error_code: str = "PROJECT_CONVERSATION_NOT_FOUND",
    ):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND, detail=detail, error_code=error_code
        )
