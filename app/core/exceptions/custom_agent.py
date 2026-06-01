"""Custom-agent exception classes mapped to the API error contract."""

from fastapi import status

from .http import CustomHTTPException


class CustomAgentValidationError(CustomHTTPException):
    """400 — invalid prompt, model, tools, skills, names, or attachment order."""

    def __init__(
        self,
        detail: str = "Custom agent validation failed",
        error_code: str = "CUSTOM_AGENT_VALIDATION_FAILED",
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST, detail=detail, error_code=error_code
        )


class CustomAgentForbiddenError(CustomHTTPException):
    """403 — user does not own the agent, conversation, device, tool, or skill."""

    def __init__(
        self,
        detail: str = "Access denied to this custom agent resource",
        error_code: str = "CUSTOM_AGENT_FORBIDDEN",
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN, detail=detail, error_code=error_code
        )


class CustomAgentNotFoundError(CustomHTTPException):
    """404 — the custom agent is missing or soft-deleted."""

    def __init__(
        self,
        detail: str = "Custom agent not found",
        error_code: str = "CUSTOM_AGENT_NOT_FOUND",
    ):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND, detail=detail, error_code=error_code
        )


class CustomAgentInUseError(CustomHTTPException):
    """409 — update/delete/detach while the runtime agent is active or paused."""

    def __init__(
        self,
        detail: str = "Custom agent is currently active or paused and cannot be modified",
        error_code: str = "CUSTOM_AGENT_IN_USE",
    ):
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail, error_code=error_code)
