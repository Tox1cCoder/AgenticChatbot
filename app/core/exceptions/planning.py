from enum import Enum
from typing import Optional


class PauseReason(str, Enum):
    RECURSION_LIMIT = "recursion_limit"
    RATE_LIMIT = "rate_limit"
    MAX_TASKS_REACHED = "max_tasks_reached"


class PlanExecutionPausedException(Exception):
    def __init__(
        self,
        reason: PauseReason,
        user_message: str,
        original_error: Optional[str] = None,
    ):
        self.reason = reason
        self.user_message = user_message
        self.original_error = original_error
        super().__init__(user_message)

    @classmethod
    def from_exception(cls, exc: Exception) -> Optional["PlanExecutionPausedException"]:
        message = str(exc).lower()

        if "recursion limit" in message or "graph_recursion" in message:
            return cls(
                reason=PauseReason.RECURSION_LIMIT,
                user_message=(
                    "Plan execution paused because the workflow reached its limit. "
                    "Send a message to continue from the next task."
                ),
                original_error=str(exc),
            )

        rate_limit_markers = [
            "rate limit",
            "quota exceeded",
            "resourceexhausted",
            "too many requests",
            "429",
        ]
        if any(marker in message for marker in rate_limit_markers):
            return cls(
                reason=PauseReason.RATE_LIMIT,
                user_message=(
                    "Plan execution paused due to rate limiting. "
                    "Please wait a moment and send a message to continue."
                ),
                original_error=str(exc),
            )

        return None
