from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from ..services.event_streaming.events import V3StreamEvent


class IWorkflowRuntime(ABC):
    """Minimal workflow runtime port consumed by the service layer."""

    @abstractmethod
    async def initialize(self) -> None:
        """Warm workflow dependencies needed for request execution."""
        pass

    @abstractmethod
    async def execute_request(self, request: Any) -> Any:
        """Execute a workflow request and return the runtime response."""
        pass

    @abstractmethod
    async def execute_request_stream(self, request: Any) -> AsyncIterator[V3StreamEvent]:
        """Stream workflow execution events for a request."""
        pass

    @abstractmethod
    async def resume(self, thread_id: str, user_input: str | None = None) -> Any:
        """Resume a paused workflow thread."""
        pass

    @abstractmethod
    async def resume_with_decisions_stream(
        self,
        thread_id: str,
        decisions: list[Any],
    ) -> AsyncIterator[V3StreamEvent]:
        """Resume an interrupted workflow thread with explicit tool decisions."""
        pass

    @abstractmethod
    async def resume_with_continuation_stream(
        self,
        thread_id: str,
        resume: Any,
    ) -> AsyncIterator[V3StreamEvent]:
        """Resume a turn paused at its execution budget.

        Separate from ``resume_with_decisions_stream`` on purpose: that one
        answers a human-approval interrupt by addressing decisions to pending
        ``action_requests``, which a budget pause has none of.
        """
        pass

    @abstractmethod
    async def compact_checkpoint_after_terminal_response(self, thread_id: str | None) -> None:
        """Clear transient checkpoint transcript after durable response persistence."""
        pass

    @abstractmethod
    def invalidate_history_cache(self, conversation_id: str) -> None:
        """Drop any cached history tied to the given conversation."""
        pass
