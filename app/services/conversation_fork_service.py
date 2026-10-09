"""Fork a conversation through one completed assistant reply.

The service owns the boundary: request identity (the fingerprint), the source
turn lock, translating every refusal into the API's typed HTTP errors, and the
post-commit compaction notification. The snapshot itself is one transaction in
:class:`~app.repositories.conversation_fork.ConversationForkRepository`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any
from uuid import UUID

from app.ai.workflow.contracts import WorkflowRoutingException
from app.core.exceptions import AuthorizationException, ResourceNotFoundException
from app.core.exceptions.http import CustomHTTPException
from app.repositories.conversation_fork import (
    ConversationForkRepository,
    ForkError,
    ForkErrorCode,
    ForkPlan,
)
from app.schemas.conversation import ConversationForkRequest, ConversationRead
from app.services.conversation_service import ConversationService
from app.services.conversation_turn_coordinator import ConversationTurnCoordinator

logger = logging.getLogger(__name__)

_TURN_CONFLICT = ForkErrorCode.TURN_CONFLICT.value

#: Status and display text per refusal. Source missing/denied are absent on
#: purpose: they reuse the existing conversation exceptions below.
_HTTP_ERRORS: dict[ForkErrorCode, tuple[int, str]] = {
    ForkErrorCode.MESSAGE_NOT_FOUND: (404, "The selected message is not in this conversation"),
    ForkErrorCode.POINT_NOT_COMPLETED: (409, "Only a completed assistant reply can be forked"),
    ForkErrorCode.REQUEST_CONFLICT: (409, "This request ID was already used for another fork"),
    ForkErrorCode.RESULT_DELETED: (410, "The conversation created by this request was deleted"),
    ForkErrorCode.DOCUMENT_CONTEXT_UNSUPPORTED: (
        409,
        "Conversations with uploaded documents cannot be forked yet",
    ),
    ForkErrorCode.HISTORY_TOO_LARGE: (413, "The conversation history is too large to fork"),
    ForkErrorCode.TURN_CONFLICT: (409, "The conversation is busy; retry the fork"),
}


def fork_request_fingerprint(
    *, source_conversation_id: UUID, message_id: UUID, title: str | None
) -> str:
    """SHA-256 of the canonical JSON of what the client asked for.

    The *requested* title is hashed (null included), never the generated
    default, so a retry is recognized even after the source is renamed.
    """
    canonical = json.dumps(
        {
            "source_conversation_id": str(source_conversation_id),
            "message_id": str(message_id),
            "title": title,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ConversationForkService:
    """Creates, or replays, the child conversation for one fork request."""

    def __init__(
        self,
        *,
        repository: ConversationForkRepository,
        turn_coordinator: ConversationTurnCoordinator,
        compaction_publisher: Callable[[UUID], Any] | None,
        lock_timeout_seconds: float,
        max_messages: int,
        max_bytes: int,
    ) -> None:
        self.repository = repository
        self.turn_coordinator = turn_coordinator
        self.compaction_publisher = compaction_publisher
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.max_messages = int(max_messages)
        self.max_bytes = int(max_bytes)

    async def fork_conversation(
        self,
        *,
        source_conversation_id: UUID,
        owner_id: UUID,
        request: ConversationForkRequest,
    ) -> tuple[ConversationRead, bool]:
        """Return ``(child, created)``; ``created`` is True only for a new child."""
        plan = self._plan(source_conversation_id, owner_id, request)
        try:
            child, created = await self._fork(plan)
        except ForkError as exc:
            raise _http_error(exc.code) from exc
        except WorkflowRoutingException as exc:
            if exc.error.code != _TURN_CONFLICT:
                raise
            raise _http_error(ForkErrorCode.TURN_CONFLICT) from exc
        if created:
            self._publish_compaction(child.id)
        return ConversationService._convert_to_read_schema(child), created

    async def _fork(self, plan: ForkPlan) -> tuple[Any, bool]:
        replay = await self.repository.apreflight(plan=plan)
        if replay is not None:
            return replay, False
        async with self.turn_coordinator.hold(
            str(plan.source_conversation_id), request_id=str(plan.request_id)
        ):
            return await self.repository.afork_snapshot(plan=plan)

    def _plan(
        self, source_conversation_id: UUID, owner_id: UUID, request: ConversationForkRequest
    ) -> ForkPlan:
        return ForkPlan(
            source_conversation_id=source_conversation_id,
            owner_id=owner_id,
            message_id=request.message_id,
            request_id=request.request_id,
            fingerprint=fork_request_fingerprint(
                source_conversation_id=source_conversation_id,
                message_id=request.message_id,
                title=request.title,
            ),
            title=request.title,
            max_messages=self.max_messages,
            max_bytes=self.max_bytes,
            lock_timeout_seconds=self.lock_timeout_seconds,
        )

    def _publish_compaction(self, child_id: UUID) -> None:
        """Best effort: the pending job is committed and the reconcile sweep dispatches it."""
        if self.compaction_publisher is None:
            return
        try:
            self.compaction_publisher(child_id)
        except Exception:
            logger.warning(
                "Fork compaction notification failed code=broker_publish_failed conversation_id=%s",
                child_id,
                exc_info=True,
            )


def _http_error(code: ForkErrorCode) -> CustomHTTPException:
    if code is ForkErrorCode.SOURCE_NOT_FOUND:
        return ResourceNotFoundException(detail="Conversation not found", error_code=code.value)
    if code is ForkErrorCode.SOURCE_ACCESS_DENIED:
        return AuthorizationException(
            detail="Access denied to this conversation", error_code=code.value
        )
    status_code, detail = _HTTP_ERRORS[code]
    return CustomHTTPException(status_code=status_code, detail=detail, error_code=code.value)


__all__ = ["ConversationForkService", "fork_request_fingerprint"]
