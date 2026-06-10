from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.models.enums import MessageRole
from app.schemas.message import MessageCreate, MessageRead
from app.schemas.workflow import (
    WorkflowExecutionRequest,
    WorkflowPlanningContext,
    WorkflowResponse,
    WorkflowResponseMessage,
)
from app.services.event_streaming.events import V3StreamEvent, make_event
from app.services.message_service import MessageService


def _message_row(*, conversation_id, sender: int, content: str) -> SimpleNamespace:
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id=uuid4(),
        conversation_id=conversation_id,
        sender=sender,
        content=content,
        message_metadata={},
        feedback=None,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


@pytest.mark.asyncio
async def test_message_service_accumulates_v3_text_and_persists_once():
    conversation_id = uuid4()
    user_id = uuid4()
    persisted = []

    service = MessageService.__new__(MessageService)
    service.repository = SimpleNamespace(
        create=lambda entity: _message_row(
            conversation_id=conversation_id,
            sender=MessageRole.user.value,
            content=entity["content"],
        )
    )
    service.conversation_validation_utils = SimpleNamespace(
        validate_conversation_access=lambda *_args: None,
        conversation_repository=SimpleNamespace(
            get_by_id=lambda _conversation_id: SimpleNamespace(title="Existing chat")
        ),
    )
    workflow_request = WorkflowExecutionRequest(
        message="hello",
        conversation_id=str(conversation_id),
        user_id=str(user_id),
        planning=WorkflowPlanningContext(),
    )
    service._build_user_message_workflow_request = AsyncMock(
        return_value=(user_id, None, workflow_request)
    )

    async def source(_request):
        yield make_event("message_delta", sequence=1, data={"text": "hello"})
        yield make_event(
            "complete",
            sequence=2,
            data={
                "response": WorkflowResponse(
                    message=WorkflowResponseMessage(content="hello"),
                    metadata={},
                )
            },
        )

    service.ai_service = SimpleNamespace(
        invalidate_history_cache=lambda *_args: None,
        execute_request_stream=source,
    )
    service._persist_completed_workflow_response = AsyncMock(
        side_effect=lambda **kwargs: persisted.append(kwargs) or MessageRead.model_validate(
            _message_row(
                conversation_id=conversation_id,
                sender=MessageRole.assistant.value,
                content="hello",
            )
        )
    )
    service._compact_checkpoint_after_persist = AsyncMock()

    events = [
        event
        async for event in service.create_message_stream(
            MessageCreate(conversation_id=conversation_id, content="hello"),
            user_id,
        )
    ]

    assert all(isinstance(event, V3StreamEvent) for event in events)
    assert any(event.type == "message_delta" and event.data["text"] == "hello" for event in events)
    assert events[-1].type == "complete"
    assert events[-1].data["message"]["content"] == "hello"
    assert len(persisted) == 1


@pytest.mark.asyncio
async def test_resume_stream_accepts_v3_events_and_persists_once():
    conversation_id = uuid4()
    user_id = uuid4()
    bot_message_id = uuid4()
    persisted = []

    service = MessageService.__new__(MessageService)
    service.hitl_interrupt_repository = None
    service._validate_and_claim_interrupt_resume = lambda **_kwargs: None
    service._get_conversation_context = lambda *_args: (user_id, None)
    service._revalidate_resume_custom_agent = lambda *_args: None
    service._audit_interrupt_resume_decisions = lambda **_kwargs: None
    service._resolve_custom_agents_state = lambda *_args: {}
    service._clear_redis_interrupt = lambda *_args: None
    service._sync_response_plan_state = lambda **_kwargs: False

    async def resume_source(**_kwargs):
        yield make_event("message_delta", sequence=1, data={"text": "resumed"})
        yield make_event(
            "complete",
            sequence=2,
            data={
                "response": WorkflowResponse(
                    message=WorkflowResponseMessage(content="resumed"),
                    metadata={},
                )
            },
        )

    service.ai_service = SimpleNamespace(
        invalidate_history_cache=lambda *_args: None,
        resume_interrupted_execution_stream=resume_source,
    )
    service._persist_completed_workflow_response = AsyncMock(
        side_effect=lambda **kwargs: persisted.append(kwargs) or MessageRead.model_validate(
            _message_row(
                conversation_id=conversation_id,
                sender=MessageRole.assistant.value,
                content="resumed",
            )
        )
    )
    service._compact_checkpoint_after_persist = AsyncMock()

    events = [
        event
        async for event in service.resume_message_creation_stream(
            thread_id=str(conversation_id),
            conversation_id=conversation_id,
            user_id=user_id,
            decisions=[],
            bot_message_id=bot_message_id,
        )
    ]

    assert all(isinstance(event, V3StreamEvent) for event in events)
    assert any(
        event.type == "message_delta" and event.data["text"] == "resumed" for event in events
    )
    assert events[-1].type == "complete"
    assert events[-1].data["message"]["content"] == "resumed"
    assert len(persisted) == 1
