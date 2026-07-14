from __future__ import annotations

import asyncio
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
        side_effect=lambda **kwargs: (
            persisted.append(kwargs)
            or MessageRead.model_validate(
                _message_row(
                    conversation_id=conversation_id,
                    sender=MessageRole.assistant.value,
                    content="hello",
                )
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
        side_effect=lambda **kwargs: (
            persisted.append(kwargs)
            or MessageRead.model_validate(
                _message_row(
                    conversation_id=conversation_id,
                    sender=MessageRole.assistant.value,
                    content="resumed",
                )
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


def _claimed_resume_service(*, source, lifecycle_events):
    conversation_id = uuid4()
    user_id = uuid4()
    interrupt_id = "claimed-interrupt"

    service = MessageService.__new__(MessageService)
    service.hitl_interrupt_repository = SimpleNamespace(
        mark_failed=lambda identifier, *, resolution_source: lifecycle_events.append(
            ("failed", identifier, resolution_source)
        )
    )
    service._validate_and_claim_interrupt_resume = lambda **_kwargs: SimpleNamespace()
    service._get_conversation_context = lambda *_args: (user_id, None)
    service._revalidate_resume_custom_agent = lambda *_args: None
    service._audit_interrupt_resume_decisions = lambda **_kwargs: None
    service._resolve_custom_agents_state = lambda *_args: {}
    service._clear_redis_interrupt = lambda *_args: None
    service._sync_response_plan_state = lambda **_kwargs: False
    service._create_bot_response_message = lambda **_kwargs: SimpleNamespace(
        model_dump=lambda **_kwargs: {},
    )
    service.ai_service = SimpleNamespace(
        invalidate_history_cache=lambda *_args: None,
        resume_interrupted_execution_stream=source,
    )

    return service, conversation_id, user_id, interrupt_id


class _PausedRegistryRecorder:
    def __init__(self):
        self.calls = []

    def clear_paused_for_conversation(self, user_id, conversation_id):
        self.calls.append((user_id, conversation_id))


@pytest.mark.asyncio
async def test_claimed_resume_stream_error_marks_interrupt_failed_before_terminal_event():
    lifecycle_events = []

    async def source(**_kwargs):
        yield make_event("error", sequence=1, data={"error": "upstream failure"})

    service, conversation_id, user_id, interrupt_id = _claimed_resume_service(
        source=source,
        lifecycle_events=lifecycle_events,
    )

    events = []
    async for event in service.resume_message_creation_stream(
        thread_id=str(conversation_id),
        conversation_id=conversation_id,
        user_id=user_id,
        decisions=[],
        interrupt_id=interrupt_id,
    ):
        lifecycle_events.append(("event", event.type))
        events.append(event)

    assert events[-1].type == "error"
    assert events[-1].data["error_code"] == "INTERRUPT_FAILED"
    assert events[-1].data["status_code"] == 500
    assert lifecycle_events == [
        ("failed", interrupt_id, "stream_error"),
        ("event", "error"),
    ]


@pytest.mark.asyncio
async def test_closing_claimed_resume_stream_marks_interrupt_failed_for_client_disconnect(
    monkeypatch,
):
    lifecycle_events = []

    async def source(**_kwargs):
        yield make_event("message_delta", sequence=1, data={"text": ""})
        await asyncio.Event().wait()

    service, conversation_id, user_id, interrupt_id = _claimed_resume_service(
        source=source,
        lifecycle_events=lifecycle_events,
    )
    registry = _PausedRegistryRecorder()
    monkeypatch.setattr(
        "app.services.message_service.get_generation_registry",
        lambda: registry,
    )
    stream = service.resume_message_creation_stream(
        thread_id=str(conversation_id),
        conversation_id=conversation_id,
        user_id=user_id,
        decisions=[],
        interrupt_id=interrupt_id,
    )

    assert (await anext(stream)).type == "message_delta"
    await stream.aclose()

    assert ("failed", interrupt_id, "client_disconnect") in lifecycle_events
    assert registry.calls == [(user_id, conversation_id)]


@pytest.mark.asyncio
async def test_claimed_resume_setup_exception_marks_failed_and_yields_typed_error(monkeypatch):
    lifecycle_events = []

    async def source(**_kwargs):
        yield make_event("complete", sequence=1, data={})

    service, conversation_id, user_id, interrupt_id = _claimed_resume_service(
        source=source,
        lifecycle_events=lifecycle_events,
    )

    def fail_context(*_args):
        raise RuntimeError("setup failed")

    service._get_conversation_context = fail_context
    registry = _PausedRegistryRecorder()
    monkeypatch.setattr(
        "app.services.message_service.get_generation_registry",
        lambda: registry,
    )

    events = [
        event
        async for event in service.resume_message_creation_stream(
            thread_id=str(conversation_id),
            conversation_id=conversation_id,
            user_id=user_id,
            decisions=[],
            interrupt_id=interrupt_id,
        )
    ]

    assert lifecycle_events == [("failed", interrupt_id, "stream_exception")]
    assert events[-1].type == "error"
    assert events[-1].data["error_code"] == "INTERRUPT_FAILED"
    assert events[-1].data["status_code"] == 500
    assert registry.calls == [(user_id, conversation_id)]


@pytest.mark.asyncio
async def test_claimed_incomplete_resume_stream_clears_paused_registry(monkeypatch):
    lifecycle_events = []

    async def source(**_kwargs):
        if False:
            yield make_event("message_delta", sequence=1, data={"text": "unused"})

    service, conversation_id, user_id, interrupt_id = _claimed_resume_service(
        source=source,
        lifecycle_events=lifecycle_events,
    )
    registry = _PausedRegistryRecorder()
    monkeypatch.setattr(
        "app.services.message_service.get_generation_registry",
        lambda: registry,
    )

    events = [
        event
        async for event in service.resume_message_creation_stream(
            thread_id=str(conversation_id),
            conversation_id=conversation_id,
            user_id=user_id,
            decisions=[],
            interrupt_id=interrupt_id,
        )
    ]

    assert events[-1].type == "error"
    assert ("failed", interrupt_id, "stream_incomplete") in lifecycle_events
    assert registry.calls == [(user_id, conversation_id)]


@pytest.mark.asyncio
async def test_claimed_resume_completion_persistence_failure_marks_failed_before_resolved():
    conversation_id = uuid4()
    user_id = uuid4()
    interrupt_id = "claimed-interrupt"
    transitions = []
    state = {"value": "resolving"}

    class StatefulRepository:
        def mark_resolved(self, _interrupt_id):
            assert state["value"] == "resolving"
            state["value"] = "resolved"
            transitions.append("resolved")

        def mark_failed(self, _interrupt_id, *, resolution_source):
            if state["value"] != "resolving":
                return False
            state["value"] = "failed"
            transitions.append(("failed", resolution_source))
            return True

    async def source(**_kwargs):
        yield make_event(
            "complete",
            sequence=1,
            data={
                "response": WorkflowResponse(
                    message=WorkflowResponseMessage(content="completed"),
                    metadata={},
                )
            },
        )

    service = MessageService.__new__(MessageService)
    service.hitl_interrupt_repository = StatefulRepository()
    service._validate_and_claim_interrupt_resume = lambda **_kwargs: SimpleNamespace()
    service._get_conversation_context = lambda *_args: (user_id, None)
    service._revalidate_resume_custom_agent = lambda *_args: None
    service._audit_interrupt_resume_decisions = lambda **_kwargs: None
    service._resolve_custom_agents_state = lambda *_args: {}
    service._clear_redis_interrupt = lambda *_args: None
    service._sync_response_plan_state = lambda **_kwargs: False
    service._create_bot_response_message = lambda **_kwargs: SimpleNamespace(
        model_dump=lambda **_kwargs: {},
    )
    service.ai_service = SimpleNamespace(
        invalidate_history_cache=lambda *_args: None,
        resume_interrupted_execution_stream=source,
    )
    service._persist_completed_workflow_response = AsyncMock(
        side_effect=RuntimeError("completion persistence failed")
    )
    service._compact_checkpoint_after_persist = AsyncMock()

    events = [
        event
        async for event in service.resume_message_creation_stream(
            thread_id=str(conversation_id),
            conversation_id=conversation_id,
            user_id=user_id,
            decisions=[],
            interrupt_id=interrupt_id,
        )
    ]

    assert state["value"] == "failed"
    assert transitions == [("failed", "stream_exception")]
    assert events[-1].data["error_code"] == "INTERRUPT_FAILED"
    assert events[-1].data["status_code"] == 500


@pytest.mark.asyncio
async def test_claimed_nested_interrupt_durable_creation_failure_marks_failed_not_resolved():
    conversation_id = uuid4()
    user_id = uuid4()
    interrupt_id = "claimed-interrupt"
    bot_message_id = uuid4()
    paused_message_id = uuid4()
    error_message_id = uuid4()
    transitions = []
    state = {"value": "resolving"}
    created_messages = []
    deleted_message_ids = []

    class StatefulRepository:
        def create(self, **_kwargs):
            raise RuntimeError("follow-up interrupt persistence failed")

        def mark_resolved(self, _interrupt_id):
            assert state["value"] == "resolving"
            state["value"] = "resolved"
            transitions.append("resolved")

        def mark_failed(self, _interrupt_id, *, resolution_source):
            if state["value"] != "resolving":
                return False
            state["value"] = "failed"
            transitions.append(("failed", resolution_source))
            return True

    async def source(**_kwargs):
        yield make_event(
            "interrupt",
            sequence=1,
            data={
                "interrupt": {
                    "interrupt_id": "follow-up-interrupt",
                    "action_requests": [],
                    "thread_id": str(conversation_id),
                    "conversation_id": str(conversation_id),
                    "metadata": {},
                }
            },
        )

    service = MessageService.__new__(MessageService)
    service.hitl_interrupt_repository = StatefulRepository()
    service._validate_and_claim_interrupt_resume = lambda **_kwargs: SimpleNamespace()
    service._get_conversation_context = lambda *_args: (user_id, None)
    service._revalidate_resume_custom_agent = lambda *_args: None
    service._audit_interrupt_resume_decisions = lambda **_kwargs: None
    service._resolve_custom_agents_state = lambda *_args: {}
    service._clear_redis_interrupt = lambda *_args: None
    service._sync_response_plan_state = lambda **_kwargs: False
    service.repository = SimpleNamespace(
        delete=lambda message_id: deleted_message_ids.append(message_id) or True,
    )

    def create_bot_response_message(**kwargs):
        created_messages.append((kwargs["message_id"], kwargs["metadata"]))
        message_id = paused_message_id if len(created_messages) == 1 else error_message_id
        return SimpleNamespace(
            id=message_id,
            model_dump=lambda **_kwargs: {"id": str(message_id)},
        )

    service._create_bot_response_message = create_bot_response_message
    service.redis_client = None
    service.task_plan_service = None
    service.ai_service = SimpleNamespace(
        invalidate_history_cache=lambda *_args: None,
        resume_interrupted_execution_stream=source,
    )

    events = [
        event
        async for event in service.resume_message_creation_stream(
            thread_id=str(conversation_id),
            conversation_id=conversation_id,
            user_id=user_id,
            decisions=[],
            interrupt_id=interrupt_id,
            bot_message_id=bot_message_id,
        )
    ]

    assert state["value"] == "failed"
    assert transitions == [("failed", "stream_exception")]
    assert events[-1].type == "error"
    assert [message_id for message_id, _metadata in created_messages] == [bot_message_id, None]
    assert created_messages[0][1]["paused"] is True
    assert created_messages[1][1] == {"error": "follow-up interrupt persistence failed"}
    assert events[-1].message_id == str(error_message_id)
    assert events[-1].data["message"]["id"] == str(error_message_id)
    assert deleted_message_ids == [paused_message_id]
    assert events[-1].data["error_code"] == "INTERRUPT_FAILED"
    assert events[-1].data["status_code"] == 500
