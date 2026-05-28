"""Tests for the rich-item progressive stream events and AI SDK transport.

See response_format.md Task 5.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.api.ai_sdk import (
    StreamState,
    _build_ui_message_stream_response,
    project_ai_sdk_message_for_capability,
)
from app.core.config import settings
from app.core.rich_response import select_transient_upsert_items
from app.models.enums import MessageRole
from app.schemas.message import MessageCreate
from app.schemas.workflow import (
    WorkflowExecutionRequest,
    WorkflowPlanningContext,
    WorkflowResponse,
    WorkflowResponseMessage,
)
from app.services.ai_service import AIService
from app.services.message_service import MessageService
from app.services.stream_events import build_canonical_rich_items_event

SAFE_WIDGET_ITEM = {
    "id": "widget:w-1",
    "type": "live_widget",
    "display_policy": "inline_or_append",
    "payload": {
        "widget_id": "w-1",
        "session_id": "conv-1",
        "widget_type": "chart",
        "status": "active",
        "version": 1,
        "connection_endpoint": "/widgets/w-1/connection",
    },
}


def test_canonical_rich_items_event_contains_safe_upserts():
    event = build_canonical_rich_items_event(items=[SAFE_WIDGET_ITEM])
    assert event["type"] == "rich_items"
    assert event["operation"] == "upsert"
    assert event["items"][0]["id"] == "widget:w-1"


def test_ai_service_builds_transient_item_for_live_widget_tool_result():
    items = AIService._build_tool_end_rich_items(
        render={"version": 1, "type": "live_widget"},
        result=json.dumps(
            {
                "widget_id": "w-1",
                "session_id": "conv-1",
                "widget_type": "chart",
                "status": "active",
                "version": 1,
            }
        ),
        tool_call_id="call-widget",
        tool_name="widget_create",
    )

    [item] = items
    assert item["id"] == SAFE_WIDGET_ITEM["id"]
    assert item["type"] == "live_widget"
    assert item["payload"] == SAFE_WIDGET_ITEM["payload"]


def _widget_tool_end_event() -> dict[str, Any]:
    return {
        "type": "tool_end",
        "name": "widget_create",
        "tool_call_id": "call-widget",
        "result": json.dumps(
            {
                "widget_id": "w-1",
                "session_id": "conv-1",
                "widget_type": "chart",
                "status": "active",
                "version": 1,
            }
        ),
        "render": {"version": 1, "type": "live_widget"},
    }


@pytest.mark.asyncio
async def test_ai_service_suppresses_rich_items_without_capable_request(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)

    async def source(_request):
        yield _widget_tool_end_event()

    service = AIService.__new__(AIService)
    service.workflow = SimpleNamespace(execute_request_stream=source)
    events = [
        event
        async for event in service.execute_request_stream(
            WorkflowExecutionRequest(
                message="show a widget",
                inline_rich_response_v1=False,
            )
        )
    ]

    assert all(event.get("type") != "rich_items" for event in events)


@pytest.mark.asyncio
async def test_ai_service_suppresses_rich_items_while_rollout_disabled(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", False)

    async def source(_request):
        yield _widget_tool_end_event()

    service = AIService.__new__(AIService)
    service.workflow = SimpleNamespace(execute_request_stream=source)
    events = [
        event
        async for event in service.execute_request_stream(
            WorkflowExecutionRequest(
                message="show a widget",
                inline_rich_response_v1=True,
            )
        )
    ]

    assert all(event.get("type") != "rich_items" for event in events)


@pytest.mark.asyncio
async def test_ai_service_suppresses_resume_rich_items_without_capability(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)

    async def source(*_args, **_kwargs):
        yield _widget_tool_end_event()

    service = AIService.__new__(AIService)
    service.checkpointer = object()
    service.workflow = SimpleNamespace(resume_with_decisions_stream=source)
    events = [
        event
        async for event in service.resume_interrupted_execution_stream(
            "thread-1",
            [],
            inline_rich_response_v1=False,
        )
    ]

    assert all(event.get("type") != "rich_items" for event in events)


@pytest.mark.asyncio
async def test_ai_sdk_maps_rich_items_to_transient_data_event():
    async def source() -> Any:
        yield build_canonical_rich_items_event(items=[SAFE_WIDGET_ITEM])
        yield {
            "type": "complete",
            "message": {
                "content": "Result\n\n<!--rich:widget:w-1-->",
                "message_metadata": {"rich_items": [SAFE_WIDGET_ITEM]},
            },
        }

    state = StreamState(
        message_id="m-1",
        text_id="t-1",
        reasoning_id="r-1",
        inline_rich_response_v1=True,
    )
    response = _build_ui_message_stream_response(lambda: source(), state)
    chunks = [chunk async for chunk in response.body_iterator]
    payloads = []
    for line in "".join(chunks).splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            try:
                payloads.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                continue
    event = next(payload for payload in payloads if payload.get("type") == "data-rich-items")
    assert event["data"]["items"][0]["id"] == "widget:w-1"
    assert event["transient"] is True


@pytest.mark.asyncio
async def test_ai_sdk_complete_does_not_emit_unselected_image_file_parts():
    async def source() -> Any:
        yield {
            "type": "complete",
            "message": {
                "content": "No relevant image selected.",
                "message_metadata": {
                    "rich_items_version": 1,
                    "images": [{"url": "https://img.test/hidden.png", "mime": "image/png"}],
                    "rich_items": [],
                },
            },
        }

    state = StreamState(
        message_id="m-2",
        text_id="t-2",
        reasoning_id="r-2",
        inline_rich_response_v1=True,
    )
    response = _build_ui_message_stream_response(lambda: source(), state)
    content = "".join([chunk async for chunk in response.body_iterator])
    assert '"type":"file"' not in content
    assert "hidden.png" not in content


def test_transient_upserts_do_not_accept_unselected_images():
    hidden_image = {
        "id": "image:tool:c1:0",
        "type": "image",
        "display_policy": "inline_only",
        "alt_text": "Hidden image",
        "payload": {"url": "https://img.test/hidden.png", "mime_type": "image/png"},
    }
    assert select_transient_upsert_items([hidden_image]) == []


def test_non_capable_history_projection_removes_standalone_markers():
    message = {
        "content": "Intro\n\n<!--rich:image:tool:c1:0-->\n\nConclusion",
        "messageMetadata": {"rich_items_version": 1, "rich_items": []},
    }
    projected = project_ai_sdk_message_for_capability(message, inline_rich_response_v1=False)
    assert "<!--rich:" not in projected["content"]


def test_capable_projection_preserves_markers():
    message = {
        "content": "Intro\n\n<!--rich:image:tool:c1:0-->\n\nConclusion",
        "messageMetadata": {"rich_items_version": 1, "rich_items": []},
    }
    projected = project_ai_sdk_message_for_capability(message, inline_rich_response_v1=True)
    assert "<!--rich:image:tool:c1:0-->" in projected["content"]


@pytest.mark.asyncio
async def test_non_capable_stream_does_not_receive_rich_data_parts():
    async def source() -> Any:
        yield build_canonical_rich_items_event(items=[SAFE_WIDGET_ITEM])
        yield {"type": "complete", "message": {"content": "Answer", "message_metadata": {}}}

    state = StreamState(
        message_id="m-3",
        text_id="t-3",
        reasoning_id="r-3",
        inline_rich_response_v1=False,
    )
    response = _build_ui_message_stream_response(lambda: source(), state)
    content = "".join([chunk async for chunk in response.body_iterator])
    assert '"data-rich-items"' not in content


def _message_row(*, conversation_id, sender: int, content: str) -> SimpleNamespace:
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id=uuid4(),
        created_at=now,
        updated_at=now,
        deleted_at=None,
        conversation_id=conversation_id,
        sender=sender,
        content=content,
        message_metadata={},
        feedback=None,
    )


@pytest.mark.asyncio
async def test_message_service_forwards_rich_items_during_new_message_stream():
    conversation_id = uuid4()
    user_id = uuid4()
    service = MessageService.__new__(MessageService)
    service.repository = SimpleNamespace(
        create=lambda _entity: _message_row(
            conversation_id=conversation_id,
            sender=MessageRole.user.value,
            content="show a widget",
        )
    )
    service.conversation_validation_utils = SimpleNamespace(
        validate_conversation_access=lambda *_args: None,
        conversation_repository=SimpleNamespace(
            get_by_id=lambda _conversation_id: SimpleNamespace(title="Existing chat")
        ),
    )

    async def source(_request):
        yield build_canonical_rich_items_event(items=[SAFE_WIDGET_ITEM])
        yield {
            "type": "complete",
            "response": WorkflowResponse(
                message=WorkflowResponseMessage(content="done"),
                metadata={},
            ),
        }

    service.ai_service = SimpleNamespace(
        invalidate_history_cache=lambda *_args: None,
        execute_request_stream=source,
    )
    workflow_request = WorkflowExecutionRequest(
        message="show a widget",
        conversation_id=str(conversation_id),
        user_id=str(user_id),
        planning=WorkflowPlanningContext(),
    )
    service._build_user_message_workflow_request = AsyncMock(
        return_value=(user_id, None, workflow_request)
    )
    service._persist_completed_workflow_response = AsyncMock(
        return_value=_message_row(
            conversation_id=conversation_id,
            sender=MessageRole.assistant.value,
            content="done",
        )
    )
    service._compact_checkpoint_after_persist = AsyncMock()

    events = [
        event
        async for event in service.create_message_stream(
            MessageCreate(conversation_id=conversation_id, content="show a widget"),
            user_id,
        )
    ]

    assert any(event.get("type") == "rich_items" for event in events)


@pytest.mark.asyncio
async def test_message_service_forwards_rich_items_during_resume_stream():
    conversation_id = uuid4()
    user_id = uuid4()
    service = MessageService.__new__(MessageService)
    service.hitl_interrupt_repository = None
    service._validate_and_claim_interrupt_resume = lambda **_kwargs: None
    service._get_conversation_context = lambda *_args: (user_id, None)
    service._audit_interrupt_resume_decisions = lambda **_kwargs: None
    service._clear_redis_interrupt = lambda *_args: None
    service._create_bot_response_message = lambda **_kwargs: _message_row(
        conversation_id=conversation_id,
        sender=MessageRole.assistant.value,
        content="error",
    )

    async def source(*_args, **_kwargs):
        yield build_canonical_rich_items_event(items=[SAFE_WIDGET_ITEM])
        yield {"type": "error", "error": "terminal"}

    service.ai_service = SimpleNamespace(resume_interrupted_execution_stream=source)

    events = [
        event
        async for event in service.resume_message_creation_stream(
            thread_id="thread-1",
            conversation_id=conversation_id,
            user_id=user_id,
            decisions=[],
        )
    ]

    assert any(event.get("type") == "rich_items" for event in events)


def test_live_widget_rich_item_does_not_embed_state():
    """Widget rich items must stay compact — state arrives over the WebSocket."""
    from app.core.response_constants import _widget_rich_item_from_live_widget

    item = _widget_rich_item_from_live_widget(
        {
            "widget_id": "w-1",
            "session_id": "conv-1",
            "widget_type": "chart",
            "title": "Meaningful Chart",
            "status": "active",
            "version": 1,
            "state": {
                "labels": ["A", "B"],
                "datasets": [{"data": [1, 2]}],
                "presentation": {"caption": "should not leak"},
                "actions": [{"key": "k", "type": "assistant_message", "message_template": "x"}],
            },
        }
    )

    assert "state" not in item["payload"]
    assert "presentation" not in item["payload"]
    assert "actions" not in item["payload"]
    assert item["type"] == "live_widget"
    assert item["display_policy"] == "inline_or_append"
    assert item["payload"]["widget_id"] == "w-1"
