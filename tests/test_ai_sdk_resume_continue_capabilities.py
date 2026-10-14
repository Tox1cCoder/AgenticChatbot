"""The web-draft capability reaches the adapter on Chat, Resume, and Continue.

A capable client that approved a tool call or continued a paused turn used to
get the legacy preview behavior on the resumed stream, because Resume and
Continue built their stream state without the flag. The flag shapes this
response's presentation only; it is never durable model state.
"""

from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest

from app.api.ai_sdk import (
    AISDKChatRequest,
    chat_ui_message_stream,
    continue_generation_ai_sdk,
    resume_interrupt_ai_sdk,
)
from app.schemas.message import ContinueGenerationRequest, InterruptResumeRequest
from app.services.event_streaming.events import make_event

_CONVERSATION = "11111111-1111-4111-8111-111111111111"
_GENERATION = "22222222-2222-4222-8222-222222222222"
_CONTINUATION = "33333333-3333-4333-8333-333333333333"


def _web_answer_source():
    async def source():
        preview = {"operation": "replace", "text": "Draft"}
        yield make_event("answer_preview", sequence=1, data=preview)
        yield make_event("message_delta", sequence=2, data={"text": "Final"})
        final = {"id": "m-1", "content": "Final"}
        yield make_event("complete", sequence=3, data={"message": final})

    return source()


class _FakeMessageService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def create_message_stream(self, message_create, current_user_id, **kwargs):
        self.calls.append(("chat", {"message_create": message_create, **kwargs}))
        return _web_answer_source()

    def resume_message_creation_stream(self, **kwargs):
        self.calls.append(("resume", kwargs))
        return _web_answer_source()

    def continue_message_generation_stream(self, **kwargs):
        self.calls.append(("continue", kwargs))
        return _web_answer_source()


def _resume_body(**extra) -> dict:
    return {
        "threadId": "thread-1",
        "interruptId": "int-1",
        "conversationId": _CONVERSATION,
        "decisions": [{"type": "approve"}],
        **extra,
    }


def _continue_body(**extra) -> dict:
    return {
        "conversationId": _CONVERSATION,
        "generationId": _GENERATION,
        "continuationId": _CONTINUATION,
        "idempotencyKey": "continue-test-1",
        "expectedVersion": 1,
        **extra,
    }


async def _call(operation: str, extra: dict, service: _FakeMessageService):
    user_id = uuid4()
    if operation == "chat":
        payload = AISDKChatRequest.model_validate(
            {"messages": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}], **extra}
        )
        return await chat_ui_message_stream(
            conversation_id=UUID(_CONVERSATION),
            payload=payload,
            message_service=service,
            current_user_id=user_id,
        )
    if operation == "resume":
        return await resume_interrupt_ai_sdk(
            resume_request=InterruptResumeRequest.model_validate(_resume_body(**extra)),
            message_service=service,
            current_user_id=user_id,
        )
    return await continue_generation_ai_sdk(
        continue_request=ContinueGenerationRequest.model_validate(_continue_body(**extra)),
        message_service=service,
        current_user_id=user_id,
    )


async def _types_and_deltas(response) -> tuple[list[str], list[str]]:
    body = "".join([chunk async for chunk in response.body_iterator])
    payloads = [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]
    return (
        [p["type"] for p in payloads],
        [p["delta"] for p in payloads if p["type"] == "text-delta"],
    )


@pytest.mark.parametrize("key", ["streamWebAnswerV1", "stream_web_answer_v1"])
def test_continue_schema_accepts_web_stream_capability(key):
    value = ContinueGenerationRequest.model_validate(
        {
            "conversationId": "11111111-1111-4111-8111-111111111111",
            "generationId": "22222222-2222-4222-8222-222222222222",
            "continuationId": "33333333-3333-4333-8333-333333333333",
            "idempotencyKey": "continue-test-1",
            "expectedVersion": 1,
            key: True,
        }
    )
    assert value.stream_web_answer_v1 is True


@pytest.mark.parametrize("key", ["streamWebAnswerV1", "stream_web_answer_v1"])
def test_resume_schema_accepts_web_stream_capability(key):
    value = InterruptResumeRequest.model_validate(_resume_body(**{key: True}))
    assert value.stream_web_answer_v1 is True


@pytest.mark.parametrize("model", [InterruptResumeRequest, ContinueGenerationRequest])
def test_web_stream_capability_defaults_to_off(model):
    body = _resume_body() if model is InterruptResumeRequest else _continue_body()
    assert model.model_validate(body).stream_web_answer_v1 is False


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["chat", "resume", "continue"])
@pytest.mark.parametrize("key", ["streamWebAnswerV1", "stream_web_answer_v1"])
async def test_capable_client_gets_draft_text_and_reconciliation(operation, key):
    response = await _call(operation, {key: True}, _FakeMessageService())
    types, deltas = await _types_and_deltas(response)

    assert deltas == ["Draft"]
    assert types.count("data-answer-reconcile") == 1
    assert "data-answer-preview" not in types


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["chat", "resume", "continue"])
@pytest.mark.parametrize("extra", [{}, {"streamWebAnswerV1": False}], ids=["omitted", "false"])
async def test_client_without_capability_keeps_the_validated_fallback(operation, extra):
    response = await _call(operation, extra, _FakeMessageService())
    types, deltas = await _types_and_deltas(response)

    assert deltas == ["Final"]
    assert "data-answer-reconcile" not in types
    assert "data-answer-preview" in types


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["chat", "resume", "continue"])
@pytest.mark.parametrize("rich", [True, False])
@pytest.mark.parametrize("web", [True, False])
async def test_web_and_rich_capabilities_are_independent(operation, rich, web):
    service = _FakeMessageService()
    response = await _call(
        operation,
        {"inlineRichResponseV1": rich, "streamWebAnswerV1": web},
        service,
    )
    types, _ = await _types_and_deltas(response)

    ((_, kwargs),) = service.calls
    if operation == "chat":
        passed_rich = kwargs["message_create"].inline_rich_response_v1
    else:
        passed_rich = kwargs["inline_rich_response_v1"]
    assert passed_rich is rich
    assert ("data-answer-reconcile" in types) is web
    assert not any(key.startswith("stream_web") for key in kwargs)
