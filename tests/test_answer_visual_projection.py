"""Per-feature capability projection of answer visuals.

Stored metadata is canonical. A client that did not advertise a feature gets
that feature's derived Markdown at the marker position instead of a marker it
cannot render; everything it does support is left alone.
"""

from __future__ import annotations

import copy
import json
from uuid import UUID, uuid4

import pytest

from app.core.answer_visuals import ResolvedEvidence, build_answer_visual
from app.services.event_streaming.ai_sdk_projection import (
    project_ai_sdk_message_for_capability,
)
from app.services.event_streaming.events import make_event

_VISUAL_ID = "visual:steps1"


def _visual() -> dict:
    item = build_answer_visual(
        {
            "variant": "steps",
            "title": "Reset the router",
            "steps": [
                {"title": "Unplug it"},
                {"title": "Wait | 30 seconds", "evidence": [{"kind": "web", "id": "S1"}]},
            ],
        },
        resolve_evidence=lambda ref: ResolvedEvidence(
            title="ISP help", url="https://isp.example/reset"
        ),
        turn_key="turn-1",
    )
    item.pop("_internal")
    item["id"] = _VISUAL_ID
    return item


_TOOL_RENDER = {
    "id": "tool:t1",
    "type": "tool_render",
    "display_policy": "inline_or_append",
    "payload": {"render": {"type": "table"}},
    "provenance": {},
}


def _message(content: str, items: list[dict] | None = None) -> dict:
    return {
        "id": "m-1",
        "role": "assistant",
        "content": content,
        "metadata": {
            "rich_items_version": 1,
            "rich_items": items if items is not None else [_visual(), _TOOL_RENDER],
            "rich_reference_warnings": [],
        },
    }


def _fallback() -> str:
    return _visual()["payload"]["fallback_markdown"]


_CONTENT = "Try this.\n\n<!--rich:visual:steps1-->\n\nThen check.\n\n<!--rich:tool:t1-->"


def _project(message: dict, *, rich: bool, visuals: bool) -> dict:
    return project_ai_sdk_message_for_capability(
        message, inline_rich_response_v1=rich, answer_visuals_v1=visuals
    )


def test_fully_capable_client_receives_the_canonical_message():
    message = _message(_CONTENT)

    assert _project(message, rich=True, visuals=True) == message


def test_base_rich_client_gets_fallback_for_visuals_and_keeps_other_items():
    projected = _project(_message(_CONTENT), rich=True, visuals=False)

    assert projected["content"] == (
        f"Try this.\n\n{_fallback()}\n\nThen check.\n\n<!--rich:tool:t1-->"
    )
    assert [item["id"] for item in projected["metadata"]["rich_items"]] == ["tool:t1"]


def test_non_rich_client_gets_fallback_and_no_markers():
    projected = _project(_message(_CONTENT), rich=False, visuals=False)

    assert projected["content"] == f"Try this.\n\n{_fallback()}\n\nThen check.\n\n"
    assert "rich_items" not in projected["metadata"]


def test_visual_capability_without_base_capability_is_not_honoured():
    projected = _project(_message(_CONTENT), rich=False, visuals=True)

    assert _fallback() in projected["content"]
    assert "<!--rich:" not in projected["content"]


def test_fallback_keeps_values_and_citations():
    projected = _project(_message(_CONTENT), rich=False, visuals=False)

    assert "Wait \\| 30 seconds" not in projected["content"]  # steps are not table cells
    assert "Wait | 30 seconds [isp.example](https://isp.example/reset)" in projected["content"]


def test_projection_never_mutates_the_canonical_message():
    message = _message(_CONTENT)
    before = copy.deepcopy(message)

    _project(message, rich=True, visuals=False)
    _project(message, rich=False, visuals=False)

    assert message == before


def test_marker_inside_code_fence_is_not_expanded():
    content = "Example:\n\n```\n<!--rich:visual:steps1-->\n```\n"
    projected = _project(_message(content), rich=True, visuals=False)

    assert projected["content"] == content


def test_repeated_marker_expands_once():
    content = "<!--rich:visual:steps1-->\n\nAgain:\n\n<!--rich:visual:steps1-->"
    projected = _project(_message(content), rich=True, visuals=False)

    assert projected["content"].count("**Reset the router**") == 1
    assert "<!--rich:visual:steps1-->" not in projected["content"]


def test_marker_inside_a_paragraph_becomes_its_own_block():
    content = "Before <!--rich:visual:steps1--> after."
    projected = _project(_message(content), rich=True, visuals=False)

    assert projected["content"] == f"Before\n\n{_fallback()}\n\nafter."


def test_visual_marker_without_a_descriptor_is_removed_for_an_unsupported_client():
    content = "Answer.\n\n<!--rich:visual:gone-->"
    projected = _project(_message(content, items=[_TOOL_RENDER]), rich=True, visuals=False)

    assert projected["content"] == "Answer."


def test_fallback_is_never_re_expanded():
    item = _visual()
    item["payload"]["fallback_markdown"] = "**T**\n\n1. see &lt;!--rich:tool:t1-->"
    projected = _project(_message(_CONTENT, items=[item, _TOOL_RENDER]), rich=True, visuals=False)

    assert projected["content"].count("<!--rich:tool:t1-->") == 1


# ---------------------------------------------------------------------------
# Streaming adapter
# ---------------------------------------------------------------------------


def _visual_stream(deltas: list[str], final_content: str):
    async def source():
        for index, delta in enumerate(deltas, start=1):
            yield make_event("message_delta", sequence=index, data={"text": delta})
        final = _message(final_content)
        yield make_event("complete", sequence=len(deltas) + 1, data={"message": final})

    return source()


async def _frames(*, rich: bool, visuals: bool, deltas: list[str], final_content: str):
    from app.services.event_streaming.ai_sdk_v6 import AISDKV6StreamAdapter, AISDKV6StreamState

    state = AISDKV6StreamState(
        "m-1", "t-1", "r-1", inline_rich_response_v1=rich, answer_visuals_v1=visuals
    )
    adapter = AISDKV6StreamAdapter(
        lambda: _visual_stream(deltas, final_content), state, heartbeat_interval_seconds=60
    )
    body = "".join([frame async for frame in adapter.iter_sse()])
    return [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]


@pytest.mark.asyncio
async def test_unsupported_stream_never_shows_a_split_visual_marker_and_reconciles():
    deltas = ["Try this.\n\n<!--ri", "ch:visual:st", "eps1-->\n\nThen check."]
    frames = await _frames(
        rich=True, visuals=False, deltas=deltas, final_content="Try this.\n\n"
        "<!--rich:visual:steps1-->\n\nThen check."
    )

    streamed = "".join(f["delta"] for f in frames if f["type"] == "text-delta")
    assert "<!--rich:visual" not in streamed
    [reconcile] = [f for f in frames if f["type"] == "data-answer-reconcile"]
    assert _fallback() in reconcile["data"]["content"]


@pytest.mark.asyncio
async def test_capable_stream_passes_the_visual_marker_through():
    frames = await _frames(
        rich=True,
        visuals=True,
        deltas=["Try this.\n\n<!--rich:visual:steps1-->"],
        final_content="Try this.\n\n<!--rich:visual:steps1-->",
    )

    streamed = "".join(f["delta"] for f in frames if f["type"] == "text-delta")
    assert "<!--rich:visual:steps1-->" in streamed
    assert not any(f["type"] == "data-answer-reconcile" for f in frames)
    [message] = [f for f in frames if f["type"] == "data-assistant-message"]
    ids = [item["id"] for item in message["data"]["message"]["metadata"]["rich_items"]]
    assert _VISUAL_ID in ids


@pytest.mark.asyncio
async def test_unsupported_stream_without_visual_markers_does_not_reconcile():
    frames = await _frames(
        rich=True, visuals=False, deltas=["Plain answer."], final_content="Plain answer."
    )

    assert not any(f["type"] == "data-answer-reconcile" for f in frames)


def test_transient_upserts_never_publish_an_answer_visual():
    from app.core.rich_response import select_transient_upsert_items

    assert select_transient_upsert_items([_visual()]) == []


# ---------------------------------------------------------------------------
# Request propagation
# ---------------------------------------------------------------------------


_CONVERSATION = "11111111-1111-4111-8111-111111111111"


class _FakeMessageService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def _source(self):
        content = "Try this.\n\n<!--rich:visual:steps1-->"
        return _visual_stream([content], content)

    def create_message_stream(self, message_create, current_user_id, **kwargs):
        self.calls.append(("chat", {"message_create": message_create, **kwargs}))
        return self._source()

    def resume_message_creation_stream(self, **kwargs):
        self.calls.append(("resume", kwargs))
        return self._source()

    def continue_message_generation_stream(self, **kwargs):
        self.calls.append(("continue", kwargs))
        return self._source()


async def _call(operation: str, extra: dict, service: _FakeMessageService):
    from app.api.ai_sdk import (
        AISDKChatRequest,
        chat_ui_message_stream,
        continue_generation_ai_sdk,
        resume_interrupt_ai_sdk,
    )
    from app.schemas.message import ContinueGenerationRequest, InterruptResumeRequest

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
        body = {
            "threadId": "thread-1",
            "interruptId": "int-1",
            "conversationId": _CONVERSATION,
            "decisions": [{"type": "approve"}],
            **extra,
        }
        return await resume_interrupt_ai_sdk(
            resume_request=InterruptResumeRequest.model_validate(body),
            message_service=service,
            current_user_id=user_id,
        )
    body = {
        "conversationId": _CONVERSATION,
        "generationId": "22222222-2222-4222-8222-222222222222",
        "continuationId": "33333333-3333-4333-8333-333333333333",
        "idempotencyKey": "continue-test-1",
        "expectedVersion": 1,
        **extra,
    }
    return await continue_generation_ai_sdk(
        continue_request=ContinueGenerationRequest.model_validate(body),
        message_service=service,
        current_user_id=user_id,
    )


@pytest.fixture
def _visuals_rollout(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "answer_visuals_enabled", True)


async def _payloads(response) -> list[dict]:
    body = "".join([chunk async for chunk in response.body_iterator])
    return [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["chat", "resume", "continue"])
@pytest.mark.parametrize("key", ["answerVisualsV1", "answer_visuals_v1"])
@pytest.mark.usefixtures("_visuals_rollout")
async def test_every_operation_delivers_visuals_to_a_capable_client(operation, key):
    service = _FakeMessageService()
    response = await _call(operation, {"inlineRichResponseV1": True, key: True}, service)
    payloads = await _payloads(response)

    streamed = "".join(p["delta"] for p in payloads if p["type"] == "text-delta")
    assert "<!--rich:visual:steps1-->" in streamed
    assert not any(p["type"] == "data-answer-reconcile" for p in payloads)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["chat", "resume", "continue"])
@pytest.mark.usefixtures("_visuals_rollout")
async def test_every_operation_defaults_to_the_visual_fallback(operation):
    service = _FakeMessageService()
    response = await _call(operation, {"inlineRichResponseV1": True}, service)
    payloads = await _payloads(response)

    streamed = "".join(p["delta"] for p in payloads if p["type"] == "text-delta")
    assert "<!--rich:visual" not in streamed
    [reconcile] = [p for p in payloads if p["type"] == "data-answer-reconcile"]
    assert _fallback() in reconcile["data"]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("requested", [True, False])
@pytest.mark.usefixtures("_visuals_rollout")
async def test_chat_carries_the_effective_visual_capability_into_generation(requested):
    service = _FakeMessageService()
    response = await _call(
        "chat", {"inlineRichResponseV1": True, "answerVisualsV1": requested}, service
    )
    await _payloads(response)

    ((_, kwargs),) = service.calls
    assert kwargs["message_create"].answer_visuals_v1 is requested


@pytest.mark.asyncio
async def test_chat_does_not_request_visuals_while_the_rollout_gate_is_off(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "answer_visuals_enabled", False)
    service = _FakeMessageService()
    response = await _call(
        "chat", {"inlineRichResponseV1": True, "answerVisualsV1": True}, service
    )
    await _payloads(response)

    ((_, kwargs),) = service.calls
    assert kwargs["message_create"].answer_visuals_v1 is False


def _history(*, rich: bool, visuals: bool) -> dict:
    from datetime import UTC, datetime
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from app.api.ai_sdk import get_conversation_messages_ai_sdk
    from app.repositories.utils.pagination import PaginationMeta
    from app.schemas.pagination import MessagePaginationParams

    stored = _message(_CONTENT)
    row = SimpleNamespace(
        id=uuid4(),
        sender=2,
        content=stored["content"],
        created_at=datetime.now(UTC),
        message_metadata=stored["metadata"],
    )
    service = MagicMock()
    service.get_conversation_messages.return_value = SimpleNamespace(
        items=[row], meta=PaginationMeta.calculate(1, 100, 1)
    )
    response = get_conversation_messages_ai_sdk(
        uuid4(),
        service,
        uuid4(),
        MessagePaginationParams(),
        inline_rich_response_v1=rich,
        answer_visuals_v1=visuals,
    )
    return response.data.messages[0].model_dump(by_alias=True)


@pytest.mark.usefixtures("_visuals_rollout")
def test_legacy_history_of_a_capable_turn_reads_as_markdown():
    message = _history(rich=False, visuals=False)

    assert _fallback() in message["content"]
    assert "<!--rich:" not in message["content"]
    assert "rich_items" not in (message["metadata"] or {})


@pytest.mark.usefixtures("_visuals_rollout")
def test_capable_history_keeps_the_visual():
    message = _history(rich=True, visuals=True)

    assert "<!--rich:visual:steps1-->" in message["content"]
    assert _VISUAL_ID in [item["id"] for item in message["metadata"]["rich_items"]]


@pytest.mark.usefixtures("_visuals_rollout")
def test_base_rich_history_drops_the_descriptor_and_keeps_other_items():
    message = _history(rich=True, visuals=False)

    assert _fallback() in message["content"]
    assert [item["id"] for item in message["metadata"]["rich_items"]] == ["tool:t1"]


def test_effective_capability_requires_base_and_rollout(monkeypatch):
    from app.core.config import settings
    from app.services.event_streaming.ai_sdk_projection import effective_answer_visuals

    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "answer_visuals_enabled", True)
    assert effective_answer_visuals(base=True, requested=True) is True
    assert effective_answer_visuals(base=False, requested=True) is False
    monkeypatch.setattr(settings, "answer_visuals_enabled", False)
    assert effective_answer_visuals(base=True, requested=True) is False
