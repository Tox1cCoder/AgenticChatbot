"""Task 10: direct-SDK provider calls (vision + captioning) are recorded.

These paths call the Gemini SDK's ``generate_content`` synchronously rather
than through LangChain. Each attempt is recorded as its own operation
(``vision`` / ``image_caption``) with provider-reported usage, inheriting the
bound request/document ownership context.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from prometheus_client import CollectorRegistry

from app.observability.model_usage import ModelUsageMetrics
from app.repositories.model_usage import RecordEventCommand, RecordResult
from app.usage.context import bind_usage_context
from app.usage.recorder import ModelUsageRecorder
from app.usage.types import UsageContext


class _FakeRepo:
    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self.commands: list[RecordEventCommand] = []
        self._fail_with = fail_with

    def record_event(self, command: RecordEventCommand) -> RecordResult:
        if self._fail_with is not None:
            raise self._fail_with
        self.commands.append(command)
        return RecordResult(inserted=True, event_id=uuid4())


def _recorder(repo: _FakeRepo) -> ModelUsageRecorder:
    return ModelUsageRecorder(
        repository=repo,
        enqueue_failed_write=lambda payload: None,
        metrics=ModelUsageMetrics(registry=CollectorRegistry()),
    )


def _gemini_response(text: str = "a serene fox") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=12,
            total_token_count=112,
        ),
    )


def _gemini_client(response) -> SimpleNamespace:
    return SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_kwargs: response))


# ---------------------------------------------------------------------------
# Vision (chat_agent Gemini generate_content)
# ---------------------------------------------------------------------------


def _chat_agent(recorder):
    from app.ai.agents.chat_agent import ChatAgent

    agent = ChatAgent.__new__(ChatAgent)
    agent.recorder = recorder
    # agent_id is a read-only property on ChatAgent ("chat_agent").
    return agent


@pytest.mark.asyncio
async def test_vision_gemini_records_provider_usage():
    repo = _FakeRepo()
    agent = _chat_agent(_recorder(repo))
    client = _gemini_client(_gemini_response())
    user_id = uuid4()

    with (
        bind_usage_context(UsageContext(user_id=user_id, operation="workflow")),
        agent._begin_vision_usage() as operation,
    ):
        response = await agent._invoke_vision_gemini(
            client, model="gemini-3-pro", parts=[], config=None, operation=operation
        )

    assert response.text == "a serene fox"
    assert len(repo.commands) == 1
    command = repo.commands[0]
    assert command.status == "success"
    assert command.provider == "gemini"
    assert command.model == "gemini-3-pro"
    assert command.context.operation == "vision"
    assert command.context.agent_id == "chat_agent"
    assert command.context.user_id == user_id
    assert command.usage.input_tokens == 100
    assert command.usage.output_tokens == 12


@pytest.mark.asyncio
async def test_vision_gemini_without_recorder_records_nothing():
    agent = _chat_agent(None)
    client = _gemini_client(_gemini_response())

    with agent._begin_vision_usage() as operation:
        assert operation is None
        response = await agent._invoke_vision_gemini(
            client, model="gemini-3-pro", parts=[], config=None, operation=operation
        )

    assert response.text == "a serene fox"


# ---------------------------------------------------------------------------
# Captioning (document_processing_service Gemini generate_content)
# ---------------------------------------------------------------------------


def _caption_service(recorder, client):
    from app.services.document_processing_service import DocumentProcessingService

    service = DocumentProcessingService.__new__(DocumentProcessingService)
    service.recorder = recorder
    service.gemini_client = client
    service.settings = SimpleNamespace(
        image_caption_model="gemini-2.5-flash",
        image_caption_max_retry_attempts=3,
        image_caption_retry_delay_seconds=0.001,
    )
    return service


@pytest.mark.asyncio
async def test_caption_records_image_caption_operation():
    repo = _FakeRepo()
    service = _caption_service(_recorder(repo), _gemini_client(_gemini_response("a cat")))
    user_id, document_id = uuid4(), uuid4()

    with bind_usage_context(
        UsageContext(user_id=user_id, document_id=document_id, operation="document_index")
    ):
        caption = await service._generate_image_caption_with_retry(b"\x89PNG", "img.png")

    assert caption == "a cat"
    assert len(repo.commands) == 1
    command = repo.commands[0]
    assert command.status == "success"
    assert command.provider == "gemini"
    assert command.model == "gemini-2.5-flash"
    assert command.context.operation == "image_caption"
    assert command.context.user_id == user_id
    assert command.context.document_id == document_id
    assert command.usage.input_tokens == 100


@pytest.mark.asyncio
async def test_caption_records_one_event_per_retry_attempt(monkeypatch):
    from google.genai import errors as genai_errors

    monkeypatch.setattr(
        "app.services.document_processing_service.asyncio.sleep",
        _async_noop,
    )
    rate_limit = genai_errors.ClientError(
        429,
        {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [{"retryDelay": "1s"}]}},
    )
    calls = iter([rate_limit, _gemini_response("recovered")])

    def _generate_content(**_kwargs):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    client = SimpleNamespace(models=SimpleNamespace(generate_content=_generate_content))
    repo = _FakeRepo()
    service = _caption_service(_recorder(repo), client)

    with bind_usage_context(UsageContext(user_id=uuid4(), operation="document_index")):
        caption = await service._generate_image_caption_with_retry(b"\x89PNG", "img.png")

    assert caption == "recovered"
    assert [c.attempt for c in repo.commands] == [1, 2]
    assert repo.commands[0].status == "error"
    assert repo.commands[1].status == "success"


@pytest.mark.asyncio
async def test_caption_without_recorder_returns_caption():
    service = _caption_service(None, _gemini_client(_gemini_response("a dog")))

    caption = await service._generate_image_caption_with_retry(b"\x89PNG", "img.png")

    assert caption == "a dog"


async def _async_noop(*_args, **_kwargs) -> None:
    return None
