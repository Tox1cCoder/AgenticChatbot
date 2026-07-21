"""Task 9: image-generation streaming usage is recorded exactly once.

``ImageGeneratorAgent._generate_images`` opens one recorder streaming-attempt
handle immediately before each provider stream, consumes every event
(including the terminal ``ImageUsage`` that arrives only after the requested
image count is satisfied), and finalizes that handle exactly once with the
terminal status -- success, error, or cancelled -- even when no usage event
arrives. The image operation is its own ledger event (operation
``image_generation``) attributed to the bound user/conversation context.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import patch
from uuid import uuid4

import pytest
from prometheus_client import CollectorRegistry

from app.ai.image_generation import ImageFinal, ImageUsage, NarrativeDelta
from app.observability.model_usage import ModelUsageMetrics
from app.repositories.model_usage import RecordEventCommand, RecordResult
from app.usage.context import bind_usage_context
from app.usage.recorder import ModelUsageRecorder
from app.usage.types import NormalizedUsage, UsageContext


class FakeRepository:
    """In-memory double for ``ModelUsageRepository`` (owns no session)."""

    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self.commands: list[RecordEventCommand] = []
        self.record_threads: list[int] = []
        self._fail_with = fail_with

    def record_event(self, command: RecordEventCommand) -> RecordResult:
        self.record_threads.append(threading.get_ident())
        if self._fail_with is not None:
            raise self._fail_with
        self.commands.append(command)
        return RecordResult(inserted=True, event_id=uuid4())


class FakeProvider:
    """Yields a scripted event list, optionally raising after the last event."""

    def __init__(self, events: list, *, raises: BaseException | None = None) -> None:
        self._events = events
        self._raises = raises

    async def stream_generate(self, request):  # noqa: ANN001 - test double
        for event in self._events:
            yield event
        if self._raises is not None:
            raise self._raises


def _recorder(repo: FakeRepository) -> ModelUsageRecorder:
    return ModelUsageRecorder(
        repository=repo,
        enqueue_failed_write=lambda payload: None,
        metrics=ModelUsageMetrics(registry=CollectorRegistry()),
    )


def _agent(recorder, *, model_name: str = "gemini-3-pro-image", max_images: int = 2):
    from app.ai.agents.image_generator_agent import ImageGeneratorAgent

    agent = ImageGeneratorAgent.__new__(ImageGeneratorAgent)
    agent.model_name = model_name
    agent.default_aspect_ratio = "1:1"
    agent.max_images = max_images
    agent.gemini_client = object()  # only used by the (patched) resolver
    agent.recorder = recorder
    return agent


def _patch_provider(provider: FakeProvider):
    return patch(
        "app.ai.agents.image_generator_agent.resolve_image_provider",
        return_value=provider,
    )


def _gemini_usage() -> NormalizedUsage:
    return NormalizedUsage(
        input_tokens=120,
        output_tokens=2048,
        total_tokens=2168,
        output_image_tokens=2048,
        source="provider_reported",
    )


@pytest.mark.asyncio
async def test_gemini_stream_usage_recorded_as_image_operation():
    repo = FakeRepository()
    recorder = _recorder(repo)
    agent = _agent(recorder)
    user_id, conversation_id = uuid4(), uuid4()
    provider = FakeProvider(
        [
            ImageFinal(index=0, data_b64="AAA", mime="image/png"),
            NarrativeDelta(text="Here is your fox."),
            ImageUsage(usage=_gemini_usage(), provider_request_id="resp-42"),
        ]
    )

    with (
        bind_usage_context(
            UsageContext(user_id=user_id, conversation_id=conversation_id, operation="workflow")
        ),
        _patch_provider(provider),
    ):
        images, narrative = await agent._generate_images("enhanced", "draw a fox")

    assert [img["data"] for img in images] == ["AAA"]
    assert narrative == "Here is your fox."
    assert len(repo.commands) == 1, "exactly one ledger event per image stream"
    command = repo.commands[0]
    assert command.status == "success"
    assert command.provider == "gemini"
    assert command.model == "gemini-3-pro-image"
    assert command.context.operation == "image_generation"
    assert command.context.agent_id == "image_generator_agent"
    assert command.context.user_id == user_id
    assert command.context.conversation_id == conversation_id
    assert command.attempt == 1
    assert command.usage.source == "provider_reported"
    assert command.usage.input_tokens == 120
    assert command.usage.output_tokens == 2048
    assert command.usage.generated_images == 1
    assert command.provider_request_id == "resp-42"


@pytest.mark.asyncio
async def test_generated_images_count_reflects_delivered_finals():
    repo = FakeRepository()
    agent = _agent(_recorder(repo), max_images=2)
    provider = FakeProvider(
        [
            ImageFinal(index=0, data_b64="AAA", mime="image/png"),
            ImageFinal(index=1, data_b64="BBB", mime="image/png"),
            ImageUsage(usage=_gemini_usage()),
        ]
    )

    with bind_usage_context(UsageContext(operation="workflow")), _patch_provider(provider):
        images, _ = await agent._generate_images("enhanced", "orig")

    assert len(images) == 2
    assert repo.commands[0].usage.generated_images == 2


@pytest.mark.asyncio
async def test_stream_records_terminal_status_when_no_usage_event_arrives():
    repo = FakeRepository()
    agent = _agent(_recorder(repo))
    provider = FakeProvider([ImageFinal(index=0, data_b64="AAA", mime="image/png")])

    with bind_usage_context(UsageContext(operation="workflow")), _patch_provider(provider):
        images, _ = await agent._generate_images("enhanced", "orig")

    assert len(images) == 1
    assert len(repo.commands) == 1
    command = repo.commands[0]
    assert command.status == "success"
    assert command.usage.source == "unavailable"
    assert command.usage.generated_images == 1


@pytest.mark.asyncio
async def test_stream_error_is_finalized_once_and_reraises():
    repo = FakeRepository()
    agent = _agent(_recorder(repo))
    provider = FakeProvider(
        [ImageFinal(index=0, data_b64="AAA", mime="image/png")],
        raises=ValueError("provider blew up"),
    )

    with (
        bind_usage_context(UsageContext(operation="workflow")),
        _patch_provider(provider),
        pytest.raises(ValueError, match="provider blew up"),
    ):
        await agent._generate_images("enhanced", "orig")

    assert len(repo.commands) == 1
    command = repo.commands[0]
    assert command.status == "error"
    assert command.error_code == "ValueError"


@pytest.mark.asyncio
async def test_stream_cancellation_is_finalized_once_and_reraises():
    repo = FakeRepository()
    agent = _agent(_recorder(repo))
    provider = FakeProvider(
        [ImageFinal(index=0, data_b64="AAA", mime="image/png")],
        raises=asyncio.CancelledError(),
    )

    with (
        bind_usage_context(UsageContext(operation="workflow")),
        _patch_provider(provider),
        pytest.raises(asyncio.CancelledError),
    ):
        await agent._generate_images("enhanced", "orig")

    assert len(repo.commands) == 1
    command = repo.commands[0]
    assert command.status == "cancelled"
    assert command.error_code is None


@pytest.mark.asyncio
async def test_openai_image_stream_records_openai_provider():
    repo = FakeRepository()
    agent = _agent(_recorder(repo), model_name="gpt-image-1", max_images=1)
    provider = FakeProvider(
        [
            ImageFinal(index=0, data_b64="ZZZ", mime="image/png"),
            ImageUsage(
                usage=NormalizedUsage(
                    input_tokens=50,
                    output_tokens=100,
                    total_tokens=150,
                    source="provider_reported",
                )
            ),
        ]
    )

    with bind_usage_context(UsageContext(operation="workflow")), _patch_provider(provider):
        await agent._generate_images("enhanced", "orig")

    command = repo.commands[0]
    assert command.provider == "openai"
    assert command.model == "gpt-image-1"
    assert command.usage.input_tokens == 50


@pytest.mark.asyncio
async def test_no_recording_when_recorder_absent():
    agent = _agent(None)
    provider = FakeProvider(
        [
            ImageFinal(index=0, data_b64="AAA", mime="image/png"),
            ImageUsage(usage=_gemini_usage()),
        ]
    )

    with bind_usage_context(UsageContext(operation="workflow")), _patch_provider(provider):
        images, _ = await agent._generate_images("enhanced", "orig")

    assert [img["data"] for img in images] == ["AAA"]


@pytest.mark.asyncio
async def test_persistence_runs_off_the_event_loop_thread():
    repo = FakeRepository()
    agent = _agent(_recorder(repo))
    loop_thread = threading.get_ident()
    provider = FakeProvider(
        [
            ImageFinal(index=0, data_b64="AAA", mime="image/png"),
            ImageUsage(usage=_gemini_usage()),
        ]
    )

    with bind_usage_context(UsageContext(operation="workflow")), _patch_provider(provider):
        await agent._generate_images("enhanced", "orig")

    assert repo.record_threads and repo.record_threads[0] != loop_thread
