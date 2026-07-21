"""Tests for the one-attempt model-usage recorder (``app.usage.recorder``).

The recorder wraps a single provider call, executes it exactly once (never
adding a provider retry loop), and records the attempt's outcome to the Task 3
repository off the event loop. Persistence failures must never break the
caller's provider path: they enqueue a content-free Celery failed-write task,
and if that enqueue also fails they are logged at bounded cardinality and
swallowed. Provider responses are always returned and provider exceptions
always re-raise regardless of persistence outcome.
"""

from __future__ import annotations

import asyncio
import json
import threading
import typing
from uuid import uuid4

import pytest
from prometheus_client import CollectorRegistry

from app.observability.model_usage import ModelUsageMetrics
from app.repositories.model_usage import RecordEventCommand, RecordResult
from app.usage.context import bind_usage_context
from app.usage.recorder import (
    ModelUsageRecorder,
    classify_error,
    classify_status,
    deserialize_record_command,
    serialize_record_command,
)
from app.usage.types import NormalizedUsage, UsageContext, UsageOperation


class FakeRepository:
    """In-memory double for ``ModelUsageRepository`` (owns no session)."""

    def __init__(self, *, fail_with: Exception | None = None, inserted: bool = True) -> None:
        self.commands: list[RecordEventCommand] = []
        self.record_threads: list[int] = []
        self._fail_with = fail_with
        self._inserted = inserted

    def record_event(self, command: RecordEventCommand) -> RecordResult:
        self.record_threads.append(threading.get_ident())
        if self._fail_with is not None:
            raise self._fail_with
        self.commands.append(command)
        return RecordResult(
            inserted=self._inserted,
            event_id=uuid4() if self._inserted else None,
        )


class CountingCall:
    """Async callable that records how many times it was invoked."""

    def __init__(self, *, result=None, raises: BaseException | None = None) -> None:
        self.result = result
        self.raises = raises
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.result


def _metrics() -> ModelUsageMetrics:
    return ModelUsageMetrics(registry=CollectorRegistry())


def test_streaming_begin_return_annotation_is_explicit_protocol():
    from app.usage.recorder import StreamingAttempt

    hints = typing.get_type_hints(ModelUsageRecorder.begin_streaming_attempt)
    assert hints["return"] is StreamingAttempt


def _recorder(repo, *, enqueue=None, metrics=None) -> ModelUsageRecorder:
    return ModelUsageRecorder(
        repository=repo,
        enqueue_failed_write=enqueue if enqueue is not None else (lambda payload: None),
        metrics=metrics or _metrics(),
    )


# --- success path --------------------------------------------------------


async def test_success_records_provider_reported_usage_and_returns_response():
    repo = FakeRepository()
    recorder = _recorder(repo)
    call = CountingCall(result={"usage_metadata": {"input_tokens": 5, "output_tokens": 7}})

    with bind_usage_context(UsageContext(operation="chat", user_id=uuid4())):
        response = await recorder.record_one_async_attempt(
            call=call, provider="gemini", model="gemini-3-flash", operation=UsageOperation()
        )

    assert response is call.result
    assert call.calls == 1, "recorder must execute the supplied call exactly once"
    assert len(repo.commands) == 1
    command = repo.commands[0]
    assert command.status == "success"
    assert command.usage.source == "provider_reported"
    assert command.usage.input_tokens == 5
    assert command.usage.output_tokens == 7
    assert command.error_code is None
    assert command.attempt == 1
    assert command.latency_ms >= 0
    assert command.started_at.tzinfo is not None
    assert command.completed_at.tzinfo is not None


async def test_success_records_metric():
    repo = FakeRepository()
    metrics = _metrics()
    recorder = _recorder(repo, metrics=metrics)
    call = CountingCall(result={"usage_metadata": {"input_tokens": 1}})

    with bind_usage_context(UsageContext(operation="chat")):
        await recorder.record_one_async_attempt(
            call=call, provider="openai", model="gpt-5", operation=UsageOperation()
        )

    value = metrics.attempts.labels(
        provider="openai", operation="chat", status="success", source="provider_reported"
    )._value.get()
    assert value == 1.0


async def test_tracking_disabled_executes_provider_without_allocating_or_recording(monkeypatch):
    from app.core.config import settings

    repo = FakeRepository()
    metrics = _metrics()
    recorder = _recorder(repo, metrics=metrics)
    operation = UsageOperation()
    call = CountingCall(result={"usage_metadata": {"input_tokens": 1}})
    monkeypatch.setattr(settings, "model_usage_tracking_enabled", False)

    assert (
        await recorder.record_one_async_attempt(
            call=call, provider="gemini", model="any-model", operation=operation
        )
        is call.result
    )
    assert call.calls == 1
    assert repo.commands == []
    assert operation.allocate_attempt() == 1
    assert metrics.render().decode("utf-8").count("model_usage_attempts_total{") == 0


async def test_tracking_disabled_stream_handle_is_a_noop(monkeypatch):
    from app.core.config import settings

    repo = FakeRepository()
    recorder = _recorder(repo)
    operation = UsageOperation()
    monkeypatch.setattr(settings, "model_usage_tracking_enabled", False)

    handle = recorder.begin_streaming_attempt(
        provider="openai", model="any-image-model", operation=operation
    )
    handle.set_usage(NormalizedUsage(total_tokens=3, source="provider_reported"))
    handle.note_generated_image()
    await handle.finalize("success")

    assert repo.commands == []
    assert operation.allocate_attempt() == 1


def test_tracking_disabled_sync_executes_provider_without_recording(monkeypatch):
    from app.core.config import settings

    repo = FakeRepository()
    recorder = _recorder(repo)
    operation = UsageOperation()
    calls = 0
    monkeypatch.setattr(settings, "model_usage_tracking_enabled", False)

    def provider_call():
        nonlocal calls
        calls += 1
        return "provider-response"

    assert (
        recorder.record_one_sync_attempt(
            call=provider_call,
            provider="openai",
            model="any-model",
            operation=operation,
        )
        == "provider-response"
    )
    assert calls == 1
    assert repo.commands == []
    assert operation.allocate_attempt() == 1


async def test_duplicate_write_emits_duplicate_not_stored_metric():
    metrics = _metrics()
    recorder = _recorder(FakeRepository(inserted=False), metrics=metrics)

    await recorder.record_one_async_attempt(
        call=CountingCall(result={}),
        provider="openai",
        model="any-model",
        operation=UsageOperation(),
    )

    assert metrics.persistence.labels(outcome="duplicate", failure_class="none")._value.get() == 1
    assert metrics.persistence.labels(outcome="stored", failure_class="none")._value.get() == 0


# --- failure classification ---------------------------------------------


async def test_timeout_exception_records_timeout_status_and_reraises():
    repo = FakeRepository()
    recorder = _recorder(repo)
    call = CountingCall(raises=TimeoutError("slow"))

    with bind_usage_context(UsageContext(operation="chat")), pytest.raises(TimeoutError):
        await recorder.record_one_async_attempt(
            call=call, provider="gemini", model="gemini-3-flash", operation=UsageOperation()
        )

    assert call.calls == 1
    assert repo.commands[0].status == "timeout"
    assert repo.commands[0].error_code == "TimeoutError"


async def test_provider_timeout_class_name_is_classified_as_timeout():
    class ProviderReadTimeout(Exception):
        pass

    repo = FakeRepository()
    recorder = _recorder(repo)
    call = CountingCall(raises=ProviderReadTimeout("upstream"))

    with bind_usage_context(UsageContext(operation="chat")), pytest.raises(ProviderReadTimeout):
        await recorder.record_one_async_attempt(
            call=call, provider="openai", model="gpt-5", operation=UsageOperation()
        )

    assert repo.commands[0].status == "timeout"
    assert repo.commands[0].error_code == "ProviderReadTimeout"


async def test_cancellation_records_cancelled_and_reraises():
    repo = FakeRepository()
    recorder = _recorder(repo)
    call = CountingCall(raises=asyncio.CancelledError())

    with (
        bind_usage_context(UsageContext(operation="chat")),
        pytest.raises(asyncio.CancelledError),
    ):
        await recorder.record_one_async_attempt(
            call=call, provider="gemini", model="gemini-3-flash", operation=UsageOperation()
        )

    assert call.calls == 1
    assert repo.commands[0].status == "cancelled"
    assert repo.commands[0].error_code is None


async def test_error_reraises_and_error_code_excludes_message():
    repo = FakeRepository()
    recorder = _recorder(repo)
    secret_message = "prompt-leak-should-never-be-recorded"
    call = CountingCall(raises=ValueError(secret_message))

    with bind_usage_context(UsageContext(operation="chat")), pytest.raises(ValueError):
        await recorder.record_one_async_attempt(
            call=call, provider="openai", model="gpt-5", operation=UsageOperation()
        )

    command = repo.commands[0]
    assert command.status == "error"
    assert command.error_code == "ValueError"
    assert secret_message not in (command.error_code or "")


# --- attempt allocation --------------------------------------------------


async def test_attempt_numbers_are_operation_scoped_and_monotonic():
    repo = FakeRepository()
    recorder = _recorder(repo)
    operation = UsageOperation()
    call = CountingCall(result={"usage": {"total_tokens": 3}})

    with bind_usage_context(UsageContext(operation="chat")):
        await recorder.record_one_async_attempt(
            call=call, provider="gemini", model="gemini-3-flash", operation=operation
        )
        await recorder.record_one_async_attempt(
            call=call, provider="gemini", model="gemini-3-flash", operation=operation
        )

    assert [c.attempt for c in repo.commands] == [1, 2]
    assert [c.event_key for c in repo.commands] == [
        f"{operation.operation_id}:1",
        f"{operation.operation_id}:2",
    ]


# --- estimate fallback ---------------------------------------------------


async def test_unavailable_usage_uses_estimate_callback():
    repo = FakeRepository()
    recorder = _recorder(repo)
    call = CountingCall(result={"no": "usage here"})

    def estimate(response):
        return NormalizedUsage(input_tokens=42, source="locally_estimated")

    with bind_usage_context(UsageContext(operation="chat")):
        await recorder.record_one_async_attempt(
            call=call,
            provider="gemini",
            model="gemini-3-flash",
            operation=UsageOperation(),
            estimate=estimate,
        )

    command = repo.commands[0]
    assert command.status == "success"
    assert command.usage.source == "locally_estimated"
    assert command.usage.input_tokens == 42


async def test_unavailable_usage_without_estimate_records_unavailable():
    repo = FakeRepository()
    recorder = _recorder(repo)
    call = CountingCall(result={"no": "usage here"})

    with bind_usage_context(UsageContext(operation="chat")):
        await recorder.record_one_async_attempt(
            call=call, provider="gemini", model="gemini-3-flash", operation=UsageOperation()
        )

    assert repo.commands[0].usage.source == "unavailable"


# --- persistence failure fallback ----------------------------------------


async def test_repository_failure_enqueues_failed_write_and_still_returns_response():
    repo = FakeRepository(fail_with=RuntimeError("db unavailable"))
    enqueued: list[dict] = []
    recorder = _recorder(repo, enqueue=enqueued.append)
    call = CountingCall(result={"usage_metadata": {"input_tokens": 9}})

    with bind_usage_context(UsageContext(operation="chat", user_id=uuid4())):
        response = await recorder.record_one_async_attempt(
            call=call, provider="gemini", model="gemini-3-flash", operation=UsageOperation()
        )

    assert response is call.result, "provider response must survive a persistence failure"
    assert len(enqueued) == 1
    payload = enqueued[0]
    # Payload must be JSON-safe primitives only.
    round_trip = json.loads(json.dumps(payload))
    assert round_trip == payload


async def test_failed_write_payload_excludes_prompts_and_responses():
    repo = FakeRepository(fail_with=RuntimeError("db unavailable"))
    enqueued: list[dict] = []
    recorder = _recorder(repo, enqueue=enqueued.append)
    call = CountingCall(
        result={
            "usage_metadata": {"input_tokens": 3},
            "prompt": "TOP SECRET USER PROMPT",
            "content": "sensitive model response body",
        }
    )

    with bind_usage_context(UsageContext(operation="chat")):
        await recorder.record_one_async_attempt(
            call=call, provider="gemini", model="gemini-3-flash", operation=UsageOperation()
        )

    serialized = json.dumps(enqueued[0])
    assert "TOP SECRET USER PROMPT" not in serialized
    assert "sensitive model response body" not in serialized
    forbidden_keys = {"prompt", "response", "responses", "messages", "content", "tool_args", "raw"}
    assert forbidden_keys.isdisjoint(enqueued[0].keys())


async def test_broker_failure_is_swallowed_and_logged(caplog):
    repo = FakeRepository(fail_with=RuntimeError("db unavailable"))

    def failing_enqueue(payload):
        raise RuntimeError("broker unreachable")

    recorder = _recorder(repo, enqueue=failing_enqueue)
    call = CountingCall(result={"usage_metadata": {"input_tokens": 2}})

    with caplog.at_level("ERROR"), bind_usage_context(UsageContext(operation="chat")):
        response = await recorder.record_one_async_attempt(
            call=call, provider="gemini", model="gemini-3-flash", operation=UsageOperation()
        )

    assert response is call.result
    assert any(
        "model_usage" in record.name or "model_usage" in record.message for record in caplog.records
    )


async def test_error_path_persistence_failure_reraises_original_exception():
    repo = FakeRepository(fail_with=RuntimeError("db unavailable"))

    def failing_enqueue(payload):
        raise RuntimeError("broker unreachable")

    recorder = _recorder(repo, enqueue=failing_enqueue)
    call = CountingCall(raises=ValueError("provider failed"))

    with (
        bind_usage_context(UsageContext(operation="chat")),
        pytest.raises(ValueError, match="provider failed"),
    ):
        await recorder.record_one_async_attempt(
            call=call, provider="openai", model="gpt-5", operation=UsageOperation()
        )


# --- non-blocking persistence --------------------------------------------


async def test_async_persistence_runs_off_the_event_loop_thread():
    repo = FakeRepository()
    recorder = _recorder(repo)
    call = CountingCall(result={"usage_metadata": {"input_tokens": 1}})
    loop_thread = threading.get_ident()

    with bind_usage_context(UsageContext(operation="chat")):
        await recorder.record_one_async_attempt(
            call=call, provider="gemini", model="gemini-3-flash", operation=UsageOperation()
        )

    assert repo.record_threads, "record_event was never invoked"
    assert repo.record_threads[0] != loop_thread, (
        "synchronous DB I/O must run via asyncio.to_thread, not on the event loop"
    )


# --- sync wrapper --------------------------------------------------------


def test_sync_wrapper_records_success_and_returns_response():
    repo = FakeRepository()
    recorder = _recorder(repo)
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        return {"usage": {"total_tokens": 11}}

    with bind_usage_context(UsageContext(operation="rag")):
        response = recorder.record_one_sync_attempt(
            call=call, provider="gemini", model="gemini-3-flash", operation=UsageOperation()
        )

    assert response == {"usage": {"total_tokens": 11}}
    assert calls["n"] == 1
    assert repo.commands[0].status == "success"
    assert repo.commands[0].usage.total_tokens == 11


def test_sync_wrapper_reraises_and_records_error():
    repo = FakeRepository()
    recorder = _recorder(repo)

    def call():
        raise ValueError("boom")

    with bind_usage_context(UsageContext(operation="rag")), pytest.raises(ValueError):
        recorder.record_one_sync_attempt(
            call=call, provider="gemini", model="gemini-3-flash", operation=UsageOperation()
        )

    assert repo.commands[0].status == "error"
    assert repo.commands[0].error_code == "ValueError"


def test_sync_wrapper_allocates_operation_scoped_attempts():
    repo = FakeRepository()
    recorder = _recorder(repo)
    operation = UsageOperation()

    def call():
        return {"usage": {"total_tokens": 1}}

    with bind_usage_context(UsageContext(operation="rag")):
        recorder.record_one_sync_attempt(
            call=call, provider="gemini", model="gemini-3-flash", operation=operation
        )
        recorder.record_one_sync_attempt(
            call=call, provider="gemini", model="gemini-3-flash", operation=operation
        )

    assert [c.attempt for c in repo.commands] == [1, 2]


def test_sync_wrapper_repository_failure_enqueues_and_returns_response():
    repo = FakeRepository(fail_with=RuntimeError("db unavailable"))
    enqueued: list[dict] = []
    recorder = _recorder(repo, enqueue=enqueued.append)

    def call():
        return {"usage": {"total_tokens": 1}}

    with bind_usage_context(UsageContext(operation="rag")):
        response = recorder.record_one_sync_attempt(
            call=call, provider="gemini", model="gemini-3-flash", operation=UsageOperation()
        )

    assert response == {"usage": {"total_tokens": 1}}
    assert len(enqueued) == 1


# --- classification helpers ----------------------------------------------


def test_classify_status_maps_known_exceptions():
    assert classify_status(TimeoutError()) == "timeout"
    assert classify_status(asyncio.TimeoutError()) == "timeout"
    assert classify_status(ValueError()) == "error"


def test_classify_error_returns_bounded_class_name_not_message():
    err = ValueError("a very long message containing PII like an email address")
    assert classify_error(err) == "ValueError"


# --- serialization round-trip --------------------------------------------


def test_serialize_deserialize_round_trip_preserves_event_key():
    from datetime import datetime, timezone

    started = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
    completed = datetime(2026, 7, 20, 12, 0, 1, tzinfo=timezone.utc)
    operation_id = uuid4()
    command = RecordEventCommand(
        operation_id=operation_id,
        attempt=3,
        context=UsageContext(operation="chat", user_id=uuid4(), conversation_id=uuid4()),
        usage=NormalizedUsage(input_tokens=5, output_tokens=7, source="provider_reported"),
        provider="gemini",
        model="gemini-3-flash",
        status="success",
        latency_ms=123,
        started_at=started,
        completed_at=completed,
    )

    payload = serialize_record_command(command)
    # Must survive a JSON round-trip.
    payload = json.loads(json.dumps(payload))
    restored = deserialize_record_command(payload)

    assert restored.event_key == command.event_key
    assert restored.attempt == 3
    assert restored.usage == command.usage
    assert restored.context == command.context
    assert restored.started_at == started
    assert restored.completed_at == completed
