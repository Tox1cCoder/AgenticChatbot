"""Forked children: first-turn memory preparation and its bounded fallback.

A fork copies a transcript prefix but never the source's memory summary (it can
cover rows past the cutoff). Until the child has its own summary,
``ConversationHistoryProvider`` trims the copied prefix to the history budget
before routing runs, so a fact early in a long prefix would be absent from the
first child turn. ``ForkMemoryPreparer`` runs the child's durable compaction job
under its normal lease before the workflow request is built, bounded by a
timeout. Every way that cannot happen proceeds with ordinary bounded history and
a structured warning (controller ruling R4); none of them fails the turn.

Fake repositories, runners and AI services only: no database, no model call.
PostgreSQL behavior is in tests/integration/test_conversation_fork_history_postgres.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.api.ai_sdk import AISDKChatRequest, chat_ui_message_stream
from app.models.enums import MessageRole
from app.repositories.conversation_compaction import SummaryJobState
from app.schemas.message import MessageCreate, MessageRead
from app.services.conversation_fork_memory import (
    ForkMemoryPreparation,
    ForkMemoryPreparer,
    applicable_history_budget,
)
from app.services.event_streaming.events import make_event
from app.services.generation_registry import get_generation_registry
from app.services.message_service import MessageService
from app.workers.conversation_compaction import WorkerOutcome

USER = MessageRole.user.value
ASSISTANT = MessageRole.assistant.value
LOGGER = "app.services.conversation_fork_memory"
COMPLETED = WorkerOutcome("completed", "idle")

CHILD = uuid4()
OWNER = uuid4()
NOW = datetime(2026, 10, 9, tzinfo=UTC)


def _settings(**overrides: Any) -> SimpleNamespace:
    values = {
        "conversation_summary_enabled": True,
        "conversation_fork_memory_prepare_timeout_seconds": 5.0,
        "router_history_max_messages": 12,
        "router_history_max_tokens": 3000,
        "chat_history_max_messages": 24,
        "chat_history_max_tokens": 9000,
        "rag_history_max_messages": 12,
        "rag_history_max_tokens": 3000,
        "search_history_max_messages": 16,
        "search_history_max_tokens": 5000,
        "planning_history_max_messages": 16,
        "planning_history_max_tokens": 5000,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _row(sequence: int, sender: int, content: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        sender=sender,
        content=content,
        message_metadata={"fork_history": True},
        sequence=sequence,
        created_at=None,
        deleted_at=None,
    )


def _prefix(turns: int, *, answer_chars: int = 0) -> list[SimpleNamespace]:
    rows = []
    for index in range(turns):
        rows.append(_row(2 * index + 1, USER, f"question {index}"))
        answer = f"answer {index}" + ("x" * answer_chars)
        rows.append(_row(2 * index + 2, ASSISTANT, answer))
    return rows


class FakeRepository:
    """The five reads/one write the preparer uses, recorded in call order."""

    def __init__(
        self,
        rows,
        *,
        job_status: str | None = "pending",
        memory: Any = None,
        later_turn: bool = False,
        available_at: datetime | None = None,
    ):
        self.rows = rows
        self.job_status = job_status
        self.memory = memory
        self.later_turn = later_turn
        self.available_at = available_at
        self.calls: list[str] = []
        self.tail_limits: list[int | None] = []
        self.excluded: list[UUID | None] = []

    def has_messages_after(self, conversation_id, after_sequence, *, excluding_message_id=None):
        self.calls.append("later")
        self.excluded.append(excluding_message_id)
        return self.later_turn

    def get_owned_valid_memory(self, conversation_id: UUID, owner_id: UUID):
        self.calls.append("memory")
        return self.memory

    def load_fork_prefix_tail(self, conversation_id: UUID, through_sequence: int, *, limit):
        self.calls.append("prefix")
        self.tail_limits.append(limit)
        rows = [row for row in self.rows if row.sequence <= through_sequence]
        return rows[-limit:] if limit else rows

    def get_job_state(self, conversation_id: UUID) -> SummaryJobState | None:
        self.calls.append("job")
        if self.job_status is None:
            return None
        return SummaryJobState(self.job_status, self.available_at)

    def request_backfill(self, conversation_id: UUID, requested_through_sequence=None) -> bool:
        self.calls.append("reopen")
        self.job_status = "pending"
        return True


class Runner:
    """Stands in for ``run_conversation_compaction_now``: the worker under its lease."""

    def __init__(
        self,
        repository: FakeRepository,
        outcome: WorkerOutcome = COMPLETED,
        *,
        produces_memory: bool = True,
        error: Exception | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self.repository = repository
        self.outcome = outcome
        self.produces_memory = produces_memory
        self.error = error
        self.gate = gate
        self.calls: list[tuple[UUID, bool]] = []

    async def __call__(self, conversation_id: UUID, *, force: bool = False) -> WorkerOutcome:
        self.calls.append((conversation_id, force))
        if self.gate is not None:
            while not self.gate.is_set():
                await asyncio.sleep(0.01)
        if self.error is not None:
            raise self.error
        if self.produces_memory:
            self.repository.memory = SimpleNamespace(last_summarized_sequence=32)
        return self.outcome


def _preparer(repository, runner=None, **settings_overrides) -> ForkMemoryPreparer:
    return ForkMemoryPreparer(
        repository=repository,
        runner=runner or Runner(repository),
        settings=_settings(**settings_overrides),
    )


async def _prepare(preparer: ForkMemoryPreparer, *, through: int | None = 40, current=None):
    return await preparer.aprepare(
        conversation_id=CHILD,
        owner_id=OWNER,
        forked_through_sequence=through,
        current_message_id=current,
    )


def _warnings(caplog) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == LOGGER and record.levelno == logging.WARNING
    ]


# ---------------------------------------------------------------------------
# when preparation is needed
# ---------------------------------------------------------------------------


def test_the_container_gives_message_service_a_preparer_over_the_durable_worker():
    """Unwired, every fork would silently run its first turns on trimmed history."""
    from app.core.container import Container
    from app.workers.conversation_compaction import run_conversation_compaction_now

    assert Container.message_service.kwargs["fork_memory_preparer"] is (
        Container.fork_memory_preparer
    )
    preparer_kwargs = Container.fork_memory_preparer.kwargs
    assert preparer_kwargs["repository"] is Container.conversation_compaction_repository
    assert preparer_kwargs["runner"].provides is run_conversation_compaction_now


def test_the_applicable_budget_is_the_tightest_one_any_agent_applies():
    assert applicable_history_budget(_settings()) == (12, 3000)
    assert applicable_history_budget(
        _settings(router_history_max_messages=0, rag_history_max_messages=0)
    ) == (16, 3000)


async def test_an_ordinary_conversation_is_never_inspected():
    repository = FakeRepository(_prefix(30))
    runner = Runner(repository)

    result = await _prepare(_preparer(repository, runner), through=None)

    assert result == ForkMemoryPreparation("skipped", "not_a_fork")
    assert repository.calls == []
    assert runner.calls == []


async def test_a_child_that_already_has_valid_memory_is_left_alone():
    repository = FakeRepository(_prefix(30), memory=SimpleNamespace(last_summarized_sequence=40))
    runner = Runner(repository)

    result = await _prepare(_preparer(repository, runner))

    assert result.status == "skipped"
    assert result.reason == "memory_present"
    assert runner.calls == []


async def test_only_the_first_turn_is_prepared(caplog):
    """Any row after the copied prefix, besides this turn's own message, ends preparation.

    Each persisted reply re-queues the job with its attempts reset, so a
    failing provider would otherwise cost a claim and a wait on every turn.
    This also covers a later turn whose memory an edit invalidated.
    """
    repository = FakeRepository(_prefix(20), job_status="retry", later_turn=True)
    runner = Runner(repository)
    current = uuid4()

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        result = await _prepare(_preparer(repository, runner), current=current)

    assert result == ForkMemoryPreparation("skipped", "not_first_turn")
    assert repository.calls == ["later"]
    assert repository.excluded == [current]
    assert runner.calls == []
    assert _warnings(caplog) == []


async def test_a_prefix_inside_the_history_budget_needs_no_summary(caplog):
    repository = FakeRepository(_prefix(3))
    runner = Runner(repository)

    result = await _prepare(_preparer(repository, runner), through=6)

    assert result == ForkMemoryPreparation("skipped", "within_history_budget")
    assert runner.calls == []
    assert repository.tail_limits == [13]
    assert _warnings(caplog) == []


async def test_a_prefix_over_the_message_budget_runs_the_childs_job_under_its_lease():
    repository = FakeRepository(_prefix(20))
    runner = Runner(repository)

    result = await _prepare(_preparer(repository, runner))

    assert result == ForkMemoryPreparation("prepared", "summary_ready")
    # force: the prefix is over the *history* budget, which can be well below
    # the background trigger; an unforced run would skip and idle the job.
    assert runner.calls == [(CHILD, True)]
    assert repository.calls == ["later", "memory", "prefix", "job", "memory"]


async def test_a_few_very_long_messages_exceed_the_token_budget():
    repository = FakeRepository(_prefix(2, answer_chars=20_000))
    runner = Runner(repository)

    result = await _prepare(_preparer(repository, runner), through=4)

    assert result.status == "prepared"
    assert runner.calls == [(CHILD, True)]


async def test_a_job_the_background_worker_skipped_is_reopened_before_running():
    # A 40-row prefix is over the router budget but under the background
    # trigger, so a worker that got there first idled the job without memory.
    repository = FakeRepository(_prefix(20), job_status="idle")
    runner = Runner(repository)

    result = await _prepare(_preparer(repository, runner))

    assert result.status == "prepared"
    assert repository.calls[:5] == ["later", "memory", "prefix", "job", "reopen"]


async def test_preparation_only_counts_rows_through_the_cutoff():
    repository = FakeRepository(_prefix(3) + _prefix(20)[6:])
    runner = Runner(repository)

    result = await _prepare(_preparer(repository, runner), through=6)

    assert result.status == "skipped"
    assert runner.calls == []


# ---------------------------------------------------------------------------
# the fallback: bounded history, a warning, and the turn proceeds
# ---------------------------------------------------------------------------


def _assert_fallback(caplog, result, reason: str) -> None:
    assert result.status == "fallback"
    assert result.reason == reason
    [warning] = _warnings(caplog)
    message = warning.getMessage()
    assert str(CHILD) in message
    assert f"reason={reason}" in message
    assert "bounded history" in message
    assert warning.fork_memory == {
        "conversation_id": str(CHILD),
        "reason": reason,
        "detail": result.detail,
    }


async def test_disabled_compaction_falls_back_without_touching_the_job(caplog):
    repository = FakeRepository(_prefix(20))
    runner = Runner(repository)

    result = await _prepare(_preparer(repository, runner, conversation_summary_enabled=False))

    _assert_fallback(caplog, result, "compaction_disabled")
    assert runner.calls == []
    assert "job" not in repository.calls


async def test_a_busy_lease_falls_back_instead_of_competing_for_it(caplog):
    repository = FakeRepository(_prefix(20), job_status="processing")
    runner = Runner(repository)

    result = await _prepare(_preparer(repository, runner))

    _assert_fallback(caplog, result, "lease_busy")
    assert runner.calls == []


async def test_a_lease_taken_between_the_check_and_the_claim_also_falls_back(caplog):
    repository = FakeRepository(_prefix(20))
    runner = Runner(repository, WorkerOutcome("noop", "not_claimed"), produces_memory=False)

    result = await _prepare(_preparer(repository, runner))

    _assert_fallback(caplog, result, "lease_busy")


async def test_a_first_turn_with_a_dead_job_falls_back_without_reviving_it(caplog):
    repository = FakeRepository(_prefix(20), job_status="dead")
    runner = Runner(repository)

    result = await _prepare(_preparer(repository, runner))

    _assert_fallback(caplog, result, "summary_job_dead")
    assert runner.calls == []
    assert "reopen" not in repository.calls


async def test_a_retry_still_in_backoff_is_not_claimed_early(caplog):
    later = datetime.now(UTC) + timedelta(minutes=5)
    repository = FakeRepository(_prefix(20), job_status="retry", available_at=later)
    runner = Runner(repository)

    result = await _prepare(_preparer(repository, runner))

    _assert_fallback(caplog, result, "summary_job_backoff")
    assert runner.calls == []


async def test_a_retry_whose_backoff_has_elapsed_runs():
    earlier = datetime.now(UTC) - timedelta(seconds=1)
    repository = FakeRepository(_prefix(20), job_status="retry", available_at=earlier)
    runner = Runner(repository)

    result = await _prepare(_preparer(repository, runner))

    assert result.status == "prepared"
    assert runner.calls == [(CHILD, True)]


async def test_a_summary_provider_failure_falls_back(caplog):
    repository = FakeRepository(_prefix(20))
    runner = Runner(repository, WorkerOutcome("retry", "generation_failed"), produces_memory=False)

    result = await _prepare(_preparer(repository, runner))

    _assert_fallback(caplog, result, "summary_failed")
    assert result.detail == "generation_failed"


async def test_a_run_that_completes_without_memory_falls_back(caplog):
    repository = FakeRepository(_prefix(20))
    runner = Runner(repository, WorkerOutcome("skipped", "idle"), produces_memory=False)

    result = await _prepare(_preparer(repository, runner))

    _assert_fallback(caplog, result, "summary_failed")
    assert result.detail == "idle"


async def test_an_unexpected_error_falls_back(caplog):
    repository = FakeRepository(_prefix(20))
    runner = Runner(repository, error=RuntimeError("broker unavailable"))

    result = await _prepare(_preparer(repository, runner))

    _assert_fallback(caplog, result, "preparation_error")
    assert result.detail == "RuntimeError"


async def test_a_slow_summary_times_out_and_the_turn_does_not_wait_for_it(caplog):
    repository = FakeRepository(_prefix(20))
    gate = threading.Event()
    runner = Runner(repository, gate=gate)
    preparer = _preparer(repository, runner, conversation_fork_memory_prepare_timeout_seconds=0.2)

    started = time.monotonic()
    try:
        result = await _prepare(preparer)
        elapsed = time.monotonic() - started
    finally:
        gate.set()

    _assert_fallback(caplog, result, "timeout")
    assert elapsed < 2.0
    assert runner.calls == [(CHILD, True)]


# ---------------------------------------------------------------------------
# the next-generation path: preparation precedes routing and history assembly
# ---------------------------------------------------------------------------


class _RecordingPreparer:
    def __init__(self, order: list[str], result: ForkMemoryPreparation) -> None:
        self.order = order
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def aprepare(self, **kwargs) -> ForkMemoryPreparation:
        self.calls.append(kwargs)
        self.order.append("prepare")
        return self.result


def _message_service(order: list[str], preparer: Any, *, through: int | None = 40):
    service = MessageService.__new__(MessageService)
    for name in (
        "_turn_coordinator",
        "generation_control_service",
        "custom_agent_service",
        "chat_image_service",
        "web_image_service",
        "tool_approval_setting_repository",
        "tool_approval_repository",
        "hitl_interrupt_repository",
        "task_plan_service",
        "project_context_service",
        "redis_client",
    ):
        setattr(service, name, None)
    service.fork_memory_preparer = preparer
    conversation = SimpleNamespace(
        id=CHILD,
        title="Trip (fork)",
        persona_prompt=None,
        planning_mode_enabled=False,
        plan_lifecycle=None,
        owner_id=OWNER,
        forked_through_sequence=through,
    )

    async def acreate(entity):
        return SimpleNamespace(
            id=uuid4(),
            conversation_id=CHILD,
            sender=USER,
            content=entity["content"],
            message_metadata={},
            feedback=None,
            created_at=NOW,
            updated_at=NOW,
            deleted_at=None,
            sequence=41,
        )

    async def aget_by_id(_conversation_id):
        return conversation

    async def avalidate(*_args):
        return None

    service.repository = SimpleNamespace(acreate=acreate)
    service.conversation_validation_utils = SimpleNamespace(
        avalidate_conversation_access=avalidate,
        conversation_repository=SimpleNamespace(aget_by_id=aget_by_id),
    )
    requests: list[Any] = []

    async def stream(request):
        order.append("workflow")
        requests.append(request)
        yield make_event("message_delta", sequence=1, data={"text": "fresh answer"})
        yield make_event("complete", sequence=2, data={"response": None})

    service.ai_service = SimpleNamespace(
        invalidate_history_cache=lambda conversation_id: order.append("invalidate"),
        execute_request_stream=stream,
    )

    async def persist(**kwargs):
        return MessageRead.model_validate(
            SimpleNamespace(
                id=kwargs["message_id"],
                conversation_id=CHILD,
                sender=ASSISTANT,
                content="fresh answer",
                message_metadata={},
                feedback=None,
                created_at=NOW,
                updated_at=NOW,
                deleted_at=None,
            )
        )

    async def no_compaction(**_kwargs):
        return None

    service._persist_completed_workflow_response = persist
    service._compact_checkpoint_after_persist = no_compaction
    return service, requests


@pytest.fixture(autouse=True)
def _clean_registry():
    registry = get_generation_registry()
    registry._store.clear()  # noqa: SLF001 - test isolation
    yield
    registry._store.clear()  # noqa: SLF001


async def _stream_turn(service: MessageService) -> list[Any]:
    stream = service.create_message_stream(
        MessageCreate(conversation_id=CHILD, content="what was my code?", role=MessageRole.user),
        OWNER,
    )
    return [event async for event in stream]


async def test_streaming_prepares_memory_before_the_workflow_runs():
    order: list[str] = []
    preparer = _RecordingPreparer(order, ForkMemoryPreparation("prepared", "summary_ready"))
    service, requests = _message_service(order, preparer)

    events = await _stream_turn(service)

    assert [event.type for event in events][-1] == "complete"
    assert order.index("prepare") < order.index("workflow")
    # The new summary changes the cache key anyway; dropping the entry is cheap.
    assert order.index("prepare") < order.index("invalidate", order.index("prepare"))
    [call] = preparer.calls
    assert (call["conversation_id"], call["owner_id"], call["forked_through_sequence"]) == (
        CHILD,
        OWNER,
        40,
    )
    # The turn's own user row is the one row after the prefix that is allowed.
    assert call["current_message_id"] == UUID(requests[0].user_message_id)
    assert requests[0].conversation_id == str(CHILD)


@pytest.mark.parametrize("reason", ["compaction_disabled", "lease_busy", "timeout"])
async def test_streaming_proceeds_with_bounded_history_when_preparation_falls_back(reason):
    order: list[str] = []
    preparer = _RecordingPreparer(order, ForkMemoryPreparation("fallback", reason))
    service, _requests = _message_service(order, preparer)

    events = await _stream_turn(service)

    assert [event.type for event in events if event.type == "error"] == []
    assert [event.type for event in events][-1] == "complete"
    assert order == ["invalidate", "prepare", "workflow"]


async def test_the_ai_sdk_entry_point_prepares_before_the_workflow_runs():
    order: list[str] = []
    preparer = _RecordingPreparer(order, ForkMemoryPreparation("prepared", "summary_ready"))
    service, _requests = _message_service(order, preparer)

    response = await chat_ui_message_stream(
        conversation_id=CHILD,
        payload=AISDKChatRequest(
            messages=[{"role": "user", "parts": [{"type": "text", "text": "what was my code?"}]}]
        ),
        message_service=service,
        current_user_id=OWNER,
    )
    body = "".join([chunk async for chunk in response.body_iterator])

    payloads = [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]
    assert "fresh answer" in "".join(p.get("delta", "") for p in payloads)
    assert order.index("prepare") < order.index("workflow")


async def test_an_ordinary_conversation_reaches_the_preparer_as_not_a_fork():
    order: list[str] = []
    preparer = _RecordingPreparer(order, ForkMemoryPreparation("skipped", "not_a_fork"))
    service, _requests = _message_service(order, preparer, through=None)

    await _stream_turn(service)

    assert preparer.calls[0]["forked_through_sequence"] is None


async def test_a_service_without_a_preparer_still_answers():
    order: list[str] = []
    service, _requests = _message_service(order, None)

    events = await _stream_turn(service)

    assert [event.type for event in events][-1] == "complete"
    assert "prepare" not in order
