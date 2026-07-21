from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.repositories.conversation_compaction import SummaryJobClaim
from app.workers.conversation_compaction import (
    ConversationCompactionWorker,
    WorkerOutcome,
    compute_retry_delay,
    dispatch_due_jobs,
    request_historical_backfill,
)


def _settings(**overrides):
    values = {
        "conversation_summary_lease_seconds": 120,
        "conversation_summary_timeout_seconds": 1,
        "conversation_summary_max_attempts": 5,
        "conversation_summary_retry_base_seconds": 5,
        "conversation_summary_retry_max_seconds": 900,
        "conversation_summary_reconcile_seconds": 60,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeRepository:
    def __init__(self, events, *, claim=True, cas=True, input_exists=True):
        self.events = events
        self.conversation_id = uuid4()
        self.owner_id = uuid4()
        self.claim_enabled = claim
        self.cas = cas
        self.input_exists = input_exists
        self.failed = []
        self.completed = []

    def claim_job(self, conversation_id, **_kwargs):
        self.events.append("lease_committed")
        if not self.claim_enabled:
            return None
        return SummaryJobClaim(conversation_id, 4, uuid4(), 1)

    def load_compaction_input(self, claim):
        self.events.append("input_loaded")
        if not self.input_exists:
            return None
        return SimpleNamespace(
            owner_id=self.owner_id,
            messages=({"sequence": 3}, {"sequence": 4}),
            summary_payload=None,
            summary_version=0,
            last_summarized_sequence=None,
        )

    def persist_memory_cas(self, claim, **kwargs):
        self.events.append("memory_cas")
        self.persist_kwargs = kwargs
        return self.cas

    def complete_claim(self, claim):
        self.events.append("complete")
        self.completed.append(claim)
        return "idle"

    def fail_claim(self, claim, **kwargs):
        self.events.append("fail")
        self.failed.append(kwargs)
        return True


class FakeCompactor:
    provider = "gemini"
    model = "gemini-2.5-flash"
    prompt_version = "v1"

    def __init__(self, events, result):
        self.events = events
        self.result = result

    async def compact(self, messages, **kwargs):
        self.events.append("provider_call")
        self.compact_kwargs = kwargs
        return self.result


def _result(*, success=True, error=None):
    memory = SimpleNamespace(
        model_dump=lambda: {"facts": ["fact"]},
    )
    return SimpleNamespace(
        success=success,
        error_code=error,
        memory=memory if success else None,
        last_summarized_sequence=4 if success else None,
        summary_token_count=3,
        input_token_count=18,
        reported_input_tokens=None,
        reported_cost_amount=None,
        reported_cost_currency=None,
        trigger=SimpleNamespace(message_count=2, token_count=20, token_strategy="fixed"),
        selection=SimpleNamespace(compactable_prefix=({"sequence": 3}, {"sequence": 4})),
    )


@pytest.mark.asyncio
async def test_lease_commits_before_provider_and_success_persists_then_completes() -> None:
    events = []
    repository = FakeRepository(events)
    worker = ConversationCompactionWorker(
        repository=repository,
        compactor=FakeCompactor(events, _result()),
        settings=_settings(),
        jitter=lambda _low, _high: 0,
    )

    outcome = await worker.execute(repository.conversation_id)

    assert outcome == WorkerOutcome("completed", "idle")
    assert events == [
        "lease_committed",
        "input_loaded",
        "provider_call",
        "memory_cas",
        "complete",
    ]
    assert repository.persist_kwargs["base_summary_version"] == 0
    assert repository.persist_kwargs["base_cursor"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("force", [False, True])
async def test_worker_passes_verified_owner_conversation_and_timeout(force) -> None:
    events = []
    repository = FakeRepository(events)
    compactor = FakeCompactor(events, _result())
    worker = ConversationCompactionWorker(
        repository=repository,
        compactor=compactor,
        settings=_settings(),
    )

    await worker.execute(repository.conversation_id, force=force)

    assert compactor.compact_kwargs["user_id"] == repository.owner_id
    assert compactor.compact_kwargs["conversation_id"] == repository.conversation_id
    assert compactor.compact_kwargs["force"] is force
    assert compactor.compact_kwargs["timeout_seconds"] == 1


@pytest.mark.asyncio
async def test_duplicate_notification_without_claim_is_idempotent_noop() -> None:
    events = []
    repository = FakeRepository(events, claim=False)
    worker = ConversationCompactionWorker(
        repository=repository,
        compactor=FakeCompactor(events, _result()),
        settings=_settings(),
    )

    outcome = await worker.execute(repository.conversation_id)

    assert outcome.code == "not_claimed"
    assert "provider_call" not in events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_code", "expected_status", "permanent"),
    [
        ("generation_failed", "retry", False),
        ("provider_timeout", "retry", False),
        ("invalid_memory", "dead", True),
        ("credential_unavailable", "dead", True),
    ],
)
async def test_failure_classification_and_sanitized_retry_state(
    error_code, expected_status, permanent
) -> None:
    events = []
    repository = FakeRepository(events)
    worker = ConversationCompactionWorker(
        repository=repository,
        compactor=FakeCompactor(events, _result(success=False, error=error_code)),
        settings=_settings(),
        jitter=lambda _low, _high: 0,
    )

    outcome = await worker.execute(repository.conversation_id)

    assert outcome.status == expected_status
    assert repository.failed[0]["permanent"] is permanent
    assert repository.failed[0]["error_code"] == error_code
    assert "fact" not in repository.failed[0]["error_code"]


@pytest.mark.asyncio
async def test_timeout_is_transient_and_uses_bounded_retry() -> None:
    events = []
    repository = FakeRepository(events)

    class SlowCompactor(FakeCompactor):
        async def compact(self, messages, **kwargs):
            await asyncio.sleep(0.05)

    worker = ConversationCompactionWorker(
        repository=repository,
        compactor=SlowCompactor(events, None),
        settings=_settings(conversation_summary_timeout_seconds=0.001),
        jitter=lambda _low, _high: 0,
    )

    outcome = await worker.execute(repository.conversation_id)

    assert outcome == WorkerOutcome("retry", "provider_timeout")
    assert repository.failed[0]["permanent"] is False


def test_exponential_backoff_is_bounded_and_jittered() -> None:
    assert compute_retry_delay(0, base_seconds=5, max_seconds=900, jitter=lambda a, b: 0) == 5
    assert compute_retry_delay(4, base_seconds=5, max_seconds=900, jitter=lambda a, b: 0) == 80
    assert compute_retry_delay(20, base_seconds=5, max_seconds=900, jitter=lambda a, b: b) == 900
    delay = compute_retry_delay(3, base_seconds=5, max_seconds=900, jitter=lambda a, b: b)
    assert 40 <= delay <= 50


@pytest.mark.asyncio
async def test_cas_conflict_does_not_complete_or_overwrite_newer_state() -> None:
    events = []
    repository = FakeRepository(events, cas=False)
    worker = ConversationCompactionWorker(
        repository=repository,
        compactor=FakeCompactor(events, _result()),
        settings=_settings(),
    )

    outcome = await worker.execute(repository.conversation_id)

    assert outcome.code == "cas_conflict"
    assert repository.completed == []
    assert repository.failed[0]["error_code"] == "cas_conflict"


def test_reconciler_debounces_before_best_effort_publish() -> None:
    events = []
    ids = [uuid4(), uuid4()]

    class Repository:
        def reconcile_due_jobs(self, **kwargs):
            events.append(("debounced", kwargs["dispatch_debounce_seconds"]))
            return ids

    def publisher(conversation_id):
        events.append(("published", conversation_id))
        if conversation_id == ids[0]:
            raise RuntimeError("broker unavailable with sensitive details")

    published = dispatch_due_jobs(
        Repository(),
        publisher,
        debounce_seconds=60,
        limit=100,
    )

    assert events[0] == ("debounced", 60)
    assert published == 1
    assert events[-1] == ("published", ids[1])


def test_backfill_is_rate_limited_idempotent_and_publishes_ids_only() -> None:
    first, second = (uuid4(), 8), (uuid4(), 12)
    calls = []

    class Repository:
        def list_backfill_candidates(self, *, limit):
            calls.append(("listed", limit))
            return [first, second]

        def request_backfill(self, conversation_id, requested_through_sequence):
            calls.append(("requested", conversation_id, requested_through_sequence))
            return conversation_id == first[0]

    published = []
    count = request_historical_backfill(
        Repository(),
        published.append,
        batch_size=2,
    )

    assert count == 1
    assert published == [str(first[0])]
    assert calls[0] == ("listed", 2)


def test_celery_compaction_task_payload_contains_only_conversation_id() -> None:
    from app.workers.conversation_compaction import compact_conversation_task

    conversation_id = str(uuid4())
    signature = compact_conversation_task.s(conversation_id)

    assert signature.args == (conversation_id,)
    assert signature.kwargs == {}


def test_disabled_compaction_tasks_do_not_build_workers(monkeypatch) -> None:
    from app.workers import conversation_compaction as worker_module

    monkeypatch.setattr(worker_module.settings, "conversation_summary_enabled", False)
    monkeypatch.setattr(
        worker_module,
        "_build_worker",
        lambda: pytest.fail("disabled compaction must not build a worker"),
    )

    assert worker_module.compact_conversation_task(str(uuid4()))["status"] == "disabled"
    assert worker_module.reconcile_conversation_summaries_task() == 0
    assert worker_module.backfill_conversation_summaries_task() == 0


def test_backfill_uses_separately_rate_limited_compaction_task() -> None:
    from app.workers import conversation_compaction as worker_module

    assert worker_module.compact_backfill_conversation_task.rate_limit == "10/m"
    assert worker_module.compact_conversation_task.rate_limit is None
