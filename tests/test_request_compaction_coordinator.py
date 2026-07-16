from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ai.request_compaction import RequestCompactionCoordinator


class _Repository:
    def __init__(self):
        self.requested = []
        self.memory = None

    def request_backfill(self, conversation_id):
        self.requested.append(conversation_id)
        return True

    def get_owned_valid_memory(self, conversation_id, user_id):
        return self.memory


def test_durable_request_persists_before_publishing():
    repository = _Repository()
    events = []
    coordinator = RequestCompactionCoordinator(
        repository=repository,
        publisher=lambda conversation_id: events.append(("publish", conversation_id)),
        runner=lambda *_args, **_kwargs: None,
        enabled=True,
    )
    conversation_id = uuid4()

    assert coordinator.request_durable(conversation_id) is True
    assert repository.requested == [conversation_id]
    assert events == [("publish", conversation_id)]


@pytest.mark.asyncio
async def test_emergency_compaction_forces_worker_and_returns_valid_memory():
    repository = _Repository()
    conversation_id = uuid4()
    user_id = uuid4()
    calls = []
    published = []

    async def runner(target, *, force):
        calls.append((target, force))
        repository.memory = SimpleNamespace(
            summary_payload={
                "facts": ["The user prefers concise answers."],
                "decisions": [],
                "constraints": [],
                "preferences": [],
                "open_questions": [],
                "tool_outcomes": [],
            },
            last_summarized_sequence=8,
        )

    coordinator = RequestCompactionCoordinator(
        repository=repository,
        publisher=published.append,
        runner=runner,
        enabled=True,
    )

    reference = await coordinator.compact_now(conversation_id, user_id)

    assert calls == [(conversation_id, True)]
    assert published == []
    assert reference is not None
    assert reference.cursor == 8
    assert "BEGIN_UNTRUSTED_CONVERSATION_MEMORY_JSON" in reference.content


@pytest.mark.asyncio
async def test_disabled_coordinator_never_requests_or_runs_work():
    repository = _Repository()
    coordinator = RequestCompactionCoordinator(
        repository=repository,
        publisher=lambda _conversation_id: pytest.fail("must not publish"),
        runner=lambda *_args, **_kwargs: pytest.fail("must not run"),
        enabled=False,
    )

    assert coordinator.request_durable(uuid4()) is False
    assert await coordinator.compact_now(uuid4(), uuid4()) is None
    assert repository.requested == []
