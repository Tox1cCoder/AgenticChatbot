from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models.enums import MessageRole
from app.repositories.message import MessageRepository


class _CommittedPersistence:
    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.events = events

    def persist_message(self, message_data):
        row = SimpleNamespace(**message_data, sequence=2)
        self.events.append(("committed", row.id))
        return row


def _message_data(*, role: MessageRole, metadata=None, content="answer"):
    return {
        "id": uuid4(),
        "conversation_id": uuid4(),
        "sender": role.value,
        "content": content,
        "message_metadata": metadata or {},
    }


@pytest.mark.parametrize(
    "terminal_metadata",
    [
        {},
        {"streamed": True},
        {"partial": True},
        {"stopped": True, "partial": True},
        {"resumed": True},
        {"error": "safe_error"},
    ],
    ids=["ordinary", "streamed", "partial", "stopped", "resumed", "error"],
)
def test_every_terminal_assistant_commit_publishes_after_atomic_persistence(
    terminal_metadata,
) -> None:
    events: list[tuple[str, object]] = []
    repository = MessageRepository(
        session_factory=lambda: None,
        compaction_repository=_CommittedPersistence(events),
        compaction_publisher=lambda conversation_id: events.append(("published", conversation_id)),
    )
    message = _message_data(role=MessageRole.assistant, metadata=terminal_metadata)

    persisted = repository.create(message)

    assert events == [
        ("committed", persisted.id),
        ("published", persisted.conversation_id),
    ]


def test_user_commit_does_not_publish_compaction_notification() -> None:
    events: list[tuple[str, object]] = []
    repository = MessageRepository(
        session_factory=lambda: None,
        compaction_repository=_CommittedPersistence(events),
        compaction_publisher=lambda conversation_id: events.append(("published", conversation_id)),
    )
    message = _message_data(role=MessageRole.user, content="question")

    persisted = repository.create(message)

    assert events == [("committed", persisted.id)]


def test_publish_failure_keeps_committed_message_and_job() -> None:
    events: list[tuple[str, object]] = []

    def fail_publish(_conversation_id):
        events.append(("publish_failed", "broker_unavailable"))
        raise RuntimeError("broker payload with private details")

    repository = MessageRepository(
        session_factory=lambda: None,
        compaction_repository=_CommittedPersistence(events),
        compaction_publisher=fail_publish,
    )
    message = _message_data(role=MessageRole.assistant)

    persisted = repository.create(message)

    assert persisted.id == message["id"]
    assert events == [
        ("committed", persisted.id),
        ("publish_failed", "broker_unavailable"),
    ]


def test_production_publisher_sends_only_conversation_id(monkeypatch) -> None:
    from app.core.config import settings
    from app.workers import conversation_compaction as worker_module

    conversation_id = uuid4()
    published = []
    monkeypatch.setattr(settings, "conversation_summary_enabled", True)
    monkeypatch.setattr(
        worker_module.compact_conversation_task,
        "delay",
        lambda value: published.append(value),
    )

    worker_module.publish_conversation_compaction(conversation_id)

    assert published == [str(conversation_id)]


@pytest.mark.parametrize(
    ("metadata", "hidden"),
    [
        ({"paused": True}, True),
        ({"interrupt": {"id": "i1"}}, True),
        ({"paused": True}, False),
    ],
)
def test_hidden_placeholders_are_excluded_from_compaction_input(metadata, hidden) -> None:
    from app.repositories.conversation_compaction import ConversationCompactionRepository

    row = SimpleNamespace(
        sender=MessageRole.assistant.value,
        content="visible" if not hidden else "",
        message_metadata=metadata,
    )

    assert ConversationCompactionRepository._is_hidden_artifact(row) is hidden
