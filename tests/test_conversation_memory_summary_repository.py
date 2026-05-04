"""Memory refactor Task 1 guards: ConversationMemorySummaryRepository contract.

Verifies the repository upserts a single summary row per conversation, returns
the persisted row in detached form, and bumps ``summary_version`` on update.
Uses fake sessions so we do not require a live database.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models.conversation_memory_summary import ConversationMemorySummary
from app.models.message import Message
from app.repositories.conversation_memory_summary import (
    ConversationMemorySummaryRepository,
)


class _Result:
    def __init__(self, rows):
        self._rows = rows if isinstance(rows, list) else [rows] if rows is not None else []

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    def __init__(
        self,
        initial_row: ConversationMemorySummary | None = None,
        messages: list[SimpleNamespace] | None = None,
    ):
        self.added: list = []
        self.committed = 0
        self.refreshed: list = []
        self.expunged: list = []
        self._row = initial_row
        self._messages = list(messages or [])

    def execute(self, stmt):
        try:
            entity = stmt.column_descriptions[0].get("entity")
        except Exception:
            entity = None
        if entity is Message:
            return _Result(self._messages)
        return _Result(self._row)

    def add(self, obj):
        self.added.append(obj)
        self._row = obj

    def commit(self):
        self.committed += 1

    def refresh(self, obj):
        self.refreshed.append(obj)

    def expunge(self, obj):
        self.expunged.append(obj)


def _factory_for(session: _FakeSession):
    @contextmanager
    def factory():
        yield session

    return factory


def test_upsert_creates_summary_for_conversation():
    session = _FakeSession()
    repo = ConversationMemorySummaryRepository(session_factory=_factory_for(session))

    conversation_id = uuid4()
    user_id = uuid4()
    message_id = uuid4()

    saved = repo.upsert(
        conversation_id=conversation_id,
        user_id=user_id,
        summary_text="- User asked about invoices",
        last_summarized_message_id=message_id,
        source_message_count=2,
        estimated_tokens=16,
    )

    assert isinstance(saved, ConversationMemorySummary)
    assert saved.conversation_id == conversation_id
    assert saved.user_id == user_id
    assert saved.last_summarized_message_id == message_id
    assert saved.summary_text == "- User asked about invoices"
    assert saved.source_message_count == 2
    assert saved.estimated_tokens == 16
    assert saved.summary_version == 1
    assert session.added == [saved]
    assert session.committed == 1
    assert saved in session.expunged


def test_upsert_updates_existing_summary_cursor():
    conversation_id = uuid4()
    user_id = uuid4()
    earlier_message_id = uuid4()
    later_message_id = uuid4()
    base = datetime(2026, 4, 29, 10, 0, tzinfo=timezone.utc)

    existing = ConversationMemorySummary(
        conversation_id=conversation_id,
        user_id=user_id,
        summary_text="- Old summary",
        last_summarized_message_id=earlier_message_id,
        source_message_count=2,
        estimated_tokens=8,
        summary_version=3,
    )
    session = _FakeSession(
        initial_row=existing,
        messages=[
            SimpleNamespace(id=earlier_message_id, created_at=base),
            SimpleNamespace(id=later_message_id, created_at=base + timedelta(seconds=1)),
        ],
    )
    repo = ConversationMemorySummaryRepository(session_factory=_factory_for(session))

    saved = repo.upsert(
        conversation_id=conversation_id,
        user_id=user_id,
        summary_text="- Updated summary",
        last_summarized_message_id=later_message_id,
        source_message_count=6,
        estimated_tokens=20,
    )

    assert saved is existing
    assert saved.summary_text == "- Updated summary"
    assert saved.last_summarized_message_id == later_message_id
    assert saved.source_message_count == 6
    assert saved.estimated_tokens == 20
    assert saved.summary_version == 4, "summary_version must monotonically increase on update"
    assert session.added == [], "existing row must not be re-added"
    assert session.committed == 1


def test_upsert_skips_incoming_older_cursor_with_same_source_count():
    conversation_id = uuid4()
    user_id = uuid4()
    older_message_id = uuid4()
    newer_message_id = uuid4()
    base = datetime(2026, 4, 29, 10, 0, tzinfo=timezone.utc)

    existing = ConversationMemorySummary(
        conversation_id=conversation_id,
        user_id=user_id,
        summary_text="- Newer summary",
        last_summarized_message_id=newer_message_id,
        source_message_count=10,
        estimated_tokens=20,
        summary_version=4,
    )
    session = _FakeSession(
        initial_row=existing,
        messages=[
            SimpleNamespace(id=older_message_id, created_at=base),
            SimpleNamespace(id=newer_message_id, created_at=base + timedelta(seconds=1)),
        ],
    )
    repo = ConversationMemorySummaryRepository(session_factory=_factory_for(session))

    saved = repo.upsert(
        conversation_id=conversation_id,
        user_id=user_id,
        summary_text="- Older racing summary",
        last_summarized_message_id=older_message_id,
        source_message_count=10,
        estimated_tokens=12,
    )

    assert saved is existing
    assert saved.summary_text == "- Newer summary"
    assert saved.last_summarized_message_id == newer_message_id
    assert saved.summary_version == 4
    assert session.committed == 0


def test_upsert_accepts_later_cursor_even_when_source_count_is_lower():
    conversation_id = uuid4()
    user_id = uuid4()
    older_message_id = uuid4()
    newer_message_id = uuid4()
    base = datetime(2026, 4, 29, 10, 0, tzinfo=timezone.utc)

    existing = ConversationMemorySummary(
        conversation_id=conversation_id,
        user_id=user_id,
        summary_text="- Earlier summary",
        last_summarized_message_id=older_message_id,
        source_message_count=60,
        estimated_tokens=20,
        summary_version=4,
    )
    session = _FakeSession(
        initial_row=existing,
        messages=[
            SimpleNamespace(id=older_message_id, created_at=base),
            SimpleNamespace(id=newer_message_id, created_at=base + timedelta(seconds=1)),
        ],
    )
    repo = ConversationMemorySummaryRepository(session_factory=_factory_for(session))

    saved = repo.upsert(
        conversation_id=conversation_id,
        user_id=user_id,
        summary_text="- Later token-triggered summary",
        last_summarized_message_id=newer_message_id,
        source_message_count=8,
        estimated_tokens=30,
    )

    assert saved is existing
    assert saved.summary_text == "- Later token-triggered summary"
    assert saved.last_summarized_message_id == newer_message_id
    assert saved.source_message_count == 8
    assert saved.summary_version == 5
    assert session.committed == 1


def test_get_by_conversation_id_returns_existing_row():
    conversation_id = uuid4()
    existing = ConversationMemorySummary(
        conversation_id=conversation_id,
        user_id=uuid4(),
        summary_text="- existing",
        last_summarized_message_id=uuid4(),
        source_message_count=4,
        estimated_tokens=12,
        summary_version=1,
    )
    session = _FakeSession(initial_row=existing)
    repo = ConversationMemorySummaryRepository(session_factory=_factory_for(session))

    found = repo.get_by_conversation_id(conversation_id)

    assert found is existing
    assert existing in session.expunged


def test_get_by_conversation_id_returns_none_when_missing():
    session = _FakeSession(initial_row=None)
    repo = ConversationMemorySummaryRepository(session_factory=_factory_for(session))

    found = repo.get_by_conversation_id(uuid4())

    assert found is None
    assert session.expunged == []


def test_repository_init_requires_session_factory():
    with pytest.raises(TypeError):
        ConversationMemorySummaryRepository()  # type: ignore[call-arg]
