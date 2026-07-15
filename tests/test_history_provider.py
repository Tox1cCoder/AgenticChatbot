"""Memory refactor Task 2/3 guards.

Verifies the canonical prompt-history queries (Task 2) and the
``ConversationHistoryProvider`` (Task 3). Tests use fake sessions and
stub repositories — no live database required.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from app.models.enums import MessageRole
from app.models.message import Message
from app.repositories.message import MessageCRUDStrategy, MessageRepository

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_message(
    *,
    conversation_id: UUID,
    sender: int,
    content: str,
    created_at: datetime,
    deleted: bool = False,
    metadata: dict | None = None,
) -> Message:
    msg = Message(
        id=uuid4(),
        conversation_id=conversation_id,
        sender=sender,
        content=content,
        created_at=created_at,
        sequence=max(1, int(created_at.timestamp())),
    )
    if deleted:
        msg.deleted_at = created_at + timedelta(seconds=1)
    msg.message_metadata = metadata or {}
    return msg


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)

    def scalar_one(self):
        return self._rows[0]

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._rows[0] if self._rows else 0


class _CaptureSession:
    """Fake session whose ``execute`` returns pre-staged rows for each call."""

    def __init__(self, programmed_results: list):
        self._programmed = list(programmed_results)
        self.statements: list = []

    def execute(self, stmt):
        self.statements.append(stmt)
        if not self._programmed:
            return _FakeResult([])
        next_rows = self._programmed.pop(0)
        return _FakeResult(next_rows)


def _factory_for(session):
    @contextmanager
    def factory():
        yield session

    return factory


# ----------------------------------------------------------------------
# Task 2: prompt-history queries
# ----------------------------------------------------------------------


def test_get_prompt_history_excludes_current_deleted_and_empty_paused():
    """Soft-deleted messages and empty paused assistant placeholders must
    not appear in returned history. The current user message is excluded
    by ``before_message_id``."""
    conversation_id = uuid4()

    base = datetime(2026, 4, 29, 10, 0, tzinfo=timezone.utc)
    old_user = _make_message(
        conversation_id=conversation_id,
        sender=MessageRole.user.value,
        content="hi there",
        created_at=base,
    )
    old_assistant = _make_message(
        conversation_id=conversation_id,
        sender=MessageRole.assistant.value,
        content="hello back",
        created_at=base + timedelta(seconds=10),
    )
    _make_message(
        conversation_id=conversation_id,
        sender=MessageRole.user.value,
        content="this should not appear",
        created_at=base + timedelta(seconds=20),
        deleted=True,
    )
    paused_empty = _make_message(
        conversation_id=conversation_id,
        sender=MessageRole.assistant.value,
        content="",
        created_at=base + timedelta(seconds=30),
        metadata={"paused": True},
    )
    interrupt_empty = _make_message(
        conversation_id=conversation_id,
        sender=MessageRole.assistant.value,
        content="",
        created_at=base + timedelta(seconds=40),
        metadata={"interrupt": {"id": "x"}},
    )
    current_message = _make_message(
        conversation_id=conversation_id,
        sender=MessageRole.user.value,
        content="current question",
        created_at=base + timedelta(seconds=50),
    )

    # Programmed sessions:
    # 1) lookup of the cursor message (returns current_message)
    # 2) main query: SQL now orders DESC + LIMIT to grab the most recent rows
    #    in the window, then the strategy reverses to ASC. The fake session
    #    ignores ``.order_by``, so we hand back rows in the DESC order the
    #    real DB would produce. ``deleted_user`` is excluded by the SQL
    #    ``deleted_at IS NULL`` filter; the strategy filters paused/interrupt
    #    empties in Python.
    session = _CaptureSession(
        [
            [current_message],
            [interrupt_empty, paused_empty, old_assistant, old_user],
        ]
    )
    repo = MessageRepository(session_factory=_factory_for(session))

    rows = repo.get_prompt_history(
        conversation_id=conversation_id,
        before_message_id=current_message.id,
        after_message_id=None,
        limit=20,
    )

    assert [row.id for row in rows] == [old_user.id, old_assistant.id]


def test_get_prompt_history_keeps_partial_assistant_with_content():
    """Partial assistant rows with non-empty content remain in history; only
    empty paused/interrupt placeholders are excluded."""
    conversation_id = uuid4()
    base = datetime(2026, 4, 29, 10, 0, tzinfo=timezone.utc)

    partial_with_content = _make_message(
        conversation_id=conversation_id,
        sender=MessageRole.assistant.value,
        content="I was about to say...",
        created_at=base,
        metadata={"paused": True},
    )
    later_user = _make_message(
        conversation_id=conversation_id,
        sender=MessageRole.user.value,
        content="follow up",
        created_at=base + timedelta(seconds=10),
    )

    # SQL orders DESC + LIMIT, then the strategy reverses to ASC. The fake
    # session does no ordering, so program rows in the DESC order the real
    # DB would return.
    session = _CaptureSession([[later_user, partial_with_content]])
    repo = MessageRepository(session_factory=_factory_for(session))

    rows = repo.get_prompt_history(
        conversation_id=conversation_id,
        before_message_id=None,
        after_message_id=None,
        limit=20,
    )

    assert [row.id for row in rows] == [partial_with_content.id, later_user.id]


def test_get_prompt_history_uses_after_cursor_when_provided():
    """When ``after_message_id`` is supplied, only messages strictly after
    its (created_at, id) are returned."""
    conversation_id = uuid4()
    base = datetime(2026, 4, 29, 10, 0, tzinfo=timezone.utc)
    cursor_msg = _make_message(
        conversation_id=conversation_id,
        sender=MessageRole.assistant.value,
        content="cursor anchor",
        created_at=base,
    )
    after = _make_message(
        conversation_id=conversation_id,
        sender=MessageRole.user.value,
        content="post-cursor",
        created_at=base + timedelta(seconds=5),
    )

    # First lookup returns cursor; main query (with cursor applied) returns the
    # one post-cursor row.
    session = _CaptureSession([[cursor_msg], [after]])
    repo = MessageRepository(session_factory=_factory_for(session))

    rows = repo.get_prompt_history(
        conversation_id=conversation_id,
        before_message_id=None,
        after_message_id=cursor_msg.id,
        limit=20,
    )

    assert [row.id for row in rows] == [after.id]


def test_count_by_conversation_id_uses_sql_count():
    """``count_by_conversation_id`` must not materialize the result set."""
    strategy = MessageCRUDStrategy(Message)

    captured: list = []

    def execute(stmt):
        captured.append(str(stmt.compile(compile_kwargs={"literal_binds": False})))
        return _FakeResult([7])

    fake_session = SimpleNamespace(execute=execute)

    total = strategy.count_by_conversation_id(fake_session, uuid4())

    assert total == 7
    assert "count" in captured[0].lower(), f"Expected SQL COUNT, got: {captured[0]}"


# ----------------------------------------------------------------------
# Task 3: ConversationHistoryProvider
# ----------------------------------------------------------------------


def _fake_settings() -> SimpleNamespace:
    return SimpleNamespace(
        memory_cache_max_conversations=64,
        memory_cache_ttl_seconds=60,
        chat_history_max_messages=20,
        rag_history_max_messages=15,
        search_history_max_messages=15,
        planning_history_max_messages=15,
        chat_history_max_tokens=8000,
        rag_history_max_tokens=8000,
        search_history_max_tokens=8000,
        planning_history_max_tokens=8000,
    )


def test_history_provider_returns_summary_plus_recent_without_overlap():
    """History provider returns the durable summary plus only messages that
    came after the summary cursor — no overlap with summarized content."""
    from app.ai.history import ConversationHistoryProvider

    conversation_id = uuid4()
    user_id = uuid4()
    after_user_id = uuid4()
    after_assistant_id = uuid4()
    current_message_id = uuid4()

    base = datetime(2026, 4, 29, 10, 0, tzinfo=timezone.utc)
    after_user = _make_message(
        conversation_id=conversation_id,
        sender=MessageRole.user.value,
        content="post-summary user",
        created_at=base,
    )
    after_user.id = after_user_id
    after_assistant = _make_message(
        conversation_id=conversation_id,
        sender=MessageRole.assistant.value,
        content="post-summary assistant",
        created_at=base + timedelta(seconds=10),
    )
    after_assistant.id = after_assistant_id

    summary = SimpleNamespace(
        summary_payload={
            "facts": ["Earlier billing discussion"],
            "decisions": [],
            "constraints": [],
            "preferences": [],
            "open_questions": [],
            "tool_outcomes": [],
        },
        last_summarized_sequence=4,
        summary_version=2,
        is_valid=True,
    )

    message_repo = MagicMock()
    message_repo.get_prompt_history.return_value = [after_user, after_assistant]

    summary_repo = MagicMock()
    summary_repo.get_owned_valid_memory.return_value = summary

    provider = ConversationHistoryProvider(
        message_repository=message_repo,
        summary_repository=summary_repo,
        settings=_fake_settings(),
    )

    context = asyncio.run(
        provider.build_context(
            conversation_id=conversation_id,
            user_id=user_id,
            current_message_id=current_message_id,
            agent_key="chat",
        )
    )

    assert context.memory is not None
    assert context.memory.role.value == "memory"
    assert [m.metadata["message_id"] for m in context.messages[1:]] == [
        str(after_user_id),
        str(after_assistant_id),
    ]
    message_repo.get_prompt_history.assert_called_once()
    kwargs = message_repo.get_prompt_history.call_args.kwargs
    assert kwargs["before_message_id"] == current_message_id
    assert kwargs["after_sequence"] == 4


def test_history_provider_trims_by_agent_budget():
    """The number of returned messages must not exceed the agent's budget."""
    from app.ai.history import ConversationHistoryProvider

    conversation_id = uuid4()
    user_id = uuid4()

    base = datetime(2026, 4, 29, 10, 0, tzinfo=timezone.utc)
    rows = [
        _make_message(
            conversation_id=conversation_id,
            sender=MessageRole.user.value if i % 2 == 0 else MessageRole.assistant.value,
            content=f"msg {i}",
            created_at=base + timedelta(seconds=i),
        )
        for i in range(40)
    ]

    message_repo = MagicMock()
    message_repo.get_prompt_history.return_value = rows

    summary_repo = MagicMock()
    summary_repo.get_owned_valid_memory.return_value = None

    settings = _fake_settings()
    provider = ConversationHistoryProvider(
        message_repository=message_repo,
        summary_repository=summary_repo,
        settings=settings,
    )

    context = asyncio.run(
        provider.build_context(
            conversation_id=conversation_id,
            user_id=user_id,
            current_message_id=uuid4(),
            agent_key="rag",
        )
    )

    assert len(context.messages) <= settings.rag_history_max_messages
    assert context.budget.agent_key == "rag"
    assert context.memory is None


def test_history_provider_invalidate_clears_cached_entries():
    """``invalidate`` must drop cached contexts for the conversation."""
    from app.ai.history import ConversationHistoryProvider

    conversation_id = uuid4()
    user_id = uuid4()

    base = datetime(2026, 4, 29, 10, 0, tzinfo=timezone.utc)
    msg = _make_message(
        conversation_id=conversation_id,
        sender=MessageRole.user.value,
        content="hello",
        created_at=base,
    )

    message_repo = MagicMock()
    message_repo.get_prompt_history.return_value = [msg]

    summary_repo = MagicMock()
    summary_repo.get_owned_valid_memory.return_value = None

    provider = ConversationHistoryProvider(
        message_repository=message_repo,
        summary_repository=summary_repo,
        settings=_fake_settings(),
    )

    asyncio.run(
        provider.build_context(
            conversation_id=conversation_id,
            user_id=user_id,
            current_message_id=None,
            agent_key="chat",
        )
    )
    asyncio.run(
        provider.build_context(
            conversation_id=conversation_id,
            user_id=user_id,
            current_message_id=None,
            agent_key="chat",
        )
    )
    # Second call should be cached — repo only hit once.
    assert message_repo.get_prompt_history.call_count == 1

    provider.invalidate(conversation_id)

    asyncio.run(
        provider.build_context(
            conversation_id=conversation_id,
            user_id=user_id,
            current_message_id=None,
            agent_key="chat",
        )
    )
    assert message_repo.get_prompt_history.call_count == 2
