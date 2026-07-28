"""An async twin must return exactly what its sync counterpart returns.

These run against a live PostgreSQL because the point of the twin is the real
transport; a mocked session would prove nothing about ``run_sync``.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.database.async_session import AsyncSessionLocal
from app.database.session import SessionLocal
from app.models.enums import MessageRole
from app.repositories.message import MessageRepository
from app.schemas.message import MessageCreate

pytestmark = pytest.mark.selector_event_loop


@pytest.fixture
def message_repository():
    return MessageRepository(
        session_factory=SessionLocal,
        async_session_factory=AsyncSessionLocal,
    )


async def test_acount_matches_sync_count(
    message_repository, seeded_message_id, seeded_conversation_id
):
    expected = message_repository.count_by_conversation_id(seeded_conversation_id)
    assert expected >= 1
    assert await message_repository.acount_by_conversation_id(seeded_conversation_id) == expected


async def test_aget_by_id_matches_sync(message_repository, seeded_message_id):
    expected = message_repository.get_by_id(seeded_message_id)
    actual = await message_repository.aget_by_id(seeded_message_id)
    assert actual is not None
    assert actual.id == expected.id
    assert actual.content == expected.content


async def test_aget_by_id_returns_none_for_missing(message_repository, require_async_db):
    assert await message_repository.aget_by_id(uuid4()) is None


async def test_acreate_persists_and_returns_a_detached_row(
    message_repository, seeded_conversation_id
):
    created = await message_repository.acreate(
        MessageCreate(
            conversation_id=seeded_conversation_id,
            role=MessageRole.user,
            content="async twin round-trip",
        )
    )
    # Reading attributes after the session closed proves expire_on_commit=False.
    assert created.content == "async twin round-trip"
    assert created.sequence >= 1
    assert message_repository.get_by_id(created.id) is not None


async def test_acreate_allocates_sequences_monotonically(
    message_repository, seeded_conversation_id
):
    """persist_message's UPDATE...RETURNING must still be transactional."""
    first = await message_repository.acreate(
        MessageCreate(
            conversation_id=seeded_conversation_id,
            role=MessageRole.user,
            content="first",
        )
    )
    second = await message_repository.acreate(
        MessageCreate(
            conversation_id=seeded_conversation_id,
            role=MessageRole.user,
            content="second",
        )
    )
    assert second.sequence == first.sequence + 1


async def test_acreate_rejects_an_unknown_conversation(message_repository, require_async_db):
    with pytest.raises(ValueError, match="conversation_not_found"):
        await message_repository.acreate(
            MessageCreate(
                conversation_id=uuid4(),
                role=MessageRole.user,
                content="orphan",
            )
        )


async def test_aget_latest_by_conversation_matches_sync(
    message_repository, seeded_message_id, seeded_conversation_id
):
    expected = message_repository.get_latest_by_conversation(seeded_conversation_id)
    actual = await message_repository.aget_latest_by_conversation(seeded_conversation_id)
    assert (actual is None) == (expected is None)
    if expected is not None:
        assert actual.id == expected.id


@pytest.fixture
def conversation_repository():
    from app.repositories.conversation import ConversationRepository

    return ConversationRepository(
        session_factory=SessionLocal,
        async_session_factory=AsyncSessionLocal,
    )


@pytest.fixture
def document_repository():
    from app.repositories.document import DocumentRepository

    return DocumentRepository(
        session_factory=SessionLocal,
        async_session_factory=AsyncSessionLocal,
    )


async def test_conversation_aget_by_id_matches_sync(
    conversation_repository, seeded_conversation_id
):
    expected = conversation_repository.get_by_id(seeded_conversation_id)
    actual = await conversation_repository.aget_by_id(seeded_conversation_id)
    assert actual is not None
    assert actual.id == expected.id
    assert actual.title == expected.title


async def test_conversation_aget_by_id_returns_none_for_missing(
    conversation_repository, require_async_db
):
    assert await conversation_repository.aget_by_id(uuid4()) is None


async def test_conversation_aupdate_persists(conversation_repository, seeded_conversation_id):
    from app.schemas.conversation import ConversationUpdate

    updated = await conversation_repository.aupdate(
        seeded_conversation_id, ConversationUpdate(title="renamed by async twin")
    )
    assert updated is not None
    assert updated.title == "renamed by async twin"
    assert conversation_repository.get_by_id(seeded_conversation_id).title == (
        "renamed by async twin"
    )


async def test_conversation_aupdate_returns_none_for_missing(
    conversation_repository, require_async_db
):
    from app.schemas.conversation import ConversationUpdate

    assert await conversation_repository.aupdate(uuid4(), ConversationUpdate(title="x")) is None


async def test_document_acount_matches_sync(document_repository, seeded_conversation_id):
    expected = document_repository.count_by_conversation(seeded_conversation_id)
    assert await document_repository.acount_by_conversation(seeded_conversation_id) == expected


# ── Phase 2: repositories on the rest of the pre-first-token path ───────────


@pytest.fixture
def task_plan_repository():
    from app.repositories.task_plan import TaskPlanRepository

    return TaskPlanRepository(
        session_factory=SessionLocal,
        async_session_factory=AsyncSessionLocal,
    )


@pytest.fixture
def custom_agent_repository():
    from app.repositories.custom_agent import CustomAgentRepository

    return CustomAgentRepository(
        session_factory=SessionLocal,
        async_session_factory=AsyncSessionLocal,
    )


async def test_task_plan_aget_by_conversation_id_matches_sync(
    task_plan_repository, seeded_conversation_id
):
    expected = task_plan_repository.get_by_conversation_id(seeded_conversation_id)
    actual = await task_plan_repository.aget_by_conversation_id(seeded_conversation_id)
    assert [row.id for row in actual] == [row.id for row in expected]


async def test_task_plan_aget_by_conversation_id_honors_include_completed(
    task_plan_repository, seeded_conversation_id
):
    expected = task_plan_repository.get_by_conversation_id(
        seeded_conversation_id, include_completed=False
    )
    actual = await task_plan_repository.aget_by_conversation_id(
        seeded_conversation_id, include_completed=False
    )
    assert [row.id for row in actual] == [row.id for row in expected]


async def test_task_plan_aget_active_or_next_matches_sync(
    task_plan_repository, seeded_conversation_id
):
    expected = task_plan_repository.get_active_or_next_task(seeded_conversation_id)
    actual = await task_plan_repository.aget_active_or_next_task(seeded_conversation_id)
    assert (actual is None) == (expected is None)
    if expected is not None:
        assert actual.id == expected.id


async def test_custom_agent_alist_attachments_matches_sync(
    custom_agent_repository, seeded_conversation_id
):
    conversation_owner = uuid4()
    expected = custom_agent_repository.list_attachments(conversation_owner, seeded_conversation_id)
    actual = await custom_agent_repository.alist_attachments(
        conversation_owner, seeded_conversation_id
    )
    assert [row[0].id for row in actual] == [row[0].id for row in expected]


async def test_aget_by_conversation_id_matches_sync(
    message_repository, seeded_message_id, seeded_conversation_id
):
    # order_by is passed explicitly: the shared strategy does
    # hasattr(Message, order_by) with no None guard, so the parameter's own
    # default raises TypeError on the sync path too (pre-existing).
    expected = message_repository.get_by_conversation_id(
        seeded_conversation_id, page=1, limit=10, order_by="created_at"
    )
    actual = await message_repository.aget_by_conversation_id(
        seeded_conversation_id, page=1, limit=10, order_by="created_at"
    )
    assert [row.id for row in actual.items] == [row.id for row in expected.items]
    assert len(actual.items) >= 1
    assert actual.meta.total == expected.meta.total
