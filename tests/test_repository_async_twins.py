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
