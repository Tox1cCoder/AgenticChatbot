"""``order_by=None`` must mean "default ordering", not TypeError.

``DefaultQueryStrategy`` guards with ``if order_by and hasattr(...)``.
``MessageCRUDStrategy`` overrides those methods and dropped the ``order_by and``
half of the guard, so passing the repository's own documented default
(``order_by: str | None = None``) raised
``TypeError: attribute name must be string, not 'NoneType'``.

HTTP callers always supply a string (``MessageOrderBy`` defaults to
``CREATED_AT``), so this was reachable from internal callers rather than the API.
"""

from __future__ import annotations

import pytest

from app.database.async_session import AsyncSessionLocal
from app.database.session import SessionLocal
from app.repositories.message import MessageRepository

pytestmark = pytest.mark.selector_event_loop


@pytest.fixture
def message_repository():
    return MessageRepository(
        session_factory=SessionLocal,
        async_session_factory=AsyncSessionLocal,
    )


def test_get_by_conversation_id_accepts_none_order_by(
    message_repository, seeded_message_id, seeded_conversation_id
):
    page = message_repository.get_by_conversation_id(seeded_conversation_id, page=1, limit=10)
    assert len(page.items) >= 1


def test_none_order_by_matches_explicit_created_at(
    message_repository, seeded_message_id, seeded_conversation_id
):
    default_page = message_repository.get_by_conversation_id(
        seeded_conversation_id, page=1, limit=10
    )
    explicit_page = message_repository.get_by_conversation_id(
        seeded_conversation_id, page=1, limit=10, order_by="created_at"
    )
    assert [row.id for row in default_page.items] == [row.id for row in explicit_page.items]


def test_unknown_order_by_still_falls_back_to_default_ordering(
    message_repository, seeded_message_id, seeded_conversation_id
):
    """The existing else-branch behavior for a bad column name is preserved."""
    page = message_repository.get_by_conversation_id(
        seeded_conversation_id, page=1, limit=10, order_by="not_a_column"
    )
    assert len(page.items) >= 1


async def test_async_twin_also_accepts_none_order_by(
    message_repository, seeded_message_id, seeded_conversation_id
):
    page = await message_repository.aget_by_conversation_id(
        seeded_conversation_id, page=1, limit=10
    )
    assert len(page.items) >= 1


def test_get_by_user_id_accepts_none_order_by(message_repository, require_async_db):
    """Same override, same missing guard, second method."""
    from uuid import uuid4

    page = message_repository.get_by_user_id(uuid4(), page=1, limit=10, order_by=None)
    assert page.items == []
