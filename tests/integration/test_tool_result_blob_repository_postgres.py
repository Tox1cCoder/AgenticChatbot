"""PostgreSQL integration test for ToolResultBlobRepository scoping.

get_for_user_and_conversation exists to guarantee one thing: a blob is only
readable when both user_id AND conversation_id match. No test in the repo
previously executed the real WHERE clause -- test_tool_result_read_tool.py
only ever talks to a FakeRepository, so a dropped or swapped predicate would
have passed the whole suite. This file runs the production method against a
live database instead, mirroring the session_factory/create_all pattern in
test_conversation_compaction_postgres.py and test_model_usage_repository_postgres.py.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker

from app.models.base import Base
from app.models.conversation import Conversation
from app.models.tool_result_blob import ToolResultBlob
from app.models.user import User
from app.repositories.tool_result_blob import ToolResultBlobRepository

ScopedBlob = tuple[ToolResultBlobRepository, UUID, UUID, UUID, UUID, UUID]


@pytest.fixture(scope="module")
def session_factory():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL integration tests")
    engine = create_engine(database_url)
    Base.metadata.create_all(
        engine,
        tables=[User.__table__, Conversation.__table__, ToolResultBlob.__table__],
    )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        engine.dispose()


@pytest.fixture()
def scoped_blob(session_factory) -> Iterator[ScopedBlob]:
    """Seed one blob plus a second, unrelated user/conversation pair.

    Two independent identities are required: a test that only ever sees one
    user or one conversation cannot tell an ANDed predicate from a dropped
    one, since both would happen to return the same result.
    """
    owner_id = uuid4()
    other_user_id = uuid4()
    conversation_id = uuid4()
    other_conversation_id = uuid4()
    blob_id = uuid4()
    with session_factory.begin() as session:
        session.add(
            User(
                id=owner_id,
                username=f"owner-{owner_id}",
                email=f"{owner_id}@example.test",
                password_hash="test",
            )
        )
        session.add(
            User(
                id=other_user_id,
                username=f"other-{other_user_id}",
                email=f"{other_user_id}@example.test",
                password_hash="test",
            )
        )
        session.add(Conversation(id=conversation_id, owner_id=owner_id, title="blob test"))
        session.add(
            Conversation(id=other_conversation_id, owner_id=other_user_id, title="unrelated")
        )
        session.add(
            ToolResultBlob(
                id=blob_id,
                conversation_id=conversation_id,
                user_id=owner_id,
                tool_name="tavily_search",
                content="full result text",
                sha256="a" * 64,
                size_bytes=17,
            )
        )
    repository = ToolResultBlobRepository(session_factory)
    yield (
        repository,
        blob_id,
        owner_id,
        conversation_id,
        other_user_id,
        other_conversation_id,
    )
    with session_factory.begin() as session:
        session.execute(delete(ToolResultBlob).where(ToolResultBlob.id == blob_id))
        session.execute(
            delete(Conversation).where(
                Conversation.id.in_([conversation_id, other_conversation_id])
            )
        )
        session.execute(delete(User).where(User.id.in_([owner_id, other_user_id])))


def test_matching_user_and_conversation_returns_the_blob(scoped_blob: ScopedBlob) -> None:
    repository, blob_id, owner_id, conversation_id, _, _ = scoped_blob

    record = repository.get_for_user_and_conversation(blob_id, owner_id, conversation_id)

    assert record is not None
    assert record.id == blob_id


def test_wrong_conversation_id_is_not_found(scoped_blob: ScopedBlob) -> None:
    """Fails if the conversation_id predicate is dropped or swapped for another column."""
    repository, blob_id, owner_id, _, _, other_conversation_id = scoped_blob

    record = repository.get_for_user_and_conversation(blob_id, owner_id, other_conversation_id)

    assert record is None


def test_wrong_user_id_is_not_found(scoped_blob: ScopedBlob) -> None:
    """Fails if the user_id predicate is dropped or swapped for another column."""
    repository, blob_id, _, conversation_id, other_user_id, _ = scoped_blob

    record = repository.get_for_user_and_conversation(blob_id, other_user_id, conversation_id)

    assert record is None
