from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.base import Base
from app.models.chat_image import ChatImage
from app.models.conversation import Conversation
from app.models.project import Project
from app.models.user import User
from app.repositories.chat_image import ChatImageRepository


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    def __init__(self, rows=None):
        self._rows = rows or []

    def execute(self, _stmt):
        return _FakeQuery(self._rows)


def _factory(session):
    @contextmanager
    def _f():
        yield session

    return _f


@pytest.fixture()
def owned_factory():
    """A real (SQLite) chat_images table, so the partial unique index is enforced."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(
        engine,
        tables=[User.__table__, Project.__table__, Conversation.__table__, ChatImage.__table__],
    )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    user_id, conversation_id = uuid4(), uuid4()
    with factory.begin() as db:
        db.add(User(id=user_id, username="u", email="u@example.test", password_hash="x"))
        db.flush()
        db.add(Conversation(id=conversation_id, owner_id=user_id, title="images"))
    try:
        yield factory, user_id, conversation_id
    finally:
        engine.dispose()


def _row(user_id, conversation_id, **overrides):
    payload = {
        "id": uuid4(),
        "conversation_id": conversation_id,
        "user_id": user_id,
        "sha256": "a" * 64,
        "size_bytes": 10,
        "content_type": "image/png",
        "storage_path": "aa/aaa.png",
    }
    payload.update(overrides)
    return payload


def _live_rows(factory) -> int:
    with factory() as db:
        return db.execute(
            select(func.count()).select_from(ChatImage).where(ChatImage.deleted_at.is_(None))
        ).scalar_one()


def test_create_persists_a_new_row(owned_factory):
    factory, user_id, conversation_id = owned_factory
    data = _row(user_id, conversation_id)

    record = ChatImageRepository(factory).create(data)

    assert record.id == data["id"]
    assert record.created_at is not None
    assert _live_rows(factory) == 1


def test_a_concurrent_duplicate_returns_the_row_that_won(owned_factory):
    """Two workers both miss the read-side check; the second insert must not fail.

    The storage service checks ``get_by_user_and_sha`` first, so reaching
    ``create`` with a duplicate means another worker inserted in between.
    """
    factory, user_id, conversation_id = owned_factory
    repository = ChatImageRepository(factory)
    winner = repository.create(_row(user_id, conversation_id))

    loser = repository.create(_row(user_id, conversation_id, storage_path="aa/other.png"))

    assert loser.id == winner.id
    assert loser.storage_path == "aa/aaa.png"
    assert _live_rows(factory) == 1


def test_a_soft_deleted_row_does_not_block_a_new_one(owned_factory):
    factory, user_id, conversation_id = owned_factory
    repository = ChatImageRepository(factory)
    repository.create(_row(user_id, conversation_id, deleted_at=datetime.now(timezone.utc)))

    fresh = repository.create(_row(user_id, conversation_id))

    assert fresh.deleted_at is None
    assert _live_rows(factory) == 1


def test_get_for_user_returns_row():
    marker = object()
    repo = ChatImageRepository(_factory(_FakeSession(rows=[marker])))
    assert repo.get_for_user(uuid4(), uuid4()) is marker
