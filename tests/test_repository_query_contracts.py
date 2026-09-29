"""Repository reads that returned the wrong rows, checked against real SQL (SQLite)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.base import Base
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.feedback import Feedback
from app.models.message import Message
from app.models.project import Project
from app.models.user import User
from app.repositories.conversation import ConversationRepository
from app.repositories.user import UserRepository

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


@compiles(JSONB, "sqlite")
def _compile_jsonb_for_sqlite(_type, _compiler, **_kwargs):
    return "JSON"


@pytest.fixture()
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(
        engine,
        tables=[
            User.__table__,
            Project.__table__,
            Conversation.__table__,
            Message.__table__,
            Feedback.__table__,
            Document.__table__,
        ],
    )
    try:
        yield sessionmaker(bind=engine, expire_on_commit=False)
    finally:
        engine.dispose()


def _user(name: str, created_at: datetime, deleted: bool = False) -> User:
    return User(
        id=uuid4(),
        username=name,
        email=f"{name}@example.test",
        password_hash="x",
        created_at=created_at,
        deleted_at=created_at if deleted else None,
    )


def _message(conversation_id: UUID, sequence: int, *, deleted: bool = False) -> Message:
    created = T0 + timedelta(minutes=sequence)
    return Message(
        id=uuid4(),
        conversation_id=conversation_id,
        sender=1,
        content=f"m{sequence}",
        sequence=sequence,
        created_at=created,
        deleted_at=created if deleted else None,
    )


def test_user_get_all_treats_skip_as_a_row_offset(session_factory):
    """``skip`` used to be passed on as a page number: skip=1 returned page 1."""
    with session_factory() as session:
        for offset, name in enumerate(("a", "b", "c")):
            session.add(_user(name, T0 + timedelta(minutes=offset)))
        session.add(_user("gone", T0 - timedelta(minutes=1), deleted=True))
        session.commit()

    repository = UserRepository(session_factory)

    assert [user.username for user in repository.get_all(skip=0, limit=2)] == ["a", "b"]
    assert [user.username for user in repository.get_all(skip=1, limit=2)] == ["b", "c"]
    assert [user.username for user in repository.get_all(skip=3, limit=2)] == []


def test_conversation_list_preview_and_count_skip_deleted_messages(session_factory):
    owner = _user("owner", T0)
    busy, quiet, empty = uuid4(), uuid4(), uuid4()
    with session_factory() as session:
        session.add(owner)
        session.flush()
        for offset, conversation_id in enumerate((busy, quiet, empty)):
            session.add(
                Conversation(
                    id=conversation_id,
                    owner_id=owner.id,
                    title=str(offset),
                    updated_at=T0 + timedelta(hours=offset),
                )
            )
        session.flush()
        for sequence in (1, 2, 3):
            session.add(_message(busy, sequence))
        session.add(_message(busy, 4, deleted=True))
        session.add(_message(quiet, 1))
        session.commit()

    page = ConversationRepository(session_factory=session_factory).get_by_owner_id(
        owner.id, include=["messages"], latest_messages=2
    )
    by_id = {conversation.id: conversation for conversation in page.items}

    assert [message.content for message in by_id[busy].messages] == ["m2", "m3"]
    assert by_id[busy].message_count == 3
    assert [message.content for message in by_id[quiet].messages] == ["m1"]
    assert by_id[quiet].message_count == 1
    assert by_id[empty].messages == []
    assert by_id[empty].message_count == 0
    assert page.meta.total == 3


def test_document_upload_time_is_stamped_at_insert(session_factory):
    """An evaluated default froze every upload at the process's import time."""
    owner = _user("uploader", T0)
    conversation_id = uuid4()
    with session_factory() as session:
        session.add(owner)
        session.flush()
        session.add(Conversation(id=conversation_id, owner_id=owner.id, title="docs"))
        session.flush()
        before = datetime.now(timezone.utc)
        document = Document(
            conversation_id=conversation_id,
            filename="a.txt",
            filename_key="a.txt",
            file_type="text/plain",
            status=1,
        )
        session.add(document)
        session.commit()

    assert document.upload_time >= before
