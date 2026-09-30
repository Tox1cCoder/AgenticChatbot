"""Deleting a project deletes the memories saved in it.

A soft delete never fires the ``ON DELETE SET NULL`` on ``user_memories.project_id``,
and the project's conversations are detached, so without an explicit write the
project's memories would stay live but unreachable from any conversation.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.base import Base
from app.models.conversation import Conversation
from app.models.project import Project
from app.models.user import User
from app.models.user_memory import UserMemory
from app.repositories.project import ProjectRepository
from app.repositories.user_memory import UserMemoryRepository


@pytest.fixture
def memory_env():
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
            UserMemory.__table__,
        ],
    )
    sf = sessionmaker(bind=engine, expire_on_commit=False)
    owner_id = uuid4()
    with sf() as s:
        s.add(
            User(
                id=owner_id,
                username=f"u_{owner_id.hex[:12]}",
                email=f"{owner_id.hex[:12]}@test.local",
                password_hash="x",
            )
        )
        s.commit()
    try:
        yield ProjectRepository(session_factory=sf), UserMemoryRepository(sf), sf, owner_id
    finally:
        engine.dispose()


def _remember(memories: UserMemoryRepository, owner_id, content: str, project_id=None):
    return memories.create(user_id=owner_id, content=content, source="t", project_id=project_id)


def _recalled(memories: UserMemoryRepository, owner_id, project_id=None) -> set:
    return {m.id for m in memories.list_for_user(owner_id, project_id=project_id)}


def test_soft_delete_deletes_the_projects_memories_only(memory_env):
    projects, memories, sf, owner_id = memory_env
    doomed = projects.create(owner_id, {"name": "Doomed"})
    kept = projects.create(owner_id, {"name": "Kept"})
    doomed_ids = [_remember(memories, owner_id, text, doomed.id).id for text in ("one", "two")]
    kept_memory = _remember(memories, owner_id, "kept", kept.id)
    global_memory = _remember(memories, owner_id, "global")

    assert projects.soft_delete_and_detach(owner_id, doomed.id) is True

    with sf() as s:
        for memory_id in doomed_ids:
            assert s.get(UserMemory, memory_id).deleted_at is not None
        assert s.get(UserMemory, kept_memory.id).deleted_at is None
        assert s.get(UserMemory, global_memory.id).deleted_at is None

    assert _recalled(memories, owner_id, doomed.id) == {global_memory.id}
    assert _recalled(memories, owner_id, kept.id) == {kept_memory.id, global_memory.id}
    assert _recalled(memories, owner_id) == {global_memory.id}


def test_soft_delete_by_another_owner_touches_no_memories(memory_env):
    projects, memories, sf, owner_id = memory_env
    project = projects.create(owner_id, {"name": "Mine"})
    memory = _remember(memories, owner_id, "m", project.id)

    assert projects.soft_delete_and_detach(uuid4(), project.id) is False

    with sf() as s:
        assert s.get(UserMemory, memory.id).deleted_at is None
