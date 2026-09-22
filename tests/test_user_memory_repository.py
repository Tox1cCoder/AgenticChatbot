"""Behaviour of UserMemoryRepository against PostgreSQL.

The project scoping lives in one SQL predicate, so it is tested against a real
database rather than a fake: `project_id = NULL` is never true in SQL, and that
detail is exactly what makes a project-less conversation see only global
memories.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import delete

from app.core.config import settings
from app.database.async_session import AsyncSessionLocal
from app.database.database import Database
from app.models.conversation import Conversation
from app.models.project import Project
from app.models.user import User
from app.models.user_memory import UserMemory
from app.repositories.user_memory import UserMemoryRepository


@pytest.fixture
def repo_env():
    db = Database(settings.database_url)
    sf = db.session
    owner_id = uuid4()
    other_id = uuid4()
    project_a = uuid4()
    project_b = uuid4()
    conv_in_a = uuid4()
    conv_no_project = uuid4()

    with sf() as s:
        for uid in (owner_id, other_id):
            s.add(
                User(
                    id=uid,
                    username=f"u_{uid.hex[:12]}",
                    email=f"{uid.hex[:12]}@test.local",
                    password_hash="x",
                )
            )
        s.flush()
        for pid, name in ((project_a, "A"), (project_b, "B")):
            s.add(Project(id=pid, owner_id=owner_id, name=name))
        s.flush()
        s.add(
            Conversation(id=conv_in_a, owner_id=owner_id, project_id=project_a, title="in A")
        )
        s.add(
            Conversation(
                id=conv_no_project, owner_id=owner_id, project_id=None, title="no project"
            )
        )
        s.commit()

    repository = UserMemoryRepository(
        session_factory=sf,
        async_session_factory=AsyncSessionLocal,
    )
    try:
        yield repository, sf, owner_id, other_id, project_a, project_b, conv_in_a, conv_no_project
    finally:
        with sf() as s:
            s.execute(delete(UserMemory).where(UserMemory.user_id.in_([owner_id, other_id])))
            s.execute(delete(Conversation).where(Conversation.id.in_([conv_in_a, conv_no_project])))
            s.execute(delete(Project).where(Project.id.in_([project_a, project_b])))
            s.execute(delete(User).where(User.id.in_([owner_id, other_id])))
            s.commit()


def _contents(rows):
    return sorted(row.content for row in rows)


class TestProjectScopedRecall:
    def test_a_project_sees_its_own_memories(self, repo_env):
        repo, _sf, owner_id, _other, project_a, _b, _c1, _c2 = repo_env
        repo.create(
            user_id=str(owner_id), content="fact in A", source="test", project_id=str(project_a)
        )

        rows = repo.list_for_user(str(owner_id), project_id=str(project_a))
        assert _contents(rows) == ["fact in A"]

    def test_a_project_never_sees_another_projects_memories(self, repo_env):
        repo, _sf, owner_id, _other, project_a, project_b, _c1, _c2 = repo_env
        repo.create(
            user_id=str(owner_id), content="fact in A", source="test", project_id=str(project_a)
        )

        rows = repo.list_for_user(str(owner_id), project_id=str(project_b))
        assert _contents(rows) == []

    def test_global_memories_are_visible_from_every_project(self, repo_env):
        repo, _sf, owner_id, _other, project_a, project_b, _c1, _c2 = repo_env
        repo.create(user_id=str(owner_id), content="global fact", source="test", project_id=None)
        repo.create(
            user_id=str(owner_id), content="fact in A", source="test", project_id=str(project_a)
        )

        assert _contents(repo.list_for_user(str(owner_id), project_id=str(project_a))) == [
            "fact in A",
            "global fact",
        ]
        assert _contents(repo.list_for_user(str(owner_id), project_id=str(project_b))) == [
            "global fact"
        ]

    def test_no_project_sees_only_global_memories(self, repo_env):
        """`project_id = NULL` is never true in SQL, so a project-less
        conversation must fall through to the IS NULL branch."""
        repo, _sf, owner_id, _other, project_a, _b, _c1, _c2 = repo_env
        repo.create(user_id=str(owner_id), content="global fact", source="test", project_id=None)
        repo.create(
            user_id=str(owner_id), content="fact in A", source="test", project_id=str(project_a)
        )

        rows = repo.list_for_user(str(owner_id), project_id=None)
        assert _contents(rows) == ["global fact"]

    def test_another_users_memories_are_never_returned(self, repo_env):
        repo, _sf, owner_id, other_id, project_a, _b, _c1, _c2 = repo_env
        repo.create(
            user_id=str(other_id), content="not yours", source="test", project_id=str(project_a)
        )

        assert repo.list_for_user(str(owner_id), project_id=str(project_a)) == []

    def test_soft_deleted_memories_are_excluded(self, repo_env):
        repo, _sf, owner_id, _other, project_a, _b, _c1, _c2 = repo_env
        record = repo.create(
            user_id=str(owner_id), content="gone", source="test", project_id=str(project_a)
        )

        assert repo.delete_for_user(str(record.id), str(owner_id)) is True
        assert repo.list_for_user(str(owner_id), project_id=str(project_a)) == []

    def test_delete_refuses_another_users_memory(self, repo_env):
        repo, _sf, owner_id, other_id, project_a, _b, _c1, _c2 = repo_env
        record = repo.create(
            user_id=str(other_id), content="not yours", source="test", project_id=str(project_a)
        )

        assert repo.delete_for_user(str(record.id), str(owner_id)) is False


class TestProjectResolution:
    def test_resolves_the_conversations_project(self, repo_env):
        repo, _sf, owner_id, _other, project_a, _b, conv_in_a, _c2 = repo_env
        assert repo.resolve_project_id(str(owner_id), str(conv_in_a)) == str(project_a)

    def test_a_conversation_without_a_project_resolves_to_global(self, repo_env):
        repo, _sf, owner_id, _other, _a, _b, _c1, conv_no_project = repo_env
        assert repo.resolve_project_id(str(owner_id), str(conv_no_project)) is None

    def test_a_conversation_the_user_does_not_own_resolves_to_global(self, repo_env):
        """Ownership is checked here so a forged conversation id cannot file a
        memory into somebody else's project."""
        repo, _sf, _owner, other_id, _a, _b, conv_in_a, _c2 = repo_env
        assert repo.resolve_project_id(str(other_id), str(conv_in_a)) is None

    def test_no_conversation_resolves_to_global(self, repo_env):
        repo, _sf, owner_id, _other, _a, _b, _c1, _c2 = repo_env
        assert repo.resolve_project_id(str(owner_id), None) is None


# ---------------------------------------------------------------------------
# Async twins
#
# Recall runs on the event loop before the first token, so the async twin is
# the one production actually uses. A twin that drifts from its sync
# counterpart is worse than no twin: the blocking path stays correct while the
# hot path quietly returns something else.
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.selector_event_loop


class TestAsyncTwins:
    async def test_alist_matches_sync_list_for_a_project(self, repo_env):
        repo, _sf, owner_id, _other, project_a, _b, _c1, _c2 = repo_env
        repo.create(user_id=str(owner_id), content="global", source="test", project_id=None)
        repo.create(
            user_id=str(owner_id), content="in A", source="test", project_id=str(project_a)
        )

        expected = repo.list_for_user(str(owner_id), project_id=str(project_a))
        actual = await repo.alist_for_user(str(owner_id), project_id=str(project_a))

        assert _contents(actual) == _contents(expected) == ["global", "in A"]

    async def test_alist_matches_sync_list_outside_any_project(self, repo_env):
        repo, _sf, owner_id, _other, project_a, _b, _c1, _c2 = repo_env
        repo.create(user_id=str(owner_id), content="global", source="test", project_id=None)
        repo.create(
            user_id=str(owner_id), content="in A", source="test", project_id=str(project_a)
        )

        expected = repo.list_for_user(str(owner_id), project_id=None)
        actual = await repo.alist_for_user(str(owner_id), project_id=None)

        assert _contents(actual) == _contents(expected) == ["global"]

    async def test_aresolve_project_id_matches_sync(self, repo_env):
        repo, _sf, owner_id, _other, _a, _b, conv_in_a, conv_no_project = repo_env

        for conversation_id in (str(conv_in_a), str(conv_no_project), None):
            expected = repo.resolve_project_id(str(owner_id), conversation_id)
            assert await repo.aresolve_project_id(str(owner_id), conversation_id) == expected
