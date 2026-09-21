"""Behaviour of ProjectRepository against PostgreSQL."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from app.core.config import settings
from app.database.database import Database
from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.project import Project, ProjectCustomAgent
from app.models.user import User
from app.repositories.project import ProjectRepository


@pytest.fixture
def repo_env():
    db = Database(settings.database_url)
    sf = db.session
    owner_id = uuid4()
    other_id = uuid4()
    agent_a = uuid4()
    agent_b = uuid4()

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
        for aid, name in ((agent_a, "A"), (agent_b, "B")):
            s.add(
                CustomAgent(
                    id=aid,
                    owner_id=owner_id,
                    name=name,
                    slug=name.lower(),
                    prompt="p",
                    provider_type="openai",
                    model="gpt-4.1-mini",
                )
            )
        s.commit()

    repository = ProjectRepository(session_factory=sf)
    try:
        yield repository, sf, owner_id, other_id, agent_a, agent_b
    finally:
        with sf() as s:
            s.execute(delete(ConversationCustomAgent))
            s.execute(delete(ProjectCustomAgent))
            for uid in (owner_id, other_id):
                s.execute(delete(Conversation).where(Conversation.owner_id == uid))
                s.execute(delete(Project).where(Project.owner_id == uid))
                s.execute(delete(CustomAgent).where(CustomAgent.owner_id == uid))
                s.execute(delete(User).where(User.id == uid))
            s.commit()


def test_create_and_list_by_owner(repo_env):
    repository, _sf, owner_id, other_id, _a, _b = repo_env

    created = repository.create(owner_id, {"name": "Roadmap", "instructions": "Be brief."})

    assert created.name == "Roadmap"
    assert [p.id for p in repository.list_by_owner(owner_id)] == [created.id]
    assert repository.list_by_owner(other_id) == []


def test_get_owned_returns_none_for_another_owner(repo_env):
    repository, _sf, owner_id, other_id, _a, _b = repo_env
    project = repository.create(owner_id, {"name": "Roadmap"})

    assert repository.get_owned(owner_id, project.id) is not None
    assert repository.get_owned(other_id, project.id) is None


def test_get_live_is_ownership_agnostic_but_hides_deleted(repo_env):
    repository, _sf, owner_id, _other, _a, _b = repo_env
    project = repository.create(owner_id, {"name": "Roadmap"})

    assert repository.get_live(project.id) is not None

    repository.soft_delete_and_detach(owner_id, project.id)

    assert repository.get_live(project.id) is None


def test_soft_delete_detaches_conversations_without_deleting_them(repo_env):
    repository, sf, owner_id, _other, _a, _b = repo_env
    project = repository.create(owner_id, {"name": "Roadmap"})
    conversation_id = uuid4()
    with sf() as s:
        s.add(
            Conversation(
                id=conversation_id, owner_id=owner_id, title="t", project_id=project.id
            )
        )
        s.commit()

    assert repository.soft_delete_and_detach(owner_id, project.id) is True

    with sf() as s:
        conversation = s.get(Conversation, conversation_id)
        assert conversation is not None, "the conversation must survive the project delete"
        assert conversation.project_id is None


def test_replace_agents_sets_order(repo_env):
    repository, _sf, owner_id, _other, agent_a, agent_b = repo_env
    project = repository.create(owner_id, {"name": "Roadmap"})

    repository.replace_agents(owner_id, project.id, [agent_b, agent_a])

    assert [a.id for a in repository.list_agents(owner_id, project.id)] == [agent_b, agent_a]


def test_seed_inserts_project_agents_onto_a_conversation(repo_env):
    repository, sf, owner_id, _other, agent_a, agent_b = repo_env
    project = repository.create(owner_id, {"name": "Roadmap"})
    repository.replace_agents(owner_id, project.id, [agent_a, agent_b])
    conversation_id = uuid4()
    with sf() as s:
        s.add(Conversation(id=conversation_id, owner_id=owner_id, title="t"))
        s.commit()

    inserted = repository.seed_conversation_agents(owner_id, project.id, conversation_id)

    assert inserted == 2
    with sf() as s:
        rows = (
            s.execute(
                select(ConversationCustomAgent.custom_agent_id)
                .where(ConversationCustomAgent.conversation_id == conversation_id)
                .order_by(ConversationCustomAgent.agent_order.asc())
            )
            .scalars()
            .all()
        )
    assert rows == [agent_a, agent_b]


def test_seed_is_idempotent_and_never_removes(repo_env):
    repository, sf, owner_id, _other, agent_a, agent_b = repo_env
    project = repository.create(owner_id, {"name": "Roadmap"})
    repository.replace_agents(owner_id, project.id, [agent_a])
    conversation_id = uuid4()
    with sf() as s:
        s.add(Conversation(id=conversation_id, owner_id=owner_id, title="t"))
        s.add(
            ConversationCustomAgent(
                owner_id=owner_id,
                conversation_id=conversation_id,
                custom_agent_id=agent_b,
                agent_order=0,
            )
        )
        s.commit()

    first = repository.seed_conversation_agents(owner_id, project.id, conversation_id)
    second = repository.seed_conversation_agents(owner_id, project.id, conversation_id)

    assert (first, second) == (1, 0)
    with sf() as s:
        rows = (
            s.execute(
                select(ConversationCustomAgent.custom_agent_id)
                .where(ConversationCustomAgent.conversation_id == conversation_id)
                .order_by(ConversationCustomAgent.agent_order.asc())
            )
            .scalars()
            .all()
        )
    assert rows == [agent_b, agent_a], "the pre-existing attachment is kept and stays first"


def test_conversation_counts_excludes_soft_deleted(repo_env):
    repository, sf, owner_id, _other, _a, _b = repo_env
    project = repository.create(owner_id, {"name": "Roadmap"})
    with sf() as s:
        s.add(Conversation(owner_id=owner_id, title="live", project_id=project.id))
        s.add(
            Conversation(
                owner_id=owner_id,
                title="gone",
                project_id=project.id,
                deleted_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
            )
        )
        s.commit()

    assert repository.conversation_counts(owner_id) == {project.id: 1}
