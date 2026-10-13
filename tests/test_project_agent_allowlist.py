"""A project's agent set is an allowlist for its conversations.

A conversation's effective agents are its attachments that are also in its
project's set. Out-of-set rows are kept, not deleted, so leaving the project
brings them back. An empty project set allows no agents.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from app.core.config import settings
from app.core.exceptions.custom_agent import CustomAgentForbiddenError
from app.database.database import Database
from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.project import Project, ProjectCustomAgent
from app.models.user import User
from app.repositories.custom_agent import CustomAgentRepository
from app.repositories.project import ProjectRepository


@pytest.fixture
def allowlist_env(require_test_database):
    db = Database(settings.database_url)
    sf = db.session
    owner_id = uuid4()
    agent_ids = {name: uuid4() for name in ("A", "B", "C")}
    project_id = uuid4()
    conversation_id = uuid4()
    with sf() as s:
        s.add(
            User(
                id=owner_id,
                username=f"u_{owner_id.hex[:12]}",
                email=f"{owner_id.hex[:12]}@test.local",
                password_hash="x",
            )
        )
        s.flush()
        for name, agent_id in agent_ids.items():
            s.add(
                CustomAgent(
                    id=agent_id,
                    owner_id=owner_id,
                    name=name,
                    slug=name.lower(),
                    prompt="p",
                    provider_type="openai",
                    model="gpt-4.1-mini",
                )
            )
        s.add(Project(id=project_id, owner_id=owner_id, name="P"))
        s.flush()
        s.add(Conversation(id=conversation_id, owner_id=owner_id, title="t"))
        s.flush()
        for order, name in enumerate(("A", "B")):
            s.add(
                ProjectCustomAgent(
                    owner_id=owner_id,
                    project_id=project_id,
                    custom_agent_id=agent_ids[name],
                    agent_order=order,
                )
            )
        for order, name in enumerate(("A", "B", "C")):
            s.add(
                ConversationCustomAgent(
                    owner_id=owner_id,
                    conversation_id=conversation_id,
                    custom_agent_id=agent_ids[name],
                    agent_order=order,
                )
            )
        s.commit()
    try:
        yield sf, owner_id, agent_ids, project_id, conversation_id
    finally:
        with sf() as s:
            s.execute(
                delete(ConversationCustomAgent).where(ConversationCustomAgent.owner_id == owner_id)
            )
            s.execute(delete(ProjectCustomAgent).where(ProjectCustomAgent.owner_id == owner_id))
            s.execute(delete(Conversation).where(Conversation.owner_id == owner_id))
            s.execute(delete(Project).where(Project.owner_id == owner_id))
            s.execute(delete(CustomAgent).where(CustomAgent.owner_id == owner_id))
            s.execute(delete(User).where(User.id == owner_id))
            s.commit()


def _attach(sf, owner_id, project_id, conversation_id) -> None:
    ProjectRepository(session_factory=sf).attach_conversation(owner_id, project_id, conversation_id)


def _effective_names(sf, owner_id, conversation_id) -> list[str]:
    attachments = CustomAgentRepository(session_factory=sf).list_attachments(
        owner_id, conversation_id
    )
    return [agent.name for _attachment, agent in attachments]


def _stored_ids(sf, conversation_id) -> set:
    with sf() as s:
        return set(
            s.execute(
                select(ConversationCustomAgent.custom_agent_id).where(
                    ConversationCustomAgent.conversation_id == conversation_id
                )
            ).scalars()
        )


def test_loose_conversation_keeps_every_attachment(allowlist_env):
    sf, owner_id, _agents, _project_id, conversation_id = allowlist_env

    assert _effective_names(sf, owner_id, conversation_id) == ["A", "B", "C"]


def test_project_conversation_sees_only_allowed_agents(allowlist_env):
    sf, owner_id, _agents, project_id, conversation_id = allowlist_env

    _attach(sf, owner_id, project_id, conversation_id)

    assert _effective_names(sf, owner_id, conversation_id) == ["A", "B"]


def test_narrowing_the_project_hides_the_removed_agent(allowlist_env):
    sf, owner_id, agents, project_id, conversation_id = allowlist_env
    _attach(sf, owner_id, project_id, conversation_id)

    ProjectRepository(session_factory=sf).replace_agents(owner_id, project_id, [agents["A"]])

    assert _effective_names(sf, owner_id, conversation_id) == ["A"]


def test_empty_project_set_allows_no_agents(allowlist_env):
    sf, owner_id, _agents, project_id, conversation_id = allowlist_env
    _attach(sf, owner_id, project_id, conversation_id)

    ProjectRepository(session_factory=sf).replace_agents(owner_id, project_id, [])

    assert _effective_names(sf, owner_id, conversation_id) == []


def test_attaching_an_agent_outside_the_project_is_refused(allowlist_env):
    sf, owner_id, agents, project_id, conversation_id = allowlist_env
    _attach(sf, owner_id, project_id, conversation_id)

    with pytest.raises(CustomAgentForbiddenError):
        CustomAgentRepository(session_factory=sf).replace_attachments(
            owner_id, conversation_id, [agents["A"], agents["C"]]
        )

    assert _stored_ids(sf, conversation_id) == set(agents.values()), "nothing was written"


def test_replacing_inside_a_project_keeps_hidden_rows(allowlist_env):
    sf, owner_id, agents, project_id, conversation_id = allowlist_env
    _attach(sf, owner_id, project_id, conversation_id)

    CustomAgentRepository(session_factory=sf).replace_attachments(
        owner_id, conversation_id, [agents["B"]]
    )

    assert _effective_names(sf, owner_id, conversation_id) == ["B"]
    assert _stored_ids(sf, conversation_id) == {agents["B"], agents["C"]}

    ProjectRepository(session_factory=sf).detach_conversation(
        owner_id, project_id, conversation_id
    )
    assert sorted(_effective_names(sf, owner_id, conversation_id)) == ["B", "C"]


def test_runtime_roster_excludes_agents_outside_the_project(allowlist_env):
    from unittest.mock import MagicMock

    from app.services.custom_agent_service import CustomAgentService

    sf, owner_id, agents, project_id, conversation_id = allowlist_env
    _attach(sf, owner_id, project_id, conversation_id)
    service = CustomAgentService(
        CustomAgentRepository(session_factory=sf), MagicMock(), model_config_service=None
    )

    roster = service.build_runtime_state(owner_id, conversation_id)

    assert {entry["id"] for entry in roster.values()} == {str(agents["A"]), str(agents["B"])}
