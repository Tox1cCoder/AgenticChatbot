"""ProjectService behaviour, including the reverse-direction ownership checks."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import delete

from app.core.config import settings
from app.core.exceptions import AuthorizationException
from app.core.exceptions.project import (
    ProjectConversationNotFoundError,
    ProjectForbiddenError,
    ProjectNotFoundError,
)
from app.database.database import Database
from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.project import Project, ProjectCustomAgent
from app.models.user import User
from app.repositories.custom_agent import CustomAgentRepository
from app.repositories.project import ProjectRepository
from app.schemas.project import ProjectCreate, ProjectUpdate
from app.services.project_service import ProjectService
from app.utils.validation.conversation_validation import ConversationValidationUtils


@pytest.fixture
def service_env():
    db = Database(settings.database_url)
    sf = db.session
    owner_id = uuid4()
    other_id = uuid4()
    agent_id = uuid4()
    owner_conversation = uuid4()
    other_conversation = uuid4()

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
        s.add(
            CustomAgent(
                id=agent_id,
                owner_id=owner_id,
                name="A",
                slug="a",
                prompt="p",
                provider_type="openai",
                model="gpt-4.1-mini",
            )
        )
        s.add(Conversation(id=owner_conversation, owner_id=owner_id, title="mine"))
        s.add(Conversation(id=other_conversation, owner_id=other_id, title="theirs"))
        s.commit()

    service = ProjectService(
        repository=ProjectRepository(session_factory=sf),
        custom_agent_repository=CustomAgentRepository(session_factory=sf),
        conversation_validation_utils=ConversationValidationUtils(session_factory=sf),
    )
    try:
        yield service, owner_id, other_id, agent_id, owner_conversation, other_conversation
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


def test_create_then_read(service_env):
    service, owner_id, *_ = service_env

    created = service.create_project(owner_id, ProjectCreate(name="Roadmap", instructions="Brief."))
    fetched = service.get_project(owner_id, created.id)

    assert fetched.name == "Roadmap"
    assert fetched.instructions == "Brief."
    assert fetched.conversation_count == 0


def test_update_changes_instructions(service_env):
    service, owner_id, *_ = service_env
    created = service.create_project(owner_id, ProjectCreate(name="Roadmap"))

    updated = service.update_project(owner_id, created.id, ProjectUpdate(instructions="New."))

    assert updated.instructions == "New."


def test_missing_project_is_404(service_env):
    service, owner_id, *_ = service_env

    with pytest.raises(ProjectNotFoundError):
        service.get_project(owner_id, uuid4())


def test_another_users_project_is_403(service_env):
    """Reverse direction: unauthorised access must be refused, not merely absent."""
    service, owner_id, other_id, *_ = service_env
    created = service.create_project(owner_id, ProjectCreate(name="Roadmap"))

    with pytest.raises(ProjectForbiddenError):
        service.get_project(other_id, created.id)


def test_deleted_project_is_404_not_403(service_env):
    service, owner_id, *_ = service_env
    created = service.create_project(owner_id, ProjectCreate(name="Roadmap"))
    service.delete_project(owner_id, created.id)

    with pytest.raises(ProjectNotFoundError):
        service.get_project(owner_id, created.id)


def test_cannot_attach_another_users_conversation(service_env):
    """The conversation ownership check raises AuthorizationException (403), not a
    project error: ``ConversationValidationUtils.validate_user_owns_conversation``
    is what rejects it, since the conversation exists but belongs to someone else.
    """
    service, owner_id, _other_id, _agent, _mine, theirs = service_env
    created = service.create_project(owner_id, ProjectCreate(name="Roadmap"))

    with pytest.raises(AuthorizationException):
        service.attach_conversation(owner_id, created.id, theirs)


def test_cannot_attach_to_another_users_project(service_env):
    service, owner_id, other_id, _agent, mine, _theirs = service_env
    created = service.create_project(owner_id, ProjectCreate(name="Roadmap"))

    with pytest.raises(ProjectForbiddenError):
        service.attach_conversation(other_id, created.id, mine)


def test_attach_seeds_agents_and_detach_keeps_them(service_env):
    service, owner_id, _other, agent_id, mine, _theirs = service_env
    created = service.create_project(owner_id, ProjectCreate(name="Roadmap"))
    service.set_agents(owner_id, created.id, [agent_id])

    service.attach_conversation(owner_id, created.id, mine)
    seeded = service.repository.seed_conversation_agents(owner_id, created.id, mine)
    service.detach_conversation(owner_id, created.id, mine)

    assert seeded == 0, "attach already seeded the agent, so a re-seed inserts nothing"
    assert service.repository.conversation_project_id(mine) is None


def test_detach_against_the_wrong_project_is_404(service_env):
    service, owner_id, _other, _agent, mine, _theirs = service_env
    first = service.create_project(owner_id, ProjectCreate(name="First"))
    second = service.create_project(owner_id, ProjectCreate(name="Second"))
    service.attach_conversation(owner_id, first.id, mine)

    with pytest.raises(ProjectConversationNotFoundError):
        service.detach_conversation(owner_id, second.id, mine)


def test_attach_moves_a_conversation_between_projects(service_env):
    service, owner_id, _other, _agent, mine, _theirs = service_env
    first = service.create_project(owner_id, ProjectCreate(name="First"))
    second = service.create_project(owner_id, ProjectCreate(name="Second"))

    service.attach_conversation(owner_id, first.id, mine)
    service.attach_conversation(owner_id, second.id, mine)

    assert service.repository.conversation_project_id(mine) == second.id


def test_set_agents_rejects_an_agent_the_user_does_not_own(service_env):
    service, owner_id, *_ = service_env
    created = service.create_project(owner_id, ProjectCreate(name="Roadmap"))

    with pytest.raises(ProjectForbiddenError):
        service.set_agents(owner_id, created.id, [uuid4()])
