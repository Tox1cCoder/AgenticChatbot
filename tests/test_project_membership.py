"""Creating and listing conversations inside a project."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from app.core.config import settings
from app.core.exceptions.project import ProjectForbiddenError
from app.database.database import Database
from app.factories.conversation_factory import ConversationFactory
from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.project import Project, ProjectCustomAgent
from app.models.user import User
from app.schemas.conversation import ConversationCreate


def test_factory_carries_project_id_from_schema():
    owner_id = uuid4()
    project_id = uuid4()

    fields = ConversationFactory.create_from_schema(
        ConversationCreate(title="t", project_id=project_id), owner_id
    )

    assert fields["project_id"] == project_id


def test_factory_carries_project_id_from_dict():
    """Both factory paths, because updating only one is how a field goes missing."""
    owner_id = uuid4()
    project_id = uuid4()

    fields = ConversationFactory.create_from_dict(
        {"owner_id": owner_id, "title": "t", "project_id": project_id}
    )

    assert fields["project_id"] == project_id


def test_factory_defaults_project_id_to_none():
    fields = ConversationFactory.create_from_dict({"owner_id": uuid4(), "title": "t"})

    assert fields["project_id"] is None


@pytest.fixture
def membership_env():
    db = Database(settings.database_url)
    sf = db.session
    owner_id = uuid4()
    other_id = uuid4()
    agent_id = uuid4()
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
        s.commit()
    try:
        yield sf, owner_id, other_id, agent_id
    finally:
        with sf() as s:
            s.execute(
                delete(ConversationCustomAgent).where(
                    ConversationCustomAgent.owner_id.in_((owner_id, other_id))
                )
            )
            s.execute(
                delete(ProjectCustomAgent).where(
                    ProjectCustomAgent.owner_id.in_((owner_id, other_id))
                )
            )
            for uid in (owner_id, other_id):
                s.execute(delete(Conversation).where(Conversation.owner_id == uid))
                s.execute(delete(Project).where(Project.owner_id == uid))
                s.execute(delete(CustomAgent).where(CustomAgent.owner_id == uid))
                s.execute(delete(User).where(User.id == uid))
            s.commit()


def _service(sf):
    from app.repositories.conversation import ConversationRepository
    from app.repositories.custom_agent import CustomAgentRepository
    from app.repositories.project import ProjectRepository
    from app.services.conversation_service import ConversationService
    from app.services.project_service import ProjectService
    from app.utils.validation.conversation_validation import ConversationValidationUtils
    from app.utils.validation.user_validation import UserValidationUtils

    conversation_repository = ConversationRepository(session_factory=sf)
    project_repository = ProjectRepository(session_factory=sf)
    # Both take a session factory, not a repository; they build their own internally.
    conversation_validation_utils = ConversationValidationUtils(session_factory=sf)
    user_validation_utils = UserValidationUtils(session_factory=sf)
    project_service = ProjectService(
        repository=project_repository,
        custom_agent_repository=CustomAgentRepository(session_factory=sf),
        conversation_validation_utils=conversation_validation_utils,
    )
    return (
        ConversationService(
            conversation_repository=conversation_repository,
            user_validation_utils=user_validation_utils,
            conversation_validation_utils=conversation_validation_utils,
            project_service=project_service,
        ),
        project_service,
    )


def test_creating_in_a_project_seeds_its_agents(membership_env):
    sf, owner_id, _other_id, agent_id = membership_env
    service, project_service = _service(sf)
    from app.schemas.project import ProjectCreate

    project = project_service.create_project(owner_id, ProjectCreate(name="Roadmap"))
    project_service.set_agents(owner_id, project.id, [agent_id])

    created = service.create_conversation(
        ConversationCreate(title="t", project_id=project.id), owner_id
    )

    assert created.project_id == project.id
    with sf() as s:
        attached = s.execute(
            select(ConversationCustomAgent.custom_agent_id).where(
                ConversationCustomAgent.conversation_id == created.id
            )
        ).scalars().all()
    assert attached == [agent_id]


def test_creating_in_another_users_project_is_refused(membership_env):
    sf, owner_id, other_id, _agent_id = membership_env
    service, project_service = _service(sf)
    from app.schemas.project import ProjectCreate

    project = project_service.create_project(owner_id, ProjectCreate(name="Roadmap"))

    with pytest.raises(ProjectForbiddenError):
        service.create_conversation(
            ConversationCreate(title="t", project_id=project.id), other_id
        )


def test_listing_filters_by_project(membership_env):
    sf, owner_id, _other_id, _agent_id = membership_env
    service, project_service = _service(sf)
    from app.schemas.project import ProjectCreate

    project = project_service.create_project(owner_id, ProjectCreate(name="Roadmap"))
    inside = service.create_conversation(
        ConversationCreate(title="inside", project_id=project.id), owner_id
    )
    service.create_conversation(ConversationCreate(title="outside"), owner_id)

    filtered = service.get_by_user_id(owner_id, project_id=project.id)
    unfiltered = service.get_by_user_id(owner_id)

    assert [c.id for c in filtered.items] == [inside.id]
    assert len(unfiltered.items) == 2, "the flat list still shows every conversation"


def test_listing_with_messages_include_also_respects_project_filter(membership_env):
    """Ruling: the include=["messages"] branch uses a DIFFERENT strategy method
    (get_with_recent_messages + count_by_owner_id) than the plain path
    (get_by_owner_id). Filtering only the plain path would silently ignore the
    project filter whenever messages are requested, and would also report a
    wrong pagination total.
    """
    sf, owner_id, _other_id, _agent_id = membership_env
    service, project_service = _service(sf)
    from app.schemas.project import ProjectCreate

    project = project_service.create_project(owner_id, ProjectCreate(name="Roadmap"))
    inside = service.create_conversation(
        ConversationCreate(title="inside", project_id=project.id), owner_id
    )
    service.create_conversation(ConversationCreate(title="outside"), owner_id)

    filtered = service.get_by_user_id(owner_id, project_id=project.id, include=["messages"])

    assert [c.id for c in filtered.items] == [inside.id]
    assert filtered.meta.total == 1
