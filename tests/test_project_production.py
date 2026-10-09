"""Project transaction regressions against a dedicated PostgreSQL database."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy import event, func, select, update

from app.core.exceptions.project import ProjectConversationNotFoundError, ProjectNotFoundError
from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent
from app.schemas.conversation import ConversationCreate
from app.schemas.project import ProjectCreate
from tests.database_isolation import requires_test_database
from tests.test_project_membership import _service
from tests.test_project_membership import membership_env as _membership_env

membership_env = _membership_env

pytestmark = requires_test_database


def _prepared(membership_env):
    sf, owner, other, agent = membership_env
    conversations, projects = _service(sf)
    project = projects.create_project(owner, ProjectCreate(name="Production review"))
    projects.set_agents(owner, project.id, [agent])
    return sf, owner, agent, conversations, projects, project


def _fail_agent_insert(sf):
    with sf() as session:
        engine = session.get_bind()

    def fail(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.startswith("INSERT INTO conversation_custom_agents"):
            raise RuntimeError("simulated inherited agent insert failure")

    return engine, fail


def test_attach_rolls_back_membership_when_inheritance_fails(membership_env):
    sf, owner, _agent, conversations, projects, project = _prepared(membership_env)
    conversation = conversations.create_conversation(ConversationCreate(title="outside"), owner)
    engine, fail = _fail_agent_insert(sf)
    event.listen(engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="inherited agent insert failure"):
            projects.attach_conversation(owner, project.id, conversation.id)
    finally:
        event.remove(engine, "before_cursor_execute", fail)
    with sf() as session:
        assert session.get(Conversation, conversation.id).project_id is None
        assert session.scalar(select(func.count()).select_from(ConversationCustomAgent)) == 0


def test_create_rolls_back_conversation_when_inheritance_fails(membership_env):
    sf, owner, _agent, conversations, _projects, project = _prepared(membership_env)
    engine, fail = _fail_agent_insert(sf)
    event.listen(engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="inherited agent insert failure"):
            conversations.create_conversation(
                ConversationCreate(title="inside", project_id=project.id), owner
            )
    finally:
        event.remove(engine, "before_cursor_execute", fail)
    with sf() as session:
        assert (
            session.scalar(
                select(func.count()).select_from(Conversation).where(Conversation.owner_id == owner)
            )
            == 0
        )


@pytest.mark.parametrize("operation", ["attach", "create"])
def test_membership_rechecks_deleted_project_in_write_transaction(
    membership_env, monkeypatch, operation
):
    sf, owner, _agent, conversations, projects, project = _prepared(membership_env)
    conversation = conversations.create_conversation(ConversationCreate(title="outside"), owner)
    original = projects.require_owned

    def deleted_after_validation(owner_id, project_id):
        result = original(owner_id, project_id)
        projects.repository.soft_delete_and_detach(owner_id, project_id)
        return result

    monkeypatch.setattr(projects, "require_owned", deleted_after_validation)
    with pytest.raises(ProjectNotFoundError):
        if operation == "attach":
            projects.attach_conversation(owner, project.id, conversation.id)
        else:
            conversations.create_conversation(
                ConversationCreate(title="inside", project_id=project.id), owner
            )
    with sf() as session:
        assert session.get(Conversation, conversation.id).project_id is None
        assert (
            session.scalar(
                select(func.count()).select_from(Conversation).where(Conversation.owner_id == owner)
            )
            == 1
        )


def test_detach_does_not_erase_move_to_another_project(membership_env):
    sf, owner, _agent, conversations, projects, project = _prepared(membership_env)
    destination = projects.create_project(owner, ProjectCreate(name="Destination"))
    conversation = conversations.create_conversation(
        ConversationCreate(title="inside", project_id=project.id), owner
    )
    with sf() as session:
        engine = session.get_bind()
    moved = False

    def move_before_detach(_conn, _cursor, statement, _parameters, _context, _executemany):
        nonlocal moved
        if not moved and statement.startswith("UPDATE conversations SET"):
            moved = True
            with sf() as session:
                session.execute(
                    update(Conversation)
                    .where(Conversation.id == conversation.id)
                    .values(project_id=destination.id)
                )
                session.commit()

    event.listen(engine, "before_cursor_execute", move_before_detach)
    try:
        with pytest.raises(ProjectConversationNotFoundError):
            projects.detach_conversation(owner, project.id, conversation.id)
    finally:
        event.remove(engine, "before_cursor_execute", move_before_detach)
    assert moved
    with sf() as session:
        assert session.get(Conversation, conversation.id).project_id == destination.id


def test_concurrent_attach_is_idempotent(membership_env):
    sf, owner, agent, conversations, projects, project = _prepared(membership_env)
    conversation = conversations.create_conversation(ConversationCreate(title="outside"), owner)
    start = Barrier(4)

    def attach():
        start.wait(timeout=5)
        projects.attach_conversation(owner, project.id, conversation.id)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(attach) for _ in range(4)]
        for future in futures:
            future.result(timeout=10)
    with sf() as session:
        rows = session.execute(
            select(
                ConversationCustomAgent.custom_agent_id, ConversationCustomAgent.agent_order
            ).where(ConversationCustomAgent.conversation_id == conversation.id)
        ).all()
    assert rows == [(agent, 0)]
