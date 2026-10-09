"""Project and agent deletion must coordinate with fork snapshot writes."""

import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select, text

from app.models.base import Base
from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.project import ProjectCustomAgent
from app.models.user_memory import UserMemory
from app.repositories.conversation_fork import ConversationForkRepository, ForkPlan
from app.repositories.custom_agent import CustomAgentRepository
from app.repositories.project import ProjectRepository
from tests.integration import test_conversation_fork_postgres as support

pg = support.pg
pytestmark = pytest.mark.selector_event_loop


@pytest.fixture(scope="module")
def pg_engines():
    """Include tables project deletion uses even on an unmigrated test database."""
    factory = support.pg_engines.__wrapped__()
    sync, async_engine = next(factory)
    try:
        with sync() as session:
            Base.metadata.create_all(
                session.get_bind(), tables=[ProjectCustomAgent.__table__, UserMemory.__table__]
            )
        yield sync, async_engine
    finally:
        next(factory, None)


def _backend_pid(conn):
    driver = conn.connection.driver_connection
    return driver.info.backend_pid


def _plan(source):
    return ForkPlan(
        source.conversation_id,
        source.owner_id,
        source.id_at(2),
        uuid4(),
        "review",
        None,
        5000,
        32 * 1024 * 1024,
        3.0,
    )


def _wait_for_deletion(sf, future, pid, blocker):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if future.done():
            future.result()
            return
        with sf() as session:
            blockers = session.scalar(text("SELECT pg_blocking_pids(:pid)"), {"pid": pid})
        if blocker in blockers:
            return
        time.sleep(0.01)
    raise AssertionError("deletion neither completed nor waited for the fork")


async def test_fork_and_project_delete_complete_without_deadlock(pg, monkeypatch):
    owner = support._user(pg)
    projects = ProjectRepository(pg.sync)
    project = projects.create(owner, {"name": "Fork project"})
    source = support._six_message_source(pg, owner, project_id=project.id)
    repository = ConversationForkRepository(pg.sync, async_session_factory=pg.async_factory)
    original = repository._lock_source
    deletion_pid = Queue()
    with pg.sync() as session:
        engine = session.get_bind()

    def capture_pid(conn, _cursor, statement, _parameters, _context, _many):
        if "FROM projects" in statement and "FOR UPDATE" in statement:
            deletion_pid.put(_backend_pid(conn))

    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = []

        def lock_then_delete(session, plan):
            result = original(session, plan)
            fork_pid = session.scalar(text("SELECT pg_backend_pid()"))
            event.listen(engine, "before_cursor_execute", capture_pid)
            future = pool.submit(projects.soft_delete_and_detach, owner, project.id)
            futures.append(future)
            pid = deletion_pid.get(timeout=5)
            _wait_for_deletion(pg.sync, future, pid, fork_pid)
            return result

        monkeypatch.setattr(repository, "_lock_source", lock_then_delete)
        try:
            child, _ = await repository.afork_snapshot(plan=_plan(source))
        finally:
            event.remove(engine, "before_cursor_execute", capture_pid)
        assert futures[0].result(timeout=5) is True
    with pg.sync() as session:
        assert session.get(Conversation, source.conversation_id).project_id is None
        assert session.get(Conversation, child.id).project_id is None


async def test_fork_cannot_resurrect_deleted_agent_attachments(pg):
    source = support._six_message_source(pg)
    agents = CustomAgentRepository(pg.sync)
    agent = agents.create(
        source.owner_id,
        {
            "name": "Fork agent",
            "slug": "fork-agent",
            "prompt": "Help",
            "provider_type": "openai",
            "model": "gpt-4.1-mini",
        },
    )
    agents.replace_attachments(source.owner_id, source.conversation_id, [agent.id])
    repository = ConversationForkRepository(pg.sync, async_session_factory=pg.async_factory)
    deletion_pid = Queue()
    futures = []
    with pg.sync() as session:
        engine = session.get_bind()

    def capture_pid(conn, _cursor, statement, _parameters, _context, _many):
        if "FROM custom_agents" in statement and "FOR UPDATE" in statement:
            deletion_pid.put(_backend_pid(conn))

    with ThreadPoolExecutor(max_workers=1) as pool:

        def delete_before_insert(conn, _cursor, statement, _parameters, _context, _many):
            if not futures and statement.startswith("INSERT INTO conversation_custom_agents"):
                fork_pid = _backend_pid(conn)
                future = pool.submit(agents.delete_with_detach, source.owner_id, agent.id)
                futures.append(future)
                pid = deletion_pid.get(timeout=5)
                _wait_for_deletion(pg.sync, future, pid, fork_pid)

        event.listen(engine, "before_cursor_execute", capture_pid)
        event.listen(pg.async_engine.sync_engine, "before_cursor_execute", delete_before_insert)
        try:
            await repository.afork_snapshot(plan=_plan(source))
        finally:
            event.remove(pg.async_engine.sync_engine, "before_cursor_execute", delete_before_insert)
            event.remove(engine, "before_cursor_execute", capture_pid)
        assert futures[0].result(timeout=5) is True
    with pg.sync() as session:
        assert session.get(CustomAgent, agent.id).deleted_at is not None
        assert (
            session.scalar(
                select(func.count())
                .select_from(ConversationCustomAgent)
                .where(ConversationCustomAgent.custom_agent_id == agent.id)
            )
            == 0
        )
