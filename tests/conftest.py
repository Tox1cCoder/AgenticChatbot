"""Shared pytest configuration and fixtures.

This module is auto-loaded by pytest and provides shared utilities for all tests.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import sys
import tempfile

import pytest

# Set this before pytest imports application modules. Client settings otherwise
# fall through to the developer's live LOCALAPPDATA profile, where enabled MCP
# processes must never be started by the test suite.
_PYTEST_RUNTIME = tempfile.TemporaryDirectory(
    prefix="sample-chatbot-pytest-",
    ignore_cleanup_errors=True,
)
os.environ["CLIENT_PROFILE_ROOT"] = _PYTEST_RUNTIME.name
atexit.register(_PYTEST_RUNTIME.cleanup)

# Client settings also read a dotenv file, defaulting to the repository's own
# .env.client. Whatever a developer keeps in that file would otherwise leak into
# the suite, so point the loader at an empty file instead.
_PYTEST_CLIENT_ENV_FILE = os.path.join(_PYTEST_RUNTIME.name, "client-settings-empty")
with open(_PYTEST_CLIENT_ENV_FILE, "w", encoding="utf-8") as _handle:
    _handle.write("")
os.environ["CLIENT_ENV_FILE"] = _PYTEST_CLIENT_ENV_FILE


class FakePopen:
    """Minimal subprocess.Popen stub that records the command and supports wait()."""

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.pid = 99999

    def wait(self):
        return 0

    def terminate(self):
        pass

    def poll(self):
        return 0


def async_double(fn):
    """Wrap a sync test-double callable so it can stand in for an async twin.

    Repositories expose ``a``-prefixed async twins alongside their sync methods.
    Test doubles are built from plain lambdas, which cannot be async, so this
    adapts one lambda to serve both call styles without duplicating it.
    """

    async def _wrapped(*args, **kwargs):
        return fn(*args, **kwargs)

    return _wrapped


# Export helpers for use in test modules
__all__ = ["FakePopen", "async_double"]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "selector_event_loop: run this async test on a SelectorEventLoop "
        "(required for async psycopg on Windows)",
    )


def pytest_collection_modifyitems(config, items):
    """Keep paid/networked provider contracts out of ordinary test runs."""
    if "live_provider" in config.option.markexpr:
        return
    selected = []
    deselected = []
    for item in items:
        if item.get_closest_marker("live_provider"):
            deselected.append(item)
        else:
            selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def pytest_asyncio_loop_factories(config, item):
    """Choose the event loop implementation per test, Windows only.

    Two Windows constraints pull in opposite directions and cannot be satisfied
    by one loop type:

    * async psycopg refuses to run on a ``ProactorEventLoop`` and raises
      ``InterfaceError`` at connect time, so database tests need a
      ``SelectorEventLoop`` (``app/main.py:440`` sets the equivalent policy in
      production); and
    * ``SelectorEventLoop`` cannot spawn subprocesses, which
      ``tests/client_backend/test_skill_execution_engine.py`` genuinely does.

    Proactor therefore stays the default — preserving existing behavior — and
    async database tests opt in with ``pytest.mark.selector_event_loop``.

    Returning exactly one factory matters: pytest-asyncio parametrizes across
    every factory returned, so a two-entry mapping would run each async test
    twice.
    """
    if sys.platform != "win32":
        return {"default": asyncio.new_event_loop}
    if item.get_closest_marker("selector_event_loop"):
        return {"selector": asyncio.SelectorEventLoop}
    return {"proactor": asyncio.ProactorEventLoop}


@pytest.fixture(scope="session")
def _async_db_available() -> bool:
    """Probe the async engine once per session."""
    from sqlalchemy import text

    from app.database.async_session import AsyncSessionLocal

    async def probe() -> bool:
        try:
            async with AsyncSessionLocal() as session:
                await session.execute(text("select 1"))
            return True
        except Exception:
            return False

    if sys.platform == "win32":
        return asyncio.run(probe(), loop_factory=asyncio.SelectorEventLoop)
    return asyncio.run(probe())


@pytest.fixture
def require_async_db(_async_db_available: bool) -> None:
    """Skip a test that needs a reachable PostgreSQL."""
    if not _async_db_available:
        pytest.skip("async PostgreSQL is not reachable")


@pytest.fixture
def forbid_sync_db_on_event_loop():
    """Record every sync-engine connection checked out while a loop is running.

    This is the invariant the async migration exists to enforce: no code reached
    from an ``async def`` may touch the sync engine, because that blocks the
    event loop and stalls every other in-flight request. Counting event-loop
    ticks proves a single call is non-blocking; this proves a whole code path is.

    Yields the list of offending call sites so a test can assert it is empty, or
    assert a known count while a migration is still in progress.
    """
    import os
    import traceback

    from sqlalchemy import event

    from app.database.session import engine

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    offenders: list[str] = []

    def _on_checkout(_dbapi_connection, _connection_record, _connection_proxy):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # off-loop use (Celery, Alembic, fixtures) is fine
        frames = [
            frame
            for frame in traceback.extract_stack()
            if frame.filename.startswith(project_root) and "site-packages" not in frame.filename
        ]
        offenders.append(
            " <- ".join(
                f"{os.path.relpath(frame.filename, project_root)}:{frame.lineno}"
                for frame in frames[-2:]
            )
            or "<unknown>"
        )

    event.listen(engine.pool, "checkout", _on_checkout)
    try:
        yield offenders
    finally:
        event.remove(engine.pool, "checkout", _on_checkout)


@pytest.fixture
def seeded_conversation_id(require_async_db):
    """Insert a throwaway owner and conversation, then remove them.

    Seeding goes through the sync engine so these fixtures stay usable by tests
    that are verifying the async path — the fixture is never the thing under
    test. Messages cascade with the conversation.
    """
    from uuid import uuid4

    from app.database.session import SessionLocal
    from app.models.conversation import Conversation
    from app.models.message import Message
    from app.models.user import User

    suffix = uuid4().hex[:12]
    with SessionLocal() as session:
        owner = User(
            username=f"async-fixture-{suffix}",
            email=f"async-fixture-{suffix}@example.invalid",
            password_hash="not-a-real-hash",
        )
        session.add(owner)
        session.flush()
        conversation = Conversation(owner_id=owner.id, title="async-twin-fixture")
        session.add(conversation)
        session.commit()
        conversation_id = conversation.id
        owner_id = owner.id

    yield conversation_id

    # Persisting an assistant message also enqueues a compaction job whose
    # foreign key targets (conversation_id, sequence), so dependents must go
    # before the messages they reference.
    from app.models.conversation_memory_summary import ConversationMemorySummary
    from app.models.conversation_summary_job import ConversationSummaryJob

    with SessionLocal() as session:
        for model in (ConversationSummaryJob, ConversationMemorySummary):
            session.query(model).filter(model.conversation_id == conversation_id).delete()
        session.query(Message).filter(Message.conversation_id == conversation_id).delete()
        session.query(Conversation).filter(Conversation.id == conversation_id).delete()
        session.query(User).filter(User.id == owner_id).delete()
        session.commit()


@pytest.fixture
def seeded_message_id(seeded_conversation_id):
    """Persist one message in the seeded conversation via the sync path."""
    from app.database.session import SessionLocal
    from app.models.enums import MessageRole
    from app.repositories.message import MessageRepository
    from app.schemas.message import MessageCreate

    repository = MessageRepository(session_factory=SessionLocal)
    message = repository.create(
        MessageCreate(
            conversation_id=seeded_conversation_id,
            role=MessageRole.user,
            content="seeded",
        )
    )
    return message.id
