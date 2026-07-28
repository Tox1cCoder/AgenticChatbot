# Async SQLAlchemy Migration — Phase 1 (Foundation + Hot Path) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the FastAPI request path true async database I/O, so a slow query in one request can no longer stall every other in-flight stream, and cut the blocking DB round-trips that sit in front of time-to-first-token.

**Architecture:** One query implementation, two transports. Every query body today lives in a strategy method taking `db: Session` (e.g. `DefaultCommandStrategy.create(db, schema)`); the repository wrapper only owns the session. We add an `AsyncSession` transport that runs those *unchanged* sync-style callables via `AsyncSession.run_sync()`, which performs real async I/O through greenlet — no thread pool, no query rewrites. Repositories gain `a`-prefixed async twins alongside their existing sync methods, so callers migrate module-by-module instead of in one unreviewable big bang. The sync engine stays, permanently, for Celery workers and Alembic.

**Tech Stack:** SQLAlchemy 2.0.51 (async ORM), psycopg 3.3.4 (async mode), greenlet 3.5.3, FastAPI, dependency-injector, pytest-asyncio (`asyncio_mode = "auto"`), PostgreSQL 18.3.

## Global Constraints

- **psycopg async requires a `SelectorEventLoop`.** On Windows the default `ProactorEventLoop` raises `psycopg.InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in async mode`. `app/main.py:440` and `app/workers/cleanup_tasks.py:138-142` already set the selector policy; **pytest does not** and must be fixed (Task 2) before any async-DB test can pass.
- **Do not remove or repoint the sync engine.** `app/database/session.py` `engine`/`SessionLocal` stay exactly as they are. Celery tasks (`app/workers/`, 39 sync methods) and Alembic (`alembic.ini`, `app/alembic/env.py`) depend on them.
- **Do not modify any strategy method.** `app/repositories/command_strategy.py`, `query_strategy.py`, and the per-repository `*CRUDStrategy` classes keep their `db: Session` signatures and their `commit()`/`refresh()`/`expunge()` calls. They are the shared query implementation; `run_sync` executes them verbatim.
- **`expire_on_commit=False` and `autoflush=False`** must be set on the async session factory, matching `SessionLocal` (`app/database/session.py:18`). Repositories return detached ORM objects and callers read attributes after the session closes; changing this silently breaks them with `DetachedInstanceError`.
- **Async twins are named with an `a` prefix** (`create` → `acreate`, `get_by_id` → `aget_by_id`). Never change or delete an existing sync method signature in this phase.
- **The async URL is derived, never configured separately.** `make_url(settings.database_url).set(drivername="postgresql+psycopg")`. There must be no second `DATABASE_URL`-style setting to drift.
- Line length 100, ruff clean, functions under 100 lines (project standard, `~/.claude/rules/development.md`).

## Out of Scope for This Phase

Phase 1 delivers working software and the measurable latency win, then stops. These get their own plans:

- Migrating the remaining ~22 repositories' call sites (Phase 2, per-module sweep).
- Deleting the now-unused sync repository twins (Phase 3, only after Phase 2 proves nothing calls them).
- Converting Celery tasks to async (may never be worth it).

## File Structure

| File | Responsibility |
|---|---|
| `app/database/async_session.py` (create) | Owns the async engine, `AsyncSessionLocal`, derived async URL, and pool sizing. Mirrors `session.py` and nothing more. |
| `app/database/database.py` (modify) | Gains `async_session()` alongside `session()`, so the container can inject an async factory the same way it injects the sync one. |
| `app/repositories/session_transport.py` (create) | `RepositorySessionMixin` with `_run` (sync) and `_arun` (async via `run_sync`). The single place the transport choice lives. |
| `app/repositories/message.py`, `conversation.py`, `document.py` (modify) | Async twins for the methods the streaming hot path uses. |
| `app/core/container.py` (modify) | Wires `async_session_factory` into the three migrated repositories. |
| `app/services/message_service.py` (modify) | `create_message_stream` awaits the async twins for its pre-stream calls. |
| `app/ai/graph.py` (modify) | `_route_node`'s document count awaits the async twin. |
| `tests/conftest.py` (modify) | Selector event-loop policy for the whole test session. |
| `tests/test_async_session.py` (create) | Engine/URL/pool config and `run_sync` transport behavior. |
| `tests/test_repository_async_twins.py` (create) | Async twins return the same results as their sync counterparts. |
| `tests/test_stream_path_is_non_blocking.py` (create) | The regression test that matters: the event loop stays responsive during a DB call. |

---

### Task 1: Async engine and session factory

**Files:**
- Create: `app/database/async_session.py`
- Modify: `app/core/config.py` (add pool settings near `database_url`, line ~217)
- Test: `tests/test_async_session.py`

**Interfaces:**
- Consumes: `settings.database_url` (existing).
- Produces: `async_engine`, `AsyncSessionLocal` (an `async_sessionmaker[AsyncSession]`), `async_database_url() -> URL`, `get_async_session_factory()`. Later tasks import `AsyncSessionLocal` and `get_async_session_factory`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_async_session.py
"""The async engine must mirror the sync engine's semantics exactly."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text

from app.database.async_session import (
    AsyncSessionLocal,
    async_database_url,
    async_engine,
)
from app.database.session import engine as sync_engine


def test_async_url_is_derived_from_the_same_setting():
    url = async_database_url()
    assert url.drivername == "postgresql+psycopg"
    # Same target database as the sync engine — no second source of truth.
    assert url.database == sync_engine.url.database
    assert url.host == sync_engine.url.host


def test_async_session_factory_matches_sync_semantics():
    assert AsyncSessionLocal.kw["expire_on_commit"] is False
    assert AsyncSessionLocal.kw["autoflush"] is False


def test_pool_is_explicitly_sized():
    # Default (5 + 10) is a concurrency ceiling; sizing must be deliberate.
    assert async_engine.pool.size() >= 10


@pytest.mark.asyncio
async def test_async_session_executes_a_query():
    async with AsyncSessionLocal() as session:
        assert (await session.execute(text("select 1"))).scalar_one() == 1


@pytest.mark.asyncio
async def test_run_sync_executes_unchanged_sync_style_code():
    """This is the bridge the whole migration depends on."""

    def sync_style(sync_session):
        return sync_session.execute(text("select 42")).scalar_one()

    async with AsyncSessionLocal() as session:
        assert await session.run_sync(sync_style) == 42


@pytest.mark.asyncio
async def test_concurrent_sessions_do_not_serialize():
    async def one():
        async with AsyncSessionLocal() as session:
            await session.execute(text("select pg_sleep(0.3)"))

    started = asyncio.get_running_loop().time()
    await asyncio.gather(*(one() for _ in range(4)))
    elapsed = asyncio.get_running_loop().time() - started
    # Serialized would be ~1.2s; concurrent is ~0.3s.
    assert elapsed < 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_async_session.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.database.async_session'`

- [ ] **Step 3: Add pool settings to config**

In `app/core/config.py`, directly after the `database_url` field (line ~217-220):

```python
    db_pool_size: int = Field(
        default=20,
        description=(
            "Async engine connection pool size. Sized for concurrent streaming "
            "requests; the SQLAlchemy default of 5 is a concurrency ceiling."
        ),
    )
    db_max_overflow: int = Field(
        default=10,
        description="Additional async connections allowed above db_pool_size under burst.",
    )
```

- [ ] **Step 4: Write the async session module**

```python
# app/database/async_session.py
"""Async engine and session factory for the FastAPI request path.

Mirrors :mod:`app.database.session` for asynchronous callers. The sync engine
in that module is intentionally kept: Celery workers and Alembic are sync and
must stay that way.

The URL is derived from the one ``database_url`` setting rather than configured
separately, so the two engines can never drift onto different databases.

psycopg's async mode requires a ``SelectorEventLoop``; ``app/main.py`` and the
Celery entrypoints set that policy, and ``tests/conftest.py`` does the same for
the suite.
"""

from __future__ import annotations

from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

_ASYNC_DRIVER = "postgresql+psycopg"


def async_database_url() -> URL:
    """Return ``settings.database_url`` retargeted at the async psycopg driver."""
    return make_url(settings.database_url).set(drivername=_ASYNC_DRIVER)


async_engine = create_async_engine(
    async_database_url(),
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.api_debug,
)

# expire_on_commit=False and autoflush=False mirror SessionLocal: repositories
# hand back detached ORM objects whose loaded attributes callers still read.
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


def get_async_engine():
    """Return the shared async engine."""
    return async_engine


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the shared async session factory."""
    return AsyncSessionLocal


async def dispose_async_engine() -> None:
    """Close pooled async connections. Call on application shutdown."""
    await async_engine.dispose()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_async_session.py -v`
Expected: PASS (all 6). If every test errors with `psycopg.InterfaceError ... ProactorEventLoop`, do Task 2 first and re-run.

- [ ] **Step 6: Commit**

```bash
git add app/database/async_session.py app/core/config.py tests/test_async_session.py
git commit -m "feat: add async SQLAlchemy engine and session factory"
```

---

### Task 2: Selector event-loop policy for the test suite

**Files:**
- Modify: `tests/conftest.py`
- Test: `tests/test_async_session.py` (from Task 1, used as the proof)

**Interfaces:**
- Consumes: nothing.
- Produces: a session-scoped autouse fixture guaranteeing every async test runs on a `SelectorEventLoop`. Every later async-DB test depends on this.

- [ ] **Step 1: Confirm the failure mode is real**

Run: `.venv/Scripts/python.exe -m pytest tests/test_async_session.py::test_async_session_executes_a_query -v`
Expected on Windows: FAIL with `psycopg.InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in async mode`

- [ ] **Step 2: Add the policy fixture to conftest**

Append to `tests/conftest.py`:

```python
import asyncio
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def _selector_event_loop_policy():
    """Force a SelectorEventLoop for the whole suite.

    psycopg's async mode cannot run on Windows' default ProactorEventLoop.
    ``app/main.py`` and the Celery entrypoints set this policy in production;
    pytest-asyncio builds its loops from the process policy, so the suite has
    to set it too or every async database test fails at connect time.
    """
    if sys.platform != "win32":
        yield
        return

    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        yield
    finally:
        asyncio.set_event_loop_policy(previous)
```

- [ ] **Step 3: Run the async tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_async_session.py -v`
Expected: PASS (all 6)

- [ ] **Step 4: Verify the existing suite is unaffected**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 2729 passed, 1 failed. The one failure must be
`test_alembic_full_chain_postgres.py::test_readme_tracks_migration_head_and_current_graph_contract`,
which is a pre-existing README/migration-head drift unrelated to this work. Any
*other* failure is a regression from this task — stop and fix it.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py
git commit -m "test: force SelectorEventLoop so async psycopg works under pytest"
```

---

### Task 3: Repository session transport mixin

**Files:**
- Create: `app/repositories/session_transport.py`
- Modify: `app/database/database.py`
- Test: `tests/test_repository_session_transport.py`

**Interfaces:**
- Consumes: `AsyncSessionLocal` from Task 1.
- Produces:
  - `Database.async_session()` — an `@asynccontextmanager` yielding `AsyncSession`.
  - `RepositorySessionMixin` with `_run(work: Callable[[Session], T]) -> T` and
    `async _arun(work: Callable[[Session], T]) -> T`, plus
    `__init__`-injected `session_factory` and optional `async_session_factory`.
  Tasks 4-5 inherit this mixin; Task 6 injects `async_session_factory`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repository_session_transport.py
"""Both transports must execute the *same* sync-style callable."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.database.async_session import AsyncSessionLocal
from app.database.session import SessionLocal
from app.repositories.session_transport import RepositorySessionMixin


class _Probe(RepositorySessionMixin):
    pass


def _work(session):
    """Ordinary sync-style query code — identical for both transports."""
    return session.execute(text("select 7")).scalar_one()


def test_sync_transport_runs_the_work():
    probe = _Probe(session_factory=SessionLocal)
    assert probe._run(_work) == 7


@pytest.mark.asyncio
async def test_async_transport_runs_the_same_work():
    probe = _Probe(session_factory=SessionLocal, async_session_factory=AsyncSessionLocal)
    assert await probe._arun(_work) == 7


@pytest.mark.asyncio
async def test_async_transport_without_a_factory_is_a_clear_error():
    probe = _Probe(session_factory=SessionLocal)
    with pytest.raises(RuntimeError, match="async_session_factory"):
        await probe._arun(_work)


@pytest.mark.asyncio
async def test_async_transport_propagates_work_exceptions():
    def boom(session):
        raise ValueError("query failed")

    probe = _Probe(session_factory=SessionLocal, async_session_factory=AsyncSessionLocal)
    with pytest.raises(ValueError, match="query failed"):
        await probe._arun(boom)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_repository_session_transport.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.repositories.session_transport'`

- [ ] **Step 3: Write the mixin**

```python
# app/repositories/session_transport.py
"""Shared session transports for repositories.

A repository operation is a sync-style callable taking a ``Session`` — exactly
what the existing ``*CRUDStrategy`` methods already are. This mixin runs such a
callable over either transport:

* ``_run``  — the sync engine, for Celery workers, Alembic, and callers not yet
  migrated.
* ``_arun`` — an ``AsyncSession``, via ``run_sync``. SQLAlchemy drives the
  callable on a greenlet and performs the actual I/O asynchronously, so the
  query body needs no rewriting and the event loop is never blocked.

Keeping both here means the query implementation exists once.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from sqlalchemy.orm import Session

T = TypeVar("T")


class RepositorySessionMixin:
    """Provide sync and async execution of sync-style repository work."""

    def __init__(
        self,
        session_factory: Callable[[], Any],
        async_session_factory: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.session_factory = session_factory
        self.async_session_factory = async_session_factory
        super().__init__(**kwargs)

    def _run(self, work: Callable[[Session], T]) -> T:
        """Execute ``work`` on the sync engine."""
        with self.session_factory() as session:
            return work(session)

    async def _arun(self, work: Callable[[Session], T]) -> T:
        """Execute ``work`` on the async engine without blocking the event loop."""
        if self.async_session_factory is None:
            raise RuntimeError(
                f"{type(self).__name__} was constructed without an "
                "async_session_factory; wire it in app/core/container.py before "
                "calling an async repository method."
            )
        async with self.async_session_factory() as session:
            return await session.run_sync(work)
```

- [ ] **Step 4: Add the async session context manager to Database**

In `app/database/database.py`, add the import and the method:

```python
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.async_session import AsyncSessionLocal
from app.database.session import SessionLocal
```

and inside `class Database`, after `session()`:

```python
    @asynccontextmanager
    async def async_session(self) -> AsyncIterator[AsyncSession]:
        """Provide an async database session as a context manager.

        Mirrors :meth:`session` for the async request path.
        """
        session: AsyncSession = AsyncSessionLocal()
        try:
            yield session
        except Exception:
            logger.exception("Async session rollback because of exception")
            await session.rollback()
            raise
        finally:
            await session.close()
```

Add `AsyncIterator` to the `collections.abc` import at the top of the file.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_repository_session_transport.py -v`
Expected: PASS (all 4)

- [ ] **Step 6: Commit**

```bash
git add app/repositories/session_transport.py app/database/database.py tests/test_repository_session_transport.py
git commit -m "feat: add sync and async repository session transports"
```

---

### Task 4: Async twins on MessageRepository and ConversationCompactionRepository

**Files:**
- Modify: `app/repositories/message.py`
- Modify: `app/repositories/conversation_compaction.py`
- Test: `tests/test_repository_async_twins.py`

**Interfaces:**
- Consumes: `RepositorySessionMixin` (Task 3).
- Produces:
  - `ConversationCompactionRepository._persist_message_in_session(session, message_data) -> Message`
    (extracted, shared), `ConversationCompactionRepository.apersist_message(message_data) -> Message`.
  - `MessageRepository.acreate`, `aget_by_id`, `acount_by_conversation_id`,
    `aget_by_conversation_id`, `aget_latest_by_conversation`.
  Task 7 calls `acreate`; Phase 2 calls the rest.

**Two things to know before you start (they are not what you'd guess):**

1. **`MessageRepository.create` does not own a session.** It builds
   `message_data` and delegates to
   `self._compaction_repository.persist_message(...)`, which owns the session and
   runs a multi-statement transaction. So `acreate` must delegate to an async
   twin *on the compaction repository* — that is why this task touches two files.
2. **`MessageRepository.get_prompt_history` takes `db: Session` as its first
   parameter.** It is strategy-style code, not a session-owning wrapper, so it
   gets no twin. Callers already supply the session.

Do not convert all 26 methods at once — an unused twin is untested surface.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repository_async_twins.py
"""An async twin must return exactly what its sync counterpart returns."""

from __future__ import annotations

import pytest

from app.database.async_session import AsyncSessionLocal
from app.database.session import SessionLocal
from app.repositories.message import MessageRepository


@pytest.fixture
def message_repository():
    return MessageRepository(
        session_factory=SessionLocal,
        async_session_factory=AsyncSessionLocal,
    )


@pytest.mark.asyncio
async def test_acount_matches_sync_count(message_repository, seeded_conversation_id):
    expected = message_repository.count_by_conversation_id(seeded_conversation_id)
    assert await message_repository.acount_by_conversation_id(seeded_conversation_id) == expected


@pytest.mark.asyncio
async def test_aget_by_id_matches_sync(message_repository, seeded_message_id):
    expected = message_repository.get_by_id(seeded_message_id)
    actual = await message_repository.aget_by_id(seeded_message_id)
    assert actual is not None
    assert actual.id == expected.id
    assert actual.content == expected.content


@pytest.mark.asyncio
async def test_aget_by_id_returns_none_for_missing(message_repository):
    from uuid import uuid4

    assert await message_repository.aget_by_id(uuid4()) is None


@pytest.mark.asyncio
async def test_acreate_persists_and_returns_a_detached_row(
    message_repository, seeded_conversation_id
):
    from app.models.enums import MessageRole
    from app.schemas.message import MessageCreate

    created = await message_repository.acreate(
        MessageCreate(
            conversation_id=seeded_conversation_id,
            role=MessageRole.user,
            content="async twin round-trip",
        )
    )
    # Reading attributes after the session closed proves expire_on_commit=False.
    assert created.content == "async twin round-trip"
    assert message_repository.get_by_id(created.id) is not None
```

Add these fixtures to `tests/conftest.py` (they seed one throwaway conversation
and message through the existing sync path, then clean up):

```python
@pytest.fixture
def seeded_conversation_id():
    from app.database.session import SessionLocal
    from app.models.conversation import Conversation
    from app.repositories.user import UserRepository

    user_repo = UserRepository(session_factory=SessionLocal)
    owner = user_repo.get_or_create_test_user()
    with SessionLocal() as session:
        conversation = Conversation(owner_id=owner.id, title="async-twin-fixture")
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
        conversation_id = conversation.id
    yield conversation_id
    with SessionLocal() as session:
        session.query(Conversation).filter(Conversation.id == conversation_id).delete()
        session.commit()


@pytest.fixture
def seeded_message_id(seeded_conversation_id):
    from app.database.session import SessionLocal
    from app.models.enums import MessageRole
    from app.repositories.message import MessageRepository
    from app.schemas.message import MessageCreate

    repo = MessageRepository(session_factory=SessionLocal)
    message = repo.create(
        MessageCreate(
            conversation_id=seeded_conversation_id,
            role=MessageRole.user,
            content="seeded",
        )
    )
    return message.id
```

If `UserRepository` has no `get_or_create_test_user`, read
`app/repositories/user.py` and use whatever creation method exists, or insert a
`User` row directly the same way the conversation is inserted above.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_repository_async_twins.py -v`
Expected: FAIL — `AttributeError: 'MessageRepository' object has no attribute 'acount_by_conversation_id'`

- [ ] **Step 3: Make MessageRepository use the mixin**

Change the class declaration and `__init__` in `app/repositories/message.py`.
Replace:

```python
class MessageRepository:
    """Repository for Message model using session factory pattern"""

    def __init__(
        self,
        session_factory: callable,
        compaction_repository: ConversationCompactionRepository | None = None,
        compaction_publisher: Callable[[UUID], Any] | None = None,
    ):
        """Initialize repository with session factory for dependency injection."""
        self.session_factory = session_factory
```

with:

```python
class MessageRepository(RepositorySessionMixin):
    """Repository for Message model using session factory pattern"""

    def __init__(
        self,
        session_factory: callable,
        compaction_repository: ConversationCompactionRepository | None = None,
        compaction_publisher: Callable[[UUID], Any] | None = None,
        async_session_factory: Callable[[], Any] | None = None,
    ):
        """Initialize repository with session factory for dependency injection."""
        super().__init__(
            session_factory=session_factory,
            async_session_factory=async_session_factory,
        )
```

and add the import:

```python
from app.repositories.session_transport import RepositorySessionMixin
```

Keep `self._crud_strategy` and `self._compaction_publisher` exactly as they are,
after the `super().__init__` call. **One line does have to change**: the
fallback-constructed compaction repository must receive the async factory too,
or `acreate` raises `RuntimeError` whenever no compaction repository was
injected. Replace:

```python
        self._compaction_repository = compaction_repository or ConversationCompactionRepository(
            session_factory
        )
```

with:

```python
        self._compaction_repository = compaction_repository or ConversationCompactionRepository(
            session_factory,
            async_session_factory=async_session_factory,
        )
```

- [ ] **Step 4: Add the six async twins**

Add directly after each corresponding sync method so the pair stays together.
Each twin delegates to the *same* strategy call the sync method uses:

```python
    async def acount_by_conversation_id(self, conversation_id: UUID) -> int:
        """Async twin of :meth:`count_by_conversation_id`."""
        return await self._arun(
            lambda session: self._crud_strategy.count_by_conversation_id(
                session, conversation_id
            )
        )

    async def aget_by_id(self, id: UUID) -> Message | None:
        """Async twin of :meth:`get_by_id`."""
        return await self._arun(lambda session: self._crud_strategy.get_by_id(session, id))

    async def aget_by_conversation_id(
        self,
        conversation_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: str | None = None,
        order_direction: str = "asc",
        include_feedback: bool = False,
    ) -> Paginator[Message]:
        """Async twin of :meth:`get_by_conversation_id`."""
        return await self._arun(
            lambda session: self._crud_strategy.get_by_conversation_id(
                session,
                conversation_id,
                page,
                limit,
                order_by,
                order_direction,
                include_feedback,
            )
        )

    async def aget_latest_by_conversation(self, conversation_id: UUID) -> Message | None:
        """Async twin of :meth:`get_latest_by_conversation`."""
        return await self._arun(
            lambda session: self._crud_strategy.get_latest_by_conversation(
                session, conversation_id
            )
        )
```

- [ ] **Step 4b: Extract the persist transaction so both transports share it**

In `app/repositories/conversation_compaction.py`, `persist_message` owns a
multi-statement transaction. Split it into a session-taking body plus two thin
wrappers, so the transaction exists exactly once. Replace the existing
`persist_message` with:

```python
    def _persist_message_in_session(
        self, session: Session, message_data: Mapping[str, Any]
    ) -> Message:
        """Allocate a sequence, insert the message, and request assistant work once.

        Session-taking body shared by :meth:`persist_message` and
        :meth:`apersist_message`. Runs unchanged under ``AsyncSession.run_sync``.
        """
        data = dict(message_data)
        conversation_id = data["conversation_id"]
        data.pop("sequence", None)
        now = self._utcnow()

        allocated = session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(next_message_sequence=Conversation.next_message_sequence + 1)
            .returning(Conversation.next_message_sequence - 1)
        ).scalar_one_or_none()
        if allocated is None:
            session.rollback()
            raise ValueError("conversation_not_found")

        message = Message(**data, sequence=int(allocated), feedback=None)
        session.add(message)
        session.flush()
        if message.sender == MessageRole.assistant.value:
            session.execute(
                self._job_upsert_statement(
                    conversation_id=conversation_id,
                    requested_through_sequence=message.sequence,
                    now=now,
                )
            )
        session.commit()
        session.expunge(message)
        return message

    def persist_message(self, message_data: Mapping[str, Any]) -> Message:
        """Allocate a sequence, insert a message, and request assistant work once."""
        return self._run(lambda session: self._persist_message_in_session(session, message_data))

    async def apersist_message(self, message_data: Mapping[str, Any]) -> Message:
        """Async twin of :meth:`persist_message`."""
        return await self._arun(
            lambda session: self._persist_message_in_session(session, message_data)
        )
```

Apply the Step 3 mixin change to `ConversationCompactionRepository` as well, and
add `from sqlalchemy.orm import Session` if it is not already imported.

- [ ] **Step 4c: Add MessageRepository.acreate**

`create` builds `message_data` with no database access, then delegates. Mirror
that exactly, awaiting the compaction twin and keeping the publisher side effect
outside the transaction:

```python
    async def acreate(self, input_schema: MessageCreate) -> Message:
        """Async twin of :meth:`create`."""
        if isinstance(input_schema, dict):
            message_data = input_schema
        else:
            message_data = MessageFactory.create_from_schema_with_role(
                input_schema,
                input_schema.role,
            )
        message = await self._compaction_repository.apersist_message(message_data)
        if message.sender == MessageRole.assistant.value and self._compaction_publisher is not None:
            try:
                self._compaction_publisher(message.conversation_id)
            except Exception:
                logger.exception(
                    "Compaction publish failed for conversation %s", message.conversation_id
                )
        return message
```

Copy the `except` block's comment and behavior verbatim from the existing
`create` — the message and its coalesced job are already committed there, so the
failure must stay swallowed rather than propagating.

Note there is no `aget_prompt_history`: that method takes `db: Session` from its
caller and owns no session.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_repository_async_twins.py -v`
Expected: PASS (all 4)

- [ ] **Step 6: Verify no sync regression**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q -k "message or repository or conversation"`
Expected: all pass. The mixin changed how `session_factory` is assigned, so this
proves every existing sync caller still works.

- [ ] **Step 7: Commit**

```bash
git add app/repositories/message.py app/repositories/conversation_compaction.py \
        tests/test_repository_async_twins.py tests/conftest.py
git commit -m "feat: add async twins for the message repository hot path"
```

---

### Task 5: Async twins on ConversationRepository and DocumentRepository

**Files:**
- Modify: `app/repositories/conversation.py`, `app/repositories/document.py`
- Test: `tests/test_repository_async_twins.py` (extend)

**Interfaces:**
- Consumes: `RepositorySessionMixin` (Task 3).
- Produces: `ConversationRepository.aget_by_id`, `ConversationRepository.aupdate`,
  `DocumentRepository.acount_by_conversation`. Task 7 uses the conversation
  twins; Task 8 uses the document twin.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_repository_async_twins.py`:

```python
@pytest.fixture
def conversation_repository():
    from app.repositories.conversation import ConversationRepository

    return ConversationRepository(
        session_factory=SessionLocal,
        async_session_factory=AsyncSessionLocal,
    )


@pytest.fixture
def document_repository():
    from app.repositories.document import DocumentRepository

    return DocumentRepository(
        session_factory=SessionLocal,
        async_session_factory=AsyncSessionLocal,
    )


@pytest.mark.asyncio
async def test_conversation_aget_by_id_matches_sync(
    conversation_repository, seeded_conversation_id
):
    expected = conversation_repository.get_by_id(seeded_conversation_id)
    actual = await conversation_repository.aget_by_id(seeded_conversation_id)
    assert actual is not None
    assert actual.id == expected.id
    assert actual.title == expected.title


@pytest.mark.asyncio
async def test_document_acount_matches_sync(document_repository, seeded_conversation_id):
    expected = document_repository.count_by_conversation(seeded_conversation_id)
    assert await document_repository.acount_by_conversation(seeded_conversation_id) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_repository_async_twins.py -v -k "conversation_aget or document_acount"`
Expected: FAIL — `AttributeError: ... has no attribute 'aget_by_id'`

- [ ] **Step 3: Apply the mixin and add the twins**

For each of the two repositories, make the same three edits as Task 4 Step 3:
add `from app.repositories.session_transport import RepositorySessionMixin`,
change the class to inherit it, and replace the `self.session_factory = ...`
assignment with `super().__init__(session_factory=..., async_session_factory=...)`
while adding the `async_session_factory` parameter. Then add:

```python
    # app/repositories/conversation.py
    async def aget_by_id(self, id: UUID) -> Conversation | None:
        """Async twin of :meth:`get_by_id`."""
        return await self._arun(lambda session: self._crud_strategy.get_by_id(session, id))

    async def aupdate(
        self, id: UUID, input_schema: ConversationUpdate
    ) -> Conversation | None:
        """Async twin of :meth:`update`.

        Both statements run inside one ``run_sync`` call so they share the
        transaction, exactly as the sync method's single ``with`` block does.
        """

        def work(session):
            db_obj = self._crud_strategy.get_by_id(session, id)
            if db_obj is None:
                return None
            return self._crud_strategy.update(session, db_obj, input_schema)

        return await self._arun(work)
```

```python
    # app/repositories/document.py
    # NOTE: this repository queries directly rather than through a strategy.
    # Keep the query identical to the sync method's.
    async def acount_by_conversation(self, conversation_id: UUID) -> int:
        """Async twin of :meth:`count_by_conversation`."""
        return await self._arun(
            lambda db: db.query(Document)
            .filter(Document.conversation_id == conversation_id)
            .count()
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_repository_async_twins.py -v`
Expected: PASS (all 6)

- [ ] **Step 5: Commit**

```bash
git add app/repositories/conversation.py app/repositories/document.py tests/test_repository_async_twins.py
git commit -m "feat: add async twins for conversation and document repositories"
```

---

### Task 6: Inject the async session factory through the container

**Files:**
- Modify: `app/core/container.py`
- Modify: `app/main.py` (shutdown disposal)
- Test: `tests/test_container_async_wiring.py`

**Interfaces:**
- Consumes: `Database.async_session` (Task 3), the four mixin repositories (Tasks 4-5).
- Produces: container-built `message_repository`, `conversation_compaction_repository`,
  `conversation_repository`, and `document_repository` that have a working
  `async_session_factory`. Tasks 7-8 rely on the app's real wiring, not
  hand-built repositories.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_container_async_wiring.py
"""Container-built repositories must be able to use the async transport."""

from __future__ import annotations

import pytest

from app.core.container import Container


@pytest.fixture(scope="module")
def container():
    return Container()


@pytest.mark.parametrize(
    "provider_name",
    [
        "message_repository",
        "conversation_repository",
        "document_repository",
        "conversation_compaction_repository",
    ],
)
def test_repository_receives_an_async_session_factory(container, provider_name):
    repository = getattr(container, provider_name)()
    assert repository.async_session_factory is not None


def test_message_repository_compaction_delegate_is_also_async_capable(container):
    """acreate() delegates into the compaction repository, so it needs the factory too."""
    repository = container.message_repository()
    assert repository._compaction_repository.async_session_factory is not None


@pytest.mark.asyncio
async def test_container_repository_can_query_asynchronously(container):
    from sqlalchemy import text

    repository = container.message_repository()
    assert await repository._arun(lambda s: s.execute(text("select 1")).scalar_one()) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_container_async_wiring.py -v`
Expected: FAIL — `assert None is not None`

- [ ] **Step 3: Wire the async factory for the four repositories**

In `app/core/container.py`, add one line to each of the four affected
`providers.Factory(...)` declarations. `conversation_compaction_repository` is
included because `MessageRepository.acreate` delegates into it:

```python
    conversation_compaction_repository = providers.Factory(
        ConversationCompactionRepository,
        session_factory=db.provided.session,
        async_session_factory=db.provided.async_session,
    )

    message_repository = providers.Factory(
        MessageRepository,
        session_factory=db.provided.session,
        async_session_factory=db.provided.async_session,
        compaction_repository=conversation_compaction_repository,
        compaction_publisher=providers.Object(publish_conversation_compaction),
    )
```

Do the same one-line addition for `conversation_repository` and
`document_repository`. Leave the other ~21 repository providers untouched — they
have no async twins yet, and `async_session_factory` defaults to `None`.

- [ ] **Step 4: Dispose the async engine on shutdown**

In `app/main.py`, inside the existing lifespan/shutdown handler, add:

```python
    from app.database.async_session import dispose_async_engine

    await dispose_async_engine()
```

Place it alongside the other shutdown cleanup. If the app uses
`@app.on_event("shutdown")` rather than a lifespan context manager, add it to
that handler instead.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_container_async_wiring.py -v`
Expected: PASS (all 4)

- [ ] **Step 6: Commit**

```bash
git add app/core/container.py app/main.py tests/test_container_async_wiring.py
git commit -m "feat: wire async session factory into hot-path repositories"
```

---

### Task 7: Make the streaming pre-flight non-blocking

**Files:**
- Modify: `app/services/message_service.py` (`create_message_stream`, lines ~961-1036)
- Test: `tests/test_stream_path_is_non_blocking.py`

**Interfaces:**
- Consumes: `MessageRepository.acreate`, `ConversationRepository.aget_by_id` (Tasks 4-5).
- Produces: no new public interface. `create_message_stream` keeps its signature
  and event sequence exactly.

**Context:** these are the blocking calls in front of the first streamed event —
`validate_conversation_access` (line ~961), `repository.create` (~969),
`conversation_repository.get_by_id` (~1003). Each currently stalls the whole
event loop, so one slow query delays *every* concurrent stream's first token.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stream_path_is_non_blocking.py
"""The event loop must stay responsive while the stream path hits the DB.

This is the regression test for the whole migration: it fails if any DB call on
the pre-stream path runs synchronously on the loop.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text

from app.database.async_session import AsyncSessionLocal


@pytest.mark.asyncio
async def test_a_slow_query_does_not_stall_the_event_loop():
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("select pg_sleep(0.4)"))
    finally:
        beat.cancel()

    # ~20 ticks if the loop kept running; ~0 if the query blocked it.
    assert ticks >= 10, f"event loop stalled during the query (only {ticks} ticks)"


@pytest.mark.asyncio
async def test_sync_session_does_stall_the_loop_for_contrast():
    """Characterizes the behavior being removed, so the test above has meaning."""
    from app.database.session import SessionLocal

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    try:
        with SessionLocal() as session:
            session.execute(text("select pg_sleep(0.4)"))
    finally:
        beat.cancel()

    assert ticks <= 2, f"expected the sync session to block the loop, got {ticks} ticks"
```

- [ ] **Step 2: Run test to verify the contrast holds**

Run: `.venv/Scripts/python.exe -m pytest tests/test_stream_path_is_non_blocking.py -v`
Expected: both PASS. The second test failing means the sync engine is not
actually blocking and the premise of this plan needs rechecking — stop and
investigate before continuing.

- [ ] **Step 3: Await the async twins in create_message_stream**

In `app/services/message_service.py`, in `create_message_stream`:

```python
-        created_message = self.repository.create(message_entity)
+        created_message = await self.repository.acreate(message_entity)
```

```python
-            conversation = self.conversation_validation_utils.conversation_repository.get_by_id(
-                message_create_data.conversation_id
-            )
+            conversation = (
+                await self.conversation_validation_utils.conversation_repository.aget_by_id(
+                    message_create_data.conversation_id
+                )
+            )
```

`validate_conversation_access` is a sync helper wrapping its own repository
call. Read `app/services/utils/conversation_validation_utils.py`, add an
`async def avalidate_conversation_access` twin that awaits
`conversation_repository.aget_by_id` and performs the identical checks in the
identical order (raising the same exceptions), then call it here:

```python
-        self.conversation_validation_utils.validate_conversation_access(
+        await self.conversation_validation_utils.avalidate_conversation_access(
             user_id, message_create_data.conversation_id
         )
```

Keep the sync method — other callers still use it.

- [ ] **Step 4: Verify the streaming tests still pass**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q -k "stream or message_service or conversation_access"`
Expected: all pass. Watch specifically for an unawaited-coroutine warning, which
means a call site was missed.

- [ ] **Step 5: Commit**

```bash
git add app/services/message_service.py app/services/utils/conversation_validation_utils.py tests/test_stream_path_is_non_blocking.py
git commit -m "perf: make the streaming pre-flight database calls non-blocking"
```

---

### Task 8: Make the router's document lookup non-blocking

**Files:**
- Modify: `app/ai/graph.py` (`_route_node` line ~953, `_conversation_has_documents` line ~1034)
- Test: `tests/test_graph_route_node_async_documents.py`

**Interfaces:**
- Consumes: `DocumentRepository.acount_by_conversation` (Task 5).
- Produces: `MultiAgentWorkflow._aconversation_has_documents(conversation_id) -> bool`,
  replacing the sync call inside `_route_node`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_route_node_async_documents.py
"""Routing must not block the loop on its document-count query."""

from __future__ import annotations

import pytest

from app.ai.graph import MultiAgentWorkflow


class _AsyncDocumentRepository:
    def __init__(self, count: int):
        self._count = count
        self.sync_calls = 0

    def count_by_conversation(self, conversation_id):
        self.sync_calls += 1
        return self._count

    async def acount_by_conversation(self, conversation_id):
        return self._count


@pytest.mark.asyncio
async def test_uses_the_async_twin_and_not_the_sync_method():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    repository = _AsyncDocumentRepository(count=3)
    workflow.document_repository = repository

    assert await workflow._aconversation_has_documents(str(__import__("uuid").uuid4())) is True
    assert repository.sync_calls == 0, "routing still used the blocking sync method"


@pytest.mark.asyncio
async def test_returns_false_without_a_repository():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.document_repository = None
    assert await workflow._aconversation_has_documents("whatever") is False


@pytest.mark.asyncio
async def test_returns_false_for_a_malformed_conversation_id():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.document_repository = _AsyncDocumentRepository(count=1)
    assert await workflow._aconversation_has_documents("not-a-uuid") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph_route_node_async_documents.py -v`
Expected: FAIL — `AttributeError: 'MultiAgentWorkflow' object has no attribute '_aconversation_has_documents'`

- [ ] **Step 3: Add the async helper and use it**

In `app/ai/graph.py`, directly after `_conversation_has_documents`:

```python
    async def _aconversation_has_documents(self, conversation_id: str | None) -> bool:
        """Async twin of :meth:`_conversation_has_documents`.

        Routing runs before the first token, so this query must not block the
        event loop and stall other in-flight streams.
        """
        if not conversation_id or not self.document_repository:
            return False
        try:
            return await self.document_repository.acount_by_conversation(UUID(conversation_id)) > 0
        except (ValueError, Exception):
            return False
```

Then in `_route_node`:

```python
-        has_documents = self._conversation_has_documents(conversation_id)
+        has_documents = await self._aconversation_has_documents(conversation_id)
```

Keep the sync method — non-streaming paths still call it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph_route_node_async_documents.py -v`
Expected: PASS (all 3)

- [ ] **Step 5: Verify routing tests still pass**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q -k "route or graph or stickiness"`
Expected: all pass. Existing router tests build `MultiAgentWorkflow` with
`document_repository = None`, which the new helper handles.

- [ ] **Step 6: Commit**

```bash
git add app/ai/graph.py tests/test_graph_route_node_async_documents.py
git commit -m "perf: make routing's document lookup non-blocking"
```

---

### Task 9: Full verification and phase handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-07-28-async-sqlalchemy-foundation.md` (this file — record results)

- [ ] **Step 1: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: every test passes except the known pre-existing
`test_alembic_full_chain_postgres.py::test_readme_tracks_migration_head_and_current_graph_contract`.
Any other failure blocks the phase.

- [ ] **Step 2: Lint and format the changed files**

```bash
.venv/Scripts/python.exe -m ruff check app/ tests/
.venv/Scripts/python.exe -m ruff format --check app/database app/repositories
```
Expected: no new findings versus the pre-existing baseline.

- [ ] **Step 3: Confirm Celery and Alembic still work on the sync engine**

```bash
.venv/Scripts/python.exe -c "from app.workers.celery_app import celery_app; print('celery imports ok')"
.venv/Scripts/python.exe -m alembic current
```
Expected: both succeed. This proves the sync engine was not disturbed.

- [ ] **Step 4: Verify the app boots and serves a request**

Start the app the documented way (selector-loop launch, see `app/main.py:440`)
and confirm `GET /health` responds, then send one chat message end-to-end and
confirm tokens still stream.

- [ ] **Step 5: Record the measured outcome in this file**

Append a "Results" section stating: full-suite result, whether the loop-stall
test passes, and observed behavior under two concurrent streams (previously one
stream's DB work delayed the other's first token).

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/plans/2026-07-28-async-sqlalchemy-foundation.md
git commit -m "docs: record async foundation phase 1 results"
```

---

## Phase 2 Preview (separate plan)

Phase 1 leaves 22 repositories sync-only and their async callers still blocking.
Phase 2 sweeps them module by module, one commit per module, in descending
traffic order. The mechanical recipe per repository is fixed by Tasks 4-5:
inherit `RepositorySessionMixin`, replace the `session_factory` assignment with
`super().__init__(...)`, add an `a`-prefixed twin per session-owning method
(168 session-owning methods exist across the codebase; Phase 1 converts 8), wire
`async_session_factory` in the container, then await the twins at the call
sites. Phase 3 removes sync twins that Phase 2 leaves unreferenced, keeping only
what Celery and Alembic use.

Suggested Phase 2 order, by request-path impact:
`agent_model_config`, `custom_agent`, `task_plan`, `hitl_interrupt`,
`tool_approval_setting`, `client_device`, `chat_image`, `user`, `feedback`,
`skill_setting`, `model_provider`, `document_chunk`, `document_image`,
`document_parse_artifact`, `tool_result_blob`, `user_memory`,
`conversation_compaction`, `model_usage`.
