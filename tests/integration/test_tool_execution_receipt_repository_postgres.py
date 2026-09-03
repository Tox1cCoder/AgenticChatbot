"""PostgreSQL integration test for ToolExecutionReceiptRepository.

Everything the receipt design rests on is a database guarantee, and none of it
is observable against a fake: the unique index on ``execution_key`` is what
makes two concurrent replays resolve to one provider call, and the
``WHERE status IN (...)`` clauses are what stop a recorded outcome from being
overwritten. A read-then-write repository, or a missing index, would pass every
unit test in ``test_tool_execution_receipt_service.py`` and duplicate real
effects in production.

Mirrors the ``session_factory``/``create_all`` pattern in
``test_tool_result_blob_repository_postgres.py``. The module-wide
``selector_event_loop`` marker is required, not cosmetic: async psycopg raises
``InterfaceError`` at connect time on Windows' default ``ProactorEventLoop``
(see ``pytest_asyncio_loop_factories`` in ``tests/conftest.py``).
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.models.base import Base
from app.models.conversation import Conversation
from app.models.tool_execution_receipt import ReceiptStatus, ToolExecutionReceipt
from app.models.user import User
from app.repositories.tool_execution_receipt import ToolExecutionReceiptRepository
from app.services.tool_execution_receipt_service import (
    MutationExecutionScope,
    execution_key,
)

pytestmark = pytest.mark.selector_event_loop

Seeded = tuple[ToolExecutionReceiptRepository, UUID, UUID, UUID, UUID, object]


def _async_url(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg://"):
        return database_url
    if database_url.startswith("postgresql+psycopg2://"):
        return database_url.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


@pytest.fixture(scope="module")
def engines():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL integration tests")
    engine = create_engine(database_url)
    Base.metadata.create_all(
        engine,
        tables=[User.__table__, Conversation.__table__, ToolExecutionReceipt.__table__],
    )
    async_engine = create_async_engine(_async_url(database_url))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    async_factory = async_sessionmaker(bind=async_engine, expire_on_commit=False)
    try:
        yield factory, async_factory
    finally:
        engine.dispose()
        if sys.platform == "win32":
            asyncio.run(async_engine.dispose(), loop_factory=asyncio.SelectorEventLoop)
        else:
            asyncio.run(async_engine.dispose())


@pytest.fixture()
def seeded(engines) -> Iterator[Seeded]:
    """Two independent owners, so an ANDed predicate is distinguishable."""
    session_factory, async_session_factory = engines
    owner_id = uuid4()
    other_user_id = uuid4()
    conversation_id = uuid4()
    other_conversation_id = uuid4()
    with session_factory.begin() as session:
        session.add(
            User(
                id=owner_id,
                username=f"owner-{owner_id}",
                email=f"{owner_id}@example.test",
                password_hash="test",
            )
        )
        session.add(
            User(
                id=other_user_id,
                username=f"other-{other_user_id}",
                email=f"{other_user_id}@example.test",
                password_hash="test",
            )
        )
        session.add(Conversation(id=conversation_id, owner_id=owner_id, title="receipt test"))
        session.add(
            Conversation(id=other_conversation_id, owner_id=other_user_id, title="unrelated")
        )

    repository = ToolExecutionReceiptRepository(
        session_factory=session_factory, async_session_factory=async_session_factory
    )
    yield (
        repository,
        owner_id,
        conversation_id,
        other_user_id,
        other_conversation_id,
        session_factory,
    )
    with session_factory.begin() as session:
        session.execute(
            delete(ToolExecutionReceipt).where(
                ToolExecutionReceipt.user_id.in_([owner_id, other_user_id])
            )
        )
        session.execute(
            delete(Conversation).where(
                Conversation.id.in_([conversation_id, other_conversation_id])
            )
        )
        session.execute(delete(User).where(User.id.in_([owner_id, other_user_id])))


def _scope(owner_id: UUID, conversation_id: UUID, **overrides) -> MutationExecutionScope:
    payload = {
        "thread_id": f"routing-v2:{conversation_id}:turn-1",
        "dispatch_id": "d1",
        "task_id": "w1",
        "tool_call_id": f"call-{uuid4()}",
        "tool_id": "mcp::write",
        "user_id": owner_id,
        "conversation_id": conversation_id,
        "turn_id": "turn-1",
    }
    payload.update(overrides)
    return MutationExecutionScope(**payload)


def _row(session_factory, key: str) -> ToolExecutionReceipt | None:
    with session_factory() as session:
        return (
            session.execute(
                select(ToolExecutionReceipt).where(ToolExecutionReceipt.execution_key == key)
            )
            .scalars()
            .first()
        )


async def test_a_fresh_key_is_reserved(seeded: Seeded) -> None:
    repository, owner_id, conversation_id, _, _, session_factory = seeded
    scope = _scope(owner_id, conversation_id)
    key = execution_key(scope)

    record = await repository.areserve(scope=scope, key=key)

    assert record.fresh is True
    assert record.status is ReceiptStatus.RESERVED
    assert _row(session_factory, key).status is ReceiptStatus.RESERVED


async def test_the_unique_key_makes_a_second_reservation_read_the_first(seeded: Seeded) -> None:
    """Fails if the unique index is missing: both callers would insert."""
    repository, owner_id, conversation_id, _, _, session_factory = seeded
    scope = _scope(owner_id, conversation_id)
    key = execution_key(scope)

    await repository.areserve(scope=scope, key=key)
    second = await repository.areserve(scope=scope, key=key)

    assert second.fresh is False
    assert second.status is ReceiptStatus.RESERVED
    with session_factory() as session:
        rows = (
            session.execute(
                select(ToolExecutionReceipt).where(ToolExecutionReceipt.execution_key == key)
            )
            .scalars()
            .all()
        )
    assert len(rows) == 1


async def test_concurrent_reservations_produce_exactly_one_winner(seeded: Seeded) -> None:
    repository, owner_id, conversation_id, _, _, _ = seeded
    scope = _scope(owner_id, conversation_id)
    key = execution_key(scope)

    records = await asyncio.gather(*(repository.areserve(scope=scope, key=key) for _ in range(4)))

    assert sum(1 for record in records if record.fresh) == 1


async def test_a_completed_receipt_returns_its_recorded_result(seeded: Seeded) -> None:
    repository, owner_id, conversation_id, _, _, _ = seeded
    scope = _scope(owner_id, conversation_id)
    key = execution_key(scope)

    await repository.areserve(scope=scope, key=key)
    await repository.acomplete(
        key=key,
        result={"content": "created", "artifact_ref": "blob:1"},
        provider_receipt_id="prov-1",
    )
    record = await repository.areserve(scope=scope, key=key)

    assert record.status is ReceiptStatus.COMPLETED
    assert record.result == {"content": "created", "artifact_ref": "blob:1"}
    assert record.provider_receipt_id == "prov-1"


async def test_a_terminal_receipt_is_never_overwritten(seeded: Seeded) -> None:
    """Fails if a transition is a read-then-write instead of a compare-and-set."""
    repository, owner_id, conversation_id, _, _, session_factory = seeded
    scope = _scope(owner_id, conversation_id)
    key = execution_key(scope)

    await repository.areserve(scope=scope, key=key)
    await repository.acomplete(key=key, result={"content": "created"}, provider_receipt_id=None)
    await repository.afail(key=key, error_code="tool_execution_failed")
    await repository.amark_outcome_unknown(key=key)

    row = _row(session_factory, key)
    assert row.status is ReceiptStatus.COMPLETED
    assert row.error_code is None


async def test_a_receipt_owned_by_another_user_is_not_readable(seeded: Seeded) -> None:
    """A receipt is not a global cache; a foreign row must not answer a call."""
    repository, owner_id, conversation_id, other_user_id, other_conversation_id, _ = seeded
    owner_scope = _scope(owner_id, conversation_id)
    key = execution_key(owner_scope)

    await repository.areserve(scope=owner_scope, key=key)
    await repository.acomplete(
        key=key, result={"content": "someone elses effect"}, provider_receipt_id=None
    )

    intruder_scope = owner_scope.model_copy(
        update={"user_id": other_user_id, "conversation_id": other_conversation_id}
    )
    record = await repository.areserve(scope=intruder_scope, key=key)

    assert record.fresh is True
    assert record.result is None


async def test_outcome_unknown_is_listed_for_reconciliation(seeded: Seeded) -> None:
    repository, owner_id, conversation_id, _, _, _ = seeded
    scope = _scope(owner_id, conversation_id)
    key = execution_key(scope)

    await repository.areserve(scope=scope, key=key)
    await repository.amark_outcome_unknown(key=key)

    unresolved = await repository.alist_unresolved(user_id=owner_id)
    assert [row.execution_key for row in unresolved] == [key]


async def test_a_bounded_result_is_stored_and_read_back(seeded: Seeded) -> None:
    repository, owner_id, conversation_id, _, _, _ = seeded
    scope = _scope(owner_id, conversation_id)
    key = execution_key(scope)

    await repository.areserve(scope=scope, key=key)
    await repository.acomplete(key=key, result={"content": "x" * 4000}, provider_receipt_id=None)
    record = await repository.areserve(scope=scope, key=key)

    assert len(record.result["content"]) == 4000
