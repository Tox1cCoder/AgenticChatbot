"""PostgreSQL integration test for GenerationRepository.

Everything the lifecycle rests on is a database guarantee, and none of it is
observable against a fake:

* ``WHERE version = :expected`` is what makes two concurrent Continues on one
  paused turn produce exactly one epoch increment. A read-then-write repository
  would pass every unit test and answer one question twice in production.
* the partial unique index is what stops a conversation running two turns at
  once — and, just as importantly, what lets a *paused* turn coexist with the
  next question.
* the command ledger's unique index is what makes a retried Stop return the
  first Stop's recorded result rather than stopping whatever is running when it
  lands.

Mirrors the ``session_factory``/``create_all`` pattern in
``test_tool_execution_receipt_repository_postgres.py``. The module-wide
``selector_event_loop`` marker is required, not cosmetic: async psycopg raises
``InterfaceError`` at connect time on Windows' default ``ProactorEventLoop``
(see ``pytest_asyncio_loop_factories`` in ``tests/conftest.py``).
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.producer_identity import current_producer_token
from app.models.base import Base
from app.models.conversation import Conversation
from app.models.generation import (
    Generation,
    GenerationCommand,
    GenerationCommandAction,
    GenerationStatus,
)
from app.models.user import User
from app.repositories.generation import GenerationRepository
from app.schemas.generation import CreateGeneration
from app.services.generation_reaper import (
    reclaim_orphaned_generations,
    terminalize_own_generations,
)

pytestmark = pytest.mark.selector_event_loop


@dataclass
class Seeded:
    repository: GenerationRepository
    owner_id: UUID
    conversation_id: UUID
    other_user_id: UUID
    other_conversation_id: UUID
    session_factory: object


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
        tables=[
            User.__table__,
            Conversation.__table__,
            Generation.__table__,
            GenerationCommand.__table__,
        ],
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
        for user_id, label in ((owner_id, "owner"), (other_user_id, "other")):
            session.add(
                User(
                    id=user_id,
                    username=f"{label}-{user_id}",
                    email=f"{user_id}@example.test",
                    password_hash="test",
                )
            )
        session.add(Conversation(id=conversation_id, owner_id=owner_id, title="generation test"))
        session.add(
            Conversation(id=other_conversation_id, owner_id=other_user_id, title="unrelated")
        )

    repository = GenerationRepository(
        session_factory=session_factory, async_session_factory=async_session_factory
    )
    try:
        yield Seeded(
            repository=repository,
            owner_id=owner_id,
            conversation_id=conversation_id,
            other_user_id=other_user_id,
            other_conversation_id=other_conversation_id,
            session_factory=session_factory,
        )
    finally:
        with session_factory.begin() as session:
            session.execute(
                delete(GenerationCommand).where(
                    GenerationCommand.generation_id.in_(
                        select(Generation.id).where(
                            Generation.user_id.in_([owner_id, other_user_id])
                        )
                    )
                )
            )
            session.execute(
                delete(Generation).where(Generation.user_id.in_([owner_id, other_user_id]))
            )
            session.execute(
                delete(Conversation).where(
                    Conversation.id.in_([conversation_id, other_conversation_id])
                )
            )
            session.execute(delete(User).where(User.id.in_([owner_id, other_user_id])))


def _create(seeded: Seeded, *, turn: str = "turn-1", **overrides) -> CreateGeneration:
    values: dict = {
        "conversation_id": seeded.conversation_id,
        "user_id": seeded.owner_id,
        "logical_turn_id": turn,
        "checkpoint_thread_id": f"routing-v2:{turn}",
        "active_agent_id": "chat_agent",
    }
    values.update(overrides)
    return CreateGeneration(**values)


# ----------------------------------------------------------------------
# creation and owner scoping
# ----------------------------------------------------------------------


async def test_a_new_generation_starts_at_version_one_and_epoch_zero(seeded: Seeded) -> None:
    snapshot = await seeded.repository.acreate(_create(seeded))

    assert snapshot.status is GenerationStatus.STARTING
    assert snapshot.version == 1
    assert snapshot.execution_epoch == 0
    assert snapshot.continuation_available is False


async def test_deleting_the_assistant_message_releases_the_generation(seeded: Seeded) -> None:
    """A generation row must not be able to veto a message deletion.

    Created with a plain foreign key, ``DELETE FROM messages`` raised
    ``ForeignKeyViolation`` for any message a generation referenced — the
    bookkeeping outranking the thing it books. ``ON DELETE SET NULL`` keeps the
    lifecycle history, where ``CASCADE`` would silently destroy it.
    """
    from sqlalchemy import delete as sql_delete

    from app.models.enums import MessageRole
    from app.models.message import Message

    snapshot = await seeded.repository.acreate(_create(seeded, turn="turn-fk"))
    message_id = uuid4()
    with seeded.session_factory() as session:  # type: ignore[operator]
        session.add(
            Message(
                id=message_id,
                conversation_id=seeded.conversation_id,
                sender=MessageRole.assistant,
                content="a partial answer",
                sequence=1,
            )
        )
        session.commit()
        session.execute(
            Generation.__table__.update()
            .where(Generation.id == snapshot.generation_id)
            .values(assistant_message_id=message_id)
        )
        session.commit()

        session.execute(sql_delete(Message).where(Message.id == message_id))
        session.commit()

        remaining = session.execute(
            select(Generation.id, Generation.assistant_message_id).where(
                Generation.id == snapshot.generation_id
            )
        ).one()

    assert remaining.id == snapshot.generation_id
    assert remaining.assistant_message_id is None


async def test_a_generation_is_reachable_by_its_logical_turn(seeded: Seeded) -> None:
    """The route a caller holding only a turn id takes.

    An older Stop endpoint carries the user message id, which is the logical
    turn. Resolving it to "whatever is active in this conversation" instead
    would let a delayed Stop cancel a turn it was never issued against.
    """
    snapshot = await seeded.repository.acreate(_create(seeded, turn="turn-lookup"))

    found = await seeded.repository.aget_by_logical_turn(
        "turn-lookup", seeded.owner_id, seeded.conversation_id
    )

    assert found is not None
    assert found.generation_id == snapshot.generation_id


async def test_a_logical_turn_lookup_is_owner_and_conversation_scoped(seeded: Seeded) -> None:
    await seeded.repository.acreate(_create(seeded, turn="turn-scoped"))

    assert (
        await seeded.repository.aget_by_logical_turn(
            "turn-scoped", seeded.other_user_id, seeded.conversation_id
        )
        is None
    )
    assert (
        await seeded.repository.aget_by_logical_turn(
            "turn-scoped", seeded.owner_id, seeded.other_conversation_id
        )
        is None
    )


async def test_an_unknown_logical_turn_is_not_resolved_to_a_neighbour(seeded: Seeded) -> None:
    """A stale turn id must find nothing, not the turn running now."""
    await seeded.repository.acreate(_create(seeded, turn="turn-current"))

    assert (
        await seeded.repository.aget_by_logical_turn(
            "turn-that-never-existed", seeded.owner_id, seeded.conversation_id
        )
        is None
    )


async def test_a_generation_is_not_readable_by_another_user(seeded: Seeded) -> None:
    snapshot = await seeded.repository.acreate(_create(seeded))

    assert (
        await seeded.repository.aget_owned(
            snapshot.generation_id, seeded.other_user_id, seeded.conversation_id
        )
        is None
    )


async def test_a_generation_is_not_readable_through_another_conversation(
    seeded: Seeded,
) -> None:
    """The id is public; the conversation predicate is what makes it safe."""
    snapshot = await seeded.repository.acreate(_create(seeded))

    assert (
        await seeded.repository.aget_owned(
            snapshot.generation_id, seeded.owner_id, seeded.other_conversation_id
        )
        is None
    )


async def test_the_logical_turn_is_claimed_once(seeded: Seeded) -> None:
    await seeded.repository.acreate(_create(seeded, turn="turn-unique"))

    with pytest.raises(IntegrityError):
        await seeded.repository.acreate(
            _create(
                seeded,
                turn="turn-unique",
                conversation_id=seeded.other_conversation_id,
                user_id=seeded.other_user_id,
            )
        )


async def test_a_conversation_cannot_run_two_turns_at_once(seeded: Seeded) -> None:
    await seeded.repository.acreate(_create(seeded, turn="turn-a"))

    with pytest.raises(IntegrityError):
        await seeded.repository.acreate(_create(seeded, turn="turn-b"))


async def test_a_paused_turn_does_not_block_the_next_question(seeded: Seeded) -> None:
    """``continuable`` sits outside the partial index on purpose.

    A paused turn holds no worker. If it kept the conversation locked, a user
    who ignored Continue could never ask anything else.
    """
    first = await seeded.repository.acreate(_create(seeded, turn="turn-paused"))
    paused = await seeded.repository.atransition(
        generation_id=first.generation_id,
        user_id=seeded.owner_id,
        conversation_id=seeded.conversation_id,
        expected_statuses=(GenerationStatus.STARTING,),
        expected_version=first.version,
        values={
            "status": GenerationStatus.CONTINUABLE,
            "continuation_id": uuid4(),
            "continuation_available": True,
        },
    )
    assert paused is not None

    second = await seeded.repository.acreate(_create(seeded, turn="turn-next"))

    assert second.status is GenerationStatus.STARTING


# ----------------------------------------------------------------------
# compare-and-set transitions
# ----------------------------------------------------------------------


async def test_a_transition_increments_the_version(seeded: Seeded) -> None:
    created = await seeded.repository.acreate(_create(seeded))

    running = await seeded.repository.atransition(
        generation_id=created.generation_id,
        user_id=seeded.owner_id,
        conversation_id=seeded.conversation_id,
        expected_statuses=(GenerationStatus.STARTING,),
        expected_version=created.version,
        values={"status": GenerationStatus.RUNNING},
    )

    assert running is not None
    assert running.version == created.version + 1
    assert running.status is GenerationStatus.RUNNING


async def test_a_stale_version_is_refused_rather_than_applied(seeded: Seeded) -> None:
    created = await seeded.repository.acreate(_create(seeded))
    await seeded.repository.atransition(
        generation_id=created.generation_id,
        user_id=seeded.owner_id,
        conversation_id=seeded.conversation_id,
        expected_statuses=(GenerationStatus.STARTING,),
        expected_version=created.version,
        values={"status": GenerationStatus.RUNNING},
    )

    late = await seeded.repository.atransition(
        generation_id=created.generation_id,
        user_id=seeded.owner_id,
        conversation_id=seeded.conversation_id,
        expected_statuses=(GenerationStatus.STARTING,),
        expected_version=created.version,
        values={"status": GenerationStatus.STOPPED},
    )

    assert late is None


async def test_a_transition_from_the_wrong_status_is_refused(seeded: Seeded) -> None:
    created = await seeded.repository.acreate(_create(seeded))

    refused = await seeded.repository.atransition(
        generation_id=created.generation_id,
        user_id=seeded.owner_id,
        conversation_id=seeded.conversation_id,
        expected_statuses=(GenerationStatus.CONTINUABLE,),
        expected_version=created.version,
        values={"status": GenerationStatus.CONTINUING},
    )

    assert refused is None


async def test_another_users_transition_cannot_touch_the_row(seeded: Seeded) -> None:
    created = await seeded.repository.acreate(_create(seeded))

    refused = await seeded.repository.atransition(
        generation_id=created.generation_id,
        user_id=seeded.other_user_id,
        conversation_id=seeded.conversation_id,
        expected_statuses=(GenerationStatus.STARTING,),
        expected_version=created.version,
        values={"status": GenerationStatus.STOPPED},
    )

    assert refused is None


async def test_concurrent_continues_produce_exactly_one_epoch_increment(
    seeded: Seeded,
) -> None:
    """The property the whole table exists for.

    Two Continues on one paused turn must not both advance the epoch: that
    would run the specialist twice and answer one question twice.
    """
    created = await seeded.repository.acreate(_create(seeded, turn="turn-race"))
    paused = await seeded.repository.atransition(
        generation_id=created.generation_id,
        user_id=seeded.owner_id,
        conversation_id=seeded.conversation_id,
        expected_statuses=(GenerationStatus.STARTING,),
        expected_version=created.version,
        values={
            "status": GenerationStatus.CONTINUABLE,
            "continuation_id": uuid4(),
            "continuation_available": True,
        },
    )
    assert paused is not None

    async def attempt():
        return await seeded.repository.atransition(
            generation_id=created.generation_id,
            user_id=seeded.owner_id,
            conversation_id=seeded.conversation_id,
            expected_statuses=(GenerationStatus.CONTINUABLE,),
            expected_version=paused.version,
            values={
                "status": GenerationStatus.CONTINUING,
                "execution_epoch": paused.execution_epoch + 1,
                "continuation_available": False,
            },
        )

    results = await asyncio.gather(attempt(), attempt(), return_exceptions=True)
    winners = [item for item in results if isinstance(item, object) and item is not None]
    winners = [item for item in winners if not isinstance(item, BaseException)]

    assert len(winners) == 1
    assert winners[0].execution_epoch == paused.execution_epoch + 1


async def test_the_version_is_owned_by_the_transition_not_the_caller(seeded: Seeded) -> None:
    created = await seeded.repository.acreate(_create(seeded))

    with pytest.raises(ValueError, match="version"):
        await seeded.repository.atransition(
            generation_id=created.generation_id,
            user_id=seeded.owner_id,
            conversation_id=seeded.conversation_id,
            expected_statuses=(GenerationStatus.STARTING,),
            expected_version=created.version,
            values={"status": GenerationStatus.RUNNING, "version": 99},
        )


async def test_the_active_lookup_finds_the_running_turn(seeded: Seeded) -> None:
    created = await seeded.repository.acreate(_create(seeded))

    active = await seeded.repository.aget_active_for_conversation(
        seeded.conversation_id, seeded.owner_id
    )

    assert active is not None
    assert active.generation_id == created.generation_id


async def test_the_active_lookup_ignores_a_finished_turn(seeded: Seeded) -> None:
    created = await seeded.repository.acreate(_create(seeded))
    await seeded.repository.atransition(
        generation_id=created.generation_id,
        user_id=seeded.owner_id,
        conversation_id=seeded.conversation_id,
        expected_statuses=(GenerationStatus.STARTING,),
        expected_version=created.version,
        values={"status": GenerationStatus.COMPLETED},
    )

    assert (
        await seeded.repository.aget_active_for_conversation(
            seeded.conversation_id, seeded.owner_id
        )
        is None
    )


# ----------------------------------------------------------------------
# command ledger
# ----------------------------------------------------------------------


async def test_a_fresh_command_is_claimed(seeded: Seeded) -> None:
    created = await seeded.repository.acreate(_create(seeded))

    claim = await seeded.repository.aclaim_command(
        generation_id=created.generation_id,
        idempotency_key="stop-key-0001",
        action=GenerationCommandAction.STOP,
        fence=created.version,
    )

    assert claim.claimed is True
    assert claim.result is None


async def test_a_replayed_command_returns_the_first_result(seeded: Seeded) -> None:
    created = await seeded.repository.acreate(_create(seeded))
    await seeded.repository.aclaim_command(
        generation_id=created.generation_id,
        idempotency_key="stop-key-0002",
        action=GenerationCommandAction.STOP,
        fence=created.version,
    )
    await seeded.repository.arecord_command_result(
        generation_id=created.generation_id,
        idempotency_key="stop-key-0002",
        result={"status": "stopped"},
    )

    replay = await seeded.repository.aclaim_command(
        generation_id=created.generation_id,
        idempotency_key="stop-key-0002",
        action=GenerationCommandAction.STOP,
        fence=created.version,
    )

    assert replay.claimed is False
    assert replay.result == {"status": "stopped"}


async def test_a_replay_reports_the_fence_the_command_was_issued_against(
    seeded: Seeded,
) -> None:
    """R5: a delayed Stop must be recognisable as belonging to an old epoch.

    The ledger records the fence the *client* issued against, not the row's
    version now. That is what lets the service refuse the replay as stale
    instead of cancelling whatever epoch happens to be running.
    """
    created = await seeded.repository.acreate(_create(seeded))
    await seeded.repository.aclaim_command(
        generation_id=created.generation_id,
        idempotency_key="stop-key-0003",
        action=GenerationCommandAction.STOP,
        fence=1,
    )
    for expected, values in (
        (1, {"status": GenerationStatus.RUNNING}),
        (2, {"status": GenerationStatus.CONTINUABLE}),
    ):
        await seeded.repository.atransition(
            generation_id=created.generation_id,
            user_id=seeded.owner_id,
            conversation_id=seeded.conversation_id,
            expected_statuses=(
                GenerationStatus.STARTING,
                GenerationStatus.RUNNING,
            ),
            expected_version=expected,
            values=values,
        )

    replay = await seeded.repository.aclaim_command(
        generation_id=created.generation_id,
        idempotency_key="stop-key-0003",
        action=GenerationCommandAction.STOP,
        fence=3,
    )

    assert replay.claimed is False
    assert replay.fence == 1


async def test_concurrent_claims_of_one_key_produce_one_winner(seeded: Seeded) -> None:
    created = await seeded.repository.acreate(_create(seeded))

    async def claim():
        return await seeded.repository.aclaim_command(
            generation_id=created.generation_id,
            idempotency_key="stop-key-0004",
            action=GenerationCommandAction.STOP,
            fence=created.version,
        )

    results = await asyncio.gather(claim(), claim(), return_exceptions=True)
    claims = [item for item in results if not isinstance(item, BaseException)]

    assert len([item for item in claims if item.claimed]) == 1


async def test_a_recorded_result_is_never_overwritten(seeded: Seeded) -> None:
    created = await seeded.repository.acreate(_create(seeded))
    await seeded.repository.aclaim_command(
        generation_id=created.generation_id,
        idempotency_key="stop-key-0005",
        action=GenerationCommandAction.STOP,
        fence=created.version,
    )
    await seeded.repository.arecord_command_result(
        generation_id=created.generation_id,
        idempotency_key="stop-key-0005",
        result={"status": "stopped"},
    )
    await seeded.repository.arecord_command_result(
        generation_id=created.generation_id,
        idempotency_key="stop-key-0005",
        result={"status": "something-else"},
    )

    replay = await seeded.repository.aclaim_command(
        generation_id=created.generation_id,
        idempotency_key="stop-key-0005",
        action=GenerationCommandAction.STOP,
        fence=created.version,
    )

    assert replay.result == {"status": "stopped"}


async def test_two_generations_may_share_one_idempotency_key(seeded: Seeded) -> None:
    """The key is scoped to its generation, not global."""
    first = await seeded.repository.acreate(_create(seeded, turn="turn-key-a"))
    await seeded.repository.atransition(
        generation_id=first.generation_id,
        user_id=seeded.owner_id,
        conversation_id=seeded.conversation_id,
        expected_statuses=(GenerationStatus.STARTING,),
        expected_version=first.version,
        values={"status": GenerationStatus.COMPLETED},
    )
    second = await seeded.repository.acreate(_create(seeded, turn="turn-key-b"))

    for generation_id in (first.generation_id, second.generation_id):
        claim = await seeded.repository.aclaim_command(
            generation_id=generation_id,
            idempotency_key="shared-key-0001",
            action=GenerationCommandAction.CONTINUE,
            fence=1,
        )
        assert claim.claimed is True


# ----------------------------------------------------------------------
# reclaiming a turn whose worker is gone
# ----------------------------------------------------------------------
#
# The partial unique index admits one active row per conversation, which is
# what stops two turns running at once. Its cost is that a row left active by a
# worker that died -- a crash, a hard kill, or uvicorn's reloader replacing the
# child on a code change -- blocks that conversation permanently: every later
# turn's INSERT raises UniqueViolation and the API reports
# ``conversation_turn_conflict``, which no retry can clear because no worker
# exists to finish the turn.
#
# What makes reclaiming safe rather than reckless is that it is scoped to a
# named producer. "Fail everything active at startup" would be correct only
# under an assumption this application refuses to encode -- that there is one
# worker -- and would otherwise terminalize a peer's streaming turn.


def _dead_producer_token() -> str:
    """A token for a process that is provably not running on this host."""
    import psutil

    for candidate in range(2**22, 2**22 + 5_000):
        if not psutil.pid_exists(candidate):
            return f"{socket.gethostname()}:{candidate}:1"
    raise AssertionError("no unused pid found in the probed range")


def _stamp(seeded: Seeded, generation_id: UUID, token: str | None) -> None:
    """Write a producer directly, standing in for another worker's INSERT."""
    with seeded.session_factory.begin() as session:
        session.execute(
            update(Generation).where(Generation.id == generation_id).values(producer_token=token)
        )


def _read(seeded: Seeded, generation_id: UUID) -> Generation:
    with seeded.session_factory() as session:
        row = session.get(Generation, generation_id)
        assert row is not None
        return row


async def test_a_new_generation_records_the_process_producing_it(seeded: Seeded) -> None:
    snapshot = await seeded.repository.acreate(_create(seeded, turn="turn-producer"))

    assert _read(seeded, snapshot.generation_id).producer_token == current_producer_token()


async def test_the_active_producers_are_what_the_reaper_reads(seeded: Seeded) -> None:
    """Only active rows matter: a terminal row blocks nothing."""
    live = await seeded.repository.acreate(_create(seeded, turn="turn-live"))
    _stamp(seeded, live.generation_id, "host-a:11:1")
    done = await seeded.repository.acreate(
        _create(
            seeded,
            turn="turn-done",
            conversation_id=seeded.other_conversation_id,
            user_id=seeded.other_user_id,
        )
    )
    _stamp(seeded, done.generation_id, "host-b:22:1")
    await seeded.repository.atransition(
        generation_id=done.generation_id,
        user_id=seeded.other_user_id,
        conversation_id=seeded.other_conversation_id,
        expected_statuses=(GenerationStatus.STARTING,),
        expected_version=done.version,
        values={"status": GenerationStatus.COMPLETED},
    )

    tokens = await seeded.repository.aget_active_producer_tokens()

    assert "host-a:11:1" in tokens
    assert "host-b:22:1" not in tokens


async def test_failing_a_producers_generations_terminalizes_them(seeded: Seeded) -> None:
    orphan = await seeded.repository.acreate(_create(seeded, turn="turn-orphan"))
    _stamp(seeded, orphan.generation_id, "gone-host:99:1")

    reclaimed = await seeded.repository.afail_active_by_producer(
        ["gone-host:99:1"], terminal_reason="producer_lost"
    )

    row = _read(seeded, orphan.generation_id)
    assert reclaimed == 1
    assert row.status is GenerationStatus.FAILED
    assert row.terminal_reason == "producer_lost"
    assert row.terminal_at is not None


async def test_reclaiming_bumps_the_version_so_a_command_against_it_is_stale(
    seeded: Seeded,
) -> None:
    """A client holding the old fence must be refused, not silently applied."""
    orphan = await seeded.repository.acreate(_create(seeded, turn="turn-fence"))
    _stamp(seeded, orphan.generation_id, "gone-host:98:1")

    await seeded.repository.afail_active_by_producer(
        ["gone-host:98:1"], terminal_reason="producer_lost"
    )

    assert _read(seeded, orphan.generation_id).version == orphan.version + 1


async def test_failing_one_producer_leaves_another_producers_turn_running(
    seeded: Seeded,
) -> None:
    """The whole point of naming the producer: a peer's turn is untouched."""
    mine = await seeded.repository.acreate(_create(seeded, turn="turn-mine"))
    _stamp(seeded, mine.generation_id, "host-mine:1:1")
    theirs = await seeded.repository.acreate(
        _create(
            seeded,
            turn="turn-theirs",
            conversation_id=seeded.other_conversation_id,
            user_id=seeded.other_user_id,
        )
    )
    _stamp(seeded, theirs.generation_id, "host-theirs:2:1")

    await seeded.repository.afail_active_by_producer(
        ["host-mine:1:1"], terminal_reason="producer_lost"
    )

    assert _read(seeded, mine.generation_id).status is GenerationStatus.FAILED
    assert _read(seeded, theirs.generation_id).status is GenerationStatus.STARTING


async def test_failing_a_producer_leaves_its_paused_turn_alone(seeded: Seeded) -> None:
    """``continuable`` holds no worker and blocks nothing, so it is not an orphan.

    Terminalizing it would destroy a turn the user can still continue.
    """
    paused = await seeded.repository.acreate(_create(seeded, turn="turn-paused"))
    _stamp(seeded, paused.generation_id, "gone-host:97:1")
    await seeded.repository.atransition(
        generation_id=paused.generation_id,
        user_id=seeded.owner_id,
        conversation_id=seeded.conversation_id,
        expected_statuses=(GenerationStatus.STARTING,),
        expected_version=paused.version,
        values={"status": GenerationStatus.CONTINUABLE, "continuation_available": True},
    )

    reclaimed = await seeded.repository.afail_active_by_producer(
        ["gone-host:97:1"], terminal_reason="producer_lost"
    )

    assert reclaimed == 0
    assert _read(seeded, paused.generation_id).status is GenerationStatus.CONTINUABLE


async def test_a_conversation_can_start_a_new_turn_once_its_orphan_is_reclaimed(
    seeded: Seeded,
) -> None:
    """The bug this exists for, asserted end to end.

    Before reclaiming, the conversation's next question cannot even be
    inserted; afterwards it can.
    """
    orphan = await seeded.repository.acreate(_create(seeded, turn="turn-blocking"))
    _stamp(seeded, orphan.generation_id, "gone-host:96:1")

    with pytest.raises(IntegrityError):
        await seeded.repository.acreate(_create(seeded, turn="turn-blocked"))

    await seeded.repository.afail_active_by_producer(
        ["gone-host:96:1"], terminal_reason="producer_lost"
    )

    revived = await seeded.repository.acreate(_create(seeded, turn="turn-unblocked"))
    assert revived.status is GenerationStatus.STARTING


# ----------------------------------------------------------------------
# the reaper's policy
# ----------------------------------------------------------------------


async def test_the_startup_sweep_reclaims_a_turn_whose_worker_is_gone(seeded: Seeded) -> None:
    orphan = await seeded.repository.acreate(_create(seeded, turn="turn-sweep-dead"))
    _stamp(seeded, orphan.generation_id, _dead_producer_token())

    reclaimed = await reclaim_orphaned_generations(seeded.repository)

    assert reclaimed == 1
    assert _read(seeded, orphan.generation_id).terminal_reason == "producer_lost"


async def test_the_startup_sweep_leaves_this_processes_own_turn_running(seeded: Seeded) -> None:
    """This process is alive by definition, so its rows are not orphans."""
    mine = await seeded.repository.acreate(_create(seeded, turn="turn-sweep-mine"))

    reclaimed = await reclaim_orphaned_generations(seeded.repository)

    assert reclaimed == 0
    assert _read(seeded, mine.generation_id).status is GenerationStatus.STARTING


async def test_the_startup_sweep_leaves_a_turn_produced_on_another_host(
    seeded: Seeded,
) -> None:
    """This host cannot see that process; guessing would reap a live turn."""
    remote = await seeded.repository.acreate(_create(seeded, turn="turn-sweep-remote"))
    _stamp(seeded, remote.generation_id, f"not-{socket.gethostname()}:1234:1")

    reclaimed = await reclaim_orphaned_generations(seeded.repository)

    assert reclaimed == 0
    assert _read(seeded, remote.generation_id).status is GenerationStatus.STARTING


async def test_the_startup_sweep_leaves_a_turn_that_names_no_producer(seeded: Seeded) -> None:
    """Rows written before the column existed are unknown, not dead."""
    legacy = await seeded.repository.acreate(_create(seeded, turn="turn-sweep-legacy"))
    _stamp(seeded, legacy.generation_id, None)

    reclaimed = await reclaim_orphaned_generations(seeded.repository)

    assert reclaimed == 0
    assert _read(seeded, legacy.generation_id).status is GenerationStatus.STARTING


async def test_shutdown_terminalizes_this_workers_own_active_turn(seeded: Seeded) -> None:
    """A reload kills the child, so the row is failed before the process goes.

    Recorded as ``producer_shutdown`` rather than ``producer_lost``: the
    process knew it was leaving, which is a different fact from a crash.
    """
    mine = await seeded.repository.acreate(_create(seeded, turn="turn-shutdown"))

    terminalized = await terminalize_own_generations(seeded.repository)

    row = _read(seeded, mine.generation_id)
    assert terminalized == 1
    assert row.status is GenerationStatus.FAILED
    assert row.terminal_reason == "producer_shutdown"


async def test_shutdown_leaves_another_workers_turn_alone(seeded: Seeded) -> None:
    theirs = await seeded.repository.acreate(_create(seeded, turn="turn-shutdown-peer"))
    _stamp(seeded, theirs.generation_id, "peer-host:5:1")

    terminalized = await terminalize_own_generations(seeded.repository)

    assert terminalized == 0
    assert _read(seeded, theirs.generation_id).status is GenerationStatus.STARTING
