"""Concurrent control commands against a real PostgreSQL.

Every guarantee here is a database guarantee, and none of it is observable
against a fake. A read-then-write repository passes every unit test and still
answers one question twice in production:

* two Continues on one paused turn would both observe ``continuable``, both
  pass the legality check, and both increment the epoch — two answers to one
  question, and two assistant messages;
* two Stops would both transition and both publish, so a worker could be
  cancelled twice and the second command would report against a version the
  first had already moved;
* a Stop racing a Continue must produce exactly one winner, and the loser must
  be told it lost rather than silently applied to whatever state resulted.

Two predicates settle these, and it is worth knowing which does what, because
the tests below were briefly wrong about it. ``atransition`` filters on
*status* as well as ``version``, and for most races the status predicate is
what refuses the loser: the winner's transition moves the row out of the set
the loser's command declared, so the second ``UPDATE`` matches nothing. The
version fence matters where it does not — a status that legitimately allows the
same command twice, of which ``stop_requested`` is the example, since a Stop
may be issued from it. ``test_the_version_fence_alone_refuses_a_second_stop``
isolates that case, and it is the one test here that fails if the fence is
removed.

``RETURNING`` is what tells the winner what it won. The command ledger's unique
index is what makes a *replay* return the first attempt's recorded result
instead of executing again.

Sessions are separate on purpose. Two coroutines sharing one SQLAlchemy session
serialize through that session and would show a race that cannot happen and
hide one that can.

The module-wide ``selector_event_loop`` marker is required, not cosmetic: async
psycopg raises ``InterfaceError`` at connect time on Windows' default
``ProactorEventLoop``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.models.base import Base
from app.models.conversation import Conversation
from app.models.enums import MessageRole
from app.models.generation import Generation, GenerationCommand, GenerationStatus
from app.models.message import Message
from app.models.user import User
from app.repositories.generation import GenerationRepository
from app.schemas.generation import (
    ContinueGenerationCommand,
    CreateGeneration,
    MarkContinuable,
    StopGenerationCommand,
)
from app.services.generation_control_bus import InMemoryGenerationControlBus
from app.services.generation_control_service import (
    ContinuationUnavailable,
    GenerationControlError,
    GenerationControlService,
)

pytestmark = pytest.mark.selector_event_loop


@dataclass
class Seeded:
    owner_id: UUID
    conversation_id: UUID
    #: A real row, because ``MarkContinuable.assistant_message_id`` is required
    #: and carries a foreign key. That is the schema enforcing persist-before-
    #: offer: there is no such thing as a continuable turn whose partial answer
    #: was never written down.
    assistant_message_id: UUID
    session_factory: object
    async_session_factory: object


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
            Message.__table__,
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
    session_factory, async_session_factory = engines
    owner_id = uuid4()
    conversation_id = uuid4()
    assistant_message_id = uuid4()
    with session_factory.begin() as session:
        session.add(
            User(
                id=owner_id,
                username=f"racer-{owner_id}",
                email=f"{owner_id}@example.test",
                password_hash="test",
            )
        )
        session.add(Conversation(id=conversation_id, owner_id=owner_id, title="race test"))
        session.flush()
        session.add(
            Message(
                id=assistant_message_id,
                conversation_id=conversation_id,
                sender=MessageRole.assistant,
                content="a validated partial answer",
                sequence=1,
            )
        )

    try:
        yield Seeded(
            owner_id=owner_id,
            conversation_id=conversation_id,
            assistant_message_id=assistant_message_id,
            session_factory=session_factory,
            async_session_factory=async_session_factory,
        )
    finally:
        with session_factory.begin() as session:
            session.execute(
                delete(GenerationCommand).where(
                    GenerationCommand.generation_id.in_(
                        select(Generation.id).where(Generation.user_id == owner_id)
                    )
                )
            )
            session.execute(delete(Generation).where(Generation.user_id == owner_id))
            session.execute(delete(Message).where(Message.conversation_id == conversation_id))
            session.execute(delete(Conversation).where(Conversation.id == conversation_id))
            session.execute(delete(User).where(User.id == owner_id))


def _service(seeded: Seeded) -> GenerationControlService:
    """A control service over its *own* repository handle.

    Each racer gets one of these, so the two commands reach PostgreSQL as
    genuinely concurrent statements rather than being serialized by a shared
    session.
    """
    repository = GenerationRepository(
        session_factory=seeded.session_factory,
        async_session_factory=seeded.async_session_factory,
    )
    return GenerationControlService(
        repository=repository, bus=InMemoryGenerationControlBus(), stop_wait_seconds=0.05
    )


async def _paused(seeded: Seeded, *, turn: str) -> tuple[GenerationControlService, object]:
    """A turn parked at `continuable` with a live continuation id."""
    service = _service(seeded)
    started = await service.start_generation(
        CreateGeneration(
            conversation_id=seeded.conversation_id,
            user_id=seeded.owner_id,
            logical_turn_id=turn,
            checkpoint_thread_id=f"wf2:conv:{turn}",
            active_agent_id="chat_agent",
        )
    )
    running = await service.mark_running(
        generation_id=started.generation_id,
        user_id=seeded.owner_id,
        conversation_id=seeded.conversation_id,
        expected_version=started.version,
    )
    paused = await service.mark_continuable(
        MarkContinuable(
            generation_id=running.generation_id,
            conversation_id=seeded.conversation_id,
            user_id=seeded.owner_id,
            expected_version=running.version,
            assistant_message_id=seeded.assistant_message_id,
            research_accounting={"searches": 2},
        )
    )
    return service, paused


def _row(seeded: Seeded, generation_id: UUID) -> Generation:
    with seeded.session_factory() as session:  # type: ignore[operator]
        return session.execute(
            select(Generation).where(Generation.id == generation_id)
        ).scalar_one()


def _continue(seeded: Seeded, paused, *, key: str) -> ContinueGenerationCommand:
    return ContinueGenerationCommand(
        generation_id=paused.generation_id,
        continuation_id=paused.continuation_id,
        conversation_id=seeded.conversation_id,
        user_id=seeded.owner_id,
        idempotency_key=key,
        expected_version=paused.version,
    )


def _stop(seeded: Seeded, snapshot, *, key: str) -> StopGenerationCommand:
    return StopGenerationCommand(
        generation_id=snapshot.generation_id,
        conversation_id=seeded.conversation_id,
        user_id=seeded.owner_id,
        idempotency_key=key,
        expected_version=snapshot.version,
    )


def _settled(results: list) -> tuple[list, list]:
    """Split gathered results into winners and typed refusals."""
    winners = [item for item in results if not isinstance(item, BaseException)]
    refusals = [item for item in results if isinstance(item, GenerationControlError)]
    unexpected = [
        item
        for item in results
        if isinstance(item, BaseException) and not isinstance(item, GenerationControlError)
    ]
    assert not unexpected, f"a race produced an untyped failure: {unexpected}"
    return winners, refusals


# ----------------------------------------------------------------------
# Continue against Continue
# ----------------------------------------------------------------------


async def test_two_concurrent_continues_produce_one_epoch_increment(seeded: Seeded) -> None:
    """The defect this table exists to prevent: one question, two answers."""
    owner, paused = await _paused(seeded, turn="race-continue-1")
    other = _service(seeded)

    results = await asyncio.gather(
        owner.prepare_continue(_continue(seeded, paused, key="continue-key-aaaa")),
        other.prepare_continue(_continue(seeded, paused, key="continue-key-bbbb")),
        return_exceptions=True,
    )
    winners, refusals = _settled(results)

    assert len(winners) == 1, f"both Continues were accepted: {winners}"
    assert len(refusals) == 1
    row = _row(seeded, paused.generation_id)
    assert row.execution_epoch == 1
    assert row.status is GenerationStatus.CONTINUING


async def test_the_losing_continue_is_told_it_lost(seeded: Seeded) -> None:
    """A silent loser would leave a client waiting for a stream that never starts."""
    owner, paused = await _paused(seeded, turn="race-continue-2")
    other = _service(seeded)

    results = await asyncio.gather(
        owner.prepare_continue(_continue(seeded, paused, key="continue-key-cccc")),
        other.prepare_continue(_continue(seeded, paused, key="continue-key-dddd")),
        return_exceptions=True,
    )
    _winners, refusals = _settled(results)

    assert refusals and refusals[0].code in {"stale_command", "continuation_unavailable"}


async def test_a_spent_continuation_cannot_open_a_second_epoch(seeded: Seeded) -> None:
    """Sequential, not concurrent: the id is consumed, so the retry has nothing."""
    owner, paused = await _paused(seeded, turn="race-continue-3")
    await owner.prepare_continue(_continue(seeded, paused, key="continue-key-eeee"))

    with pytest.raises(GenerationControlError):
        await owner.prepare_continue(_continue(seeded, paused, key="continue-key-ffff"))

    assert _row(seeded, paused.generation_id).execution_epoch == 1


async def test_a_replayed_continue_returns_its_own_recorded_lease(seeded: Seeded) -> None:
    """Same key twice is a retry, not a second command."""
    owner, paused = await _paused(seeded, turn="race-continue-4")
    command = _continue(seeded, paused, key="continue-key-gggg")

    first = await owner.prepare_continue(command)
    second = await owner.prepare_continue(command)

    assert second.execution_epoch == first.execution_epoch
    assert second.paused_epoch == first.paused_epoch
    assert _row(seeded, paused.generation_id).execution_epoch == 1


# ----------------------------------------------------------------------
# Stop against Stop
# ----------------------------------------------------------------------


async def test_two_concurrent_stops_produce_one_transition(seeded: Seeded) -> None:
    owner, paused = await _paused(seeded, turn="race-stop-1")
    other = _service(seeded)
    running = await owner._repository.aget_owned(
        paused.generation_id, seeded.owner_id, seeded.conversation_id
    )

    results = await asyncio.gather(
        owner.request_stop(_stop(seeded, running, key="stop-key-aaaa")),
        other.request_stop(_stop(seeded, running, key="stop-key-bbbb")),
        return_exceptions=True,
    )
    winners, _refusals = _settled(results)

    # A paused turn resolves immediately: nothing is running to cancel.
    assert _row(seeded, paused.generation_id).status is GenerationStatus.COMPLETED_PARTIAL
    assert winners, "both Stops failed"


async def test_the_version_never_moves_backwards_under_concurrency(seeded: Seeded) -> None:
    """Monotonic versions are what make every later fence meaningful."""
    owner, paused = await _paused(seeded, turn="race-stop-2")
    other = _service(seeded)
    before = _row(seeded, paused.generation_id).version

    await asyncio.gather(
        owner.request_stop(_stop(seeded, paused, key="stop-key-cccc")),
        other.request_stop(_stop(seeded, paused, key="stop-key-dddd")),
        return_exceptions=True,
    )

    after = _row(seeded, paused.generation_id).version
    assert after > before
    # One legal transition, so exactly one increment.
    assert after == before + 1


async def test_the_repository_version_fence_refuses_the_second_writer(
    seeded: Seeded,
) -> None:
    """The fence itself, isolated from every predicate that shadows it.

    This one is asserted against ``atransition`` directly rather than through
    the service, and the reason is worth recording. Every other race in this
    file is settled *before* the fence is reached:

    * the service's own ``_require_fresh`` refuses a command whose version has
      already moved, which covers the sequential retry; and
    * ``atransition``'s ``status IN (...)`` predicate refuses a concurrent
      loser whose winner moved the row out of the declared set.

    So deleting ``WHERE version = :expected`` leaves every service-level test
    here green. What it does not survive is two writers that read the same
    version and declare a status set the winner's transition stays inside —
    which is what this constructs. Both target ``stop_requested``, a status
    that is itself in ``_STOPPABLE``, so only the version tells them apart.
    """
    repository = GenerationRepository(
        session_factory=seeded.session_factory,
        async_session_factory=seeded.async_session_factory,
    )
    created = await repository.acreate(
        CreateGeneration(
            conversation_id=seeded.conversation_id,
            user_id=seeded.owner_id,
            logical_turn_id="race-fence-1",
            checkpoint_thread_id="wf2:conv:race-fence-1",
            active_agent_id="chat_agent",
        )
    )
    stoppable = (
        GenerationStatus.STARTING,
        GenerationStatus.RUNNING,
        GenerationStatus.STOP_REQUESTED,
    )

    async def transition():
        return await repository.atransition(
            generation_id=created.generation_id,
            user_id=seeded.owner_id,
            conversation_id=seeded.conversation_id,
            expected_statuses=stoppable,
            expected_version=created.version,
            values={"status": GenerationStatus.STOP_REQUESTED},
        )

    results = await asyncio.gather(transition(), transition())

    applied = [result for result in results if result is not None]
    assert len(applied) == 1, "both writers applied against the same version"
    assert [result for result in results if result is None], "no writer was refused"
    # One transition, so one increment. A second would make every later fence
    # a client holds wrong by one.
    assert _row(seeded, created.generation_id).version == created.version + 1


async def test_a_replayed_stop_returns_the_first_answer(seeded: Seeded) -> None:
    owner, paused = await _paused(seeded, turn="race-stop-3")
    command = _stop(seeded, paused, key="stop-key-eeee")

    first = await owner.request_stop(command)
    second = await owner.request_stop(command)

    assert second == first
    assert _row(seeded, paused.generation_id).version == first.version


# ----------------------------------------------------------------------
# Continue against Stop
# ----------------------------------------------------------------------


async def test_a_stop_and_a_continue_on_one_paused_turn_leave_one_winner(
    seeded: Seeded,
) -> None:
    owner, paused = await _paused(seeded, turn="race-mixed-1")
    other = _service(seeded)

    results = await asyncio.gather(
        owner.prepare_continue(_continue(seeded, paused, key="continue-key-hhhh")),
        other.request_stop(_stop(seeded, paused, key="stop-key-ffff")),
        return_exceptions=True,
    )
    winners, refusals = _settled(results)

    assert len(winners) == 1, f"both commands were accepted: {winners}"
    assert len(refusals) == 1
    row = _row(seeded, paused.generation_id)
    # Whichever won, the row is in exactly one of the two legal outcomes.
    assert row.status in {GenerationStatus.CONTINUING, GenerationStatus.COMPLETED_PARTIAL}
    if row.status is GenerationStatus.COMPLETED_PARTIAL:
        assert row.execution_epoch == 0, "a losing Continue still spent an epoch"
    else:
        assert row.execution_epoch == 1


async def test_a_delayed_stop_replay_after_a_continue_is_refused_as_stale(
    seeded: Seeded,
) -> None:
    """R5, against the real fence.

    A Stop settles against epoch 0; a Continue is accepted and moves the row
    on; a delayed retry of the Stop arrives carrying the old fence. Executing
    it would cancel epoch 1, which nobody asked to stop. A single
    "last command" column cannot refuse this — the key no longer matches
    anything, so the command looks new.
    """
    owner, paused = await _paused(seeded, turn="race-mixed-2")
    stale_fence = paused.version

    lease = await owner.prepare_continue(_continue(seeded, paused, key="continue-key-iiii"))
    assert lease.execution_epoch == 1

    with pytest.raises(GenerationControlError) as caught:
        await owner.request_stop(
            StopGenerationCommand(
                generation_id=paused.generation_id,
                conversation_id=seeded.conversation_id,
                user_id=seeded.owner_id,
                idempotency_key="stop-key-delayed-retry",
                expected_version=stale_fence,
            )
        )

    assert caught.value.code == "stale_command"
    assert _row(seeded, paused.generation_id).status is GenerationStatus.CONTINUING


async def test_a_refused_command_is_recorded_so_its_replay_is_refused_identically(
    seeded: Seeded,
) -> None:
    """A refusal must not be reconsidered against a later epoch.

    Without the refusal in the ledger, the replay would be evaluated afresh —
    and by then the row may have moved into a state where the command *is*
    legal, which is the opposite of what a fence is for.
    """
    owner, paused = await _paused(seeded, turn="race-mixed-3")
    await owner.prepare_continue(_continue(seeded, paused, key="continue-key-jjjj"))

    command = StopGenerationCommand(
        generation_id=paused.generation_id,
        conversation_id=seeded.conversation_id,
        user_id=seeded.owner_id,
        idempotency_key="stop-key-refused-once",
        expected_version=paused.version,
    )
    with pytest.raises(GenerationControlError) as first:
        await owner.request_stop(command)
    with pytest.raises(GenerationControlError) as second:
        await owner.request_stop(command)

    assert type(second.value) is type(first.value)
    assert second.value.code == first.value.code


# ----------------------------------------------------------------------
# no duplicate assistant message
# ----------------------------------------------------------------------


async def test_one_assistant_message_survives_a_continue_race(seeded: Seeded) -> None:
    """The user-visible consequence of losing the epoch race.

    Two accepted Continues would each write an assistant message for one
    question. The row holds a single ``assistant_message_id``, so the count of
    distinct ids across the race is the observable.
    """
    owner, paused = await _paused(seeded, turn="race-message-1")
    other = _service(seeded)

    results = await asyncio.gather(
        owner.prepare_continue(_continue(seeded, paused, key="continue-key-kkkk")),
        other.prepare_continue(_continue(seeded, paused, key="continue-key-llll")),
        return_exceptions=True,
    )
    winners, _refusals = _settled(results)

    assert len(winners) == 1
    row = _row(seeded, paused.generation_id)
    assert row.execution_epoch == 1
    # The consumed continuation is cleared, so no second Continue can be built.
    assert row.continuation_id is None
    assert row.continuation_available is False


async def test_a_continuation_becomes_unusable_the_moment_it_is_consumed(
    seeded: Seeded,
) -> None:
    owner, paused = await _paused(seeded, turn="race-message-2")

    await owner.prepare_continue(_continue(seeded, paused, key="continue-key-mmmm"))
    refreshed = await owner.aget_snapshot(
        generation_id=paused.generation_id,
        user_id=seeded.owner_id,
        conversation_id=seeded.conversation_id,
    )

    with pytest.raises(ContinuationUnavailable):
        await owner.prepare_continue(
            ContinueGenerationCommand(
                generation_id=paused.generation_id,
                continuation_id=paused.continuation_id,
                conversation_id=seeded.conversation_id,
                user_id=seeded.owner_id,
                idempotency_key="continue-key-nnnn",
                expected_version=refreshed.version,
            )
        )
