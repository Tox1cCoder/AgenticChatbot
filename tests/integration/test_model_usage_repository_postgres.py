"""PostgreSQL integration tests for the model-usage repository.

Mirrors ``tests/integration/test_conversation_compaction_postgres.py``: a
module-scoped engine bound to ``TEST_DATABASE_URL`` (skipped if unset), an
explicit ``create_all`` table list covering every FK target of the usage
tables, and per-test seed/cleanup via a tenant factory fixture.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy import event as sa_event
from sqlalchemy.orm import sessionmaker

from app.models.base import Base
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.enums import MessageRole
from app.models.message import Message
from app.models.model_usage import ModelUsageEvent, ModelUsageMinute
from app.models.user import User
from app.repositories.model_usage import (
    ModelUsageReferenceError,
    ModelUsageRepository,
    RecordEventCommand,
    compute_rollup_key,
)
from app.usage.types import NormalizedUsage, UsageContext

TenantFactory = Callable[..., tuple[UUID, UUID | None, UUID | None]]


@pytest.fixture(scope="module")
def session_factory():
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
            Document.__table__,
            ModelUsageEvent.__table__,
            ModelUsageMinute.__table__,
        ],
    )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        engine.dispose()


@pytest.fixture()
def repository(session_factory) -> ModelUsageRepository:
    return ModelUsageRepository(session_factory)


@pytest.fixture()
def tenant_factory(session_factory) -> Iterator[TenantFactory]:
    """Create disposable (user, conversation, message) tuples and clean them up.

    Cleanup deletes rollups/events for the created users first (neither table
    is reachable through cascade from a test that deletes the conversation or
    user itself mid-test), then messages, conversations, and users -- an order
    that respects the RESTRICT-by-default FKs (Message -> Conversation,
    Conversation -> User) regardless of what a given test already removed.
    """
    created_user_ids: list[UUID] = []
    created_conversation_ids: list[UUID] = []
    created_message_ids: list[UUID] = []

    def _create(
        *, with_conversation: bool = True, with_message: bool = False
    ) -> tuple[UUID, UUID | None, UUID | None]:
        if with_message and not with_conversation:
            raise ValueError("with_message requires with_conversation")
        user_id = uuid4()
        conversation_id = uuid4() if with_conversation else None
        message_id = uuid4() if with_message else None
        with session_factory.begin() as session:
            session.add(
                User(
                    id=user_id,
                    username=f"usage-{user_id}",
                    email=f"{user_id}@example.test",
                    password_hash="test",
                )
            )
            if conversation_id is not None:
                session.add(Conversation(id=conversation_id, owner_id=user_id, title="usage test"))
            if message_id is not None:
                session.add(
                    Message(
                        id=message_id,
                        conversation_id=conversation_id,
                        sender=MessageRole.user.value,
                        content="usage test message",
                        message_metadata={},
                        sequence=1,
                    )
                )
        created_user_ids.append(user_id)
        if conversation_id is not None:
            created_conversation_ids.append(conversation_id)
        if message_id is not None:
            created_message_ids.append(message_id)
        return user_id, conversation_id, message_id

    yield _create

    with session_factory.begin() as session:
        session.execute(
            delete(ModelUsageMinute).where(ModelUsageMinute.user_id.in_(created_user_ids))
        )
        session.execute(
            delete(ModelUsageEvent).where(ModelUsageEvent.user_id.in_(created_user_ids))
        )
        if created_message_ids:
            session.execute(delete(Message).where(Message.id.in_(created_message_ids)))
        if created_conversation_ids:
            session.execute(
                delete(Conversation).where(Conversation.id.in_(created_conversation_ids))
            )
        session.execute(delete(User).where(User.id.in_(created_user_ids)))


def _command(
    *,
    user_id: UUID,
    conversation_id: UUID | None,
    request_message_id: UUID | None = None,
    operation_id: UUID | None = None,
    attempt: int = 1,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    operation: str = "chat",
    agent_id: str | None = None,
    status: str = "success",
    usage_source: str = "provider_reported",
    input_tokens: int | None = 100,
    output_tokens: int | None = 50,
    total_tokens: int | None = 150,
    reasoning_tokens: int | None = None,
    cached_input_tokens: int | None = None,
    input_text_tokens: int | None = None,
    input_image_tokens: int | None = None,
    output_text_tokens: int | None = None,
    output_image_tokens: int | None = None,
    generated_images: int = 0,
    latency_ms: int = 250,
    started_at: datetime | None = None,
) -> RecordEventCommand:
    started = started_at or datetime.now(timezone.utc).replace(second=0, microsecond=0)
    return RecordEventCommand(
        operation_id=operation_id or uuid4(),
        attempt=attempt,
        context=UsageContext(
            user_id=user_id,
            conversation_id=conversation_id,
            request_message_id=request_message_id,
            operation=operation,
            agent_id=agent_id,
        ),
        usage=NormalizedUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_input_tokens=cached_input_tokens,
            input_text_tokens=input_text_tokens,
            input_image_tokens=input_image_tokens,
            output_text_tokens=output_text_tokens,
            output_image_tokens=output_image_tokens,
            generated_images=generated_images,
            source=usage_source,
        ),
        provider=provider,
        model=model,
        status=status,
        latency_ms=latency_ms,
        started_at=started,
        completed_at=started + timedelta(milliseconds=latency_ms),
    )


def _rollup_key_for(command: RecordEventCommand, conversation_id: UUID | None) -> str:
    bucket = command.started_at.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return compute_rollup_key(
        bucket_start_utc=bucket,
        user_id=command.context.user_id,
        conversation_id=conversation_id,
        provider=command.provider,
        model=command.model,
        operation=command.context.operation,
        agent_id=command.context.agent_id,
        status=command.status,
        usage_source=command.usage.source,
    )


def test_record_event_is_idempotent_and_increments_rollup_once(
    repository, tenant_factory, session_factory
) -> None:
    user_id, conversation_id, message_id = tenant_factory(with_message=True)
    command = _command(
        user_id=user_id, conversation_id=conversation_id, request_message_id=message_id
    )

    first = repository.record_event(command)
    second = repository.record_event(command)

    assert first.inserted is True
    assert first.event_id is not None
    assert second.inserted is False
    assert second.event_id is None

    with session_factory() as session:
        events = list(
            session.execute(
                select(ModelUsageEvent).where(ModelUsageEvent.operation_id == command.operation_id)
            ).scalars()
        )
        assert len(events) == 1
        assert events[0].request_message_id == message_id

        minute = session.get(ModelUsageMinute, _rollup_key_for(command, conversation_id))
        assert minute is not None
        assert minute.request_count == 1
        assert minute.input_tokens_sum == 100
        assert minute.input_tokens_known_count == 1
        assert minute.latency_ms_sum == command.latency_ms


def test_unknown_tokens_remain_null_while_rollup_uses_zero_for_sum(
    repository, tenant_factory, session_factory
) -> None:
    user_id, conversation_id, _ = tenant_factory()
    command = _command(
        user_id=user_id,
        conversation_id=conversation_id,
        input_tokens=100,
        output_tokens=None,
        total_tokens=None,
    )

    result = repository.record_event(command)
    assert result.inserted is True

    with session_factory() as session:
        event = session.get(ModelUsageEvent, result.event_id)
        assert event.input_tokens == 100
        assert event.output_tokens is None
        assert event.total_tokens is None

        minute = session.get(ModelUsageMinute, _rollup_key_for(command, conversation_id))
        assert minute.request_count == 1
        assert minute.input_tokens_sum == 100
        assert minute.input_tokens_known_count == 1
        assert minute.output_tokens_sum == 0
        assert minute.output_tokens_known_count == 0
        assert minute.total_tokens_sum == 0
        assert minute.total_tokens_known_count == 0


def test_same_logical_operation_distinct_attempts_are_recorded(
    repository, tenant_factory, session_factory
) -> None:
    user_id, conversation_id, _ = tenant_factory()
    operation_id = uuid4()
    started = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    first_command = _command(
        user_id=user_id,
        conversation_id=conversation_id,
        operation_id=operation_id,
        attempt=1,
        status="error",
        started_at=started,
    )
    second_command = _command(
        user_id=user_id,
        conversation_id=conversation_id,
        operation_id=operation_id,
        attempt=2,
        status="error",
        started_at=started,
    )

    first = repository.record_event(first_command)
    second = repository.record_event(second_command)

    assert first.inserted is True
    assert second.inserted is True
    assert first.event_id != second.event_id

    with session_factory() as session:
        events = list(
            session.execute(
                select(ModelUsageEvent).where(ModelUsageEvent.operation_id == operation_id)
            ).scalars()
        )
        assert {event.attempt for event in events} == {1, 2}
        assert {event.event_key for event in events} == {
            f"{operation_id}:1",
            f"{operation_id}:2",
        }

        minute = session.get(ModelUsageMinute, _rollup_key_for(first_command, conversation_id))
        assert minute.request_count == 2
        assert minute.input_tokens_sum == 200
        assert minute.input_tokens_known_count == 2


def test_user_and_conversation_filters_never_cross_tenants(
    repository, tenant_factory, session_factory
) -> None:
    user_a, conversation_a, _ = tenant_factory()
    user_b, conversation_b, _ = tenant_factory()
    window_start = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(
        minutes=5
    )
    window_end = window_start + timedelta(minutes=10)
    started = window_start + timedelta(minutes=1)

    repository.record_event(
        _command(user_id=user_a, conversation_id=conversation_a, started_at=started)
    )
    repository.record_event(
        _command(user_id=user_b, conversation_id=conversation_b, started_at=started)
    )

    totals_a = repository.get_summary_totals(
        user_id=user_a, start_inclusive=window_start, end_exclusive=window_end
    )
    totals_b = repository.get_summary_totals(
        user_id=user_b, start_inclusive=window_start, end_exclusive=window_end
    )
    assert totals_a.request_count == 1
    assert totals_b.request_count == 1

    series_a = repository.get_minute_series(
        user_id=user_a, start_inclusive=window_start, end_exclusive=window_end
    )
    assert len(series_a) == 1
    assert all(row.user_id == user_a for row in series_a)

    breakdown_a = repository.get_dimension_breakdown(
        user_id=user_a,
        start_inclusive=window_start,
        end_exclusive=window_end,
        dimension="provider",
    )
    assert len(breakdown_a) == 1
    assert breakdown_a[0].dimension_value == "openai"
    assert breakdown_a[0].totals.request_count == 1

    cross_tenant_conversation = repository.get_summary_totals(
        user_id=user_a,
        start_inclusive=window_start,
        end_exclusive=window_end,
        conversation_id=conversation_b,
    )
    assert cross_tenant_conversation.request_count == 0

    with pytest.raises(ValueError):
        repository.get_summary_totals(
            user_id=None, start_inclusive=window_start, end_exclusive=window_end
        )
    with pytest.raises(ValueError):
        repository.get_minute_series(
            user_id=None, start_inclusive=window_start, end_exclusive=window_end
        )
    with pytest.raises(ValueError):
        repository.get_dimension_breakdown(
            user_id=None,
            start_inclusive=window_start,
            end_exclusive=window_end,
            dimension="provider",
        )


def test_bucket_series_aggregates_intervals_in_sql_and_stays_tenant_bounded(
    repository, tenant_factory
) -> None:
    user_a, conversation_a, _ = tenant_factory()
    user_b, conversation_b, _ = tenant_factory()
    start = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=10)
    intervals = [
        (start, start + timedelta(minutes=2)),
        (start + timedelta(minutes=2), start + timedelta(minutes=4)),
        (start + timedelta(minutes=4), start + timedelta(minutes=6)),
    ]

    repository.record_event(
        _command(
            user_id=user_a,
            conversation_id=conversation_a,
            provider="openai",
            total_tokens=150,
            started_at=start,
        )
    )
    repository.record_event(
        _command(
            user_id=user_a,
            conversation_id=conversation_a,
            provider="gemini",
            input_tokens=10,
            output_tokens=10,
            total_tokens=20,
            started_at=start + timedelta(minutes=1),
        )
    )
    repository.record_event(
        _command(
            user_id=user_a,
            conversation_id=conversation_a,
            total_tokens=30,
            started_at=start + timedelta(minutes=2),
        )
    )
    repository.record_event(
        _command(
            user_id=user_b,
            conversation_id=conversation_b,
            total_tokens=999,
            started_at=start,
        )
    )

    rows = repository.get_bucket_series(
        user_id=user_a,
        bucket_intervals=intervals,
        conversation_id=conversation_a,
    )

    assert len(rows) == 2
    assert len(rows) <= len(intervals)
    assert [row.bucket_start_utc for row in rows] == [intervals[0][0], intervals[1][0]]
    assert [row.totals.request_count for row in rows] == [2, 1]
    assert [row.totals.total_tokens_sum for row in rows] == [170, 30]
    assert [row.totals.total_tokens_known_count for row in rows] == [2, 1]

    cross_tenant = repository.get_bucket_series(
        user_id=user_a,
        bucket_intervals=intervals,
        conversation_id=conversation_b,
    )
    assert cross_tenant == []


def test_latest_conversation_event_breaks_timestamp_attempt_ties_by_id(
    repository, tenant_factory
) -> None:
    user_id, conversation_id, _ = tenant_factory()
    started = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    first = repository.record_event(
        _command(
            user_id=user_id,
            conversation_id=conversation_id,
            provider="openai",
            model="gpt-4o",
            attempt=1,
            started_at=started,
        )
    )
    second = repository.record_event(
        _command(
            user_id=user_id,
            conversation_id=conversation_id,
            provider="gemini",
            model="gemini-2.5-flash",
            attempt=1,
            started_at=started,
        )
    )
    expected_id = max(first.event_id, second.event_id)

    latest = repository.get_latest_conversation_event(
        user_id=user_id, conversation_id=conversation_id
    )

    assert latest is not None
    assert latest.id == expected_id


def test_reconcile_minute_rebuilds_exactly_from_raw_events(
    repository, tenant_factory, session_factory
) -> None:
    user_id, conversation_id, _ = tenant_factory()
    started = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    command = _command(
        user_id=user_id,
        conversation_id=conversation_id,
        started_at=started,
        input_tokens=42,
        output_tokens=8,
        latency_ms=123,
    )
    result = repository.record_event(command)
    assert result.inserted is True
    rollup_key = _rollup_key_for(command, conversation_id)

    # A rollup row with no backing raw event at all: reconciliation must
    # purge it outright, not just leave its drifted sums in place.
    phantom_rollup_key = compute_rollup_key(
        bucket_start_utc=started,
        user_id=user_id,
        conversation_id=conversation_id,
        provider=command.provider,
        model=command.model,
        operation=command.context.operation,
        agent_id=command.context.agent_id,
        status="cancelled",
        usage_source=command.usage.source,
    )
    with session_factory.begin() as session:
        minute = session.get(ModelUsageMinute, rollup_key)
        minute.request_count = 999
        minute.input_tokens_sum = 555555
        minute.latency_ms_sum = 999999
        session.add(
            ModelUsageMinute(
                rollup_key=phantom_rollup_key,
                bucket_start_utc=started,
                user_id=user_id,
                conversation_id=conversation_id,
                provider=command.provider,
                model=command.model,
                operation=command.context.operation,
                agent_id=command.context.agent_id,
                status="cancelled",
                usage_source=command.usage.source,
                request_count=5,
            )
        )

    rebuilt_count = repository.reconcile_minute_range(
        start_inclusive=started, end_exclusive=started + timedelta(minutes=1)
    )
    assert rebuilt_count == 1

    with session_factory() as session:
        minute = session.get(ModelUsageMinute, rollup_key)
        assert minute.request_count == 1
        assert minute.input_tokens_sum == 42
        assert minute.input_tokens_known_count == 1
        assert minute.output_tokens_sum == 8
        assert minute.latency_ms_sum == 123
        assert session.get(ModelUsageMinute, phantom_rollup_key) is None


def test_reconcile_twice_is_exactly_idempotent_across_chunks_and_small_batches(
    repository, tenant_factory, session_factory
) -> None:
    user_id, conversation_id, _ = tenant_factory()
    start = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=5)
    for minute_offset in (0, 1, 2, 3):
        repository.record_event(
            _command(
                user_id=user_id,
                conversation_id=conversation_id,
                operation_id=uuid4(),
                started_at=start + timedelta(minutes=minute_offset),
                input_tokens=10 + minute_offset,
            )
        )

    def state():
        with session_factory() as session:
            rows = session.execute(
                select(ModelUsageMinute)
                .where(ModelUsageMinute.user_id == user_id)
                .order_by(ModelUsageMinute.rollup_key)
            ).scalars()
            return [
                (
                    row.rollup_key,
                    row.bucket_start_utc,
                    row.request_count,
                    row.input_tokens_sum,
                    row.input_tokens_known_count,
                )
                for row in rows
            ]

    first_count = repository.reconcile_minute_range(
        start_inclusive=start,
        end_exclusive=start + timedelta(minutes=5),
        chunk_minutes=2,
        insert_batch_size=1,
    )
    first_state = state()
    second_count = repository.reconcile_minute_range(
        start_inclusive=start,
        end_exclusive=start + timedelta(minutes=5),
        chunk_minutes=2,
        insert_batch_size=1,
    )

    assert first_count == second_count == 4
    assert state() == first_state


def test_reconcile_server_aggregation_preserves_all_sums_and_known_counts(
    repository, tenant_factory, session_factory
) -> None:
    user_id, conversation_id, _ = tenant_factory()
    minute = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    repository.record_event(
        _command(
            user_id=user_id,
            conversation_id=conversation_id,
            started_at=minute,
            reasoning_tokens=7,
            cached_input_tokens=3,
            input_text_tokens=70,
            input_image_tokens=30,
            output_text_tokens=40,
            output_image_tokens=10,
            generated_images=2,
        )
    )
    repository.record_event(
        _command(
            user_id=user_id,
            conversation_id=conversation_id,
            started_at=minute + timedelta(seconds=1),
            reasoning_tokens=None,
            cached_input_tokens=4,
            input_text_tokens=None,
            input_image_tokens=20,
            output_text_tokens=50,
            output_image_tokens=None,
            generated_images=1,
        )
    )

    assert (
        repository.reconcile_minute_range(
            start_inclusive=minute,
            end_exclusive=minute + timedelta(minutes=1),
            chunk_minutes=1,
            insert_batch_size=1,
        )
        == 1
    )

    with session_factory() as session:
        row = session.execute(
            select(ModelUsageMinute).where(ModelUsageMinute.user_id == user_id)
        ).scalar_one()
        assert (row.reasoning_tokens_sum, row.reasoning_tokens_known_count) == (7, 1)
        assert (row.cached_input_tokens_sum, row.cached_input_tokens_known_count) == (7, 2)
        assert (row.input_text_tokens_sum, row.input_text_tokens_known_count) == (70, 1)
        assert (row.input_image_tokens_sum, row.input_image_tokens_known_count) == (50, 2)
        assert (row.output_text_tokens_sum, row.output_text_tokens_known_count) == (90, 2)
        assert (row.output_image_tokens_sum, row.output_image_tokens_known_count) == (10, 1)
        assert row.generated_images_sum == 3


def test_reconcile_minute_lock_serializes_same_minute_but_not_other_minute(
    repository, tenant_factory, session_factory
) -> None:
    user_id, conversation_id, _ = tenant_factory()
    minute = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=2)
    repository.record_event(
        _command(user_id=user_id, conversation_id=conversation_id, started_at=minute)
    )
    locks_acquired = threading.Event()
    release_reconcile = threading.Event()
    same_minute_done = threading.Event()
    other_minute_done = threading.Event()
    errors: list[BaseException] = []
    engine = session_factory.kw["bind"]

    def pause_after_exclusive_locks(
        _connection, _cursor, statement, _parameters, _context, _executemany
    ):
        if "pg_advisory_xact_lock(" in statement:
            locks_acquired.set()
            if not release_reconcile.wait(5):
                raise TimeoutError("test did not release reconciliation")

    def run_reconcile():
        try:
            repository.reconcile_minute_range(
                start_inclusive=minute,
                end_exclusive=minute + timedelta(minutes=1),
                chunk_minutes=1,
            )
        except BaseException as exc:
            errors.append(exc)

    def record_at(started_at, done):
        try:
            repository.record_event(
                _command(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    started_at=started_at,
                )
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    sa_event.listen(engine, "after_cursor_execute", pause_after_exclusive_locks)
    try:
        reconcile_thread = threading.Thread(target=run_reconcile)
        reconcile_thread.start()
        assert locks_acquired.wait(5)

        same_thread = threading.Thread(target=record_at, args=(minute, same_minute_done))
        other_thread = threading.Thread(
            target=record_at,
            args=(minute + timedelta(minutes=1), other_minute_done),
        )
        same_thread.start()
        other_thread.start()

        assert other_minute_done.wait(2), "a different UTC minute must not share the lock"
        assert not same_minute_done.wait(0.2), "same-minute write must wait for reconciliation"
        release_reconcile.set()
        reconcile_thread.join(5)
        same_thread.join(5)
        other_thread.join(5)
        assert not reconcile_thread.is_alive()
        assert not same_thread.is_alive()
        assert not other_thread.is_alive()
        assert errors == []
    finally:
        release_reconcile.set()
        sa_event.remove(engine, "after_cursor_execute", pause_after_exclusive_locks)

    with session_factory() as session:
        raw_count = session.scalar(
            select(func.count(ModelUsageEvent.id)).where(
                ModelUsageEvent.user_id == user_id,
                ModelUsageEvent.started_at >= minute,
                ModelUsageEvent.started_at < minute + timedelta(minutes=1),
            )
        )
        rollup_count = session.scalar(
            select(func.sum(ModelUsageMinute.request_count)).where(
                ModelUsageMinute.user_id == user_id,
                ModelUsageMinute.bucket_start_utc == minute,
            )
        )
    assert raw_count == rollup_count == 2


def test_rollup_fk_deletes_do_not_mutate_hashed_dimensions(
    repository, tenant_factory, session_factory
) -> None:
    user_id, conversation_id, _ = tenant_factory()
    started = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    command = _command(user_id=user_id, conversation_id=conversation_id, started_at=started)
    result = repository.record_event(command)
    assert result.inserted is True
    rollup_key = _rollup_key_for(command, conversation_id)

    with session_factory() as session:
        assert session.get(ModelUsageMinute, rollup_key) is not None

    with session_factory.begin() as session:
        session.execute(delete(Conversation).where(Conversation.id == conversation_id))

    with session_factory() as session:
        assert session.get(ModelUsageMinute, rollup_key) is None
        event = session.get(ModelUsageEvent, result.event_id)
        assert event is not None
        assert event.conversation_id is None

        # The row must be gone outright -- not silently updated in place to
        # the dimensions a NULL conversation_id would hash to.
        orphaned_rollup_key = compute_rollup_key(
            bucket_start_utc=started,
            user_id=user_id,
            conversation_id=None,
            provider=command.provider,
            model=command.model,
            operation=command.context.operation,
            agent_id=command.context.agent_id,
            status=command.status,
            usage_source=command.usage.source,
        )
        assert session.get(ModelUsageMinute, orphaned_rollup_key) is None

    with session_factory.begin() as session:
        session.execute(delete(User).where(User.id == user_id))

    with session_factory() as session:
        assert session.get(ModelUsageEvent, result.event_id) is None


def test_cleanup_deletes_raw_older_than_90_days_and_rollups_older_than_2_years(
    repository, tenant_factory, session_factory
) -> None:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    raw_user_id, raw_conversation_id, _ = tenant_factory()
    raw_cutoff = now - timedelta(days=90)
    old_event_ids = [
        repository.record_event(
            _command(
                user_id=raw_user_id,
                conversation_id=raw_conversation_id,
                started_at=raw_cutoff - timedelta(minutes=minute_offset),
            )
        ).event_id
        for minute_offset in (1, 2, 3)
    ]
    recent_event = repository.record_event(
        _command(
            user_id=raw_user_id,
            conversation_id=raw_conversation_id,
            started_at=now - timedelta(days=1),
        )
    )

    deleted_raw = repository.delete_raw_events_older_than(raw_cutoff, batch_size=1)
    assert deleted_raw == 3
    with session_factory() as session:
        for event_id in old_event_ids:
            assert session.get(ModelUsageEvent, event_id) is None
        assert session.get(ModelUsageEvent, recent_event.event_id) is not None

    rollup_user_id, rollup_conversation_id, _ = tenant_factory()
    rollup_cutoff = now - timedelta(days=730)
    old_started = rollup_cutoff - timedelta(minutes=1)
    recent_started = now - timedelta(days=1)
    repository.record_event(
        _command(
            user_id=rollup_user_id, conversation_id=rollup_conversation_id, started_at=old_started
        )
    )
    repository.record_event(
        _command(
            user_id=rollup_user_id,
            conversation_id=rollup_conversation_id,
            started_at=recent_started,
        )
    )

    deleted_rollups = repository.delete_rollups_older_than(rollup_cutoff)
    assert deleted_rollups == 1

    old_rollup_key = compute_rollup_key(
        bucket_start_utc=old_started,
        user_id=rollup_user_id,
        conversation_id=rollup_conversation_id,
        provider="openai",
        model="gpt-4o-mini",
        operation="chat",
        agent_id=None,
        status="success",
        usage_source="provider_reported",
    )
    recent_rollup_key = compute_rollup_key(
        bucket_start_utc=recent_started,
        user_id=rollup_user_id,
        conversation_id=rollup_conversation_id,
        provider="openai",
        model="gpt-4o-mini",
        operation="chat",
        agent_id=None,
        status="success",
        usage_source="provider_reported",
    )
    with session_factory() as session:
        assert session.get(ModelUsageMinute, old_rollup_key) is None
        assert session.get(ModelUsageMinute, recent_rollup_key) is not None


def test_health_snapshot_uses_complete_minutes_and_content_free_aggregates(
    repository, tenant_factory
) -> None:
    user_id, conversation_id, _ = tenant_factory()
    complete_minute = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    event_minute = complete_minute - timedelta(minutes=1)
    repository.record_event(
        _command(
            user_id=user_id,
            conversation_id=conversation_id,
            started_at=event_minute,
        )
    )

    result = repository.get_model_usage_health_snapshot(
        now=complete_minute + timedelta(seconds=45),
        lookback_minutes=10,
    )

    assert result == {
        "raw_event_count": 1,
        "unattributed_event_count": 0,
        "rollup_request_count": 1,
        "latest_event_minute": event_minute,
        "latest_rollup_minute": event_minute,
    }


def test_record_event_rejects_unpersisted_request_message_id(
    repository, tenant_factory, session_factory
) -> None:
    user_id, conversation_id, _ = tenant_factory()
    reserved_assistant_message_id = uuid4()  # never persisted as a Message row
    command = _command(
        user_id=user_id,
        conversation_id=conversation_id,
        request_message_id=reserved_assistant_message_id,
    )

    with pytest.raises(ModelUsageReferenceError):
        repository.record_event(command)

    with session_factory() as session:
        event = session.execute(
            select(ModelUsageEvent).where(ModelUsageEvent.event_key == command.event_key)
        ).scalar_one_or_none()
        assert event is None
