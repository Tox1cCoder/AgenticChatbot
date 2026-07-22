"""Repository for the per-user model-usage ledger and its minute rollups.

``ModelUsageRepository.record_event`` performs the entire idempotent
insert-then-aggregate flow (data-contract.md sections 4.1-4.2) inside one
transaction: insert the immutable event with ``ON CONFLICT DO NOTHING`` on
``event_key``, and only when a new row was actually inserted, fold its
contribution into the UTC-minute rollup with ``ON CONFLICT DO UPDATE``.

The remaining methods are bounded read and maintenance queries: summary
totals, a minute time series, a dimension breakdown, the latest visible
assistant-message context gauge, minute-range reconciliation (rebuild rollups
exactly from raw events), and batched deletion of aged raw events / rollups.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, TypeVar
from uuid import UUID

from sqlalchemy import DateTime, Integer, and_, column, delete, desc, func, select, text, values
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.conversation import Conversation
from app.models.enums import MessageRole
from app.models.message import Message
from app.models.model_usage import NULLABLE_TOKEN_FIELDS, ModelUsageEvent, ModelUsageMinute
from app.usage.types import NormalizedUsage, UsageContext, UsageStatus

# rollup_key derivation version (data-contract.md §4.2): a leading version
# marker so a future change to the dimension list or encoding can mint a
# distinct hash space instead of silently colliding with v1 keys.
_ROLLUP_KEY_VERSION = "v1"
_MINUTE_LOCK_NAMESPACE = (
    int.from_bytes(
        hashlib.sha256(b"model_usage_minute_v1").digest()[:4],
        byteorder="big",
    )
    & 0x7FFFFFFF
)
_UINT32_SIZE = 1 << 32

UsageDimension = Literal["provider", "model", "operation", "agent_id", "status", "usage_source"]

_DIMENSION_COLUMNS: dict[str, Any] = {
    "provider": ModelUsageMinute.provider,
    "model": ModelUsageMinute.model,
    "operation": ModelUsageMinute.operation,
    "agent_id": ModelUsageMinute.agent_id,
    "status": ModelUsageMinute.status,
    "usage_source": ModelUsageMinute.usage_source,
}

# Column names summed/counted by every bounded totals query, in a fixed
# order shared by SQL aggregation and reconciliation.
_TOTALS_FIELD_NAMES: tuple[str, ...] = (
    "request_count",
    "latency_ms_sum",
    "generated_images_sum",
    *(f"{field}_{suffix}" for field in NULLABLE_TOKEN_FIELDS for suffix in ("sum", "known_count")),
)


def _require_aware(value: datetime, field_name: str) -> datetime:
    """Raise unless ``value`` carries explicit timezone/UTC-offset info."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime")
    return value


def _require_minute_aligned_utc(value: datetime, field_name: str) -> datetime:
    """Return ``value`` normalized to UTC, rejecting non-minute-aligned instants."""
    utc_value = _require_aware(value, field_name).astimezone(timezone.utc)
    if utc_value.second != 0 or utc_value.microsecond != 0:
        raise ValueError(f"{field_name} must be minute-aligned (second=microsecond=0 UTC)")
    return utc_value


def _minute_lock_key(bucket_start_utc: datetime) -> int:
    aligned = _require_minute_aligned_utc(bucket_start_utc, "bucket_start_utc")
    minute = int(aligned.timestamp() // 60)
    if not 0 <= minute < _UINT32_SIZE:
        raise ValueError("bucket_start_utc is outside the advisory-lock key range")
    return (_MINUTE_LOCK_NAMESPACE * _UINT32_SIZE) + minute


def compute_rollup_key(
    *,
    bucket_start_utc: datetime,
    user_id: UUID | None,
    conversation_id: UUID | None,
    provider: str,
    model: str,
    operation: str,
    agent_id: str | None,
    status: str,
    usage_source: str,
) -> str:
    """Deterministically hash a rollup's dimensions to its ``rollup_key``.

    Per data-contract.md §4.2, this is a SHA-256 digest over the UTF-8 JSON
    encoding of a *versioned, ordered array* -- not delimiter-joined strings
    -- with explicit ``null`` entries for absent dimensions and an RFC 3339
    UTC minute for the bucket. ``reconcile_minute_range`` calls this exact
    function so a rebuilt row's key always matches the one ``record_event``
    would have produced for the same dimensions.
    """
    bucket_utc = _require_minute_aligned_utc(bucket_start_utc, "bucket_start_utc")
    ordered_dimensions = [
        _ROLLUP_KEY_VERSION,
        bucket_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        str(user_id) if user_id is not None else None,
        str(conversation_id) if conversation_id is not None else None,
        provider,
        model,
        operation,
        agent_id,
        status,
        usage_source,
    ]
    encoded = json.dumps(ordered_dimensions, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RecordEventCommand:
    """Storage-facing command for one provider-call attempt's outcome.

    Combines the domain ``UsageContext``/``NormalizedUsage`` types from
    ``app.usage.types`` with the attempt identity and outcome fields those
    types don't carry: ``operation_id``/``attempt`` (allocated by
    ``UsageOperation``), the resolved ``provider``/``model``/``status``,
    timing, and the optional ``provider_request_id``/``error_code``.
    """

    operation_id: UUID
    attempt: int
    context: UsageContext
    usage: NormalizedUsage
    provider: str
    model: str
    status: UsageStatus
    latency_ms: int
    started_at: datetime
    completed_at: datetime
    provider_request_id: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.attempt < 1:
            raise ValueError("RecordEventCommand.attempt must be >= 1")
        if self.latency_ms < 0:
            raise ValueError("RecordEventCommand.latency_ms must be >= 0")
        _require_aware(self.started_at, "RecordEventCommand.started_at")
        _require_aware(self.completed_at, "RecordEventCommand.completed_at")

    @property
    def event_key(self) -> str:
        """Stable idempotency key: ``<operation_id>:<attempt>`` (data-contract.md §4.1)."""
        return f"{self.operation_id}:{self.attempt}"


@dataclass(frozen=True)
class RecordResult:
    """Outcome of ``record_event``: whether a new row was actually inserted."""

    inserted: bool
    event_id: UUID | None


@dataclass(frozen=True)
class UsageTotals:
    """Aggregated ``ModelUsageMinute`` totals over a bounded window."""

    request_count: int
    latency_ms_sum: int
    generated_images_sum: int
    input_tokens_sum: int
    input_tokens_known_count: int
    output_tokens_sum: int
    output_tokens_known_count: int
    total_tokens_sum: int
    total_tokens_known_count: int
    reasoning_tokens_sum: int
    reasoning_tokens_known_count: int
    cached_input_tokens_sum: int
    cached_input_tokens_known_count: int
    input_text_tokens_sum: int
    input_text_tokens_known_count: int
    input_image_tokens_sum: int
    input_image_tokens_known_count: int
    output_text_tokens_sum: int
    output_text_tokens_known_count: int
    output_image_tokens_sum: int
    output_image_tokens_known_count: int


@dataclass(frozen=True)
class DimensionUsageTotals:
    """One grouped row from ``get_dimension_breakdown``."""

    dimension_value: str | None
    totals: UsageTotals


@dataclass(frozen=True)
class ConversationUsageTotals:
    """One bounded conversation aggregate joined to its display title."""

    conversation_id: UUID
    title: str | None
    totals: UsageTotals


@dataclass(frozen=True)
class BucketUsageTotals:
    """One SQL-aggregated response bucket, bounded by caller-supplied UTC instants."""

    bucket_start_utc: datetime
    bucket_end_utc: datetime
    totals: UsageTotals


class ModelUsageReferenceError(Exception):
    """Raised when ``record_event`` references a row that doesn't exist yet.

    Wraps the underlying ``IntegrityError`` from a foreign-key violation --
    e.g. a reserved-but-unpersisted assistant message UUID supplied as
    ``request_message_id`` (data-contract.md §4.1) -- so callers get an
    explicit, catchable error instead of a raw SQLAlchemy/DBAPI exception.
    The originating error is kept on ``__cause__`` via ``raise ... from exc``.
    """


def _sum_expressions() -> list[Any]:
    return [
        func.coalesce(func.sum(getattr(ModelUsageMinute, name)), 0).label(name)
        for name in _TOTALS_FIELD_NAMES
    ]


def _totals_from_mapping(mapping: Mapping[str, Any]) -> UsageTotals:
    return UsageTotals(**{name: int(mapping[name]) for name in _TOTALS_FIELD_NAMES})


_T = TypeVar("_T")


def _partition_rows(rows: Iterable[_T], *, batch_size: int) -> Iterator[list[_T]]:
    """Yield bounded insertion partitions without materializing ``rows``."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    batch: list[_T] = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class ModelUsageRepository:
    """Own the model-usage ledger's write transaction and bounded queries."""

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    # -- Write path ---------------------------------------------------

    def record_event(self, command: RecordEventCommand) -> RecordResult:
        """Insert one ledger event and fold it into its minute rollup atomically.

        Idempotent retries of the same ``(operation_id, attempt)`` resolve
        to the same ``event_key``; a conflict there means the attempt was
        already recorded, so the rollup upsert is skipped entirely and the
        transaction rolls back to a no-op read.
        """
        started_at = command.started_at
        bucket_start_utc = started_at.astimezone(timezone.utc).replace(second=0, microsecond=0)

        event_values: dict[str, Any] = {
            "event_key": command.event_key,
            "operation_id": command.operation_id,
            "attempt": command.attempt,
            "user_id": command.context.user_id,
            "conversation_id": command.context.conversation_id,
            "request_message_id": command.context.request_message_id,
            "document_id": command.context.document_id,
            "correlation_id": command.context.correlation_id,
            "langsmith_run_id": command.context.langsmith_run_id,
            "provider_request_id": command.provider_request_id,
            "provider": command.provider,
            "model": command.model,
            "operation": command.context.operation,
            "agent_id": command.context.agent_id,
            "status": command.status,
            "usage_source": command.usage.source,
            "generated_images": command.usage.generated_images,
            "latency_ms": command.latency_ms,
            "error_code": command.error_code,
            "started_at": started_at,
            "completed_at": command.completed_at,
        }
        for field in NULLABLE_TOKEN_FIELDS:
            event_values[field] = getattr(command.usage, field)

        event_insert = (
            insert(ModelUsageEvent)
            .values(**event_values)
            .on_conflict_do_nothing(index_elements=["event_key"])
            .returning(ModelUsageEvent.id)
        )

        rollup_key = compute_rollup_key(
            bucket_start_utc=bucket_start_utc,
            user_id=command.context.user_id,
            conversation_id=command.context.conversation_id,
            provider=command.provider,
            model=command.model,
            operation=command.context.operation,
            agent_id=command.context.agent_id,
            status=command.status,
            usage_source=command.usage.source,
        )
        minute_upsert = self._minute_upsert_statement(
            rollup_key=rollup_key,
            bucket_start_utc=bucket_start_utc,
            command=command,
        )

        with self.session_factory() as session:
            try:
                session.execute(
                    select(func.pg_advisory_xact_lock_shared(_minute_lock_key(bucket_start_utc)))
                )
                inserted_id = session.execute(event_insert).scalar_one_or_none()
            except IntegrityError as exc:
                session.rollback()
                raise ModelUsageReferenceError(
                    "record_event references a row that does not exist yet "
                    "(user_id, conversation_id, request_message_id, or document_id)"
                ) from exc
            if inserted_id is None:
                session.rollback()
                return RecordResult(inserted=False, event_id=None)
            session.execute(minute_upsert)
            session.commit()
            return RecordResult(inserted=True, event_id=inserted_id)

    @staticmethod
    def _minute_upsert_statement(
        *, rollup_key: str, bucket_start_utc: datetime, command: RecordEventCommand
    ):
        table = ModelUsageMinute.__table__
        contribution: dict[str, Any] = {
            "rollup_key": rollup_key,
            "bucket_start_utc": bucket_start_utc,
            "user_id": command.context.user_id,
            "conversation_id": command.context.conversation_id,
            "provider": command.provider,
            "model": command.model,
            "operation": command.context.operation,
            "agent_id": command.context.agent_id,
            "status": command.status,
            "usage_source": command.usage.source,
            "request_count": 1,
            "latency_ms_sum": command.latency_ms,
            "generated_images_sum": command.usage.generated_images,
        }
        increment_columns = ["request_count", "latency_ms_sum", "generated_images_sum"]
        for field in NULLABLE_TOKEN_FIELDS:
            value = getattr(command.usage, field)
            contribution[f"{field}_sum"] = value if value is not None else 0
            contribution[f"{field}_known_count"] = 0 if value is None else 1
            increment_columns.append(f"{field}_sum")
            increment_columns.append(f"{field}_known_count")

        incoming = insert(table).values(**contribution)
        aggregate_increments = {
            column: table.c[column] + incoming.excluded[column] for column in increment_columns
        }
        return incoming.on_conflict_do_update(
            index_elements=["rollup_key"],
            set_=aggregate_increments,
        )

    # -- Bounded queries (every method requires a non-null user_id) ---

    def get_summary_totals(
        self,
        *,
        user_id: UUID,
        start_inclusive: datetime,
        end_exclusive: datetime,
        conversation_id: UUID | None = None,
    ) -> UsageTotals:
        """Return summed totals across every minute rollup in the window."""
        if user_id is None:
            raise ValueError("get_summary_totals requires a non-null user_id")
        start = _require_aware(start_inclusive, "start_inclusive")
        end = _require_aware(end_exclusive, "end_exclusive")
        statement = select(*_sum_expressions()).where(
            ModelUsageMinute.user_id == user_id,
            ModelUsageMinute.bucket_start_utc >= start,
            ModelUsageMinute.bucket_start_utc < end,
        )
        if conversation_id is not None:
            statement = statement.where(ModelUsageMinute.conversation_id == conversation_id)
        with self.session_factory() as session:
            row = session.execute(statement).mappings().one()
        return _totals_from_mapping(row)

    def get_minute_series(
        self,
        *,
        user_id: UUID,
        start_inclusive: datetime,
        end_exclusive: datetime,
        conversation_id: UUID | None = None,
    ) -> list[ModelUsageMinute]:
        """Return raw per-minute rollup rows in the window, oldest first."""
        if user_id is None:
            raise ValueError("get_minute_series requires a non-null user_id")
        start = _require_aware(start_inclusive, "start_inclusive")
        end = _require_aware(end_exclusive, "end_exclusive")
        statement = (
            select(ModelUsageMinute)
            .where(
                ModelUsageMinute.user_id == user_id,
                ModelUsageMinute.bucket_start_utc >= start,
                ModelUsageMinute.bucket_start_utc < end,
            )
            .order_by(ModelUsageMinute.bucket_start_utc)
        )
        if conversation_id is not None:
            statement = statement.where(ModelUsageMinute.conversation_id == conversation_id)
        with self.session_factory() as session:
            rows = list(session.execute(statement).scalars().all())
            for row in rows:
                session.expunge(row)
            return rows

    def get_bucket_series(
        self,
        *,
        user_id: UUID,
        bucket_intervals: Sequence[tuple[datetime, datetime]],
        conversation_id: UUID | None = None,
    ) -> list[BucketUsageTotals]:
        """Aggregate usage into ordered UTC intervals in one bounded SQL query."""
        if user_id is None:
            raise ValueError("get_bucket_series requires a non-null user_id")
        if not bucket_intervals:
            return []
        if len(bucket_intervals) > 1500:
            raise ValueError("get_bucket_series accepts at most 1500 intervals")

        normalized: list[tuple[int, datetime, datetime]] = []
        previous_end: datetime | None = None
        for index, (raw_start, raw_end) in enumerate(bucket_intervals):
            start = _require_minute_aligned_utc(raw_start, "bucket interval start")
            end = _require_minute_aligned_utc(raw_end, "bucket interval end")
            if end <= start:
                raise ValueError("bucket intervals must have end after start")
            if previous_end is not None and start < previous_end:
                raise ValueError("bucket intervals must be ordered and non-overlapping")
            normalized.append((index, start, end))
            previous_end = end

        interval_values = (
            values(
                column("bucket_index", Integer),
                column("bucket_start_utc", DateTime(timezone=True)),
                column("bucket_end_utc", DateTime(timezone=True)),
                name="usage_bucket_values",
            )
            .data(normalized)
            .cte("usage_buckets")
        )
        statement = (
            select(
                interval_values.c.bucket_start_utc,
                interval_values.c.bucket_end_utc,
                *_sum_expressions(),
            )
            .select_from(
                interval_values.join(
                    ModelUsageMinute,
                    and_(
                        ModelUsageMinute.bucket_start_utc >= interval_values.c.bucket_start_utc,
                        ModelUsageMinute.bucket_start_utc < interval_values.c.bucket_end_utc,
                    ),
                )
            )
            .where(ModelUsageMinute.user_id == user_id)
            .group_by(
                interval_values.c.bucket_index,
                interval_values.c.bucket_start_utc,
                interval_values.c.bucket_end_utc,
            )
            .order_by(interval_values.c.bucket_index)
        )
        if conversation_id is not None:
            statement = statement.where(ModelUsageMinute.conversation_id == conversation_id)
        with self.session_factory() as session:
            rows = session.execute(statement).mappings().all()
        return [
            BucketUsageTotals(
                bucket_start_utc=row["bucket_start_utc"],
                bucket_end_utc=row["bucket_end_utc"],
                totals=_totals_from_mapping(row),
            )
            for row in rows
        ]

    def get_dimension_breakdown(
        self,
        *,
        user_id: UUID,
        start_inclusive: datetime,
        end_exclusive: datetime,
        dimension: UsageDimension,
        conversation_id: UUID | None = None,
        limit: int | None = None,
    ) -> list[DimensionUsageTotals]:
        """Return totals grouped by one dimension column within the window."""
        if user_id is None:
            raise ValueError("get_dimension_breakdown requires a non-null user_id")
        if dimension not in _DIMENSION_COLUMNS:
            raise ValueError(f"Unsupported usage dimension: {dimension!r}")
        if limit is not None and not 1 <= limit <= 20:
            raise ValueError("get_dimension_breakdown limit must be between 1 and 20")
        start = _require_aware(start_inclusive, "start_inclusive")
        end = _require_aware(end_exclusive, "end_exclusive")
        raw_column = _DIMENSION_COLUMNS[dimension]
        public_column = func.coalesce(raw_column, "unknown")
        statement = (
            select(public_column.label("dimension_value"), *_sum_expressions())
            .where(
                ModelUsageMinute.user_id == user_id,
                ModelUsageMinute.bucket_start_utc >= start,
                ModelUsageMinute.bucket_start_utc < end,
            )
            .group_by(public_column)
        )
        if conversation_id is not None:
            statement = statement.where(ModelUsageMinute.conversation_id == conversation_id)
        if limit is None:
            statement = statement.order_by(public_column)
        else:
            statement = statement.order_by(desc("total_tokens_sum"), public_column.asc()).limit(
                limit
            )
        with self.session_factory() as session:
            rows = session.execute(statement).mappings().all()
        return [
            DimensionUsageTotals(
                dimension_value=row["dimension_value"],
                totals=_totals_from_mapping(row),
            )
            for row in rows
        ]

    def get_top_conversations(
        self,
        *,
        user_id: UUID,
        start_inclusive: datetime,
        end_exclusive: datetime,
        conversation_id: UUID | None = None,
        limit: int = 20,
    ) -> list[ConversationUsageTotals]:
        """Return the highest known-token conversation totals, bounded in SQL."""
        if user_id is None:
            raise ValueError("get_top_conversations requires a non-null user_id")
        if not 1 <= limit <= 20:
            raise ValueError("get_top_conversations limit must be between 1 and 20")
        start = _require_aware(start_inclusive, "start_inclusive")
        end = _require_aware(end_exclusive, "end_exclusive")
        statement = (
            select(
                ModelUsageMinute.conversation_id.label("conversation_id"),
                Conversation.title.label("title"),
                *_sum_expressions(),
            )
            .join(Conversation, Conversation.id == ModelUsageMinute.conversation_id)
            .where(
                ModelUsageMinute.user_id == user_id,
                Conversation.owner_id == user_id,
                Conversation.deleted_at.is_(None),
                ModelUsageMinute.conversation_id.is_not(None),
                ModelUsageMinute.bucket_start_utc >= start,
                ModelUsageMinute.bucket_start_utc < end,
            )
            .group_by(ModelUsageMinute.conversation_id, Conversation.title)
            .order_by(
                desc("total_tokens_sum"),
                ModelUsageMinute.conversation_id.asc(),
            )
            .limit(limit)
        )
        if conversation_id is not None:
            statement = statement.where(ModelUsageMinute.conversation_id == conversation_id)
        with self.session_factory() as session:
            rows = session.execute(statement).mappings().all()
        return [
            ConversationUsageTotals(
                conversation_id=row["conversation_id"],
                title=row["title"],
                totals=_totals_from_mapping(row),
            )
            for row in rows
        ]

    def get_latest_conversation_context_window(
        self, *, user_id: UUID, conversation_id: UUID
    ) -> dict[str, Any] | None:
        """Return context metadata from the latest visible assistant message."""
        if user_id is None:
            raise ValueError("get_latest_conversation_context_window requires a non-null user_id")
        statement = (
            select(Message.message_metadata["context_window"])
            .join(Conversation, Conversation.id == Message.conversation_id)
            .where(
                Conversation.owner_id == user_id,
                Conversation.id == conversation_id,
                Conversation.deleted_at.is_(None),
                Message.deleted_at.is_(None),
                Message.sender == MessageRole.assistant.value,
                Message.message_metadata.op("?")("context_window"),
            )
            .order_by(
                Message.sequence.desc(),
                Message.id.desc(),
            )
            .limit(1)
        )
        with self.session_factory() as session:
            value = session.execute(statement).scalar_one_or_none()
        return dict(value) if isinstance(value, dict) else None

    # -- Maintenance (not user-scoped: whole-table reconciliation/cleanup) --

    def reconcile_minute_range(
        self,
        *,
        start_inclusive: datetime,
        end_exclusive: datetime,
        chunk_minutes: int = 60,
        insert_batch_size: int = 1_000,
    ) -> int:
        """Rebuild every rollup whose bucket falls in the range from raw events.

        Discards whatever rollup state currently exists for
        ``[start_inclusive, end_exclusive)`` -- including rows with no
        backing events left -- and recomputes it directly from
        ``model_usage_events``, so the result is exact rather than merely
        corrected. Bounds must already be minute-aligned UTC instants.
        Returns the number of rollup rows written.
        """
        if chunk_minutes < 1:
            raise ValueError("chunk_minutes must be >= 1")
        if insert_batch_size < 1:
            raise ValueError("insert_batch_size must be >= 1")
        chunk_start = _require_minute_aligned_utc(start_inclusive, "start_inclusive")
        end = _require_minute_aligned_utc(end_exclusive, "end_exclusive")
        if end < chunk_start:
            raise ValueError("end_exclusive must not precede start_inclusive")
        written = 0
        while chunk_start < end:
            chunk_end = min(end, chunk_start + timedelta(minutes=chunk_minutes))
            written += self._reconcile_minute_chunk(
                start_inclusive=chunk_start,
                end_exclusive=chunk_end,
                insert_batch_size=insert_batch_size,
            )
            chunk_start = chunk_end
        return written

    def _reconcile_minute_chunk(
        self,
        *,
        start_inclusive: datetime,
        end_exclusive: datetime,
        insert_batch_size: int,
    ) -> int:
        """Atomically replace one bounded time chunk from server aggregates."""
        bucket = func.date_trunc("minute", ModelUsageEvent.started_at, "UTC").label(
            "bucket_start_utc"
        )
        dimensions = (
            ModelUsageEvent.user_id,
            ModelUsageEvent.conversation_id,
            ModelUsageEvent.provider,
            ModelUsageEvent.model,
            ModelUsageEvent.operation,
            ModelUsageEvent.agent_id,
            ModelUsageEvent.status,
            ModelUsageEvent.usage_source,
        )
        aggregate_expressions = [
            func.count(ModelUsageEvent.id).label("request_count"),
            func.coalesce(func.sum(ModelUsageEvent.latency_ms), 0).label("latency_ms_sum"),
            func.coalesce(func.sum(ModelUsageEvent.generated_images), 0).label(
                "generated_images_sum"
            ),
        ]
        for field in NULLABLE_TOKEN_FIELDS:
            event_column = getattr(ModelUsageEvent, field)
            aggregate_expressions.extend(
                (
                    func.coalesce(func.sum(event_column), 0).label(f"{field}_sum"),
                    func.count(event_column).label(f"{field}_known_count"),
                )
            )
        aggregate_statement = (
            select(bucket, *dimensions, *aggregate_expressions)
            .where(
                ModelUsageEvent.started_at >= start_inclusive,
                ModelUsageEvent.started_at < end_exclusive,
            )
            .group_by(bucket, *dimensions)
            .order_by(bucket)
            .execution_options(stream_results=True, yield_per=insert_batch_size)
        )

        with self.session_factory() as session:
            session.execute(
                text(
                    "SELECT pg_advisory_xact_lock("
                    "(:namespace * 4294967296::bigint) + "
                    "floor(extract(epoch FROM minute_at) / 60)::bigint) "
                    "FROM generate_series("
                    ":start_inclusive, "
                    ":end_exclusive - interval '1 minute', "
                    "interval '1 minute'"
                    ") AS minute_at "
                    "ORDER BY minute_at"
                ),
                {
                    "namespace": _MINUTE_LOCK_NAMESPACE,
                    "start_inclusive": start_inclusive,
                    "end_exclusive": end_exclusive,
                },
            ).all()
            session.execute(
                delete(ModelUsageMinute).where(
                    ModelUsageMinute.bucket_start_utc >= start_inclusive,
                    ModelUsageMinute.bucket_start_utc < end_exclusive,
                )
            )
            aggregates = session.execute(aggregate_statement).mappings()

            def rebuilt_rows() -> Iterator[dict[str, Any]]:
                for aggregate in aggregates:
                    row = dict(aggregate)
                    row["rollup_key"] = compute_rollup_key(
                        bucket_start_utc=row["bucket_start_utc"],
                        user_id=row["user_id"],
                        conversation_id=row["conversation_id"],
                        provider=row["provider"],
                        model=row["model"],
                        operation=row["operation"],
                        agent_id=row["agent_id"],
                        status=row["status"],
                        usage_source=row["usage_source"],
                    )
                    yield row

            written = 0
            for batch in _partition_rows(rebuilt_rows(), batch_size=insert_batch_size):
                session.execute(insert(ModelUsageMinute), batch)
                written += len(batch)
            session.commit()
            return written

    def delete_raw_events_older_than(self, cutoff: datetime, *, batch_size: int = 500) -> int:
        """Delete ``model_usage_events`` rows started before ``cutoff``, in batches."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        cutoff = _require_aware(cutoff, "cutoff")
        total_deleted = 0
        with self.session_factory() as session:
            while True:
                batch_ids = list(
                    session.execute(
                        select(ModelUsageEvent.id)
                        .where(ModelUsageEvent.started_at < cutoff)
                        .order_by(ModelUsageEvent.started_at)
                        .limit(batch_size)
                    ).scalars()
                )
                if not batch_ids:
                    break
                session.execute(delete(ModelUsageEvent).where(ModelUsageEvent.id.in_(batch_ids)))
                session.commit()
                total_deleted += len(batch_ids)
                if len(batch_ids) < batch_size:
                    break
        return total_deleted

    def delete_rollups_older_than(self, cutoff: datetime, *, batch_size: int = 500) -> int:
        """Delete ``model_usage_minute`` rows bucketed before ``cutoff``, in batches."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        cutoff = _require_aware(cutoff, "cutoff")
        total_deleted = 0
        with self.session_factory() as session:
            while True:
                batch_keys = list(
                    session.execute(
                        select(ModelUsageMinute.rollup_key)
                        .where(ModelUsageMinute.bucket_start_utc < cutoff)
                        .order_by(ModelUsageMinute.bucket_start_utc)
                        .limit(batch_size)
                    ).scalars()
                )
                if not batch_keys:
                    break
                session.execute(
                    delete(ModelUsageMinute).where(ModelUsageMinute.rollup_key.in_(batch_keys))
                )
                session.commit()
                total_deleted += len(batch_keys)
                if len(batch_keys) < batch_size:
                    break
        return total_deleted

    def get_model_usage_health_snapshot(
        self,
        *,
        now: datetime,
        lookback_minutes: int,
    ) -> dict[str, Any]:
        """Return bounded, content-free ledger/rollup health aggregates.

        Only complete UTC minutes are inspected.  The result contains counts
        and timestamps, never tenant, conversation, model, request, or trace
        dimensions, so it is safe for an unauthenticated operational endpoint.
        """
        if lookback_minutes < 1:
            raise ValueError("lookback_minutes must be >= 1")
        complete_minute = (
            _require_aware(now, "now").astimezone(timezone.utc).replace(second=0, microsecond=0)
        )
        start = complete_minute - timedelta(minutes=lookback_minutes)
        raw_statement = select(
            func.count(ModelUsageEvent.id).label("raw_event_count"),
            func.count(ModelUsageEvent.id)
            .filter(ModelUsageEvent.user_id.is_(None))
            .label("unattributed_event_count"),
            func.max(func.date_trunc("minute", ModelUsageEvent.started_at)).label(
                "latest_event_minute"
            ),
        ).where(
            ModelUsageEvent.started_at >= start,
            ModelUsageEvent.started_at < complete_minute,
        )
        rollup_statement = select(
            func.coalesce(func.sum(ModelUsageMinute.request_count), 0).label(
                "rollup_request_count"
            ),
            func.max(ModelUsageMinute.bucket_start_utc).label("latest_rollup_minute"),
        ).where(
            ModelUsageMinute.bucket_start_utc >= start,
            ModelUsageMinute.bucket_start_utc < complete_minute,
        )
        with self.session_factory() as session:
            raw = session.execute(raw_statement).mappings().one()
            rollup = session.execute(rollup_statement).mappings().one()
        return {
            "raw_event_count": int(raw["raw_event_count"] or 0),
            "unattributed_event_count": int(raw["unattributed_event_count"] or 0),
            "rollup_request_count": int(rollup["rollup_request_count"] or 0),
            "latest_event_minute": raw["latest_event_minute"],
            "latest_rollup_minute": rollup["latest_rollup_minute"],
        }
