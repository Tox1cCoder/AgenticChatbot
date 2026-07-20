"""Per-user model-usage analytics ledger.

``ModelUsageEvent`` is an immutable per-attempt row written once the
outcome of a provider call is known. ``ModelUsageMinute`` is the UTC-minute
rollup keyed by a deterministic ``rollup_key`` hash so that dimensions with
``NULL`` values (e.g. no conversation) still dedupe correctly under
PostgreSQL's distinct-null uniqueness semantics.
"""

import uuid

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base

# Token/image fields whose per-event value may be unknown (NULL) and whose
# minute rollup therefore tracks a companion "known" coverage count.
NULLABLE_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "input_text_tokens",
    "input_image_tokens",
    "output_text_tokens",
    "output_image_tokens",
)


def _token_check_constraints(table_name: str) -> list[CheckConstraint]:
    return [
        CheckConstraint(
            f"{field} IS NULL OR {field} >= 0",
            name=f"ck_{table_name}_{field}_nonnegative",
        )
        for field in NULLABLE_TOKEN_FIELDS
    ]


# model_usage_minute's check-constraint names use a short, abbreviated scheme
# ("ck_mu_minute_<column>_nonneg" instead of
# "ck_model_usage_minute_<column>_nonnegative") because the full-length scheme
# pushes several *_known_count names past PostgreSQL's 63-character
# identifier limit (e.g. "cached_input_tokens_known_count" would produce a
# 65-character name). Keep this prefix/suffix pair in sync with the migration
# in app/alembic/versions/y2z3a4b5c6d7_add_model_usage_ledger.py.
_MINUTE_CHECK_PREFIX = "ck_mu_minute_"
_MINUTE_CHECK_SUFFIX = "_nonneg"


def _minute_check_name(column_name: str) -> str:
    return f"{_MINUTE_CHECK_PREFIX}{column_name}{_MINUTE_CHECK_SUFFIX}"


def _rollup_sum_check_constraints() -> list[CheckConstraint]:
    checks = []
    for field in NULLABLE_TOKEN_FIELDS:
        for suffix in ("sum", "known_count"):
            column_name = f"{field}_{suffix}"
            checks.append(
                CheckConstraint(
                    f"{column_name} >= 0",
                    name=_minute_check_name(column_name),
                )
            )
    return checks


class ModelUsageEvent(Base):
    """Immutable ledger row for a single application-controlled provider attempt."""

    __tablename__ = "model_usage_events"
    __table_args__ = (
        UniqueConstraint("event_key", name="uq_model_usage_events_event_key"),
        UniqueConstraint("operation_id", "attempt", name="uq_model_usage_events_operation_attempt"),
        CheckConstraint("attempt >= 1", name="ck_model_usage_events_attempt_positive"),
        CheckConstraint(
            "generated_images >= 0",
            name="ck_model_usage_events_generated_images_nonnegative",
        ),
        CheckConstraint("latency_ms >= 0", name="ck_model_usage_events_latency_ms_nonnegative"),
        *_token_check_constraints("model_usage_events"),
        Index("ix_model_usage_events_user_started", "user_id", "started_at"),
        Index(
            "ix_model_usage_events_conversation_started",
            "conversation_id",
            "started_at",
        ),
        Index(
            "ix_model_usage_events_provider_model_started",
            "provider",
            "model",
            "started_at",
        ),
        Index("ix_model_usage_events_operation_started", "operation", "started_at"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Idempotency: same normalized command replayed after a ledger-write
    # failure resolves to the same event_key ("<operation_id>:<attempt>").
    event_key = Column(String(160), nullable=False)
    operation_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    attempt = Column(Integer, nullable=False)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL")
    )
    request_message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id", ondelete="SET NULL"))
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"))

    correlation_id = Column(String(128), nullable=True)
    langsmith_run_id = Column(UUID(as_uuid=True), nullable=True)
    provider_request_id = Column(String(255), nullable=True)

    provider = Column(String(32), nullable=False)
    model = Column(String(255), nullable=False)
    operation = Column(String(64), nullable=False)
    agent_id = Column(String(128), nullable=True)
    status = Column(String(16), nullable=False)
    usage_source = Column(String(32), nullable=False)

    input_tokens = Column(BigInteger, nullable=True)
    output_tokens = Column(BigInteger, nullable=True)
    total_tokens = Column(BigInteger, nullable=True)
    reasoning_tokens = Column(BigInteger, nullable=True)
    cached_input_tokens = Column(BigInteger, nullable=True)
    input_text_tokens = Column(BigInteger, nullable=True)
    input_image_tokens = Column(BigInteger, nullable=True)
    output_text_tokens = Column(BigInteger, nullable=True)
    output_image_tokens = Column(BigInteger, nullable=True)

    generated_images = Column(Integer, nullable=False)
    latency_ms = Column(BigInteger, nullable=False)

    error_code = Column(String(128), nullable=True)

    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    user = relationship("User", back_populates="usage_events")

    def __repr__(self) -> str:
        return (
            f"<ModelUsageEvent(id={self.id}, operation_id={self.operation_id}, "
            f"attempt={self.attempt}, status={self.status!r})>"
        )


class ModelUsageMinute(Base):
    """UTC-minute aggregate keyed by a deterministic SHA-256 rollup_key.

    ``rollup_key`` is derived from a versioned, ordered JSON array of
    ``bucket_start_utc`` plus every dimension (with explicit ``null``
    entries), so it is the primary key: a single non-null column avoids
    PostgreSQL's distinct-null uniqueness behavior for a composite key that
    otherwise includes nullable dimensions such as ``conversation_id``.
    """

    __tablename__ = "model_usage_minute"
    __table_args__ = (
        *_rollup_sum_check_constraints(),
        CheckConstraint(
            "generated_images_sum >= 0",
            name=_minute_check_name("generated_images_sum"),
        ),
        CheckConstraint("request_count >= 0", name=_minute_check_name("request_count")),
        CheckConstraint("latency_ms_sum >= 0", name=_minute_check_name("latency_ms_sum")),
        Index("ix_model_usage_minute_user_bucket", "user_id", "bucket_start_utc"),
        Index(
            "ix_model_usage_minute_conversation_bucket",
            "conversation_id",
            "bucket_start_utc",
        ),
    )

    rollup_key = Column(String(64), primary_key=True)

    bucket_start_utc = Column(DateTime(timezone=True), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"))
    provider = Column(String(32), nullable=False)
    model = Column(String(255), nullable=False)
    operation = Column(String(64), nullable=False)
    agent_id = Column(String(128), nullable=True)
    status = Column(String(16), nullable=False)
    usage_source = Column(String(32), nullable=False)

    input_tokens_sum = Column(BigInteger, nullable=False, default=0, server_default=text("0"))
    input_tokens_known_count = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    output_tokens_sum = Column(BigInteger, nullable=False, default=0, server_default=text("0"))
    output_tokens_known_count = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    total_tokens_sum = Column(BigInteger, nullable=False, default=0, server_default=text("0"))
    total_tokens_known_count = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    reasoning_tokens_sum = Column(BigInteger, nullable=False, default=0, server_default=text("0"))
    reasoning_tokens_known_count = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    cached_input_tokens_sum = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    cached_input_tokens_known_count = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    input_text_tokens_sum = Column(BigInteger, nullable=False, default=0, server_default=text("0"))
    input_text_tokens_known_count = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    input_image_tokens_sum = Column(BigInteger, nullable=False, default=0, server_default=text("0"))
    input_image_tokens_known_count = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    output_text_tokens_sum = Column(BigInteger, nullable=False, default=0, server_default=text("0"))
    output_text_tokens_known_count = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    output_image_tokens_sum = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    output_image_tokens_known_count = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )

    generated_images_sum = Column(BigInteger, nullable=False, default=0, server_default=text("0"))
    request_count = Column(BigInteger, nullable=False, default=0, server_default=text("0"))
    latency_ms_sum = Column(BigInteger, nullable=False, default=0, server_default=text("0"))

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    def __repr__(self) -> str:
        return (
            f"<ModelUsageMinute(rollup_key={self.rollup_key!r}, "
            f"bucket_start_utc={self.bucket_start_utc}, request_count={self.request_count})>"
        )
