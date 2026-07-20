"""add model usage ledger (model_usage_events, model_usage_minute)

Revision ID: y2z3a4b5c6d7
Revises: x1y2z3a4b5c6
Create Date: 2026-07-20 00:00:00.000000

Adds the per-user model-usage analytics persistence layer:

- ``model_usage_events``: immutable per-attempt provider-call ledger.
- ``model_usage_minute``: UTC-minute rollup keyed by a deterministic
  ``rollup_key`` (SHA-256 hex) instead of a composite key, since several
  rollup dimensions are nullable and PostgreSQL treats NULLs as distinct
  under a plain UNIQUE constraint.

No data backfill: both tables start empty.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = "y2z3a4b5c6d7"
down_revision: str | Sequence[str] | None = "x1y2z3a4b5c6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Token/image fields whose per-event value may be unknown (NULL) and whose
# minute rollup therefore tracks a companion "known" coverage count.
_NULLABLE_TOKEN_FIELDS = (
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


def _event_token_columns() -> list[sa.Column]:
    return [sa.Column(field, sa.BigInteger(), nullable=True) for field in _NULLABLE_TOKEN_FIELDS]


def _event_token_checks() -> list[sa.CheckConstraint]:
    return [
        sa.CheckConstraint(
            f"{field} IS NULL OR {field} >= 0",
            name=f"ck_model_usage_events_{field}_nonnegative",
        )
        for field in _NULLABLE_TOKEN_FIELDS
    ]


def _rollup_sum_columns() -> list[sa.Column]:
    columns = []
    for field in _NULLABLE_TOKEN_FIELDS:
        columns.append(
            sa.Column(f"{field}_sum", sa.BigInteger(), nullable=False, server_default=sa.text("0"))
        )
        columns.append(
            sa.Column(
                f"{field}_known_count",
                sa.BigInteger(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
    return columns


# model_usage_minute's check-constraint names use a short, abbreviated scheme
# ("ck_mu_minute_<column>_nonneg" instead of
# "ck_model_usage_minute_<column>_nonnegative") because the full-length scheme
# pushes several *_known_count names past PostgreSQL's 63-character
# identifier limit (e.g. "cached_input_tokens_known_count" would produce a
# 65-character name and SQLAlchemy raises IdentifierError on DDL emission).
# Keep this prefix/suffix pair in sync with app/models/model_usage.py.
_MINUTE_CHECK_PREFIX = "ck_mu_minute_"
_MINUTE_CHECK_SUFFIX = "_nonneg"


def _minute_check_name(column_name: str) -> str:
    return f"{_MINUTE_CHECK_PREFIX}{column_name}{_MINUTE_CHECK_SUFFIX}"


def _rollup_sum_checks() -> list[sa.CheckConstraint]:
    checks = []
    for field in _NULLABLE_TOKEN_FIELDS:
        for suffix in ("sum", "known_count"):
            column_name = f"{field}_{suffix}"
            checks.append(
                sa.CheckConstraint(
                    f"{column_name} >= 0",
                    name=_minute_check_name(column_name),
                )
            )
    return checks


def upgrade() -> None:
    op.create_table(
        "model_usage_events",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("event_key", sa.String(length=160), nullable=False),
        sa.Column("operation_id", UUID(as_uuid=True), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), nullable=True),
        sa.Column("conversation_id", UUID(as_uuid=True), nullable=True),
        sa.Column("request_message_id", UUID(as_uuid=True), nullable=True),
        sa.Column("document_id", UUID(as_uuid=True), nullable=True),
        sa.Column("correlation_id", sa.String(length=128), nullable=True),
        sa.Column("langsmith_run_id", UUID(as_uuid=True), nullable=True),
        sa.Column("provider_request_id", sa.String(length=255), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("operation", sa.String(length=64), nullable=False),
        sa.Column("agent_id", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("usage_source", sa.String(length=32), nullable=False),
        *_event_token_columns(),
        sa.Column("generated_images", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.BigInteger(), nullable=False),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_model_usage_events"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_model_usage_events_user",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_model_usage_events_conversation",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["request_message_id"],
            ["messages.id"],
            name="fk_model_usage_events_request_message",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_model_usage_events_document",
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint("event_key", name="uq_model_usage_events_event_key"),
        sa.UniqueConstraint(
            "operation_id",
            "attempt",
            name="uq_model_usage_events_operation_attempt",
        ),
        sa.CheckConstraint("attempt >= 1", name="ck_model_usage_events_attempt_positive"),
        sa.CheckConstraint(
            "generated_images >= 0",
            name="ck_model_usage_events_generated_images_nonnegative",
        ),
        sa.CheckConstraint("latency_ms >= 0", name="ck_model_usage_events_latency_ms_nonnegative"),
        *_event_token_checks(),
    )

    op.create_table(
        "model_usage_minute",
        sa.Column("rollup_key", sa.String(length=64), nullable=False),
        sa.Column("bucket_start_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), nullable=True),
        sa.Column("conversation_id", UUID(as_uuid=True), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("operation", sa.String(length=64), nullable=False),
        sa.Column("agent_id", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("usage_source", sa.String(length=32), nullable=False),
        *_rollup_sum_columns(),
        sa.Column(
            "generated_images_sum",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("request_count", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("latency_ms_sum", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("rollup_key", name="pk_model_usage_minute"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_model_usage_minute_user",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_model_usage_minute_conversation",
            ondelete="CASCADE",
        ),
        *_rollup_sum_checks(),
        sa.CheckConstraint(
            "generated_images_sum >= 0",
            name=_minute_check_name("generated_images_sum"),
        ),
        sa.CheckConstraint(
            "request_count >= 0",
            name=_minute_check_name("request_count"),
        ),
        sa.CheckConstraint(
            "latency_ms_sum >= 0",
            name=_minute_check_name("latency_ms_sum"),
        ),
    )

    op.create_index(
        "ix_model_usage_events_operation_id",
        "model_usage_events",
        ["operation_id"],
        unique=False,
    )
    op.create_index(
        "ix_model_usage_events_user_started",
        "model_usage_events",
        ["user_id", "started_at"],
        unique=False,
    )
    op.create_index(
        "ix_model_usage_events_conversation_started",
        "model_usage_events",
        ["conversation_id", "started_at"],
        unique=False,
    )
    op.create_index(
        "ix_model_usage_events_provider_model_started",
        "model_usage_events",
        ["provider", "model", "started_at"],
        unique=False,
    )
    op.create_index(
        "ix_model_usage_events_operation_started",
        "model_usage_events",
        ["operation", "started_at"],
        unique=False,
    )
    op.create_index(
        "ix_model_usage_minute_user_bucket",
        "model_usage_minute",
        ["user_id", "bucket_start_utc"],
        unique=False,
    )
    op.create_index(
        "ix_model_usage_minute_conversation_bucket",
        "model_usage_minute",
        ["conversation_id", "bucket_start_utc"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_model_usage_minute_conversation_bucket", table_name="model_usage_minute")
    op.drop_index("ix_model_usage_minute_user_bucket", table_name="model_usage_minute")
    op.drop_table("model_usage_minute")

    op.drop_index("ix_model_usage_events_operation_started", table_name="model_usage_events")
    op.drop_index("ix_model_usage_events_provider_model_started", table_name="model_usage_events")
    op.drop_index("ix_model_usage_events_conversation_started", table_name="model_usage_events")
    op.drop_index("ix_model_usage_events_user_started", table_name="model_usage_events")
    op.drop_index("ix_model_usage_events_operation_id", table_name="model_usage_events")
    op.drop_table("model_usage_events")
