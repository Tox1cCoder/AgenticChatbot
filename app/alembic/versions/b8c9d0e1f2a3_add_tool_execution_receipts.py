"""add durable tool execution receipts

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-09-03 00:00:00.000000

A LangGraph checkpoint is written after a node returns, so a mutation that
reached its provider and then lost the process leaves no record and the replay
calls the provider again. This table is reserved before the call and completed
alongside the effect, which is what makes the two observable as one.

The unique index on ``execution_key`` is the mechanism rather than a safety
net: two concurrent replays race to insert, one wins, and the loser reads the
winner's row instead of invoking.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ENUM, JSONB, UUID

revision: str = "b8c9d0e1f2a3"
down_revision: str | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_STATUS_VALUES = ("reserved", "completed", "failed", "outcome_unknown")

# ``create_type=False`` is load-bearing. ``op.create_table`` auto-creates any
# enum type a column references, with no ``checkfirst``, so leaving it True
# makes the CREATE TYPE run twice and the migration abort on DuplicateObject.
# The type is created once, explicitly, below.
_STATUS_ENUM = ENUM(*_STATUS_VALUES, name="tool_execution_receipt_status", create_type=False)


def upgrade() -> None:
    ENUM(*_STATUS_VALUES, name="tool_execution_receipt_status").create(
        op.get_bind(), checkfirst=True
    )

    op.create_table(
        "tool_execution_receipts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("execution_key", sa.String(length=64), nullable=False),
        sa.Column("status", _STATUS_ENUM, nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", UUID(as_uuid=True), nullable=False),
        sa.Column("turn_id", sa.String(length=160), nullable=False),
        sa.Column("thread_id", sa.String(length=320), nullable=False),
        sa.Column("dispatch_id", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.String(length=160), nullable=False),
        sa.Column("tool_call_id", sa.String(length=255), nullable=False),
        sa.Column("qualified_tool_id", sa.String(length=512), nullable=False),
        sa.Column("provider_idempotency", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("result_json", JSONB, nullable=True),
        sa.Column("artifact_ref", sa.String(length=512), nullable=True),
        sa.Column("provider_receipt_id", sa.String(length=255), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
    )

    # One row per execution identity. Everything else in the design depends on
    # this being enforced by the database rather than by a read-then-write.
    op.create_index(
        "uq_tool_execution_receipts_execution_key",
        "tool_execution_receipts",
        ["execution_key"],
        unique=True,
    )
    op.create_index("ix_tool_execution_receipts_user_id", "tool_execution_receipts", ["user_id"])
    op.create_index(
        "ix_tool_execution_receipts_conversation_id",
        "tool_execution_receipts",
        ["conversation_id"],
    )
    op.create_index(
        "ix_tool_execution_receipts_thread_id", "tool_execution_receipts", ["thread_id"]
    )
    op.create_index(
        "ix_tool_execution_receipts_qualified_tool_id",
        "tool_execution_receipts",
        ["qualified_tool_id"],
    )
    op.create_index(
        "ix_tool_execution_receipts_owner_status",
        "tool_execution_receipts",
        ["user_id", "status"],
    )
    op.create_index(
        "ix_tool_execution_receipts_conversation_turn",
        "tool_execution_receipts",
        ["conversation_id", "turn_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_tool_execution_receipts_conversation_turn", table_name="tool_execution_receipts"
    )
    op.drop_index("ix_tool_execution_receipts_owner_status", table_name="tool_execution_receipts")
    op.drop_index(
        "ix_tool_execution_receipts_qualified_tool_id", table_name="tool_execution_receipts"
    )
    op.drop_index("ix_tool_execution_receipts_thread_id", table_name="tool_execution_receipts")
    op.drop_index(
        "ix_tool_execution_receipts_conversation_id", table_name="tool_execution_receipts"
    )
    op.drop_index("ix_tool_execution_receipts_user_id", table_name="tool_execution_receipts")
    op.drop_index("uq_tool_execution_receipts_execution_key", table_name="tool_execution_receipts")
    op.drop_table("tool_execution_receipts")
    ENUM(*_STATUS_VALUES, name="tool_execution_receipt_status").drop(op.get_bind(), checkfirst=True)
