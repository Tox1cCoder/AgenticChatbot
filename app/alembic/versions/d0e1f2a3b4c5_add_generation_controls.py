"""add the generation lifecycle and its command ledger

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
Create Date: 2026-09-07 00:00:00.000000

A turn used to be identified by the message it produced, so there was nothing
to Stop during the part of the turn that takes longest. ``generations`` owns
identity instead: it exists before the first model call and outlives every
worker that touches it.

Two indexes are the mechanism, not a safety net:

* ``uq_generations_active_per_conversation`` is partial over the in-flight
  statuses, so one conversation cannot have two turns running. ``continuable``
  is deliberately outside the predicate — a paused turn holds no worker and
  must not block the next question.
* ``uq_generation_commands_key`` makes a command claim a race that the database
  settles. The loser reads the winner's recorded result, which is what makes a
  retried Stop return "stopped" instead of stopping something else.

``generation_commands`` records the ``fence`` — the lifecycle version the
client issued against — because a single "last command" column cannot fence a
delayed replay: Stop completes against epoch 0, Continue overwrites the
column, and the late retry of the Stop cancels epoch 1.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ENUM, JSONB, UUID

revision: str = "d0e1f2a3b4c5"
down_revision: str | None = "c9d0e1f2a3b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_STATUS_VALUES = (
    "starting",
    "running",
    "finalizing_after_limit",
    "continuable",
    "continuing",
    "stop_requested",
    "stopped",
    "completed",
    "completed_partial",
    "failed",
)
_ACTION_VALUES = ("stop", "continue")

# The statuses that mean a worker is doing work, or is being asked to stop.
# Kept in sync with ``app.models.generation.ACTIVE_STATUSES`` by
# ``test_generation_migration_matches_the_model``.
_ACTIVE_STATUSES = (
    "continuing",
    "finalizing_after_limit",
    "running",
    "starting",
    "stop_requested",
)
_ACTIVE_PREDICATE = "status IN ({})".format(", ".join(f"'{value}'" for value in _ACTIVE_STATUSES))

# ``create_type=False`` is load-bearing. ``op.create_table`` auto-creates any
# enum type a column references, with no ``checkfirst``, so leaving it True
# makes the CREATE TYPE run twice and the migration abort on DuplicateObject.
# Both types are created once, explicitly, below.
_STATUS_ENUM = ENUM(*_STATUS_VALUES, name="generation_status", create_type=False)
_ACTION_ENUM = ENUM(*_ACTION_VALUES, name="generation_command_action", create_type=False)


def upgrade() -> None:
    ENUM(*_STATUS_VALUES, name="generation_status").create(op.get_bind(), checkfirst=True)
    ENUM(*_ACTION_VALUES, name="generation_command_action").create(
        op.get_bind(), checkfirst=True
    )

    op.create_table(
        "generations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("terminal_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("logical_turn_id", sa.String(length=160), nullable=False),
        sa.Column("checkpoint_thread_id", sa.String(length=320), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", UUID(as_uuid=True), nullable=False),
        sa.Column("status", _STATUS_ENUM, nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("execution_epoch", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("active_agent_id", sa.String(length=160), nullable=True),
        sa.Column("terminal_reason", sa.String(length=128), nullable=True),
        sa.Column("execution_budget", JSONB, nullable=True),
        sa.Column("research_accounting", JSONB, nullable=True),
        sa.Column("assistant_message_id", UUID(as_uuid=True), nullable=True),
        sa.Column("continuation_id", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "continuation_available", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("continuation_block_reason", sa.String(length=128), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.ForeignKeyConstraint(["assistant_message_id"], ["messages.id"]),
    )

    op.create_index(
        "uq_generations_logical_turn_id", "generations", ["logical_turn_id"], unique=True
    )
    op.create_index(
        "uq_generations_active_per_conversation",
        "generations",
        ["conversation_id"],
        unique=True,
        postgresql_where=sa.text(_ACTIVE_PREDICATE),
    )
    op.create_index("ix_generations_user_id", "generations", ["user_id"])
    op.create_index("ix_generations_conversation_id", "generations", ["conversation_id"])
    op.create_index("ix_generations_owner_status", "generations", ["user_id", "status"])
    op.create_index(
        "ix_generations_checkpoint_thread", "generations", ["checkpoint_thread_id"]
    )
    op.create_index("ix_generations_continuation_id", "generations", ["continuation_id"])

    op.create_table(
        "generation_commands",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("generation_id", UUID(as_uuid=True), nullable=False),
        sa.Column("idempotency_key", sa.String(length=160), nullable=False),
        sa.Column("action", _ACTION_ENUM, nullable=False),
        sa.Column("fence", sa.Integer(), nullable=False),
        sa.Column("result", JSONB, nullable=True),
        sa.ForeignKeyConstraint(["generation_id"], ["generations.id"], ondelete="CASCADE"),
    )

    op.create_index(
        "uq_generation_commands_key",
        "generation_commands",
        ["generation_id", "idempotency_key"],
        unique=True,
    )
    op.create_index(
        "ix_generation_commands_generation_id", "generation_commands", ["generation_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_generation_commands_generation_id", table_name="generation_commands")
    op.drop_index("uq_generation_commands_key", table_name="generation_commands")
    op.drop_table("generation_commands")

    op.drop_index("ix_generations_continuation_id", table_name="generations")
    op.drop_index("ix_generations_checkpoint_thread", table_name="generations")
    op.drop_index("ix_generations_owner_status", table_name="generations")
    op.drop_index("ix_generations_conversation_id", table_name="generations")
    op.drop_index("ix_generations_user_id", table_name="generations")
    op.drop_index("uq_generations_active_per_conversation", table_name="generations")
    op.drop_index("uq_generations_logical_turn_id", table_name="generations")
    op.drop_table("generations")

    ENUM(*_ACTION_VALUES, name="generation_command_action").drop(
        op.get_bind(), checkfirst=True
    )
    ENUM(*_STATUS_VALUES, name="generation_status").drop(op.get_bind(), checkfirst=True)
