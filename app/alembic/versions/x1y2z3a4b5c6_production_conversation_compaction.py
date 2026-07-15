"""production conversation compaction

Revision ID: x1y2z3a4b5c6
Revises: w7x8y9z0a1b2
Create Date: 2026-07-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "x1y2z3a4b5c6"
down_revision: str | Sequence[str] | None = "w7x8y9z0a1b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column(
            "next_message_sequence",
            sa.BigInteger(),
            nullable=True,
            server_default=sa.text("1"),
        ),
    )
    op.add_column(
        "messages",
        sa.Column("sequence", sa.BigInteger(), nullable=True),
    )
    op.execute(
        """
        WITH ranked AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY conversation_id
                    ORDER BY created_at, id
                ) AS sequence
            FROM messages
        )
        UPDATE messages AS message
        SET sequence = ranked.sequence
        FROM ranked
        WHERE message.id = ranked.id
        """
    )
    op.execute(
        """
        UPDATE conversations AS conversation
        SET next_message_sequence = COALESCE(
            (
                SELECT MAX(sequence) + 1
                FROM messages
                WHERE messages.conversation_id = conversation.id
            ),
            1
        )
        """
    )
    op.alter_column(
        "conversations",
        "next_message_sequence",
        existing_type=sa.BigInteger(),
        nullable=False,
        server_default=sa.text("1"),
    )
    op.alter_column(
        "messages",
        "sequence",
        existing_type=sa.BigInteger(),
        nullable=False,
    )
    op.create_check_constraint(
        "ck_conversations_next_message_sequence_positive",
        "conversations",
        "next_message_sequence > 0",
    )
    op.create_check_constraint(
        "ck_messages_sequence_positive",
        "messages",
        "sequence > 0",
    )
    op.create_unique_constraint(
        "uq_messages_conversation_sequence",
        "messages",
        ["conversation_id", "sequence"],
    )
    op.create_index(
        "ix_messages_prompt_history",
        "messages",
        ["conversation_id", "sequence"],
        unique=False,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    op.rename_table(
        "conversation_memory_summaries",
        "conversation_memory_summaries_legacy",
    )
    op.drop_index(
        "ux_conversation_memory_summaries_conversation_id",
        table_name="conversation_memory_summaries_legacy",
    )
    op.drop_index(
        "ix_conversation_memory_summaries_user_id",
        table_name="conversation_memory_summaries_legacy",
    )
    op.drop_index(
        "ix_conversation_memory_summaries_last_message",
        table_name="conversation_memory_summaries_legacy",
    )

    op.create_table(
        "conversation_memory_summaries",
        sa.Column("conversation_id", UUID(as_uuid=True), nullable=False),
        sa.Column(
            "summary_payload",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "summary_schema_version",
            sa.SmallInteger(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        sa.Column("last_summarized_sequence", sa.BigInteger(), nullable=True),
        sa.Column(
            "summary_version",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        sa.Column(
            "source_message_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "source_token_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "summary_token_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("provider", sa.String(length=64), nullable=False, server_default=sa.text("''")),
        sa.Column("model", sa.String(length=255), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "tokenizer",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "prompt_version",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "is_valid",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint(
            "conversation_id",
            name="pk_conversation_memory_summaries",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_memory_summary_conversation",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id", "last_summarized_sequence"],
            ["messages.conversation_id", "messages.sequence"],
            name="fk_memory_summary_conversation_sequence",
        ),
        sa.CheckConstraint(
            "summary_schema_version > 0",
            name="ck_memory_summary_schema_version_positive",
        ),
        sa.CheckConstraint(
            "summary_version > 0",
            name="ck_memory_summary_version_positive",
        ),
        sa.CheckConstraint(
            "source_message_count >= 0",
            name="ck_memory_summary_source_message_count_nonnegative",
        ),
        sa.CheckConstraint(
            "source_token_count >= 0",
            name="ck_memory_summary_source_token_count_nonnegative",
        ),
        sa.CheckConstraint(
            "summary_token_count >= 0",
            name="ck_memory_summary_token_count_nonnegative",
        ),
    )
    op.execute(
        """
        INSERT INTO conversation_memory_summaries (
            conversation_id,
            summary_payload,
            summary_schema_version,
            last_summarized_sequence,
            summary_version,
            source_message_count,
            source_token_count,
            summary_token_count,
            provider,
            model,
            tokenizer,
            prompt_version,
            is_valid,
            created_at,
            updated_at
        )
        SELECT
            legacy.conversation_id,
            '{}'::jsonb,
            1,
            NULL,
            GREATEST(COALESCE(legacy.summary_version, 0) + 1, 1),
            0,
            0,
            0,
            'legacy',
            '',
            'legacy-invalidated',
            'structured-memory-v1',
            false,
            legacy.created_at,
            now()
        FROM conversation_memory_summaries_legacy AS legacy
        JOIN conversations ON conversations.id = legacy.conversation_id
        ON CONFLICT (conversation_id) DO NOTHING
        """
    )
    op.drop_table("conversation_memory_summaries_legacy")

    op.create_table(
        "conversation_summary_jobs",
        sa.Column("conversation_id", UUID(as_uuid=True), nullable=False),
        sa.Column("requested_through_sequence", sa.BigInteger(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("lease_token", UUID(as_uuid=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error_code", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint(
            "conversation_id",
            name="pk_conversation_summary_jobs",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_summary_job_conversation",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id", "requested_through_sequence"],
            ["messages.conversation_id", "messages.sequence"],
            name="fk_summary_job_conversation_sequence",
        ),
        sa.CheckConstraint(
            "status IN ('idle', 'pending', 'processing', 'retry', 'dead')",
            name="ck_summary_jobs_status",
        ),
        sa.CheckConstraint(
            "attempt_count >= 0",
            name="ck_summary_jobs_attempt_count_nonnegative",
        ),
    )
    op.create_index(
        "ix_summary_jobs_due",
        "conversation_summary_jobs",
        ["status", "available_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_summary_jobs_due", table_name="conversation_summary_jobs")
    op.drop_table("conversation_summary_jobs")

    op.create_table(
        "conversation_memory_summaries_legacy",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("conversation_id", UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("last_summarized_message_id", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "source_message_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "estimated_tokens",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "summary_version",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="conversation_memory_summaries_pkey"),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="conversation_memory_summaries_conversation_id_fkey",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="conversation_memory_summaries_user_id_fkey",
        ),
        sa.ForeignKeyConstraint(
            ["last_summarized_message_id"],
            ["messages.id"],
            name="conversation_memory_summaries_last_summarized_message_id_fkey",
        ),
    )
    op.execute(
        """
        INSERT INTO conversation_memory_summaries_legacy (
            id,
            conversation_id,
            user_id,
            summary_text,
            last_summarized_message_id,
            source_message_count,
            estimated_tokens,
            summary_version,
            created_at,
            updated_at
        )
        SELECT
            gen_random_uuid(),
            summary.conversation_id,
            conversation.owner_id,
            summary.summary_payload::text,
            cursor_message.id,
            summary.source_message_count,
            summary.summary_token_count,
            LEAST(summary.summary_version, 2147483647)::integer,
            summary.created_at,
            summary.updated_at
        FROM conversation_memory_summaries AS summary
        JOIN conversations AS conversation ON conversation.id = summary.conversation_id
        LEFT JOIN messages AS cursor_message
          ON cursor_message.conversation_id = summary.conversation_id
         AND cursor_message.sequence = summary.last_summarized_sequence
        """
    )
    op.drop_table("conversation_memory_summaries")
    op.rename_table(
        "conversation_memory_summaries_legacy",
        "conversation_memory_summaries",
    )
    op.create_index(
        "ux_conversation_memory_summaries_conversation_id",
        "conversation_memory_summaries",
        ["conversation_id"],
        unique=True,
    )
    op.create_index(
        "ix_conversation_memory_summaries_user_id",
        "conversation_memory_summaries",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_conversation_memory_summaries_last_message",
        "conversation_memory_summaries",
        ["last_summarized_message_id"],
        unique=False,
    )

    op.drop_index("ix_messages_prompt_history", table_name="messages")
    op.drop_constraint(
        "uq_messages_conversation_sequence",
        "messages",
        type_="unique",
    )
    op.drop_constraint(
        "ck_messages_sequence_positive",
        "messages",
        type_="check",
    )
    op.drop_constraint(
        "ck_conversations_next_message_sequence_positive",
        "conversations",
        type_="check",
    )
    op.drop_column("messages", "sequence")
    op.drop_column("conversations", "next_message_sequence")
