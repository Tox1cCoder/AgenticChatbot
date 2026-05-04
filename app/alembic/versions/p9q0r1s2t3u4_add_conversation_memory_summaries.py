"""add_conversation_memory_summaries

Revision ID: p9q0r1s2t3u4
Revises: o6p7q8r9s0t1
Create Date: 2026-04-29 00:00:00.000000

Adds the durable conversation-memory summary table introduced by the
memory refactor. Each conversation gets at most one row holding the rolling
summary plus a ``messages.id`` cursor pointing at the newest message folded
into that summary.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = "p9q0r1s2t3u4"
down_revision: str | Sequence[str] | None = "o6p7q8r9s0t1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create conversation_memory_summaries table."""

    op.create_table(
        "conversation_memory_summaries",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "conversation_id",
            UUID(as_uuid=True),
            sa.ForeignKey("conversations.id"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column("summary_text", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "last_summarized_message_id",
            UUID(as_uuid=True),
            sa.ForeignKey("messages.id"),
            nullable=True,
        ),
        sa.Column(
            "source_message_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "estimated_tokens",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "summary_version",
            sa.Integer(),
            nullable=False,
            server_default="1",
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
            onupdate=sa.text("now()"),
        ),
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
    )
    op.create_index(
        "ix_conversation_memory_summaries_last_message",
        "conversation_memory_summaries",
        ["last_summarized_message_id"],
    )


def downgrade() -> None:
    """Drop conversation_memory_summaries table."""

    op.drop_index(
        "ix_conversation_memory_summaries_last_message",
        table_name="conversation_memory_summaries",
    )
    op.drop_index(
        "ix_conversation_memory_summaries_user_id",
        table_name="conversation_memory_summaries",
    )
    op.drop_index(
        "ux_conversation_memory_summaries_conversation_id",
        table_name="conversation_memory_summaries",
    )
    op.drop_table("conversation_memory_summaries")
