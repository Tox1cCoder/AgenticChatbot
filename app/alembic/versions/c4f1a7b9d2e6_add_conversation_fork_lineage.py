"""conversations: add fork lineage and idempotency columns

Revision ID: c4f1a7b9d2e6
Revises: 9b132832d676
Create Date: 2026-10-09 00:00:00.000000

Additive only. Existing conversations keep every new column NULL; nothing is
backfilled. The lineage foreign keys are ``ON DELETE SET NULL`` so hard-deleting
a source conversation or message never blocks on, or removes, its forks, and the
checks deliberately do not tie them to ``forked_through_sequence`` for that
reason. The (owner_id, fork_request_id) unique constraint has no
``deleted_at`` predicate: a retry of a soft-deleted fork must still find it.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "c4f1a7b9d2e6"
down_revision: str | Sequence[str] | None = "9b132832d676"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("forked_from_conversation_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "conversations",
        sa.Column("forked_from_message_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "conversations", sa.Column("forked_through_sequence", sa.BigInteger(), nullable=True)
    )
    op.add_column(
        "conversations",
        sa.Column("fork_request_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "conversations",
        sa.Column("fork_request_fingerprint", sa.String(length=64), nullable=True),
    )
    op.create_foreign_key(
        "fk_conversations_forked_from_conversation_id",
        "conversations",
        "conversations",
        ["forked_from_conversation_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_conversations_forked_from_message_id",
        "conversations",
        "messages",
        ["forked_from_message_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        op.f("ix_conversations_forked_from_conversation_id"),
        "conversations",
        ["forked_from_conversation_id"],
    )
    op.create_index(
        op.f("ix_conversations_forked_from_message_id"),
        "conversations",
        ["forked_from_message_id"],
    )
    op.create_unique_constraint(
        "uq_conversations_owner_fork_request", "conversations", ["owner_id", "fork_request_id"]
    )
    op.create_check_constraint(
        "ck_conversations_forked_through_sequence_positive",
        "conversations",
        "forked_through_sequence IS NULL OR forked_through_sequence > 0",
    )
    op.create_check_constraint(
        "ck_conversations_fork_request_fields_together",
        "conversations",
        "(fork_request_id IS NULL) = (fork_request_fingerprint IS NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_conversations_fork_request_fields_together", "conversations", type_="check"
    )
    op.drop_constraint(
        "ck_conversations_forked_through_sequence_positive", "conversations", type_="check"
    )
    op.drop_constraint("uq_conversations_owner_fork_request", "conversations", type_="unique")
    op.drop_index(op.f("ix_conversations_forked_from_message_id"), table_name="conversations")
    op.drop_index(op.f("ix_conversations_forked_from_conversation_id"), table_name="conversations")
    op.drop_constraint(
        "fk_conversations_forked_from_message_id", "conversations", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_conversations_forked_from_conversation_id", "conversations", type_="foreignkey"
    )
    op.drop_column("conversations", "fork_request_fingerprint")
    op.drop_column("conversations", "fork_request_id")
    op.drop_column("conversations", "forked_through_sequence")
    op.drop_column("conversations", "forked_from_message_id")
    op.drop_column("conversations", "forked_from_conversation_id")
