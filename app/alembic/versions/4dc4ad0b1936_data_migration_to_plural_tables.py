"""data_migration_to_plural_tables

Revision ID: 4dc4ad0b1936
Revises: ca57b3ea95db
Create Date: 2025-09-15 16:41:05.651513

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "4dc4ad0b1936"
down_revision: str | None = "ca57b3ea95db"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Create new plural tables with correct structure
    op.create_table(
        "users",
        sa.Column("username", sa.String(length=50), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("avatar_url", sa.String(length=2048), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_index(op.f("ix_users_id"), "users", ["id"], unique=False)
    op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)

    op.create_table(
        "conversations",
        sa.Column("owner_id", sa.UUID(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_conversations_id"), "conversations", ["id"], unique=False)
    op.create_index(op.f("ix_conversations_owner_id"), "conversations", ["owner_id"], unique=False)

    op.create_table(
        "messages",
        sa.Column("conversation_id", sa.UUID(), nullable=False),
        sa.Column("sender", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_message_conversation_created",
        "messages",
        ["conversation_id", "created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_messages_conversation_id"),
        "messages",
        ["conversation_id"],
        unique=False,
    )
    op.create_index(op.f("ix_messages_id"), "messages", ["id"], unique=False)

    # Data migration from old tables to new tables
    op.execute(
        """
        INSERT INTO users (id, username, email, password_hash, avatar_url, created_at, updated_at, deleted_at)
        SELECT id, username, email, password_hash, avatar_url, created_at, updated_at, NULL as deleted_at
        FROM "user"
    """
    )

    op.execute(
        """
        INSERT INTO conversations (id, owner_id, title, created_at, updated_at, deleted_at)
        SELECT id, user_id as owner_id, title, created_at, updated_at, NULL as deleted_at
        FROM conversation
    """
    )

    # Convert enum sender to integer (assuming 'user'=1, 'assistant'=2)
    op.execute(
        """
        INSERT INTO messages (id, conversation_id, sender, content, created_at, updated_at, deleted_at)
        SELECT id, conversation_id,
               CASE
                   WHEN sender = 'user' THEN 1
                   WHEN sender = 'assistant' THEN 2
                   ELSE 1
               END as sender,
               content, created_at, updated_at, NULL as deleted_at
        FROM message
    """
    )

    # Update feedback table to add deleted_at
    op.add_column("feedback", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))

    # Drop existing foreign key constraints from feedback table before dropping referenced tables
    op.drop_constraint("feedback_message_id_fkey", "feedback", type_="foreignkey")
    op.drop_constraint("feedback_user_id_fkey", "feedback", type_="foreignkey")

    # Drop old constraint and create indexes
    op.drop_constraint("feedback_message_id_key", "feedback", type_="unique")
    op.create_index(
        "idx_feedbacks_message_user",
        "feedback",
        ["message_id", "user_id"],
        unique=False,
    )
    op.create_index(op.f("ix_feedback_id"), "feedback", ["id"], unique=False)
    op.create_index(op.f("ix_feedback_message_id"), "feedback", ["message_id"], unique=True)
    op.create_index(op.f("ix_feedback_user_id"), "feedback", ["user_id"], unique=False)

    # Now safe to drop old tables (data preserved in new tables, constraints removed)
    op.drop_table("message")
    op.drop_table("conversation")
    op.drop_table("user")

    # Create foreign keys on feedback table to point to new plural tables
    op.create_foreign_key("feedback_user_id_fkey", "feedback", "users", ["user_id"], ["id"])
    op.create_foreign_key(
        "feedback_message_id_fkey", "feedback", "messages", ["message_id"], ["id"]
    )


def downgrade() -> None:
    # Reverse migration - recreate old tables and move data back
    op.drop_constraint("feedback_message_id_fkey", "feedback", type_="foreignkey")
    op.drop_constraint("feedback_user_id_fkey", "feedback", type_="foreignkey")

    # Recreate old tables
    op.create_table(
        "user",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("username", sa.String(length=50), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("avatar_url", sa.String(length=2048), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "conversation",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "message",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("conversation_id", sa.UUID(), nullable=False),
        sa.Column(
            "sender",
            postgresql.ENUM("user", "assistant", name="message_role"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversation.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Move data back
    op.execute(
        'INSERT INTO "user" SELECT id, username, email, password_hash, avatar_url, created_at, updated_at FROM users'
    )
    op.execute(
        "INSERT INTO conversation SELECT id, owner_id as user_id, title, created_at, updated_at FROM conversations"
    )
    op.execute(
        """
        INSERT INTO message (id, conversation_id, sender, content, created_at, updated_at)
        SELECT id, conversation_id,
               CASE
                   WHEN sender = 1 THEN 'user'::message_role
                   WHEN sender = 2 THEN 'assistant'::message_role
                   ELSE 'user'::message_role
               END as sender,
               content, created_at, updated_at
        FROM messages
    """
    )

    # Drop new tables and restore feedback
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("users")

    op.drop_column("feedback", "deleted_at")
    op.drop_index("idx_feedbacks_message_user", table_name="feedback")
    op.drop_index(op.f("ix_feedback_id"), table_name="feedback")
    op.drop_index(op.f("ix_feedback_message_id"), table_name="feedback")
    op.drop_index(op.f("ix_feedback_user_id"), table_name="feedback")
    op.create_unique_constraint("feedback_message_id_key", "feedback", ["message_id"])
