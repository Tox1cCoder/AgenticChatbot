"""Update database schema to match ERD requirements

Revision ID: update_erd_compliance
Revises: db36b7c1df40
Create Date: 2025-09-11 12:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "update_erd_compliance"
down_revision = "db36b7c1df40"
branch_labels = None
depends_on = None


def upgrade():
    """Update schema to match ERD requirements"""

    # 1. Rename tables to match ERD (remove plural 's')
    op.rename_table("users", "user")
    op.rename_table("conversations", "conversation")
    op.rename_table("messages", "message")
    op.rename_table("feedbacks", "feedback")

    # 2. Update foreign key references after table renames
    op.drop_constraint("messages_conversation_id_fkey", "message", type_="foreignkey")
    op.create_foreign_key(
        "message_conversation_id_fkey",
        "message",
        "conversation",
        ["conversation_id"],
        ["id"],
    )

    op.drop_constraint("messages_parent_message_id_fkey", "message", type_="foreignkey")
    op.drop_column("message", "parent_message_id")  # ERD doesn't show parent_message_id

    op.drop_constraint("feedback_message_id_fkey", "feedback", type_="foreignkey")
    op.create_foreign_key(
        "feedback_message_id_fkey", "feedback", "message", ["message_id"], ["id"]
    )

    op.drop_constraint("feedback_user_id_fkey", "feedback", type_="foreignkey")
    op.create_foreign_key(
        "feedback_user_id_fkey", "feedback", "user", ["user_id"], ["id"]
    )

    op.drop_constraint("conversations_user_id_fkey", "conversation", type_="foreignkey")
    op.create_foreign_key(
        "conversation_user_id_fkey", "conversation", "user", ["user_id"], ["id"]
    )

    # 3. Rename message.role to message.sender as per ERD
    op.alter_column("message", "role", new_column_name="sender")

    # 4. Add unique constraint on feedback.message_id as per ERD
    op.create_unique_constraint("uq_feedback_message_id", "feedback", ["message_id"])

    # 5. Update indexes after table renames
    op.drop_index("idx_messages_conversation_created", table_name="message")
    op.create_index(
        "idx_message_conversation_created", "message", ["conversation_id", "created_at"]
    )

    op.drop_index("idx_feedback_message_user", table_name="feedback")
    op.create_index("idx_feedback_message_user", "feedback", ["message_id", "user_id"])


def downgrade():
    """Revert schema changes"""

    # Reverse all changes
    op.drop_index("idx_message_conversation_created", table_name="message")
    op.create_index(
        "idx_messages_conversation_created",
        "message",
        ["conversation_id", "created_at"],
    )

    op.drop_index("idx_feedbacks_message_user", table_name="feedbacks")
    op.create_index("idx_feedback_message_user", "feedbacks", ["message_id", "user_id"])

    op.drop_constraint("uq_feedbacks_message_id", "feedbacks", type_="unique")

    op.alter_column("message", "sender", new_column_name="role")

    op.add_column(
        "message", sa.Column("parent_message_id", postgresql.UUID(), nullable=True)
    )
    op.create_foreign_key(
        "messages_parent_message_id_fkey",
        "message",
        "message",
        ["parent_message_id"],
        ["id"],
    )

    op.drop_constraint("message_conversation_id_fkey", "message", type_="foreignkey")
    op.create_foreign_key(
        "messages_conversation_id_fkey",
        "message",
        "conversation",
        ["conversation_id"],
        ["id"],
    )

    op.drop_constraint("feedbacks_message_id_fkey", "feedbacks", type_="foreignkey")
    op.create_foreign_key(
        "feedback_message_id_fkey", "feedbacks", "message", ["message_id"], ["id"]
    )

    op.drop_constraint("feedbacks_user_id_fkey", "feedbacks", type_="foreignkey")
    op.create_foreign_key(
        "feedback_user_id_fkey", "feedbacks", "users", ["user_id"], ["id"]
    )

    op.drop_constraint("conversation_user_id_fkey", "conversation", type_="foreignkey")
    op.create_foreign_key(
        "conversations_user_id_fkey", "conversation", "users", ["user_id"], ["id"]
    )

    op.rename_table("conversation", "conversations")
    op.rename_table("message", "messages")
    op.rename_table("feedbacks", "feedback")
