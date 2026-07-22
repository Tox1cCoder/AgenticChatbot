"""Add chat_images table for externalized chat image bytes.

Revision ID: d7e8f9a0b1c2
Revises: c6d7e8f9a0b1
Create Date: 2026-07-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "d7e8f9a0b1c2"
down_revision: str | None = "c6d7e8f9a0b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chat_images",
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
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column(
            "content_type",
            sa.String(length=128),
            nullable=False,
            server_default="image/png",
        ),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_chat_images_conversation_id", "chat_images", ["conversation_id"])
    op.create_index("ix_chat_images_user_id", "chat_images", ["user_id"])
    op.create_index("ix_chat_images_sha256", "chat_images", ["sha256"])


def downgrade() -> None:
    op.drop_index("ix_chat_images_sha256", table_name="chat_images")
    op.drop_index("ix_chat_images_user_id", table_name="chat_images")
    op.drop_index("ix_chat_images_conversation_id", table_name="chat_images")
    op.drop_table("chat_images")
