"""add_tool_result_blobs

Revision ID: q0r1s2t3u4v5
Revises: p9q0r1s2t3u4
Create Date: 2026-05-06 00:00:00.000000

Adds the durable storage table for offloaded tool result payloads.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = "q0r1s2t3u4v5"
down_revision: str | Sequence[str] | None = "p9q0r1s2t3u4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create tool_result_blobs table."""

    op.create_table(
        "tool_result_blobs",
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
        sa.Column("tool_call_id", sa.String(length=255), nullable=True),
        sa.Column("tool_name", sa.String(length=255), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column(
            "content_type",
            sa.String(length=128),
            nullable=False,
            server_default="text/plain",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_index(
        "ix_tool_result_blobs_conversation_id",
        "tool_result_blobs",
        ["conversation_id"],
    )
    op.create_index(
        "ix_tool_result_blobs_user_id",
        "tool_result_blobs",
        ["user_id"],
    )
    op.create_index(
        "ix_tool_result_blobs_tool_call_id",
        "tool_result_blobs",
        ["tool_call_id"],
    )
    op.create_index(
        "ix_tool_result_blobs_tool_name",
        "tool_result_blobs",
        ["tool_name"],
    )


def downgrade() -> None:
    """Drop tool_result_blobs table."""

    op.drop_index("ix_tool_result_blobs_tool_name", table_name="tool_result_blobs")
    op.drop_index("ix_tool_result_blobs_tool_call_id", table_name="tool_result_blobs")
    op.drop_index("ix_tool_result_blobs_user_id", table_name="tool_result_blobs")
    op.drop_index(
        "ix_tool_result_blobs_conversation_id", table_name="tool_result_blobs"
    )
    op.drop_table("tool_result_blobs")
