"""Add owned references for selected remote rich images.

Revision ID: f9a0b1c2d3e4
Revises: e8f9a0b1c2d3
Create Date: 2026-07-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "f9a0b1c2d3e4"
down_revision: str | None = "e8f9a0b1c2d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "web_image_references",
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
        sa.Column("upstream_url", sa.String(length=4096), nullable=False),
        sa.Column("expected_mime", sa.String(length=128), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_web_image_references_conversation_id",
        "web_image_references",
        ["conversation_id"],
    )
    op.create_index(
        "ix_web_image_references_user_id",
        "web_image_references",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_web_image_references_user_id", table_name="web_image_references")
    op.drop_index("ix_web_image_references_conversation_id", table_name="web_image_references")
    op.drop_table("web_image_references")
