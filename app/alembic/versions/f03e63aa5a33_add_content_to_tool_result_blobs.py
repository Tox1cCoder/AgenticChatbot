"""add content to tool_result_blobs

Revision ID: f03e63aa5a33
Revises: u7v8w9x0y1z2
Create Date: 2026-06-12 14:19:19.785518

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f03e63aa5a33'
down_revision: str | None = 'u7v8w9x0y1z2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("tool_result_blobs", sa.Column("content", sa.Text(), nullable=True))
    op.alter_column(
        "tool_result_blobs",
        "storage_path",
        existing_type=sa.String(length=1024),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "tool_result_blobs",
        "storage_path",
        existing_type=sa.String(length=1024),
        nullable=False,
    )
    op.drop_column("tool_result_blobs", "content")
