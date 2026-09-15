"""Add pending, selected, and released web-image lifecycle state.

Revision ID: a3b4c5d6e7f9
Revises: f2a3b4c5d6e7
Create Date: 2026-09-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a3b4c5d6e7f9"
down_revision: str | None = "f2a3b4c5d6e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "web_image_references",
        sa.Column(
            "lifecycle_state",
            sa.String(length=16),
            server_default="pending",
            nullable=False,
        ),
    )
    op.add_column(
        "web_image_references",
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_web_image_references_lifecycle_state",
        "web_image_references",
        ["lifecycle_state"],
    )
    op.create_index(
        "ix_web_image_references_expires_at",
        "web_image_references",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_web_image_references_expires_at", table_name="web_image_references")
    op.drop_index(
        "ix_web_image_references_lifecycle_state", table_name="web_image_references"
    )
    op.drop_column("web_image_references", "expires_at")
    op.drop_column("web_image_references", "lifecycle_state")
