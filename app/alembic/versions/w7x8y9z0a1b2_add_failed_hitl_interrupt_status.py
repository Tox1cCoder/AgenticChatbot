"""add failed HITL interrupt status

Revision ID: w7x8y9z0a1b2
Revises: v1w2x3y4z5a6
Create Date: 2026-07-14
"""

from collections.abc import Sequence

from alembic import op

revision: str = "w7x8y9z0a1b2"
down_revision: str | None = "v1w2x3y4z5a6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TYPE hitl_interrupt_status ADD VALUE IF NOT EXISTS 'failed'")


def downgrade() -> None:
    # PostgreSQL does not support removing an enum label safely.
    pass
