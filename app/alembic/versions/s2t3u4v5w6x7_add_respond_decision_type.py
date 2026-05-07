"""add_respond_decision_type

Revision ID: s2t3u4v5w6x7
Revises: r1s2t3u4v5w6
Create Date: 2026-05-06 00:02:00.000000

Extends the ``decision_type`` PostgreSQL enum with the ``respond`` value
used by HITL respond decisions.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "s2t3u4v5w6x7"
down_revision: str | Sequence[str] | None = "r1s2t3u4v5w6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add ``respond`` to the decision_type enum."""

    op.execute("ALTER TYPE decision_type ADD VALUE IF NOT EXISTS 'respond'")


def downgrade() -> None:
    """No-op: PostgreSQL does not support removing enum values cleanly."""
