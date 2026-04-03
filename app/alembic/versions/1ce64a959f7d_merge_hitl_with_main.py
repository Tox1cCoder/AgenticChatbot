"""merge_hitl_with_main

Revision ID: 1ce64a959f7d
Revises: be1969e4b7f2, h1i2j3k4l5m6
Create Date: 2026-03-09 14:45:41.251799

"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "1ce64a959f7d"
down_revision: str | None = ("be1969e4b7f2", "h1i2j3k4l5m6")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
