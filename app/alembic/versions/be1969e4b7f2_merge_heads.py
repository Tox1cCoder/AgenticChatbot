"""merge_heads

Revision ID: be1969e4b7f2
Revises: c2d3e4f5g6h7, f6a7b8c9d0e1
Create Date: 2026-01-20 14:53:10.551629

"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "be1969e4b7f2"
down_revision: str | None = ("c2d3e4f5g6h7", "f6a7b8c9d0e1")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
