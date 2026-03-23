"""merge allow custom model branch with device metadata branch

Revision ID: m4n5o6p7q8r9
Revises: 0f1e2d3c4b5a, l3m4n5o6p7q8
Create Date: 2026-03-23 11:45:00.000000

"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "m4n5o6p7q8r9"
down_revision: str | Sequence[str] | None = ("0f1e2d3c4b5a", "l3m4n5o6p7q8")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
