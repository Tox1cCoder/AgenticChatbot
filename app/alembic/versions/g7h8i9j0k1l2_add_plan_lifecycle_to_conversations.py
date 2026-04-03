"""add_plan_lifecycle_to_conversations

Revision ID: g7h8i9j0k1l2
Revises: 6c6598a9eb26
Create Date: 2026-03-12 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "g7h8i9j0k1l2"
down_revision: str | Sequence[str] | None = "6c6598a9eb26"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add plan_lifecycle column (nullable VARCHAR) to conversations.

    Using a plain VARCHAR instead of a PG enum avoids requiring a new type
    in the database and simplifies future extensions to the lifecycle values.
    """
    op.add_column(
        "conversations",
        sa.Column("plan_lifecycle", sa.String(20), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("conversations", "plan_lifecycle")
