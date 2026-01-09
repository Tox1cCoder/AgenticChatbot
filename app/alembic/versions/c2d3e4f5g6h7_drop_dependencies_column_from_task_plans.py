"""drop_dependencies_column_from_task_plans

Revision ID: c2d3e4f5g6h7
Revises: b1c2d3e4f5g6
Create Date: 2026-01-09 13:42:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "c2d3e4f5g6h7"
down_revision: Union[str, Sequence[str], None] = ("b1c2d3e4f5g6", "d4e29f078096")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Drop the dependencies column from task_plans table.

    The dependencies feature has been removed from the planning agent.
    Tasks now execute in order without dependency tracking.
    """
    op.drop_column("task_plans", "dependencies")


def downgrade() -> None:
    """Re-add the dependencies column to task_plans table."""
    op.add_column(
        "task_plans",
        sa.Column(
            "dependencies",
            JSONB,
            nullable=True,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
