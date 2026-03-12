"""add_task_execution_metrics

Revision ID: d4e29f078096
Revises: b1c2d3e4f5g6
Create Date: 2025-12-10 13:57:23.771003

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e29f078096"
down_revision: str | None = "b1c2d3e4f5g6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add task execution metrics columns to task_plans table
    op.add_column("task_plans", sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "task_plans",
        sa.Column("estimated_duration_minutes", sa.Integer(), nullable=True),
    )
    op.add_column("task_plans", sa.Column("actual_duration_minutes", sa.Integer(), nullable=True))
    op.add_column(
        "task_plans",
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("task_plans", sa.Column("completion_confidence", sa.Float(), nullable=True))


def downgrade() -> None:
    # Remove task execution metrics columns from task_plans table
    op.drop_column("task_plans", "completion_confidence")
    op.drop_column("task_plans", "retry_count")
    op.drop_column("task_plans", "actual_duration_minutes")
    op.drop_column("task_plans", "estimated_duration_minutes")
    op.drop_column("task_plans", "started_at")
