"""Add provider-native reasoning selection to agent model configs.

Revision ID: e8f9a0b1c2d3
Revises: d7e8f9a0b1c2
Create Date: 2026-07-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e8f9a0b1c2d3"
down_revision: str | None = "d7e8f9a0b1c2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_model_configs",
        sa.Column("reasoning_effort", sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_model_configs", "reasoning_effort")
