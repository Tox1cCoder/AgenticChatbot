"""add_allow_custom_model_flag

Revision ID: 0f1e2d3c4b5a
Revises: f6a7b8c9d0e1
Create Date: 2026-03-20 16:35:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0f1e2d3c4b5a"
down_revision = "f6a7b8c9d0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_model_configs",
        sa.Column(
            "allow_custom_model",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.alter_column("agent_model_configs", "allow_custom_model", server_default=None)


def downgrade() -> None:
    op.drop_column("agent_model_configs", "allow_custom_model")
