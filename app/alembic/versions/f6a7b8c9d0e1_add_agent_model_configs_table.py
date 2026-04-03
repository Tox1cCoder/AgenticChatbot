"""add_agent_model_configs_table

Revision ID: f6a7b8c9d0e1
Revises: e5f6g7h8i9j0
Create Date: 2026-01-20 13:50:00.000000

"""

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision = "f6a7b8c9d0e1"
down_revision = "e5f6g7h8i9j0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """
    Create agent_model_configs table for persistent per-agent model selection.

    Stores provider/model/temperature per (user_id, agent_key) with a unique constraint.
    """
    op.create_table(
        "agent_model_configs",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            default=uuid.uuid4,
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("agent_key", sa.Text(), nullable=False),
        sa.Column("provider_type", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_agent_model_configs_user_id",
        ),
    )

    op.create_index("idx_agent_model_configs_user_id", "agent_model_configs", ["user_id"])
    op.create_index(
        "idx_agent_model_configs_user_agent",
        "agent_model_configs",
        ["user_id", "agent_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("idx_agent_model_configs_user_agent", table_name="agent_model_configs")
    op.drop_index("idx_agent_model_configs_user_id", table_name="agent_model_configs")
    op.drop_table("agent_model_configs")
