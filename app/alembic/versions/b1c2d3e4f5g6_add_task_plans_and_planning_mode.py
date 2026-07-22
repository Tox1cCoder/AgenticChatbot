"""add_task_plans_and_planning_mode

Revision ID: b1c2d3e4f5g6
Revises: a3b4c5d6e7f8, a1b2c3d4e5f6
Create Date: 2025-12-02 10:00:00.000000

"""

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ENUM, JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "b1c2d3e4f5g6"
down_revision: str | Sequence[str] | None = ("a3b4c5d6e7f8", "a1b2c3d4e5f6")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """
    Create task_plans table and add planning_mode_enabled to conversations.

    This migration adds support for task planning functionality:
    - task_plans table for storing individual tasks with dependencies and metadata
    - planning_mode_enabled column on conversations to enable/disable planning mode
    """
    # Create task_status enum if it doesn't exist
    op.execute(
        """
        DO $$ BEGIN
            CREATE TYPE task_status AS ENUM ('pending', 'in_progress', 'completed', 'skipped');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """
    )

    # Create task_plans table
    op.create_table(
        "task_plans",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            default=uuid.uuid4,
            nullable=False,
        ),
        sa.Column("conversation_id", UUID(as_uuid=True), nullable=False),
        sa.Column("task_order", sa.Integer(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "status",
            ENUM(
                "pending",
                "in_progress",
                "completed",
                "skipped",
                name="task_status",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("dependencies", JSONB, nullable=True, server_default=sa.text("'[]'::jsonb")),
        sa.Column("task_metadata", JSONB, nullable=True, server_default=sa.text("'{}'::jsonb")),
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
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_task_plans_conversation_id",
        ),
    )

    # Create indexes for efficient querying
    op.create_index("ix_task_plans_id", "task_plans", ["id"])
    op.create_index("ix_task_plans_conversation_id", "task_plans", ["conversation_id"])
    op.create_index("ix_task_plans_status", "task_plans", ["status"])
    op.create_index(
        "ix_task_plans_conversation_order",
        "task_plans",
        ["conversation_id", "task_order"],
    )

    # Add planning_mode_enabled column to conversations table
    op.add_column(
        "conversations",
        sa.Column(
            "planning_mode_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    """Drop task_plans table, planning_mode_enabled column, and enum."""
    # Drop planning_mode_enabled column from conversations
    op.drop_column("conversations", "planning_mode_enabled")

    # Drop indexes
    op.drop_index("ix_task_plans_conversation_order", table_name="task_plans")
    op.drop_index("ix_task_plans_status", table_name="task_plans")
    op.drop_index("ix_task_plans_conversation_id", table_name="task_plans")
    op.drop_index("ix_task_plans_id", table_name="task_plans")

    # Drop table
    op.drop_table("task_plans")

    # Drop enum type
    op.execute("DROP TYPE task_status")
