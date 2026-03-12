"""add_tool_approvals_table

Revision ID: a3b4c5d6e7f8
Revises: a2b3c4d5e6f7
Create Date: 2025-11-26 10:00:00.000000

"""

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "a3b4c5d6e7f8"
down_revision: str | None = "a2b3c4d5e6f7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """
    Create tool_approvals table for HITL audit trail.

    This table tracks all human approval decisions for tool calls in HITL workflows,
    providing a complete audit trail for compliance, debugging, and analytics.
    """
    # Create decision_type enum if it doesn't exist
    op.execute(
        """
        DO $$ BEGIN
            CREATE TYPE decision_type AS ENUM ('accept', 'edit', 'reject');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """
    )

    # Create tool_approvals table
    op.create_table(
        "tool_approvals",
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
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("conversation_id", UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("interrupt_id", sa.String(length=255), nullable=False),
        sa.Column("tool_name", sa.String(length=255), nullable=False),
        sa.Column("tool_call_id", sa.String(length=255), nullable=False),
        sa.Column("original_args", JSONB, nullable=False),
        sa.Column("modified_args", JSONB, nullable=True),
        sa.Column(
            "decision",
            sa.Enum("accept", "edit", "reject", name="decision_type", create_type=False),
            nullable=False,
        ),
        sa.Column(
            "decided_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_tool_approvals_conversation_id",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_tool_approvals_user_id"),
    )

    # Create indexes for efficient querying
    op.create_index("ix_tool_approvals_id", "tool_approvals", ["id"])
    op.create_index("ix_tool_approvals_conversation_id", "tool_approvals", ["conversation_id"])
    op.create_index("ix_tool_approvals_user_id", "tool_approvals", ["user_id"])
    op.create_index("ix_tool_approvals_interrupt_id", "tool_approvals", ["interrupt_id"])
    op.create_index("ix_tool_approvals_decided_at", "tool_approvals", ["decided_at"])


def downgrade() -> None:
    """Drop tool_approvals table and enum."""
    # Drop indexes
    op.drop_index("ix_tool_approvals_decided_at", table_name="tool_approvals")
    op.drop_index("ix_tool_approvals_interrupt_id", table_name="tool_approvals")
    op.drop_index("ix_tool_approvals_user_id", table_name="tool_approvals")
    op.drop_index("ix_tool_approvals_conversation_id", table_name="tool_approvals")
    op.drop_index("ix_tool_approvals_id", table_name="tool_approvals")

    # Drop table
    op.drop_table("tool_approvals")

    # Drop enum type
    op.execute("DROP TYPE decision_type")
