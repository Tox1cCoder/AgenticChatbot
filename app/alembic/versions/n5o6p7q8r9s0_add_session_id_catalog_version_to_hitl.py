"""Add session_id and catalog_version to hitl_interrupts and tool_approvals.

Revision ID: n5o6p7q8r9s0
Revises: m4n5o6p7q8r9
Create Date: 2026-04-08

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "n5o6p7q8r9s0"
down_revision: str | Sequence[str] | None = "m4n5o6p7q8r9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add session_id and catalog_version to hitl_interrupts
    op.add_column(
        "hitl_interrupts",
        sa.Column("session_id", sa.String(255), nullable=True),
    )
    op.add_column(
        "hitl_interrupts",
        sa.Column("catalog_version", sa.Integer(), nullable=True),
    )
    op.add_column(
        "hitl_interrupts",
        sa.Column("tool_instance_id", sa.String(64), nullable=True),
    )

    # Add session_id, catalog_version, and tool_instance_id to tool_approvals
    op.add_column(
        "tool_approvals",
        sa.Column("session_id", sa.String(255), nullable=True),
    )
    op.add_column(
        "tool_approvals",
        sa.Column("catalog_version", sa.Integer(), nullable=True),
    )
    op.add_column(
        "tool_approvals",
        sa.Column("tool_instance_id", sa.String(64), nullable=True),
    )

    # Indexes for resume validation queries
    op.create_index(
        "ix_hitl_interrupts_session_id",
        "hitl_interrupts",
        ["session_id"],
    )
    op.create_index(
        "ix_tool_approvals_session_id",
        "tool_approvals",
        ["session_id"],
    )
    op.create_index(
        "ix_tool_approvals_tool_instance_id",
        "tool_approvals",
        ["tool_instance_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_tool_approvals_tool_instance_id", table_name="tool_approvals")
    op.drop_index("ix_tool_approvals_session_id", table_name="tool_approvals")
    op.drop_index("ix_hitl_interrupts_session_id", table_name="hitl_interrupts")

    op.drop_column("tool_approvals", "tool_instance_id")
    op.drop_column("tool_approvals", "catalog_version")
    op.drop_column("tool_approvals", "session_id")

    op.drop_column("hitl_interrupts", "tool_instance_id")
    op.drop_column("hitl_interrupts", "catalog_version")
    op.drop_column("hitl_interrupts", "session_id")
