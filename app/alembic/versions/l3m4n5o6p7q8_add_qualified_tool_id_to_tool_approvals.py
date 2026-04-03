"""add qualified tool id to tool approvals

Revision ID: l3m4n5o6p7q8
Revises: k2l3m4n5o6p7
Create Date: 2026-03-19 15:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "l3m4n5o6p7q8"
down_revision: str | None = "k2l3m4n5o6p7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "tool_approvals",
        sa.Column("qualified_tool_id", sa.String(length=512), nullable=True),
    )
    op.create_index(
        op.f("ix_tool_approvals_qualified_tool_id"),
        "tool_approvals",
        ["qualified_tool_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_tool_approvals_qualified_tool_id"), table_name="tool_approvals")
    op.drop_column("tool_approvals", "qualified_tool_id")
