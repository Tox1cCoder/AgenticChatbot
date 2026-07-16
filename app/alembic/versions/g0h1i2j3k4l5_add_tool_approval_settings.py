"""add tool_approval_settings (per-user HITL approval policy)

Revision ID: g0h1i2j3k4l5
Revises: f03e63aa5a33
Create Date: 2026-06-22 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "g0h1i2j3k4l5"
down_revision: str | None = "f03e63aa5a33"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tool_approval_settings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scope_type", sa.String(length=16), nullable=False),
        sa.Column("scope_value", sa.String(length=512), nullable=False),
        sa.Column("require_approval", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id", "scope_type", "scope_value", name="uq_tool_approval_settings_user_scope"
        ),
        sa.CheckConstraint(
            "scope_type IN ('server', 'tool')",
            name="ck_tool_approval_settings_scope_type",
        ),
    )
    op.create_index(
        op.f("ix_tool_approval_settings_id"), "tool_approval_settings", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_tool_approval_settings_user_id"),
        "tool_approval_settings",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_tool_approval_settings_scope_value"),
        "tool_approval_settings",
        ["scope_value"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_tool_approval_settings_scope_value"), table_name="tool_approval_settings"
    )
    op.drop_index(op.f("ix_tool_approval_settings_user_id"), table_name="tool_approval_settings")
    op.drop_index(op.f("ix_tool_approval_settings_id"), table_name="tool_approval_settings")
    op.drop_table("tool_approval_settings")
