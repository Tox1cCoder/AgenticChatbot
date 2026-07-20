"""scope editable HITL settings to client devices and tool origins

Revision ID: z3a4b5c6d7e8
Revises: y2z3a4b5c6d7
Create Date: 2026-07-20 00:00:00.000000

Legacy rows are reset because their account-wide identity cannot be mapped
reliably to one of a user's devices.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "z3a4b5c6d7e8"
down_revision: str | None = "y2z3a4b5c6d7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(
        "uq_tool_approval_settings_user_scope",
        "tool_approval_settings",
        type_="unique",
    )

    # No device can be inferred safely from a legacy account-wide rule.
    op.execute("DELETE FROM tool_approval_settings")
    op.add_column(
        "tool_approval_settings",
        sa.Column("device_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "tool_approval_settings",
        sa.Column("tool_origin", sa.String(length=32), nullable=True),
    )
    op.create_foreign_key(
        "fk_tool_approval_settings_device_id_client_devices",
        "tool_approval_settings",
        "client_devices",
        ["device_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_check_constraint(
        "ck_tool_approval_settings_tool_origin",
        "tool_approval_settings",
        "tool_origin IN ('client_mcp', 'client_skill')",
    )
    op.create_unique_constraint(
        "uq_tool_approval_settings_user_device_origin_scope",
        "tool_approval_settings",
        ["user_id", "device_id", "tool_origin", "scope_type", "scope_value"],
    )
    op.create_index(
        "ix_tool_approval_settings_device_id",
        "tool_approval_settings",
        ["device_id"],
        unique=False,
    )
    op.create_index(
        "ix_tool_approval_settings_tool_origin",
        "tool_approval_settings",
        ["tool_origin"],
        unique=False,
    )
    op.alter_column("tool_approval_settings", "device_id", nullable=False)
    op.alter_column("tool_approval_settings", "tool_origin", nullable=False)


def downgrade() -> None:
    # Legacy rows cannot be restored after the approved one-time reset.
    op.drop_index("ix_tool_approval_settings_tool_origin", table_name="tool_approval_settings")
    op.drop_index("ix_tool_approval_settings_device_id", table_name="tool_approval_settings")
    op.drop_constraint(
        "uq_tool_approval_settings_user_device_origin_scope",
        "tool_approval_settings",
        type_="unique",
    )
    op.drop_constraint(
        "ck_tool_approval_settings_tool_origin",
        "tool_approval_settings",
        type_="check",
    )
    op.drop_constraint(
        "fk_tool_approval_settings_device_id_client_devices",
        "tool_approval_settings",
        type_="foreignkey",
    )
    op.drop_column("tool_approval_settings", "tool_origin")
    op.drop_column("tool_approval_settings", "device_id")
    op.create_unique_constraint(
        "uq_tool_approval_settings_user_scope",
        "tool_approval_settings",
        ["user_id", "scope_type", "scope_value"],
    )
