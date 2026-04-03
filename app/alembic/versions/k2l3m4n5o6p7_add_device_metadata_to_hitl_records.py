"""add device metadata columns to hitl interrupts and tool approvals

Revision ID: k2l3m4n5o6p7
Revises: j1k2l3m4n5o6
Create Date: 2026-03-17 01:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "k2l3m4n5o6p7"
down_revision: str | None = "j1k2l3m4n5o6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "hitl_interrupts",
        sa.Column("device_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "hitl_interrupts",
        sa.Column(
            "interrupt_metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_index(
        op.f("ix_hitl_interrupts_device_id"), "hitl_interrupts", ["device_id"], unique=False
    )
    op.create_foreign_key(
        "fk_hitl_interrupts_device_id_client_devices",
        "hitl_interrupts",
        "client_devices",
        ["device_id"],
        ["id"],
    )

    op.add_column(
        "tool_approvals",
        sa.Column("device_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "tool_approvals",
        sa.Column("tool_origin", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "tool_approvals",
        sa.Column("server_name", sa.String(length=255), nullable=True),
    )
    op.create_index(
        op.f("ix_tool_approvals_device_id"), "tool_approvals", ["device_id"], unique=False
    )
    op.create_index(
        op.f("ix_tool_approvals_tool_origin"),
        "tool_approvals",
        ["tool_origin"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_tool_approvals_device_id_client_devices",
        "tool_approvals",
        "client_devices",
        ["device_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_tool_approvals_device_id_client_devices",
        "tool_approvals",
        type_="foreignkey",
    )
    op.drop_index(op.f("ix_tool_approvals_tool_origin"), table_name="tool_approvals")
    op.drop_index(op.f("ix_tool_approvals_device_id"), table_name="tool_approvals")
    op.drop_column("tool_approvals", "server_name")
    op.drop_column("tool_approvals", "tool_origin")
    op.drop_column("tool_approvals", "device_id")

    op.drop_constraint(
        "fk_hitl_interrupts_device_id_client_devices",
        "hitl_interrupts",
        type_="foreignkey",
    )
    op.drop_index(op.f("ix_hitl_interrupts_device_id"), table_name="hitl_interrupts")
    op.drop_column("hitl_interrupts", "interrupt_metadata_json")
    op.drop_column("hitl_interrupts", "device_id")
