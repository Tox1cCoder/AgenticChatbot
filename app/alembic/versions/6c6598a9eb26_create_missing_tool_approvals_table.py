"""Reconcile the application schema and create tool approvals if missing.

Checkpoint and OAuth tables are outside this revision's ownership and remain
untouched. Task-execution metrics are intentionally retired: upgrade drops
their columns and values, while downgrade can restore only their schema and
historical NULL/zero defaults, not the discarded values.

Revision ID: 6c6598a9eb26
Revises: 1ce64a959f7d
Create Date: 2026-03-10 13:59:13.889216
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "6c6598a9eb26"
down_revision: str | None = "1ce64a959f7d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _column_names(table_name: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}


def _create_tool_approvals_if_missing() -> None:
    if sa.inspect(op.get_bind()).has_table("tool_approvals"):
        return

    op.execute(
        """
        DO $$ BEGIN
            CREATE TYPE decision_type AS ENUM ('accept', 'edit', 'reject');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
        """
    )
    op.create_table(
        "tool_approvals",
        sa.Column("id", sa.UUID(), nullable=False),
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
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("conversation_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("interrupt_id", sa.String(length=255), nullable=False),
        sa.Column("tool_name", sa.String(length=255), nullable=False),
        sa.Column("tool_call_id", sa.String(length=255), nullable=False),
        sa.Column("original_args", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("modified_args", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "decision",
            postgresql.ENUM(
                "accept",
                "edit",
                "reject",
                name="decision_type",
                create_type=False,
            ),
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
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_tool_approvals_user_id",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    for index_name, columns in (
        ("ix_tool_approvals_conversation_id", ["conversation_id"]),
        ("ix_tool_approvals_decided_at", ["decided_at"]),
        ("ix_tool_approvals_id", ["id"]),
        ("ix_tool_approvals_interrupt_id", ["interrupt_id"]),
        ("ix_tool_approvals_user_id", ["user_id"]),
    ):
        op.create_index(op.f(index_name), "tool_approvals", columns, unique=False)


def upgrade() -> None:
    """Apply application-owned reconciliation, intentionally retiring task metrics."""
    _create_tool_approvals_if_missing()

    for index_name, table_name, columns in (
        ("ix_agent_model_configs_id", "agent_model_configs", ["id"]),
        ("ix_agent_model_configs_user_id", "agent_model_configs", ["user_id"]),
        ("ix_document_images_document_id", "document_images", ["document_id"]),
        ("ix_hitl_interrupts_assistant_message_id", "hitl_interrupts", ["assistant_message_id"]),
        ("ix_model_providers_id", "model_providers", ["id"]),
        ("ix_model_providers_user_id", "model_providers", ["user_id"]),
    ):
        op.create_index(
            op.f(index_name),
            table_name,
            columns,
            unique=False,
            if_not_exists=True,
        )

    if "file_size" in _column_names("documents"):
        op.drop_column("documents", "file_size")

    op.drop_index(
        op.f("ix_hitl_interrupts_status_expires_at"),
        table_name="hitl_interrupts",
        if_exists=True,
    )
    hitl_foreign_keys = sa.inspect(op.get_bind()).get_foreign_keys("hitl_interrupts")
    if not any(
        foreign_key["constrained_columns"] == ["assistant_message_id"]
        and foreign_key["referred_table"] == "messages"
        and foreign_key["referred_columns"] == ["id"]
        and foreign_key["referred_schema"] in (None, "public")
        for foreign_key in hitl_foreign_keys
    ):
        op.create_foreign_key(
            None,
            "hitl_interrupts",
            "messages",
            ["assistant_message_id"],
            ["id"],
        )

    for retired_column in ("extra_metadata", "citations"):
        if retired_column in _column_names("messages"):
            op.drop_column("messages", retired_column)

    op.drop_index(
        op.f("ix_task_plans_conversation_order"),
        table_name="task_plans",
        if_exists=True,
    )
    op.drop_index(
        op.f("ix_task_plans_status"),
        table_name="task_plans",
        if_exists=True,
    )
    op.create_index(
        "idx_task_plan_conversation_order",
        "task_plans",
        ["conversation_id", "task_order"],
        unique=False,
        if_not_exists=True,
    )
    for retired_column in (
        "estimated_duration_minutes",
        "started_at",
        "completion_confidence",
        "retry_count",
        "actual_duration_minutes",
    ):
        if retired_column in _column_names("task_plans"):
            op.drop_column("task_plans", retired_column)


def downgrade() -> None:
    """Restore the canonical application schema at ``1ce64a959f7d``.

    Externally managed checkpoint tables and unrelated OAuth data are not
    created, dropped, or altered in either direction. The task-execution
    metric columns are restored structurally, but values removed by upgrade
    cannot be recovered: nullable metrics return as NULL and ``retry_count``
    returns with its historical zero default.
    """
    task_plan_column_types = {
        "started_at": sa.DateTime(timezone=True),
        "estimated_duration_minutes": sa.Integer(),
        "actual_duration_minutes": sa.Integer(),
        "retry_count": sa.Integer(),
        "completion_confidence": sa.Float(),
    }
    for column_name, column_type in task_plan_column_types.items():
        if column_name not in _column_names("task_plans"):
            op.add_column(
                "task_plans",
                sa.Column(
                    column_name,
                    column_type,
                    server_default=sa.text("0") if column_name == "retry_count" else None,
                    nullable=column_name != "retry_count",
                ),
            )

    op.drop_index(
        "idx_task_plan_conversation_order",
        table_name="task_plans",
        if_exists=True,
    )
    op.create_index(
        op.f("ix_task_plans_status"),
        "task_plans",
        ["status"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_task_plans_conversation_order"),
        "task_plans",
        ["conversation_id", "task_order"],
        unique=False,
        if_not_exists=True,
    )

    for index_name, table_name in (
        ("ix_model_providers_user_id", "model_providers"),
        ("ix_model_providers_id", "model_providers"),
        ("ix_agent_model_configs_user_id", "agent_model_configs"),
        ("ix_agent_model_configs_id", "agent_model_configs"),
        ("ix_document_images_document_id", "document_images"),
    ):
        op.drop_index(op.f(index_name), table_name=table_name, if_exists=True)

    op.create_index(
        op.f("ix_hitl_interrupts_status_expires_at"),
        "hitl_interrupts",
        ["status", "expires_at"],
        unique=False,
        if_not_exists=True,
    )
