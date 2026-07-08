"""schema contract cleanup

Revision ID: v1w2x3y4z5a6
Revises: g0h1i2j3k4l5
Create Date: 2026-07-08
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "v1w2x3y4z5a6"
down_revision: str | None = "g0h1i2j3k4l5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_agent_model_configs_user_id", table_name="agent_model_configs")
    op.drop_index("ix_agent_model_configs_id", table_name="agent_model_configs")
    op.drop_index("ix_model_providers_user_id", table_name="model_providers")
    op.drop_index("ix_model_providers_id", table_name="model_providers")
    op.drop_index("ix_document_images_document_id", table_name="document_images")
    op.drop_index("idx_document_chunks_qdrant_point_id", table_name="document_chunks")

    for table_name, index_name in (
        ("client_devices", "ix_client_devices_id"),
        ("conversation_custom_agents", "ix_conversation_custom_agents_id"),
        ("conversations", "ix_conversations_id"),
        ("custom_agents", "ix_custom_agents_id"),
        ("document_parse_artifacts", "ix_document_parse_artifacts_id"),
        ("feedbacks", "ix_feedbacks_id"),
        ("messages", "ix_messages_id"),
        ("skill_settings", "ix_skill_settings_id"),
        ("task_plans", "ix_task_plans_id"),
        ("tool_approval_settings", "ix_tool_approval_settings_id"),
        ("tool_approvals", "ix_tool_approvals_id"),
        ("users", "ix_users_id"),
    ):
        op.drop_index(index_name, table_name=table_name)

    op.drop_table("conversation_device_bindings")


def downgrade() -> None:
    op.create_table(
        "conversation_device_bindings",
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("device_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("bound_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "bound_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(["bound_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.ForeignKeyConstraint(["device_id"], ["client_devices.id"]),
        sa.PrimaryKeyConstraint("conversation_id"),
    )
    op.create_index(
        "ix_conversation_device_bindings_bound_at",
        "conversation_device_bindings",
        ["bound_at"],
    )
    op.create_index(
        "ix_conversation_device_bindings_bound_by_user_id",
        "conversation_device_bindings",
        ["bound_by_user_id"],
    )
    op.create_index(
        "ix_conversation_device_bindings_conversation_id",
        "conversation_device_bindings",
        ["conversation_id"],
    )
    op.create_index(
        "ix_conversation_device_bindings_device_id",
        "conversation_device_bindings",
        ["device_id"],
    )

    op.create_index("ix_agent_model_configs_user_id", "agent_model_configs", ["user_id"])
    op.create_index("ix_agent_model_configs_id", "agent_model_configs", ["id"])
    op.create_index("ix_model_providers_user_id", "model_providers", ["user_id"])
    op.create_index("ix_model_providers_id", "model_providers", ["id"])
    op.create_index("ix_document_images_document_id", "document_images", ["document_id"])
    op.create_index("idx_document_chunks_qdrant_point_id", "document_chunks", ["qdrant_point_id"])

    for table_name, index_name in (
        ("client_devices", "ix_client_devices_id"),
        ("conversation_custom_agents", "ix_conversation_custom_agents_id"),
        ("conversations", "ix_conversations_id"),
        ("custom_agents", "ix_custom_agents_id"),
        ("document_parse_artifacts", "ix_document_parse_artifacts_id"),
        ("feedbacks", "ix_feedbacks_id"),
        ("messages", "ix_messages_id"),
        ("skill_settings", "ix_skill_settings_id"),
        ("task_plans", "ix_task_plans_id"),
        ("tool_approval_settings", "ix_tool_approval_settings_id"),
        ("tool_approvals", "ix_tool_approvals_id"),
        ("users", "ix_users_id"),
    ):
        op.create_index(index_name, table_name, ["id"])
