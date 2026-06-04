"""add custom agents and conversation attachments

Revision ID: u7v8w9x0y1z2
Revises: t6u7v8w9x0y1
Create Date: 2026-05-29 00:00:00.000000

Adds per-user custom agents (``custom_agents``) and their per-conversation
attachments (``conversation_custom_agents``). The live (non-deleted) slug is
unique per owner via a partial unique index so soft-deleting an agent frees
its slug for reuse.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "u7v8w9x0y1z2"
down_revision: str | Sequence[str] | None = "t6u7v8w9x0y1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "custom_agents",
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
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("provider_type", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.Column("reasoning_effort", sa.String(length=32), nullable=True),
        sa.Column(
            "tool_refs",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "skill_refs",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_custom_agents_id"), "custom_agents", ["id"], unique=False)
    op.create_index(op.f("ix_custom_agents_owner_id"), "custom_agents", ["owner_id"], unique=False)
    op.create_index(
        "ix_custom_agents_owner_deleted",
        "custom_agents",
        ["owner_id", "deleted_at"],
        unique=False,
    )
    op.create_index(
        "uq_custom_agents_owner_slug_active",
        "custom_agents",
        ["owner_id", "slug"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    op.create_table(
        "conversation_custom_agents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("custom_agent_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_order", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.ForeignKeyConstraint(["custom_agent_id"], ["custom_agents.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "conversation_id",
            "custom_agent_id",
            name="uq_conversation_custom_agents_conv_agent",
        ),
    )
    op.create_index(
        op.f("ix_conversation_custom_agents_id"),
        "conversation_custom_agents",
        ["id"],
        unique=False,
    )
    op.create_index(
        "ix_conversation_custom_agents_owner_conv",
        "conversation_custom_agents",
        ["owner_id", "conversation_id"],
        unique=False,
    )
    op.create_index(
        "ix_conversation_custom_agents_custom_agent_id",
        "conversation_custom_agents",
        ["custom_agent_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_conversation_custom_agents_custom_agent_id",
        table_name="conversation_custom_agents",
    )
    op.drop_index(
        "ix_conversation_custom_agents_owner_conv",
        table_name="conversation_custom_agents",
    )
    op.drop_index(
        op.f("ix_conversation_custom_agents_id"),
        table_name="conversation_custom_agents",
    )
    op.drop_table("conversation_custom_agents")

    op.drop_index("uq_custom_agents_owner_slug_active", table_name="custom_agents")
    op.drop_index("ix_custom_agents_owner_deleted", table_name="custom_agents")
    op.drop_index(op.f("ix_custom_agents_owner_id"), table_name="custom_agents")
    op.drop_index(op.f("ix_custom_agents_id"), table_name="custom_agents")
    op.drop_table("custom_agents")
