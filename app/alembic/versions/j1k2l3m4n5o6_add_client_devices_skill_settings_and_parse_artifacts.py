"""add client devices, conversation bindings, parse artifacts, and skill settings

Revision ID: j1k2l3m4n5o6
Revises: i9j0k1l2m3n4
Create Date: 2026-03-17 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "j1k2l3m4n5o6"
down_revision: str | None = "i9j0k1l2m3n4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "client_devices",
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
        sa.Column("device_identifier", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column(
            "platform",
            sa.String(length=64),
            nullable=False,
            server_default="windows",
        ),
        sa.Column("app_version", sa.String(length=64), nullable=True),
        sa.Column("runtime_version", sa.String(length=64), nullable=True),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="offline",
        ),
        sa.Column(
            "capabilities_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "device_identifier",
            name="uq_client_devices_user_device_id",
        ),
    )
    op.create_index(op.f("ix_client_devices_id"), "client_devices", ["id"], unique=False)
    op.create_index(op.f("ix_client_devices_user_id"), "client_devices", ["user_id"], unique=False)
    op.create_index(
        op.f("ix_client_devices_device_identifier"),
        "client_devices",
        ["device_identifier"],
        unique=False,
    )
    op.create_index(
        op.f("ix_client_devices_last_seen_at"),
        "client_devices",
        ["last_seen_at"],
        unique=False,
    )
    op.create_index(op.f("ix_client_devices_status"), "client_devices", ["status"], unique=False)

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
        op.f("ix_conversation_device_bindings_conversation_id"),
        "conversation_device_bindings",
        ["conversation_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_conversation_device_bindings_device_id"),
        "conversation_device_bindings",
        ["device_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_conversation_device_bindings_bound_by_user_id"),
        "conversation_device_bindings",
        ["bound_by_user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_conversation_device_bindings_bound_at"),
        "conversation_device_bindings",
        ["bound_at"],
        unique=False,
    )

    op.create_table(
        "document_parse_artifacts",
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
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("artifact_type", sa.String(length=64), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("checksum_sha256", sa.String(length=128), nullable=True),
        sa.Column(
            "artifact_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "artifact_type",
            "storage_path",
            name="uq_document_parse_artifact_path",
        ),
    )
    op.create_index(
        op.f("ix_document_parse_artifacts_id"),
        "document_parse_artifacts",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_document_parse_artifacts_document_id"),
        "document_parse_artifacts",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_document_parse_artifacts_artifact_type"),
        "document_parse_artifacts",
        ["artifact_type"],
        unique=False,
    )

    op.create_table(
        "skill_settings",
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
        sa.Column("skill_name", sa.String(length=255), nullable=False),
        sa.Column(
            "enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "skill_name", name="uq_skill_settings_user_skill"),
    )
    op.create_index(op.f("ix_skill_settings_id"), "skill_settings", ["id"], unique=False)
    op.create_index(op.f("ix_skill_settings_user_id"), "skill_settings", ["user_id"], unique=False)
    op.create_index(
        op.f("ix_skill_settings_skill_name"),
        "skill_settings",
        ["skill_name"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_skill_settings_skill_name"), table_name="skill_settings")
    op.drop_index(op.f("ix_skill_settings_user_id"), table_name="skill_settings")
    op.drop_index(op.f("ix_skill_settings_id"), table_name="skill_settings")
    op.drop_table("skill_settings")

    op.drop_index(
        op.f("ix_document_parse_artifacts_artifact_type"),
        table_name="document_parse_artifacts",
    )
    op.drop_index(
        op.f("ix_document_parse_artifacts_document_id"),
        table_name="document_parse_artifacts",
    )
    op.drop_index(op.f("ix_document_parse_artifacts_id"), table_name="document_parse_artifacts")
    op.drop_table("document_parse_artifacts")

    op.drop_index(
        op.f("ix_conversation_device_bindings_bound_at"),
        table_name="conversation_device_bindings",
    )
    op.drop_index(
        op.f("ix_conversation_device_bindings_bound_by_user_id"),
        table_name="conversation_device_bindings",
    )
    op.drop_index(
        op.f("ix_conversation_device_bindings_device_id"),
        table_name="conversation_device_bindings",
    )
    op.drop_index(
        op.f("ix_conversation_device_bindings_conversation_id"),
        table_name="conversation_device_bindings",
    )
    op.drop_table("conversation_device_bindings")

    op.drop_index(op.f("ix_client_devices_status"), table_name="client_devices")
    op.drop_index(op.f("ix_client_devices_last_seen_at"), table_name="client_devices")
    op.drop_index(op.f("ix_client_devices_device_identifier"), table_name="client_devices")
    op.drop_index(op.f("ix_client_devices_user_id"), table_name="client_devices")
    op.drop_index(op.f("ix_client_devices_id"), table_name="client_devices")
    op.drop_table("client_devices")
