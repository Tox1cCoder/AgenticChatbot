"""add_model_providers_table

Revision ID: e5f6g7h8i9j0
Revises: d4e29f078096
Create Date: 2026-01-20 10:00:00.000000

"""

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision = "e5f6g7h8i9j0"
down_revision = "d4e29f078096"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """
    Create model_providers table for multi-provider AI model configuration.

    This table stores encrypted API keys for different AI providers (OpenAI, Anthropic, etc.)
    per user, enabling dynamic model switching across conversations.
    """
    # Create model_providers table
    op.create_table(
        "model_providers",
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
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("provider_type", sa.Text(), nullable=False),
        sa.Column("api_key_encrypted", sa.Text(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("provider_metadata", JSONB, nullable=True, server_default="{}"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_model_providers_user_id",
        ),
    )

    # Create indexes for efficient querying
    op.create_index("idx_model_providers_user_id", "model_providers", ["user_id"])

    # Create unique constraint on user_id + provider_type (excluding soft-deleted rows)
    op.execute(
        """
        CREATE UNIQUE INDEX idx_model_providers_user_type
        ON model_providers(user_id, provider_type)
        WHERE deleted_at IS NULL;
        """
    )

    # Create partial index for default providers
    op.execute(
        """
        CREATE INDEX idx_model_providers_user_default
        ON model_providers(user_id, is_default)
        WHERE is_default = true;
        """
    )


def downgrade() -> None:
    """
    Drop model_providers table and all related indexes.
    """
    op.drop_index("idx_model_providers_user_default", table_name="model_providers")
    op.drop_index("idx_model_providers_user_type", table_name="model_providers")
    op.drop_index("idx_model_providers_user_id", table_name="model_providers")
    op.drop_table("model_providers")
