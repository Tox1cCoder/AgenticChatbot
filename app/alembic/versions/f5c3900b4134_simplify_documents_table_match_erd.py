"""simplify_documents_table_match_erd

Revision ID: f5c3900b4134
Revises: 5c89d07a2037
Create Date: 2025-09-30 09:31:32.948363

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f5c3900b4134"
down_revision: str | None = "5c89d07a2037"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Drop unnecessary columns to match ERD
    op.drop_constraint("documents_uploader_id_fkey", "documents", type_="foreignkey")
    op.drop_column("documents", "uploader_id")
    op.drop_column("documents", "file_size")
    op.drop_column("documents", "processing_started_at")
    op.drop_column("documents", "processing_completed_at")
    op.drop_column("documents", "error_message")
    op.drop_column("documents", "qdrant_collection_name")
    op.drop_column("documents", "chunk_count")
    op.drop_column("documents", "created_at")
    op.drop_column("documents", "updated_at")


def downgrade() -> None:
    # Re-add columns if needed to rollback
    op.add_column(
        "documents",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.add_column(
        "documents",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.add_column(
        "documents",
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "documents",
        sa.Column("qdrant_collection_name", sa.String(length=100), nullable=True),
    )
    op.add_column("documents", sa.Column("error_message", sa.Text(), nullable=True))
    op.add_column(
        "documents",
        sa.Column("processing_completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("documents", sa.Column("file_size", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("uploader_id", sa.UUID(), nullable=True))
    op.create_foreign_key(
        "documents_uploader_id_fkey", "documents", "users", ["uploader_id"], ["id"]
    )
