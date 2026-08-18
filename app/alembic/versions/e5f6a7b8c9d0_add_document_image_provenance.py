"""Add bbox, section_path, and content_sha256 provenance to document_images.

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-08-13
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "e5f6a7b8c9d0"
down_revision: str | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document_images",
        sa.Column("bbox", JSONB, nullable=True),
    )
    op.add_column(
        "document_images",
        sa.Column(
            "section_path",
            JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "document_images",
        sa.Column("content_sha256", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "idx_document_images_content_sha256",
        "document_images",
        ["content_sha256"],
    )


def downgrade() -> None:
    op.drop_index("idx_document_images_content_sha256", table_name="document_images")
    op.drop_column("document_images", "content_sha256")
    op.drop_column("document_images", "section_path")
    op.drop_column("document_images", "bbox")
