"""Add a PostgreSQL full-text index for active chunk lexical retrieval.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-08-13
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: str | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "idx_document_chunks_content_simple_fts",
        "document_chunks",
        [sa.text("to_tsvector('simple', content)")],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_document_chunks_content_simple_fts",
        table_name="document_chunks",
    )
