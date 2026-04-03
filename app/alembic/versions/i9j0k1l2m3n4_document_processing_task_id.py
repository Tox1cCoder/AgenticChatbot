"""document: add processing_task_id column

Revision ID: i9j0k1l2m3n4
Revises: h8i9j0k1l2m3
Create Date: 2025-01-01 00:00:00.000000

Workstream D — Document task ownership.

Adds processing_task_id VARCHAR(255) NULL to the documents table, indexed
for fast lookups in the ownership-verification path of GET /documents/task/{id}.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "i9j0k1l2m3n4"
down_revision: str | Sequence[str] | None = "h8i9j0k1l2m3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("processing_task_id", sa.String(255), nullable=True),
    )
    op.create_index(
        "idx_document_processing_task_id",
        "documents",
        ["processing_task_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_document_processing_task_id", table_name="documents")
    op.drop_column("documents", "processing_task_id")
