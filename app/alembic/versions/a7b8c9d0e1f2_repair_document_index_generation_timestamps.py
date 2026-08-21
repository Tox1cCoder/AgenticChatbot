"""Repair document index generation lifecycle timestamps.

Revision ``c3d4e5f6a7b8`` was edited twice after it had already been applied,
so its ``create_table`` gained ``retired_at`` and ``failed_at`` that no
already-stamped database ever received. Alembic never re-runs an applied
revision, which left those databases serving an ORM that selects two columns
PostgreSQL does not have. This online repair adds only the columns that are
absent, so databases migrated from the edited file are untouched.

Revision ID: a7b8c9d0e1f2
Revises: e5f6a7b8c9d0
Create Date: 2026-08-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: str | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "document_index_generations"
_LIFECYCLE_COLUMNS = ("retired_at", "failed_at")


def upgrade() -> None:
    """Add whichever lifecycle timestamps the stamped database is missing."""
    if op.get_context().as_sql:
        raise RuntimeError(
            f"migration {revision} requires an online PostgreSQL connection; "
            "offline SQL generation is unsupported"
        )
    inspector = sa.inspect(op.get_bind())
    present = {column["name"] for column in inspector.get_columns(_TABLE, schema="public")}
    for column_name in _LIFECYCLE_COLUMNS:
        if column_name not in present:
            op.add_column(
                _TABLE,
                sa.Column(column_name, sa.DateTime(timezone=True), nullable=True),
            )


def downgrade() -> None:
    """Keep the columns: ``c3d4e5f6a7b8`` declares them and owns the table."""
