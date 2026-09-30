"""document_chunks timestamp defaults and legacy columns

Revision ID: c0033ee1e8cd
Revises: 60adc43e534e
Create Date: 2026-09-30 00:00:00.000000

Two leftovers of how ``document_chunks`` came to exist.

``1d24e8e1ec28`` created the thin table with ``page_number`` and
``content_preview``, and ``o6p7q8r9s0t1`` altered it in place when it already
existed. That branch never gave ``created_at``/``updated_at`` a server default
and never dropped the two legacy columns, so a database built from the chain
has both problems. A database where the table did not exist yet took the
``_create_full_schema`` branch instead: the application database is one of
those, and it already has ``now()`` defaults and no legacy columns (checked
read-only on 2026-09-30).

So every statement here must hold on both shapes:

* ``SET DEFAULT now()`` is idempotent. The model has had a Python-side default
  since ``0a51a1e7``; the server default is defence for raw SQL inserts.
* The columns are dropped with ``IF EXISTS``. Nothing reads them: the model has
  no such attributes and no raw SQL in ``app/``, ``client_backend/`` or
  ``scripts/`` names them. ``content_preview`` was copied into ``content`` by
  ``o6p7q8r9s0t1``'s backfill, and ``page_number`` was superseded by
  ``page_start``/``page_end``. On the application database there is no data to
  lose because the columns are absent.

The downgrade restores the chain's previous shape: the columns come back empty
and the defaults are removed, including on a database that had them before
this revision.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c0033ee1e8cd"
down_revision: str | Sequence[str] | None = "60adc43e534e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TIMESTAMP_COLUMNS = ("created_at", "updated_at")
_LEGACY_COLUMNS = ("page_number", "content_preview")


def upgrade() -> None:
    """Give the timestamps a server default and drop the unread legacy columns."""

    for column in _TIMESTAMP_COLUMNS:
        op.alter_column("document_chunks", column, server_default=sa.text("now()"))
    for column in _LEGACY_COLUMNS:
        op.execute(f"ALTER TABLE document_chunks DROP COLUMN IF EXISTS {column}")


def downgrade() -> None:
    """Put back the chain's previous shape; the legacy columns come back empty."""

    op.add_column("document_chunks", sa.Column("page_number", sa.Integer(), nullable=True))
    op.add_column("document_chunks", sa.Column("content_preview", sa.Text(), nullable=True))
    for column in _TIMESTAMP_COLUMNS:
        op.alter_column("document_chunks", column, server_default=None)
