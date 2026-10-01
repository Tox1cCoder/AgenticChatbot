"""model_providers: rebuild idx_model_providers_user_type as a partial index

Revision ID: 9b132832d676
Revises: e983df693ada
Create Date: 2026-09-30 00:00:00.000000

``e5f6g7h8i9j0`` creates this unique index with ``WHERE deleted_at IS NULL`` and
the model declares it so, but the application database was found on 2026-09-30
(read-only check of ``pg_indexes``) with a plain UNIQUE index under the same
name. With it, a soft-deleted provider keeps occupying its (user_id,
provider_type) pair, so that provider type can never be added again.

The upgrade drops the index whatever its definition and recreates it partial,
which makes it idempotent: a database the chain built gets the same index back.
Before touching anything it refuses if live rows already duplicate (user_id,
provider_type), which only a database missing the index can hold, because a bare
CREATE UNIQUE INDEX failure would block API startup with no explanation.

The downgrade changes nothing: the partial index is what every earlier revision
defines, and restoring the plain one would restore the bug.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9b132832d676"
down_revision: str | Sequence[str] | None = "e983df693ada"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX = "idx_model_providers_user_type"
_LIVE_DUPLICATE_GROUPS = (
    "SELECT count(*) FROM (SELECT 1 FROM model_providers WHERE deleted_at IS NULL "
    "GROUP BY user_id, provider_type HAVING count(*) > 1) duplicates"
)


def upgrade() -> None:
    """Refuse on live duplicates, then rebuild the index partial."""

    if op.get_context().as_sql:
        raise RuntimeError(
            "9b132832d676 counts duplicate providers before rebuilding a unique index "
            "and requires an online PostgreSQL connection"
        )
    duplicates = op.get_bind().execute(sa.text(_LIVE_DUPLICATE_GROUPS)).scalar_one()
    if duplicates:
        raise RuntimeError(
            f"9b132832d676 changed nothing: model_providers has {duplicates} (user_id, "
            "provider_type) groups with more than one live row (deleted_at IS NULL), and "
            f"{_INDEX} allows one. Soft-delete the extra rows, then start again."
        )
    op.drop_index(_INDEX, table_name="model_providers", if_exists=True)
    op.create_index(
        _INDEX,
        "model_providers",
        ["user_id", "provider_type"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )


def downgrade() -> None:
    """Nothing to undo; see the module docstring."""
