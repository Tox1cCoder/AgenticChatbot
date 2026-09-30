"""add_users_token_version

Revision ID: 60adc43e534e
Revises: de19068933b7
Create Date: 2026-09-30 00:00:00.000000

Adds ``users.token_version``. Access and refresh tokens carry it as the ``ver``
claim, and a token whose ``ver`` is below the user's current version is refused,
so bumping the column revokes every token issued before the bump.

Existing rows get 0, and a token issued before this revision has no ``ver`` and
counts as version 0, so nobody is logged out by the upgrade. No existing row can
violate the column: it is new, and the constant server default fills it (a
catalog-only change on PostgreSQL 11+, no table rewrite).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "60adc43e534e"
down_revision: str | Sequence[str] | None = "de19068933b7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the per-user token version, 0 for every existing user."""

    op.add_column(
        "users",
        sa.Column("token_version", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )


def downgrade() -> None:
    """Drop the column. Tokens are then no longer revocable by version."""

    op.drop_column("users", "token_version")
