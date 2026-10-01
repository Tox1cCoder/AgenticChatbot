"""server_secrets: keys the server generated for itself

Revision ID: e983df693ada
Revises: 371ffaf3a087
Create Date: 2026-09-30 00:00:00.000000

Holds the token signing key and the provider-key encryption key when neither is
configured in the environment; ``app/core/server_secrets.py`` is the only reader
and writer. A value configured in the environment always wins and is never
copied here. Anyone holding a database backup holds whatever this table holds.

The downgrade refuses while the table holds the encryption key and
``model_providers`` holds keys encrypted with it: dropping the table would leave
them undecryptable. Set ``MODEL_ENCRYPTION_KEY`` to the stored value and delete
its row first.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e983df693ada"
down_revision: str | Sequence[str] | None = "371ffaf3a087"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_STRANDED_PROVIDER_KEYS = (
    "SELECT count(*) FROM model_providers WHERE api_key_encrypted <> '' "
    "AND EXISTS (SELECT 1 FROM server_secrets WHERE name = 'model_encryption_key')"
)


def upgrade() -> None:
    op.create_table(
        "server_secrets",
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    """Drop the table, unless that would strand encrypted provider keys."""

    if op.get_context().as_sql:
        raise RuntimeError(
            "e983df693ada checks for provider keys encrypted with the stored key "
            "before dropping it and requires an online PostgreSQL connection"
        )
    stranded = op.get_bind().execute(sa.text(_STRANDED_PROVIDER_KEYS)).scalar_one()
    if stranded:
        raise RuntimeError(
            f"e983df693ada changed nothing: {stranded} model_providers rows hold API keys "
            "encrypted with the key stored in server_secrets, and dropping the table would "
            "leave them undecryptable. Set MODEL_ENCRYPTION_KEY to that key (SELECT value "
            "FROM server_secrets WHERE name = 'model_encryption_key'), delete that row, "
            "then downgrade again."
        )
    op.drop_table("server_secrets")
