"""drop the stray chat_images sha256 index

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-09-03 00:00:00.000000

``ix_chat_images_sha256`` exists on the live database but no migration ever
created it -- ``d7e8f9a0b1c2`` creates only the ``conversation_id`` and
``user_id`` indexes. It came from a ``Base.metadata.create_all`` against a
model that has since dropped ``index=True``, so a database built from
migrations does not have it and a database built from the models did. That
divergence is what made ``alembic check`` permanently dirty.

Dropping it rather than declaring it: the only query touching ``sha256`` is
``ChatImageRepository.get_by_user_and_sha``, which filters on ``user_id``
first, and ``pg_stat_user_indexes`` recorded 0 scans on this index against
3217 on ``ix_chat_images_user_id``. Keeping an index the planner never chooses
would mean carrying write cost for nothing.

``IF EXISTS`` / ``IF NOT EXISTS`` are deliberate: whether the index is present
depends on how a given database was built, so neither direction may assume it.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "c9d0e1f2a3b4"
down_revision: str | None = "b8c9d0e1f2a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chat_images_sha256")


def downgrade() -> None:
    op.execute("CREATE INDEX IF NOT EXISTS ix_chat_images_sha256 ON chat_images (sha256)")
