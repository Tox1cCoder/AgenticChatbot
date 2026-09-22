"""scope_user_memories_to_projects

Revision ID: 9778bb07ea35
Revises: 9a8b7c6d5e4f
Create Date: 2026-09-22 00:00:00.000000

Adds the project scope to agent-editable user memory.

``project_id`` is nullable and NULL means "global": a memory saved from a
conversation that belongs to no project. Recall reads
``project_id = :project OR project_id IS NULL``, so a project sees its own
memories plus the global ones, and never another project's. Existing rows
predate projects and are therefore global by construction - no backfill is
needed, and none is possible, since nothing recorded where they came from.

``ondelete="SET NULL"`` rather than CASCADE: deleting a project must not
destroy facts the user asked to be remembered. They fall back to global.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = "9778bb07ea35"
down_revision: str | Sequence[str] | None = "9a8b7c6d5e4f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable project scope and the recall index."""

    op.add_column(
        "user_memories",
        sa.Column("project_id", UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_user_memories_project_id",
        "user_memories",
        "projects",
        ["project_id"],
        ["id"],
        ondelete="SET NULL",
    )
    # Covers the recall predicate (user + scope, live rows only).
    op.create_index(
        "ix_user_memories_user_project",
        "user_memories",
        ["user_id", "project_id", "deleted_at"],
    )


def downgrade() -> None:
    """Drop the project scope. Every surviving memory becomes global."""

    op.drop_index("ix_user_memories_user_project", table_name="user_memories")
    op.drop_constraint("fk_user_memories_project_id", "user_memories", type_="foreignkey")
    op.drop_column("user_memories", "project_id")
