"""add leading timestamp indexes for model-usage maintenance

Revision ID: a4b5c6d7e8f9
Revises: z3a4b5c6d7e8
Create Date: 2026-07-21 00:00:00.000000

The maintenance paths scan all tenants by time. Existing tenant-leading
composite indexes cannot support those predicates efficiently, so these
standalone indexes are created concurrently to avoid blocking production
writes while PostgreSQL builds them.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a4b5c6d7e8f9"
down_revision: str | None = "z3a4b5c6d7e8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        # CREATE INDEX CONCURRENTLY may leave an invalid same-name artifact if
        # interrupted. Always remove it (or a completed prior artifact) so a
        # retry deterministically rebuilds a valid index.
        op.drop_index(
            "ix_model_usage_events_started_at",
            table_name="model_usage_events",
            if_exists=True,
            postgresql_concurrently=True,
        )
        op.create_index(
            "ix_model_usage_events_started_at",
            "model_usage_events",
            ["started_at"],
            unique=False,
            postgresql_concurrently=True,
        )
        op.drop_index(
            "ix_model_usage_minute_bucket_start_utc",
            table_name="model_usage_minute",
            if_exists=True,
            postgresql_concurrently=True,
        )
        op.create_index(
            "ix_model_usage_minute_bucket_start_utc",
            "model_usage_minute",
            ["bucket_start_utc"],
            unique=False,
            postgresql_concurrently=True,
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.drop_index(
            "ix_model_usage_minute_bucket_start_utc",
            table_name="model_usage_minute",
            if_exists=True,
            postgresql_concurrently=True,
        )
        op.drop_index(
            "ix_model_usage_events_started_at",
            table_name="model_usage_events",
            if_exists=True,
            postgresql_concurrently=True,
        )
