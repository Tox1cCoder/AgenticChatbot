"""task_plan: unique (conversation_id, task_order) constraint

Revision ID: h8i9j0k1l2m3
Revises: g7h8i9j0k1l2
Create Date: 2025-01-01 00:00:00.000000

Workstream B – DB hardening.

Steps (upgrade):
  1. Resequence duplicate task_order values per conversation using
     ROW_NUMBER() so the unique constraint can be applied cleanly.
  2. Drop the old non-unique index idx_task_plan_conversation_order.
  3. Add the DEFERRABLE INITIALLY DEFERRED unique constraint
     uq_task_plan_conversation_order.

Steps (downgrade):
  1. Drop the unique constraint.
  2. Recreate the old non-unique index.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "h8i9j0k1l2m3"
down_revision: str | Sequence[str] | None = "g7h8i9j0k1l2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DEDUP_SQL = """
WITH ranked AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY conversation_id
            ORDER BY task_order, created_at, id
        ) - 1 AS new_order
    FROM task_plans
)
UPDATE task_plans t
SET task_order = ranked.new_order
FROM ranked
WHERE t.id = ranked.id
  AND t.task_order != ranked.new_order;
"""


def upgrade() -> None:
    # 1. Fix any duplicate task_order values before adding the constraint.
    op.execute(_DEDUP_SQL)

    # 2. Drop the old non-unique index (may not exist on fresh DBs, so ignore).
    op.execute("DROP INDEX IF EXISTS idx_task_plan_conversation_order;")

    # 3. Add the DEFERRABLE unique constraint.
    op.create_unique_constraint(
        "uq_task_plan_conversation_order",
        "task_plans",
        ["conversation_id", "task_order"],
        deferrable=True,
        initially="DEFERRED",
    )


def downgrade() -> None:
    op.drop_constraint("uq_task_plan_conversation_order", "task_plans", type_="unique")
    op.create_index(
        "idx_task_plan_conversation_order",
        "task_plans",
        ["conversation_id", "task_order"],
    )
