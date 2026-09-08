"""let a deleted message release the generation that referenced it

Revision ID: e1f2a3b4c5d6
Revises: d0e1f2a3b4c5
Create Date: 2026-09-08 00:00:00.000000

``generations.assistant_message_id`` was created with a plain foreign key, so
deleting an assistant message that any generation referenced raised
``ForeignKeyViolation`` and the delete failed outright:

    update or delete on table "messages" violates foreign key constraint
    "generations_assistant_message_id_fkey" on table "generations"

That is the wrong direction of authority. A generation row is bookkeeping
*about* a turn; it must not be able to veto the deletion of the message the
turn produced. ``ON DELETE SET NULL`` is the same choice
``model_usage_events.request_message_id`` already makes for the same reason,
and it keeps the lifecycle history — status, epochs, terminal reason — which
``CASCADE`` would silently destroy.

The constraint is dropped and recreated under an explicit name. The original
was unnamed, so PostgreSQL generated ``generations_assistant_message_id_fkey``;
that name is what the drop targets, and the replacement is named so a future
migration never has to guess again.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "e1f2a3b4c5d6"
down_revision: str | None = "d0e1f2a3b4c5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_GENERATED_NAME = "generations_assistant_message_id_fkey"
_EXPLICIT_NAME = "fk_generations_assistant_message"


def upgrade() -> None:
    op.drop_constraint(_GENERATED_NAME, "generations", type_="foreignkey")
    op.create_foreign_key(
        _EXPLICIT_NAME,
        "generations",
        "messages",
        ["assistant_message_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(_EXPLICIT_NAME, "generations", type_="foreignkey")
    op.create_foreign_key(
        _GENERATED_NAME,
        "generations",
        "messages",
        ["assistant_message_id"],
        ["id"],
    )
