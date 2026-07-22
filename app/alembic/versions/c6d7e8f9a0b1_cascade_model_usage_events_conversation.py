"""Cascade model-usage events when their conversation is deleted.

Revision ID: c6d7e8f9a0b1
Revises: b5c6d7e8f9a0
Create Date: 2026-07-22
"""

from collections.abc import Sequence

from alembic import op

revision: str = "c6d7e8f9a0b1"
down_revision: str | None = "b5c6d7e8f9a0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _replace_conversation_fk(*, ondelete: str) -> None:
    op.drop_constraint(
        "fk_model_usage_events_conversation",
        "model_usage_events",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_model_usage_events_conversation",
        "model_usage_events",
        "conversations",
        ["conversation_id"],
        ["id"],
        ondelete=ondelete,
    )


def upgrade() -> None:
    """Delete conversation-scoped raw usage with its owning conversation."""
    _replace_conversation_fk(ondelete="CASCADE")


def downgrade() -> None:
    """Restore the prior anonymizing foreign-key action."""
    _replace_conversation_fk(ondelete="SET NULL")
