"""Repair the tool-approval table and decision enum at the current head.

This online-only repair covers databases already stamped at the preceding
head. It canonicalizes known lowercase/uppercase enum labels and recreates a
missing table exactly. Unknown enum labels, unsupported enum dependents, and
malformed tables fail rather than risking data loss.

Revision ID: b5c6d7e8f9a0
Revises: a4b5c6d7e8f9
Create Date: 2026-07-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.alembic.migration_helpers.tool_approval_repair_v1 import (
    CURRENT_DECISION_LABELS,
    canonicalize_decision_type,
    create_tool_approvals,
    require_online,
    validate_tool_approvals_schema,
)

revision: str = "b5c6d7e8f9a0"
down_revision: str | None = "a4b5c6d7e8f9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Canonicalize or recreate the current tool-approval schema."""
    require_online(op, revision)
    connection = op.get_bind()
    canonicalize_decision_type(connection, CURRENT_DECISION_LABELS)
    if not sa.inspect(connection).has_table("tool_approvals", schema="public"):
        create_tool_approvals(op, current=True)
    validate_tool_approvals_schema(connection, current=True)


def downgrade() -> None:
    """Deliberately preserve repaired schema and data as a safe no-op."""
