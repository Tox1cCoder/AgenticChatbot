"""Alembic autogenerate filters.

Application migrations own only application tables. LangGraph checkpoint
tables and Alembic's version table are managed externally and must never be
dropped by app-model autogenerate.
"""

from __future__ import annotations

from typing import Any

EXTERNAL_TABLE_NAMES = {
    "alembic_version",
    "checkpoint_blobs",
    "checkpoint_migrations",
    "checkpoint_writes",
    "checkpoints",
}


def include_name(
    name: str | None,
    type_: str,
    parent_names: dict[str, Any],
) -> bool:
    if type_ == "table" and name in EXTERNAL_TABLE_NAMES:
        return False
    return True
