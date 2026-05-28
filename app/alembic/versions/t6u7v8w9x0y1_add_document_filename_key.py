"""document: add filename_key with backfill and per-conversation uniqueness

Revision ID: t6u7v8w9x0y1
Revises: s2t3u4v5w6x7
Create Date: 2026-05-19 00:00:00.000000

RAG batch-upload overhaul (Phase 2).

Adds ``documents.filename_key`` for race-safe duplicate-filename detection
scoped to a conversation. Backfills the column from existing filenames
using the same casefold/NFC normalization that the application now uses
at request time. Historical same-conversation duplicates are preserved
by suffixing older duplicate keys with ``::legacy::<document_id>`` so
the unique constraint can be created without deleting data.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "t6u7v8w9x0y1"
down_revision: str | Sequence[str] | None = "s2t3u4v5w6x7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _normalize(filename: str) -> str:
    name = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    name = unicodedata.normalize("NFC", name)
    return name.casefold()


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("filename_key", sa.String(length=255), nullable=True),
    )

    bind = op.get_bind()
    rows = bind.execute(sa.text("SELECT id, conversation_id, filename FROM documents")).fetchall()

    # First pass: compute the normalized key for every row.
    keyed: list[tuple[str, str, str]] = []
    for row_id, conversation_id, filename in rows:
        key = _normalize(filename or "")
        # Defensive fallback — empty normalized name should never happen in
        # practice but we don't want to leave the column nullable forever.
        if not key:
            key = f"unknown::{row_id}"
        keyed.append((str(row_id), str(conversation_id), key))

    # Detect duplicates within the same conversation. Keep the first
    # canonical key, suffix older duplicates with ``::legacy::<id>`` so the
    # unique constraint can be created without deleting any history.
    seen: dict[tuple[str, str], str] = {}
    for row_id, conversation_id, key in keyed:
        composite_key = (conversation_id, key)
        if composite_key in seen:
            stored_key = f"{key}::legacy::{row_id}"
        else:
            stored_key = key
            seen[composite_key] = row_id

        bind.execute(
            sa.text("UPDATE documents SET filename_key = :k WHERE id = :id"),
            {"k": stored_key, "id": row_id},
        )

    op.alter_column(
        "documents",
        "filename_key",
        existing_type=sa.String(length=255),
        nullable=False,
    )

    op.create_unique_constraint(
        "uq_documents_conversation_filename_key",
        "documents",
        ["conversation_id", "filename_key"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_documents_conversation_filename_key",
        "documents",
        type_="unique",
    )
    op.drop_column("documents", "filename_key")
