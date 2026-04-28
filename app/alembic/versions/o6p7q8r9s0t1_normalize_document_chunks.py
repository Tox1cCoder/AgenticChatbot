"""normalize_document_chunks

Revision ID: o6p7q8r9s0t1
Revises: n5o6p7q8r9s0
Create Date: 2026-04-24

Phase 2 of the RAG overhaul: promote ``document_chunks`` into the canonical
content store that backs retrieval.

The live database may or may not already have the original thin
``document_chunks`` table created by migration ``1d24e8e1ec28``. This
migration handles both cases:

* If the table exists, it is altered in place. Existing rows are backfilled
  with placeholder ``content_sha256`` / ``content`` values and marked
  ``index_status = 'needs_reindex'`` so the Phase 10 reindex job can rebuild
  them before production traffic switches over.
* If the table does not exist (because the historical migration was never
  applied against this database, or the table was dropped manually), the
  normalized schema is created from scratch.

The ORM model is the source of truth for the final column list. See
``app/models/document_chunk.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "o6p7q8r9s0t1"
down_revision: str | None = "n5o6p7q8r9s0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _table_exists(bind, name: str) -> bool:
    return sa.inspect(bind).has_table(name)


def _create_full_schema() -> None:
    """Create document_chunks with the final normalized schema."""
    op.create_table(
        "document_chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("document_id", UUID(as_uuid=True), nullable=False),
        sa.Column("parse_artifact_id", UUID(as_uuid=True), nullable=True),
        sa.Column("qdrant_point_id", sa.String(length=100), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("section_path", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column(
            "block_provenance", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
        sa.Column("chunk_metadata", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "index_status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("index_error", sa.Text(), nullable=True),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("embedding_model", sa.String(), nullable=True),
        sa.Column("embedding_dimension", sa.Integer(), nullable=True),
        sa.Column("qdrant_collection_name", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
            name="document_chunks_document_id_fkey",
        ),
        sa.ForeignKeyConstraint(
            ["parse_artifact_id"],
            ["document_parse_artifacts.id"],
            ondelete="SET NULL",
            name="document_chunks_parse_artifact_id_fkey",
        ),
        sa.UniqueConstraint("qdrant_point_id", name="document_chunks_qdrant_point_id_key"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_document_index"),
    )


def _add_new_columns_to_existing_table() -> None:
    """Add the new columns as nullable so existing rows survive."""
    op.add_column("document_chunks", sa.Column("content", sa.Text(), nullable=True))
    op.add_column(
        "document_chunks", sa.Column("content_sha256", sa.String(length=64), nullable=True)
    )
    op.add_column("document_chunks", sa.Column("char_count", sa.Integer(), nullable=True))
    op.add_column(
        "document_chunks", sa.Column("parse_artifact_id", UUID(as_uuid=True), nullable=True)
    )
    op.add_column("document_chunks", sa.Column("page_start", sa.Integer(), nullable=True))
    op.add_column("document_chunks", sa.Column("page_end", sa.Integer(), nullable=True))
    op.add_column("document_chunks", sa.Column("section_path", JSONB(), nullable=True))
    op.add_column("document_chunks", sa.Column("block_provenance", JSONB(), nullable=True))
    op.add_column("document_chunks", sa.Column("chunk_metadata", JSONB(), nullable=True))
    op.add_column("document_chunks", sa.Column("index_status", sa.String(length=32), nullable=True))
    op.add_column("document_chunks", sa.Column("index_error", sa.Text(), nullable=True))
    op.add_column(
        "document_chunks", sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("document_chunks", sa.Column("embedding_model", sa.String(), nullable=True))
    op.add_column("document_chunks", sa.Column("embedding_dimension", sa.Integer(), nullable=True))
    op.add_column(
        "document_chunks", sa.Column("qdrant_collection_name", sa.String(), nullable=True)
    )
    op.add_column(
        "document_chunks", sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True)
    )


def _backfill_existing_rows(bind) -> None:
    rows = bind.execute(
        sa.text("SELECT id, content_preview, token_count FROM document_chunks")
    ).fetchall()
    for row in rows:
        content_value = row.content_preview or ""
        sha = sha256(content_value.encode("utf-8")).hexdigest()
        bind.execute(
            sa.text(
                """
                UPDATE document_chunks
                SET content = :content,
                    content_sha256 = :sha,
                    char_count = :cc,
                    token_count = COALESCE(token_count, 0),
                    section_path = COALESCE(section_path, '[]'::jsonb),
                    block_provenance = COALESCE(block_provenance, '[]'::jsonb),
                    chunk_metadata = COALESCE(chunk_metadata, '{}'::jsonb),
                    index_status = 'needs_reindex',
                    updated_at = NOW()
                WHERE id = :id
                """
            ),
            {"content": content_value, "sha": sha, "cc": len(content_value), "id": row.id},
        )


def _tighten_existing_table() -> None:
    for col in (
        "content",
        "content_sha256",
        "char_count",
        "token_count",
        "section_path",
        "block_provenance",
        "chunk_metadata",
        "index_status",
        "updated_at",
    ):
        op.alter_column("document_chunks", col, nullable=False)

    op.drop_constraint(
        "document_chunks_document_id_fkey",
        "document_chunks",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "document_chunks_document_id_fkey",
        "document_chunks",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "document_chunks_parse_artifact_id_fkey",
        "document_chunks",
        "document_parse_artifacts",
        ["parse_artifact_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.alter_column(
        "document_chunks",
        "qdrant_point_id",
        existing_type=sa.String(length=100),
        nullable=True,
    )
    op.create_unique_constraint(
        "uq_document_chunk_document_index",
        "document_chunks",
        ["document_id", "chunk_index"],
    )


def _create_indexes() -> None:
    op.create_index(
        "idx_document_chunks_document_id",
        "document_chunks",
        ["document_id"],
        if_not_exists=True,
    )
    op.create_index(
        "idx_document_chunks_parse_artifact_id",
        "document_chunks",
        ["parse_artifact_id"],
        if_not_exists=True,
    )
    op.create_index(
        "idx_document_chunks_qdrant_point_id",
        "document_chunks",
        ["qdrant_point_id"],
        if_not_exists=True,
    )
    op.create_index(
        "idx_document_chunks_content_sha256",
        "document_chunks",
        ["content_sha256"],
        if_not_exists=True,
    )
    op.create_index(
        "idx_document_chunks_index_status",
        "document_chunks",
        ["index_status"],
        if_not_exists=True,
    )


def _promote_document_image_chunk_fk(bind) -> None:
    # Only add the FK if it does not already exist.
    insp = sa.inspect(bind)
    fk_names = {fk["name"] for fk in insp.get_foreign_keys("document_images")}
    if "document_images_chunk_id_fkey" in fk_names:
        return

    # Historical rows may carry chunk_id values that point at chunks that no
    # longer exist (or were never created against this DB). Null those out
    # before the FK is enforced.
    bind.execute(
        sa.text(
            """
            UPDATE document_images
            SET chunk_id = NULL
            WHERE chunk_id IS NOT NULL
              AND chunk_id NOT IN (SELECT id FROM document_chunks)
            """
        )
    )

    op.create_foreign_key(
        "document_images_chunk_id_fkey",
        "document_images",
        "document_chunks",
        ["chunk_id"],
        ["id"],
        ondelete="SET NULL",
    )


def upgrade() -> None:
    bind = op.get_bind()

    if not _table_exists(bind, "document_chunks"):
        _create_full_schema()
    else:
        _add_new_columns_to_existing_table()
        _backfill_existing_rows(bind)
        _tighten_existing_table()

    _create_indexes()
    _promote_document_image_chunk_fk(bind)


def downgrade() -> None:
    bind = op.get_bind()

    insp = sa.inspect(bind)
    fk_names = {fk["name"] for fk in insp.get_foreign_keys("document_images")}
    if "document_images_chunk_id_fkey" in fk_names:
        op.drop_constraint(
            "document_images_chunk_id_fkey",
            "document_images",
            type_="foreignkey",
        )

    for idx in (
        "idx_document_chunks_index_status",
        "idx_document_chunks_content_sha256",
        "idx_document_chunks_qdrant_point_id",
        "idx_document_chunks_parse_artifact_id",
        "idx_document_chunks_document_id",
    ):
        op.drop_index(idx, table_name="document_chunks", if_exists=True)

    op.drop_constraint(
        "uq_document_chunk_document_index",
        "document_chunks",
        type_="unique",
    )

    op.drop_constraint(
        "document_chunks_parse_artifact_id_fkey",
        "document_chunks",
        type_="foreignkey",
    )
    op.drop_constraint(
        "document_chunks_document_id_fkey",
        "document_chunks",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "document_chunks_document_id_fkey",
        "document_chunks",
        "documents",
        ["document_id"],
        ["id"],
    )

    op.alter_column(
        "document_chunks",
        "qdrant_point_id",
        existing_type=sa.String(length=100),
        nullable=False,
    )

    for name in (
        "updated_at",
        "qdrant_collection_name",
        "embedding_dimension",
        "embedding_model",
        "indexed_at",
        "index_error",
        "index_status",
        "chunk_metadata",
        "block_provenance",
        "section_path",
        "page_end",
        "page_start",
        "parse_artifact_id",
        "char_count",
        "content_sha256",
        "content",
    ):
        op.drop_column("document_chunks", name)
