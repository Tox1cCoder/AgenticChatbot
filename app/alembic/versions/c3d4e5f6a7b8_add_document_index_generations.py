"""Add atomic document index generations.

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-08-13
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "c3d4e5f6a7b8"
down_revision: str | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "document_index_generations",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("embedding_provider", sa.String(length=64), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("chunking_version", sa.String(length=64), nullable=False),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_document_index_generations_document",
        "document_index_generations",
        ["document_id"],
    )
    op.create_index(
        "idx_document_index_generations_status",
        "document_index_generations",
        ["status"],
    )
    op.create_index(
        "uq_document_index_generation_active",
        "document_index_generations",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    op.add_column(
        "document_chunks",
        sa.Column("index_generation_id", UUID(as_uuid=True), nullable=True),
    )

    bind = op.get_bind()
    documents = bind.execute(sa.text("SELECT id FROM documents ORDER BY id")).fetchall()
    for document in documents:
        generation_id = uuid.uuid4()
        bind.execute(
            sa.text(
                """
                INSERT INTO document_index_generations (
                    id, document_id, status, embedding_provider, embedding_model,
                    embedding_dimension, chunking_version, created_at, activated_at
                )
                SELECT :generation_id, d.id, 'active',
                       COALESCE(c.embedding_provider, 'legacy'),
                       COALESCE(c.embedding_model, 'legacy'),
                       COALESCE(c.embedding_dimension, 0),
                       'legacy-v1', NOW(), NOW()
                FROM documents d
                LEFT JOIN LATERAL (
                    SELECT
                        NULL::varchar AS embedding_provider,
                        embedding_model,
                        embedding_dimension
                    FROM document_chunks
                    WHERE document_id = d.id
                    ORDER BY chunk_index
                    LIMIT 1
                ) c ON TRUE
                WHERE d.id = :document_id
                """
            ),
            {"generation_id": generation_id, "document_id": document.id},
        )
        bind.execute(
            sa.text(
                "UPDATE document_chunks SET index_generation_id = :generation_id "
                "WHERE document_id = :document_id"
            ),
            {"generation_id": generation_id, "document_id": document.id},
        )

    op.alter_column("document_chunks", "index_generation_id", nullable=False)
    op.create_foreign_key(
        "document_chunks_index_generation_id_fkey",
        "document_chunks",
        "document_index_generations",
        ["index_generation_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "idx_document_chunks_index_generation_id",
        "document_chunks",
        ["index_generation_id"],
    )
    op.drop_constraint(
        "uq_document_chunk_document_index", "document_chunks", type_="unique"
    )
    op.create_unique_constraint(
        "uq_document_chunk_generation_index",
        "document_chunks",
        ["document_id", "index_generation_id", "chunk_index"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text(
            "DELETE FROM document_chunks c USING document_index_generations g "
            "WHERE c.index_generation_id = g.id AND g.status <> 'active'"
        )
    )
    op.drop_constraint(
        "uq_document_chunk_generation_index", "document_chunks", type_="unique"
    )
    op.create_unique_constraint(
        "uq_document_chunk_document_index",
        "document_chunks",
        ["document_id", "chunk_index"],
    )
    op.drop_index("idx_document_chunks_index_generation_id", table_name="document_chunks")
    op.drop_constraint(
        "document_chunks_index_generation_id_fkey",
        "document_chunks",
        type_="foreignkey",
    )
    op.drop_column("document_chunks", "index_generation_id")
    op.drop_index(
        "uq_document_index_generation_active",
        table_name="document_index_generations",
    )
    op.drop_index(
        "idx_document_index_generations_status",
        table_name="document_index_generations",
    )
    op.drop_index(
        "idx_document_index_generations_document",
        table_name="document_index_generations",
    )
    op.drop_table("document_index_generations")
