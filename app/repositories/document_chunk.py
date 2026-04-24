"""Repository for the canonical document_chunks content store.

Follows the sync session-factory style used by the rest of the server's
data layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import UUID

from sqlalchemy.orm import joinedload

from app.models.document_chunk import DocumentChunk


class DocumentChunkRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    def replace_document_chunks(
        self,
        document_id: UUID,
        chunk_rows: list[dict[str, Any]],
    ) -> list[DocumentChunk]:
        """Replace all chunks for a document in a single transaction.

        ``chunk_rows`` is a list of kwargs dicts; each one is used to construct
        a ``DocumentChunk`` row. Callers are responsible for populating
        ``content``, ``content_sha256``, ``char_count``, ``token_count`` and
        any deterministic ``id`` / ``qdrant_point_id`` they want.
        """
        with self.session_factory() as db:
            db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).delete(synchronize_session=False)

            created: list[DocumentChunk] = []
            for row in chunk_rows:
                chunk = DocumentChunk(document_id=document_id, **row)
                db.add(chunk)
                created.append(chunk)

            db.commit()
            for chunk in created:
                db.refresh(chunk)
            return created

    def delete_by_document(self, document_id: UUID) -> int:
        with self.session_factory() as db:
            deleted = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.document_id == document_id)
                .delete(synchronize_session=False)
            )
            db.commit()
            return deleted

    def mark_indexed(
        self,
        chunk_id: UUID,
        *,
        point_id: str,
        embedding_model: str,
        embedding_dimension: int,
        collection_name: str,
    ) -> DocumentChunk | None:
        with self.session_factory() as db:
            chunk = db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()
            if chunk is None:
                return None
            chunk.qdrant_point_id = point_id
            chunk.embedding_model = embedding_model
            chunk.embedding_dimension = embedding_dimension
            chunk.qdrant_collection_name = collection_name
            chunk.index_status = "indexed"
            chunk.index_error = None
            chunk.indexed_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(chunk)
            return chunk

    def mark_index_failed(self, chunk_id: UUID, error: str) -> DocumentChunk | None:
        with self.session_factory() as db:
            chunk = db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()
            if chunk is None:
                return None
            chunk.index_status = "failed"
            chunk.index_error = error
            db.commit()
            db.refresh(chunk)
            return chunk

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    def get_by_document_ordered(self, document_id: UUID) -> list[DocumentChunk]:
        with self.session_factory() as db:
            return (
                db.query(DocumentChunk)
                .options(joinedload(DocumentChunk.document))
                .filter(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index.asc())
                .all()
            )

    def get_by_ids(self, chunk_ids: Iterable[UUID]) -> list[DocumentChunk]:
        ids = list(chunk_ids)
        if not ids:
            return []
        with self.session_factory() as db:
            return (
                db.query(DocumentChunk)
                .options(joinedload(DocumentChunk.document))
                .filter(DocumentChunk.id.in_(ids))
                .all()
            )

    def get_by_qdrant_point_ids(self, point_ids: Iterable[str]) -> list[DocumentChunk]:
        ids = [p for p in point_ids if p]
        if not ids:
            return []
        with self.session_factory() as db:
            return (
                db.query(DocumentChunk)
                .options(joinedload(DocumentChunk.document))
                .filter(DocumentChunk.qdrant_point_id.in_(ids))
                .all()
            )
