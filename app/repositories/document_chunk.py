"""Repository for the canonical document_chunks content store.

Follows the sync session-factory style used by the rest of the server's
data layer.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.orm import joinedload

from app.models.conversation import Conversation
from app.models.document import Document
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
            db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete(
                synchronize_session=False
            )

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

    def mark_indexed_bulk(
        self,
        chunk_ids: list[UUID],
        *,
        point_ids: list[str],
        embedding_model: str,
        embedding_dimension: int,
        collection_name: str,
    ) -> int:
        """Bulk update all chunks with embedding metadata in a single operation.

        Args:
            chunk_ids: List of chunk IDs to update.
            point_ids: List of Qdrant point IDs (parallel to chunk_ids).
            embedding_model: Name of the embedding model.
            embedding_dimension: Dimension of the embeddings.
            collection_name: Name of the Qdrant collection.

        Returns:
            The number of chunks updated.
        """
        if not chunk_ids:
            return 0

        with self.session_factory() as db:
            now = datetime.now(timezone.utc)
            mappings = [
                {
                    "id": chunk_id,
                    "qdrant_point_id": point_id,
                    "embedding_model": embedding_model,
                    "embedding_dimension": embedding_dimension,
                    "qdrant_collection_name": collection_name,
                    "index_status": "indexed",
                    "index_error": None,
                    "indexed_at": now,
                }
                for chunk_id, point_id in zip(chunk_ids, point_ids, strict=True)
            ]
            db.bulk_update_mappings(DocumentChunk, mappings)
            db.commit()
            return len(mappings)

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

    def get_by_document_for_scope(
        self,
        document_id: UUID,
        *,
        user_id: Any | None = None,
        conversation_id: Any | None = None,
    ) -> list[DocumentChunk]:
        """Return chunks for ``document_id`` only when the parent ``Document``
        matches the given server-context filters.

        Auth filtering happens in the SQL ``WHERE`` clause via a join on
        ``Document``. Returns an empty list when filters don't match.
        """
        with self.session_factory() as db:
            query = (
                db.query(DocumentChunk)
                .options(joinedload(DocumentChunk.document))
                .join(Document, DocumentChunk.document_id == Document.id)
                .filter(DocumentChunk.document_id == document_id)
            )
            if conversation_id is not None:
                query = query.filter(Document.conversation_id == conversation_id)
            if user_id is not None:
                # Documents are scoped via ``Conversation.owner_id``; the model
                # exposes this through the relationship, so we restrict by
                # joining to the conversation owner.
                from app.models.conversation import Conversation

                query = query.join(
                    Conversation, Document.conversation_id == Conversation.id
                ).filter(Conversation.owner_id == user_id)
            return query.order_by(DocumentChunk.chunk_index.asc()).all()

    def get_window_for_scope(
        self,
        document_id: UUID,
        user_id: Any | None,
        conversation_id: Any | None,
        start_chunk: int,
        max_chunks: int,
    ) -> list[DocumentChunk]:
        """Return one bounded, ownership-filtered window of document chunks."""
        if user_id is None and conversation_id is None:
            return []

        bounded_start = max(0, int(start_chunk))
        bounded_limit = min(20, max(1, int(max_chunks)))

        with self.session_factory() as db:
            query = (
                db.query(DocumentChunk)
                .options(joinedload(DocumentChunk.document))
                .join(Document, DocumentChunk.document_id == Document.id)
                .filter(DocumentChunk.document_id == document_id)
            )
            if conversation_id is not None:
                query = query.filter(Document.conversation_id == conversation_id)
            if user_id is not None:
                query = query.join(
                    Conversation, Document.conversation_id == Conversation.id
                ).filter(Conversation.owner_id == user_id)
            return (
                query.order_by(DocumentChunk.chunk_index.asc())
                .offset(bounded_start)
                .limit(bounded_limit)
                .all()
            )

    def has_chunk_after_for_scope(
        self,
        document_id: UUID,
        user_id: Any | None,
        conversation_id: Any | None,
        after_offset: int,
    ) -> bool:
        """Check for a later chunk without hydrating content outside the window."""
        if user_id is None and conversation_id is None:
            return False

        with self.session_factory() as db:
            query = (
                db.query(DocumentChunk.id)
                .join(Document, DocumentChunk.document_id == Document.id)
                .filter(DocumentChunk.document_id == document_id)
            )
            if conversation_id is not None:
                query = query.filter(Document.conversation_id == conversation_id)
            if user_id is not None:
                query = query.join(
                    Conversation, Document.conversation_id == Conversation.id
                ).filter(Conversation.owner_id == user_id)
            return (
                query.order_by(DocumentChunk.chunk_index.asc())
                .offset(max(0, int(after_offset)))
                .first()
                is not None
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

    def get_by_ids_for_scope(
        self,
        chunk_ids: Iterable[UUID],
        *,
        user_id: Any | None = None,
        conversation_id: Any | None = None,
    ) -> list[DocumentChunk]:
        """Return chunks by id only when parent documents match server scope."""
        ids = list(chunk_ids)
        if not ids:
            return []

        with self.session_factory() as db:
            query = (
                db.query(DocumentChunk)
                .options(joinedload(DocumentChunk.document))
                .join(Document, DocumentChunk.document_id == Document.id)
                .filter(DocumentChunk.id.in_(ids))
            )
            if conversation_id is not None:
                query = query.filter(Document.conversation_id == conversation_id)
            if user_id is not None:
                query = query.join(
                    Conversation, Document.conversation_id == Conversation.id
                ).filter(Conversation.owner_id == user_id)
            return query.all()

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
