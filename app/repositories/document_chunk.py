"""Repository for the canonical document_chunks content store.

Follows the sync session-factory style used by the rest of the server's
data layer.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, literal, literal_column
from sqlalchemy.orm import joinedload

from app.models.conversation import Conversation
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_index_generation import DocumentIndexGeneration


def postgres_lexical_expressions(query_text: str):
    """Build the indexed PostgreSQL simple-language match and raw rank."""
    language = literal_column("'simple'")
    vector = func.to_tsvector(language, DocumentChunk.content)
    tsquery = func.plainto_tsquery(language, query_text)
    return vector.op("@@")(tsquery), func.ts_rank_cd(vector, tsquery).label("lexical_score")


class DocumentChunkRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    def create_generation_chunks(
        self,
        document_id: UUID,
        index_generation_id: UUID,
        chunk_rows: list[dict[str, Any]],
    ) -> list[DocumentChunk]:
        """Create an inactive generation's chunks without touching active rows.

        ``chunk_rows`` is a list of kwargs dicts; each one is used to construct
        a ``DocumentChunk`` row. Callers are responsible for populating
        ``content``, ``content_sha256``, ``char_count``, ``token_count`` and
        any deterministic ``id`` / ``qdrant_point_id`` they want.
        """
        with self.session_factory() as db:
            created: list[DocumentChunk] = []
            for row in chunk_rows:
                chunk = DocumentChunk(
                    document_id=document_id,
                    index_generation_id=index_generation_id,
                    **row,
                )
                db.add(chunk)
                created.append(chunk)

            db.commit()
            for chunk in created:
                db.refresh(chunk)
            return created

    def delete_generation(self, index_generation_id: UUID) -> int:
        with self.session_factory() as db:
            deleted = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.index_generation_id == index_generation_id)
                .delete(synchronize_session=False)
            )
            db.commit()
            return deleted

    def get_by_generation_ordered(self, index_generation_id: UUID) -> list[DocumentChunk]:
        with self.session_factory() as db:
            return (
                db.query(DocumentChunk)
                .options(joinedload(DocumentChunk.document))
                .filter(DocumentChunk.index_generation_id == index_generation_id)
                .order_by(DocumentChunk.chunk_index.asc())
                .all()
            )

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
    @staticmethod
    def _active(query):
        return query.join(
            DocumentIndexGeneration,
            DocumentChunk.index_generation_id == DocumentIndexGeneration.id,
        ).filter(DocumentIndexGeneration.status == "active")

    def get_by_document_ordered(self, document_id: UUID) -> list[DocumentChunk]:
        with self.session_factory() as db:
            query = (
                db.query(DocumentChunk)
                .options(joinedload(DocumentChunk.document))
                .filter(DocumentChunk.document_id == document_id)
            )
            return self._active(query).order_by(DocumentChunk.chunk_index.asc()).all()

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
        if not user_id or not conversation_id:
            return []

        with self.session_factory() as db:
            query = (
                db.query(DocumentChunk)
                .options(joinedload(DocumentChunk.document))
                .join(Document, DocumentChunk.document_id == Document.id)
                .filter(DocumentChunk.document_id == document_id)
            )
            query = query.filter(Document.conversation_id == conversation_id)
            query = query.join(Conversation, Document.conversation_id == Conversation.id).filter(
                Conversation.owner_id == user_id
            )
            return self._active(query).order_by(DocumentChunk.chunk_index.asc()).all()

    def get_window_for_scope(
        self,
        document_id: UUID,
        user_id: Any | None,
        conversation_id: Any | None,
        start_chunk: int,
        max_chunks: int,
    ) -> list[DocumentChunk]:
        """Return one bounded, ownership-filtered window of document chunks."""
        if not user_id or not conversation_id:
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
            query = query.filter(Document.conversation_id == conversation_id)
            query = query.join(Conversation, Document.conversation_id == Conversation.id).filter(
                Conversation.owner_id == user_id
            )
            return (
                self._active(query)
                .order_by(DocumentChunk.chunk_index.asc())
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
        if not user_id or not conversation_id:
            return False

        with self.session_factory() as db:
            query = (
                db.query(DocumentChunk.id)
                .join(Document, DocumentChunk.document_id == Document.id)
                .filter(DocumentChunk.document_id == document_id)
            )
            query = query.filter(Document.conversation_id == conversation_id)
            query = query.join(Conversation, Document.conversation_id == Conversation.id).filter(
                Conversation.owner_id == user_id
            )
            return (
                self._active(query)
                .order_by(DocumentChunk.chunk_index.asc())
                .offset(max(0, int(after_offset)))
                .first()
                is not None
            )

    def get_by_ids(self, chunk_ids: Iterable[UUID]) -> list[DocumentChunk]:
        ids = list(chunk_ids)
        if not ids:
            return []
        with self.session_factory() as db:
            query = (
                db.query(DocumentChunk)
                .options(joinedload(DocumentChunk.document))
                .filter(DocumentChunk.id.in_(ids))
            )
            return self._active(query).all()

    def get_by_ids_for_scope(
        self,
        chunk_ids: Iterable[UUID],
        *,
        user_id: Any | None = None,
        conversation_id: Any | None = None,
    ) -> list[DocumentChunk]:
        """Return chunks by id only when parent documents match server scope."""
        ids = list(chunk_ids)
        if not ids or not user_id or not conversation_id:
            return []

        with self.session_factory() as db:
            query = (
                db.query(DocumentChunk)
                .options(joinedload(DocumentChunk.document))
                .join(Document, DocumentChunk.document_id == Document.id)
                .filter(DocumentChunk.id.in_(ids))
            )
            query = query.filter(Document.conversation_id == conversation_id)
            query = query.join(Conversation, Document.conversation_id == Conversation.id).filter(
                Conversation.owner_id == user_id
            )
            return self._active(query).all()

    def get_active_by_ids_for_scope(
        self,
        chunk_ids: Iterable[UUID],
        *,
        user_id: Any | None = None,
        conversation_id: Any | None = None,
    ) -> list[DocumentChunk]:
        """Authorize candidate IDs against active SQL generations and scope."""
        return self.get_by_ids_for_scope(
            chunk_ids,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    def get_context_expansion_for_scope(
        self,
        chunk_id: UUID,
        *,
        document_id: UUID,
        user_id: Any | None = None,
        conversation_id: Any | None = None,
        max_neighbors: int = 2,
    ) -> list[DocumentChunk]:
        """Return bounded parent/adjacent rows from the seed's active generation."""
        if not user_id or not conversation_id or not chunk_id or not document_id:
            return []
        bounded_limit = min(8, max(0, int(max_neighbors)))
        if bounded_limit == 0:
            return []

        with self.session_factory() as db:
            seed_query = (
                db.query(DocumentChunk)
                .join(Document, DocumentChunk.document_id == Document.id)
                .join(Conversation, Document.conversation_id == Conversation.id)
                .join(
                    DocumentIndexGeneration,
                    DocumentChunk.index_generation_id == DocumentIndexGeneration.id,
                )
                .filter(DocumentChunk.id == chunk_id)
                .filter(DocumentChunk.document_id == document_id)
                .filter(Document.conversation_id == conversation_id)
                .filter(Conversation.owner_id == user_id)
                .filter(DocumentIndexGeneration.status == "active")
            )
            seed = seed_query.first()
            if seed is None:
                return []

            metadata = dict(seed.chunk_metadata or {})
            parent_id = None
            try:
                raw_parent_id = metadata.get("parent_chunk_id")
                if raw_parent_id:
                    parent_id = UUID(str(raw_parent_id))
            except (TypeError, ValueError, AttributeError):
                parent_id = None

            query = (
                db.query(DocumentChunk)
                .options(joinedload(DocumentChunk.document))
                .filter(DocumentChunk.document_id == seed.document_id)
                .filter(DocumentChunk.index_generation_id == seed.index_generation_id)
                .filter(DocumentChunk.id != seed.id)
            )
            adjacent_min = int(seed.chunk_index) - bounded_limit
            adjacent_max = int(seed.chunk_index) + bounded_limit
            adjacency = DocumentChunk.chunk_index.between(adjacent_min, adjacent_max)
            query = query.filter(
                adjacency if parent_id is None else (adjacency | (DocumentChunk.id == parent_id))
            )
            rows = query.all()
            return sorted(
                rows,
                key=lambda row: (
                    0 if parent_id is not None and row.id == parent_id else 1,
                    abs(int(row.chunk_index) - int(seed.chunk_index)),
                    int(row.chunk_index),
                    str(row.id),
                ),
            )[:bounded_limit]

    def get_active_generation_ids_for_scope(
        self,
        *,
        user_id: Any | None = None,
        conversation_id: Any | None = None,
    ) -> list[UUID]:
        if not user_id or not conversation_id:
            return []
        with self.session_factory() as db:
            rows = (
                db.query(DocumentIndexGeneration.id)
                .join(Document, DocumentIndexGeneration.document_id == Document.id)
                .join(Conversation, Document.conversation_id == Conversation.id)
                .filter(Document.conversation_id == conversation_id)
                .filter(Conversation.owner_id == user_id)
                .filter(DocumentIndexGeneration.status == "active")
                .order_by(DocumentIndexGeneration.id.asc())
                .all()
            )
            return [row.id for row in rows]

    def search_active_lexical_for_scope(
        self,
        query_text: str,
        *,
        user_id: Any | None = None,
        conversation_id: Any | None = None,
        limit: int = 40,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return deterministic lexical candidates from active authorized rows."""
        query_text = str(query_text or "").strip()
        if not query_text or not user_id or not conversation_id:
            return []
        bounded_limit = max(1, int(limit))
        with self.session_factory() as db:
            base = (
                db.query(DocumentChunk)
                .options(joinedload(DocumentChunk.document))
                .join(Document, DocumentChunk.document_id == Document.id)
                .join(Conversation, Document.conversation_id == Conversation.id)
                .join(
                    DocumentIndexGeneration,
                    DocumentChunk.index_generation_id == DocumentIndexGeneration.id,
                )
                .filter(Document.conversation_id == conversation_id)
                .filter(Conversation.owner_id == user_id)
                .filter(DocumentIndexGeneration.status == "active")
            )
            if db.get_bind().dialect.name == "postgresql":
                match, score = postgres_lexical_expressions(query_text)
                rows = (
                    base.add_columns(score)
                    .filter(match)
                    .order_by(score.desc(), DocumentChunk.id.asc())
                    .limit(bounded_limit)
                    .all()
                )
            else:
                terms = [term.casefold() for term in query_text.split() if term]
                if not terms:
                    return []
                score = literal(1.0).label("lexical_score")
                rows = (
                    base.add_columns(score)
                    .filter(
                        and_(*(func.lower(DocumentChunk.content).contains(term) for term in terms))
                    )
                    .order_by(DocumentChunk.id.asc())
                    .limit(bounded_limit)
                    .all()
                )
            return [(chunk, float(raw_score)) for chunk, raw_score in rows]

    def get_by_qdrant_point_ids(self, point_ids: Iterable[str]) -> list[DocumentChunk]:
        ids = [p for p in point_ids if p]
        if not ids:
            return []
        with self.session_factory() as db:
            query = (
                db.query(DocumentChunk)
                .options(joinedload(DocumentChunk.document))
                .filter(DocumentChunk.qdrant_point_id.in_(ids))
            )
            return self._active(query).all()
