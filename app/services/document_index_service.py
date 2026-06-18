"""Document index service.

The single owner of chunk persistence in PostgreSQL, vector embedding, and
Qdrant upsert. Exposes:

  * ``index_document`` — replace SQL chunks, embed in batches, upsert
    Qdrant points, mark chunks indexed.
  * ``delete_document_index`` — remove Qdrant points and SQL chunks for a
    document.
  * ``reindex_document`` — re-embed existing SQL chunks and sync Qdrant.
  * ``ensure_collection`` — create the configured Qdrant collection at
    the configured vector dimension if absent; validate vector size if
    present. The single owner of collection bootstrap.

All write paths are idempotent by ``document_id``. Qdrant payloads hold
only lookup metadata; canonical text lives in SQL.
"""

from __future__ import annotations

import logging
import time as _time
import uuid
import warnings
from collections.abc import Iterable
from typing import Any
from uuid import UUID

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.models.document_chunk import DocumentChunk
from app.repositories.document_chunk import DocumentChunkRepository
from app.services.document_chunk_builder import BuiltChunk

logger = logging.getLogger(__name__)


class DocumentIndexService:
    def __init__(
        self,
        *,
        chunk_repository: DocumentChunkRepository,
        qdrant_client: Any,
        embedding_service: Any,
        collection_name: str,
        embedding_model_name: str | None = None,
        embedding_dimension: int | None = None,
        embedding_provider: str | None = None,
        index_batch_size: int | None = None,
        qdrant_upsert_batch_size: int = 1000,
    ):
        self.chunk_repository = chunk_repository
        self.qdrant_client = qdrant_client
        self.embedding_service = embedding_service
        self.collection_name = collection_name
        # Prefer values reported by the embedding service when caller didn't
        # supply explicit overrides. Keeping all three as kwargs lets tests
        # pin specific values without poking at the service stub.
        self.embedding_model_name = embedding_model_name or getattr(
            embedding_service, "model_name", "unknown"
        )
        self.embedding_dimension = (
            int(embedding_dimension)
            if embedding_dimension is not None
            else int(getattr(embedding_service, "dimension", 0))
        )
        self.embedding_provider = embedding_provider or getattr(
            embedding_service, "provider", "unknown"
        )
        # index_batch_size is deprecated; batching is now internal to the
        # embedding service. Accept but ignore the parameter.
        if index_batch_size is not None:
            warnings.warn(
                "index_batch_size is deprecated. Batching is now internal to "
                "the embedding service (rag_embedding_batch_size). This parameter is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.qdrant_upsert_batch_size = max(1, int(qdrant_upsert_batch_size))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ensure_collection(self) -> None:
        """Create the configured Qdrant collection if it does not exist.

        Raises ``ValueError`` if the collection exists but its configured
        vector size does not match ``embedding_dimension``. This is the
        only place collections are created.
        """
        try:
            collections = self.qdrant_client.get_collections()
            exists = any(
                getattr(c, "name", None) == self.collection_name
                for c in getattr(collections, "collections", []) or []
            )
        except Exception as exc:
            logger.warning(
                "Could not connect to Qdrant for ensure_collection: %s. Collection check skipped.",
                exc,
            )
            return

        if not exists:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            return

        info = self.qdrant_client.get_collection(self.collection_name)
        try:
            actual_size = int(info.config.params.vectors.size)
        except AttributeError:
            return
        if actual_size != self.embedding_dimension:
            raise ValueError(
                f"Collection '{self.collection_name}' has vector size "
                f"{actual_size}, expected {self.embedding_dimension}"
            )

    def index_document(
        self,
        *,
        document: Any,
        built_chunks: list[BuiltChunk],
        parse_artifact_id: UUID | None,
        timing_sink: dict | None = None,
    ) -> list[DocumentChunk]:
        """Replace chunks for ``document`` with ``built_chunks`` and index them."""
        document_id = self._coerce_uuid(document.id)

        chunk_rows = [self._built_chunk_to_row(bc, parse_artifact_id) for bc in built_chunks]
        persisted = self.chunk_repository.replace_document_chunks(document_id, chunk_rows)

        # Delete any existing Qdrant points for this document (idempotent reindex).
        self._delete_points_for_document(document_id)

        try:
            self._embed_and_upsert(
                document=document,
                persisted_chunks=persisted,
                timing_sink=timing_sink,
            )
        except Exception as exc:
            for chunk in persisted:
                try:
                    self.chunk_repository.mark_index_failed(chunk.id, str(exc))
                except Exception:
                    logger.exception(
                        "Failed to mark chunk %s as failed", getattr(chunk, "id", None)
                    )
            raise

        self.chunk_repository.mark_indexed_bulk(
            [chunk.id for chunk in persisted],
            point_ids=[str(self._point_id_for_chunk(chunk.id)) for chunk in persisted],
            embedding_model=self.embedding_model_name,
            embedding_dimension=self.embedding_dimension,
            collection_name=self.collection_name,
        )

        return persisted

    def delete_document_index(self, document_id: UUID) -> None:
        document_id = self._coerce_uuid(document_id)
        self._delete_points_for_document(document_id)
        self.chunk_repository.delete_by_document(document_id)

    def reindex_document(self, document_id: UUID) -> list[DocumentChunk]:
        document_id = self._coerce_uuid(document_id)
        chunks = self.chunk_repository.get_by_document_ordered(document_id)
        if not chunks:
            return []

        self._delete_points_for_document(document_id)

        # Reindex only sees stored chunk content. Conversation/user payload
        # fields are reconstructed from the chunk's joined Document row when
        # available; otherwise they're omitted (the original index pass set
        # them, and a stale collection should be recreated rather than
        # patched in place).
        self._embed_and_upsert(
            document=None,
            persisted_chunks=chunks,
        )
        self.chunk_repository.mark_indexed_bulk(
            [chunk.id for chunk in chunks],
            point_ids=[str(self._point_id_for_chunk(chunk.id)) for chunk in chunks],
            embedding_model=self.embedding_model_name,
            embedding_dimension=self.embedding_dimension,
            collection_name=self.collection_name,
        )
        return chunks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _built_chunk_to_row(
        self,
        bc: BuiltChunk,
        parse_artifact_id: UUID | None,
    ) -> dict[str, Any]:
        return {
            "id": uuid.uuid4(),
            "parse_artifact_id": parse_artifact_id,
            "chunk_index": bc.chunk_index,
            "content": bc.content,
            "content_sha256": bc.content_sha256,
            "char_count": bc.char_count,
            "token_count": bc.token_count,
            "page_start": bc.page_start,
            "page_end": bc.page_end,
            "section_path": list(bc.section_path),
            "block_provenance": list(bc.block_provenance),
            "chunk_metadata": dict(bc.metadata),
            "index_status": "pending",
        }

    def _embed_and_upsert(
        self,
        *,
        document: Any,
        persisted_chunks: Iterable[DocumentChunk],
        timing_sink: dict | None = None,
    ) -> None:
        persisted = list(persisted_chunks)
        if not persisted:
            return

        title = self._title_for_document(document, persisted)

        # Single embed_documents call for all chunks — batching is internal
        # to the embedding service (rag_embedding_batch_size).
        texts = [chunk.content for chunk in persisted]
        titles = [title] * len(texts)

        embed_t0 = _time.monotonic()
        vectors = self.embedding_service.embed_documents(texts, titles=titles)
        embed_s = _time.monotonic() - embed_t0

        # Build points for all chunks.
        points: list[PointStruct] = []
        for chunk, vector in zip(persisted, vectors, strict=True):
            points.append(
                PointStruct(
                    id=self._point_id_for_chunk(chunk.id),
                    vector=list(vector),
                    payload=self._payload_for_chunk(document, chunk),
                )
            )

        # Upsert to Qdrant in batches by qdrant_upsert_batch_size to avoid
        # overwhelming the server with a single large request.
        upsert_t0 = _time.monotonic()
        for batch in _batched(points, self.qdrant_upsert_batch_size):
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
        upsert_s = _time.monotonic() - upsert_t0

        if timing_sink is not None:
            timing_sink["embed_s"] = embed_s
            timing_sink["upsert_s"] = upsert_s

    @staticmethod
    def _title_for_document(document: Any, chunks: list[DocumentChunk]) -> str | None:
        """Best-effort document title for the Gemini doc-format prompt."""
        if document is not None:
            for attr in ("filename", "title", "name"):
                value = getattr(document, attr, None)
                if value:
                    return str(value)
            return None
        if chunks:
            doc_obj = getattr(chunks[0], "document", None)
            if doc_obj is not None:
                value = getattr(doc_obj, "filename", None)
                if value:
                    return str(value)
        return None

    def _payload_for_chunk(self, document: Any, chunk: DocumentChunk) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "document_id": str(chunk.document_id),
            "chunk_id": str(chunk.id),
            "chunk_index": chunk.chunk_index,
            "content_sha256": chunk.content_sha256,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section_path": list(chunk.section_path or []),
            "embedding_model": self.embedding_model_name,
            "embedding_provider": self.embedding_provider,
            "modality": "text",
        }
        if document is not None:
            if getattr(document, "conversation_id", None) is not None:
                payload["conversation_id"] = str(document.conversation_id)
            if getattr(document, "user_id", None) is not None:
                payload["user_id"] = str(document.user_id)
        return payload

    def _delete_points_for_document(self, document_id: UUID) -> None:
        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=str(document_id)),
                            )
                        ]
                    )
                ),
            )
        except Exception:
            logger.exception("Qdrant delete-by-filter failed for document_id=%s", document_id)
            raise

    @staticmethod
    def _point_id_for_chunk(chunk_id: UUID) -> str:
        return str(chunk_id)

    @staticmethod
    def _coerce_uuid(value: Any) -> UUID:
        if isinstance(value, UUID):
            return value
        return UUID(str(value))


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]
