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
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
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
from app.repositories.document_index_generation import DocumentIndexGenerationRepository
from app.services.document_chunk_builder import BuiltChunk
from app.usage.types import UsageContext

logger = logging.getLogger(__name__)


class DocumentIndexService:
    def __init__(
        self,
        *,
        chunk_repository: DocumentChunkRepository,
        generation_repository: DocumentIndexGenerationRepository,
        qdrant_client: Any,
        embedding_service: Any,
        collection_name: str,
        embedding_model_name: str | None = None,
        embedding_dimension: int | None = None,
        embedding_provider: str | None = None,
        chunking_version: str = "structure-v2",
        qdrant_upsert_batch_size: int = 1000,
    ):
        self.chunk_repository = chunk_repository
        self.generation_repository = generation_repository
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
        self.chunking_version = chunking_version
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
        usage_context: UsageContext | None = None,
        activate: bool = True,
    ) -> list[DocumentChunk]:
        """Build and verify a replacement generation before atomically activating it."""
        document_id = self._coerce_uuid(document.id)
        old_generation = self.generation_repository.get_active(document_id)
        generation = self.generation_repository.create(
            document_id=document_id,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model_name,
            embedding_dimension=self.embedding_dimension,
            chunking_version=self.chunking_version,
        )
        chunk_rows = [self._built_chunk_to_row(bc, parse_artifact_id) for bc in built_chunks]
        persisted: list[DocumentChunk] = []

        try:
            persisted = self.chunk_repository.create_generation_chunks(
                document_id, generation.id, chunk_rows
            )
            self._embed_and_upsert(
                document=document,
                persisted_chunks=persisted,
                index_generation_id=generation.id,
                timing_sink=timing_sink,
                usage_context=usage_context,
            )
            self.chunk_repository.mark_indexed_bulk(
                [chunk.id for chunk in persisted],
                point_ids=[str(self._point_id_for_chunk(chunk.id)) for chunk in persisted],
                embedding_model=self.embedding_model_name,
                embedding_dimension=self.embedding_dimension,
                collection_name=self.collection_name,
            )
            self._verify_generation(document_id, generation.id, persisted)
            self.generation_repository.mark_ready(generation.id)
            if activate:
                self._set_generation_active(document_id, generation.id, True)
                try:
                    self.generation_repository.activate(generation.id)
                except Exception:
                    try:
                        self._set_generation_active(document_id, generation.id, False)
                    except Exception:
                        logger.exception(
                            "Failed to restore generation %s Qdrant payloads", generation.id
                        )
                    raise
        except Exception as exc:
            for chunk in persisted:
                try:
                    self.chunk_repository.mark_index_failed(chunk.id, str(exc))
                except Exception:
                    logger.exception(
                        "Failed to mark chunk %s as failed", getattr(chunk, "id", None)
                    )
            try:
                self.generation_repository.mark_failed(
                    generation.id, self._failure_code(exc)
                )
            except Exception:
                logger.exception("Failed to mark generation %s failed", generation.id)
            raise

        if activate and old_generation is not None and old_generation.id != generation.id:
            try:
                self._set_generation_active(document_id, old_generation.id, False)
            except Exception:
                # SQL is authoritative. Active-only hydration prevents retired
                # rows from leaking while reconciliation repairs Qdrant flags.
                logger.exception(
                    "Retired generation payload cleanup deferred for document_id=%s",
                    document_id,
                )

        return persisted

    def delete_document_index(self, document_id: UUID) -> None:
        document_id = self._coerce_uuid(document_id)
        self._delete_points_for_document(document_id)
        self.chunk_repository.delete_by_document(document_id)

    def reindex_document(
        self, document_id: UUID, *, activate: bool = True
    ) -> list[DocumentChunk]:
        document_id = self._coerce_uuid(document_id)
        chunks = self.chunk_repository.get_by_document_ordered(document_id)
        if not chunks:
            return []
        document = chunks[0].document
        built = [
            BuiltChunk(
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                content_sha256=chunk.content_sha256,
                char_count=chunk.char_count,
                token_count=chunk.token_count,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=tuple(chunk.section_path or ()),
                block_provenance=tuple(chunk.block_provenance or ()),
                metadata=dict(chunk.chunk_metadata or {}),
            )
            for chunk in chunks
        ]
        return self.index_document(
            document=document,
            built_chunks=built,
            parse_artifact_id=chunks[0].parse_artifact_id,
            activate=activate,
        )

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
        index_generation_id: UUID,
        timing_sink: dict | None = None,
        usage_context: UsageContext | None = None,
    ) -> None:
        persisted = list(persisted_chunks)
        if not persisted:
            return

        title = self._title_for_document(document, persisted)

        # Single embed_documents call for all chunks — batching is internal
        # to the embedding service (rag_embedding_batch_size). usage_context is
        # passed explicitly because the embedding batches run in a
        # ThreadPoolExecutor that does not inherit the bound request context.
        texts = [chunk.content for chunk in persisted]
        titles = [title] * len(texts)

        embed_t0 = _time.monotonic()
        vectors = self.embedding_service.embed_documents(
            texts, titles=titles, usage_context=usage_context
        )
        embed_s = _time.monotonic() - embed_t0

        # Build points for all chunks.
        points: list[PointStruct] = []
        for chunk, vector in zip(persisted, vectors, strict=True):
            points.append(
                PointStruct(
                    id=self._point_id_for_chunk(chunk.id),
                    vector=list(vector),
                    payload=self._payload_for_chunk(document, chunk, index_generation_id),
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

    def _payload_for_chunk(
        self, document: Any, chunk: DocumentChunk, index_generation_id: UUID
    ) -> dict[str, Any]:
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
            "index_generation": str(index_generation_id),
            "is_active": False,
        }
        if document is not None:
            if getattr(document, "conversation_id", None) is not None:
                payload["conversation_id"] = str(document.conversation_id)
            if getattr(document, "user_id", None) is not None:
                payload["user_id"] = str(document.user_id)
        return payload

    def _verify_generation(
        self,
        document_id: UUID,
        generation_id: UUID,
        chunks: list[DocumentChunk],
    ) -> None:
        generation_filter = self._generation_filter(document_id, generation_id)
        counted = self.qdrant_client.count(
            collection_name=self.collection_name,
            count_filter=generation_filter,
            exact=True,
        )
        actual_count = int(getattr(counted, "count", -1))
        if actual_count != len(chunks):
            raise ValueError(
                f"generation point count mismatch: expected {len(chunks)}, got {actual_count}"
            )

        point_ids = [str(self._point_id_for_chunk(chunk.id)) for chunk in chunks]
        points = self.qdrant_client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=True,
        )
        if len(points) != len(chunks):
            raise ValueError(
                f"generation point retrieval mismatch: expected {len(chunks)}, got {len(points)}"
            )
        expected_document = str(document_id)
        expected_generation = str(generation_id)
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            if payload.get("document_id") != expected_document:
                raise ValueError("generation point document scope mismatch")
            if payload.get("index_generation") != expected_generation:
                raise ValueError("generation point ownership mismatch")
            vector = getattr(point, "vector", None)
            if not isinstance(vector, list) or len(vector) != self.embedding_dimension:
                raise ValueError("generation point vector dimension mismatch")

    def _set_generation_active(
        self, document_id: UUID, generation_id: UUID, is_active: bool
    ) -> None:
        self.qdrant_client.set_payload(
            collection_name=self.collection_name,
            payload={"is_active": is_active},
            points=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=str(document_id)),
                        ),
                        FieldCondition(
                            key="index_generation",
                            match=MatchValue(value=str(generation_id)),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def reconcile_active_payloads(self, document_id: UUID) -> UUID | None:
        """Make Qdrant activity flags match the authoritative SQL generation."""
        document_id = self._coerce_uuid(document_id)
        active = self.generation_repository.get_active(document_id)
        document_filter = FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=str(document_id))
                    )
                ]
            )
        )
        self.qdrant_client.set_payload(
            collection_name=self.collection_name,
            payload={"is_active": False},
            points=document_filter,
            wait=True,
        )
        if active is not None:
            self._set_generation_active(document_id, active.id, True)
            return active.id
        return None

    def purge_retired_generations(
        self, document_id: UUID, older_than: datetime
    ) -> list[UUID]:
        """Delete retired SQL/Qdrant generations after a caller-chosen rollback window."""
        document_id = self._coerce_uuid(document_id)
        purged: list[UUID] = []
        for generation in self.generation_repository.retired_before(document_id, older_than):
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(
                    filter=self._generation_filter(document_id, generation.id)
                ),
                wait=True,
            )
            self.chunk_repository.delete_generation(generation.id)
            self.generation_repository.delete(generation.id)
            purged.append(generation.id)
        return purged

    def purge_retired_after_hours(self, document_id: UUID, hours: int) -> list[UUID]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max(0, int(hours)))
        return self.purge_retired_generations(document_id, cutoff)

    @staticmethod
    def _failure_code(exc: Exception) -> str:
        name = type(exc).__name__.upper()
        return f"INDEX_BUILD_{name}"[:64]

    @staticmethod
    def _generation_filter(document_id: UUID, generation_id: UUID) -> Filter:
        return Filter(
            must=[
                FieldCondition(
                    key="document_id", match=MatchValue(value=str(document_id))
                ),
                FieldCondition(
                    key="index_generation", match=MatchValue(value=str(generation_id))
                ),
            ]
        )

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
