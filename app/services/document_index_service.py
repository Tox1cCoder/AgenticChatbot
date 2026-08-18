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
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    KeywordIndexParams,
    KeywordIndexType,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.models.document_chunk import DocumentChunk
from app.repositories.document_chunk import DocumentChunkRepository
from app.repositories.document_index_generation import DocumentIndexGenerationRepository
from app.schemas.document_image import DocumentImageUpdate
from app.services.document_chunk_builder import BuiltChunk
from app.services.rag_cache import NullRAGExactCache, RAGExactCache, document_embedding_key
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
        document_image_repository: Any | None = None,
        multimodal_image_embeddings_enabled: bool = False,
        cache: RAGExactCache | None = None,
        metrics: Any | None = None,
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
        # Task 11: native multimodal image points. Both stay optional/off by
        # default so existing text-only callers are unaffected.
        self.document_image_repository = document_image_repository
        self.multimodal_image_embeddings_enabled = bool(multimodal_image_embeddings_enabled)
        self._payload_indexes_ready = False
        # Task 12: exact content-addressed embedding cache, tenant-scoped by
        # document.user_id. Defaults to a no-op so a caller that never wires a
        # cache is byte-for-byte identical to the pre-Task-12 code path.
        self.cache = cache if cache is not None else NullRAGExactCache()
        self.metrics = metrics

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
            self._ensure_payload_indexes()
            return

        info = self.qdrant_client.get_collection(self.collection_name)
        try:
            actual_size = int(info.config.params.vectors.size)
        except AttributeError:
            self._ensure_payload_indexes()
            return
        if actual_size != self.embedding_dimension:
            raise ValueError(
                f"Collection '{self.collection_name}' has vector size "
                f"{actual_size}, expected {self.embedding_dimension}"
            )
        self._ensure_payload_indexes()

    def index_document(
        self,
        *,
        document: Any,
        built_chunks: list[BuiltChunk],
        parse_artifact_id: UUID | None,
        image_rows: Sequence[Any] = (),
        timing_sink: dict | None = None,
        usage_context: UsageContext | None = None,
        activate: bool = True,
    ) -> list[DocumentChunk]:
        """Build and verify a replacement generation before atomically activating it.

        ``image_rows`` are already-persisted ``DocumentImage`` rows (nullable
        ``chunk_id``) supplied by the caller. When
        ``multimodal_image_embeddings_enabled`` is true, each row is embedded
        and upserted as a separate ``modality="image"`` Qdrant point *before*
        verification; both text and image point counts are verified before
        ``mark_ready``. Only once the generation is verified does this method
        link each row's ``chunk_id`` to the page-matching chunk — never
        earlier, so a build that fails partway never repoints another
        generation's images.
        """
        if not built_chunks:
            raise ValueError("document indexing requires at least one chunk")
        self.ensure_collection()
        if not self._payload_indexes_ready:
            raise RuntimeError(
                "Qdrant payload indexes are unavailable; refusing document upsert"
            )
        document_id = self._coerce_uuid(document.id)
        generation = self.generation_repository.create(
            document_id=document_id,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model_name,
            embedding_dimension=self.embedding_dimension,
            chunking_version=self.chunking_version,
        )
        chunk_rows = [self._built_chunk_to_row(bc, parse_artifact_id) for bc in built_chunks]
        persisted: list[DocumentChunk] = []
        image_points: list[PointStruct] = []
        activation_outcome_unknown = False

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
            image_points = self._embed_and_upsert_images(
                document=document,
                image_rows=image_rows,
                index_generation_id=generation.id,
                usage_context=usage_context,
            )
            self._verify_generation(document_id, generation.id, persisted)
            self._verify_image_points(document_id, generation.id, image_points)
            self.generation_repository.mark_ready(generation.id)
            # Link only after verification succeeds (finding 3): a failure
            # above must never repoint another generation's images.
            self._link_image_chunks(persisted_chunks=persisted, image_rows=image_rows)
            if activate:
                self._set_generation_active(document_id, generation.id, True)
                try:
                    self.generation_repository.activate(generation.id)
                except Exception as activation_error:
                    confirmed = self._confirm_active_generation(
                        document_id, generation.id
                    )
                    if confirmed is True:
                        logger.warning(
                            "Generation %s activation commit was confirmed after error: %s",
                            generation.id,
                            activation_error,
                        )
                    elif confirmed is False:
                        try:
                            self._set_generation_active(
                                document_id, generation.id, False
                            )
                        except Exception:
                            logger.exception(
                                "Failed to restore generation %s Qdrant payloads",
                                generation.id,
                            )
                        raise
                    else:
                        # The transaction outcome is unknown. Do not hide or
                        # mark failed a generation that may be SQL-active.
                        activation_outcome_unknown = True
                        raise RuntimeError(
                            "index generation activation outcome is unknown; "
                            "run payload reconciliation"
                        ) from activation_error
        except Exception as exc:
            if not activation_outcome_unknown:
                for chunk in persisted:
                    try:
                        self.chunk_repository.mark_index_failed(chunk.id, str(exc))
                    except Exception:
                        logger.exception(
                            "Failed to mark chunk %s as failed",
                            getattr(chunk, "id", None),
                        )
                try:
                    self.generation_repository.mark_failed(
                        generation.id, self._failure_code(exc)
                    )
                except Exception:
                    logger.exception("Failed to mark generation %s failed", generation.id)
            raise

        if activate:
            try:
                self.reconcile_active_payloads(document_id)
            except Exception:
                # SQL is authoritative. Active-only hydration prevents retired
                # rows from leaking while reconciliation repairs Qdrant flags.
                logger.exception(
                    "Retired generation payload cleanup deferred for document_id=%s",
                    document_id,
                )

        return persisted

    def _ensure_payload_indexes(self) -> None:
        if self._payload_indexes_ready:
            return
        tenant_schema = KeywordIndexParams(type=KeywordIndexType.KEYWORD, is_tenant=True)
        try:
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema=tenant_schema,
                wait=True,
            )
        except UnexpectedResponse as exc:
            if not self._tenant_index_is_unsupported(exc):
                raise
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )
        for field_name in (
            "conversation_id",
            "document_id",
            "modality",
            "index_generation",
        ):
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )
        self.qdrant_client.create_payload_index(
            collection_name=self.collection_name,
            field_name="is_active",
            field_schema=PayloadSchemaType.BOOL,
            wait=True,
        )
        self._payload_indexes_ready = True

    @staticmethod
    def _tenant_index_is_unsupported(exc: UnexpectedResponse) -> bool:
        content = getattr(exc, "content", b"")
        message = content.decode("utf-8", errors="replace").casefold()
        return getattr(exc, "status_code", None) in {400, 422} and (
            "is_tenant" in message
            and any(token in message for token in ("unknown", "unsupported", "extra"))
        )

    def _confirm_active_generation(
        self, document_id: UUID, generation_id: UUID
    ) -> bool | None:
        try:
            active = self.generation_repository.get_active(document_id)
        except Exception:
            logger.exception(
                "Could not confirm activation state for generation %s", generation_id
            )
            return None
        return active is not None and active.id == generation_id

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

        # Single embed_documents call for all *uncached* chunks — batching is
        # internal to the embedding service (rag_embedding_batch_size).
        # usage_context is passed explicitly because the embedding batches run
        # in a ThreadPoolExecutor that does not inherit the bound request
        # context.
        texts = [chunk.content for chunk in persisted]
        titles = [title] * len(texts)

        embed_t0 = _time.monotonic()
        vectors = self._embed_documents_cached(
            document=document,
            persisted=persisted,
            texts=texts,
            titles=titles,
            usage_context=usage_context,
        )
        embed_s = _time.monotonic() - embed_t0
        self._record_stage("embedding", embed_s, modality="text")

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

    def _embed_documents_cached(
        self,
        *,
        document: Any,
        persisted: list[DocumentChunk],
        texts: list[str],
        titles: list[str | None],
        usage_context: UsageContext | None,
    ) -> list[list[float]]:
        """Embed only the chunks whose content hash misses the exact cache.

        Keyed by content SHA-256 under the current tenant/provider/model/
        dimension/format-version -- never by chunk id, so two chunks with
        identical text (even across documents) share one cache entry. With
        the default no-op cache every lookup misses, so every chunk is
        embedded exactly as before Task 12.
        """
        tenant = str(getattr(document, "user_id", "") or "")
        format_version = str(
            getattr(self.embedding_service, "document_format_version", "doc-fmt-v1")
        )
        keys = [
            document_embedding_key(
                tenant=tenant,
                provider=self.embedding_provider,
                model=self.embedding_model_name,
                dimension=self.embedding_dimension,
                format_version=format_version,
                content_sha256=str(getattr(chunk, "content_sha256", "") or ""),
            )
            for chunk in persisted
        ]

        vectors: list[list[float] | None] = [None] * len(persisted)
        missing_indices: list[int] = []
        for index, key in enumerate(keys):
            cached = self.cache.get_document_embedding(key, dimension=self.embedding_dimension)
            self._record_cache_result("document_embedding", hit=cached is not None)
            if cached is not None:
                vectors[index] = cached
            else:
                missing_indices.append(index)

        if missing_indices:
            fresh = self.embedding_service.embed_documents(
                [texts[index] for index in missing_indices],
                titles=[titles[index] for index in missing_indices],
                usage_context=usage_context,
            )
            for index, vector in zip(missing_indices, fresh, strict=True):
                resolved = list(vector)
                vectors[index] = resolved
                self.cache.set_document_embedding(keys[index], resolved)

        return [vector for vector in vectors if vector is not None]

    def _record_stage(self, stage: str, elapsed_seconds: float, **labels: Any) -> None:
        recorder = getattr(self.metrics, "stage", None)
        if not callable(recorder):
            return
        try:
            recorder(stage, elapsed_seconds=elapsed_seconds, labels=labels)
        except Exception:
            logger.exception("Failed to record RAG stage metric for %s", stage)

    def _record_cache_result(self, cache_name: str, *, hit: bool) -> None:
        recorder = getattr(self.metrics, "cache_result", None)
        if not callable(recorder):
            return
        result = "hit" if hit else ("miss" if getattr(self.cache, "enabled", False) else "disabled")
        try:
            recorder(cache_name, result)
        except Exception:
            logger.exception("Failed to record RAG cache metric for %s", cache_name)

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
        generation_filter = self._generation_filter(document_id, generation_id, modality="text")
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

    # ------------------------------------------------------------------
    # Task 11: image linking + native multimodal embeddings
    # ------------------------------------------------------------------
    def _embed_and_upsert_images(
        self,
        *,
        document: Any,
        image_rows: Sequence[Any],
        index_generation_id: UUID,
        usage_context: UsageContext | None,
    ) -> list[PointStruct]:
        """Embed and upsert native ``modality="image"`` points, if enabled.

        Only runs when ``multimodal_image_embeddings_enabled`` is true;
        caption-first retrieval never depends on it. chunk_id linking is a
        separate step (``_link_image_chunks``) run only after this
        generation is verified, so a build that fails here never touches
        another generation's image rows (review finding 3).

        An unreadable image file is skipped (logged by id only, never its
        path or bytes) rather than aborting the whole generation, matching
        how ``RAGImageSelector`` tolerates the same condition (finding 7).
        """
        image_rows = list(image_rows)
        if not image_rows:
            return []
        if self.document_image_repository is None:
            raise ValueError(
                "document_image_repository is required to index image_rows"
            )
        if not self.multimodal_image_embeddings_enabled:
            return []

        points: list[PointStruct] = []
        for image in image_rows:
            image_bytes = self._read_image_bytes(image)
            if image_bytes is None:
                continue
            mime_type = str(getattr(image, "mime_type", "") or "application/octet-stream")
            vector = self.embedding_service.embed_image(
                image_bytes,
                mime_type=mime_type,
                usage_context=usage_context,
            )
            points.append(
                self._point_for_image(
                    document,
                    image,
                    list(vector),
                    index_generation_id=index_generation_id,
                )
            )

        for batch in _batched(points, self.qdrant_upsert_batch_size):
            self.qdrant_client.upsert(collection_name=self.collection_name, points=batch)
        return points

    def _link_image_chunks(
        self,
        *,
        persisted_chunks: list[DocumentChunk],
        image_rows: Sequence[Any],
    ) -> None:
        """Link each image row's chunk_id by page against this generation.

        Called only after the generation has been verified and marked
        ready. Linking earlier (before we know the build will succeed) let a
        later failure leave an older, still-active generation's images
        pointing at a chunk that belongs to the new, doomed generation —
        purging that failed generation then nulled the link via
        ``ondelete="SET NULL"`` (review finding 3).
        """
        image_rows = list(image_rows)
        if not image_rows:
            return
        if self.document_image_repository is None:
            raise ValueError(
                "document_image_repository is required to index image_rows"
            )
        for image in image_rows:
            matched_chunk_id = self._chunk_id_for_image_page(
                getattr(image, "page_number", None), persisted_chunks
            )
            if matched_chunk_id is not None and matched_chunk_id != getattr(
                image, "chunk_id", None
            ):
                self.document_image_repository.update(
                    image.id, DocumentImageUpdate(chunk_id=matched_chunk_id)
                )

    def _read_image_bytes(self, image: Any) -> bytes | None:
        try:
            return self._resolve_image_path(image).read_bytes()
        except OSError:
            logger.warning(
                "Skipping unreadable image file: image_id=%s", getattr(image, "id", None)
            )
            return None

    def _verify_image_points(
        self,
        document_id: UUID,
        generation_id: UUID,
        image_points: list[PointStruct],
    ) -> None:
        if not image_points:
            return
        generation_filter = self._generation_filter(document_id, generation_id, modality="image")
        counted = self.qdrant_client.count(
            collection_name=self.collection_name,
            count_filter=generation_filter,
            exact=True,
        )
        actual_count = int(getattr(counted, "count", -1))
        if actual_count != len(image_points):
            raise ValueError(
                "image generation point count mismatch: "
                f"expected {len(image_points)}, got {actual_count}"
            )

        point_ids = [str(point.id) for point in image_points]
        points = self.qdrant_client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=True,
        )
        if len(points) != len(image_points):
            raise ValueError(
                "image generation point retrieval mismatch: "
                f"expected {len(image_points)}, got {len(points)}"
            )
        expected_document = str(document_id)
        expected_generation = str(generation_id)
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            if payload.get("document_id") != expected_document:
                raise ValueError("image generation point document scope mismatch")
            if payload.get("index_generation") != expected_generation:
                raise ValueError("image generation point ownership mismatch")

    def _point_for_image(
        self,
        document: Any,
        image: Any,
        vector: list[float] | None = None,
        *,
        index_generation_id: UUID | None = None,
    ) -> PointStruct:
        payload: dict[str, Any] = {
            "document_id": str(getattr(image, "document_id", getattr(document, "id", ""))),
            "image_id": str(image.id),
            "page_number": getattr(image, "page_number", None),
            "content_sha256": getattr(image, "content_sha256", None),
            "section_path": list(getattr(image, "section_path", None) or []),
            "embedding_model": self.embedding_model_name,
            "embedding_provider": self.embedding_provider,
            "modality": "image",
            "index_generation": (
                str(index_generation_id) if index_generation_id is not None else None
            ),
            "is_active": False,
        }
        if document is not None:
            if getattr(document, "conversation_id", None) is not None:
                payload["conversation_id"] = str(document.conversation_id)
            if getattr(document, "user_id", None) is not None:
                payload["user_id"] = str(document.user_id)
        return PointStruct(
            id=self._point_id_for_image(image.id, index_generation_id),
            vector=list(vector) if vector else [],
            payload=payload,
        )

    @staticmethod
    def _resolve_image_path(image: Any) -> Path:
        path = Path(str(image.image_path))
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    @staticmethod
    def _chunk_id_for_image_page(
        page_number: int | None, persisted_chunks: list[DocumentChunk]
    ) -> UUID | None:
        if not persisted_chunks:
            return None
        if page_number is None:
            return persisted_chunks[0].id
        for chunk in persisted_chunks:
            page_start = getattr(chunk, "page_start", None)
            page_end = getattr(chunk, "page_end", None)
            if page_start is None and page_end is None:
                continue
            start = page_start if page_start is not None else page_end
            end = page_end if page_end is not None else page_start
            if start <= page_number <= end:
                return chunk.id
        return persisted_chunks[0].id

    @staticmethod
    def _point_id_for_image(image_id: UUID, index_generation_id: UUID | None) -> str:
        """Derive a point id scoped to (image, generation).

        ``DocumentImage`` rows survive reindexing — unlike chunks, which get
        fresh rows (and therefore fresh point ids) every generation — so
        reusing ``str(image_id)`` directly would let a new generation's
        upsert overwrite the still-active generation's point in place. A
        later failure + purge of the new generation would then delete the
        point the active generation depends on (review finding 2).
        """
        namespace = image_id if isinstance(image_id, UUID) else UUID(str(image_id))
        generation_key = str(index_generation_id) if index_generation_id is not None else "none"
        return str(uuid.uuid5(namespace, generation_key))

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
        for _attempt in range(8):
            active = self.generation_repository.get_active(document_id)
            self._apply_payload_reconciliation(document_id, active)
            confirmed = self.generation_repository.get_active(document_id)
            active_id = getattr(active, "id", None)
            confirmed_id = getattr(confirmed, "id", None)
            if active_id == confirmed_id:
                return confirmed_id
        raise RuntimeError(
            f"Qdrant payload reconciliation did not converge for document {document_id}"
        )

    def _apply_payload_reconciliation(self, document_id: UUID, active: Any) -> None:
        if active is not None:
            # Preserve availability: make the authoritative generation visible
            # before attempting cleanup of stale payload flags.
            self._set_generation_active(document_id, active.id, True)
            retired_filter = FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=str(document_id)),
                        )
                    ],
                    must_not=[
                        FieldCondition(
                            key="index_generation",
                            match=MatchValue(value=str(active.id)),
                        )
                    ],
                )
            )
            self.qdrant_client.set_payload(
                collection_name=self.collection_name,
                payload={"is_active": False},
                points=retired_filter,
                wait=True,
            )
            return

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

    def purge_retired_generations(
        self, document_id: UUID, older_than: datetime
    ) -> list[UUID]:
        """Delete retired SQL/Qdrant generations after a caller-chosen rollback window."""
        document_id = self._coerce_uuid(document_id)
        purged: list[UUID] = []
        for generation in self.generation_repository.purgeable_before(
            document_id, older_than
        ):
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
    def _generation_filter(
        document_id: UUID, generation_id: UUID, *, modality: str | None = None
    ) -> Filter:
        must = [
            FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
            FieldCondition(key="index_generation", match=MatchValue(value=str(generation_id))),
        ]
        if modality is not None:
            must.append(FieldCondition(key="modality", match=MatchValue(value=modality)))
        return Filter(must=must)

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
