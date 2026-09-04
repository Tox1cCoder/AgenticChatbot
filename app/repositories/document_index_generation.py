"""Transactional lifecycle operations for document index generations."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import and_, or_, select

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_image import DocumentImage
from app.models.document_index_generation import DocumentIndexGeneration


class DocumentIndexGenerationRepository:
    FAILURE_CODE_LIMIT = 64

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def create(
        self,
        *,
        document_id: UUID,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        chunking_version: str,
    ) -> DocumentIndexGeneration:
        generation = DocumentIndexGeneration(
            document_id=document_id,
            status="building",
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            chunking_version=chunking_version,
        )
        with self.session_factory() as db:
            db.add(generation)
            db.commit()
            db.refresh(generation)
        return generation

    def get(self, generation_id: UUID) -> DocumentIndexGeneration | None:
        with self.session_factory() as db:
            return db.get(DocumentIndexGeneration, generation_id)

    def get_active(self, document_id: UUID) -> DocumentIndexGeneration | None:
        with self.session_factory() as db:
            return (
                db.query(DocumentIndexGeneration)
                .filter(
                    DocumentIndexGeneration.document_id == document_id,
                    DocumentIndexGeneration.status == "active",
                )
                .first()
            )

    def get_latest_failed(self, document_id: UUID) -> DocumentIndexGeneration | None:
        with self.session_factory() as db:
            return (
                db.query(DocumentIndexGeneration)
                .filter(
                    DocumentIndexGeneration.document_id == document_id,
                    DocumentIndexGeneration.status == "failed",
                )
                .order_by(DocumentIndexGeneration.created_at.desc())
                .first()
            )

    def list_for_document(self, document_id: UUID) -> list[DocumentIndexGeneration]:
        with self.session_factory() as db:
            return (
                db.query(DocumentIndexGeneration)
                .filter(DocumentIndexGeneration.document_id == document_id)
                .order_by(DocumentIndexGeneration.created_at.asc())
                .all()
            )

    def mark_ready(self, generation_id: UUID) -> DocumentIndexGeneration:
        with self.session_factory() as db:
            generation = db.get(DocumentIndexGeneration, generation_id)
            if generation is None:
                raise ValueError(f"unknown index generation: {generation_id}")
            if generation.status != "building":
                raise ValueError(f"generation {generation_id} is not building")
            generation.status = "ready"
            generation.failure_code = None
            generation.failed_at = None
            db.commit()
            db.refresh(generation)
            return generation

    def mark_failed(self, generation_id: UUID, failure_code: str) -> DocumentIndexGeneration:
        bounded_code = str(failure_code or "INDEX_BUILD_FAILED")[: self.FAILURE_CODE_LIMIT]
        with self.session_factory() as db:
            generation = db.get(DocumentIndexGeneration, generation_id)
            if generation is None:
                raise ValueError(f"unknown index generation: {generation_id}")
            if generation.status == "active":
                raise ValueError("an active generation cannot be marked failed")
            generation.status = "failed"
            generation.failure_code = bounded_code
            generation.failed_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(generation)
            return generation

    def activate(self, generation_id: UUID) -> DocumentIndexGeneration:
        """Atomically retire the current generation and activate ``generation_id``.

        Locking the parent document serializes two concurrent replacements even
        when they target different generation rows. The partial unique index is
        the final database-level invariant.
        """
        with self.session_factory() as db:
            try:
                target = db.execute(
                    select(DocumentIndexGeneration)
                    .where(DocumentIndexGeneration.id == generation_id)
                    .with_for_update()
                ).scalar_one_or_none()
                if target is None:
                    raise ValueError(f"unknown index generation: {generation_id}")
                if target.status not in {"building", "ready"}:
                    raise ValueError(f"generation {generation_id} cannot be activated")

                # PostgreSQL row-lock; SQLite ignores it but the partial unique
                # index still guards the invariant in repository tests.
                if db.get_bind().dialect.name == "postgresql":
                    db.execute(
                        select(Document.id)
                        .where(Document.id == target.document_id)
                        .with_for_update()
                    ).scalar_one_or_none()

                retired_at = datetime.now(timezone.utc)
                old_active_ids = [
                    row.id
                    for row in db.query(DocumentIndexGeneration.id)
                    .filter(
                        DocumentIndexGeneration.document_id == target.document_id,
                        DocumentIndexGeneration.status == "active",
                        DocumentIndexGeneration.id != target.id,
                    )
                    .all()
                ]
                self._rebind_images(db, old_active_ids, target.id)
                (
                    db.query(DocumentIndexGeneration)
                    .filter(
                        DocumentIndexGeneration.document_id == target.document_id,
                        DocumentIndexGeneration.status == "active",
                        DocumentIndexGeneration.id != target.id,
                    )
                    .update(
                        {"status": "retired", "retired_at": retired_at},
                        synchronize_session=False,
                    )
                )
                db.flush()
                target.status = "active"
                target.failure_code = None
                target.activated_at = datetime.now(timezone.utc)
                target.retired_at = None
                target.failed_at = None
                db.commit()
                db.refresh(target)
                return target
            except Exception:
                db.rollback()
                raise

    @staticmethod
    def _rebind_images(db, old_generation_ids: list[UUID], target_generation_id: UUID) -> None:
        if not old_generation_ids:
            return
        old_chunks = (
            db.query(DocumentChunk.id, DocumentChunk.chunk_index)
            .filter(DocumentChunk.index_generation_id.in_(old_generation_ids))
            .all()
        )
        target_by_index = {
            row.chunk_index: row.id
            for row in db.query(DocumentChunk.id, DocumentChunk.chunk_index)
            .filter(DocumentChunk.index_generation_id == target_generation_id)
            .all()
        }
        old_index_by_id = {row.id: row.chunk_index for row in old_chunks}
        if not old_index_by_id:
            return
        images = db.query(DocumentImage).filter(DocumentImage.chunk_id.in_(old_index_by_id)).all()
        for image in images:
            image.chunk_id = target_by_index.get(old_index_by_id[image.chunk_id])

    def purgeable_before(
        self, document_id: UUID, cutoff: datetime
    ) -> list[DocumentIndexGeneration]:
        with self.session_factory() as db:
            return (
                db.query(DocumentIndexGeneration)
                .filter(
                    DocumentIndexGeneration.document_id == document_id,
                    or_(
                        and_(
                            DocumentIndexGeneration.status == "retired",
                            DocumentIndexGeneration.retired_at.is_not(None),
                            DocumentIndexGeneration.retired_at < cutoff,
                        ),
                        and_(
                            DocumentIndexGeneration.status == "failed",
                            DocumentIndexGeneration.failed_at.is_not(None),
                            DocumentIndexGeneration.failed_at < cutoff,
                        ),
                    ),
                )
                .order_by(DocumentIndexGeneration.created_at.asc())
                .all()
            )

    def retired_before(self, document_id: UUID, cutoff: datetime) -> list[DocumentIndexGeneration]:
        """Backward-compatible alias for all safe-to-purge inactive generations."""
        return self.purgeable_before(document_id, cutoff)

    def delete(self, generation_id: UUID) -> bool:
        with self.session_factory() as db:
            generation = db.get(DocumentIndexGeneration, generation_id)
            if generation is None:
                return False
            if generation.status == "active":
                raise ValueError("active generation cannot be purged")
            db.delete(generation)
            db.commit()
            return True
