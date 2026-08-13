"""Transactional lifecycle operations for document index generations."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select

from app.models.document import Document
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

                (
                    db.query(DocumentIndexGeneration)
                    .filter(
                        DocumentIndexGeneration.document_id == target.document_id,
                        DocumentIndexGeneration.status == "active",
                        DocumentIndexGeneration.id != target.id,
                    )
                    .update({"status": "retired"}, synchronize_session=False)
                )
                db.flush()
                target.status = "active"
                target.failure_code = None
                target.activated_at = datetime.now(timezone.utc)
                db.commit()
                db.refresh(target)
                return target
            except Exception:
                db.rollback()
                raise

    def retired_before(
        self, document_id: UUID, cutoff: datetime
    ) -> list[DocumentIndexGeneration]:
        with self.session_factory() as db:
            return (
                db.query(DocumentIndexGeneration)
                .filter(
                    DocumentIndexGeneration.document_id == document_id,
                    DocumentIndexGeneration.status.in_(("retired", "failed")),
                    DocumentIndexGeneration.created_at < cutoff,
                )
                .order_by(DocumentIndexGeneration.created_at.asc())
                .all()
            )

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
