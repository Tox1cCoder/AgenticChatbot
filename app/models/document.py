import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("idx_document_processing_task_id", "processing_task_id"),
        UniqueConstraint(
            "conversation_id",
            "filename_key",
            name="uq_documents_conversation_filename_key",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    # Normalized, casefolded filename used to detect same-conversation duplicates.
    filename_key = Column(String(255), nullable=False)
    file_type = Column(String(100), nullable=False)
    status = Column(Integer, nullable=False, default=1)  # 1=processing, 2=ready, 3=failed
    # Callable, not ``datetime.now(timezone.utc)``: an evaluated default is
    # computed once at import, stamping every document a process uploads with
    # that process's start time and making upload_time ordering arbitrary.
    upload_time = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    # Celery task ID persisted at enqueue time for ownership-safe status polling.
    processing_task_id = Column(String(255), nullable=True)

    # Relationships. ``chunks`` and ``index_generations`` cascade in the
    # database (ON DELETE CASCADE), so they are passive: deleting a document
    # must not load every chunk first. ``images`` and ``parse_artifacts`` have
    # plain foreign keys, so the ORM cascade is the only thing deleting them.
    conversation = relationship("Conversation", back_populates="documents")
    images = relationship("DocumentImage", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    index_generations = relationship(
        "DocumentIndexGeneration",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    parse_artifacts = relationship(
        "DocumentParseArtifact",
        back_populates="document",
        cascade="all, delete-orphan",
    )
