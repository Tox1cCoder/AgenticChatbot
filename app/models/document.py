import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (Index("idx_document_processing_task_id", "processing_task_id"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(100), nullable=False)
    status = Column(Integer, nullable=False, default=1)  # 1=processing, 2=ready, 3=failed
    upload_time = Column(
        DateTime(timezone=True), nullable=False, default=datetime.now(timezone.utc)
    )
    # Celery task ID persisted at enqueue time for ownership-safe status polling.
    processing_task_id = Column(String(255), nullable=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="documents")
    images = relationship("DocumentImage", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )
