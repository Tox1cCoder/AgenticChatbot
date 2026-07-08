import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class DocumentImage(Base):
    __tablename__ = "document_images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    chunk_id = Column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    image_path = Column(String(500), nullable=False)
    image_caption = Column(Text, nullable=True)
    page_number = Column(Integer, nullable=True)
    mime_type = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.now(timezone.utc))

    # Relationships
    document = relationship("Document", back_populates="images")
    chunk = relationship("DocumentChunk", back_populates="images")

    __table_args__ = (
        Index("idx_document_images_document_id", "document_id"),
        Index("idx_document_images_chunk_id", "chunk_id"),
    )
