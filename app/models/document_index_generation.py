"""Versioned ownership and activation state for a document vector index."""

from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class DocumentIndexGeneration(Base):
    __tablename__ = "document_index_generations"
    __table_args__ = (
        Index("idx_document_index_generations_document", "document_id"),
        Index("idx_document_index_generations_status", "status"),
        Index(
            "uq_document_index_generation_active",
            "document_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
            sqlite_where=text("status = 'active'"),
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    status = Column(String(16), nullable=False, default="building")
    embedding_provider = Column(String(64), nullable=False)
    embedding_model = Column(String(255), nullable=False)
    embedding_dimension = Column(Integer, nullable=False)
    chunking_version = Column(String(64), nullable=False)
    failure_code = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    activated_at = Column(DateTime(timezone=True), nullable=True)
    retired_at = Column(DateTime(timezone=True), nullable=True)
    failed_at = Column(DateTime(timezone=True), nullable=True)

    document = relationship("Document", back_populates="index_generations")
    chunks = relationship(
        "DocumentChunk",
        back_populates="index_generation",
        cascade="all, delete-orphan",
    )
