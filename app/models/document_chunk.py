"""ORM model for the normalized ``document_chunks`` table.

This is the canonical content store for retrieved document chunks. Qdrant
only keeps embeddings plus lookup IDs; the text and all authorization-sensitive
metadata live here.
"""

from __future__ import annotations

import uuid

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "index_generation_id",
            "chunk_index",
            name="uq_document_chunk_generation_index",
        ),
        Index("idx_document_chunks_document_id", "document_id"),
        Index("idx_document_chunks_parse_artifact_id", "parse_artifact_id"),
        Index("idx_document_chunks_content_sha256", "content_sha256"),
        Index("idx_document_chunks_index_status", "index_status"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    index_generation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("document_index_generations.id", ondelete="CASCADE"),
        nullable=False,
    )
    parse_artifact_id = Column(
        UUID(as_uuid=True),
        ForeignKey("document_parse_artifacts.id", ondelete="SET NULL"),
        nullable=True,
    )
    qdrant_point_id = Column(String(100), nullable=True, unique=True)

    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_sha256 = Column(String(64), nullable=False)
    char_count = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False)

    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)

    section_path = Column(JSONB, nullable=False, default=list)
    block_provenance = Column(JSONB, nullable=False, default=list)
    chunk_metadata = Column(JSONB, nullable=False, default=dict)

    index_status = Column(String(32), nullable=False, default="pending")
    index_error = Column(Text, nullable=True)
    indexed_at = Column(DateTime(timezone=True), nullable=True)

    embedding_model = Column(String, nullable=True)
    embedding_dimension = Column(Integer, nullable=True)
    qdrant_collection_name = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    document = relationship("Document", back_populates="chunks")
    index_generation = relationship("DocumentIndexGeneration", back_populates="chunks")
    parse_artifact = relationship("DocumentParseArtifact", back_populates="chunks")
    images = relationship("DocumentImage", back_populates="chunk")

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk(id={self.id}, document_id={self.document_id}, "
            f"chunk_index={self.chunk_index}, index_status='{self.index_status}')>"
        )
