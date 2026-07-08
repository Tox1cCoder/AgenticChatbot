"""Model for tracking document parse artifacts from MinerU processing."""

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class DocumentParseArtifact(Base):
    """
    Represents a parse artifact generated from document processing (e.g., MinerU).

    Artifacts can include:
    - Resolved MinerU markdown output
    - content_list.json
    - Parse metadata needed for retrieval/debugging

    Images are stored separately in document_images table.
    """

    __tablename__ = "document_parse_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "artifact_type",
            "storage_path",
            name="uq_document_parse_artifact_path",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )

    # Parent document
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id"),
        nullable=False,
        index=True,
    )

    # Artifact metadata
    artifact_type = Column(String(64), nullable=False, index=True)
    storage_path = Column(String(1024), nullable=False)
    mime_type = Column(String(255), nullable=True)
    size_bytes = Column(Integer, nullable=True)
    checksum_sha256 = Column(String(128), nullable=True)

    # Additional metadata (e.g., page numbers, parse options, etc.)
    artifact_metadata = Column(JSONB, nullable=False, default=dict)

    # Relationships
    document = relationship("Document", back_populates="parse_artifacts")
    chunks = relationship("DocumentChunk", back_populates="parse_artifact")

    def __repr__(self) -> str:
        return (
            f"<DocumentParseArtifact(id={self.id}, type='{self.artifact_type}', "
            f"document_id={self.document_id})>"
        )
