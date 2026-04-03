"""
Repository for managing document parse artifacts.

Parse artifacts store the output of document processing operations (MinerU, etc.).
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.models.document_parse_artifact import DocumentParseArtifact


class DocumentParseArtifactRepository:
    """Repository for DocumentParseArtifact model."""

    def __init__(self, session: Session | AsyncSession):
        self.session = session

    async def create(
        self,
        document_id: UUID,
        artifact_type: str,
        storage_path: str,
        mime_type: str,
        size_bytes: int,
        checksum_sha256: str | None = None,
        artifact_metadata: dict | None = None,
    ) -> DocumentParseArtifact:
        """
        Create a new parse artifact record.

        Args:
            document_id: The parent document ID.
            artifact_type: Type of artifact (e.g., "mineru_markdown", "content_list_json").
            storage_path: Path where the artifact is stored.
            mime_type: MIME type of the artifact.
            size_bytes: Size of the artifact in bytes.
            checksum_sha256: Optional SHA256 checksum.
            artifact_metadata: Optional metadata about the artifact.

        Returns:
            The created DocumentParseArtifact instance.
        """
        artifact = DocumentParseArtifact(
            document_id=document_id,
            artifact_type=artifact_type,
            storage_path=storage_path,
            mime_type=mime_type,
            size_bytes=size_bytes,
            checksum_sha256=checksum_sha256,
            artifact_metadata=artifact_metadata or {},
        )

        self.session.add(artifact)
        await self.session.commit()
        await self.session.refresh(artifact)

        return artifact

    async def get_by_id(self, artifact_id: UUID) -> DocumentParseArtifact | None:
        """Get an artifact by its ID."""
        result = await self.session.execute(
            select(DocumentParseArtifact).where(DocumentParseArtifact.id == artifact_id)
        )
        return result.scalar_one_or_none()

    async def list_by_document(
        self,
        document_id: UUID,
        artifact_type: str | None = None,
    ) -> list[DocumentParseArtifact]:
        """
        List all artifacts for a document.

        Args:
            document_id: The document ID.
            artifact_type: Optional filter by artifact type.

        Returns:
            List of DocumentParseArtifact instances.
        """
        query = select(DocumentParseArtifact).where(
            DocumentParseArtifact.document_id == document_id
        )

        if artifact_type:
            query = query.where(DocumentParseArtifact.artifact_type == artifact_type)

        result = await self.session.execute(query.order_by(DocumentParseArtifact.created_at))
        return list(result.scalars().all())

    async def get_by_document_and_type(
        self,
        document_id: UUID,
        artifact_type: str,
    ) -> DocumentParseArtifact | None:
        """
        Get a specific artifact by document and type.

        Args:
            document_id: The document ID.
            artifact_type: The artifact type.

        Returns:
            The artifact if found, None otherwise.
        """
        result = await self.session.execute(
            select(DocumentParseArtifact).where(
                DocumentParseArtifact.document_id == document_id,
                DocumentParseArtifact.artifact_type == artifact_type,
            )
        )
        return result.scalar_one_or_none()

    async def delete(self, artifact_id: UUID) -> bool:
        """
        Delete an artifact record.

        Note: This does NOT delete the actual file from storage.
        Caller is responsible for cleanup.

        Args:
            artifact_id: The artifact ID.

        Returns:
            True if deleted, False if not found.
        """
        artifact = await self.get_by_id(artifact_id)
        if not artifact:
            return False

        await self.session.delete(artifact)
        await self.session.commit()

        return True

    async def delete_by_document(self, document_id: UUID) -> int:
        """
        Delete all artifacts for a document.

        Note: This does NOT delete the actual files from storage.
        Caller is responsible for cleanup.

        Args:
            document_id: The document ID.

        Returns:
            Number of artifacts deleted.
        """
        artifacts = await self.list_by_document(document_id)

        for artifact in artifacts:
            await self.session.delete(artifact)

        await self.session.commit()

        return len(artifacts)
