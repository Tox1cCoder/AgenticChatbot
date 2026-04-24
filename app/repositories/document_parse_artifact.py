"""Repository for document parse artifacts.

Parse artifacts store the output of document processing operations (MinerU
markdown, content-list JSON, raw source copies, etc). Images live in their
own table and are not duplicated here.

Sync, session-factory-based — matches the rest of the server's DB layer.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.models.document_parse_artifact import DocumentParseArtifact


class DocumentParseArtifactRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    def create(
        self,
        *,
        document_id: UUID,
        artifact_type: str,
        storage_path: str,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        checksum_sha256: str | None = None,
        artifact_metadata: dict[str, Any] | None = None,
    ) -> DocumentParseArtifact:
        artifact = DocumentParseArtifact(
            document_id=document_id,
            artifact_type=artifact_type,
            storage_path=storage_path,
            mime_type=mime_type,
            size_bytes=size_bytes,
            checksum_sha256=checksum_sha256,
            artifact_metadata=artifact_metadata or {},
        )
        with self.session_factory() as session:
            session.add(artifact)
            session.commit()
            session.refresh(artifact)
        return artifact

    def replace_for_document(
        self,
        *,
        document_id: UUID,
        artifacts: list[dict[str, Any]],
    ) -> list[DocumentParseArtifact]:
        """Replace every artifact for the given document in a single transaction.

        Callers are responsible for cleaning up the old files on disk after
        the DB rows are gone.
        """
        created: list[DocumentParseArtifact] = []
        with self.session_factory() as session:
            session.query(DocumentParseArtifact).filter(
                DocumentParseArtifact.document_id == document_id
            ).delete(synchronize_session=False)

            for row in artifacts:
                artifact = DocumentParseArtifact(
                    document_id=document_id,
                    artifact_type=row["artifact_type"],
                    storage_path=row["storage_path"],
                    mime_type=row.get("mime_type"),
                    size_bytes=row.get("size_bytes"),
                    checksum_sha256=row.get("checksum_sha256"),
                    artifact_metadata=row.get("artifact_metadata") or {},
                )
                session.add(artifact)
                created.append(artifact)

            session.commit()
            for artifact in created:
                session.refresh(artifact)
        return created

    def delete_by_document(self, document_id: UUID) -> int:
        with self.session_factory() as session:
            deleted = (
                session.query(DocumentParseArtifact)
                .filter(DocumentParseArtifact.document_id == document_id)
                .delete(synchronize_session=False)
            )
            session.commit()
            return deleted

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    def get_by_id(self, artifact_id: UUID) -> DocumentParseArtifact | None:
        with self.session_factory() as session:
            return (
                session.query(DocumentParseArtifact)
                .filter(DocumentParseArtifact.id == artifact_id)
                .first()
            )

    def list_by_document(
        self,
        document_id: UUID,
        artifact_type: str | None = None,
    ) -> list[DocumentParseArtifact]:
        with self.session_factory() as session:
            query = session.query(DocumentParseArtifact).filter(
                DocumentParseArtifact.document_id == document_id
            )
            if artifact_type is not None:
                query = query.filter(DocumentParseArtifact.artifact_type == artifact_type)
            return query.all()

    def get_by_document_and_type(
        self,
        document_id: UUID,
        artifact_type: str,
    ) -> DocumentParseArtifact | None:
        with self.session_factory() as session:
            return (
                session.query(DocumentParseArtifact)
                .filter(
                    DocumentParseArtifact.document_id == document_id,
                    DocumentParseArtifact.artifact_type == artifact_type,
                )
                .first()
            )
