from typing import Optional, List
import logging
import os
from uuid import UUID
from sqlalchemy.orm import Session
import asyncio

from app.ai.agents.rag_agent import RAGAgent
from app.core.config import get_settings
from app.core.exceptions.validation import FileValidationError
from app.core.exceptions.resource import ResourceNotFoundException
from app.interfaces.document_service_interface import IDocumentService
from app.repositories.document import DocumentRepository
from app.schemas.document import (
    DocumentCreate,
    DocumentResponse,
    DocumentUpdate,
    DocumentListResponse,
    DocumentStatus,
)
from app.database.session import get_db

logger = logging.getLogger(__name__)


class DocumentService(IDocumentService):
    """Service for handling document operations"""

    def __init__(self):
        """Initialize document service."""
        pass

    async def create_document(self, document_data: DocumentCreate) -> DocumentResponse:
        """Create a new document record."""
        db: Session = next(get_db())
        try:
            document_repo = DocumentRepository(db)
            document = document_repo.create(document_data)
            return DocumentResponse.model_validate(document)
        finally:
            db.close()

    async def get_document(self, document_id: UUID) -> Optional[DocumentResponse]:
        """Get document by ID."""
        db: Session = next(get_db())
        try:
            document_repo = DocumentRepository(db)
            document = document_repo.get_by_id(document_id)
            if document:
                return DocumentResponse.model_validate(document)
            return None
        finally:
            db.close()

    async def update_document(
        self, document_id: UUID, document_data: DocumentUpdate
    ) -> Optional[DocumentResponse]:
        """Update document."""
        db: Session = next(get_db())
        try:
            document_repo = DocumentRepository(db)
            document = document_repo.update(document_id, document_data)
            if document:
                return DocumentResponse.model_validate(document)
            return None
        finally:
            db.close()

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete document and its vectors from Qdrant."""
        db: Session = next(get_db())
        try:
            document_repo = DocumentRepository(db)

            # Delete vectors from Qdrant first
            try:
                settings = get_settings()
                rag_agent = RAGAgent(
                    qdrant_url=settings.qdrant_url,
                    collection_name=settings.qdrant_collection_name,
                )
                await rag_agent.initialize()
                result = await rag_agent.delete_document_vectors(str(document_id))
                await rag_agent.cleanup()

                if result.get("success"):
                    logger.info(
                        f"Successfully deleted vectors for document {document_id}"
                    )
                else:
                    logger.warning(
                        f"Failed to delete vectors for document {document_id}: {result.get('error')}"
                    )
            except Exception as e:
                logger.error(f"Error deleting vectors for document {document_id}: {e}")

            # Delete from database
            return document_repo.delete(document_id)
        finally:
            db.close()

    async def get_documents_by_conversation(
        self, conversation_id: UUID, page: int = 1, page_size: int = 20
    ) -> DocumentListResponse:
        """Get paginated documents for a conversation."""
        db: Session = next(get_db())
        try:
            document_repo = DocumentRepository(db)
            documents, total = document_repo.get_by_conversation_id(
                conversation_id, page, page_size
            )

            document_responses = [
                DocumentResponse.model_validate(doc) for doc in documents
            ]

            total_pages = (total + page_size - 1) // page_size

            return DocumentListResponse(
                documents=document_responses,
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
            )
        finally:
            db.close()

    async def update_status(
        self, document_id: UUID, status: DocumentStatus
    ) -> Optional[DocumentResponse]:
        """Update document status."""
        update_data = DocumentUpdate(status=status)
        return await self.update_document(document_id, update_data)

    async def validate_and_create_document(
        self,
        filename: str,
        file_content: bytes,
        content_type: str,
        conversation_id: UUID,
    ) -> DocumentResponse:
        """Validate file and create document record with validations."""
        # Validate filename
        if not filename:
            raise FileValidationError(detail="No file provided")

        # Get settings
        settings = get_settings()
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024

        # Validate file size
        if len(file_content) > max_size_bytes:
            raise FileValidationError(
                detail=f"File size exceeds maximum allowed size of {settings.max_file_size_mb}MB"
            )

        # Validate file extension
        allowed_extensions = {".txt", ".pdf", ".docx"}
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise FileValidationError(
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )

        # Create document record
        document_data = DocumentCreate(
            conversation_id=conversation_id,
            filename=filename,
            file_type=content_type or "unknown",
            status=DocumentStatus.PROCESSING.value,
        )

        return await self.create_document(document_data)
