from typing import Optional, List
import logging
from uuid import UUID
from sqlalchemy.orm import Session

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
        """Delete document."""
        db: Session = next(get_db())
        try:
            document_repo = DocumentRepository(db)
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
