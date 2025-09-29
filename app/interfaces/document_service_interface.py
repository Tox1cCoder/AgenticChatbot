from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from app.schemas.document import (
    DocumentCreate,
    DocumentResponse,
    DocumentUpdate,
    DocumentListResponse,
    DocumentStatus,
)


class IDocumentService(ABC):
    """Interface for Document service operations"""

    @abstractmethod
    async def create_document(self, document_data: DocumentCreate) -> DocumentResponse:
        """Create a new document"""
        pass

    @abstractmethod
    async def get_document(self, document_id: UUID) -> Optional[DocumentResponse]:
        """Get document by ID"""
        pass

    @abstractmethod
    async def update_document(
        self, document_id: UUID, document_data: DocumentUpdate
    ) -> Optional[DocumentResponse]:
        """Update document"""
        pass

    @abstractmethod
    async def delete_document(self, document_id: UUID) -> bool:
        """Delete document"""
        pass

    @abstractmethod
    async def get_documents_by_conversation(
        self, conversation_id: UUID, page: int = 1, page_size: int = 20
    ) -> DocumentListResponse:
        """Get paginated documents for a conversation"""
        pass

    @abstractmethod
    async def update_status(
        self, document_id: UUID, status: DocumentStatus
    ) -> Optional[DocumentResponse]:
        """Update document status"""
        pass