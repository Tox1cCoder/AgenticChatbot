from abc import ABC, abstractmethod
from uuid import UUID

from app.schemas.document import (
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
    DocumentStatus,
    DocumentUpdate,
)


class IDocumentService(ABC):
    """Interface for Document service operations"""

    @abstractmethod
    async def create_document(self, document_data: DocumentCreate) -> DocumentResponse:
        """Create a new document"""
        pass

    @abstractmethod
    async def get_document(self, document_id: UUID) -> DocumentResponse | None:
        """Get document by ID"""
        pass

    @abstractmethod
    async def update_document(
        self, document_id: UUID, document_data: DocumentUpdate
    ) -> DocumentResponse | None:
        """Update document"""
        pass

    @abstractmethod
    async def set_processing_task_id(
        self, document_id: UUID, task_id: str
    ) -> DocumentResponse | None:
        """Persist the background-processing task ID for a document."""
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
    ) -> DocumentResponse | None:
        """Update document status"""
        pass

    @abstractmethod
    async def validate_and_create_document(
        self,
        filename: str,
        file_size: int,
        content_type: str,
        conversation_id: UUID,
    ) -> DocumentResponse:
        """Validate file and create document record"""
        pass
