from typing import Optional
import logging
from uuid import UUID

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.ai.agents.rag_agent import RAGAgent
from app.core.config import get_settings
from app.core.exceptions.validation import FileValidationError
from app.interfaces.document_service_interface import IDocumentService
from app.repositories.document import DocumentRepository
from app.services.document_processing_service import DocumentProcessingService
from app.utils.validation.document_validation import DocumentValidationUtils
from app.core.events import get_event_bus, DocumentEvent, DocumentEventData
from app.schemas.document import (
    DocumentCreate,
    DocumentResponse,
    DocumentUpdate,
    DocumentListResponse,
    DocumentStatus,
)

logger = logging.getLogger(__name__)


class DocumentService(IDocumentService):
    """Service for handling document operations"""

    def __init__(
        self,
        document_repository: DocumentRepository,
        document_processing_service: DocumentProcessingService,
        document_validation_utils: DocumentValidationUtils,
        qdrant_client: QdrantClient,
        embedding_model: SentenceTransformer,
    ):
        """Initialize document service with injected dependencies."""
        self.repository = document_repository
        self.processing_service = document_processing_service
        self.document_validation_utils = document_validation_utils
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self._event_bus = get_event_bus()

    async def create_document(self, document_data: DocumentCreate) -> DocumentResponse:
        """Create a new document record."""
        document = self.repository.create(document_data)
        return DocumentResponse.model_validate(document)

    async def get_document(self, document_id: UUID) -> Optional[DocumentResponse]:
        """Get document by ID."""
        document = self.repository.get_by_id(document_id)
        if document:
            return DocumentResponse.model_validate(document)
        return None

    async def update_document(
        self, document_id: UUID, document_data: DocumentUpdate
    ) -> Optional[DocumentResponse]:
        """Update document."""
        document = self.repository.update(document_id, document_data)
        if document:
            return DocumentResponse.model_validate(document)
        return None

    async def set_processing_task_id(
        self, document_id: UUID, task_id: str
    ) -> Optional[DocumentResponse]:
        """Persist the background-processing task ID for a document."""
        document = self.repository.set_processing_task_id(document_id, task_id)
        if document:
            return DocumentResponse.model_validate(document)
        return None

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete document and its vectors from Qdrant."""
        # Fetch document to include details and validate existence
        existing = self.repository.get_by_id(document_id)
        if not existing:
            return False

        try:
            settings = get_settings()
            rag_agent = RAGAgent(
                settings=settings,
                qdrant_client=self.qdrant_client,
                embedding_model=self.embedding_model,
                collection_name=settings.qdrant_collection_name,
            )
            await rag_agent.initialize()
            result = await rag_agent.delete_document_vectors(str(document_id))
            await rag_agent.cleanup()

            if result.get("success"):
                logger.info(f"Successfully deleted vectors for document {document_id}")
            else:
                logger.warning(
                    f"Failed to delete vectors for document {document_id}: {result.get('error')}"
                )
        except Exception as e:
            logger.error(f"Error deleting vectors for document {document_id}: {e}")

        # Delete from database
        deleted = self.repository.delete(document_id)

        if deleted:
            # Emit DELETED event
            await self._event_bus.emit(
                DocumentEvent.DELETED,
                DocumentEventData(
                    document_id=document_id,
                    conversation_id=existing.conversation_id,
                    filename=existing.filename,
                    status="DELETED",
                ),
            )

        return deleted

    async def get_documents_by_conversation(
        self, conversation_id: UUID, page: int = 1, page_size: int = 20
    ) -> DocumentListResponse:
        """Get paginated documents for a conversation."""
        documents, total = self.repository.get_by_conversation_id(
            conversation_id, page, page_size
        )

        document_responses = [DocumentResponse.model_validate(doc) for doc in documents]

        total_pages = (total + page_size - 1) // page_size

        return DocumentListResponse(
            documents=document_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    async def update_status(
        self, document_id: UUID, status: DocumentStatus
    ) -> Optional[DocumentResponse]:
        """Update document status."""
        update_data = DocumentUpdate(status=status)
        return await self.update_document(document_id, update_data)

    async def validate_and_create_document(
        self,
        filename: str,
        file_size: int,
        content_type: str,
        conversation_id: UUID,
    ) -> DocumentResponse:
        """Validate file and create document record with validations."""
        # Validate filename
        if not filename:
            raise FileValidationError(detail="No file provided")

        # Use processing service for validation
        await self.processing_service.validate_upload_file(filename, file_size)

        # Create document record
        document_data = DocumentCreate(
            conversation_id=conversation_id,
            filename=filename,
            file_type=content_type or "unknown",
            status=DocumentStatus.PROCESSING.value,
        )

        return await self.create_document(document_data)
