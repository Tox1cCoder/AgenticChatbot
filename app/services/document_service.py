import asyncio
import logging
import unicodedata
from uuid import UUID

from app.core.exceptions.validation import (
    DuplicateDocumentFilenameError,
    FileValidationError,
)
from app.interfaces.document_service_interface import IDocumentService
from app.repositories.document import DocumentRepository
from app.schemas.document import (
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
    DocumentStatus,
    DocumentUpdate,
)
from app.services.document_index_service import DocumentIndexService
from app.services.document_processing_service import DocumentProcessingService
from app.utils.validation.document_validation import DocumentValidationUtils

logger = logging.getLogger(__name__)


def normalize_document_filename(filename: str) -> str:
    """Return the casefolded, path-stripped, NFC-normalized filename key.

    Used to detect same-conversation duplicates across case and Unicode
    normalization variants. Raises ``FileValidationError`` for empty
    inputs because an empty normalized key cannot be stored.
    """
    name = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name:
        raise FileValidationError(detail="Filename is empty after normalization.")
    name = unicodedata.normalize("NFC", name)
    return name.casefold()


class DocumentService(IDocumentService):
    """Service for handling document CRUD operations.

    Vector/chunk cleanup is delegated to ``DocumentIndexService`` — deletion
    must not instantiate the agent runtime.

    The methods are coroutines and the repository is synchronous, so every
    repository and index call runs in a worker thread. Run on the loop, each
    one blocked every other request and in-flight stream for its round trip.
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        document_processing_service: DocumentProcessingService,
        document_validation_utils: DocumentValidationUtils,
        document_index_service: DocumentIndexService,
    ):
        self.repository = document_repository
        self.processing_service = document_processing_service
        self.document_validation_utils = document_validation_utils
        self.index_service = document_index_service

    async def create_document(self, document_data: DocumentCreate) -> DocumentResponse:
        document = await asyncio.to_thread(self.repository.create, document_data)
        return DocumentResponse.model_validate(document)

    async def get_document(self, document_id: UUID) -> DocumentResponse | None:
        document = await asyncio.to_thread(self.repository.get_by_id, document_id)
        if document:
            return DocumentResponse.model_validate(document)
        return None

    async def update_document(
        self, document_id: UUID, document_data: DocumentUpdate
    ) -> DocumentResponse | None:
        document = await asyncio.to_thread(self.repository.update, document_id, document_data)
        if document:
            return DocumentResponse.model_validate(document)
        return None

    async def set_processing_task_id(
        self, document_id: UUID, task_id: str
    ) -> DocumentResponse | None:
        document = await asyncio.to_thread(
            self.repository.set_processing_task_id, document_id, task_id
        )
        if document:
            return DocumentResponse.model_validate(document)
        return None

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete document and its vectors / chunks via the index service."""
        return await asyncio.to_thread(self._delete_document, document_id)

    def _delete_document(self, document_id: UUID) -> bool:
        existing = self.repository.get_by_id(document_id)
        if not existing:
            return False

        try:
            self.index_service.delete_document_index(document_id)
        except Exception as exc:
            logger.error(
                "Failed to clean chunks/vectors for document %s: %s",
                document_id,
                exc,
                exc_info=True,
            )

        return self.repository.delete(document_id)

    async def get_documents_by_conversation(
        self, conversation_id: UUID, page: int = 1, page_size: int = 20
    ) -> DocumentListResponse:
        documents, total = await asyncio.to_thread(
            self.repository.get_by_conversation_id, conversation_id, page, page_size
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
    ) -> DocumentResponse | None:
        update_data = DocumentUpdate(status=status)
        return await self.update_document(document_id, update_data)

    async def validate_and_create_document(
        self,
        filename: str,
        file_size: int,
        content_type: str,
        conversation_id: UUID,
    ) -> DocumentResponse:
        if not filename:
            raise FileValidationError(detail="No file provided")

        await self.processing_service.validate_upload_file(filename, file_size)

        filename_key = normalize_document_filename(filename)
        existing = await asyncio.to_thread(
            self.repository.get_by_conversation_and_filename_key, conversation_id, filename_key
        )
        if existing is not None:
            if existing.status == DocumentStatus.FAILED.value:
                # Auto-replace: delete the failed document so the new upload
                # can proceed with the same filename. Through delete_document,
                # because a failure can come after some chunks were indexed and
                # a bare row delete left those vectors behind.
                await self.delete_document(existing.id)
            else:
                raise DuplicateDocumentFilenameError(
                    detail=(f"A document named '{filename}' already exists in this conversation.")
                )

        document_data = DocumentCreate(
            conversation_id=conversation_id,
            filename=filename,
            filename_key=filename_key,
            file_type=content_type or "unknown",
            status=DocumentStatus.PROCESSING.value,
        )

        return await self.create_document(document_data)
