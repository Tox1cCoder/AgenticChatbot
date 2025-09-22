from typing import Dict, Any, Optional
import tempfile
import os
import logging
from pathlib import Path

from app.services.ai_service import AIService
from app.core.exceptions import DocumentProcessingError, FileValidationError

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for handling document upload and processing operations."""

    def __init__(self):
        self.logger = logging.getLogger("document_upload_service")
        self.allowed_types = [".pdf", ".txt", ".docx"]
        self.max_file_size = 10 * 1024 * 1024  # 10MB

    async def upload_and_process_document(
        self, file_content: bytes, filename: str, file_size: int, user_id: str
    ) -> Dict[str, Any]:
        """
        Upload and process a document for RAG functionality.

        Args:
            file_content: The file content as bytes
            filename: Original filename
            file_size: Size of the file in bytes
            user_id: ID of the user uploading the document

        Returns:
            Dict with processing results

        Raises:
            FileValidationError: If file validation fails
            DocumentProcessingError: If document processing fails
        """

        # Validate file
        self._validate_file(filename, file_size)

        temp_file_path = None
        try:
            # Create temporary file
            temp_file_path = await self._create_temp_file(file_content, filename)

            # Process the document
            processing_result = await self._process_document(
                temp_file_path, filename, user_id
            )

            self.logger.info(
                f"Successfully processed document: {filename} for user {user_id}"
            )

            return {
                "filename": filename,
                "file_size": file_size,
                "chunks_created": processing_result.get("chunks_created", 0),
                "processing_time": processing_result.get("processing_time", 0),
            }

        except Exception as e:
            self.logger.error(f"Document processing failed for {filename}: {str(e)}")
            raise DocumentProcessingError(f"Failed to process document: {str(e)}")

        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to cleanup temp file: {cleanup_error}")

    def _validate_file(self, filename: str, file_size: int) -> None:
        """
        Validate uploaded file.

        Args:
            filename: Name of the file
            file_size: Size of the file in bytes

        Raises:
            FileValidationError: If validation fails
        """

        # Validate file type
        file_extension = Path(filename).suffix.lower()
        if file_extension not in self.allowed_types:
            raise FileValidationError(
                f"Unsupported file type '{file_extension}'. "
                f"Allowed types: {', '.join(self.allowed_types)}"
            )

        # Validate file size
        if file_size > self.max_file_size:
            max_size_mb = self.max_file_size / (1024 * 1024)
            raise FileValidationError(f"File size exceeds {max_size_mb}MB limit")

    async def _create_temp_file(self, file_content: bytes, filename: str) -> str:
        """
        Create a temporary file with the uploaded content.

        Args:
            file_content: Content of the file
            filename: Original filename

        Returns:
            Path to the temporary file
        """

        file_extension = Path(filename).suffix.lower()

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as temp_file:
            temp_file.write(file_content)
            return temp_file.name

    async def _process_document(
        self, file_path: str, filename: str, user_id: str
    ) -> Dict[str, Any]:
        """
        Process the document using AI service.

        Args:
            file_path: Path to the temporary file
            filename: Original filename
            user_id: ID of the user

        Returns:
            Processing results
        """

        # Get AI service and RAG agent
        ai_service = AIService()
        rag_agent = ai_service.workflow.rag_agent

        # Process the document
        return await rag_agent.process_document(
            file_path=file_path, filename=filename, user_id=user_id
        )
