"""
Document service interface definition
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class IDocumentService(ABC):
    """Interface for Document service operations"""

    @abstractmethod
    async def upload_and_process_document(
        self, file_content: bytes, filename: str, file_size: int, user_id: str
    ) -> Dict[str, Any]:
        """
        Upload and process a document for RAG functionality.

        Args:
            file_content: The file content as bytes
            filename: Original filename
            file_size: Size of the file in bytes
            user_id: User ID for the document

        Returns:
            Dict containing document processing results
        """
        pass
