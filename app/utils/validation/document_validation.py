"""
Document validation utilities
"""

from uuid import UUID

from app.core.exceptions import AuthorizationException, ResourceNotFoundException
from app.repositories.document import DocumentRepository


class DocumentValidationUtils:
    """Utilities for document-related validations"""

    def __init__(self, session_factory: callable):
        """Initialize validation utils with session factory for dependency injection."""
        self.session_factory = session_factory
        self.document_repository = DocumentRepository(session_factory)

    def validate_document_exists(self, document_id: UUID):
        """Validate that a document exists"""
        if not self.document_repository.exists(document_id):
            raise ResourceNotFoundException(
                detail="Document not found", error_code="DOCUMENT_NOT_FOUND"
            )

    def validate_user_owns_document(self, user_id: UUID, document_id: UUID):
        """Validate that the user owns the document via conversation ownership"""
        if not self.document_repository.user_owns_document(user_id, document_id):
            raise AuthorizationException(
                detail="Access denied to this document",
                error_code="DOCUMENT_ACCESS_DENIED",
            )

    def validate_document_access(self, user_id: UUID, document_id: UUID):
        """
        Validate document exists and user has access
        """
        self.validate_document_exists(document_id)
        self.validate_user_owns_document(user_id, document_id)
