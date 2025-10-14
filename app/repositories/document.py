from typing import List, Optional, Tuple
from uuid import UUID
from sqlalchemy import desc

from app.models.document import Document
from app.models.conversation import Conversation
from app.schemas.document import DocumentCreate, DocumentUpdate


class DocumentRepository:
    """Repository for document operations"""

    def __init__(self, session_factory: callable):
        """Initialize repository with session factory for dependency injection."""
        self.session_factory = session_factory

    def create(self, document_data: DocumentCreate) -> Document:
        """Create a new document"""
        with self.session_factory() as db:
            db_document = Document(
                conversation_id=document_data.conversation_id,
                filename=document_data.filename,
                file_type=document_data.file_type,
                status=document_data.status,
            )
            db.add(db_document)
            db.commit()
            db.refresh(db_document)
            return db_document

    def get_by_id(self, document_id: UUID) -> Optional[Document]:
        """Get document by ID"""
        with self.session_factory() as db:
            return db.query(Document).filter(Document.id == document_id).first()

    def get_by_conversation_id(
        self, conversation_id: UUID, page: int = 1, page_size: int = 20
    ) -> Tuple[List[Document], int]:
        """Get paginated documents for a conversation with total count"""
        with self.session_factory() as db:
            skip = (page - 1) * page_size

            query = db.query(Document).filter(
                Document.conversation_id == conversation_id
            )

            total = query.count()
            documents = (
                query.order_by(desc(Document.upload_time))
                .offset(skip)
                .limit(page_size)
                .all()
            )

            return documents, total

    def update(
        self, document_id: UUID, update_data: DocumentUpdate
    ) -> Optional[Document]:
        """Update document with new data"""
        with self.session_factory() as db:
            db_document = db.query(Document).filter(Document.id == document_id).first()
            if not db_document:
                return None

            update_dict = update_data.model_dump(exclude_unset=True)

            for key, value in update_dict.items():
                setattr(db_document, key, value)

            db.commit()
            db.refresh(db_document)
            return db_document

    def delete(self, document_id: UUID) -> bool:
        """Delete a document"""
        with self.session_factory() as db:
            db_document = db.query(Document).filter(Document.id == document_id).first()
            if not db_document:
                return False

            db.delete(db_document)
            db.commit()
            return True

    def get_by_status(
        self, status: int, page: int = 1, page_size: int = 20
    ) -> Tuple[List[Document], int]:
        """Get paginated documents by status with total count"""
        with self.session_factory() as db:
            skip = (page - 1) * page_size

            query = db.query(Document).filter(Document.status == status)

            total = query.count()
            documents = (
                query.order_by(desc(Document.upload_time))
                .offset(skip)
                .limit(page_size)
                .all()
            )

            return documents, total

    def count_all(self) -> int:
        """Count total documents"""
        with self.session_factory() as db:
            return db.query(Document).count()

    def count_by_conversation(self, conversation_id: UUID) -> int:
        """Count documents for a conversation"""
        with self.session_factory() as db:
            return (
                db.query(Document)
                .filter(Document.conversation_id == conversation_id)
                .count()
            )

    # Authorization helpers
    def exists(self, document_id: UUID) -> bool:
        """Check if a document exists by ID."""
        with self.session_factory() as db:
            return (
                db.query(Document).filter(Document.id == document_id).first()
                is not None
            )

    def user_owns_document(self, user_id: UUID, document_id: UUID) -> bool:
        """Check if a user owns the document via its conversation ownership."""
        with self.session_factory() as db:
            q = (
                db.query(Document)
                .join(Conversation, Conversation.id == Document.conversation_id)
                .filter(Document.id == document_id, Conversation.owner_id == user_id)
            )
            return db.query(q.exists()).scalar()
