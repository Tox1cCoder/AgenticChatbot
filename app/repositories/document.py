from typing import List, Optional, Tuple
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.document import Document
from app.schemas.document import DocumentCreate, DocumentUpdate


class DocumentRepository:
    """Repository for document operations"""

    def __init__(self, db: Session):
        self.db = db

    def create(self, document_data: DocumentCreate) -> Document:
        """Create a new document"""
        db_document = Document(
            conversation_id=document_data.conversation_id,
            filename=document_data.filename,
            file_type=document_data.file_type,
            status=document_data.status,
        )
        self.db.add(db_document)
        self.db.commit()
        self.db.refresh(db_document)
        return db_document

    def get_by_id(self, document_id: UUID) -> Optional[Document]:
        """Get document by ID"""
        return self.db.query(Document).filter(Document.id == document_id).first()

    def get_by_conversation_id(
        self, conversation_id: UUID, page: int = 1, page_size: int = 20
    ) -> Tuple[List[Document], int]:
        """Get paginated documents for a conversation with total count"""
        skip = (page - 1) * page_size
        
        query = self.db.query(Document).filter(
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

    def update(self, document_id: UUID, update_data: DocumentUpdate) -> Optional[Document]:
        """Update document with new data"""
        db_document = self.get_by_id(document_id)
        if not db_document:
            return None

        update_dict = update_data.model_dump(exclude_unset=True)

        for key, value in update_dict.items():
            setattr(db_document, key, value)

        self.db.commit()
        self.db.refresh(db_document)
        return db_document

    def delete(self, document_id: UUID) -> bool:
        """Delete a document"""
        db_document = self.get_by_id(document_id)
        if not db_document:
            return False

        self.db.delete(db_document)
        self.db.commit()
        return True

    def get_by_status(
        self, status: int, page: int = 1, page_size: int = 20
    ) -> Tuple[List[Document], int]:
        """Get paginated documents by status with total count"""
        skip = (page - 1) * page_size
        
        query = self.db.query(Document).filter(Document.status == status)
        
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
        return self.db.query(Document).count()

    def count_by_conversation(self, conversation_id: UUID) -> int:
        """Count documents for a conversation"""
        return (
            self.db.query(Document)
            .filter(Document.conversation_id == conversation_id)
            .count()
        )