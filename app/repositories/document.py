from collections.abc import Callable
from typing import Any
from uuid import UUID

from sqlalchemy import desc
from sqlalchemy.exc import IntegrityError, MultipleResultsFound

from app.core.exceptions.validation import DuplicateDocumentFilenameError
from app.models.conversation import Conversation
from app.models.document import Document
from app.repositories.session_transport import RepositorySessionMixin
from app.schemas.document import DocumentCreate, DocumentUpdate


class DocumentRepository(RepositorySessionMixin):
    """Repository for document operations"""

    def __init__(
        self,
        session_factory: callable,
        async_session_factory: Callable[[], Any] | None = None,
    ):
        """Initialize repository with session factory for dependency injection."""
        super().__init__(
            session_factory=session_factory,
            async_session_factory=async_session_factory,
        )

    def create(self, document_data: DocumentCreate) -> Document:
        """Create a new document.

        Translates the unique-constraint violation on
        ``(conversation_id, filename_key)`` into ``DuplicateDocumentFilenameError``
        so callers can present a per-file rejection without recovering from
        a raw SQLAlchemy error.
        """
        with self.session_factory() as db:
            db_document = Document(
                conversation_id=document_data.conversation_id,
                filename=document_data.filename,
                filename_key=document_data.filename_key,
                file_type=document_data.file_type,
                status=document_data.status,
            )
            db.add(db_document)
            try:
                db.commit()
            except IntegrityError as exc:
                db.rollback()
                raise DuplicateDocumentFilenameError(
                    detail=(
                        f"A document named '{document_data.filename}' already exists "
                        "in this conversation."
                    )
                ) from exc
            db.refresh(db_document)
            return db_document

    def filename_exists_in_conversation(self, conversation_id: UUID, filename_key: str) -> bool:
        """Return True if a document with this normalized filename already exists."""
        with self.session_factory() as db:
            query = db.query(Document).filter(
                Document.conversation_id == conversation_id,
                Document.filename_key == filename_key,
            )
            return bool(db.query(query.exists()).scalar())

    def get_by_conversation_and_filename_key(
        self, conversation_id: UUID, filename_key: str
    ) -> Document | None:
        """Return the existing document for a (conversation, filename_key) pair."""
        with self.session_factory() as db:
            return (
                db.query(Document)
                .filter(
                    Document.conversation_id == conversation_id,
                    Document.filename_key == filename_key,
                )
                .first()
            )

    def get_by_id(self, document_id: UUID) -> Document | None:
        """Get document by ID"""
        with self.session_factory() as db:
            return db.query(Document).filter(Document.id == document_id).first()

    def get_by_processing_task_id(self, task_id: str) -> Document | None:
        """Get document by its Celery processing task ID."""
        with self.session_factory() as db:
            query = db.query(Document).filter(Document.processing_task_id == task_id)
            try:
                return query.one_or_none()
            except MultipleResultsFound:
                return None

    def set_processing_task_id(self, document_id: UUID, task_id: str) -> Document | None:
        """Persist the Celery processing task ID for a document."""
        with self.session_factory() as db:
            db_document = db.query(Document).filter(Document.id == document_id).first()
            if not db_document:
                return None

            db_document.processing_task_id = task_id
            db.commit()
            db.refresh(db_document)
            return db_document

    def get_by_conversation_id(
        self, conversation_id: UUID, page: int = 1, page_size: int = 20
    ) -> tuple[list[Document], int]:
        """Get paginated documents for a conversation with total count"""
        with self.session_factory() as db:
            skip = (page - 1) * page_size

            query = db.query(Document).filter(Document.conversation_id == conversation_id)

            total = query.count()
            documents = (
                query.order_by(desc(Document.upload_time)).offset(skip).limit(page_size).all()
            )

            return documents, total

    def update(self, document_id: UUID, update_data: DocumentUpdate) -> Document | None:
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
    ) -> tuple[list[Document], int]:
        """Get paginated documents by status with total count"""
        with self.session_factory() as db:
            skip = (page - 1) * page_size

            query = db.query(Document).filter(Document.status == status)

            total = query.count()
            documents = (
                query.order_by(desc(Document.upload_time)).offset(skip).limit(page_size).all()
            )

            return documents, total

    def count_all(self) -> int:
        """Count total documents"""
        with self.session_factory() as db:
            return db.query(Document).count()

    @staticmethod
    def _count_by_conversation_in_session(db, conversation_id: UUID) -> int:
        """Counting body shared by both transports.

        This repository queries directly rather than through a strategy, so the
        query itself lives here.
        """
        return db.query(Document).filter(Document.conversation_id == conversation_id).count()

    def count_by_conversation(self, conversation_id: UUID) -> int:
        """Count documents for a conversation"""
        return self._run(lambda db: self._count_by_conversation_in_session(db, conversation_id))

    async def acount_by_conversation(self, conversation_id: UUID) -> int:
        """Async twin of :meth:`count_by_conversation`."""
        return await self._arun(
            lambda db: self._count_by_conversation_in_session(db, conversation_id)
        )

    # Authorization helpers
    def exists(self, document_id: UUID) -> bool:
        """Check if a document exists by ID."""
        with self.session_factory() as db:
            return db.query(Document).filter(Document.id == document_id).first() is not None

    def user_owns_document(self, user_id: UUID, document_id: UUID) -> bool:
        """Check if a user owns the document via its conversation ownership."""
        with self.session_factory() as db:
            q = (
                db.query(Document)
                .join(Conversation, Conversation.id == Document.conversation_id)
                .filter(Document.id == document_id, Conversation.owner_id == user_id)
            )
            return db.query(q.exists()).scalar()
