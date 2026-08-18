"""Repository for DocumentImage operations."""

from typing import Any
from uuid import UUID

from sqlalchemy import asc
from sqlalchemy.orm import joinedload

from app.models.conversation import Conversation
from app.models.document import Document
from app.models.document_image import DocumentImage
from app.schemas.document_image import DocumentImageCreate, DocumentImageUpdate


def _coerce_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


class DocumentImageRepository:
    def __init__(self, session_factory: callable):
        self.session_factory = session_factory

    def create(self, image_data: DocumentImageCreate) -> DocumentImage:
        with self.session_factory() as db:
            db_image = DocumentImage(
                document_id=image_data.document_id,
                chunk_id=image_data.chunk_id,
                image_path=image_data.image_path,
                image_caption=image_data.image_caption,
                page_number=image_data.page_number,
                mime_type=image_data.mime_type,
                bbox=image_data.bbox,
                section_path=list(image_data.section_path or []),
                content_sha256=image_data.content_sha256,
            )
            db.add(db_image)
            db.commit()
            db.refresh(db_image)
            return db_image

    def get_by_id(self, image_id: UUID) -> DocumentImage | None:
        with self.session_factory() as db:
            return db.query(DocumentImage).filter(DocumentImage.id == image_id).first()

    def get_by_document_id(self, document_id: UUID) -> list[DocumentImage]:
        with self.session_factory() as db:
            return (
                db.query(DocumentImage)
                .filter(DocumentImage.document_id == document_id)
                .order_by(asc(DocumentImage.page_number))
                .all()
            )

    def get_by_document_for_scope(
        self,
        document_id: Any,
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[DocumentImage]:
        """Return images for ``document_id`` only when the parent ``Document``
        matches the given server-context filters.

        Auth filtering happens in the SQL ``WHERE`` clause via a join on
        ``Document``. Returns an empty list when ``document_id`` or
        ``conversation_id`` is not a valid UUID, or when filters don't match.
        """
        if not user_id or not conversation_id:
            return []

        doc_uuid = _coerce_uuid(document_id)
        if doc_uuid is None:
            return []

        conv_uuid = _coerce_uuid(conversation_id)
        if conv_uuid is None:
            return []

        with self.session_factory() as db:
            query = (
                db.query(DocumentImage)
                .join(Document, DocumentImage.document_id == Document.id)
                .filter(DocumentImage.document_id == doc_uuid)
            )
            query = query.filter(Document.conversation_id == conv_uuid)
            query = query.join(
                Conversation, Document.conversation_id == Conversation.id
            ).filter(Conversation.owner_id == user_id)
            return query.order_by(asc(DocumentImage.page_number)).all()

    def get_by_id_for_scope(
        self,
        image_id: Any,
        *,
        user_id: str | None,
        conversation_id: str | None,
    ) -> DocumentImage | None:
        image_uuid = _coerce_uuid(image_id)
        conv_uuid = _coerce_uuid(conversation_id)
        if image_uuid is None or conv_uuid is None or not user_id:
            return None
        with self.session_factory() as db:
            return (
                db.query(DocumentImage)
                # Eager-load so callers (e.g. RAGRetriever.search_images) can
                # read image.document.filename after the session closes.
                .options(joinedload(DocumentImage.document))
                .join(Document, DocumentImage.document_id == Document.id)
                .join(Conversation, Document.conversation_id == Conversation.id)
                .filter(DocumentImage.id == image_uuid)
                .filter(Document.conversation_id == conv_uuid)
                .filter(Conversation.owner_id == user_id)
                .first()
            )

    def get_by_chunk_id_for_scope(
        self,
        chunk_id: Any,
        *,
        user_id: str | None,
        conversation_id: str | None,
    ) -> list[DocumentImage]:
        chunk_uuid = _coerce_uuid(chunk_id)
        conv_uuid = _coerce_uuid(conversation_id)
        if chunk_uuid is None or conv_uuid is None or not user_id:
            return []
        with self.session_factory() as db:
            return (
                db.query(DocumentImage)
                .join(Document, DocumentImage.document_id == Document.id)
                .join(Conversation, Document.conversation_id == Conversation.id)
                .filter(DocumentImage.chunk_id == chunk_uuid)
                .filter(Document.conversation_id == conv_uuid)
                .filter(Conversation.owner_id == user_id)
                .order_by(asc(DocumentImage.page_number))
                .all()
            )

    def get_by_chunk_id(self, chunk_id: UUID) -> list[DocumentImage]:
        with self.session_factory() as db:
            return db.query(DocumentImage).filter(DocumentImage.chunk_id == chunk_id).all()

    def update(self, image_id: UUID, update_data: DocumentImageUpdate) -> DocumentImage | None:
        with self.session_factory() as db:
            db_image = db.query(DocumentImage).filter(DocumentImage.id == image_id).first()
            if not db_image:
                return None

            update_dict = update_data.model_dump(exclude_unset=True)
            for field, value in update_dict.items():
                setattr(db_image, field, value)

            db.commit()
            db.refresh(db_image)
            return db_image

    def delete(self, image_id: UUID) -> bool:
        with self.session_factory() as db:
            db_image = db.query(DocumentImage).filter(DocumentImage.id == image_id).first()
            if not db_image:
                return False

            db.delete(db_image)
            db.commit()
            return True

    def delete_by_document_id(self, document_id: UUID) -> int:
        with self.session_factory() as db:
            images = db.query(DocumentImage).filter(DocumentImage.document_id == document_id).all()
            count = len(images)

            for image in images:
                db.delete(image)

            db.commit()
            return count

    def get_image_paths_by_document_id(self, document_id: UUID) -> list[str]:
        with self.session_factory() as db:
            images = (
                db.query(DocumentImage.image_path)
                .filter(DocumentImage.document_id == document_id)
                .all()
            )
            return [img[0] for img in images]
