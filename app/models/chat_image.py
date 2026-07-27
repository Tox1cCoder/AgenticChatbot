"""Model for chat image bytes offloaded out of message metadata."""

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class ChatImage(Base):
    """A single stored chat image (user attachment or generated image).

    Bytes live on content-addressed disk under ``storage_path``; the message
    metadata only carries an ``image_id`` reference. Ownership rows are
    deduplicated per ``(user_id, sha256)``: identical content re-stored by the
    same owner (e.g. a resumed run re-persisting a generated image) reuses the
    existing row rather than inserting a duplicate, so one row may be referenced
    from several messages/conversations and keeps the first conversation's
    ``conversation_id``. Distinct users still get distinct rows.
    """

    __tablename__ = "chat_images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True
    )
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    sha256 = Column(String(64), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    content_type = Column(String(128), nullable=False, default="image/png")
    storage_path = Column(String(1024), nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    conversation = relationship("Conversation", backref="chat_images")
    user = relationship("User", backref="chat_images")

    def __repr__(self) -> str:
        return f"<ChatImage(id={self.id}, sha256='{self.sha256[:12]}', size={self.size_bytes})>"
