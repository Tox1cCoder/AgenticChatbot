"""Owned references to selected third-party rich images."""

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, LargeBinary, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class WebImageReference(Base):
    """A user-owned, opaque reference to one selected upstream image.

    New provider-selected references store retrieval metadata without fetching
    the upstream during registration. On render, ``WebImageService`` fetches and
    validates the remote bytes before serving them. ``content`` remains nullable
    for records that already carry cached bytes.
    """

    __tablename__ = "web_image_references"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True
    )
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    upstream_url = Column(String(4096), nullable=False)
    expected_mime = Column(String(128), nullable=True)
    provider = Column(String(32), nullable=False)
    content = Column(LargeBinary, nullable=True)
    cached_width = Column(Integer, nullable=True)
    cached_height = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    conversation = relationship("Conversation", backref="web_image_references")
    user = relationship("User", backref="web_image_references")

    def __repr__(self) -> str:
        return f"<WebImageReference(id={self.id}, provider={self.provider!r})>"
