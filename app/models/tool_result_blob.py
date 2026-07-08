"""Model for offloaded tool result blob storage."""

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class ToolResultBlob(Base):
    """
    Persistent record for large tool outputs that have been offloaded
    out of the model-visible ToolMessage.

    The full payload lives in the ``content`` column. ``storage_path`` is a
    legacy field: rows created before content moved to Postgres keep their
    payload on disk relative to the configured storage root. The model only
    sees a preview plus a blob_id pointer it can use later to read the full
    result.
    """

    __tablename__ = "tool_result_blobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True
    )
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    tool_call_id = Column(String(255), nullable=True, index=True)
    tool_name = Column(String(255), nullable=False, index=True)

    storage_path = Column(String(1024), nullable=True)
    content = Column(Text, nullable=True)
    sha256 = Column(String(64), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    content_type = Column(String(128), nullable=False, default="text/plain")

    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    conversation = relationship("Conversation", backref="tool_result_blobs")
    user = relationship("User", backref="tool_result_blobs")

    def __repr__(self) -> str:
        return (
            f"<ToolResultBlob(id={self.id}, tool_name='{self.tool_name}', "
            f"size_bytes={self.size_bytes})>"
        )
