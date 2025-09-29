from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from app.models.base import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False
    )
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    status = Column(
        Integer, nullable=False, default=1
    )  # 1=processing, 2=ready, 3=failed
    upload_time = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    # Relationships
    conversation = relationship("Conversation", back_populates="documents")
