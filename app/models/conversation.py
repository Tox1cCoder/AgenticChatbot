from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import Column, String, ForeignKey, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base

# Create independent base for this model
Base = declarative_base()


class Conversation(Base):
    __tablename__ = "conversations"

    # Independent attribute declarations - no inheritance
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    owner_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    title = Column(String(255), nullable=False)

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title='{self.title}', owner_id={self.owner_id})>"

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, owner_id={self.owner_id}, title='{self.title}')>"
