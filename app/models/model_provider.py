"""
Model Provider database model.

Stores encrypted API keys for different AI providers per user, enabling multi-provider model configuration.
"""

import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class ModelProvider(Base):
    """
    Stores user-specific API keys for AI providers.

    Keys are encrypted at rest using Fernet symmetric encryption.
    """

    __tablename__ = "model_providers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    provider_type = Column(Text, nullable=False)  # 'gemini', 'openai', 'anthropic'
    api_key_encrypted = Column(Text, nullable=False)
    is_default = Column(Boolean, default=False, nullable=False)

    # Store provider-specific settings (e.g., organization ID, custom endpoints)
    provider_metadata = Column(JSONB, nullable=True, default=dict)

    # Relationships
    user = relationship("User", back_populates="model_providers")

    # Indexes
    __table_args__ = (
        Index("idx_model_providers_user_id", "user_id"),
        Index(
            "idx_model_providers_user_type",
            "user_id",
            "provider_type",
            unique=True,
            postgresql_where=(deleted_at.is_(None)),
        ),
        Index(
            "idx_model_providers_user_default",
            "user_id",
            "is_default",
            postgresql_where=(is_default),
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelProvider(id={self.id}, user_id={self.user_id}, provider={self.provider_type})>"
        )
