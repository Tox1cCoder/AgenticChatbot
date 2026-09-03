"""
Agent model config database model.

Stores persistent per-agent provider/model selection per user.
"""

import uuid

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class AgentModelConfig(Base):
    """
    Persistent per-agent model configuration for a user.

    One row per (user_id, agent_key).
    """

    __tablename__ = "agent_model_configs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    agent_key = Column(Text, nullable=False)  # 'chat'|'rag'|'search'|'planning'
    provider_type = Column(Text, nullable=False)  # 'gemini'|'openai' (for now)
    model = Column(Text, nullable=False)
    allow_custom_model = Column(Boolean, default=False, nullable=False)
    temperature = Column(Float, nullable=True)
    # String(32), not Text: migration e8f9a0b1c2d3 shipped VARCHAR(32) and the
    # values are short provider effort levels. Declaring Text here made the
    # model permanently disagree with the live column.
    reasoning_effort = Column(String(32), nullable=True)

    user = relationship("User", back_populates="agent_model_configs")

    __table_args__ = (
        Index("idx_agent_model_configs_user_id", "user_id"),
        Index("idx_agent_model_configs_user_agent", "user_id", "agent_key", unique=True),
    )

    def __repr__(self) -> str:
        return (
            f"<AgentModelConfig(id={self.id}, user_id={self.user_id}, "
            f"agent_key={self.agent_key}, provider_type={self.provider_type}, model={self.model}, "
            f"allow_custom_model={self.allow_custom_model})>"
        )
