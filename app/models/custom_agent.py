"""Models for per-user custom agents and their per-conversation attachments."""

import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.models.base import Base


class CustomAgent(Base):
    """
    A user-owned custom agent definition.

    Runtime identity (``custom_agent:<id>``) is derived from ``id`` at request
    time and is never persisted here; this table stores only the model identity
    (``provider_type``/``model``) plus the restricted tool/skill allowlists.
    """

    __tablename__ = "custom_agents"
    __table_args__ = (
        # Per-user uniqueness of the live (non-deleted) slug. A soft-deleted
        # agent frees its slug so the user can recreate one with the same name.
        Index(
            "uq_custom_agents_owner_slug_active",
            "owner_id",
            "slug",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
        Index("ix_custom_agents_owner_deleted", "owner_id", "deleted_at"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    name = Column(String(255), nullable=False)
    slug = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    prompt = Column(Text, nullable=False)

    # Model identity (validated against the model catalog / credentials).
    provider_type = Column(String(64), nullable=False)
    model = Column(String(255), nullable=False)
    temperature = Column(Float, nullable=True)
    reasoning_effort = Column(String(32), nullable=True)

    # Restricted allowlists: exact tool and skill references (see schemas).
    tool_refs = Column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    skill_refs = Column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))

    enabled = Column(Boolean, nullable=False, default=True, server_default=text("true"))

    def __repr__(self) -> str:
        return (
            f"<CustomAgent(id={self.id}, slug='{self.slug}', "
            f"owner_id={self.owner_id}, deleted={self.deleted_at is not None})>"
        )


class ConversationCustomAgent(Base):
    """Attachment row linking a custom agent to a single conversation."""

    __tablename__ = "conversation_custom_agents"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "custom_agent_id",
            name="uq_conversation_custom_agents_conv_agent",
        ),
        Index("ix_conversation_custom_agents_owner_conv", "owner_id", "conversation_id"),
        Index("ix_conversation_custom_agents_custom_agent_id", "custom_agent_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    custom_agent_id = Column(UUID(as_uuid=True), ForeignKey("custom_agents.id"), nullable=False)
    agent_order = Column(Integer, nullable=False, default=0, server_default=text("0"))

    def __repr__(self) -> str:
        return (
            f"<ConversationCustomAgent(conversation_id={self.conversation_id}, "
            f"custom_agent_id={self.custom_agent_id}, order={self.agent_order})>"
        )
