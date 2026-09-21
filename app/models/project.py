"""Models for user-owned projects and their default custom agents."""

import uuid

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import Base


class Project(Base):
    """A container grouping conversations under one set of instructions.

    Owner-scoped and soft-deleted, mirroring :class:`CustomAgent`. There is
    deliberately no slug and no uniqueness on ``name``: a project is addressed
    by id everywhere, and duplicate names are allowed.
    """

    __tablename__ = "projects"
    __table_args__ = (Index("ix_projects_owner_deleted", "owner_id", "deleted_at"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    instructions = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name='{self.name}', owner_id={self.owner_id})>"


class ProjectCustomAgent(Base):
    """A custom agent in a project's default set.

    Structurally identical to :class:`ConversationCustomAgent` so that seeding
    a conversation is a straight row copy rather than a translation.
    """

    __tablename__ = "project_custom_agents"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "custom_agent_id",
            name="uq_project_custom_agents_project_agent",
        ),
        Index("ix_project_custom_agents_owner_project", "owner_id", "project_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    custom_agent_id = Column(UUID(as_uuid=True), ForeignKey("custom_agents.id"), nullable=False)
    agent_order = Column(Integer, nullable=False, default=0, server_default=text("0"))

    def __repr__(self) -> str:
        return (
            f"<ProjectCustomAgent(project_id={self.project_id}, "
            f"custom_agent_id={self.custom_agent_id}, order={self.agent_order})>"
        )
