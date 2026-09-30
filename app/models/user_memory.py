"""Model for agent-editable user memory items."""

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class UserMemory(Base):
    """Persistent user memory entry written by an agent on behalf of a user."""

    __tablename__ = "user_memories"
    # The recall index from migration 9778bb07ea35 (user + project scope, live
    # rows). Declared here so autogenerate no longer proposes dropping it.
    __table_args__ = (
        Index("ix_user_memories_user_project", "user_id", "project_id", "deleted_at"),
        # Serves the ``SET NULL`` from projects, which the recall index cannot:
        # it leads with user_id.
        Index("ix_user_memories_project_id", "project_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    # NULL means "global": saved from a conversation that belongs to no
    # project. Recall matches this project OR NULL, so a project sees its own
    # memories plus the global ones and never another project's.
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="SET NULL"),
        nullable=True,
    )
    content = Column(Text, nullable=False)
    source = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", backref="memories")

    def __repr__(self) -> str:
        return f"<UserMemory(id={self.id}, user_id={self.user_id}, project_id={self.project_id})>"
