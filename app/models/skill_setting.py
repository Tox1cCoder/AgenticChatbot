"""Model for tracking per-user skill enable/disable settings."""

import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class SkillSetting(Base):
    """
    Per-user skill enable/disable setting.

    Skills are per-device, but the enable/disable preference is persisted
    per-user to allow consistent behavior across devices for the same user.
    """

    __tablename__ = "skill_settings"
    __table_args__ = (
        UniqueConstraint("user_id", "skill_name", name="uq_skill_settings_user_skill"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )

    # Ownership
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    # Skill reference
    skill_name = Column(String(255), nullable=False, index=True)

    # State
    enabled = Column(Boolean, nullable=False, default=True)

    # Relationships
    user = relationship("User", backref="skill_settings")

    def __repr__(self) -> str:
        return (
            f"<SkillSetting(id={self.id}, skill_name='{self.skill_name}', "
            f"enabled={self.enabled}, user_id={self.user_id})>"
        )
