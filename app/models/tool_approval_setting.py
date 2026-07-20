"""Per-user Human-in-the-Loop approval setting (server-scoped or tool-scoped)."""

import uuid

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class ToolApprovalSetting(Base):
    """
    A per-user rule that gates tool calls behind human approval.

    ``scope_type`` is "server" (``scope_value`` = MCP server name, applies to all
    its tools) or "tool" (``scope_value`` = qualified tool id ``"<server>::<tool>"``
    or a bare tool name, overrides the server default). ``require_approval`` is the
    explicit decision for that scope.
    """

    __tablename__ = "tool_approval_settings"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "device_id",
            "tool_origin",
            "scope_type",
            "scope_value",
            name="uq_tool_approval_settings_user_device_origin_scope",
        ),
        CheckConstraint(
            "scope_type IN ('server', 'tool')",
            name="ck_tool_approval_settings_scope_type",
        ),
        CheckConstraint(
            "tool_origin IN ('client_mcp', 'client_skill')",
            name="ck_tool_approval_settings_tool_origin",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    device_id = Column(
        UUID(as_uuid=True),
        ForeignKey("client_devices.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    tool_origin = Column(String(32), nullable=False, index=True)
    scope_type = Column(String(16), nullable=False)  # "server" | "tool"
    scope_value = Column(String(512), nullable=False, index=True)
    require_approval = Column(Boolean, nullable=False, default=True)

    user = relationship("User", backref="tool_approval_settings")
    device = relationship("ClientDevice", backref="tool_approval_settings")

    def __repr__(self) -> str:
        return (
            f"<ToolApprovalSetting(user_id={self.user_id}, device_id={self.device_id}, "
            f"tool_origin='{self.tool_origin}', scope_type='{self.scope_type}', "
            f"scope_value='{self.scope_value}', require_approval={self.require_approval})>"
        )
