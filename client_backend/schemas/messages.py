"""
Message-related schemas for the client backend.

These mirror server schemas where applicable for proxy consistency.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of a message sender."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageRequest(BaseModel):
    """Request to send a message."""

    conversation_id: str
    content: str
    attachments: list[str] = Field(default_factory=list)


class MessageResponse(BaseModel):
    """Response containing a message."""

    id: str
    conversation_id: str
    role: MessageRole
    content: str
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class StreamEventType(str, Enum):
    """Types of stream events."""

    TEXT_DELTA = "text_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    ERROR = "error"
    DONE = "done"


class StreamEvent(BaseModel):
    """Event in a streaming message response."""

    type: StreamEventType
    data: Any
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


class ConversationSummary(BaseModel):
    """Summary of a conversation."""

    id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationListResponse(BaseModel):
    """Response for listing conversations."""

    conversations: list[ConversationSummary]
    total: int
    page: int
    page_size: int
