from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict
from app.models.enums import MessageRole
from app.utils.case_conversion import to_camel_case as to_camel
from app.schemas.feedback import FeedbackRead
from app.ai.schemas import InterruptResponse, InterruptDecision


class MessageCreate(BaseModel):
    conversation_id: UUID = Field(
        ..., description="Conversation ID this message belongs to"
    )
    content: str = Field(..., min_length=1, description="Message content")
    role: MessageRole = Field(
        default=MessageRole.user, description="Message role: user=1, assistant=2"
    )
    attachments: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional image attachments with structure {name: str, mime: str, data: str (base64)}",
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MessageUpdate(BaseModel):
    content: Optional[str] = Field(None, min_length=1, description="Message content")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MessageRead(BaseModel):
    model_config = ConfigDict(
        from_attributes=True, alias_generator=to_camel, populate_by_name=True
    )

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    conversation_id: UUID
    sender: int = Field(..., description="Message sender: 1=user, 2=assistant")
    content: str = Field(..., min_length=1, description="Message content")
    message_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="""Message metadata including persona used and RAG citations.
        
        For RAG responses, includes:
        - documents_cited: List of documents referenced, grouped by source file
          [
            {
              "document_id": "uuid",
              "source": "report.pdf",
              "document_number": 1,
              "chunks": [{"chunk_index": 0, "score": 0.85}],
              "total_chunks": 1,
              "avg_score": 0.85
            }
          ]
        - chunks_retrieved: Total number of chunks retrieved
        - documents_found: Number of unique documents found
        - citations: Legacy flat list of all chunk citations (backward compatibility)
        """,
    )
    feedback: Optional[FeedbackRead] = Field(
        default=None, description="Feedback for this message (when requested)"
    )
    interrupt: Optional[InterruptResponse] = Field(
        default=None,
        description="Interrupt information when tool execution requires human approval",
    )


class InterruptResumeRequest(BaseModel):
    """Request to resume execution after handling interrupts."""

    thread_id: str = Field(..., description="Thread ID from the interrupt response")
    interrupt_id: Optional[str] = Field(
        default=None, description="Interrupt ID returned from the HITL middleware"
    )
    conversation_id: UUID = Field(..., description="Conversation ID")
    decisions: List[InterruptDecision] = Field(
        ..., description="Approval/rejection/edit decisions"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

class MessageInDB(BaseModel):
    model_config = ConfigDict(
        from_attributes=True, alias_generator=to_camel, populate_by_name=True
    )

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    conversation_id: UUID
    sender: int = Field(
        ..., description="Message sender: 1=user, 2=assistant, 3=system"
    )
    content: str = Field(..., min_length=1, description="Message content")
    message_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Message metadata including persona used"
    )
