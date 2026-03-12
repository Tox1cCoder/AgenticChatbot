from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from app.models.enums import MessageRole
from app.utils.case_conversion import (
    to_camel_case as to_camel,
    convert_dict_keys_to_snake_case,
)
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
    model_config_field: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="modelConfig",
        description="""Optional per-message model configuration for provider/model selection.
        
        Structure:
        {
          "all": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.7},
          "chat": null,      # Optional per-agent override
          "rag": null,       # Optional per-agent override
          "search": null,    # Optional per-agent override
          "planning": null   # Optional per-agent override
        }
        
        If 'all' is set, applies to all agents unless per-agent override is present.
        If omitted, uses system defaults (Gemini).
        """,
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
    content: str = Field(..., description="Message content")
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
              "chunks": [
                {
                  "chunk_index": 0,
                  "score": 0.85,
                  "character_count": 1500,
                  "content": "The actual chunk text content...",
                  "page_number": 5
                }
              ],
              "total_chunks": 1,
              "avg_score": 0.85
            }
          ]
        - chunks_retrieved: Total number of chunks retrieved
        - documents_found: Number of unique documents found
        - citations: Legacy flat list of all chunk citations (backward compatibility)
        - images: List of document images included in the response for visual reference
          [
            {
              "data": "base64_encoded_image_data",
              "mime": "image/jpeg",
              "name": "Image description or caption",
              "page_number": 5,
              "caption": "Original image caption"
            }
          ]
        - has_images: Boolean indicating if images were included
        - images_count: Number of images included

        For canvas responses, includes:
        - canvas_artifact: Self-contained renderable code artifact
          {
            "content":  "<full self-contained HTML / SVG document>",
            "language": "html" | "svg" | "react",
            "title":    "Short human-readable title",
            "editable": true
          }
        """,
    )
    feedback: Optional[FeedbackRead] = Field(
        default=None, description="Feedback for this message (when requested)"
    )
    interrupt: Optional[InterruptResponse] = Field(
        default=None,
        description="Interrupt information when tool execution requires human approval",
    )
    suggested_questions: Optional[List[str]] = Field(
        default=None,
        description="0-3 follow-up question suggestions for continuing the conversation",
    )

    @model_validator(mode="after")
    def _populate_from_metadata(self):
        # Populate fields from message_metadata for persisted messages
        if not isinstance(self.message_metadata, dict):
            return self

        # Populate interrupt from metadata
        if self.interrupt is None:
            payload = self.message_metadata.get("interrupt")
            if payload:
                try:
                    self.interrupt = (
                        payload
                        if isinstance(payload, InterruptResponse)
                        else InterruptResponse.model_validate(payload)
                    )
                except Exception:
                    pass

        # Populate suggested_questions from metadata
        if self.suggested_questions is None:
            suggestions = self.message_metadata.get("suggested_questions")
            if isinstance(suggestions, list):
                self.suggested_questions = [
                    s for s in suggestions if isinstance(s, str)
                ]

        return self


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

    @field_validator("decisions", mode="before")
    @classmethod
    def _normalize_decisions_keys(cls, value):
        # Accept both camelCase and snake_case decision payloads.
        if isinstance(value, list):
            normalized = []
            for v in value:
                if isinstance(v, dict):
                    v = convert_dict_keys_to_snake_case(v)
                normalized.append(v)
            return normalized
        return value

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
    content: str = Field(..., description="Message content")
    message_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Message metadata including persona used"
    )


class StopGenerationRequest(BaseModel):
    """Request to stop an in-flight streaming generation."""

    conversation_id: UUID = Field(..., description="Conversation ID")
    user_message_id: UUID = Field(
        ..., description="User message ID from the user_message_created SSE event"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class StopGenerationResponse(BaseModel):
    """Response from the stop generation endpoint."""

    status: str = Field(
        ..., description="'cancelled' or 'not_inflight'"
    )
    message: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Persisted assistant message (partial or final), if available",
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
