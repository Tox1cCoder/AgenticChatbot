from __future__ import annotations

import contextlib
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.rich_response import sanitize_public_rich_metadata
from app.models.enums import MessageRole
from app.schemas.feedback import FeedbackRead
from app.schemas.workflow import InterruptDecision, InterruptResponse
from app.utils.case_conversion import (
    convert_dict_keys_to_snake_case,
)
from app.utils.case_conversion import (
    to_camel_case as to_camel,
)


class MessageCreate(BaseModel):
    conversation_id: UUID = Field(..., description="Conversation ID this message belongs to")
    content: str = Field(..., min_length=1, description="Message content")
    device_id: UUID | None = Field(
        default=None,
        description="Optional client device ID used for device-local tool dispatch.",
    )
    role: MessageRole = Field(
        default=MessageRole.user, description="Message role: user=1, assistant=2"
    )
    attachments: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "Optional image attachments with structure {name: str, mime: str, data: str (base64)}"
        ),
    )
    model_config_field: dict[str, Any] | None = Field(
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
    inline_rich_response_v1: bool = Field(
        default=False,
        description=(
            "When true, the client declares it can render the inline rich-response "
            "v1 contract (HTML-comment marker syntax + `rich_items` registry). "
            "Required to receive marker-bearing assistant content over AI SDK "
            "endpoints and `data-rich-items` transient events."
        ),
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MessageUpdate(BaseModel):
    content: str | None = Field(None, min_length=1, description="Message content")

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
    conversation_id: UUID
    sender: int = Field(..., description="Message sender: 1=user, 2=assistant")
    content: str = Field(..., description="Message content")
    message_metadata: dict[str, Any] | None = Field(
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
            "title":    "Short human-readable title"
          }
        """,
    )
    feedback: FeedbackRead | None = Field(
        default=None, description="Feedback for this message (when requested)"
    )
    interrupt: InterruptResponse | None = Field(
        default=None,
        description="Interrupt information when tool execution requires human approval",
    )
    suggested_questions: list[str] | None = Field(
        default=None,
        description="0-3 follow-up question suggestions for continuing the conversation",
    )

    @model_validator(mode="after")
    def _populate_from_metadata(self):
        # Populate fields from message_metadata for persisted messages
        if not isinstance(self.message_metadata, dict):
            return self
        self.message_metadata = sanitize_public_rich_metadata(self.message_metadata)

        # Populate interrupt from metadata
        if self.interrupt is None:
            payload = self.message_metadata.get("interrupt")
            if payload:
                with contextlib.suppress(Exception):
                    self.interrupt = (
                        payload
                        if isinstance(payload, InterruptResponse)
                        else InterruptResponse.model_validate(payload)
                    )

        # Populate suggested_questions from metadata
        if self.suggested_questions is None:
            suggestions = self.message_metadata.get("suggested_questions")
            if isinstance(suggestions, list):
                self.suggested_questions = [s for s in suggestions if isinstance(s, str)]

        return self


class InterruptResumeRequest(BaseModel):
    """Request to resume execution after handling interrupts."""

    thread_id: str = Field(..., description="Thread ID from the interrupt response")
    interrupt_id: str = Field(
        ..., min_length=1, description="Interrupt ID returned from the HITL middleware"
    )
    conversation_id: UUID = Field(..., description="Conversation ID")
    device_id: UUID | None = Field(
        default=None,
        description="Optional client device ID used to validate sidecar-scoped interrupt resumes.",
    )
    decisions: list[InterruptDecision] = Field(..., description="Approval/rejection/edit decisions")
    inline_rich_response_v1: bool = Field(
        default=False,
        description=(
            "When true, the client requests the inline rich-response v1 contract "
            "for the resumed response (marker-bearing content + `data-rich-items`)."
        ),
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
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
    conversation_id: UUID
    sender: int = Field(..., description="Message sender: 1=user, 2=assistant, 3=system")
    content: str = Field(..., description="Message content")
    message_metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="Message metadata including persona used"
    )


class StopGenerationRequest(BaseModel):
    """Request to stop an in-flight streaming generation.

    Two ways to name the turn, because the durable lifecycle arrived after the
    endpoint did. ``generation_id`` is canonical and comes from ``run_start``;
    ``user_message_id`` is the turn-scoped form, resolved through the logical
    turn so a stale id stops nothing rather than the turn running now.

    ``expected_version`` fences the command (R5). Without it a delayed replay of
    a Stop issued against one epoch executes against whatever epoch is running
    when it lands. It is optional only for clients that predate ``run_start``;
    a client that has a version must send it.
    """

    conversation_id: UUID = Field(..., description="Conversation ID")
    generation_id: UUID | None = Field(
        default=None, description="Generation ID from the run_start SSE event"
    )
    user_message_id: UUID | None = Field(
        default=None, description="User message ID from the user_message_created SSE event"
    )
    idempotency_key: str | None = Field(
        default=None,
        min_length=8,
        max_length=160,
        description="Replay key; the same key returns the first attempt's recorded result",
    )
    expected_version: int | None = Field(
        default=None, ge=1, description="Lifecycle version this command was issued against"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    @model_validator(mode="after")
    def _one_identity(self) -> StopGenerationRequest:
        if self.generation_id is None and self.user_message_id is None:
            raise ValueError("either generationId or userMessageId is required")
        return self


class ContinueGenerationRequest(BaseModel):
    """Request to continue a paused or stopped generation.

    ``continuation_id`` is single-use, which is what stops a replayed Continue
    from opening a second epoch on the same answer. Both it and the version come
    from the ``continuation_available`` event.
    """

    conversation_id: UUID = Field(..., description="Conversation ID")
    generation_id: UUID = Field(..., description="Generation ID to continue")
    continuation_id: UUID = Field(
        ..., description="Single-use continuation ID from continuation_available"
    )
    idempotency_key: str = Field(
        ..., min_length=8, max_length=160, description="Replay key for this Continue"
    )
    expected_version: int = Field(
        ..., ge=1, description="Lifecycle version this command was issued against"
    )
    inline_rich_response_v1: bool = False

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class GenerationSnapshotResponse(BaseModel):
    """The lifecycle state every transport publishes.

    Deliberately omits the checkpoint thread (a resume handle) and the budget
    and research-accounting blobs (server bookkeeping). ``version`` is here
    because a client cannot issue a fenced command without it.
    """

    generation_id: UUID
    logical_turn_id: str
    conversation_id: UUID
    status: str
    version: int
    execution_epoch: int
    continuation_id: UUID | None = None
    continuation_available: bool = False
    continuation_block_reason: str | None = None
    assistant_message_id: UUID | None = None
    terminal_reason: str | None = None

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class StopGenerationResponse(BaseModel):
    """Response from the stop generation endpoint."""

    status: str = Field(..., description="'cancelled', 'stop_requested' or 'not_inflight'")
    message: dict[str, Any] | None = Field(
        default=None,
        description="Persisted assistant message (partial or final), if available",
    )
    generation: GenerationSnapshotResponse | None = Field(
        default=None,
        description="Durable lifecycle snapshot; the field a client should read",
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
