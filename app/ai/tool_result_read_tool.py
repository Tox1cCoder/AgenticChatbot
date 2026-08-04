"""Internal tool that reads an offloaded tool result the model was shown a preview of.

Scoping is deliberately blunt: a blob is readable only from the conversation and
user that produced it, and every failure — unknown id, wrong conversation,
missing context, malformed id — returns the same not-found payload so the tool
cannot be used to probe for other conversations' results.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from uuid import UUID

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..core.config import settings
from .tool_context import get_tool_context

logger = logging.getLogger(__name__)

_DESCRIPTION = (
    "Read the full text of a large tool result that was offloaded and shown to you "
    "only as a preview. Pass the blob_id printed in the offload notice. Returns a "
    "bounded slice plus next_offset; call again with that offset to continue. Use "
    "this instead of repeating a search whose result was truncated."
)

_NOT_FOUND = {
    "status": "error",
    "error_type": "not_found",
    "retryable": False,
    "hint": (
        "No offloaded result with that blob_id exists in this conversation. Use the "
        "blob_id exactly as printed in the offload notice."
    ),
}


class ReadToolResultInput(BaseModel):
    blob_id: str = Field(description="The blob_id printed in the offload notice.")
    offset: int = Field(default=0, ge=0, description="Character offset to read from.")
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Characters to return. Clamped to the configured maximum.",
    )


def create_read_tool_result_tool(
    *,
    repository: Any | None = None,
    service: Any | None = None,
) -> StructuredTool:
    """Build the ``read_tool_result`` tool, resolving DI lazily when not injected."""

    async def _read(blob_id: str, offset: int = 0, limit: int | None = None) -> str:
        context = get_tool_context()
        identity = _scoped_identity(blob_id, context.user_id, context.conversation_id)
        if identity is None:
            return json.dumps(_NOT_FOUND)
        resolved_repository, resolved_service = _resolve(repository, service)
        if resolved_repository is None or resolved_service is None:
            return json.dumps(_NOT_FOUND)

        parsed_blob_id, user_uuid, conversation_uuid = identity
        record = await asyncio.to_thread(
            resolved_repository.get_for_user_and_conversation,
            parsed_blob_id,
            user_uuid,
            conversation_uuid,
        )
        if record is None:
            return json.dumps(_NOT_FOUND)
        text = await asyncio.to_thread(resolved_service.read_text, record)

        cap = max(1, int(getattr(settings, "tool_result_read_max_chars", 8000)))
        window = cap if limit is None else max(1, min(int(limit), cap))
        start = max(0, int(offset))
        chunk = text[start : start + window]
        end = start + len(chunk)
        return json.dumps(
            {
                "blob_id": str(parsed_blob_id),
                "offset": start,
                "returned_chars": len(chunk),
                "total_chars": len(text),
                "next_offset": end if end < len(text) else None,
                "content": chunk,
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        coroutine=_read,
        name="read_tool_result",
        description=_DESCRIPTION,
        args_schema=ReadToolResultInput,
        metadata={
            "tool_origin": "internal",
            "qualified_tool_id": "internal::read_tool_result",
        },
    )


def _scoped_identity(
    blob_id: str,
    user_id: str | None,
    conversation_id: str | None,
) -> tuple[UUID, UUID, UUID] | None:
    if not user_id or not conversation_id:
        return None
    try:
        return (
            UUID(str(blob_id).strip()),
            UUID(str(user_id)),
            UUID(str(conversation_id)),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _resolve(repository: Any | None, service: Any | None) -> tuple[Any | None, Any | None]:
    if repository is not None and service is not None:
        return repository, service
    try:
        from ..core.container import Container

        container = Container()
        return (
            repository or container.tool_result_blob_repository(),
            service or container.tool_result_blob_service(),
        )
    except Exception as exc:  # pragma: no cover - DI unavailable in some contexts
        logger.debug("Tool result blob access unavailable: %s", exc)
        return None, None
