"""Internal tool that reads an offloaded tool result the model was shown a preview of.

Scoping is deliberately blunt: a blob is readable only from the conversation and
user that produced it. Unknown id, wrong conversation, missing context, a
malformed id and a record whose text cannot be read all return the same
not-found payload, so the tool cannot be used to probe for other conversations'
results.

A storage or database fault is the one distinguishable outcome: it returns a
retryable ``unavailable`` payload. It says only that the store failed, never
whether a blob exists, and unlike a corrupt record it is worth retrying.
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
    "this instead of re-running the tool whose result was truncated."
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

_UNAVAILABLE = {
    "status": "error",
    "error_type": "unavailable",
    "retryable": True,
    "hint": (
        "The offloaded result could not be loaded because its storage was "
        "unreachable. Retry once, then continue without the full text."
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

        text, failure = await _load_text(resolved_repository, resolved_service, identity)
        if failure is not None:
            return json.dumps(failure)
        return _slice_payload(identity[0], text or "", offset=offset, limit=limit)

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


async def _load_text(
    repository: Any,
    service: Any,
    identity: tuple[UUID, UUID, UUID],
) -> tuple[str | None, dict[str, Any] | None]:
    """Return the blob text, or the error payload to report in its place."""

    blob_id, user_uuid, conversation_uuid = identity
    try:
        record = await asyncio.to_thread(
            repository.get_for_user_and_conversation,
            blob_id,
            user_uuid,
            conversation_uuid,
        )
    except Exception as exc:
        logger.warning("Tool result blob lookup failed for %s: %s", blob_id, exc)
        return None, _UNAVAILABLE
    if record is None:
        return None, _NOT_FOUND

    try:
        return await asyncio.to_thread(service.read_text, record), None
    except (ValueError, OSError) as exc:
        # A corrupt record and a vanished legacy file are both unreadable
        # forever, so they are the not-found case rather than a retry.
        logger.warning("Tool result blob %s is unreadable: %s", blob_id, exc)
        return None, _NOT_FOUND
    except Exception as exc:
        logger.warning("Tool result blob %s could not be read: %s", blob_id, exc)
        return None, _UNAVAILABLE


def _slice_payload(blob_id: UUID, text: str, *, offset: int, limit: int | None) -> str:
    cap = max(1, int(settings.tool_result_read_max_chars))
    window = cap if limit is None else max(1, min(int(limit), cap))
    chunk = text[offset : offset + window]
    end = offset + len(chunk)
    return json.dumps(
        {
            "blob_id": str(blob_id),
            "offset": offset,
            "returned_chars": len(chunk),
            "total_chars": len(text),
            "next_offset": end if end < len(text) else None,
            "content": chunk,
        },
        ensure_ascii=False,
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
        from ..core.container import get_container

        # The process-wide container, not a fresh ``Container()``: instantiating
        # the declarative container rebuilds its ``Database`` singleton, so every
        # read_tool_result call would open a new engine and connection pool.
        container = get_container()
        return (
            repository or container.tool_result_blob_repository(),
            service or container.tool_result_blob_service(),
        )
    except Exception as exc:  # pragma: no cover - DI unavailable in some contexts
        logger.debug("Tool result blob access unavailable: %s", exc)
        return None, None
