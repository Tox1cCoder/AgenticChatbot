"""Internal tool that retrieves evidence from an offloaded tool result.

The model is shown a preview and a ``blob_id``; this tool answers one question
against the stored text. It is deliberately not a pager: an offset cursor
turned every large result into a read-again loop that spent the turn locating
the answer instead of using it.

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
import re
from typing import Any
from uuid import UUID

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from ..core.config import settings
from .focused_tool_result import select_focused_excerpts
from .tool_context import get_tool_context

logger = logging.getLogger(__name__)

#: Word characters, matching how the retriever itself tokenizes an objective.
_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)

READ_TOOL_RESULT_DESCRIPTION = (
    "Retrieve the passages most relevant to a specific objective from a large "
    "offloaded tool result. Pass the blob_id and a precise question or fact to "
    "find. The response is bounded and source-addressed; do not call repeatedly "
    "with the same objective."
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
        "unreachable. Retry once, then answer from the preview alone."
    ),
}


class ReadToolResultInput(BaseModel):
    blob_id: str = Field(description="The blob_id printed in the offload notice.")
    objective: str = Field(
        min_length=3,
        max_length=500,
        description=(
            "The exact fact or question to find in the stored result. Ranking is "
            "driven by these words, so name the fact, not the topic."
        ),
    )
    max_excerpts: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Passages to return. Clamped to the configured maximum.",
    )
    max_chars: int | None = Field(
        default=None,
        ge=1000,
        le=80_000,
        description="Response size ceiling. Clamped to the configured maximum.",
    )

    @field_validator("objective")
    @classmethod
    def _require_a_stated_objective(cls, value: str) -> str:
        """Reject an objective that is only whitespace or punctuation.

        ``min_length`` counts spaces, so three of them pass a length the field
        meant as "name the fact". Ranking against no terms answers "no passage
        matched" for a payload that may hold the answer, and the model has no
        way to tell that from a genuine miss.
        """

        collapsed = " ".join(str(value or "").split())
        if len(collapsed) < 3 or not _TOKEN_RE.search(collapsed):
            raise ValueError("objective must name the fact to find")
        return collapsed


def create_read_tool_result_tool(
    *,
    repository: Any | None = None,
    service: Any | None = None,
) -> StructuredTool:
    """Build the ``read_tool_result`` tool, resolving DI lazily when not injected."""

    async def _read(
        blob_id: str,
        objective: str,
        max_excerpts: int | None = None,
        max_chars: int | None = None,
    ) -> str:
        context = get_tool_context()
        identity = _scoped_identity(blob_id, context.user_id, context.conversation_id)
        if identity is None:
            return _refused(_NOT_FOUND)
        resolved_repository, resolved_service = _resolve(repository, service)
        if resolved_repository is None or resolved_service is None:
            return _refused(_NOT_FOUND)

        text, failure = await _load_text(resolved_repository, resolved_service, identity)
        if failure is not None:
            return _refused(failure)
        focused = select_focused_excerpts(
            text or "",
            objective=objective,
            max_excerpts=_clamp(max_excerpts, settings.tool_result_focus_max_excerpts),
            max_chars=_clamp(max_chars, settings.tool_result_focus_max_chars),
        )
        serialized = focused.model_dump_json()
        _observe(
            "matched" if focused.excerpts else "no_match",
            blob_chars=len(text or ""),
            model_chars=len(serialized),
            excerpts=len(focused.excerpts),
            omitted=focused.omitted_candidates,
            truncated=focused.truncated,
        )
        return serialized

    return StructuredTool.from_function(
        coroutine=_read,
        name="read_tool_result",
        description=READ_TOOL_RESULT_DESCRIPTION,
        args_schema=ReadToolResultInput,
        metadata={
            "tool_origin": "internal",
            "qualified_tool_id": "internal::read_tool_result",
        },
    )


def _observe(
    outcome: str,
    *,
    blob_chars: int = 0,
    model_chars: int = 0,
    excerpts: int = 0,
    omitted: int = 0,
    truncated: bool = False,
) -> None:
    """One line per call, whatever the outcome.

    Counts and one enum only. The objective is user-derived text and is never
    written here, so this stays safe to emit unconditionally -- which is the
    point: a refusal that logs nothing is indistinguishable from a call that
    never happened.
    """

    logger.info(
        "focused_tool_result_read outcome=%s blob_chars=%d model_chars=%d "
        "excerpts=%d omitted=%d truncated=%s",
        outcome,
        blob_chars,
        model_chars,
        excerpts,
        omitted,
        truncated,
    )


def _refused(payload: dict[str, Any]) -> str:
    """Serialize a refusal and record it under its own outcome."""

    serialized = json.dumps(payload)
    _observe(str(payload["error_type"]), model_chars=len(serialized))
    return serialized


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


def _clamp(requested: int | None, configured: int) -> int:
    """A caller may ask for less than the configured ceiling, never for more."""

    ceiling = max(1, int(configured))
    return ceiling if requested is None else max(1, min(int(requested), ceiling))


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
