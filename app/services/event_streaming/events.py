"""Canonical internal v3 stream event model.

This is the single internal streaming contract for the backend. It is
independent of LangChain's upstream ``astream_events(version="v3")`` API —
"v3" here names *our* schema. Public adapters (AI SDK v6, Streamlit internal
SSE) consume these events and project them to their respective wire formats.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

StreamSchemaVersion = Literal["v3"]

StreamEventType = Literal[
    "run_start",
    "message_start",
    "message_delta",
    "message_end",
    "reasoning_start",
    "reasoning_delta",
    "reasoning_end",
    "tool_call_delta",
    "tool_call_available",
    "tool_execution_start",
    "tool_execution_delta",
    "tool_execution_end",
    "agent_selected",
    "subagent_start",
    "subagent_message_delta",
    "subagent_tool_call_available",
    "subagent_tool_execution_start",
    "subagent_tool_execution_end",
    "subagent_end",
    "state_snapshot",
    "rich_items",
    "image_preview",
    "interrupt",
    "complete",
    "error",
    "heartbeat",
    "user_message_created",
    "title_updated",
]


class SubagentRef(BaseModel):
    id: str
    name: str
    path: list[str] = Field(default_factory=list)
    status: Literal["running", "completed", "failed", "timeout", "requires_approval"]


class V3StreamEvent(BaseModel):
    schema_version: StreamSchemaVersion = "v3"
    type: StreamEventType
    sequence: int
    run_id: str | None = None
    conversation_id: str | None = None
    message_id: str | None = None
    agent: str | None = None
    node: str | None = None
    namespace: list[str] = Field(default_factory=list)
    subagent: SubagentRef | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def make_event(
    event_type: StreamEventType,
    *,
    sequence: int,
    run_id: str | None = None,
    conversation_id: str | None = None,
    message_id: str | None = None,
    agent: str | None = None,
    node: str | None = None,
    namespace: list[str] | None = None,
    subagent: SubagentRef | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> V3StreamEvent:
    return V3StreamEvent(
        type=event_type,
        sequence=sequence,
        run_id=run_id,
        conversation_id=conversation_id,
        message_id=message_id,
        agent=agent,
        node=node,
        namespace=list(namespace or []),
        subagent=subagent,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        data=dict(data or {}),
        metadata=dict(metadata or {}),
    )


# ---------------------------------------------------------------------------
# Versioned ``image_preview`` delivery union (schema v2)
#
# The wire event ``type`` stays ``"image_preview"`` (compatibility name); the
# ``data`` payload is versioned. v2 carries an explicit ``delivery`` sub-object
# so a final image is always delivered by protected reference (never inline
# base64), while transient provider partials may be delivered inline under a
# small wire budget. v1 inline payloads (top-level ``data_b64``) remain
# READABLE during the compatibility window; the readers below tolerate both.
# ---------------------------------------------------------------------------

IMAGE_PREVIEW_SCHEMA_VERSION = 2

IMAGE_PREVIEW_STATUS_PARTIAL = "partial"
IMAGE_PREVIEW_STATUS_FINAL = "final"
IMAGE_PREVIEW_STATUS_SKIPPED = "preview_skipped"


def build_image_preview_reference_data(
    *,
    image_index: Any,
    item_id: str,
    image_id: Any,
    url: str,
    media_type: str | None,
    seq: int,
) -> dict[str, Any]:
    """Build the schema-v2 ``data`` for a FINAL image delivered by protected
    reference. Never carries base64."""
    return {
        "schema_version": IMAGE_PREVIEW_SCHEMA_VERSION,
        "item_id": item_id,
        "image_index": image_index,
        "status": IMAGE_PREVIEW_STATUS_FINAL,
        "seq": seq,
        "media_type": media_type or "image/png",
        "delivery": {"kind": "reference", "image_id": image_id, "url": url},
    }


def build_image_preview_inline_data(
    *,
    image_index: Any,
    item_id: str,
    data_url: str,
    media_type: str | None,
    seq: int,
    status: str = IMAGE_PREVIEW_STATUS_PARTIAL,
) -> dict[str, Any]:
    """Build the schema-v2 ``data`` for an inline (transient) preview."""
    return {
        "schema_version": IMAGE_PREVIEW_SCHEMA_VERSION,
        "item_id": item_id,
        "image_index": image_index,
        "status": status,
        "seq": seq,
        "media_type": media_type or "image/png",
        "delivery": {"kind": "inline", "data_url": data_url},
    }


def build_image_preview_skipped_data(
    *,
    image_index: Any,
    item_id: str | None,
    media_type: str | None,
    seq: int,
    reason: str = "oversized_inline_preview",
) -> dict[str, Any]:
    """Build the schema-v2 ``data`` for a preview the wire budget skipped.

    A structured status (never a silent drop) so oversized/skipped previews
    stay observable (FR-IMG-007). Carries no base64.
    """
    return {
        "schema_version": IMAGE_PREVIEW_SCHEMA_VERSION,
        "item_id": item_id,
        "image_index": image_index,
        "status": IMAGE_PREVIEW_STATUS_SKIPPED,
        "seq": seq,
        "media_type": media_type or "image/png",
        "reason": reason,
    }


def resolve_image_preview_delivery(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a v1 or v2 ``image_preview`` ``data`` payload for the wire.

    Returns a dict with ``item_id``, ``image_index``, ``status``, ``seq``,
    ``media_type``, ``kind`` (``inline``/``reference``/``none``), ``url``
    (assembled ``data:`` URL for inline, or the protected reference URL),
    ``image_id`` and ``reason``. Never returns raw base64 beyond the assembled
    inline ``data:`` URL.
    """
    media_type = data.get("media_type") or data.get("mime") or "image/png"
    status = data.get("status") or IMAGE_PREVIEW_STATUS_FINAL
    url: str | None = None
    kind = "none"
    image_id = None

    delivery = data.get("delivery")
    if isinstance(delivery, dict):
        delivery_kind = delivery.get("kind")
        if delivery_kind == "reference":
            url = delivery.get("url")
            image_id = delivery.get("image_id")
            kind = "reference"
        elif delivery_kind == "inline":
            url = delivery.get("data_url")
            kind = "inline"
    else:
        # v1 read-compat: inline base64 at the top level.
        data_b64 = data.get("data_b64")
        if data_b64:
            url = f"data:{media_type};base64,{data_b64}"
            kind = "inline"

    return {
        "item_id": data.get("item_id"),
        "image_index": data.get("image_index"),
        "status": status,
        "seq": data.get("seq") or 0,
        "media_type": media_type,
        "kind": kind,
        "url": url,
        "image_id": image_id,
        "reason": data.get("reason"),
    }


def inline_preview_wire_size(data: dict[str, Any]) -> int:
    """Character length of the inline base64/data-url payload (0 if none)."""
    delivery = data.get("delivery")
    if isinstance(delivery, dict):
        if delivery.get("kind") == "inline":
            return len(delivery.get("data_url") or "")
        return 0
    return len(data.get("data_b64") or "")


def apply_inline_preview_wire_budget(data: dict[str, Any], *, budget: int) -> dict[str, Any]:
    """Second-defense inline budget at serialization time.

    Downgrade an oversized inline preview to a structured ``preview_skipped``
    status (FR-IMG-007) rather than shipping an outsized inline payload or
    silently dropping it. Reference/skip/empty deliveries pass through.
    """
    if budget <= 0:
        return data
    if inline_preview_wire_size(data) <= budget:
        return data
    return build_image_preview_skipped_data(
        image_index=data.get("image_index"),
        item_id=data.get("item_id"),
        media_type=data.get("media_type") or data.get("mime") or "image/png",
        seq=data.get("seq") or 0,
        reason="oversized_inline_wire_budget",
    )


# Shared by both public adapters (AI SDK `data-subagent`, Streamlit `subagent`)
# so the wire `phase` discriminator stays consistent across protocols.
SUBAGENT_PHASE_BY_EVENT: dict[str, str] = {
    "subagent_start": "start",
    "subagent_end": "end",
    "subagent_tool_call_available": "tool",
    "subagent_tool_execution_start": "tool",
    "subagent_tool_execution_end": "tool",
    "subagent_message_delta": "delta",
}
