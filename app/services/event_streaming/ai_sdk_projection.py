"""Projection helpers for AI SDK-facing response payloads.

Single source of truth for how persisted messages and stream side-channel
payloads project onto the Vercel AI SDK wire format. Both the API layer
(``app.api.ai_sdk``) and the stream adapter (``ai_sdk_v6``) import from here,
so the service layer never depends on the API layer.

Wire rules enforced here (2026-07-02 response-format cleanup):

- ``metadata`` is the only message-metadata field; the ``messageMetadata``
  mirror is gone.
- Legacy renderer fields (``images``, ``live_widgets``, ``canvas_artifact``,
  image counters, ``pending_tool_calls``) and internal ``_``-prefixed keys are
  scrubbed from every AI SDK metadata projection. Images surface as ``file``
  parts; widgets/canvas/tool renders surface only through ``rich_items``.
- Additive nested metadata such as ``context_window`` is preserved verbatim.
  The projection deliberately has no nested allowlist, so model-limit fields,
  token splits, ratios, and future fields survive both history and final-event
  paths without adding a mandatory stream event.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from app.core.rich_response import strip_inline_rich_markers

LEGACY_METADATA_KEYS = frozenset(
    {
        "images",
        "has_images",
        "images_count",
        "agentic_images_count",
        "live_widgets",
        "canvas_artifact",
        "pending_tool_calls",
    }
)

# Database-redundant / write-only debug fields some agents persist into
# message metadata. Clients already know the conversation from the route, and
# nothing reads the counters — they are wire noise.
DB_REDUNDANT_METADATA_KEYS = frozenset(
    {
        "conversation_id",
        "has_tool_calls",
        "context_messages",
    }
)

_SCRUBBED_METADATA_KEYS = LEGACY_METADATA_KEYS | DB_REDUNDANT_METADATA_KEYS

_METADATA_KEYS = ("message_metadata", "messageMetadata", "metadata")


def find_message_metadata(message: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first metadata dict on a message payload.

    Persisted message dumps use ``message_metadata``; already-projected
    payloads use ``metadata``. The persisted key wins when both are present.
    """
    for key in _METADATA_KEYS:
        value = message.get(key)
        if isinstance(value, dict):
            return value
    return None


def scrub_legacy_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop legacy renderer, database-redundant, and internal keys from an
    AI SDK projection.

    Persistence is untouched — the Streamlit path still reads the legacy
    fields from the database. Callers that need image ``file`` parts must
    extract them before scrubbing. Values are intentionally not traversed:
    nested contracts such as ``context_window`` and unknown future fields are
    additive and pass through unchanged.
    """
    return {
        key: value
        for key, value in metadata.items()
        if key not in _SCRUBBED_METADATA_KEYS and not key.startswith("_")
    }


def is_v1_rich_items_message(metadata: dict[str, Any] | None) -> bool:
    if not isinstance(metadata, dict):
        return False
    return metadata.get("rich_items_version") == 1


def _extract_mime_from_data_url(value: str) -> str | None:
    payload = value.strip()
    if not payload.startswith("data:"):
        return None

    header, _, _ = payload.partition(",")
    mime = header[5:].split(";")[0].strip()
    return mime if "/" in mime else None


def _normalize_image_item_to_file_part(item: Any) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None

    mime = (
        item.get("mime")
        or item.get("mimeType")
        or item.get("mediaType")
        or item.get("contentType")
        or "image/png"
    )
    mime = str(mime).strip() if mime else "image/png"

    candidate_values: list[Any] = [
        item.get("url"),
        item.get("data"),
        item.get("base64"),
        item.get("image"),
        item.get("source"),
    ]

    for candidate in candidate_values:
        if isinstance(candidate, dict):
            candidate = candidate.get("url") or candidate.get("data") or candidate.get("base64")
        if not isinstance(candidate, str):
            continue

        raw_value = candidate.strip()
        if not raw_value:
            continue

        if raw_value.startswith("data:"):
            detected_mime = _extract_mime_from_data_url(raw_value)
            if detected_mime:
                mime = detected_mime
            return {"url": raw_value, "mediaType": mime}

        if raw_value.startswith(("http://", "https://", "blob:")):
            return {"url": raw_value, "mediaType": mime}

        try:
            base64.b64decode(raw_value, validate=False)
        except Exception:
            continue

        return {
            "url": f"data:{mime};base64,{raw_value}",
            "mediaType": mime,
        }

    return None


def _extract_image_file_parts_from_metadata(
    metadata: dict[str, Any] | None,
) -> list[dict[str, str]]:
    if not isinstance(metadata, dict):
        return []

    images = metadata.get("images")
    if not isinstance(images, list):
        return []

    file_parts: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in images:
        file_part = _normalize_image_item_to_file_part(item)
        if not file_part:
            continue

        key = (file_part["url"], file_part["mediaType"])
        if key in seen:
            continue
        seen.add(key)
        file_parts.append(file_part)

    return file_parts


def extract_image_file_parts_from_message(
    message: dict[str, Any],
) -> list[dict[str, str]]:
    if not isinstance(message, dict):
        return []
    return _extract_image_file_parts_from_metadata(find_message_metadata(message))


def selected_image_file_parts_from_rich_items(
    metadata: dict[str, Any] | None,
) -> list[dict[str, str]]:
    if not isinstance(metadata, dict):
        return []
    rich_items = metadata.get("rich_items")
    if not isinstance(rich_items, list):
        return []
    file_parts: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in rich_items:
        if not isinstance(item, dict) or item.get("type") != "image":
            continue
        payload = item.get("payload") or {}
        url = payload.get("url")
        data = payload.get("data")
        mime_type = payload.get("mime_type") or "image/png"
        if url:
            file_part = {"url": str(url), "mediaType": str(mime_type)}
        elif data:
            file_part = {"url": f"data:{mime_type};base64,{data}", "mediaType": str(mime_type)}
        else:
            continue
        key = (file_part["url"], file_part["mediaType"])
        if key in seen:
            continue
        seen.add(key)
        file_parts.append(file_part)
    return file_parts


def project_ai_sdk_message_for_capability(
    message: dict[str, Any],
    *,
    inline_rich_response_v1: bool,
) -> dict[str, Any]:
    """Project a persisted assistant message for an AI SDK consumer.

    When ``inline_rich_response_v1`` is True, the message passes through
    unchanged (markers preserved, ``rich_items`` available). When False, the
    standalone HTML-comment marker lines are stripped from ``content`` so
    non-capable clients do not render them verbatim, and ``rich_items`` /
    ``rich_items_version`` keys are removed from any metadata field present.
    """
    if not isinstance(message, dict):
        return message
    if inline_rich_response_v1:
        return message
    projected = dict(message)
    content = projected.get("content")
    if isinstance(content, str) and "<!--rich:" in content:
        projected["content"] = strip_inline_rich_markers(content)
    for meta_key in _METADATA_KEYS:
        meta = projected.get(meta_key)
        if isinstance(meta, dict):
            scrubbed = {
                k: v
                for k, v in meta.items()
                if k not in {"rich_items", "rich_items_version", "rich_reference_warnings"}
            }
            projected[meta_key] = scrubbed
    return projected


def visible_image_file_parts(
    message: dict[str, Any],
    *,
    is_v1: bool | None = None,
) -> list[dict[str, str]]:
    """Resolve the image ``file`` parts a message may expose on the wire.

    For v1 rich-items messages, file parts come only from finalized
    ``rich_items`` (selected images) so hidden candidates cannot leak — even
    when the capability projection has already stripped the rich keys for a
    non-capable client (pass ``is_v1`` computed from the original metadata).
    Other messages fall back to the persisted legacy image metadata, which is
    scrubbed from the wire separately.
    """
    metadata = find_message_metadata(message)
    if is_v1 is None:
        is_v1 = is_v1_rich_items_message(metadata)
    if is_v1:
        return selected_image_file_parts_from_rich_items(metadata)
    return extract_image_file_parts_from_message(message)


def attach_image_parts_to_message(
    message: dict[str, Any],
    *,
    is_v1: bool | None = None,
) -> dict[str, Any]:
    """Embed the message's visible images as AI SDK ``file`` parts."""
    if not isinstance(message, dict):
        return message

    payload = dict(message)
    image_parts = visible_image_file_parts(payload, is_v1=is_v1)
    if not image_parts:
        return payload

    existing_parts = payload.get("parts")
    parts: list[dict[str, Any]] = (
        [p for p in existing_parts if isinstance(p, dict)]
        if isinstance(existing_parts, list)
        else []
    )

    if not isinstance(existing_parts, list):
        content = payload.get("content")
        if isinstance(content, str) and content.strip():
            parts.append({"type": "text", "text": content})

    existing_urls = {
        p.get("url") for p in parts if p.get("type") == "file" and isinstance(p.get("url"), str)
    }

    for file_part in image_parts:
        url = file_part["url"]
        if url in existing_urls:
            continue
        parts.append(
            {
                "type": "file",
                "url": url,
                "mediaType": file_part["mediaType"],
            }
        )
        existing_urls.add(url)

    payload["parts"] = parts
    return payload


def ensure_leading_text_part(message: dict[str, Any]) -> None:
    """Guarantee ``parts`` exists and carries the message text as a part.

    AI SDK v5+ clients render from ``parts``; a history message without a
    ``text`` part would display as empty even though ``content`` is set.
    """
    parts = message.get("parts")
    if not isinstance(parts, list):
        parts = []
    has_text = any(isinstance(p, dict) and p.get("type") == "text" for p in parts)
    content = message.get("content")
    if not has_text and isinstance(content, str) and content.strip():
        parts.insert(0, {"type": "text", "text": content})
    message["parts"] = parts


def project_ai_sdk_message_event(
    message: dict[str, Any],
    *,
    include_content: bool = False,
) -> dict[str, Any]:
    """Project a persisted message into a stream metadata side-channel.

    The AI SDK stream has already delivered body text through ``text-delta``
    (assistant) or received it from the client (user), so ``content`` is only
    included on request. The projection keeps durable metadata and file parts
    while dropping database-only fields such as ``sender``,
    ``conversation_id``, and ``updated_at``, and scrubbing legacy renderer
    metadata.
    """
    if not isinstance(message, dict):
        return {}

    projected: dict[str, Any] = {}
    message_id = message.get("id")
    if message_id not in (None, ""):
        projected["id"] = str(message_id)

    role = message.get("role")
    if not isinstance(role, str) or not role:
        sender = message.get("sender")
        if sender == 1 or sender == "1":
            role = "user"
        elif sender == 2 or sender == "2":
            role = "assistant"
    if isinstance(role, str) and role:
        projected["role"] = role

    created_at = message.get("createdAt") or message.get("created_at")
    if isinstance(created_at, str) and created_at:
        projected["createdAt"] = created_at

    if include_content:
        content = message.get("content")
        if isinstance(content, str):
            projected["content"] = content

    metadata = find_message_metadata(message)
    if isinstance(metadata, dict):
        projected["metadata"] = scrub_legacy_metadata(metadata)

    parts = message.get("parts")
    if isinstance(parts, list):
        clean_parts = [part for part in parts if isinstance(part, dict)]
        if is_v1_rich_items_message(metadata):
            allowed_urls = {
                part["url"] for part in selected_image_file_parts_from_rich_items(metadata)
            }
            clean_parts = [
                part
                for part in clean_parts
                if part.get("type") != "file" or part.get("url") in allowed_urls
            ]
        projected["parts"] = clean_parts

    return projected


def clean_tool_output(value: Any) -> Any:
    """Clean tool output by extracting actual data from LangChain Content objects."""
    if value is None:
        return None

    if isinstance(value, list):
        cleaned = []
        for item in value:
            if isinstance(item, dict):
                if "type" in item and item.get("type") == "text" and "text" in item:
                    cleaned.append(item["text"])
                else:
                    cleaned.append(clean_tool_output(item))
            else:
                cleaned.append(clean_tool_output(item))

        if len(cleaned) == 1:
            return cleaned[0]
        return cleaned

    if isinstance(value, dict):
        if "type" in value and value.get("type") == "text" and "text" in value:
            return value["text"]
        return {k: clean_tool_output(v) for k, v in value.items()}

    return value


def coerce_json_object(value: Any) -> Any:
    """
    Vercel AI SDK UI message stream expects tool ``input``/``output`` to be
    JSON-serializable. Returns the exact tool result without any wrapping,
    parsing JSON-looking strings so clients receive objects instead of
    double-encoded text.
    """
    if value is None:
        return None

    if isinstance(value, (dict, list)):
        return value

    if isinstance(value, (int, float, bool)):
        return value

    if isinstance(value, str):
        s = value.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                pass
        return value

    return str(value)
