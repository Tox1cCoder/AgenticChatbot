from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anyio import ClosedResourceError

from ..core.config import settings
from ..core.rich_image_selection import (
    image_aspect_ratio_ok,
    image_url_scheme,
    is_junk_image_url,
)
from ..core.rich_response import (
    GENERIC_IMAGE_ALT_TEXT,
    RichDisplayPolicy,
    RichItemType,
)
from ..observability.rich_images import rich_image_metrics
from .client_runtime_tools import (
    CLIENT_TOOL_PREFIX,
    get_active_client_runtime_session,
    get_client_runtime_tools,
    get_client_tool_device_id,
    is_client_tool,
)
from .tool_error_policy import (
    ToolErrorKind,
    ToolErrorSummary,
    build_tool_error_payloads,
    classify_tool_error,
)
from .tool_execution_policy import (
    AmbiguousToolExecutionPolicyError,
    ToolExecutionPolicy,
    ToolExecutionPolicyValidationError,
    resolve_tool_execution_policy,
    tool_policy_context,
)
from .tool_result_rendering import normalize_tool_result_for_rendering
from .tool_scope import is_client_only_scope
from .tool_search_tool import create_tool_search_tool
from .utils import make_json_safe, normalize_tool_call

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Tool names that may load additional tools dynamically
TOOL_LOADING_TOOLS = {"tool_search"}
_RETRY_COMPATIBILITY_ALLOWLIST = frozenset({("internal", "internal::tool_search")})
_WIDGET_ARTIFACT_TOOLS = {"widget_create", "widget_update"}
_WIDGET_SESSION_BOUND_TOOLS = {"widget_create", "session_list_widgets"}
_FULL_MODEL_HANDOFF_TOOLS = {"dispatch_subagents"}
_TYPED_WEB_IMAGE_TOOLS = frozenset({"tavily_search", "brave_image_search"})

# Render-type values that should never produce a public tool_render candidate.
# Live-widget renders have a dedicated `widget:<id>` candidate; error/text/json
# noise is not meaningful as an inline placement and would only clutter the
# inventory.
_NON_INLINE_RENDER_TYPES = {
    "live_widget",
    "error",
    "text",
    "json",
    "subagent_dispatch",
}


def _guess_mime_from_url(url: str) -> str:
    lowered = url.lower().split("?", 1)[0]
    if lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
        return "image/jpeg"
    if lowered.endswith(".webp"):
        return "image/webp"
    if lowered.endswith(".gif"):
        return "image/gif"
    return "image/png"


def _short_digest(value: str) -> str:
    """Return a short stable digest, used only to keep ids distinct."""
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:8]


def order_tavily_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order Tavily image dicts by provenance strength.

    Source-bound before query-level, then higher parent score, then lower parent
    rank, with original provider order as the stable tie-breaker. Ordering never
    rejects a candidate.
    """

    def sort_key(indexed: tuple[int, dict[str, Any]]) -> tuple[int, int, float, int, int]:
        index, image = indexed
        query_level = 1 if image.get("query_level") else 0
        raw_score = image.get("result_score")
        has_score = 0 if isinstance(raw_score, (int, float)) else 1
        score = -float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
        raw_rank = image.get("result_rank")
        rank = int(raw_rank) if isinstance(raw_rank, int) and raw_rank >= 0 else 10**6
        return (query_level, has_score, score, rank, index)

    return [image for _, image in sorted(enumerate(images), key=sort_key)]


def build_image_candidates_from_tool_result(
    result_text: str,
    *,
    tool_call_id: str | None,
    tool_name: str,
) -> list[dict[str, Any]]:
    """Build typed rich-item image candidates from a tool result payload.

    Each candidate is a public-shape dict (matching the discriminated
    ``RichItem`` schema) with deterministic id, source, provenance, and
    payload. The candidate dicts are safe to forward to
    ``build_bot_metadata()`` as transient `_rich_item_candidates`.

    A Brave result with 2+ unique eligible candidates collapses into a
    single ``image_group`` item (see ``_group_image_candidates``) so the model
    has one marker id to copy instead of choosing among several. Tavily
    results and single-candidate Brave results are returned as individual
    ``image`` items.
    """
    if not result_text:
        return []

    try:
        parsed = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(parsed, dict):
        return []

    images = parsed.get("images")
    if not isinstance(images, list):
        return []

    result_provider = str(parsed.get("provider") or "").strip().lower()
    metric_provider = (
        "brave"
        if result_provider.startswith("brave") or tool_name == "brave_image_search"
        else "tavily"
        if result_provider == "tavily" or tool_name == "tavily_search"
        else "other"
    )
    with suppress(Exception):
        rich_image_metrics.record_discovery(
            provider=metric_provider,
            result_count=len(images),
        )

    def _reject(reason: str) -> None:
        # Telemetry is best-effort: a metrics fault must never drop a candidate
        # decision or fail the surrounding tool result.
        with suppress(Exception):
            rich_image_metrics.record_candidate(provider=metric_provider, outcome=reason)

    if metric_provider == "tavily":
        # This pre-filter runs before the candidate loop below, so a non-dict
        # entry must be counted here (once) rather than silently dropped —
        # otherwise Tavily malformed entries would never reach the loop's own
        # `rejected_malformed` branch and the counters would diverge by
        # provider for no reason.
        dict_images: list[dict[str, Any]] = []
        for image in images:
            if isinstance(image, dict):
                dict_images.append(image)
            else:
                _reject("rejected_malformed")
        images = order_tavily_images(dict_images)

    result_query = str(parsed.get("query") or "").strip()
    candidates: list[dict[str, Any]] = []
    seen_display_urls: set[str] = set()
    candidate_cap = max(1, int(getattr(settings, "rich_image_candidate_max_count", 8)))
    minimum_width = max(1, int(getattr(settings, "rich_image_min_width_px", 320)))
    minimum_height = max(1, int(getattr(settings, "rich_image_min_height_px", 180)))

    for index, image in enumerate(images):
        if not isinstance(image, dict):
            _reject("rejected_malformed")
            continue
        url = image.get("url")
        data = image.get("data") or image.get("b64_data")
        if not url and not data:
            _reject("rejected_malformed")
            continue
        provider = str(image.get("provider") or "").strip().lower()
        original_url = str(url or "").strip()
        thumbnail_url = str(image.get("thumbnail_url") or "").strip()
        is_brave = provider.startswith("brave") or tool_name == "brave_image_search"
        display_url = thumbnail_url if is_brave and thumbnail_url else original_url
        width = image.get("width")
        height = image.get("height")
        if display_url:
            if image_url_scheme(display_url) != "https":
                _reject("rejected_scheme")
                continue
            if display_url in seen_display_urls:
                _reject("rejected_duplicate")
                continue
            if is_junk_image_url(display_url):
                _reject("rejected_junk_url")
                continue
            if not image_aspect_ratio_ok(
                width,
                height,
                minimum=float(getattr(settings, "rich_image_min_aspect_ratio", 0.2)),
                maximum=float(getattr(settings, "rich_image_max_aspect_ratio", 5.0)),
            ):
                _reject("rejected_aspect_ratio")
                continue
            if isinstance(width, int) and width < minimum_width:
                _reject("rejected_dimensions")
                continue
            if isinstance(height, int) and height < minimum_height:
                _reject("rejected_dimensions")
                continue
        payload: dict[str, Any] = {}
        mime_type = image.get("mime_type") or image.get("mimeType")
        if display_url:
            payload["url"] = display_url
            if not mime_type:
                mime_type = _guess_mime_from_url(display_url)
        elif data:
            payload["data"] = str(data)
            if not mime_type:
                mime_type = "image/png"
        payload["mime_type"] = str(mime_type)
        source_url = image.get("source_url")
        if source_url:
            payload["source_url"] = str(source_url)
        description = image.get("description")
        if description:
            payload["description"] = str(description)
        if isinstance(width, int) and width > 0:
            payload["width"] = width
        if isinstance(height, int) and height > 0:
            payload["height"] = height
        candidate_id_base = tool_call_id or tool_name or "tool"
        provenance: dict[str, Any] = {
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "index": index,
        }
        # Provider-specific metadata (Brave thumbnails/dimensions/source domain)
        # is not part of the narrow public ImagePayload schema, so it is kept in
        # provenance rather than risking forbidden payload extras.
        for meta_key in (
            "thumbnail_url",
            "source_domain",
            "source_title",
            "provider",
            "result_rank",
            "result_score",
            "query_level",
        ):
            meta_value = image.get(meta_key)
            if meta_value is not None:
                provenance[meta_key] = meta_value
        if display_url and original_url:
            provenance["original_image_digests"] = {
                display_url: hashlib.sha256(original_url.encode("utf-8")).hexdigest()
            }
        if result_query:
            provenance["query"] = result_query
        candidates.append(
            {
                "id": f"image:tool:{candidate_id_base}:{index}",
                "type": RichItemType.image.value,
                "source": (
                    "web_search"
                    if tool_name == "tavily_search"
                    else "image_search"
                    if is_brave
                    else "tool_image"
                ),
                "display_policy": RichDisplayPolicy.inline_only.value,
                "alt_text": str(description or image.get("alt") or GENERIC_IMAGE_ALT_TEXT),
                "title": image.get("title"),
                "payload": payload,
                "provenance": provenance,
            }
        )
        if display_url:
            seen_display_urls.add(display_url)
        with suppress(Exception):
            rich_image_metrics.record_candidate(provider=metric_provider, outcome="eligible")
        if len(candidates) >= candidate_cap:
            break
    if metric_provider == "brave" and len(candidates) >= 2:
        return [
            _group_image_candidates(
                candidates,
                tool_call_id=tool_call_id,
                query=result_query,
                metric_provider=metric_provider,
            )
        ]
    return candidates


def _group_image_candidates(
    candidates: list[dict[str, Any]],
    *,
    tool_call_id: str | None,
    query: str,
    metric_provider: str,
) -> dict[str, Any]:
    """Collapse eligible image-search candidates into one image_group item.

    ``provenance["provider"]`` prefers the raw per-image provider string so a
    group spells its provider the same way a single image does — ``provenance``
    is client-visible metadata, and two spellings of one provider is a trap for
    anyone reading it. The per-image copy is conditional, so ``metric_provider``
    (the caller's classified "brave"/"tavily"/"other" label) is the fallback that
    keeps the field from ever being None. Both normalize identically for metrics.
    """
    cap = max(2, int(getattr(settings, "rich_image_group_max_items", 3)))
    selected: list[dict[str, Any]] = []
    seen_locators: set[str] = set()
    for candidate in candidates:
        payload = candidate.get("payload") or {}
        display_url = str(payload.get("url") or "").strip()
        locators = {f"display::{display_url}"} if display_url else set()
        provenance = candidate.get("provenance") or {}
        digests = provenance.get("original_image_digests")
        if display_url and isinstance(digests, dict):
            digest = digests.get(display_url)
            if digest:
                locators.add(f"original::{digest}")
        if seen_locators.intersection(locators):
            continue
        selected.append(candidate)
        seen_locators.update(locators)
        if len(selected) >= cap:
            break
    if len(selected) == 1:
        return selected[0]
    cells: list[dict[str, Any]] = []
    for candidate in selected:
        payload = candidate.get("payload") or {}
        cell: dict[str, Any] = {
            "url": payload.get("url"),
            "mime_type": payload.get("mime_type") or "image/jpeg",
        }
        for key in ("source_url", "description", "width", "height"):
            value = payload.get(key)
            if value is not None:
                cell[key] = value
        cells.append(cell)
    first_provenance = selected[0].get("provenance") or {}
    original_image_digests: dict[str, str] = {}
    for candidate in selected:
        candidate_provenance = candidate.get("provenance") or {}
        digests = candidate_provenance.get("original_image_digests")
        if isinstance(digests, dict):
            original_image_digests.update(
                {
                    str(display_url): str(digest)
                    for display_url, digest in digests.items()
                    if display_url and digest
                }
            )
    # Without a tool_call_id, two groups in one turn would both land on the same
    # literal id and collide. The query discriminates them; per-image ids get the
    # same protection from their trailing index.
    discriminator = tool_call_id or f"q{_short_digest(query)}"
    group = {
        "id": f"imagegroup:tool:{discriminator}",
        "type": RichItemType.image_group.value,
        "source": "image_search",
        "display_policy": RichDisplayPolicy.inline_only.value,
        "alt_text": f"Images of {query}" if query else GENERIC_IMAGE_ALT_TEXT,
        "payload": {"items": cells},
        "provenance": {
            "tool_call_id": tool_call_id,
            "tool": first_provenance.get("tool"),
            "provider": first_provenance.get("provider") or metric_provider,
            "query": query,
        },
    }
    if original_image_digests:
        group["provenance"]["original_image_digests"] = original_image_digests
    return group


def _extract_image_content_blocks(result: Any) -> list[dict[str, Any]]:
    """Return standard MCP image content blocks from a raw tool result."""
    safe_result = make_json_safe(result)
    if isinstance(safe_result, str):
        try:
            safe_result = json.loads(safe_result)
        except (json.JSONDecodeError, TypeError):
            return []

    if isinstance(safe_result, dict):
        content = safe_result.get("content")
        if isinstance(content, dict):
            blocks = [content]
        elif isinstance(content, list):
            blocks = content
        elif str(safe_result.get("type") or "").lower() == "image":
            blocks = [safe_result]
        else:
            blocks = []
    elif isinstance(safe_result, list):
        blocks = safe_result
    else:
        blocks = []

    return [
        block
        for block in blocks
        if isinstance(block, dict) and str(block.get("type") or "").lower() == "image"
    ]


def _image_payload_from_content_block(block: dict[str, Any]) -> dict[str, Any] | None:
    raw_url = block.get("url") or block.get("src") or block.get("image_url")
    if isinstance(raw_url, dict):
        raw_url = raw_url.get("url")
    url = str(raw_url).strip() if raw_url not in (None, "") else ""

    raw_data = block.get("data") or block.get("base64") or block.get("b64_data")
    data = str(raw_data).strip() if raw_data not in (None, "") else ""
    mime_type = str(block.get("mimeType") or block.get("mime_type") or "").strip()
    if data.startswith("data:"):
        header, separator, encoded = data.partition(",")
        if separator:
            inferred_mime = header[5:].split(";", 1)[0].strip()
            mime_type = inferred_mime or mime_type
            data = encoded

    if not url and not data:
        return None
    if not mime_type:
        mime_type = _guess_mime_from_url(url) if url else "image/png"

    payload: dict[str, Any] = {"mime_type": mime_type}
    if url:
        payload["url"] = url
    else:
        payload["data"] = data
    return payload


def build_image_candidates_from_tool_content(
    result: Any,
    *,
    tool_call_id: str | None,
    tool_name: str,
) -> list[dict[str, Any]]:
    """Promote MCP image content blocks into typed rich-image candidates.

    The normalized model text intentionally reduces image blocks to a short
    textual marker, so harvesting must happen from the original result before
    artifact render metadata redacts large inline data.
    """
    candidates: list[dict[str, Any]] = []
    candidate_id_base = tool_call_id or tool_name or "tool"
    for index, block in enumerate(_extract_image_content_blocks(result)):
        payload = _image_payload_from_content_block(block)
        if payload is None:
            continue
        description = block.get("description") or block.get("alt")
        candidates.append(
            {
                "id": f"image:tool-content:{candidate_id_base}:{index}",
                "type": RichItemType.image.value,
                "source": "tool_image",
                "display_policy": RichDisplayPolicy.inline_only.value,
                "alt_text": str(description or GENERIC_IMAGE_ALT_TEXT),
                "title": block.get("title"),
                "payload": payload,
                "provenance": {
                    "tool_call_id": tool_call_id,
                    "tool": tool_name,
                    "content_block_index": index,
                },
            }
        )
    return candidates


def extract_images_from_tool_content(result: Any) -> list[dict[str, str]]:
    """Build the legacy metadata image shape from MCP image blocks."""
    images: list[dict[str, str]] = []
    for block in _extract_image_content_blocks(result):
        payload = _image_payload_from_content_block(block)
        if payload is None:
            continue
        image: dict[str, str] = {
            "mime": str(payload["mime_type"]),
            "description": str(block.get("description") or block.get("alt") or ""),
        }
        if payload.get("url"):
            image["url"] = str(payload["url"])
        elif payload.get("data"):
            image["data"] = str(payload["data"])
        images.append(image)
    return images


def build_tool_render_candidate(
    render: dict[str, Any] | None,
    *,
    tool_call_id: str | None,
    tool_name: str,
) -> dict[str, Any] | None:
    """Build a ``tool_render`` rich-item candidate from a normalized render.

    Returns ``None`` for render types that have a dedicated rich-item channel
    (live widgets) or are not useful as an inline placement (error/text/etc.).
    """
    if not isinstance(render, dict) or not tool_call_id:
        return None
    render_type = render.get("type")
    if not isinstance(render_type, str):
        return None
    if render_type in _NON_INLINE_RENDER_TYPES:
        return None
    return {
        "id": f"tool:{tool_call_id}",
        "type": RichItemType.tool_render.value,
        "source": "tool",
        "display_policy": RichDisplayPolicy.inline_or_append.value,
        "title": render.get("title"),
        "payload": {"render": render},
        "provenance": {
            "tool_call_id": tool_call_id,
            "tool": tool_name,
        },
    }


def build_live_widget_candidate_from_tool_result(
    result: str | dict[str, Any] | None,
    *,
    tool_name: str,
) -> dict[str, Any] | None:
    """Build a public widget mount record without exposing widget state."""
    if tool_name not in _WIDGET_ARTIFACT_TOOLS or result in (None, ""):
        return None
    try:
        parsed = json.loads(result) if isinstance(result, str) else result
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict) or not parsed.get("widget_id"):
        return None

    widget_id = str(parsed["widget_id"])
    return {
        "id": f"widget:{widget_id}",
        "type": RichItemType.live_widget.value,
        "source": "widget_tool",
        "display_policy": RichDisplayPolicy.inline_or_append.value,
        "title": parsed.get("title"),
        "payload": {
            "widget_id": widget_id,
            "session_id": str(parsed.get("session_id") or ""),
            "status": str(parsed.get("status") or "active"),
            "version": int(parsed.get("version") or 1),
            "connection_endpoint": f"/widgets/{widget_id}/connection",
        },
    }


def _attach_rich_candidates_to_artifact(
    artifact: dict[str, Any],
    *,
    raw_result: Any,
    result_text: str,
    render: dict[str, Any] | None,
    tool_call_id: str | None,
    tool_name: str,
) -> None:
    """Compute rich-item candidates for a tool result and attach them as
    ``artifact["_rich_item_candidates"]`` for the graph layer to lift into
    ``context["rich_item_candidates"]``.
    """
    candidates: list[dict[str, Any]] = []
    candidates.extend(
        build_image_candidates_from_tool_result(
            result_text, tool_call_id=tool_call_id, tool_name=tool_name
        )
    )
    candidates.extend(
        build_image_candidates_from_tool_content(
            raw_result,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )
    )
    live_widget = build_live_widget_candidate_from_tool_result(
        result_text,
        tool_name=tool_name,
    )
    if live_widget is not None:
        candidates.append(live_widget)
    tool_render = build_tool_render_candidate(
        render, tool_call_id=tool_call_id, tool_name=tool_name
    )
    if tool_render is not None:
        candidates.append(tool_render)
    if candidates:
        artifact["_rich_item_candidates"] = candidates


def extract_images_from_tool_result(
    result_text: str,
    *,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
) -> list[dict[str, str]]:
    """Legacy URL/description extraction used by callers that only need the
    ``tool_images`` shape. New callers should prefer
    ``build_image_candidates_from_tool_result()`` to receive typed records.

    The kwargs are optional so the existing call sites keep working; when
    provided they are ignored here (candidate construction is the candidate
    builder's responsibility).
    """
    if not result_text:
        return []

    try:
        parsed = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(parsed, dict):
        return []

    images = parsed.get("images")
    if not isinstance(images, list):
        return []

    extracted: list[dict[str, str]] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        url = image.get("url")
        if not url:
            continue
        extracted.append(
            {
                "url": str(url),
                "description": str(image.get("description") or ""),
            }
        )
    return extracted


def _resolve_offload_service():
    """Resolve the tool result blob service from the DI container, if configured."""

    if not getattr(settings, "tool_result_offload_enabled", False):
        return None
    try:
        from ..core.container import Container

        container = Container()
        return container.tool_result_blob_service()
    except Exception as exc:
        logger.debug("Tool result offload service unavailable: %s", exc)
        return None


def apply_tool_output_offload(
    *,
    output_text: str | None,
    tool_call_id: str | None,
    tool_name: str | None,
    conversation_id: str | None,
    user_id: str | None,
    offload_service: Any | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Offload large tool outputs to durable storage.

    Returns ``(public_text, blob_info)`` where ``public_text`` is the text the
    model and artifact should now show (preview + offload notice when
    offloaded) and ``blob_info`` carries the artifact metadata to merge.
    """

    if output_text is None:
        return output_text, None
    service = offload_service if offload_service is not None else _resolve_offload_service()
    if service is None or not conversation_id or not user_id:
        return output_text, None

    threshold = getattr(service, "threshold_chars", None) or getattr(
        settings, "tool_result_offload_threshold_chars", 0
    )
    if not threshold or len(output_text) <= int(threshold):
        return output_text, None

    try:
        from uuid import UUID

        offload = service.offload_if_large(
            conversation_id=UUID(str(conversation_id)),
            user_id=UUID(str(user_id)),
            tool_call_id=tool_call_id,
            tool_name=tool_name or "unknown",
            output_text=output_text,
        )
    except Exception as exc:
        logger.warning(
            "Failed to offload large tool output for %s: %s",
            tool_name,
            exc,
        )
        return output_text, None

    blob_id = offload.get("blob_id")
    if not blob_id:
        return output_text, None

    return offload.get("output", output_text), {
        "blob_id": blob_id,
        "blob_size_bytes": offload.get("size_bytes"),
        "output_truncated": True,
    }


def _apply_offload_to_outputs_and_artifacts(
    *,
    outputs: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    conversation_id: str | None,
    user_id: str | None,
    offload_service: Any | None = None,
) -> None:
    """Walk paired outputs/artifacts and offload any large outputs in-place."""

    if not outputs or not artifacts:
        return
    service = offload_service if offload_service is not None else _resolve_offload_service()
    if service is None or not conversation_id or not user_id:
        return

    artifact_index = {
        artifact.get("tool_call_id"): artifact
        for artifact in artifacts
        if artifact.get("tool_call_id")
    }

    for output in outputs:
        # The Planning supervisor must receive complete worker answers from
        # dispatch_subagents; replacing that ToolMessage with a blob preview
        # would hide the result it needs to reconcile todos. Skill terminal
        # errors carry the same flag: their content must reach the model uncut.
        if output.get("name") in _FULL_MODEL_HANDOFF_TOOLS or output.get("preserve_full_content"):
            continue
        tool_call_id = output.get("tool_call_id")
        public_text, blob_info = apply_tool_output_offload(
            output_text=output.get("content"),
            tool_call_id=tool_call_id,
            tool_name=output.get("name"),
            conversation_id=conversation_id,
            user_id=user_id,
            offload_service=service,
        )
        if blob_info is None:
            continue
        output["content"] = public_text
        artifact = artifact_index.get(tool_call_id)
        if artifact is not None:
            artifact["output"] = public_text
            artifact.update(blob_info)


def build_tool_artifact(
    *,
    tool_call_id: str | None,
    tool_name: str,
    tool_args: Any,
    output_text: str | None,
    error: str | None,
    status: str | None = None,
    max_output_chars: int = 0,
    render: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "tool_call_id": tool_call_id,
        "tool": tool_name,
        "args": tool_args,
        "output": None,
        "error": error,
        "status": status or ("error" if error else "success"),
    }

    if output_text is not None:
        if tool_name in _WIDGET_ARTIFACT_TOOLS:
            output_text = _compact_widget_artifact_output(output_text)
        artifact["output"] = (
            output_text[:max_output_chars]
            if max_output_chars > 0 and len(output_text) > max_output_chars
            else output_text
        )

    if render is not None:
        artifact["render"] = render

    return artifact


def _compact_widget_artifact_output(output_text: str) -> str:
    """Store a compact, parseable widget descriptor in artifacts.

    Widget tool results can include a large ``state`` payload. The backend only
    needs the stable mount metadata to derive ``live_widgets``, so keep a small
    JSON object here to avoid truncating the artifact into invalid JSON.
    """
    try:
        parsed = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        return output_text

    if not isinstance(parsed, dict) or not parsed.get("widget_id"):
        return output_text

    compact = {
        "widget_id": parsed.get("widget_id"),
        "session_id": parsed.get("session_id", ""),
        "title": parsed.get("title"),
        "status": parsed.get("status", "active"),
        "version": parsed.get("version", 1),
    }
    return json.dumps(compact, separators=(",", ":"), ensure_ascii=False)


def _bind_widget_session_args(
    tool_name: str,
    tool_args: Any,
    conversation_id: str | None,
) -> Any:
    """Bind widget session-scoped tools to the active conversation.

    Widget tools operate inside the current conversation. Models may still emit
    placeholders like ``current_session`` or stale IDs, so normalize those
    arguments here before the MCP tool is invoked.
    """
    if tool_name not in _WIDGET_SESSION_BOUND_TOOLS:
        return tool_args
    if not conversation_id or not isinstance(tool_args, dict):
        return tool_args

    bound_conversation_id = str(conversation_id)
    current_session_id = tool_args.get("session_id")
    if current_session_id == bound_conversation_id:
        return tool_args

    bound_args = dict(tool_args)
    bound_args["session_id"] = bound_conversation_id

    if current_session_id not in (None, "", bound_conversation_id):
        logger.debug(
            "Binding widget tool '%s' session_id from %r to active conversation %s",
            tool_name,
            current_session_id,
            bound_conversation_id,
        )

    return bound_args


def build_rejected_tool_artifacts(
    *,
    tool_calls: list[Any],
    rejected_feedback: dict[str, str],
    max_output_chars: int = 0,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []

    for raw_tool_call in tool_calls:
        tool_call = normalize_tool_call(raw_tool_call)
        tool_call_id = tool_call.get("id")
        if not tool_call_id or tool_call_id not in rejected_feedback:
            continue

        artifacts.append(
            build_tool_artifact(
                tool_call_id=tool_call_id,
                tool_name=tool_call.get("name") or "unknown",
                tool_args=tool_call.get("args", {}),
                output_text=rejected_feedback[tool_call_id],
                error=None,
                status="rejected",
                max_output_chars=max_output_chars,
            )
        )

    return artifacts


def _validate_client_tool_device_binding(
    tool: Any,
    context_device_id: str | None,
    tool_name: str,
) -> str | None:
    """
    Validate that a client tool is being executed on its bound device.

    This provides defense-in-depth for client tool isolation. The primary
    validation happens inside the tool's dispatch function, but this catches
    mismatches earlier in the execution path.

    Args:
        tool: The tool being executed
        context_device_id: The device_id from the current execution context
        tool_name: The tool name (for error messages)

    Returns:
        Error message if validation fails, None if validation passes
    """
    if not is_client_tool(tool):
        return None

    bound_device_id = get_client_tool_device_id(tool)
    if not bound_device_id:
        return None

    if not context_device_id:
        return (
            f"Client tool '{tool_name}' requires a device context but none was provided. "
            "Ensure the request includes device_id."
        )

    if str(context_device_id) != str(bound_device_id):
        logger.warning(
            "Client tool device mismatch: tool '%s' bound to device %s but context has device %s",
            tool_name,
            bound_device_id,
            context_device_id,
        )
        return (
            f"Client tool '{tool_name}' is bound to a different device. "
            "This tool cannot be executed from the current device session."
        )

    return None


async def ensure_agent_tool_map(
    agent: Any,
    conversation_id: str | None = None,
    user_id: str | None = None,
    device_id: str | None = None,
    tool_scope: str | None = None,
    internal_tools: list[Any] | None = None,
) -> dict[str, Any]:
    """
    Build a tool map for executing tool calls.

    In deferred mode, the execution map is built from the same reduced tool set
    used for model binding (tool_search + pinned + loaded + required internal tools).
    In non-deferred mode, it falls back to all initialized agent tools.

    Client tools are added separately and are always scoped to a specific device.
    Tool discovery may surface both server MCP tools and device-scoped client
    tools, but execution still keeps client tools bound to the active device
    session.

    Args:
        agent: The agent instance
        conversation_id: Optional conversation ID for deferred tool lookup
        user_id: Optional user ID for client tool lookup
        device_id: Optional device ID for client tool scoping

    Returns:
        Dict mapping tool names to tool objects
    """
    if not agent:
        return {}

    initialized_tools = getattr(agent, "tools", None) or []
    client_only_scope = is_client_only_scope(device_id=device_id, tool_scope=tool_scope)
    if not initialized_tools:
        if hasattr(agent, "_init_mcp"):
            await agent._init_mcp()
        elif hasattr(agent, "_init_tools"):
            await agent._init_tools()
        initialized_tools = getattr(agent, "tools", None) or []

    tools = initialized_tools

    # Keep execution permissions aligned with the tools that were actually
    # exposed to the model for this turn.
    if hasattr(agent, "_get_tools_for_binding"):
        try:
            tools = agent._get_tools_for_binding(
                conversation_id=conversation_id,
                user_id=user_id,
                device_id=device_id,
                tool_scope=tool_scope,
                internal_tools=internal_tools,
            )
        except TypeError:
            # Backward-compat fallback for non-keyword signatures.
            tools = agent._get_tools_for_binding(conversation_id)
        manages_client_tools = True
    elif settings.mcp_tool_search_enabled:
        # Safety fallback for custom agents that don't implement binding helpers.
        agent_key = getattr(agent, "agent_config_key", None)
        allowlist = None
        if agent_key:
            allowlist_key = f"{agent_key}_agent_allowed_tools"
            allowlist = getattr(settings, allowlist_key, None) or []
        tool_search = create_tool_search_tool(allowlist=allowlist)
        tools = [] if client_only_scope else list(initialized_tools)
        if not any(getattr(t, "name", None) == tool_search.name for t in tools):
            tools.append(tool_search)
        manages_client_tools = False
    else:
        tools = [] if client_only_scope else list(tools)
        manages_client_tools = False

    # Add client runtime tools (device-scoped, separate from server MCP tools)
    if not manages_client_tools and hasattr(agent, "_get_client_runtime_tools"):
        try:
            remote_tools = agent._get_client_runtime_tools(user_id=user_id, device_id=device_id)
        except TypeError:
            remote_tools = agent._get_client_runtime_tools(user_id=user_id, device_id=device_id)
        existing_names = {getattr(t, "name", None) for t in tools}
        for tool in remote_tools:
            tool_name = getattr(tool, "name", None)
            if tool_name and tool_name not in existing_names:
                tools.append(tool)
                existing_names.add(tool_name)

    tool_map = {t.name: t for t in tools if getattr(t, "name", None)}

    return tool_map


def _mark_tool_used_if_deferred(tool_name: str) -> None:
    """
    Mark a tool as used in the deferred tool state if applicable.

    This updates the LRU timestamp so frequently-used tools are less
    likely to be evicted.

    Args:
        tool_name: The name of the tool that was executed
    """
    if not settings.mcp_tool_search_enabled:
        return

    from .deferred_tool_state import get_deferred_tool_state
    from .tool_context import get_tool_context

    ctx = get_tool_context()
    if not ctx.conversation_id:
        return

    state = get_deferred_tool_state()
    state.mark_tool_used(ctx.conversation_id, ctx.agent_key, tool_name)


async def _refresh_tool_map_after_search(
    tool_map: dict[str, Any],
    agent: Any | None,
    conversation_id: str | None,
    user_id: str | None,
    device_id: str | None,
    tool_scope: str | None = None,
) -> None:
    """
    Refresh the tool map after tool_search executes to include newly loaded tools.

    This is critical for multi-client scenarios where tool_search autoloads tools
    that the model then tries to call in the same turn. Without this refresh,
    the tool map wouldn't include the newly loaded tools.

    Args:
        tool_map: The existing tool map to update in-place
        agent: The agent instance (may be None)
        conversation_id: Current conversation ID
        user_id: Current user ID
        device_id: Current device ID
    """
    if not settings.mcp_tool_search_enabled:
        return

    from .client_runtime_tools import get_client_runtime_tools
    from .deferred_tool_binding import get_deferred_tools_for_binding
    from .deferred_tool_state import get_deferred_tool_state
    from .mcp_registry import get_global_mcp_manager

    if not conversation_id:
        return

    try:
        mcp_manager = await get_global_mcp_manager()
        client_only_scope = is_client_only_scope(device_id=device_id, tool_scope=tool_scope)

        # Get the agent key for looking up loaded tools. Custom agents bind
        # loaded tools under tool_state_key, so deferred-state reads must use it
        # (not the shared agent_config_key) to find the right tools.
        agent_key = "default"
        if agent:
            agent_key = (
                getattr(agent, "tool_state_key", None)
                or getattr(agent, "agent_config_key", None)
                or "default"
            )

        # Custom agents must refresh from their own restricted binding helper so
        # stale or previously loaded tools outside the current tool_refs policy
        # cannot appear in the same-turn execution map.
        if getattr(agent, "spec", None) is not None and hasattr(agent, "_get_tools_for_binding"):
            try:
                refreshed_tools = agent._get_tools_for_binding(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    device_id=device_id,
                    tool_scope=tool_scope,
                )
            except TypeError:
                refreshed_tools = agent._get_tools_for_binding(conversation_id)

            for tool in refreshed_tools:
                tool_name = getattr(tool, "name", None)
                if tool_name and tool_name not in tool_map:
                    tool_map[tool_name] = tool
            return

        # Get all MCP tools from the manager
        all_mcp_tools = await mcp_manager.get_tools() if mcp_manager else []

        # Get the newly loaded deferred server tools
        if not client_only_scope:
            deferred_tools = get_deferred_tools_for_binding(
                conversation_id=conversation_id,
                agent_key=agent_key,
                mcp_manager=mcp_manager,
                all_tools=all_mcp_tools,
            )

            # Add any new deferred tools to the map
            for tool in deferred_tools:
                tool_name = getattr(tool, "name", None)
                if tool_name and tool_name not in tool_map:
                    tool_map[tool_name] = tool
                    logger.debug(
                        "Added newly loaded deferred tool '%s' to tool map",
                        tool_name,
                    )

        # Also refresh client tools (they may have been loaded via tool_search)
        state = get_deferred_tool_state()
        active_session = get_active_client_runtime_session(
            user_id=user_id,
            device_id=device_id,
        )
        loaded_client_tools = state.get_loaded_client_tools(
            conversation_id,
            agent_key,
            device_id=device_id,
            session_id=active_session.session_id if active_session is not None else None,
        )

        if loaded_client_tools and device_id:
            # Get fresh client tools and add any that match loaded references
            client_tools = get_client_runtime_tools(user_id=user_id, device_id=device_id)
            for client_tool in client_tools:
                tool_name = getattr(client_tool, "name", None)
                if tool_name and tool_name not in tool_map:
                    # Check if this tool was loaded by tool_search
                    for loaded in loaded_client_tools:
                        if loaded.tool_name == tool_name:
                            tool_map[tool_name] = client_tool
                            logger.debug(
                                "Added newly loaded client tool '%s' to tool map",
                                tool_name,
                            )
                            break
    except Exception as e:
        # Don't fail the entire tool execution if refresh fails
        logger.warning("Failed to refresh tool map after tool_search: %s", e)


async def _recover_missing_tool(
    *,
    tool_name: str,
    tool_map: dict[str, Any],
    agent: Any | None,
    conversation_id: str | None,
    user_id: str | None,
    device_id: str | None,
    tool_scope: str | None = None,
) -> Any | None:
    """
    Best-effort recovery for exact-name tool calls that are absent from the
    current execution map.

    This primarily covers client tools discovered through `tool_search` where
    the model later calls the exact `client__...` name but the execution map
    was built from stale scope state.
    """
    if not tool_name:
        return None

    if settings.mcp_tool_search_enabled and conversation_id:
        await _refresh_tool_map_after_search(
            tool_map=tool_map,
            agent=agent,
            conversation_id=conversation_id,
            user_id=user_id,
            device_id=device_id,
            tool_scope=tool_scope,
        )
        recovered = tool_map.get(tool_name)
        if recovered is not None:
            return recovered

    # Server tool recovery: look up by exact name in the live MCP manager.
    # Covers the case where a server MCP tool was autoloaded via tool_search,
    # the graph paused for HITL, and the in-memory deferred state was lost
    # (server restart, process migration, or LRU eviction past the approval
    # wait window). The tool is still live on the manager, so we can rebind
    # it directly by name.
    if not tool_name.startswith(CLIENT_TOOL_PREFIX):
        if is_client_only_scope(device_id=device_id, tool_scope=tool_scope):
            return None

        # Alias-aware recovery first: the model may call the public alias
        # (e.g. "brave__search") while the live MCP manager exposes the raw name
        # ("search"). Resolve the raw name + server from the loaded deferred
        # state, then rebind under the public alias.
        try:
            from .deferred_tool_binding import _tool_with_call_name
            from .deferred_tool_state import get_deferred_tool_state

            agent_key = (
                getattr(agent, "tool_state_key", None)
                or getattr(agent, "agent_config_key", None)
                or "default"
            )
            state = get_deferred_tool_state()
            server_name = state.get_server_for_loaded_tool(conversation_id, agent_key, tool_name)
            raw_tool_name = state.get_raw_tool_name_for_loaded_tool(
                conversation_id, agent_key, tool_name
            )
            if raw_tool_name and server_name:
                from .mcp_registry import get_global_mcp_manager

                manager = await get_global_mcp_manager()
                if manager is not None:
                    for server_tool in await manager.get_tools():
                        if (
                            getattr(server_tool, "name", None) == raw_tool_name
                            and manager.get_server_for_tool(server_tool) == server_name
                        ):
                            tool_map[tool_name] = _tool_with_call_name(server_tool, tool_name)
                            logger.debug(
                                "Recovered aliased server tool '%s' (raw '%s', server '%s')",
                                tool_name,
                                raw_tool_name,
                                server_name,
                            )
                            return tool_map[tool_name]
        except Exception as exc:
            logger.warning("Failed alias-aware recovery for '%s': %s", tool_name, exc)

        if settings.mcp_tool_search_enabled and conversation_id:
            logger.debug(
                "Skipping raw server-tool recovery for unloaded deferred tool '%s'",
                tool_name,
            )
            return None

        try:
            from .mcp_registry import get_global_mcp_manager

            manager = await get_global_mcp_manager()
            if manager is not None:
                for server_tool in await manager.get_tools():
                    if getattr(server_tool, "name", None) == tool_name:
                        tool_map[tool_name] = server_tool
                        logger.debug(
                            "Recovered missing server tool '%s' from MCP manager",
                            tool_name,
                        )
                        return server_tool
        except Exception as exc:
            logger.warning("Failed recovering missing server tool '%s': %s", tool_name, exc)
        return None

    if not device_id:
        return None

    try:
        for client_tool in get_client_runtime_tools(user_id=user_id, device_id=device_id):
            candidate_name = getattr(client_tool, "name", None)
            if candidate_name != tool_name:
                continue
            tool_map[tool_name] = client_tool
            logger.debug(
                "Recovered missing client tool '%s' from active runtime catalog", tool_name
            )
            return client_tool
    except Exception as exc:
        logger.warning("Failed recovering missing client tool '%s': %s", tool_name, exc)

    return None


async def invoke_tool(tool: Any, tool_args: Any) -> Any:
    """Execute a tool, preferring async paths to avoid blocking the event loop.

    Priority: coroutine attr → ainvoke → invoke (via asyncio.to_thread) → callable.
    Sync fallbacks are wrapped in ``asyncio.to_thread`` so MCP or other I/O-bound
    tools never block the running event loop.
    """
    if getattr(tool, "coroutine", None):
        return await tool.ainvoke(tool_args)

    ainvoke = getattr(tool, "ainvoke", None)
    if callable(ainvoke):
        return await ainvoke(tool_args)

    invoke = getattr(tool, "invoke", None)
    if callable(invoke):
        return await asyncio.to_thread(invoke, tool_args)

    if callable(tool):
        return await asyncio.to_thread(tool, tool_args)

    raise TypeError("Tool has no invoke/ainvoke and is not callable")


@dataclass(frozen=True)
class AttemptOutcome:
    result: Any = None
    exception: BaseException | None = None
    elapsed_ms: int = 0
    cancellation_attempted: bool = False
    cancellation_completed: bool = False
    timeout_phase: str = "none"


def _consume_abandoned_task_exception(task: asyncio.Task[Any]) -> None:
    """Retrieve a terminal exception from work abandoned at its hard deadline."""
    with suppress(asyncio.CancelledError):
        task.exception()


def _cancel_and_consume_task(task: asyncio.Task[Any]) -> None:
    """Cancel child work and consume its terminal state without awaiting it."""
    task.cancel()
    task.add_done_callback(_consume_abandoned_task_exception)


async def invoke_tool_attempt(
    tool: Any,
    tool_args: Any,
    *,
    policy: ToolExecutionPolicy,
    remaining_total_seconds: float | None,
) -> AttemptOutcome:
    """Invoke one tool attempt under soft-cancel and hard-abandon deadlines."""
    started_at = time.monotonic()
    if remaining_total_seconds is None or remaining_total_seconds == math.inf:
        cumulative_seconds = None
    elif math.isnan(remaining_total_seconds) or remaining_total_seconds == -math.inf:
        cumulative_seconds = 0.0
    else:
        cumulative_seconds = max(0.0, remaining_total_seconds)

    if cumulative_seconds == 0:
        return AttemptOutcome(
            exception=TimeoutError("Tool cumulative deadline exhausted before attempt start"),
            elapsed_ms=max(0, round((time.monotonic() - started_at) * 1000)),
            timeout_phase="hard_timeout",
        )

    if policy.outer_timeout_disabled:
        soft_seconds = cumulative_seconds
        hard_seconds = cumulative_seconds
    else:
        soft_seconds = policy.timeout_seconds
        hard_seconds = policy.hard_timeout_seconds
        if cumulative_seconds is not None:
            soft_seconds = min(soft_seconds, cumulative_seconds)
            hard_seconds = min(hard_seconds, cumulative_seconds)

    hard_deadline = started_at + hard_seconds if hard_seconds is not None else None
    task = asyncio.create_task(invoke_tool(tool, tool_args))

    try:
        done, _ = await asyncio.wait({task}, timeout=soft_seconds)
    except asyncio.CancelledError:
        _cancel_and_consume_task(task)
        raise
    if done:
        try:
            result = task.result()
        except BaseException as exc:
            return AttemptOutcome(
                exception=exc,
                elapsed_ms=max(0, round((time.monotonic() - started_at) * 1000)),
            )
        return AttemptOutcome(
            result=result,
            elapsed_ms=max(0, round((time.monotonic() - started_at) * 1000)),
        )

    task.cancel()
    hard_wait_seconds = (
        None if hard_deadline is None else max(0.0, hard_deadline - time.monotonic())
    )
    try:
        done, _ = await asyncio.wait({task}, timeout=hard_wait_seconds)
    except asyncio.CancelledError:
        _cancel_and_consume_task(task)
        raise
    cancellation_completed = policy.cancellation != "abandon_only"

    if done:
        try:
            task.result()
        except asyncio.CancelledError:
            timeout_exception: BaseException = TimeoutError(f"Tool timed out after {soft_seconds}s")
        except BaseException as exc:
            return AttemptOutcome(
                exception=exc,
                elapsed_ms=max(0, round((time.monotonic() - started_at) * 1000)),
                cancellation_attempted=True,
                cancellation_completed=cancellation_completed,
                timeout_phase="soft_timeout",
            )
        else:
            timeout_exception = TimeoutError(f"Tool timed out after {soft_seconds}s")

        return AttemptOutcome(
            exception=timeout_exception,
            elapsed_ms=max(0, round((time.monotonic() - started_at) * 1000)),
            cancellation_attempted=True,
            cancellation_completed=cancellation_completed,
            timeout_phase="soft_timeout",
        )

    task.add_done_callback(_consume_abandoned_task_exception)
    return AttemptOutcome(
        exception=TimeoutError(
            f"Tool timed out after {soft_seconds}s and did not stop before {hard_seconds}s"
        ),
        elapsed_ms=max(0, round((time.monotonic() - started_at) * 1000)),
        cancellation_attempted=True,
        cancellation_completed=False,
        timeout_phase="hard_timeout",
    )


def _structured_tool_error(result: Any) -> str | None:
    """Recognize MCP and repository-standard error result envelopes."""
    safe_result = make_json_safe(result)
    if not isinstance(safe_result, dict):
        return None

    is_error = safe_result.get("isError") is True or safe_result.get("is_error") is True
    error_value = safe_result.get("error")
    if not is_error and error_value in (None, "", False):
        return None

    if error_value not in (None, "", False):
        if isinstance(error_value, str):
            return error_value.strip() or "Tool execution failed"
        try:
            return json.dumps(error_value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(error_value)

    content = safe_result.get("content")
    blocks = content if isinstance(content, list) else [content]
    text_parts = [
        str(block.get("text")).strip()
        for block in blocks
        if isinstance(block, dict)
        and str(block.get("type") or "").lower() == "text"
        and str(block.get("text") or "").strip()
    ]
    return "\n".join(text_parts) or str(safe_result.get("message") or "Tool execution failed")


async def invoke_tool_with_policy(
    tool: Any,
    tool_args: Any,
    *,
    tool_name: str,
    tool_map: dict[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any] | None, str | None, dict[str, Any]]:
    """Invoke retries and server-MCP reconnects under one resolved policy deadline."""
    invocation_kind = _tool_invocation_kind(tool)
    try:
        policy = resolve_tool_execution_policy(
            tool,
            exposed_tool_name=tool_name,
            invocation_kind=invocation_kind,
        )
    except (AmbiguousToolExecutionPolicyError, ToolExecutionPolicyValidationError) as exc:
        logger.error("Invalid tool execution policy for %s: %s", tool_name, exc)
        summary = ToolErrorSummary(
            error_type="configuration",
            failure_retryable=False,
            message=("Tool execution is unavailable because its server policy is invalid."),
            hint="Use another available tool or report the configuration problem.",
            attempts=0,
        )
        model_content, artifact_detail = build_tool_error_payloads(
            summary,
            tool_name=tool_name,
            exception=exc,
            policy_retry_allowed=False,
        )
        return (
            None,
            artifact_detail,
            model_content,
            {
                "attempts": 0,
                "attempt_history": [],
            },
        )
    policy_snapshot = _policy_snapshot(policy)
    policy_retry_allowed = (policy.retry_safe or policy.idempotent) or (
        policy.identity.tool_origin,
        policy.identity.qualified_tool_id,
    ) in _RETRY_COMPATIBILITY_ALLOWLIST
    started_at = time.monotonic()
    attempts = 0
    attempt_history: list[dict[str, Any]] = []
    current_tool = tool

    # A closed in-memory MCP send stream rejects the request before it can be
    # handed to the server. It is therefore safe to refresh that transport once
    # even for a mutating tool whose ordinary retry policy is intentionally
    # disabled. This is distinct from ``BrokenResourceError``, which can occur
    # after transport progress and must still respect retry_safe/idempotent.
    pre_send_session_retry_limit = policy.max_attempts + 1

    while attempts < pre_send_session_retry_limit:
        remaining_total_seconds = (
            None
            if policy.outer_timeout_disabled
            else max(
                0.0,
                policy.total_timeout_seconds - (time.monotonic() - started_at),
            )
        )
        attempts += 1
        with tool_policy_context(policy):
            outcome = await invoke_tool_attempt(
                current_tool,
                tool_args,
                policy=policy,
                remaining_total_seconds=remaining_total_seconds,
            )

        if outcome.exception is None:
            attempt_history.append(
                _attempt_record(
                    policy=policy,
                    tool_name=tool_name,
                    attempt=attempts,
                    outcome=outcome,
                    summary=None,
                    policy_retry_allowed=policy_retry_allowed,
                    auto_retry_allowed=False,
                )
            )
            diagnostics = {
                "attempts": attempts,
                "attempt_history": attempt_history[-5:],
                "policy": policy_snapshot,
            }
            return outcome.result, None, None, diagnostics

        summary = classify_tool_error(
            outcome.exception,
            tool_name=tool_name,
            timeout_seconds=policy.timeout_seconds,
            attempts=attempts,
        )
        remaining_after_attempt = max(
            0.0,
            policy.total_timeout_seconds - (time.monotonic() - started_at),
        )
        ordinary_retry_allowed = (
            summary.failure_retryable
            and policy_retry_allowed
            and attempts < policy.max_attempts
            and remaining_after_attempt > 0
        )
        pre_send_session_retry_allowed = (
            isinstance(outcome.exception, ClosedResourceError)
            and policy.identity.tool_origin == "server_mcp"
            and not policy_retry_allowed
            and attempts == 1
            and attempts < pre_send_session_retry_limit
            and remaining_after_attempt > 0
        )
        auto_retry_allowed = ordinary_retry_allowed or pre_send_session_retry_allowed
        attempt_history.append(
            _attempt_record(
                policy=policy,
                tool_name=tool_name,
                attempt=attempts,
                outcome=outcome,
                summary=summary,
                policy_retry_allowed=policy_retry_allowed,
                auto_retry_allowed=auto_retry_allowed,
            )
        )

        if auto_retry_allowed:
            if (
                summary.error_type == ToolErrorKind.SESSION.value
                and policy.identity.tool_origin == "server_mcp"
            ):
                current_tool, reconnect_exc = await _reconnect_server_tool(
                    tool_name,
                    remaining_total_seconds=remaining_after_attempt,
                )
                if reconnect_exc is not None or current_tool is None:
                    final_exc = reconnect_exc or RuntimeError("MCP reconnect returned no tool")
                    attempts += 1
                    summary = classify_tool_error(
                        final_exc,
                        tool_name=tool_name,
                        timeout_seconds=policy.timeout_seconds,
                        attempts=attempts,
                    )
                    outcome = AttemptOutcome(exception=final_exc)
                    attempt_history.append(
                        _attempt_record(
                            policy=policy,
                            tool_name=tool_name,
                            attempt=attempts,
                            outcome=outcome,
                            summary=summary,
                            policy_retry_allowed=policy_retry_allowed,
                            auto_retry_allowed=False,
                        )
                    )
                else:
                    if tool_map is not None:
                        tool_map[tool_name] = current_tool
                    continue
            else:
                continue

        model_content, artifact_detail = build_tool_error_payloads(
            summary,
            tool_name=tool_name,
            exception=outcome.exception,
            policy_retry_allowed=policy_retry_allowed,
        )
        artifact_detail["attempt_history"] = attempt_history[-5:]
        artifact_detail["policy"] = policy_snapshot
        return (
            None,
            artifact_detail,
            model_content,
            {
                "attempts": attempts,
                "attempt_history": attempt_history[-5:],
                "policy": policy_snapshot,
            },
        )

    raise RuntimeError("Tool execution attempt loop ended without an outcome")


def _tool_invocation_kind(tool: Any) -> str:
    metadata = getattr(tool, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("tool_origin") in {
        "client_mcp",
        "client_skill",
    }:
        return "client_runtime"
    if getattr(tool, "coroutine", None) or callable(getattr(tool, "ainvoke", None)):
        return "native_async"
    return "sync_thread"


def _policy_snapshot(policy: ToolExecutionPolicy) -> dict[str, Any]:
    identity = policy.identity
    return make_json_safe(
        {
            "tool_origin": identity.tool_origin,
            "qualified_tool_id": identity.qualified_tool_id,
            "exposed_tool_name": identity.exposed_tool_name,
            "source_tool_name": identity.source_tool_name,
            "server_name": identity.server_name,
            "timeout_seconds": policy.timeout_seconds,
            "hard_timeout_seconds": policy.hard_timeout_seconds,
            "total_timeout_seconds": policy.total_timeout_seconds,
            "max_attempts": policy.max_attempts,
            "retry_safe": policy.retry_safe,
            "idempotent": policy.idempotent,
            "metadata_trusted": policy.metadata_trusted,
            "outer_timeout_disabled": policy.outer_timeout_disabled,
            "cancellation": policy.cancellation,
            "client_execution_timeout_seconds": (policy.client_execution_timeout_seconds),
            "client_response_timeout_seconds": policy.client_response_timeout_seconds,
            "policy_source": policy.policy_source,
            "policy_config_keys": list(policy.policy_config_keys),
            "timeout_hint": policy.timeout_hint,
        }
    )


def _attempt_record(
    *,
    policy: ToolExecutionPolicy,
    tool_name: str,
    attempt: int,
    outcome: AttemptOutcome,
    summary: ToolErrorSummary | None,
    policy_retry_allowed: bool,
    auto_retry_allowed: bool,
) -> dict[str, Any]:
    record = {
        "attempt": attempt,
        "tool_name": tool_name,
        "tool_origin": policy.identity.tool_origin,
        "qualified_tool_id": policy.identity.qualified_tool_id,
        "policy_source": policy.policy_source,
        "policy_config_keys": list(policy.policy_config_keys),
        "metadata_trusted": policy.metadata_trusted,
        "timeout_seconds": policy.timeout_seconds,
        "hard_timeout_seconds": policy.hard_timeout_seconds,
        "total_timeout_seconds": policy.total_timeout_seconds,
        "elapsed_ms": outcome.elapsed_ms,
        "failure_retryable": summary.failure_retryable if summary else False,
        "policy_retry_allowed": policy_retry_allowed,
        "auto_retry_allowed": auto_retry_allowed,
        "error_type": summary.error_type if summary else None,
        "retryable": (summary.failure_retryable and policy_retry_allowed if summary else False),
        "cancellation": policy.cancellation,
        "cancellation_attempted": outcome.cancellation_attempted,
        "cancellation_completed": outcome.cancellation_completed,
        "timeout_phase": outcome.timeout_phase,
    }
    logger.info(
        "tool_execution_attempt",
        extra={"tool_execution": make_json_safe(record)},
    )
    return record


async def _reconnect_server_tool(
    tool_name: str,
    *,
    remaining_total_seconds: float,
) -> tuple[Any | None, BaseException | None]:
    async def _reconnect() -> Any:
        from .mcp_registry import get_global_mcp_manager

        manager = await get_global_mcp_manager()
        return await manager.reconnect_and_get_tool(tool_name)

    if remaining_total_seconds <= 0:
        return None, TimeoutError("Tool cumulative deadline exhausted before reconnect")

    task = asyncio.create_task(_reconnect())
    try:
        done, _ = await asyncio.wait({task}, timeout=remaining_total_seconds)
    except asyncio.CancelledError:
        _cancel_and_consume_task(task)
        raise
    if not done:
        _cancel_and_consume_task(task)
        return None, TimeoutError("MCP reconnect exceeded the cumulative tool deadline")
    try:
        return task.result(), None
    except BaseException as exc:
        return None, exc


async def execute_tool_calls(
    *,
    tool_calls: list[Any],
    tool_map: dict[str, Any],
    capture_images: bool = True,
    artifact_max_output_chars: int = 0,
    device_id: str | None = None,
    agent: Any | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
    tool_scope: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    """
    Execute a list of tool calls and return outputs, artifacts, and images.

    For client tools (those with names starting with CLIENT_TOOL_PREFIX), the
    device_id parameter is used to validate that the tool is being executed
    on its bound device.

    When tool_search is among the tool calls, this function will automatically
    refresh the tool_map after tool_search executes to include newly loaded
    deferred tools. This ensures that tools discovered via tool_search can be
    called in the same turn without a second round-trip.

    Args:
        tool_calls: List of tool call objects to execute
        tool_map: Dict mapping tool names to tool objects (modified in-place if refresh needed)
        capture_images: Whether to extract images from tool results
        artifact_max_output_chars: Optional artifact preview cap (0 preserves the
            full result until the configured blob offload stage).
        device_id: Current device_id for client tool validation
        agent: Optional agent instance for tool map refresh
        conversation_id: Optional conversation ID for tool map refresh
        user_id: Optional user ID for tool map refresh

    Returns:
        Tuple of (outputs, artifacts, images).

    The list of public rich-item candidates produced during execution is
    attached to each artifact as ``artifact["_rich_item_candidates"]`` so the
    graph layer can lift them into ``context["rich_item_candidates"]`` without
    a separate return-shape change. The key is intentionally prefixed with an
    underscore so existing consumers (renderers, persistence) ignore it.
    """
    outputs: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    images: list[dict[str, str]] = []

    def _append_tool_error_output(
        *,
        tool_call_id: str | None,
        tool_name: str,
        tool_args: Any,
        model_content: str,
        artifact_detail: dict[str, Any],
    ) -> None:
        """Append a compact-error output/artifact pair with an error render.

        Shared by missing-name, missing-tool, device-binding, policy-timeout,
        and exception failures so every error path renders identically.
        """
        normalized = normalize_tool_result_for_rendering(model_content, tool_name=tool_name)
        render = dict(normalized.render)
        render["type"] = "error"
        render["error"] = str(
            artifact_detail.get("diagnostic")
            or artifact_detail.get("error_type")
            or "Tool execution failed"
        )
        output: dict[str, Any] = {
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": model_content,
            "render": render,
        }
        if artifact_detail.get("skill_terminal_error"):
            # The complete skill terminal error must reach the model verbatim,
            # bypassing ordinary truncation and blob offload.
            output["preserve_full_content"] = True
        outputs.append(output)
        artifact = build_tool_artifact(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_args=tool_args,
            output_text=model_content,
            error=str(artifact_detail.get("diagnostic") or artifact_detail.get("error_type")),
            max_output_chars=artifact_max_output_chars,
            render=render,
        )
        artifact.update(artifact_detail)
        artifacts.append(artifact)

    for raw_tool_call in tool_calls:
        tool_call = normalize_tool_call(raw_tool_call)
        tool_name = tool_call.get("name")
        tool_id = tool_call.get("id")
        tool_args = tool_call.get("args", {})
        tool_args = _bind_widget_session_args(tool_name or "", tool_args, conversation_id)

        # ``normalize_tool_call`` substitutes the sentinel "unknown" when no name
        # was provided, so an absent/blank name surfaces as that sentinel rather
        # than a falsy value.
        if not tool_name or tool_name == "unknown":
            summary = ToolErrorSummary(
                error_type=ToolErrorKind.ARGUMENT.value,
                failure_retryable=False,
                message="Tool name is missing.",
                hint="Provide the tool name to call, or inspect available tools if unsure.",
                attempts=1,
            )
            model_content, artifact_detail = build_tool_error_payloads(
                summary,
                tool_name=tool_name or "unknown",
                exception=ValueError("Tool name is missing."),
                policy_retry_allowed=False,
            )
            _append_tool_error_output(
                tool_call_id=tool_id,
                tool_name=tool_name or "unknown",
                tool_args=tool_args,
                model_content=model_content,
                artifact_detail=artifact_detail,
            )
            continue

        tool = tool_map.get(tool_name)
        if not tool:
            tool = await _recover_missing_tool(
                tool_name=tool_name,
                tool_map=tool_map,
                agent=agent,
                conversation_id=conversation_id,
                user_id=user_id,
                device_id=device_id,
                tool_scope=tool_scope,
            )
        if not tool:
            if tool_name.startswith(CLIENT_TOOL_PREFIX):
                summary = ToolErrorSummary(
                    error_type=ToolErrorKind.NOT_FOUND.value,
                    failure_retryable=False,
                    message=(
                        f"Client tool {tool_name} is not available for the current device session."
                    ),
                    hint=(
                        "The device may be disconnected. Ask the user to reconnect, "
                        "or use another available tool."
                    ),
                    attempts=1,
                )
            else:
                summary = ToolErrorSummary(
                    error_type=ToolErrorKind.NOT_FOUND.value,
                    failure_retryable=False,
                    message=f"Tool {tool_name} is not currently bound.",
                    hint=(
                        "Use a currently bound suitable tool if one exists. If the needed "
                        "capability is missing or ambiguous, use tool_search to discover it."
                    ),
                    attempts=1,
                )
            model_content, artifact_detail = build_tool_error_payloads(
                summary,
                tool_name=tool_name,
                exception=LookupError(summary.message),
                policy_retry_allowed=False,
            )
            _append_tool_error_output(
                tool_call_id=tool_id,
                tool_name=tool_name,
                tool_args=tool_args,
                model_content=model_content,
                artifact_detail=artifact_detail,
            )
            continue

        # Validate client tool device binding before execution
        device_error = _validate_client_tool_device_binding(tool, device_id, tool_name)
        if device_error:
            summary = ToolErrorSummary(
                error_type=ToolErrorKind.PERMISSION.value,
                failure_retryable=False,
                message=(
                    f"Client tool {tool_name} cannot execute from the current device session."
                ),
                hint="Ask the user to use the bound device, or choose another available tool.",
                attempts=1,
            )
            model_content, artifact_detail = build_tool_error_payloads(
                summary,
                tool_name=tool_name,
                exception=PermissionError(device_error),
                policy_retry_allowed=False,
            )
            _append_tool_error_output(
                tool_call_id=tool_id,
                tool_name=tool_name,
                tool_args=tool_args,
                model_content=model_content,
                artifact_detail=artifact_detail,
            )
            continue

        try:
            result, error_detail, error_content, execution_detail = await invoke_tool_with_policy(
                tool,
                tool_args,
                tool_name=tool_name,
                tool_map=tool_map,
            )
            if error_detail is not None:
                _append_tool_error_output(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    model_content=error_content,
                    artifact_detail=error_detail,
                )
                continue

            normalized_result = normalize_tool_result_for_rendering(
                result,
                tool_name=tool_name,
                error=_structured_tool_error(result),
            )
            result_text = normalized_result.model_content
            structured_error = _structured_tool_error(result)

            outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": result_text,
                    "render": normalized_result.render,
                }
            )
            artifact = build_tool_artifact(
                tool_call_id=tool_id,
                tool_name=tool_name,
                tool_args=tool_args,
                output_text=result_text,
                error=structured_error,
                max_output_chars=artifact_max_output_chars,
                render=normalized_result.render,
            )
            artifact.update(execution_detail)
            _attach_rich_candidates_to_artifact(
                artifact,
                raw_result=result,
                result_text=result_text,
                render=normalized_result.render,
                tool_call_id=tool_id,
                tool_name=tool_name,
            )
            artifacts.append(artifact)
            if capture_images and tool_name not in _TYPED_WEB_IMAGE_TOOLS:
                images.extend(extract_images_from_tool_result(result_text))
                images.extend(extract_images_from_tool_content(result))

            # Update LRU timestamp for deferred tools on successful execution
            _mark_tool_used_if_deferred(tool_name)

            # If this was a tool-loading tool (like tool_search), refresh the
            # tool map to include newly loaded deferred tools. This allows
            # subsequent tool calls in the same batch to use the discovered tools.
            if tool_name in TOOL_LOADING_TOOLS and settings.mcp_tool_search_enabled:
                await _refresh_tool_map_after_search(
                    tool_map=tool_map,
                    agent=agent,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    device_id=device_id,
                    tool_scope=tool_scope,
                )
        except Exception as exc:
            error_msg = f"Error: {exc}"
            normalized_result = normalize_tool_result_for_rendering(
                error_msg,
                tool_name=tool_name,
                error=str(exc),
            )
            outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": normalized_result.model_content,
                    "render": normalized_result.render,
                }
            )
            artifacts.append(
                build_tool_artifact(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    output_text=normalized_result.model_content,
                    error=str(exc),
                    max_output_chars=artifact_max_output_chars,
                    render=normalized_result.render,
                )
            )

    # Offloading commits the full payload to Postgres synchronously (see
    # ToolResultBlobRepository.create). Run it off the event loop thread so a
    # multi-MB commit does not stall every other concurrent request/stream;
    # this function only touches plain dicts and sync repository/service
    # calls, so threading the whole call is safe. Awaiting it keeps the
    # return values below correctly ordered after the in-place mutations.
    await asyncio.to_thread(
        _apply_offload_to_outputs_and_artifacts,
        outputs=outputs,
        artifacts=artifacts,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    return outputs, artifacts, images
