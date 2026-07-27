"""Centralized response messages and helper functions."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from .config import settings
from .rich_response import (
    RICH_ITEMS_VERSION,
    RichDisplayPolicy,
    RichItemType,
    parse_inline_rich_references,
    validate_public_rich_item,
    validate_rich_references,
)

if TYPE_CHECKING:
    from ..schemas.workflow import WorkflowResponse

_logger = logging.getLogger(__name__)
_WIDGET_TOOLS = {"widget_create", "widget_update"}


# === Response Fallback Messages ===
NO_RESPONSE_GENERATED = "No response generated"

# === Error Messages ===
ERROR_NO_RESPONSE = "Error: No response generated"
ERROR_NO_RESPONSE_RESUME = "Error: No response generated after resume"
ERROR_RESPONSE_AFTER_RESUME = "Error: No response after resuming"

# === General Errors ===
UNKNOWN_ERROR = "Unknown error"


def _extract_metadata_message(metadata: dict[str, Any] | None) -> str:
    """Derive displayable content from message metadata when body text is empty."""
    if not isinstance(metadata, dict):
        return ""

    for key in ("error", "message"):
        value = metadata.get(key)
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                return trimmed

    interrupt_payload = metadata.get("interrupt")
    if isinstance(interrupt_payload, dict):
        interrupt_metadata = interrupt_payload.get("metadata")
        if isinstance(interrupt_metadata, dict):
            for key in ("message", "reason"):
                value = interrupt_metadata.get(key)
                if isinstance(value, str):
                    trimmed = value.strip()
                    if trimmed:
                        return trimmed

    return ""


def extract_response_content(
    response: WorkflowResponse | None,
    fallback: str = NO_RESPONSE_GENERATED,
) -> str:
    """Extract content from a service-owned workflow response with fallback."""
    if response and response.message:
        metadata: dict[str, Any] = {}
        if isinstance(getattr(response, "metadata", None), dict):
            metadata.update(response.metadata)
        if isinstance(getattr(response.message, "metadata", None), dict):
            metadata.update(response.message.metadata)

        content = normalize_message_content(response.message.content, metadata)
        if content:
            return content

    if response and response.error:
        error_text = str(response.error).strip()
        if error_text:
            return error_text

    return fallback


def normalize_message_content(
    content: str | None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Normalize message content without injecting placeholder text."""
    if isinstance(content, str):
        trimmed = content.strip()
        if trimmed:
            return trimmed
    elif content is not None:
        rendered = str(content).strip()
        if rendered:
            return rendered

    return _extract_metadata_message(metadata)


def extract_live_widgets_from_artifacts(
    tool_artifacts: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Derive ``live_widgets`` metadata from widget tool artifacts.

    Scans tool artifacts for successful ``widget_create`` / ``widget_update``
    results and extracts the minimal metadata the frontend needs to mount
    each widget.
    """
    if not tool_artifacts:
        return []

    widgets_by_id: dict[str, dict[str, Any]] = {}

    for artifact in tool_artifacts:
        tool_name = artifact.get("tool") or artifact.get("tool_name")
        if tool_name not in _WIDGET_TOOLS:
            continue
        status = str(artifact.get("status") or "").strip().lower()
        if status in {"error", "failed", "rejected"}:
            continue
        if artifact.get("error"):
            continue
        raw_output = artifact.get("output")
        if raw_output in (None, ""):
            raw_output = artifact.get("tool_output")
        if raw_output in (None, ""):
            raw_output = artifact.get("result")
        if raw_output in (None, ""):
            continue

        try:
            parsed = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
        except (json.JSONDecodeError, TypeError):
            continue

        if not isinstance(parsed, dict):
            continue

        widget_id = parsed.get("widget_id")
        if not widget_id:
            continue

        widgets_by_id[str(widget_id)] = {
            "widget_id": widget_id,
            "session_id": parsed.get("session_id", ""),
            "title": parsed.get("title"),
            "status": parsed.get("status", "active"),
            "version": parsed.get("version", 1),
            "connection_endpoint": f"/widgets/{widget_id}/connection",
        }

    return list(widgets_by_id.values())


# ---------------------------------------------------------------------------
# Rich items finalization
# ---------------------------------------------------------------------------


def _widget_rich_item_from_live_widget(widget: dict[str, Any]) -> dict[str, Any]:
    """Build a public ``live_widget`` rich-item record from a live-widget entry."""
    widget_id = widget.get("widget_id")
    return {
        "id": f"widget:{widget_id}",
        "type": RichItemType.live_widget.value,
        "source": "widget_tool",
        "display_policy": RichDisplayPolicy.inline_or_append.value,
        "title": widget.get("title"),
        "payload": {
            "widget_id": widget_id,
            "session_id": widget.get("session_id", ""),
            "status": widget.get("status", "active"),
            "version": widget.get("version", 1),
            "connection_endpoint": widget.get(
                "connection_endpoint", f"/widgets/{widget_id}/connection"
            ),
        },
    }


def _is_image_candidate(candidate: dict[str, Any]) -> bool:
    return candidate.get("type") == RichItemType.image.value


def _candidate_id(candidate: dict[str, Any]) -> str | None:
    value = candidate.get("id")
    return value if isinstance(value, str) and value else None


def _normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy a candidate descriptor so finalized output is independent of
    the workflow-internal mutable list."""
    out = {key: value for key, value in candidate.items() if key not in {"_internal"}}
    # Drop temporary fields that should never reach the persisted registry.
    return out


def _canvas_rich_item_from_artifact(artifact: Any) -> dict[str, Any] | None:
    """Build a public ``canvas_artifact`` rich item from legacy canvas metadata.

    The legacy ``metadata["canvas_artifact"]`` field stays persisted for the
    Streamlit path; this promotion is what makes CanvasAgent output visible to
    AI SDK clients, whose wire projection scrubs the legacy field.
    """
    if not isinstance(artifact, dict):
        return None
    content = artifact.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    title = artifact.get("title")
    title = title if isinstance(title, str) and title.strip() else "Canvas"
    language = artifact.get("language")
    language = language if isinstance(language, str) and language.strip() else "html"
    payload: dict[str, Any] = {
        "language": language,
        "title": title,
        "content": content,
    }
    revision = artifact.get("revision")
    if isinstance(revision, int) and revision > 0:
        payload["revision"] = revision
    operation = artifact.get("operation")
    if operation in {"create", "update"}:
        payload["operation"] = operation

    return {
        "id": "canvas:main",
        "type": RichItemType.canvas_artifact.value,
        "display_policy": RichDisplayPolicy.inline_or_append.value,
        "title": title,
        "payload": payload,
    }


def _finalize_rich_items(
    *,
    content: str,
    candidates: list[dict[str, Any]],
    widget_items: list[dict[str, Any]],
    canvas_item: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Resolve the final ``rich_items`` registry and validation warnings.

    Args:
        content: The final assistant markdown.
        candidates: Transient ``_rich_item_candidates`` descriptors gathered
            during execution (already type-tagged dicts).
        widget_items: Public ``live_widget`` rich items derived from existing
            tool artifacts.
        canvas_item: Public ``canvas_artifact`` rich item derived from the
            legacy canvas metadata, when the response produced one.

    Returns:
        A tuple ``(rich_items, warnings)`` where ``rich_items`` is the
        finalized public registry and ``warnings`` is the validation log.
    """
    referenced = parse_inline_rich_references(content)
    referenced_set = set(referenced)

    rich_items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    warnings: list[dict[str, str]] = []

    def validated_public_item(item: dict[str, Any]) -> dict[str, Any] | None:
        item_id = str(item.get("id") or "")
        try:
            validated = validate_public_rich_item(
                item,
                selected_image_max_bytes=settings.rich_item_selected_image_max_bytes,
            )
        except Exception:
            warnings.append({"code": "invalid_rich_item", "id": item_id})
            return None
        return validated.model_dump(mode="json", exclude_none=True)

    # Widgets and canvas artifacts always persist (they are inline_or_append).
    always_persisted = list(widget_items)
    if canvas_item is not None:
        always_persisted.append(canvas_item)
    for item in always_persisted:
        item_id = item.get("id")
        if not item_id or item_id in seen_ids:
            continue
        public_item = validated_public_item(item)
        if public_item is None:
            continue
        rich_items.append(public_item)
        seen_ids.add(item_id)

    # Candidates: images persist only when referenced; non-images persist if
    # their policy allows append or they are referenced.
    for candidate in candidates:
        cand_id = _candidate_id(candidate)
        if cand_id is None or cand_id in seen_ids:
            continue
        is_image = _is_image_candidate(candidate)
        if is_image and cand_id not in referenced_set:
            continue
        public_candidate = validated_public_item(_normalize_candidate(candidate))
        if public_candidate is None:
            continue
        rich_items.append(public_candidate)
        seen_ids.add(cand_id)

    warnings.extend(validate_rich_references(content, rich_items))
    return rich_items, warnings


def _image_locator(payload: dict[str, Any]) -> str | None:
    """Return a stable locator (URL or short data digest) for an image record."""
    if not isinstance(payload, dict):
        return None
    url = payload.get("url") or payload.get("source_url")
    if isinstance(url, str) and url:
        return f"url::{url}"
    data = payload.get("data") or payload.get("b64_data")
    if isinstance(data, str) and data:
        return f"data::{data[:64]}"
    return None


def _filter_unreferenced_images_from_metadata_images(
    metadata: dict[str, Any],
    *,
    referenced_ids: set[str],
    candidates: list[dict[str, Any]],
) -> None:
    """For v1 messages, drop hidden image candidates from ``metadata["images"]``.

    A candidate is hidden when its rich-item id is not present in the final
    markdown. Filtering matches by candidate id when available and by URL/data
    locator otherwise, so unreferenced URLs do not leak through the legacy
    gallery field.
    """
    images = metadata.get("images")
    if not isinstance(images, list) or not images:
        return

    hidden_locators: set[str] = set()
    for candidate in candidates:
        if not _is_image_candidate(candidate):
            continue
        if candidate.get("id") in referenced_ids:
            continue
        locator = _image_locator(candidate.get("payload") or {})
        if locator:
            hidden_locators.add(locator)

    kept: list[Any] = []
    for image in images:
        if not isinstance(image, dict):
            kept.append(image)
            continue
        candidate_id = image.get("rich_item_id") or image.get("id")
        if isinstance(candidate_id, str) and candidate_id and candidate_id not in referenced_ids:
            continue
        locator = _image_locator(image)
        if locator and locator in hidden_locators:
            continue
        kept.append(image)
    if kept:
        metadata["images"] = kept
    else:
        metadata.pop("images", None)


def _reuse_stored_ref(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Promote an early-persisted ``stored_ref`` descriptor to a reference entry.

    When an image was already persisted during generation, it carries a
    ``stored_ref`` descriptor. Terminal persistence reuses it verbatim instead of
    decoding and writing the bytes a second time, dropping the transient
    ``stored_ref`` and any inline bytes so the persisted entry is a clean
    reference."""
    stored = entry.get("stored_ref")
    if not isinstance(stored, dict):
        return None
    image_id = stored.get("image_id")
    url = stored.get("url")
    if not image_id or not url:
        return None
    merged = {
        key: value
        for key, value in entry.items()
        if key not in ("data", "b64_data", "stored_ref")
    }
    merged["image_id"] = image_id
    merged["url"] = url
    merged.setdefault("mime", stored.get("mime") or entry.get("mime"))
    return merged


def externalize_metadata_images(
    images: list[dict[str, Any]] | None,
    *,
    store: Any,
) -> list[dict[str, Any]] | None:
    """Replace inline base64 in ``metadata["images"]`` entries with storage
    references. ``store`` is a callable ``(*, mime, data_b64, name) -> ref``.
    Entries already carrying a ``stored_ref`` descriptor (persisted early during
    generation) reuse that reference without re-storing. Entries already carrying
    a ``url`` (or no inline bytes) are left untouched. A store failure keeps the
    original inline entry so the image is never lost."""
    if images is None:
        return None
    if not images:
        return list(images)
    out: list[dict[str, Any]] = []
    for entry in images:
        if not isinstance(entry, dict):
            continue
        reused = _reuse_stored_ref(entry)
        if reused is not None:
            out.append(reused)  # early-persisted — reuse descriptor, no re-store
            continue
        if store is None:
            out.append(entry)
            continue
        inline_b64 = entry.get("data") or entry.get("b64_data")
        if not inline_b64:
            out.append(entry)  # remote url or already a reference
            continue
        try:
            ref = store(
                mime=entry.get("mime") or entry.get("mime_type") or "image/png",
                data_b64=inline_b64,
                name=entry.get("name") or "image",
            )
        except Exception:
            _logger.warning(
                "Generated image externalization failed; keeping inline entry "
                "code=chat_image_store_failed"
            )
            out.append(entry)
            continue
        merged = {k: v for k, v in entry.items() if k not in ("data", "b64_data")}
        merged["image_id"] = ref["image_id"]
        merged["url"] = ref["url"]
        merged.setdefault("mime", ref.get("mime"))
        out.append(merged)
    return out


def build_bot_metadata(
    response: WorkflowResponse | None,
    persona: str | None = None,
) -> dict[str, Any]:
    """Build standard bot response metadata from a workflow response.

    For workflows that produced rich-item candidates, this also materializes
    the public ``rich_items`` registry, attaches ``rich_items_version``, and
    records validation warnings.
    """
    metadata: dict[str, Any] = {}

    if response and response.metadata:
        metadata = dict(response.metadata)

    if persona:
        metadata.setdefault("persona_used", persona)

    if response and response.tool_artifacts:
        existing_artifacts = metadata.get("tool_artifacts")
        if isinstance(existing_artifacts, list):
            merged_artifacts = list(existing_artifacts)
            for artifact in response.tool_artifacts:
                if artifact not in merged_artifacts:
                    merged_artifacts.append(artifact)
            metadata["tool_artifacts"] = merged_artifacts
        else:
            metadata["tool_artifacts"] = list(response.tool_artifacts)

    if response and response.metadata and "images" in response.metadata:
        metadata["images"] = response.metadata["images"]

    # Derive live_widgets from widget tool artifacts.
    live_widgets = extract_live_widgets_from_artifacts(metadata.get("tool_artifacts"))
    if live_widgets:
        metadata["live_widgets"] = live_widgets

    # Prefer the canonical ``agent`` field. Once it exists, drop the redundant
    # custom-agent compatibility fields so persisted messages carry one shape.
    # ``custom_agent_warnings`` is intentionally preserved (still useful when
    # selected client tools or skills are unavailable).
    if "agent" in metadata:
        for redundant_key in (
            "runtime_agent_id",
            "custom_agent_id",
            "custom_agent_name",
        ):
            metadata.pop(redundant_key, None)

    if not getattr(settings, "inline_rich_response_enabled", False):
        # Keep legacy attachment/widget metadata readable while rollout is
        # disabled, but never persist the internal candidate handoff field.
        metadata.pop("_rich_item_candidates", None)
        metadata.pop("_inline_rich_response_v1", None)
        return metadata

    # ── Rich items finalization ─────────────────────────────────────────
    # The transient `_rich_item_candidates` field is the workflow-internal
    # handoff. It is consumed and removed here; it must never appear in the
    # persisted assistant metadata.
    raw_candidates = metadata.pop("_rich_item_candidates", None)
    capable_response = bool(metadata.pop("_inline_rich_response_v1", False))
    candidates: list[dict[str, Any]] = []
    if isinstance(raw_candidates, list):
        candidates = [c for c in raw_candidates if isinstance(c, dict)]

    widget_items = [_widget_rich_item_from_live_widget(widget) for widget in (live_widgets or [])]
    canvas_item = _canvas_rich_item_from_artifact(metadata.get("canvas_artifact"))

    # Only opt the message into the v1 contract when there is actual rich-item
    # activity. Legacy messages with neither markers nor candidates retain the
    # pre-feature shape so existing readers continue to function.
    content = ""
    message_obj = getattr(response, "message", None) if response is not None else None
    message_content = getattr(message_obj, "content", None) if message_obj is not None else None
    if isinstance(message_content, str):
        content = message_content
    has_markers = bool(parse_inline_rich_references(content))
    has_v1_signal = (
        bool(candidates) or has_markers or (capable_response and bool(widget_items or canvas_item))
    )

    if not has_v1_signal:
        return metadata

    rich_items, warnings = _finalize_rich_items(
        content=content,
        candidates=candidates,
        widget_items=widget_items,
        canvas_item=canvas_item,
    )

    metadata["rich_items_version"] = RICH_ITEMS_VERSION
    metadata["rich_items"] = rich_items
    metadata["rich_reference_warnings"] = warnings

    # For v1 messages, drop unreferenced image candidates from any public
    # ``metadata["images"]`` field so the legacy gallery cannot surface them.
    referenced_ids = {
        str(item.get("id"))
        for item in rich_items
        if item.get("type") == RichItemType.image.value and item.get("id")
    }
    _filter_unreferenced_images_from_metadata_images(
        metadata,
        referenced_ids=referenced_ids,
        candidates=candidates,
    )

    return metadata
