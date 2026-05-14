from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.ai.utils import make_json_safe

_WIDGET_TOOLS = {"widget_create", "widget_update"}
_CONTENT_BLOCK_TYPES = {
    "text",
    "image",
    "audio",
    "resource",
    "resource_link",
    "resourcelink",
}
_MAX_INLINE_DATA_CHARS = 64 * 1024
_MAX_STRUCTURED_CHARS = 128 * 1024


@dataclass(frozen=True)
class NormalizedToolRender:
    model_content: str
    output_preview: str
    render: dict[str, Any]


def normalize_tool_result_for_rendering(
    result: Any,
    *,
    tool_name: str,
    error: str | None = None,
) -> NormalizedToolRender:
    safe_result = make_json_safe(result)
    content_blocks = _extract_content_blocks(safe_result)
    structured_content = _extract_structured_content(safe_result)
    ui_meta = _extract_ui_meta(safe_result)
    resources = _extract_resources(safe_result)
    template_uri = _extract_template_uri(ui_meta, resources)
    text = _extract_text(content_blocks)

    if structured_content is None:
        structured_content = _infer_structured_content_from_result(safe_result, content_blocks)
    if structured_content is None:
        structured_content = _extract_dispatch_subagents_structured_content(
            safe_result,
            tool_name=tool_name,
        )

    if not text:
        text = _build_model_content(
            safe_result=safe_result,
            structured_content=structured_content,
            content_blocks=content_blocks,
            error=error,
        )

    render_type = _infer_render_type(
        tool_name=tool_name,
        error=error,
        template_uri=template_uri,
        structured_content=structured_content,
        content_blocks=content_blocks,
        resources=resources,
        text=text,
    )

    render: dict[str, Any] = {
        "version": 1,
        "type": render_type,
        "model_content": text,
        "text": text,
    }

    title = _extract_title(safe_result, ui_meta)
    if not title and render_type == "subagent_dispatch":
        title = "Planning subagents"
    if title:
        render["title"] = title
    if structured_content is not None:
        render["structured_content"] = _cap_structured_content(structured_content)
    if content_blocks:
        render["content"] = _redact_large_inline_data(content_blocks)
    if resources:
        render["resources"] = resources
    if template_uri:
        render["template_uri"] = template_uri
    if ui_meta:
        render["ui_meta"] = ui_meta
    if error:
        render["error"] = _clean_error_text(error)

    return NormalizedToolRender(
        model_content=text,
        output_preview=_preview_text(text),
        render=render,
    )


def _is_content_block_dict(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    block_type = str(value.get("type") or "").lower()
    return block_type in _CONTENT_BLOCK_TYPES


def _extract_content_blocks(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, list):
            return [make_json_safe(item) for item in content if isinstance(item, dict)]
        if isinstance(content, dict):
            return [make_json_safe(content)]

        if _is_content_block_dict(value):
            return [make_json_safe(value)]

    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        typed_blocks = [
            make_json_safe(item)
            for item in value
            if isinstance(item, dict) and isinstance(item.get("type"), str)
        ]
        if typed_blocks:
            return typed_blocks

    return []


def _extract_structured_content(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    for key in ("structuredContent", "structured_content"):
        candidate = value.get(key)
        if candidate not in (None, "", [], {}):
            return make_json_safe(candidate)
    return None


def _extract_ui_meta(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    meta = value.get("_meta") or value.get("meta")
    return make_json_safe(meta) if isinstance(meta, dict) else {}


def _extract_resources(value: Any) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []

    def add_resource(candidate: Any) -> None:
        if not isinstance(candidate, dict):
            return
        uri = candidate.get("uri") or candidate.get("url")
        if not uri:
            return
        resource = {
            "uri": str(uri),
            "mime_type": str(
                candidate.get("mimeType")
                or candidate.get("mime_type")
                or candidate.get("contentType")
                or ""
            ),
        }
        title = candidate.get("title") or candidate.get("name")
        if title:
            resource["title"] = str(title)
        resources.append(resource)

    if isinstance(value, dict):
        for key in ("resources", "resource"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                for item in candidate:
                    add_resource(item)
            else:
                add_resource(candidate)

    for block in _extract_content_blocks(value):
        if block.get("type") in {"resource", "resource_link", "resourceLink"}:
            add_resource(block.get("resource") or block)

    return resources


def _extract_template_uri(ui_meta: dict[str, Any], resources: list[dict[str, Any]]) -> str | None:
    for key in ("openai/outputTemplate", "outputTemplate", "ui/resourceUri"):
        value = ui_meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    ui_value = ui_meta.get("ui")
    if isinstance(ui_value, dict):
        for key in ("resourceUri", "resource_uri", "uri"):
            value = ui_value.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for resource in resources:
        uri = resource.get("uri")
        mime_type = str(resource.get("mime_type") or "")
        if isinstance(uri, str) and uri.startswith("ui://"):
            return uri
        if isinstance(uri, str) and mime_type == "text/html":
            return uri

    return None


def _extract_text(content_blocks: list[dict[str, Any]]) -> str:
    text_parts: list[str] = []
    media_parts: list[str] = []

    for block in content_blocks:
        block_type = str(block.get("type") or "").lower()
        if block_type == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block_type == "image":
            mime = block.get("mimeType") or block.get("mime_type") or "image"
            media_parts.append(f"[{mime} image]")
        elif block_type == "audio":
            mime = block.get("mimeType") or block.get("mime_type") or "audio"
            media_parts.append(f"[{mime} audio]")
        elif block_type in {"resource", "resource_link", "resourcelink"}:
            uri = block.get("uri") or block.get("url")
            if uri:
                media_parts.append(f"[resource: {uri}]")

    joined = "\n".join(part.strip() for part in text_parts if part.strip()).strip()
    if joined:
        return joined
    return "\n".join(media_parts).strip()


def _infer_structured_content_from_result(value: Any, content_blocks: list[dict[str, Any]]) -> Any:
    if isinstance(value, dict):
        if _is_content_block_dict(value):
            return None
        metadata_keys = {"content", "_meta", "meta", "resources", "resource"}
        remaining = {k: v for k, v in value.items() if k not in metadata_keys}
        return remaining or None

    if isinstance(value, list) and not content_blocks:
        return value

    return None


def _extract_dispatch_subagents_structured_content(
    value: Any,
    *,
    tool_name: str,
) -> Any:
    if tool_name != "dispatch_subagents":
        return None

    candidate = value
    if isinstance(value, str):
        try:
            candidate = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None

    if _looks_like_subagent_dispatch(candidate):
        return make_json_safe(candidate)
    return None


def _clean_error_text(error: Any) -> str:
    error_text = str(error or "").strip()
    if error_text.lower().startswith("error:"):
        error_text = error_text[len("error:") :].strip()
    return error_text


def _build_model_content(
    *,
    safe_result: Any,
    structured_content: Any,
    content_blocks: list[dict[str, Any]],
    error: str | None,
) -> str:
    if error:
        cleaned = _clean_error_text(error)
        return f"Error: {cleaned}" if cleaned else "Error"
    if structured_content is not None:
        return json.dumps(structured_content, ensure_ascii=False, default=str)
    if content_blocks:
        return json.dumps(content_blocks, ensure_ascii=False, default=str)
    if isinstance(safe_result, str):
        return safe_result
    return json.dumps(safe_result, ensure_ascii=False, default=str)


def _redact_large_inline_data(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            result.append(block)
            continue

        cleaned = dict(block)
        for data_key in ("data", "base64"):
            value = cleaned.get(data_key)
            if isinstance(value, str) and len(value) > _MAX_INLINE_DATA_CHARS:
                cleaned[data_key] = ""
                cleaned["_truncated"] = True
                cleaned["_original_size"] = len(value)
        result.append(cleaned)
    return result


def _cap_structured_content(value: Any) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return value
    if len(encoded) <= _MAX_STRUCTURED_CHARS:
        return value
    return {"_truncated": True, "_original_size": len(encoded)}


def _infer_render_type(
    *,
    tool_name: str,
    error: str | None,
    template_uri: str | None,
    structured_content: Any,
    content_blocks: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    text: str,
) -> str:
    if error:
        return "error"
    if tool_name == "dispatch_subagents" and _looks_like_subagent_dispatch(structured_content):
        return "subagent_dispatch"
    if template_uri:
        return "mcp_app"
    if tool_name in _WIDGET_TOOLS:
        return "live_widget"
    if any(str(block.get("type") or "").lower() == "image" for block in content_blocks):
        return "image"
    if _looks_like_chart(structured_content):
        return "chart"
    if _looks_like_table(structured_content):
        return "table"
    if resources:
        return "resource"
    if structured_content is not None:
        return "json"
    return "text" if text else "json"


def _looks_like_chart(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = set(value)
    return bool({"chart_type", "chartType"} & keys) and "labels" in keys and "datasets" in keys


def _looks_like_table(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return isinstance(value.get("columns"), list) and isinstance(value.get("rows"), list)


def _looks_like_subagent_dispatch(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("status") not in {"completed", "partial", "failed"}:
        return False
    results = value.get("results")
    if not isinstance(results, list):
        return False
    return all(isinstance(item, dict) for item in results)


def _extract_title(value: Any, ui_meta: dict[str, Any]) -> str | None:
    for candidate in (
        ui_meta.get("title"),
        value.get("title") if isinstance(value, dict) else None,
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _preview_text(value: str, max_chars: int = 1000) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip()
