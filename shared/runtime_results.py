"""Size limits for the tool results a device sends across the runtime bridge.

Two budgets, because the two kinds of content fail differently. Text keeps its
beginning and end and loses its middle: command output puts the command at the
top and the error or summary at the bottom. Images and audio are kept whole or
replaced by a note, since a truncated base64 payload is corrupt data rather
than a smaller picture.

Sizes are measured the way the bridge sends them: ``json.dumps`` with the
default ``ensure_ascii=True``, where each non-ASCII character costs six bytes.
"""

from __future__ import annotations

import json
from typing import Any

_MEDIA_TYPES = frozenset({"image", "audio"})
_OMITTED_MARKER = "\n[... {count} characters omitted to fit the device result limit ...]\n"
_TEXT_BLOCK_OVERHEAD = len(json.dumps([{"type": "text", "text": ""}])) - len('""')


def cap_tool_result(
    result: Any,
    *,
    max_text_bytes: int,
    max_media_bytes: int,
) -> tuple[Any, bool]:
    """Return ``(result, truncated)`` with the result inside both budgets.

    ``max_media_bytes`` bounds the decoded size of all image and audio blocks
    together; ``max_text_bytes`` bounds the sent size of everything else.
    """

    if isinstance(result, list) and result and all(_is_block(item) for item in result):
        return _cap_blocks(result, max_text_bytes, max_media_bytes)
    if _wire_size(result) <= max_text_bytes:
        return result, False
    if isinstance(result, str):
        return _fit_text(result, max_text_bytes), True
    text = json.dumps(result, ensure_ascii=False, default=str)
    return _fit_text(text, max_text_bytes), True


def _cap_blocks(
    blocks: list[dict[str, Any]],
    max_text_bytes: int,
    max_media_bytes: int,
) -> tuple[list[dict[str, Any]], bool]:
    truncated = False
    media_left = max_media_bytes
    placed: list[dict[str, Any]] = []
    for block in blocks:
        if not _is_media(block):
            placed.append(block)
            continue
        size = _decoded_size(block)
        if size <= media_left:
            media_left -= size
            placed.append(block)
            continue
        placed.append(_omitted_media_note(block, size, max_media_bytes))
        truncated = True

    others = [block for block in placed if not _is_media(block)]
    if _wire_size(others) <= max_text_bytes:
        return placed, truncated

    joined = "\n".join(_block_text(block) for block in others)
    fitted = {"type": "text", "text": _fit_text(joined, max_text_bytes - _TEXT_BLOCK_OVERHEAD)}
    return [fitted, *(block for block in placed if _is_media(block))], True


def _fit_text(text: str, max_wire_bytes: int) -> str:
    """Keep as much of the beginning and end as fits, with a marker between them."""

    if _wire_size(text) <= max_wire_bytes:
        return text
    low, high = 0, len(text)
    while low < high:
        keep = (low + high + 1) // 2
        if _wire_size(_cut_middle(text, keep)) <= max_wire_bytes:
            low = keep
        else:
            high = keep - 1
    candidate = _cut_middle(text, low)
    return candidate if _wire_size(candidate) <= max_wire_bytes else ""


def _cut_middle(text: str, keep: int) -> str:
    tail = keep // 2
    head = keep - tail
    marker = _OMITTED_MARKER.format(count=len(text) - keep)
    return text[:head] + marker + (text[len(text) - tail :] if tail else "")


def _omitted_media_note(block: dict[str, Any], size: int, limit: int) -> dict[str, Any]:
    kind = str(block.get("type"))
    mime = block.get("mimeType") or block.get("mime_type") or kind
    return {
        "type": "text",
        "text": (
            f"[{mime} {kind} omitted: {size:,} bytes is more than the "
            f"{limit:,}-byte media limit for one device result]"
        ),
    }


def _decoded_size(block: dict[str, Any]) -> int:
    data = block.get("data") or block.get("base64") or ""
    return len(data) * 3 // 4 if isinstance(data, str) else 0


def _block_text(block: dict[str, Any]) -> str:
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return block["text"]
    return json.dumps(block, ensure_ascii=False, default=str)


def _is_block(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("type"), str)


def _is_media(block: dict[str, Any]) -> bool:
    return block.get("type") in _MEDIA_TYPES


def _wire_size(value: Any) -> int:
    return len(json.dumps(value, default=str))
