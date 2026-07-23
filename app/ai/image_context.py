from __future__ import annotations

import mimetypes
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_LOCAL_PATH_PREFIXES = ("/", "./", "../")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")

CHAT_IMAGE_LOADER: ContextVar[Callable[[str], str | None] | None] = ContextVar(
    "chat_image_loader", default=None
)


def current_chat_image_loader() -> Callable[[str], str | None] | None:
    return CHAT_IMAGE_LOADER.get()


@contextmanager
def use_chat_image_loader(loader: Callable[[str], str | None] | None) -> Iterator[None]:
    """Install a per-run resolver that maps a stored ``image_id`` to a base64
    data URL, so historical image references can be re-sent to the model."""
    token = CHAT_IMAGE_LOADER.set(loader)
    try:
        yield
    finally:
        CHAT_IMAGE_LOADER.reset(token)


def _resolve_reference_url(attachment: dict[str, Any]) -> str | None:
    image_id = attachment.get("image_id")
    if not image_id or attachment.get("data") or attachment.get("base64"):
        return None
    loader = current_chat_image_loader()
    if loader is None:
        return None
    return loader(str(image_id))


def _clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _mime_from_data_url(data_url: str) -> str | None:
    if not data_url.startswith("data:"):
        return None
    header = data_url.split(",", 1)[0]
    media_type = header[5:].split(";", 1)[0].strip()
    return media_type or None


def _looks_like_local_path(value: str) -> bool:
    return bool(_WINDOWS_DRIVE_RE.match(value)) or value.startswith(_LOCAL_PATH_PREFIXES)


def _candidate_payloads(attachment: dict[str, Any]) -> list[Any]:
    return [
        attachment.get("data"),
        attachment.get("url"),
        attachment.get("base64"),
        attachment.get("path"),
        attachment.get("image"),
        attachment.get("source"),
    ]


# Human-readable guidance for drops the user can actually fix. Other drops
# (empty payload, non-image mime) have no reason code and stay silent.
_REJECTION_MESSAGES = {
    "blob_url": "browser blob URLs aren't supported — attach the image data instead",
    "local_path": "local file paths can't be attached — upload the image data instead",
}


def normalize_image_attachment_result(attachment: Any) -> tuple[dict[str, str] | None, str | None]:
    """Normalize an attachment to ``{name, mime, url}`` or explain why not.

    Returns ``(normalized, reason)``. ``reason`` is a short code (``"blob_url"``
    / ``"local_path"``) for drops the caller can surface to the user, else None.
    """
    if not isinstance(attachment, dict):
        return None, None

    name = _clean_str(attachment.get("name") or attachment.get("filename")) or "image"
    mime = _clean_str(
        attachment.get("mime")
        or attachment.get("mimeType")
        or attachment.get("mediaType")
        or attachment.get("contentType")
    )
    if not mime:
        mime = mimetypes.guess_type(name)[0] or "image/jpeg"

    raw_value: str | None = None
    for candidate in _candidate_payloads(attachment):
        if isinstance(candidate, dict):
            candidate = (
                candidate.get("url")
                or candidate.get("data")
                or candidate.get("base64")
                or candidate.get("path")
            )
        raw_value = _clean_str(candidate)
        if raw_value:
            break

    if not raw_value:
        return None, None
    if _looks_like_local_path(raw_value):
        return None, "local_path"

    if raw_value.startswith("data:"):
        inferred = _mime_from_data_url(raw_value)
        mime = inferred or mime
        if not mime.startswith("image/"):
            return None, None
        return {"name": name, "mime": mime, "url": raw_value}, None

    if not mime.startswith("image/"):
        return None, None

    if raw_value.startswith("blob:"):
        # Browser blob URLs are scoped to the page process and are not fetchable
        # by the backend or model provider. The UI must send data URLs instead.
        return None, "blob_url"

    if raw_value.startswith(("http://", "https://")):
        return {"name": name, "mime": mime, "url": raw_value}, None

    return {"name": name, "mime": mime, "url": f"data:{mime};base64,{raw_value}"}, None


def normalize_image_attachment(attachment: Any) -> dict[str, str] | None:
    return normalize_image_attachment_result(attachment)[0]


def describe_attachment_rejections(attachments: list[Any] | None) -> list[str]:
    """Human-readable notices for attachments dropped for a fixable reason."""
    messages: list[str] = []
    for attachment in attachments or []:
        _, reason = normalize_image_attachment_result(attachment)
        message = _REJECTION_MESSAGES.get(reason or "")
        if not message:
            continue
        label = None
        if isinstance(attachment, dict):
            label = _clean_str(attachment.get("name") or attachment.get("filename"))
        messages.append(f"{label or 'an image'}: {message}")
    return messages


def image_url_part(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def build_multimodal_content(text: Any, attachments: list[Any] | None) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    text_value = str(text or "").strip()
    if text_value:
        parts.append({"type": "text", "text": text_value})

    for attachment in attachments or []:
        if isinstance(attachment, dict):
            ref_url = _resolve_reference_url(attachment)
            if ref_url:
                parts.append(image_url_part(ref_url))
                continue
            has_ref = attachment.get("image_id") and not (
                attachment.get("data") or attachment.get("base64")
            )
            if has_ref:
                # Unresolved reference (no loader / missing): drop it rather than
                # leak the internal /chat-images URL, which the model can't fetch.
                continue
        normalized = normalize_image_attachment(attachment)
        if normalized is None:
            continue
        parts.append(image_url_part(normalized["url"]))

    return parts


def has_image_parts(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(isinstance(part, dict) and part.get("type") == "image_url" for part in content)


def attachment_memory_lines(attachments: list[Any] | None) -> list[str]:
    lines: list[str] = []
    for attachment in attachments or []:
        normalized = normalize_image_attachment(attachment)
        if normalized is None:
            continue
        lines.append(f"[Attached image: {normalized['name']}, {normalized['mime']}]")
    return lines
