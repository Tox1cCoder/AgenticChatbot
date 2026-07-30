"""Inline rich-response contract.

This module defines the public Pydantic schemas, marker parser, display-policy
helpers, and bounded prompt-inventory serializer for the inline rich-response
feature described in ``response_format.md``.

The contract is intentionally narrow:

* Assistant ``content`` remains markdown. Rich items are placed using an
  HTML-comment marker of the form ``<!--rich:<id>-->``. Prompt guidance asks
  models to put markers on their own line, while parsers also tolerate markers
  embedded in prose so malformed-but-valid model output does not leak.
* The ``rich_items`` registry is type-validated through a discriminated union.
  Each item type accepts only the fields its renderer consumes; everything else
  is rejected with ``extra="forbid"``.
* Helpers in this module never serialize raw binary data (base64, widget state,
  canvas source) into model-facing inventory blocks or transient stream
  upserts.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Iterable
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Schema version persisted with rich-item bearing assistant messages.
RICH_ITEMS_VERSION: int = 1

#: Maximum length (in characters) of an item id.
RICH_ITEM_ID_MAX_LENGTH: int = 128

#: Fallback ``alt_text`` assigned to tool-result images that arrive without a
#: description. It carries no descriptive signal, so auto-placement must not
#: treat it as one and renderers must not surface it as a caption.
GENERIC_IMAGE_ALT_TEXT: str = "Image from tool result"

#: Allowed inline raster MIME types for image payloads.
ALLOWED_IMAGE_MIME_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)

#: Allowed URL schemes for remote image and resource_link payloads. ``http``
#: is included for local development; production deployments should restrict
#: to ``https`` via a separate policy.
ALLOWED_URL_SCHEMES: frozenset[str] = frozenset({"https", "http"})

#: Same-origin authenticated image routes accepted by the public image
#: contract.  The ``/api`` forms are kept for deployments that mount the
#: application behind an API prefix.
PROTECTED_IMAGE_URL_PREFIXES: tuple[str, ...] = (
    "/chat-images/",
    "/api/chat-images/",
    "/web-images/",
    "/api/web-images/",
)


_ITEM_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-.:]+$")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RichItemType(str, Enum):
    image = "image"
    image_group = "image_group"
    live_widget = "live_widget"
    tool_render = "tool_render"
    canvas_artifact = "canvas_artifact"
    citation = "citation"
    resource_link = "resource_link"


class RichDisplayPolicy(str, Enum):
    #: Render the item only at its inline marker. Never append as a fallback.
    inline_only = "inline_only"
    #: Render the item at its inline marker if referenced; otherwise append.
    inline_or_append = "inline_or_append"


# ---------------------------------------------------------------------------
# Payload models
# ---------------------------------------------------------------------------


class PublicPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _validate_url_scheme(url: str | None) -> None:
    if url is None:
        return
    if "://" not in url:
        raise ValueError(f"invalid url: {url!r}")
    scheme = url.split("://", 1)[0].lower()
    if scheme not in ALLOWED_URL_SCHEMES:
        raise ValueError(f"unsupported url scheme {scheme!r}")


def _validate_image_url(url: str | None) -> None:
    if url is None:
        return
    if url.startswith(PROTECTED_IMAGE_URL_PREFIXES):
        return
    if url.startswith("/"):
        raise ValueError(f"unsupported protected image url: {url!r}")
    _validate_url_scheme(url)


def _validate_image_mime_and_urls(
    mime_type: str, url: str | None, source_url: str | None
) -> None:
    """Shared mime-type/URL validation for ``ImagePayload`` and
    ``ImageGroupItem``. Both models accept the same image mime categories and
    validate ``url``/``source_url`` the same way; keep that logic in one
    place instead of duplicating it per model.
    """
    if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise ValueError(
            f"unsupported image mime_type {mime_type!r}; allowed: "
            f"{sorted(ALLOWED_IMAGE_MIME_TYPES)}"
        )
    _validate_image_url(url)
    _validate_url_scheme(source_url)


class ImagePayload(PublicPayload):
    url: str | None = None
    data: str | None = None
    mime_type: str
    source_url: str | None = None
    description: str | None = None
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    caption: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _has_exactly_one_source(self) -> ImagePayload:
        if (self.url is None) == (self.data is None):
            raise ValueError("image payload requires exactly one of url or data")
        _validate_image_mime_and_urls(self.mime_type, self.url, self.source_url)
        return self


class ImageGroupItem(PublicPayload):
    """One cell of an image group. Cells are always remote or protected URLs;
    inline base64 cells are not supported because a group is only ever built
    from provider search results."""

    url: str
    mime_type: str
    source_url: str | None = None
    description: str | None = None
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_cell(self) -> ImageGroupItem:
        if not self.url:
            raise ValueError("image group cell requires a url")
        _validate_image_mime_and_urls(self.mime_type, self.url, self.source_url)
        return self


class ImageGroupPayload(PublicPayload):
    items: list[ImageGroupItem] = Field(min_length=2)


class LiveWidgetPayload(PublicPayload):
    widget_id: str
    session_id: str
    status: str
    version: int
    connection_endpoint: str


class ToolRenderPayload(PublicPayload):
    # ``render`` is the output of the existing redaction/capping normalizer.
    # We accept it as a free-form mapping because the renderer-side schemas
    # are already validated upstream.
    render: dict[str, Any]


class CanvasPayload(PublicPayload):
    language: str
    title: str
    content: str
    preferred_height: int | None = None
    revision: int | None = Field(default=None, ge=1)
    operation: Literal["create", "update"] | None = None


class CitationPayload(PublicPayload):
    source: str
    document_id: str | None = None
    page_number: int | None = None
    chunk_index: int | None = None


class ResourceLinkPayload(PublicPayload):
    url: str
    title: str | None = None
    description: str | None = None

    @model_validator(mode="after")
    def _validate_url(self) -> ResourceLinkPayload:
        _validate_url_scheme(self.url)
        return self


# ---------------------------------------------------------------------------
# Rich item models
# ---------------------------------------------------------------------------


class RichItemBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str | None = None
    display_policy: RichDisplayPolicy
    title: str | None = None
    alt_text: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_id(self) -> RichItemBase:
        if not self.id:
            raise ValueError("rich item id must be non-empty")
        if len(self.id) > RICH_ITEM_ID_MAX_LENGTH:
            raise ValueError(f"rich item id length exceeds {RICH_ITEM_ID_MAX_LENGTH} characters")
        if not _ITEM_ID_PATTERN.match(self.id):
            raise ValueError(f"rich item id contains invalid characters: {self.id!r}")
        return self


class ImageRichItem(RichItemBase):
    type: Literal[RichItemType.image]
    display_policy: Literal[RichDisplayPolicy.inline_only] = RichDisplayPolicy.inline_only
    alt_text: str
    payload: ImagePayload


class ImageGroupRichItem(RichItemBase):
    type: Literal[RichItemType.image_group]
    display_policy: Literal[RichDisplayPolicy.inline_only] = RichDisplayPolicy.inline_only
    alt_text: str
    payload: ImageGroupPayload


class LiveWidgetRichItem(RichItemBase):
    type: Literal[RichItemType.live_widget]
    payload: LiveWidgetPayload


class ToolRenderRichItem(RichItemBase):
    type: Literal[RichItemType.tool_render]
    payload: ToolRenderPayload


class CanvasRichItem(RichItemBase):
    type: Literal[RichItemType.canvas_artifact]
    payload: CanvasPayload


class CitationRichItem(RichItemBase):
    type: Literal[RichItemType.citation]
    payload: CitationPayload


class ResourceLinkRichItem(RichItemBase):
    type: Literal[RichItemType.resource_link]
    payload: ResourceLinkPayload


RichItem = Annotated[
    ImageRichItem
    | ImageGroupRichItem
    | LiveWidgetRichItem
    | ToolRenderRichItem
    | CanvasRichItem
    | CitationRichItem
    | ResourceLinkRichItem,
    Field(discriminator="type"),
]

_RICH_ITEM_ADAPTER = TypeAdapter(RichItem)


def validate_public_rich_item(
    item: Any,
    *,
    selected_image_max_bytes: int | None = None,
) -> RichItem:
    """Validate one record before it is persisted or exposed to renderers."""
    validated = _RICH_ITEM_ADAPTER.validate_python(item)
    if isinstance(validated, ImageRichItem) and validated.payload.data is not None:
        try:
            decoded = base64.b64decode(validated.payload.data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image payload contains invalid base64 data") from exc
        if selected_image_max_bytes is not None and len(decoded) > selected_image_max_bytes:
            raise ValueError("image payload exceeds decoded byte-size limit")
    return validated


# ---------------------------------------------------------------------------
# Marker parser
# ---------------------------------------------------------------------------


# Match a standalone HTML-comment marker line with up to three leading spaces
# and optional trailing whitespace. Kept for callers that still need to
# distinguish canonical block placement from the tolerant inline parser below.
_MARKER_LINE_RE = re.compile(
    r"^[ ]{0,3}<!--rich:([A-Za-z0-9_\-.:]+)-->[ \t]*$",
)
_MARKER_RE = re.compile(r"<!--rich:([A-Za-z0-9_\-.:]+)-->")
_BACKTICK_RUN_RE = re.compile(r"`+")


def _inline_code_ranges(line: str) -> list[tuple[int, int]]:
    """Return simple inline-code span ranges for one markdown line.

    The rich marker parser only needs to protect normal backtick-delimited
    code spans such as `` `<!--rich:...-->` ``. Unmatched backticks are treated
    as ordinary text, matching markdown's fallback behavior.
    """
    ranges: list[tuple[int, int]] = []
    pos = 0
    while True:
        opener = _BACKTICK_RUN_RE.search(line, pos)
        if opener is None:
            return ranges
        ticks = opener.group(0)
        closer_start = line.find(ticks, opener.end())
        if closer_start < 0:
            pos = opener.end()
            continue
        closer_end = closer_start + len(ticks)
        ranges.append((opener.start(), closer_end))
        pos = closer_end


def _overlaps_any_range(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < range_end and end > range_start for range_start, range_end in ranges)


def _iter_inline_rich_marker_matches(line: str) -> Iterable[re.Match[str]]:
    """Yield valid rich marker matches from one non-code markdown line.

    Markers may be standalone or embedded in surrounding prose. Matches inside
    inline backtick code spans are ignored so examples of the marker grammar do
    not become rich references.
    """
    if "<!--rich:" not in line:
        return
    code_ranges = _inline_code_ranges(line)
    for match in _MARKER_RE.finditer(line):
        if _overlaps_any_range(match.start(), match.end(), code_ranges):
            continue
        candidate = match.group(1)
        if len(candidate) > RICH_ITEM_ID_MAX_LENGTH:
            continue
        if not _ITEM_ID_PATTERN.match(candidate):
            continue
        yield match


def _strip_fenced_code_blocks(lines: list[str]) -> list[bool]:
    """Return a list of booleans, one per line, indicating whether the line is
    inside a fenced code block. Supports backtick and tilde fences with
    matching length (CommonMark fenced-code semantics).
    """
    inside = False
    fence_char: str | None = None
    fence_len = 0
    flags: list[bool] = []
    for raw in lines:
        # CommonMark allows up to three leading spaces before the fence.
        stripped = raw.lstrip(" ")
        leading = len(raw) - len(stripped)
        is_fence = False
        fence_match_len = 0
        fence_match_char: str | None = None
        if leading <= 3 and stripped[:3] in ("```", "~~~"):
            fence_match_char = stripped[0]
            # Count consecutive fence chars.
            i = 0
            while i < len(stripped) and stripped[i] == fence_match_char:
                i += 1
            fence_match_len = i
            is_fence = fence_match_len >= 3
        if not inside:
            if is_fence:
                inside = True
                fence_char = fence_match_char
                fence_len = fence_match_len
                # The fence opening line itself is treated as inside the block
                # for marker-parsing purposes.
                flags.append(True)
                continue
            flags.append(False)
        else:
            # Inside a fence: close only on a matching fence with length >=
            # opening length and matching character, no info string.
            closes = (
                is_fence
                and fence_match_char == fence_char
                and fence_match_len >= fence_len
                and stripped[fence_match_len:].strip() == ""
            )
            flags.append(True)
            if closes:
                inside = False
                fence_char = None
                fence_len = 0
    return flags


def _is_indented_code(line: str) -> bool:
    # An indented code block requires four spaces of indentation on a non-blank
    # line.
    if not line.strip():
        return False
    return line.startswith("    ") or line.startswith("\t")


def parse_inline_rich_references(markdown: str) -> list[str]:
    """Return ordered list of rich-item ids referenced by markers in
    ``markdown``.

    Markers may be standalone block markers or embedded in prose. Markers
    inside fenced or indented code blocks, inline code, or with invalid id
    characters/lengths are ignored. Duplicate references are preserved in
    occurrence order.
    """
    if not markdown:
        return []
    # Normalize CRLF without altering positions further.
    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    in_fence = _strip_fenced_code_blocks(lines)
    refs: list[str] = []
    for line, inside in zip(lines, in_fence, strict=False):
        if inside:
            continue
        if _is_indented_code(line):
            continue
        refs.extend(match.group(1) for match in _iter_inline_rich_marker_matches(line))
    return refs


def strip_inline_rich_markers(markdown: str) -> str:
    """Remove rich marker comments from markdown outside code contexts.

    This is used for clients that did not opt into the rich-response contract
    so marker comments cannot leak as visible text. Code fences, indented code,
    and inline code spans are preserved.
    """
    if not markdown or "<!--rich:" not in markdown:
        return markdown
    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    in_fence = _strip_fenced_code_blocks(lines)
    changed = False
    out: list[str] = []
    for line, inside in zip(lines, in_fence, strict=False):
        if inside or _is_indented_code(line):
            out.append(line)
            continue
        pieces: list[str] = []
        cursor = 0
        line_changed = False
        for match in _iter_inline_rich_marker_matches(line):
            pieces.append(line[cursor : match.start()])
            cursor = match.end()
            line_changed = True
            changed = True
        if line_changed:
            pieces.append(line[cursor:])
            out.append("".join(pieces))
        else:
            out.append(line)
    return "\n".join(out) if changed else markdown


def remove_inline_rich_reference(markdown: str, item_id: str) -> str:
    """Remove one rich-item marker outside code contexts.

    Used when optional media cannot be persisted. Other markers and literal
    marker examples in fenced, indented, or inline code remain untouched.
    """
    if not markdown or not item_id or "<!--rich:" not in markdown:
        return markdown
    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    in_fence = _strip_fenced_code_blocks(lines)
    changed = False
    output: list[str] = []
    for line, inside in zip(lines, in_fence, strict=False):
        if inside or _is_indented_code(line):
            output.append(line)
            continue
        pieces: list[str] = []
        cursor = 0
        for match in _iter_inline_rich_marker_matches(line):
            if match.group(1) != item_id:
                continue
            pieces.append(line[cursor : match.start()])
            cursor = match.end()
            changed = True
        if cursor:
            pieces.append(line[cursor:])
            output.append("".join(pieces))
        else:
            output.append(line)
    if not changed:
        return markdown
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip("\n")


# ---------------------------------------------------------------------------
# Validation warnings
# ---------------------------------------------------------------------------


def validate_rich_references(
    markdown: str, items: Iterable[RichItem | BaseModel | dict[str, Any]]
) -> list[dict[str, str]]:
    """Return a list of warning dicts for references that cannot be resolved.

    Currently emits ``{"code": "unknown_rich_item", "id": <id>}`` for each
    marker whose id is not present in ``items``. The warnings list is intended
    to be persisted alongside ``rich_items`` so renderers can show neutral
    unavailable-content blocks without crashing.
    """
    referenced = parse_inline_rich_references(markdown)
    known_ids = {
        getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
        for item in items
    }
    known_ids.discard(None)
    warnings: list[dict[str, str]] = []
    for ref in referenced:
        if ref not in known_ids:
            warnings.append({"code": "unknown_rich_item", "id": ref})
    return warnings


# ---------------------------------------------------------------------------
# Display-policy filters
# ---------------------------------------------------------------------------


def select_append_fallback_items(
    items: Iterable[RichItem], referenced_ids: set[str]
) -> list[RichItem]:
    """Return items that should be appended after the body because they are
    unreferenced and their display policy allows append.

    Image items with ``display_policy == inline_only`` are never appended even
    if their record exists, matching the selection-only image policy.
    """
    fallback: list[RichItem] = []
    for item in items:
        if item.id in referenced_ids:
            continue
        if item.display_policy != RichDisplayPolicy.inline_or_append:
            continue
        fallback.append(item)
    return fallback


def provenance_provider(item: Any) -> str:
    """Return an item's provenance provider, or ``"other"`` when absent.

    Shared so the metrics call sites that label by provider all read provenance
    the same way; three independent copies had drifted apart.
    """
    provenance = item.get("provenance") if isinstance(item, dict) else None
    if isinstance(provenance, dict):
        return str(provenance.get("provider") or "other")
    return "other"


def select_transient_upsert_items(
    items: Iterable[RichItem | BaseModel | dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the subset of ``items`` that may be streamed as transient
    ``rich_items`` upserts before final selection.

    The initial implementation streams only safe created non-image records.
    Image candidates in either shape (``image`` and ``image_group``), payloads
    carrying raw inline data, and canvas source are excluded entirely. Streaming
    an image candidate would both dump unselected candidates to the client and
    hand it pre-externalization upstream provider URLs, which the client would
    fetch directly from the third party. Returned items use the same public
    serialization as finalized rich items: null-valued keys are omitted and
    default fields such as ``provenance`` are materialized.
    """
    excluded_image_types = {RichItemType.image.value, RichItemType.image_group.value}
    safe: list[dict[str, Any]] = []
    for item in items:
        item_type = _get_type(item)
        if item_type is None or item_type in excluded_image_types:
            continue
        if item_type == RichItemType.canvas_artifact.value:
            # Canvas source is the asset; do not stream until terminal
            # finalization confirms the boundary review.
            continue
        if _payload_has_inline_binary(item):
            continue
        try:
            public_item = validate_public_rich_item(item)
        except Exception:
            continue
        safe.append(public_item.model_dump(mode="json", exclude_none=True))
    return safe


def _get_type(item: Any) -> str | None:
    raw = getattr(item, "type", None)
    if raw is None and isinstance(item, dict):
        raw = item.get("type")
    if isinstance(raw, RichItemType):
        return raw.value
    if isinstance(raw, str):
        return raw
    return None


def _payload_has_inline_binary(item: Any) -> bool:
    payload = getattr(item, "payload", None)
    if payload is None and isinstance(item, dict):
        payload = item.get("payload")
    if payload is None:
        return False
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    if not isinstance(payload, dict):
        return False
    if payload.get("data"):
        return True
    if payload.get("content") and _get_type(item) == RichItemType.canvas_artifact.value:
        return True
    return _contains_nested_inline_binary(payload)


def _contains_nested_inline_binary(value: Any) -> bool:
    if isinstance(value, list):
        return any(_contains_nested_inline_binary(item) for item in value)
    if not isinstance(value, dict):
        return False

    block_type = str(value.get("type") or "").lower()
    for key, nested in value.items():
        normalized_key = str(key)
        if normalized_key in {"base64", "b64_data"} and isinstance(nested, str) and nested.strip():
            return True
        if (
            normalized_key == "data"
            and isinstance(nested, str)
            and nested.strip()
            and block_type in {"image", "audio", "file", "resource"}
        ):
            return True
        if _contains_nested_inline_binary(nested):
            return True
    return False


# ---------------------------------------------------------------------------
# Bounded prompt inventory
# ---------------------------------------------------------------------------


_INVENTORY_HEADER = "AVAILABLE RICH ITEMS FOR OPTIONAL INLINE PLACEMENT:"
_INVENTORY_FOOTER = (
    "To display an item, copy its `<!--rich:...-->` marker (shown for that item) onto\n"
    "its own line, keeping the `rich:` prefix exactly — do not shorten it to\n"
    "`<!--<id>-->`. Use only items that materially support the answer. For an image,\n"
    "do not write a Markdown caption after the marker; the renderer owns the single\n"
    "structured figure footer. Do not invent item IDs."
)


def _summary_text(item: Any, summary_chars: int) -> str:
    title = getattr(item, "title", None)
    if title is None and isinstance(item, dict):
        title = item.get("title")
    payload = getattr(item, "payload", None)
    if payload is None and isinstance(item, dict):
        payload = item.get("payload")
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    description = None
    if isinstance(payload, dict):
        description = payload.get("description")
    pieces: list[str] = []
    if title:
        pieces.append(str(title))
    if description:
        pieces.append(str(description))
    text = " | ".join(pieces)
    if summary_chars > 0 and len(text) > summary_chars:
        text = text[: max(0, summary_chars - 1)].rstrip() + "…"
    return text


def _is_image_item(item: Any) -> bool:
    return _get_type(item) in {RichItemType.image.value, RichItemType.image_group.value}


def build_rich_item_inventory_block(
    items: Iterable[Any],
    *,
    max_items: int,
    max_chars: int,
    summary_chars: int,
    image_max_items: int,
) -> str:
    """Return a bounded human-readable inventory block listing rich items
    available for inline placement.

    The block contains only id/type/title/description text — never the raw
    payload, base64 data, or full URLs. When the item or character budget is
    exceeded, non-image items are kept in preference to image candidates so
    that an agent's created widget/tool/canvas record is never dropped before
    optional image candidates. ``image_max_items`` additionally caps how many
    image entries (an ``image_group`` counts as one) are offered at all; this
    module deliberately reads no config, so the caller must pass the
    per-answer image cap explicitly — there is no uncapped mode, since the
    parameter exists specifically to bound what reaches the model.
    """
    materialized = list(items)
    if not materialized:
        return ""

    # Stable ordering: non-image items first, preserving original order
    # within each group. This guarantees widget/tool/canvas/etc. survive
    # trimming before image candidates.
    non_image = [item for item in materialized if not _is_image_item(item)]
    image_items = [item for item in materialized if _is_image_item(item)]
    image_items = image_items[: max(0, int(image_max_items))]
    ordered = [*non_image, *image_items]
    if max_items > 0:
        ordered = ordered[:max_items]

    lines = [_INVENTORY_HEADER]
    for item in ordered:
        item_id = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
        item_type = _get_type(item) or "unknown"
        summary = _summary_text(item, summary_chars)
        marker = f"<!--rich:{item_id}-->"
        if summary:
            lines.append(f"- {marker} | {item_type} | {summary}")
        else:
            lines.append(f"- {marker} | {item_type}")
    lines.append("")
    lines.append(_INVENTORY_FOOTER)
    block = "\n".join(lines)
    if max_chars > 0 and len(block) > max_chars:
        # Trim by dropping trailing item lines (least valuable last per
        # ordering above) while keeping the header/footer guidance intact.
        while len(block) > max_chars and len(lines) > 3:
            # Remove the last non-footer, non-blank line.
            del lines[-3]
            block = "\n".join(lines)
        if len(block) > max_chars:
            # Last resort: hard truncate.
            block = block[:max_chars]
    return block


__all__ = [
    "ALLOWED_IMAGE_MIME_TYPES",
    "ALLOWED_URL_SCHEMES",
    "CanvasPayload",
    "CanvasRichItem",
    "GENERIC_IMAGE_ALT_TEXT",
    "CitationPayload",
    "CitationRichItem",
    "ImageGroupItem",
    "ImageGroupPayload",
    "ImageGroupRichItem",
    "ImagePayload",
    "ImageRichItem",
    "LiveWidgetPayload",
    "LiveWidgetRichItem",
    "RICH_ITEMS_VERSION",
    "RICH_ITEM_ID_MAX_LENGTH",
    "ResourceLinkPayload",
    "ResourceLinkRichItem",
    "RichDisplayPolicy",
    "RichItem",
    "RichItemBase",
    "RichItemType",
    "ToolRenderPayload",
    "ToolRenderRichItem",
    "build_rich_item_inventory_block",
    "parse_inline_rich_references",
    "remove_inline_rich_reference",
    "select_append_fallback_items",
    "select_transient_upsert_items",
    "strip_inline_rich_markers",
    "validate_rich_references",
    "validate_public_rich_item",
]
