"""Pure, deterministic selection for public rich-image candidates."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from .rich_response import (
    ALLOWED_IMAGE_MIME_TYPES,
    ALLOWED_URL_SCHEMES,
    PROTECTED_IMAGE_URL_PREFIXES,
)

_IMAGE_TYPES = frozenset({"image", "image_group"})
_DIRECT_SOURCES = frozenset({"rag_document", "tool_image", "generated_image"})
_REMOTE_DISCOVERY_SOURCES = frozenset({"image_search"})
_BASE64_DATA_PATTERN = re.compile(
    r"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?\Z"
)
_IMAGE_PAYLOAD_KEYS = frozenset(
    {"url", "data", "mime_type", "source_url", "description", "width", "height", "caption"}
)
_GROUP_CELL_KEYS = frozenset(
    {"url", "mime_type", "source_url", "description", "width", "height"}
)
_JUNK_IMAGE_URL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"favicon", re.IGNORECASE),
    re.compile(r"sprite", re.IGNORECASE),
    re.compile(r"spacer", re.IGNORECASE),
    re.compile(r"(?:tracking[-_]?pixel|pixel\.gif)", re.IGNORECASE),
    re.compile(r"(?:^|[/_.-])1x1(?!\d)", re.IGNORECASE),
    re.compile(r"(?:default[-_]?avatar|avatar[-_]?placeholder)", re.IGNORECASE),
    re.compile(r"/avatars?/default(?:[/_.-]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[/_.-])placeholder(?:[/_.-]|$)", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class ImageSelectionPolicy:
    """Bounds used by the in-memory image selector."""

    max_items: int
    min_width_px: int
    min_height_px: int
    min_aspect_ratio: float
    max_aspect_ratio: float


def is_image_candidate(candidate: Mapping[str, Any]) -> bool:
    """Return whether a rich candidate is an image or image group."""

    return str(candidate.get("type") or "") in _IMAGE_TYPES


def image_aspect_ratio_ok(
    width: Any,
    height: Any,
    *,
    minimum: float,
    maximum: float,
) -> bool:
    """Return false only for a known positive ratio outside the bounds."""

    if not isinstance(width, int) or not isinstance(height, int):
        return True
    if width <= 0 or height <= 0:
        return True
    return minimum <= width / height <= maximum


def image_url_scheme(url: Any) -> str | None:
    """Return a normalized URL scheme, or ``None`` for malformed input."""

    try:
        return urlsplit(str(url or "")).scheme.lower()
    except ValueError:
        return None


def is_junk_image_url(url: str) -> bool:
    """Return whether a URL path unambiguously names a non-content asset."""

    try:
        path = urlsplit(str(url or "")).path
    except ValueError:
        return True
    return bool(path) and any(pattern.search(path) for pattern in _JUNK_IMAGE_URL_PATTERNS)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _intent_rank(candidate: Mapping[str, Any]) -> int | None:
    source = str(candidate.get("source") or "")
    if source in _DIRECT_SOURCES:
        return 0
    if source == "image_search":
        return 1
    if source == "web_search":
        return None
    return 1


def _positive_dimension(value: Any) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _payload_dimensions(payload: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """Dimensions come from decoded bytes only.

    A URL's resize parameter is not an intrinsic dimension: reading ``&w=3840``
    beside a real height of 565 fabricated a 6.8 aspect ratio and rejected the
    one relevant image in the T1 trace.
    """
    return (
        _positive_dimension(payload.get("width")),
        _positive_dimension(payload.get("height")),
    )


def _payload_is_eligible(
    payload: Mapping[str, Any],
    *,
    policy: ImageSelectionPolicy,
    require_remote_url: bool,
    allow_data: bool,
) -> bool:
    allowed_keys = _IMAGE_PAYLOAD_KEYS if allow_data else _GROUP_CELL_KEYS
    if any(key not in allowed_keys for key in payload):
        return False
    url = str(payload.get("url") or "").strip()
    data = payload.get("data")
    has_url = bool(url)
    has_data = isinstance(data, str) and bool(data)
    if has_url == has_data or (has_data and not allow_data):
        return False
    if has_data and _BASE64_DATA_PATTERN.fullmatch(data) is None:
        return False
    if payload.get("mime_type") not in ALLOWED_IMAGE_MIME_TYPES:
        return False
    source_url = payload.get("source_url")
    if source_url is not None and not _is_allowed_absolute_url(source_url):
        return False
    for dimension_key in ("width", "height"):
        dimension = payload.get(dimension_key)
        if dimension is not None and (
            not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1
        ):
            return False
    caption = payload.get("caption")
    if caption is not None and (not isinstance(caption, str) or len(caption) > 500):
        return False
    if require_remote_url and has_url:
        if image_url_scheme(url) != "https" or is_junk_image_url(url):
            return False
    elif (require_remote_url and not has_data) or (
        has_url and not _is_allowed_image_url(url)
    ):
        return False
    if not require_remote_url:
        return True
    width, height = _payload_dimensions(payload)
    if width is not None and width < policy.min_width_px:
        return False
    if height is not None and height < policy.min_height_px:
        return False
    return image_aspect_ratio_ok(
        width,
        height,
        minimum=policy.min_aspect_ratio,
        maximum=policy.max_aspect_ratio,
    )


def _is_allowed_absolute_url(value: Any) -> bool:
    if not isinstance(value, str) or "://" not in value:
        return False
    return image_url_scheme(value) in ALLOWED_URL_SCHEMES


def _is_allowed_image_url(url: str) -> bool:
    if url.startswith(PROTECTED_IMAGE_URL_PREFIXES):
        return True
    if url.startswith("/"):
        return False
    return _is_allowed_absolute_url(url)


def _candidate_payloads(candidate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    payload = _mapping(candidate.get("payload"))
    if candidate.get("type") != "image_group":
        return [payload]
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, Mapping)]


def _quality_rank(candidate: Mapping[str, Any], policy: ImageSelectionPolicy) -> int:
    require_remote_url = str(candidate.get("source") or "") in _REMOTE_DISCOVERY_SOURCES
    allow_data = candidate.get("type") != "image_group"
    for payload in _candidate_payloads(candidate):
        if not _payload_is_eligible(
            payload,
            policy=policy,
            require_remote_url=require_remote_url,
            allow_data=allow_data,
        ):
            continue
        width, height = _payload_dimensions(payload)
        if width is not None and height is not None:
            return 0
    return 1


def _rank_key(
    indexed: tuple[int, dict[str, Any]],
    *,
    policy: ImageSelectionPolicy,
) -> tuple[float | int, ...]:
    index, candidate = indexed
    intent = _intent_rank(candidate)
    provenance = _mapping(candidate.get("provenance"))
    raw_result_rank = provenance.get("result_rank")
    result_rank = (
        raw_result_rank
        if isinstance(raw_result_rank, int) and raw_result_rank >= 0
        else 1_000_000
    )
    provider = str(provenance.get("provider") or "").strip().lower()
    if (
        str(candidate.get("source") or "") == "image_search"
        and provider.startswith("brave")
    ):
        return (
            99 if intent is None else intent,
            result_rank,
            _quality_rank(candidate, policy),
            index,
        )
    return (
        99 if intent is None else intent,
        _quality_rank(candidate, policy),
        result_rank,
        index,
    )


def _payload_locators(
    payload: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[str, ...]:
    locators: list[str] = []
    for key in ("url", "original_url", "source_image_url"):
        value = payload.get(key)
        if isinstance(value, str) and (normalized := value.strip()):
            locators.append(normalized)
    display_url = payload.get("url")
    original_digests = provenance.get("original_image_digests")
    if isinstance(display_url, str) and isinstance(original_digests, Mapping):
        digest = original_digests.get(display_url)
        if isinstance(digest, str) and digest:
            locators.append(f"original-digest::{digest}")
    return tuple(locators)


def _normalize_candidate(
    candidate: dict[str, Any],
    *,
    policy: ImageSelectionPolicy,
    seen_locators: set[str],
) -> dict[str, Any] | None:
    require_remote_url = (
        str(candidate.get("source") or "") in _REMOTE_DISCOVERY_SOURCES
    )
    provenance = _mapping(candidate.get("provenance"))
    payload = _mapping(candidate.get("payload"))
    if candidate.get("type") != "image_group":
        if not _payload_is_eligible(
            payload,
            policy=policy,
            require_remote_url=require_remote_url,
            allow_data=True,
        ):
            return None
        locators = _payload_locators(payload, provenance)
        if seen_locators.intersection(locators):
            return None
        seen_locators.update(locators)
        return candidate

    items = payload.get("items")
    if not isinstance(items, list):
        return None
    surviving_items: list[dict[str, Any]] = []
    claimed_locators: set[str] = set()
    for raw_item in items:
        if not isinstance(raw_item, Mapping):
            continue
        item = deepcopy(dict(raw_item))
        if not _payload_is_eligible(
            item,
            policy=policy,
            require_remote_url=require_remote_url,
            allow_data=False,
        ):
            continue
        locators = set(_payload_locators(item, provenance))
        if seen_locators.intersection(locators) or claimed_locators.intersection(locators):
            continue
        surviving_items.append(item)
        claimed_locators.update(locators)
    if not surviving_items:
        return None
    seen_locators.update(claimed_locators)
    if len(surviving_items) == 1:
        candidate["type"] = "image"
        candidate["payload"] = surviving_items[0]
        return candidate
    normalized_payload = deepcopy(dict(payload))
    normalized_payload["items"] = surviving_items
    candidate["payload"] = normalized_payload
    return candidate


def _select_ranked_images(
    indexed_candidates: list[tuple[int, dict[str, Any]]],
    policy: ImageSelectionPolicy,
) -> list[dict[str, Any]]:
    if policy.max_items <= 0:
        return []
    indexed_candidates.sort(key=lambda indexed: _rank_key(indexed, policy=policy))
    selected: list[dict[str, Any]] = []
    seen_locators: set[str] = set()
    for _, candidate in indexed_candidates:
        normalized = _normalize_candidate(
            candidate,
            policy=policy,
            seen_locators=seen_locators,
        )
        if normalized is None:
            continue
        selected.append(normalized)
        if len(selected) == policy.max_items:
            break
    return selected


def select_rich_item_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    policy: ImageSelectionPolicy,
) -> list[dict[str, Any]]:
    """Return a stable, capped candidate sequence without mutating input."""

    copied = [deepcopy(dict(candidate)) for candidate in candidates]
    non_images = [candidate for candidate in copied if not is_image_candidate(candidate)]
    indexed_images = [
        (index, candidate)
        for index, candidate in enumerate(copied)
        if is_image_candidate(candidate) and _intent_rank(candidate) is not None
    ]
    return [*non_images, *_select_ranked_images(indexed_images, policy)]
