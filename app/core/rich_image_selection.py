"""Pure, deterministic selection for public rich-image candidates."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

_IMAGE_TYPES = frozenset({"image", "image_group"})
_DIRECT_SOURCES = frozenset({"rag_document", "tool_image", "generated_image"})
_REMOTE_DISCOVERY_SOURCES = frozenset({"web_search", "image_search"})
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
_URL_WIDTH_PATTERNS = (
    re.compile(
        r"/scale-to-width-down/(?P<width>\d{1,5})(?:[/?.]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:[?&])(?:w|width)=(?P<width>\d{1,5})(?:&|$)",
        re.IGNORECASE,
    ),
)
_URL_HEIGHT_PATTERNS = (
    re.compile(
        r"(?:[?&])(?:h|height)=(?P<height>\d{1,5})(?:&|$)",
        re.IGNORECASE,
    ),
)
_URL_SIZE_PATTERN = re.compile(
    r"(?:^|[./_-])(?P<width>\d{1,5})x(?P<height>\d{1,5})(?:[./_-]|$)",
    re.IGNORECASE,
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


def is_junk_image_url(url: str) -> bool:
    """Return whether a URL path unambiguously names a non-content asset."""

    path = urlsplit(str(url or "")).path
    return bool(path) and any(pattern.search(path) for pattern in _JUNK_IMAGE_URL_PATTERNS)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _intent_rank(candidate: Mapping[str, Any]) -> int | None:
    source = str(candidate.get("source") or "")
    provenance = _mapping(candidate.get("provenance"))
    payload = _mapping(candidate.get("payload"))
    if source == "web_search" and bool(provenance.get("query_level")):
        return None
    if source in _DIRECT_SOURCES:
        return 0
    if source == "image_search":
        return 1
    if source == "web_search" and payload.get("source_url"):
        return 2
    if source == "web_search":
        return None
    return 1


def _normalized_tokens(value: object) -> frozenset[str]:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return frozenset(
        token
        for token in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
        if len(token) >= 3
    )


def _description_overlap(candidate: Mapping[str, Any]) -> float:
    provenance = _mapping(candidate.get("provenance"))
    payload = _mapping(candidate.get("payload"))
    query_tokens = _normalized_tokens(provenance.get("query"))
    if not query_tokens:
        return 0.0
    descriptive_tokens = _normalized_tokens(
        " ".join(
            str(value or "")
            for value in (
                candidate.get("title"),
                candidate.get("alt_text"),
                payload.get("description"),
                provenance.get("source_title"),
            )
        )
    )
    return len(query_tokens & descriptive_tokens) / len(query_tokens)


def _url_dimension_hint(url: str) -> tuple[int | None, int | None]:
    width: int | None = None
    height: int | None = None
    size_match = _URL_SIZE_PATTERN.search(url)
    if size_match:
        width = int(size_match.group("width"))
        height = int(size_match.group("height"))
    for pattern in _URL_WIDTH_PATTERNS:
        if match := pattern.search(url):
            width = int(match.group("width"))
            break
    for pattern in _URL_HEIGHT_PATTERNS:
        if match := pattern.search(url):
            height = int(match.group("height"))
            break
    return width, height


def _positive_dimension(value: Any) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _payload_dimensions(payload: Mapping[str, Any]) -> tuple[int | None, int | None]:
    url = str(payload.get("url") or "")
    hinted_width, hinted_height = _url_dimension_hint(url)
    return (
        _positive_dimension(payload.get("width")) or hinted_width,
        _positive_dimension(payload.get("height")) or hinted_height,
    )


def _payload_is_eligible(
    payload: Mapping[str, Any],
    *,
    policy: ImageSelectionPolicy,
    require_remote_url: bool,
) -> bool:
    url = str(payload.get("url") or "").strip()
    data = payload.get("data")
    if not url and not data:
        return False
    if require_remote_url and url:
        if urlsplit(url).scheme.lower() != "https" or is_junk_image_url(url):
            return False
    elif require_remote_url and not data:
        return False
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
    for payload in _candidate_payloads(candidate):
        if not _payload_is_eligible(
            payload,
            policy=policy,
            require_remote_url=require_remote_url,
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
    return (
        99 if intent is None else intent,
        _quality_rank(candidate, policy),
        -_description_overlap(candidate),
        result_rank,
        index,
    )


def _payload_locators(payload: Mapping[str, Any]) -> tuple[str, ...]:
    locators: list[str] = []
    for key in ("url", "original_url", "source_image_url"):
        value = payload.get(key)
        if isinstance(value, str) and (normalized := value.strip()):
            locators.append(normalized)
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
    payload = _mapping(candidate.get("payload"))
    if candidate.get("type") != "image_group":
        if not _payload_is_eligible(
            payload,
            policy=policy,
            require_remote_url=require_remote_url,
        ):
            return None
        locators = _payload_locators(payload)
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
        ):
            continue
        locators = set(_payload_locators(item))
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
