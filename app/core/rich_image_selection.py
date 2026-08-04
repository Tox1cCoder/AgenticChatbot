"""Pure, deterministic selection for public rich-image candidates."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

_IMAGE_TYPES = frozenset({"image", "image_group"})
_DIRECT_SOURCES = frozenset({"rag_document", "tool_image", "generated_image"})


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


def _rank_key(indexed: tuple[int, dict[str, Any]]) -> tuple[float | int, ...]:
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
        -_description_overlap(candidate),
        result_rank,
        index,
    )


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
    indexed_images.sort(key=_rank_key)
    max_items = max(0, policy.max_items)
    return [
        *non_images,
        *(candidate for _, candidate in indexed_images[:max_items]),
    ]
