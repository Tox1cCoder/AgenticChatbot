"""Deterministic selection of normalized Brave image-search results."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any

from ..core.config import settings
from ..observability.rich_images import rich_image_metrics
from .tool_execution import _group_image_candidates, build_image_candidates_from_tool_result
from .tool_result_rendering import provider_result_text

logger = logging.getLogger(__name__)

_OPERATIONAL_FAILURES = frozenset({"unavailable", "search_failure"})

_TIME_RANGE_WINDOW_DAYS = {"day": 1, "week": 7, "month": 31, "year": 366}


def _window_days(time_range: str | None) -> int | None:
    """Return the recency window the request explicitly declared, or ``None``.

    Only ``time_range`` declares one. ``topic`` states what kind of source to
    search, not how recent the answer must be, and a window inferred from it
    would discard images against a cutoff nobody asked for. Absent a declared
    window there is no window at all: ``page_fetched`` is a crawl time, not a
    subject date, and for most subjects a decades-old photograph is the right
    picture.
    """
    return _TIME_RANGE_WINDOW_DAYS.get(str(time_range or "").strip().lower())


def _crawled_at(candidate: Mapping[str, Any]) -> datetime | None:
    provenance = candidate.get("provenance")
    raw = provenance.get("page_fetched") if isinstance(provenance, Mapping) else None
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _drop_stale(
    candidates: list[dict[str, Any]], *, window_days: int | None
) -> list[dict[str, Any]]:
    """Drop candidates whose crawl date is known to precede the claimed window.

    An unknown crawl date is never stale: the provider does not stamp every
    result, and treating silence as staleness would disable discovery for whole
    classes of query.
    """
    if window_days is None:
        return candidates
    cutoff = datetime.now(UTC) - timedelta(days=window_days)
    return [
        candidate
        for candidate in candidates
        if (crawled := _crawled_at(candidate)) is None or crawled >= cutoff
    ]


def _object_payload(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _might_be_offensive(payload: Mapping[str, Any]) -> bool:
    safety = payload.get("safety")
    return isinstance(safety, Mapping) and bool(safety.get("might_be_offensive"))


def _confidence(candidate: Mapping[str, Any]) -> str:
    provenance = candidate.get("provenance")
    if not isinstance(provenance, Mapping):
        return ""
    return str(provenance.get("confidence") or "").strip().lower()


def _stable_deduplicate(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first candidate for each display URL or original-image digest."""

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        payload = candidate.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        display_url = str(payload.get("url") or "").strip()
        locators = {f"display::{display_url}"} if display_url else set()
        provenance = candidate.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        digests = provenance.get("original_image_digests")
        if display_url and isinstance(digests, Mapping):
            digest = digests.get(display_url)
            if digest:
                locators.add(f"original::{digest}")
        if seen.intersection(locators):
            continue
        selected.append(candidate)
        seen.update(locators)
    return selected


def select_brave_candidates(
    raw: str,
    *,
    image_query: str,
    image_intent: str | None = None,
    time_range: str | None = None,
) -> list[dict[str, Any]]:
    """Return only high-confidence Brave candidates, or medium as a fallback."""
    payload = _object_payload(raw)
    if payload.get("error") or _might_be_offensive(payload):
        return []
    candidates = build_image_candidates_from_tool_result(
        raw,
        tool_call_id=None,
        tool_name="brave_image_search",
        group_images=False,
        apply_candidate_cap=False,
    )
    # Staleness is settled before the confidence tier: an all-stale high tier
    # would otherwise shadow the fresh medium results that should be shown.
    candidates = _drop_stale(candidates, window_days=_window_days(time_range))
    high = [item for item in candidates if _confidence(item) == "high"]
    medium = [item for item in candidates if _confidence(item) == "medium"]
    tier = _stable_deduplicate(high or medium)
    if str(image_intent or "figure").lower() == "gallery" and len(tier) >= 2:
        return [
            _group_image_candidates(
                tier,
                tool_call_id=None,
                query=image_query,
                metric_provider="brave",
                max_items=max(2, int(settings.rich_image_gallery_max_items)),
            )
        ]
    return tier[: max(0, int(settings.rich_auto_place_max_images))]


def record_discovery_outcome(
    outcome: str, *, started: float | None = None
) -> list[dict[str, Any]]:
    """Best-effort terminal telemetry for a discovery attempt."""
    elapsed = 0.0 if started is None else time.perf_counter() - started
    with suppress(Exception):
        rich_image_metrics.record_discovery_outcome(
            outcome=outcome, duration_seconds=elapsed
        )
    if outcome in _OPERATIONAL_FAILURES:
        logger.warning(
            "Image discovery produced no image (%s) after %.2fs", outcome, elapsed
        )
    return []


async def discover_images(
    *,
    brave_tool: Any | None,
    image_query: str,
    image_intent: str | None = None,
    time_range: str | None = None,
) -> list[dict[str, Any]]:
    """Discover and deterministically select Brave images without verification."""
    started = time.perf_counter()
    if brave_tool is None:
        return record_discovery_outcome("unavailable", started=started)
    try:
        raw = provider_result_text(
            await brave_tool.ainvoke({"query": image_query}),
            tool_name="brave_image_search",
        )
    except Exception:
        return record_discovery_outcome("search_failure", started=started)
    payload = _object_payload(raw)
    if payload.get("error"):
        return record_discovery_outcome("search_failure", started=started)
    selected = select_brave_candidates(
        raw,
        image_query=image_query,
        image_intent=image_intent,
        time_range=time_range,
    )
    record_discovery_outcome("selected" if selected else "no_match", started=started)
    return selected
