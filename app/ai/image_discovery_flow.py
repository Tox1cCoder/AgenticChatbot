"""Deterministic selection of normalized Brave image-search results."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

from ..core.config import settings
from ..observability.rich_images import rich_image_metrics
from .tool_execution import _group_image_candidates, build_image_candidates_from_tool_result
from .tool_result_rendering import provider_result_text

logger = logging.getLogger(__name__)

_OPERATIONAL_FAILURES = frozenset({"unavailable", "search_failure"})


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


def select_brave_candidates(
    raw: str,
    *,
    image_query: str,
    image_intent: str | None = None,
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
    high = [item for item in candidates if _confidence(item) == "high"]
    medium = [item for item in candidates if _confidence(item) == "medium"]
    tier = high or medium
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
        raw, image_query=image_query, image_intent=image_intent
    )
    record_discovery_outcome("selected" if selected else "no_match", started=started)
    return selected
