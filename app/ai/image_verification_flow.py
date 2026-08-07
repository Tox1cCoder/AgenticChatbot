"""Brave discovery, guarded thumbnail download, verification, admission.

Kept separate from the research tool so the orchestration order stays readable
and each stage is testable on its own.
"""

from __future__ import annotations

import logging
import time
from contextlib import suppress
from typing import Any

from ..core.config import settings
from ..observability.rich_images import rich_image_metrics
from ..services.thumbnail_batch import fetch_thumbnails
from ..services.verified_image_bytes import remember_verified_bytes
from .tool_context import get_tool_context
from .tool_execution import (
    _group_image_candidates,
    build_image_candidates_from_tool_result,
)
from .tool_result_rendering import provider_result_text
from .visual_verifier import (
    SubmittedCandidate,
    admit_candidates,
    verify_candidates,
)

logger = logging.getLogger(__name__)

# Outcomes that mean the machinery failed, as opposed to it working and finding
# nothing worth showing. "no_match" is deliberately absent: a verifier that
# rejects every candidate is the feature doing its job.
_OPERATIONAL_FAILURES = frozenset({"timeout", "malformed", "transport", "unavailable"})


async def discover_and_verify_images(
    *,
    brave_tool: Any | None,
    web_image_service: Any | None,
    verifier_model: Any | None,
    user_request: str,
    image_query: str,
    factual_query: str,
    image_intent: str | None = None,
    recorder: Any | None = None,
) -> list[dict[str, Any]]:
    """Return public candidate dicts for verifier-approved images only.

    ``image_intent`` selects the layout and through it the cap: ``gallery``
    returns a single grid item, anything else returns individual images.
    """

    started = time.perf_counter()

    def _outcome(label: str) -> None:
        elapsed = time.perf_counter() - started
        with suppress(Exception):
            rich_image_metrics.record_verification_outcome(
                outcome=label, duration_seconds=elapsed
            )
        if label in _OPERATIONAL_FAILURES:
            # Every failure here is a successful text-only answer by design, so
            # a total outage is indistinguishable from "no good image found"
            # unless it says so. A misconfigured deadline cancelled every
            # verifier call for days and looked exactly like normal operation,
            # because only a metric nobody was watching recorded it.
            logger.warning(
                "Image verification produced no image (%s) after %.2fs; "
                "answers will be text-only until this clears.",
                label,
                elapsed,
            )

    if brave_tool is None or web_image_service is None:
        _outcome("unavailable")
        return []

    raw = provider_result_text(
        await brave_tool.ainvoke({"query": image_query}), tool_name="brave_image_search"
    )
    # group_images=False is load-bearing: the legacy path collapses two or more
    # Brave results into one capped grid, which would both hide individual images
    # from the verifier and cap discovery below the candidate budget.
    candidates = build_image_candidates_from_tool_result(
        raw,
        tool_call_id=None,
        tool_name="brave_image_search",
        group_images=False,
    )
    candidates = candidates[: max(1, int(settings.image_verification_max_candidates))]
    with suppress(Exception):
        rich_image_metrics.record_verification(stage="discovered", count=len(candidates))
    if not candidates:
        # Brave returned nothing to verify -- no separate outcome label exists
        # for "discovery was empty", so this is the same terminal state as
        # zero approvals: no image is available to show.
        _outcome("no_match")
        return []

    thumbnails = await fetch_thumbnails(
        web_image_service,
        [candidate["payload"]["url"] for candidate in candidates],
        provider="brave",
        per_item_timeout=float(settings.image_verification_thumbnail_timeout_seconds),
        batch_deadline=float(settings.image_verification_thumbnail_timeout_seconds),
    )
    with suppress(Exception):
        rich_image_metrics.record_verification(
            stage="fetched", count=sum(1 for thumbnail in thumbnails if thumbnail is not None)
        )
    submitted = [
        SubmittedCandidate(
            candidate_id=f"c{index}",
            thumbnail=thumbnail,
            title=str(candidate.get("title") or ""),
            description=str(candidate["payload"].get("description") or ""),
        )
        for index, (candidate, thumbnail) in enumerate(
            zip(candidates, thumbnails, strict=True)
        )
        if thumbnail is not None
    ]
    with suppress(Exception):
        rich_image_metrics.record_verification(stage="submitted", count=len(submitted))
    if not submitted:
        _outcome("transport")
        return []

    result = await verify_candidates(
        submitted,
        user_request=user_request,
        image_query=image_query,
        factual_query=factual_query,
        result_titles=[],
        model=verifier_model,
        timeout=float(settings.image_verification_timeout_seconds),
        recorder=recorder,
    )
    if result is None:
        _outcome("malformed")
        return []
    gallery = str(image_intent or "figure").strip().lower() == "gallery"
    approved = admit_candidates(
        result,
        submitted,
        threshold=float(settings.image_verification_confidence_threshold),
        max_items=(
            max(2, int(settings.rich_image_gallery_max_items))
            if gallery
            else max(0, int(settings.rich_auto_place_max_images))
        ),
        requested_kinds=_requested_kinds(f"{user_request} {image_query}"),
    )
    with suppress(Exception):
        rich_image_metrics.record_verification(stage="approved", count=len(approved))
    if not approved:
        _outcome("no_match")
        return []
    by_id = {f"c{index}": candidate for index, candidate in enumerate(candidates)}
    public = [
        _with_decoded_dimensions(by_id[item.candidate_id], item)
        for item in approved
        if item.candidate_id in by_id
    ]
    _hold_verified_bytes(approved)
    _outcome("approved")
    if gallery and len(public) >= 2:
        return [
            _group_image_candidates(
                public,
                tool_call_id=None,
                query=image_query,
                metric_provider="brave",
                max_items=max(2, int(settings.rich_image_gallery_max_items)),
            )
        ]
    return public


def _hold_verified_bytes(approved: list[SubmittedCandidate]) -> None:
    """Keep the approved images' validated bytes for registration to persist.

    Registration happens later, in ``message_service``, long after this tool
    call's ContextVar scope has closed — so the bytes are handed to a bounded
    turn-scoped store rather than carried in the candidate dict, which is
    serialized into response metadata and must stay small.

    Losing the hand-off is not a failure: registration then stores no bytes and
    the image is fetched again at render, exactly as before.
    """

    conversation_id = get_tool_context().conversation_id
    for candidate in approved:
        with suppress(Exception):
            remember_verified_bytes(
                conversation_id,
                candidate.thumbnail.url,
                candidate.thumbnail.image,
            )


def _with_decoded_dimensions(
    candidate: dict[str, Any], submitted: SubmittedCandidate
) -> dict[str, Any]:
    """Stamp the dimensions actually decoded from the bytes we fetched."""

    updated = {**candidate, "payload": dict(candidate["payload"])}
    updated["payload"]["width"] = submitted.thumbnail.image.width
    updated["payload"]["height"] = submitted.thumbnail.image.height
    updated["payload"]["mime_type"] = submitted.thumbnail.image.media_type
    return updated


_KIND_WORDS = {
    "portrait": ("portrait", "headshot", "chân dung"),
    "logo": ("logo", "emblem", "badge"),
    "diagram": ("diagram", "schematic", "sơ đồ"),
    "map": ("map", "bản đồ"),
    "chart": ("chart", "graph", "biểu đồ"),
    "screenshot": ("screenshot", "screen shot", "ảnh chụp màn hình"),
}


def _requested_kinds(text: str) -> frozenset[str]:
    """Kinds the user explicitly asked for, which overrides the generic bias."""

    lowered = str(text or "").casefold()
    return frozenset(
        kind for kind, words in _KIND_WORDS.items() if any(word in lowered for word in words)
    )
