"""Brave discovery, guarded thumbnail download, verification, admission.

Kept separate from the research tool so the orchestration order stays readable
and each stage is testable on its own.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..core.config import settings
from ..services.thumbnail_batch import fetch_thumbnails
from .tool_execution import (
    _group_image_candidates,
    build_image_candidates_from_tool_result,
)
from .visual_verifier import (
    SubmittedCandidate,
    admit_candidates,
    verify_candidates,
)

logger = logging.getLogger(__name__)


def _remaining_seconds(deadline_at: float) -> float:
    """Return actual wall-clock time left before ``deadline_at``, floored above 0.

    The configured per-stage settings (Brave's own timeout, the thumbnail
    timeout) are independent knobs that can drift out of sync with the overall
    image-verification deadline — e.g. a slow Brave call plus the full
    thumbnail timeout can already exceed the deadline, leaving the verifier
    call no time at all even though it was handed the full nominal setting.
    Deriving each stage's budget from what is actually left avoids handing a
    later stage a timeout the outer deadline will never let it use.
    """

    return max(0.001, deadline_at - time.monotonic())


async def discover_and_verify_images(
    *,
    brave_tool: Any | None,
    web_image_service: Any | None,
    verifier_model: Any | None,
    user_request: str,
    image_query: str,
    factual_query: str,
    image_intent: str | None = None,
) -> list[dict[str, Any]]:
    """Return public candidate dicts for verifier-approved images only.

    ``image_intent`` selects the layout and through it the cap: ``gallery``
    returns a single grid item, anything else returns individual images.
    """

    if brave_tool is None or web_image_service is None:
        return []

    # Mirrors the outer asyncio.timeout(image_verification_deadline_seconds)
    # the caller wraps this whole call in, so "remaining" below reflects real
    # time left against that same deadline rather than the raw setting.
    deadline_at = time.monotonic() + float(settings.image_verification_deadline_seconds)

    raw = str(await brave_tool.ainvoke({"query": image_query}))
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
    if not candidates:
        return []

    thumbnails = await fetch_thumbnails(
        web_image_service,
        [candidate["payload"]["url"] for candidate in candidates],
        provider="brave",
        per_item_timeout=min(
            float(settings.image_verification_thumbnail_timeout_seconds),
            _remaining_seconds(deadline_at),
        ),
        batch_deadline=_remaining_seconds(deadline_at),
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
    if not submitted:
        return []

    result = await verify_candidates(
        submitted,
        user_request=user_request,
        image_query=image_query,
        factual_query=factual_query,
        result_titles=[],
        model=verifier_model,
        timeout=_remaining_seconds(deadline_at),
    )
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
    by_id = {f"c{index}": candidate for index, candidate in enumerate(candidates)}
    public = [
        _with_decoded_dimensions(by_id[item.candidate_id], item)
        for item in approved
        if item.candidate_id in by_id
    ]
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
