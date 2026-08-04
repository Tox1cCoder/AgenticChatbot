"""Workflow boundary for canonical rich-image selection."""

from __future__ import annotations

import logging
import time
from collections.abc import MutableMapping
from contextlib import suppress
from typing import Any

from app.core.config import settings
from app.core.rich_image_selection import (
    ImageSelectionPolicy,
    is_image_candidate,
    select_rich_item_candidates,
)
from app.observability.rich_images import rich_image_metrics

logger = logging.getLogger(__name__)


def _configured_policy() -> ImageSelectionPolicy:
    return ImageSelectionPolicy(
        max_items=max(0, settings.rich_auto_place_max_images),
        min_width_px=settings.rich_image_min_width_px,
        min_height_px=settings.rich_image_min_height_px,
        min_aspect_ratio=settings.rich_image_min_aspect_ratio,
        max_aspect_ratio=settings.rich_image_max_aspect_ratio,
    )


def apply_rich_image_selection(context: MutableMapping[str, Any]) -> None:
    """Replace the context pool with its selected, fail-closed sequence."""

    context.pop("_presented_rich_image_ids", None)
    raw_candidates = context.get("rich_item_candidates")
    candidates = (
        [item for item in raw_candidates if isinstance(item, dict)]
        if isinstance(raw_candidates, list)
        else []
    )
    started_at = time.perf_counter()
    try:
        context["rich_item_candidates"] = select_rich_item_candidates(
            candidates,
            policy=_configured_policy(),
        )
    except Exception:
        logger.exception(
            "Rich image selection failed; continuing without discovered images"
        )
        context["rich_item_candidates"] = [
            item for item in candidates if not is_image_candidate(item)
        ]
    finally:
        with suppress(Exception):
            rich_image_metrics.record_selection_duration(
                time.perf_counter() - started_at
            )
