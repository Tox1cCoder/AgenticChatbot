"""Concurrent, individually-isolated thumbnail downloads for visual verification.

One blocked or malformed candidate must cost only itself. The batch deadline is
the hard bound: whatever has not arrived by then is treated as absent, because
the answer is already waiting on it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.services.web_image_service import FetchedWebImage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FetchedThumbnail:
    """One successfully fetched and decoded candidate thumbnail."""

    url: str
    image: FetchedWebImage


async def fetch_thumbnails(
    service: Any,
    urls: Sequence[str],
    *,
    provider: str,
    per_item_timeout: float,
    batch_deadline: float,
) -> list[FetchedThumbnail | None]:
    """Fetch every URL concurrently. Results align positionally with ``urls``."""

    if not urls:
        return []

    async def _one(url: str) -> FetchedThumbnail | None:
        try:
            async with asyncio.timeout(max(0.001, float(per_item_timeout))):
                image = await service.fetch_url(url, provider=provider)
        except Exception as exc:
            logger.debug("Thumbnail candidate dropped: %s", type(exc).__name__)
            return None
        return FetchedThumbnail(url=url, image=image)

    tasks = [asyncio.create_task(_one(url)) for url in urls]
    try:
        async with asyncio.timeout(max(0.001, float(batch_deadline))):
            return list(await asyncio.gather(*tasks))
    except TimeoutError:
        results: list[FetchedThumbnail | None] = []
        for task in tasks:
            if task.done() and not task.cancelled() and task.exception() is None:
                results.append(task.result())
            else:
                task.cancel()
                results.append(None)
        return results
