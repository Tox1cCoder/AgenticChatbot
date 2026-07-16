import json
import os
import sys
from pathlib import Path
from typing import Any

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

import contextlib  # noqa: E402

import httpx  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

from app.core.config import settings  # noqa: E402

BRAVE_IMAGE_SEARCH_URL = "https://api.search.brave.com/res/v1/images/search"
SUPPORTED_SAFESEARCH = ("off", "strict")

mcp = FastMCP("Brave Image Search")


def _error(message: str, *, retryable: bool = False) -> str:
    payload: dict[str, Any] = {
        "error": message,
        "provider": "brave_image_search",
        "images": [],
        "total_results": 0,
    }
    if retryable:
        payload["retryable"] = True
    return json.dumps(payload)


def _guess_mime_from_url(url: str) -> str | None:
    lowered = url.lower().split("?", 1)[0]
    if lowered.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lowered.endswith(".webp"):
        return "image/webp"
    if lowered.endswith(".gif"):
        return "image/gif"
    if lowered.endswith(".png"):
        return "image/png"
    return None


def _resolve_api_key() -> str | None:
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        with contextlib.suppress(Exception):
            api_key = settings.brave_search_api_key
    return api_key or None


def _clamp_count(count: int | None) -> int:
    default_count = int(getattr(settings, "brave_image_search_default_count", 6) or 6)
    max_count = int(getattr(settings, "brave_image_search_max_count", 10) or 10)
    if count is None:
        count = default_count
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = default_count
    return max(1, min(count, max_count))


def _resolve_safesearch(safesearch: str | None) -> tuple[str | None, str | None]:
    """Return ``(value, invalid)``. Exactly one element is non-None.

    ``None`` input falls back to the configured default. An unsupported explicit
    value is reported as ``invalid`` so the caller can return an argument error
    rather than forwarding it to Brave.
    """
    if safesearch is None:
        return str(getattr(settings, "brave_image_search_default_safesearch", "strict")), None
    value = str(safesearch).strip().lower()
    if value not in SUPPORTED_SAFESEARCH:
        return None, value
    return value, None


def _normalize_results(query: str, data: Any) -> dict[str, Any]:
    """Map raw Brave image results to the compact rich-pipeline shape.

    Brave distinguishes the page URL (``result.url``) from the direct image URL
    (``result.properties.url``). Only the direct image URL is renderable, so
    entries without it are skipped rather than falling back to the page URL.
    """
    results = data.get("results") if isinstance(data, dict) else None
    images: list[dict[str, Any]] = []
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            properties = result.get("properties")
            properties = properties if isinstance(properties, dict) else {}
            direct_url = properties.get("url")
            if not direct_url:
                continue

            title = result.get("title")
            source = result.get("source")
            meta_url = result.get("meta_url")
            hostname = meta_url.get("hostname") if isinstance(meta_url, dict) else None
            thumbnail = result.get("thumbnail")
            thumbnail_url = thumbnail.get("src") if isinstance(thumbnail, dict) else None

            image: dict[str, Any] = {
                "url": str(direct_url),
                "provider": "brave_image_search",
            }
            mime_type = _guess_mime_from_url(str(direct_url))
            if mime_type:
                image["mime_type"] = mime_type
            if result.get("url"):
                image["source_url"] = str(result["url"])
            if thumbnail_url:
                image["thumbnail_url"] = str(thumbnail_url)
            if title:
                image["title"] = str(title)
            description = title or source or hostname
            if description:
                image["description"] = str(description)
            if hostname:
                image["source_domain"] = str(hostname)
            width = properties.get("width")
            height = properties.get("height")
            if isinstance(width, int):
                image["width"] = width
            if isinstance(height, int):
                image["height"] = height

            images.append(image)

    return {
        "query": query,
        "provider": "brave_image_search",
        "images": images,
        "total_results": len(images),
    }


def brave_image_search(
    query: str,
    count: int | None = None,
    country: str | None = None,
    search_lang: str | None = None,
    safesearch: str | None = None,
) -> str:
    """Find real images of a subject using the Brave Image Search API.

    Use this when a visual reference would materially help the answer — e.g.
    "what does X look like", showcase/gallery requests, or article-style answers
    that benefit from inline photos. Returns normalized JSON with an ``images``
    array of direct, renderable image URLs and compact descriptions. Place only
    relevant returned images inline near the text they support.

    Args:
        query: What to find images of. Be specific.
        count: Number of images to return (clamped to a configured maximum).
        country: Optional 2-letter country code to localize results.
        search_lang: Optional language code for results.
        safesearch: Brave safesearch level — "off" or "strict" (default "strict").

    Returns:
        JSON string with ``query``, ``provider``, ``images`` (url, source_url,
        thumbnail_url, title, description, dimensions), and ``total_results``.
        On failure returns JSON with an ``error`` message so the answer can
        continue text-only.
    """
    api_key = _resolve_api_key()
    if not api_key:
        return _error("BRAVE_SEARCH_API_KEY not configured. Set it in environment or config.py")

    safesearch_value, invalid = _resolve_safesearch(safesearch)
    if invalid is not None:
        return _error(
            f"Unsupported safesearch '{invalid}'. Use one of {list(SUPPORTED_SAFESEARCH)}."
        )

    params: dict[str, Any] = {
        "q": query,
        "count": _clamp_count(count),
        "safesearch": safesearch_value,
    }
    if country:
        params["country"] = country
    if search_lang:
        params["search_lang"] = search_lang

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    timeout = float(getattr(settings, "brave_image_search_timeout_seconds", 2.5) or 2.5)

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(BRAVE_IMAGE_SEARCH_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        return _error(f"Brave image search timed out after {timeout}s.", retryable=True)
    except Exception as exc:
        return _error(f"Brave image search failed: {exc}")

    return json.dumps(_normalize_results(query, data))


mcp.tool()(brave_image_search)


if __name__ == "__main__":
    mcp.run(transport="stdio")
