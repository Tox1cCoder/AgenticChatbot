import json
import os
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

import contextlib  # noqa: E402
from typing import Any  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

from app.core.config import settings  # noqa: E402

mcp = FastMCP("Tavily")

SUPPORTED_SEARCH_DEPTHS = {"basic", "fast", "ultra-fast", "advanced"}
SUPPORTED_EXTRACT_DEPTHS = {"basic", "advanced"}
SUPPORTED_FORMATS = {"markdown", "text"}


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _error(message: str, *, operation: str, retryable: bool = False) -> str:
    return _json(
        {
            "error": message,
            "provider": "tavily",
            "operation": operation,
            "retryable": retryable,
        }
    )


def _resolve_api_key() -> str | None:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        with contextlib.suppress(Exception):
            api_key = settings.tavily_api_key
    return api_key or None


def _make_client():
    api_key = _resolve_api_key()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not configured. Set it in environment or config.py")
    from tavily import TavilyClient

    return TavilyClient(api_key=api_key)


def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        parsed = default
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
    return max(minimum, min(parsed, maximum))


def _choice(value: str | None, *, default: str, allowed: set[str]) -> str:
    candidate = str(value or default).strip().lower()
    return candidate if candidate in allowed else default


@mcp.tool()
def tavily_search(
    query: str,
    max_results: int | None = None,
    search_depth: str | None = None,
    include_raw_content: bool = False,
) -> str:
    """Search the web for current facts, news, recent information, or source discovery.

    Use this for broad web discovery. If the user provides a specific URL or the
    search snippets are not enough, use `tavily_extract` after search discovers the
    URL. For site structure use `tavily_map`; for bounded multi-page content use
    `tavily_crawl`.
    """
    operation = "search"
    try:
        client = _make_client()
    except Exception as exc:
        return _error(str(exc), operation=operation)

    result_count = _clamp_int(
        max_results,
        default=int(getattr(settings, "tavily_search_default_max_results", 5) or 5),
        minimum=1,
        maximum=min(int(getattr(settings, "tavily_search_max_results", 10) or 10), 20),
    )
    depth = _choice(
        search_depth,
        default=str(getattr(settings, "tavily_search_default_depth", "basic") or "basic"),
        allowed=SUPPORTED_SEARCH_DEPTHS,
    )
    params: dict[str, Any] = {
        "query": query,
        "max_results": result_count,
        "search_depth": depth,
        "include_images": bool(getattr(settings, "tavily_search_include_images", True)),
        "include_image_descriptions": bool(
            getattr(settings, "tavily_search_include_image_descriptions", True)
        ),
        "include_raw_content": include_raw_content,
        "auto_parameters": bool(getattr(settings, "tavily_search_auto_parameters", False)),
        "include_usage": True,
    }
    try:
        response = client.search(**params)
    except TimeoutError as exc:
        return _error(str(exc), operation=operation, retryable=True)
    except Exception as exc:
        return _error(f"Search failed: {exc}", operation=operation)
    return _json(_normalize_search_response(query=query, response=response))


def _normalize_search_response(*, query: str, response: Any) -> dict[str, Any]:
    response = response if isinstance(response, dict) else {}
    results = []
    for idx, result in enumerate(response.get("results") or [], 1):
        if not isinstance(result, dict):
            continue
        item = {
            "index": idx,
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
            "score": result.get("score", 0),
        }
        if result.get("raw_content"):
            item["raw_content"] = result.get("raw_content")
        if result.get("favicon"):
            item["favicon"] = result.get("favicon")
        results.append(item)

    images = []
    for image in response.get("images") or []:
        if isinstance(image, dict) and image.get("url"):
            images.append({"url": image.get("url"), "description": image.get("description", "")})

    payload = {
        "provider": "tavily",
        "operation": "search",
        "query": query,
        "answer": response.get("answer", ""),
        "images": images,
        "results": results,
        "total_results": len(results),
    }
    for key in ("auto_parameters", "usage", "request_id", "response_time"):
        if key in response:
            payload[key] = response[key]
    return payload


if __name__ == "__main__":
    mcp.run(transport="stdio")
