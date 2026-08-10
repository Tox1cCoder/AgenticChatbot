import json
import os
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

import contextlib  # noqa: E402
from typing import Any  # noqa: E402
from urllib.parse import urlsplit, urlunsplit  # noqa: E402

import requests  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from tavily import errors as tavily_errors  # noqa: E402

from app.core.config import settings  # noqa: E402

mcp = FastMCP("Tavily")

SUPPORTED_SEARCH_DEPTHS = {"basic", "fast", "ultra-fast", "advanced"}
SUPPORTED_EXTRACT_DEPTHS = {"basic", "advanced"}
SUPPORTED_FORMATS = {"markdown", "text"}
SUPPORTED_TOPICS = {"general", "news", "finance"}
SUPPORTED_TIME_RANGES = {"day", "week", "month", "year"}


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


def _optional_choice(value: str | None, *, allowed: set[str], name: str) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip().lower()
    if candidate not in allowed:
        raise ValueError(f"Unsupported {name} {value!r}")
    return candidate


def _coerce_urls(urls: str | list[str], *, maximum: int) -> list[str]:
    if isinstance(urls, str):
        raw_urls = [urls]
    elif isinstance(urls, list):
        raw_urls = urls
    else:
        raw_urls = []
    cleaned = [str(url).strip() for url in raw_urls if str(url or "").strip()]
    return cleaned[:maximum]


def _clean_string_list(values: list[str] | None) -> list[str] | None:
    if not isinstance(values, list):
        return None
    cleaned = [str(value).strip() for value in values if str(value or "").strip()]
    return cleaned or None


#: Maximum accepted query length. Longer queries are an argument error rather
#: than a silent truncation, so the model learns to split the research.
TAVILY_QUERY_MAX_LENGTH: int = 400


def validate_tavily_query(query: str) -> str:
    """Return the stripped query or raise for an empty/overlong one."""
    cleaned = str(query or "").strip()
    if not cleaned:
        raise ValueError("query must not be empty")
    if len(cleaned) > TAVILY_QUERY_MAX_LENGTH:
        raise ValueError(
            f"query exceeds {TAVILY_QUERY_MAX_LENGTH} characters; "
            "split complex research into focused subqueries"
        )
    return cleaned


def resolve_tavily_search_params(
    *,
    search_depth: str | None,
    auto_parameters: bool | None,
    default_depth: str,
    default_auto: bool,
) -> dict[str, Any]:
    """Resolve depth and automatic parameters into a mutually consistent pair.

    Tavily ignores automatic parameters when an explicit ``search_depth`` is
    present, so the two can never both be active. A ``None`` ``search_depth`` in
    the result means the caller must omit the key from the request entirely.
    """
    effective_auto = default_auto if auto_parameters is None else bool(auto_parameters)
    explicit_depth = str(search_depth or "").strip().lower()
    if explicit_depth in SUPPORTED_SEARCH_DEPTHS:
        return {"search_depth": explicit_depth, "auto_parameters": False}
    if effective_auto:
        return {"search_depth": None, "auto_parameters": True}
    return {
        "search_depth": _choice(None, default=default_depth, allowed=SUPPORTED_SEARCH_DEPTHS),
        "auto_parameters": False,
    }


@mcp.tool()
def tavily_search(
    query: str,
    max_results: int | None = None,
    search_depth: str | None = None,
    include_raw_content: bool = False,
    auto_parameters: bool | None = None,
    topic: str | None = None,
    time_range: str | None = None,
) -> str:
    """Search the web for current facts, news, recent information, or source discovery.

    Returns ranked text results with their source URLs. This tool never requests
    a provider-generated answer or images; ``brave_image_search`` is the only
    source of web images.

    Pass ``auto_parameters=True`` to let Tavily pick the search depth when the
    query intent is genuinely ambiguous; an explicit ``search_depth`` always
    wins. If the user provides a specific URL or snippets are insufficient, use
    ``tavily_extract`` after discovery.
    """
    operation = "search"
    try:
        cleaned_query = validate_tavily_query(query)
        resolved_topic = _optional_choice(topic, allowed=SUPPORTED_TOPICS, name="topic")
        resolved_time_range = _optional_choice(
            time_range, allowed=SUPPORTED_TIME_RANGES, name="time_range"
        )
    except ValueError as exc:
        return _error(str(exc), operation=operation)

    result_count = _clamp_int(
        max_results,
        default=int(getattr(settings, "tavily_search_default_max_results", 5) or 5),
        minimum=1,
        maximum=min(int(getattr(settings, "tavily_search_max_results", 10) or 10), 20),
    )
    resolved = resolve_tavily_search_params(
        search_depth=search_depth,
        auto_parameters=auto_parameters,
        default_depth=str(getattr(settings, "tavily_search_default_depth", "basic") or "basic"),
        default_auto=bool(getattr(settings, "tavily_search_auto_parameters", False)),
    )
    params: dict[str, Any] = {
        "query": cleaned_query,
        "max_results": result_count,
        "include_answer": False,
        "include_raw_content": include_raw_content,
        "auto_parameters": resolved["auto_parameters"],
        "include_usage": True,
        "timeout": 10,
        "topic": resolved_topic or "general",
    }
    if resolved["search_depth"] is not None:
        params["search_depth"] = resolved["search_depth"]
    if resolved_time_range is not None:
        params["time_range"] = resolved_time_range
    try:
        client = _make_client()
    except Exception:
        return _error("Tavily client unavailable.", operation=operation)
    try:
        response = client.search(**params)
    except Exception as exc:
        error_code, retryable = _classify_tavily_error(exc)
        return _error(f"Tavily search {error_code}.", operation=operation, retryable=retryable)
    return _json(_normalize_search_response(query=cleaned_query, response=response))


def _classify_tavily_error(exc: Exception) -> tuple[str, bool]:
    if isinstance(exc, tavily_errors.TimeoutError):
        return "timeout", True
    if isinstance(exc, tavily_errors.UsageLimitExceededError):
        return "rate_limit", True
    if isinstance(exc, tavily_errors.BadRequestError):
        return "invalid_request", False
    if isinstance(exc, (tavily_errors.InvalidAPIKeyError, tavily_errors.MissingAPIKeyError)):
        return "authentication", False
    if isinstance(exc, tavily_errors.ForbiddenError):
        return "subscription", False
    if isinstance(exc, requests.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", 0)
        if status == 429:
            return "rate_limit", True
        return ("upstream", True) if status >= 500 else ("provider_error", False)
    return "provider_error", False


def _canonical_result_key(url: str) -> str:
    parsed = urlsplit(str(url or "").strip())
    host = str(parsed.hostname or "").lower()
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), host, path, "", ""))


def _merge_content_chunks(existing: str, duplicate: str) -> str:
    chunks: list[str] = []
    for content in (existing, duplicate):
        for chunk in str(content or "").split("[...]"):
            cleaned = chunk.strip()
            if cleaned and cleaned not in chunks:
                chunks.append(cleaned)
    return " [...] ".join(chunks)


def _normalize_search_response(*, query: str, response: Any) -> dict[str, Any]:
    response = response if isinstance(response, dict) else {}
    results: list[dict[str, Any]] = []
    results_by_key: dict[str, dict[str, Any]] = {}
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
        if "published_date" in result:
            item["published_date"] = result["published_date"]
        canonical_key = _canonical_result_key(item["url"])
        if existing := results_by_key.get(canonical_key):
            existing["content"] = _merge_content_chunks(existing["content"], item["content"])
            continue
        results_by_key[canonical_key] = item
        results.append(item)

    # Field order is contractual: a truncated preview must keep facts, so
    # results lead and diagnostics trail.
    payload = {
        "results": results,
        "total_results": len(results),
        "answer": response.get("answer", ""),
        "provider": "tavily",
        "operation": "search",
        "query": query,
    }
    for key in ("auto_parameters", "usage", "request_id", "response_time"):
        if key in response:
            payload[key] = response[key]
    return payload


@mcp.tool()
def tavily_extract(
    urls: str | list[str],
    query: str | None = None,
    include_images: bool = False,
    extract_depth: str | None = None,
    format: str | None = None,
) -> str:
    """Extract page content from one or more known URLs.

    Use this when the user provides URL(s), when search found a source but
    snippets are insufficient, or when detailed source-grounded page content is
    needed. Use `tavily_search` first when you still need to discover URLs.
    """
    operation = "extract"
    max_urls = min(int(getattr(settings, "tavily_extract_max_urls", 5) or 5), 20)
    cleaned_urls = _coerce_urls(urls, maximum=max_urls)
    if not cleaned_urls:
        return _error("At least one URL is required for extraction.", operation=operation)
    try:
        client = _make_client()
    except Exception as exc:
        return _error(str(exc), operation=operation)

    params: dict[str, Any] = {
        "urls": cleaned_urls,
        "include_images": include_images,
        "extract_depth": _choice(
            extract_depth,
            default=str(getattr(settings, "tavily_extract_default_depth", "basic") or "basic"),
            allowed=SUPPORTED_EXTRACT_DEPTHS,
        ),
        "format": _choice(
            format,
            default=str(
                getattr(settings, "tavily_extract_default_format", "markdown") or "markdown"
            ),
            allowed=SUPPORTED_FORMATS,
        ),
        "timeout": float(getattr(settings, "tavily_extract_timeout_seconds", 20.0) or 20.0),
        "include_usage": True,
    }
    if query:
        params["query"] = query
        params["chunks_per_source"] = 3
    try:
        response = client.extract(**params)
    except TimeoutError as exc:
        return _error(str(exc), operation=operation, retryable=True)
    except Exception as exc:
        return _error(f"Extract failed: {exc}", operation=operation)
    return _json(_normalize_extract_response(urls=cleaned_urls, response=response))


def _normalize_extract_response(*, urls: list[str], response: Any) -> dict[str, Any]:
    response = response if isinstance(response, dict) else {}
    payload = {
        "provider": "tavily",
        "operation": "extract",
        "urls": urls,
        "results": response.get("results") or [],
        "failed_results": response.get("failed_results") or [],
        "total_results": len(response.get("results") or []),
    }
    for key in ("usage", "request_id", "response_time"):
        if key in response:
            payload[key] = response[key]
    return payload


@mcp.tool()
def tavily_map(
    url: str,
    instructions: str | None = None,
    max_depth: int | None = None,
    limit: int | None = None,
    select_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    allow_external: bool = False,
) -> str:
    """Discover URLs on a website without extracting full page bodies.

    Use this to inspect site structure, find relevant docs/pricing/legal/support
    pages, or choose URLs before extraction. Use `tavily_crawl` only when the
    task requires content from multiple pages.
    """
    operation = "map"
    if not str(url or "").strip():
        return _error("A root URL is required for mapping.", operation=operation)
    try:
        client = _make_client()
    except Exception as exc:
        return _error(str(exc), operation=operation)

    params: dict[str, Any] = {
        "url": str(url).strip(),
        "max_depth": _clamp_int(
            max_depth,
            default=int(getattr(settings, "tavily_map_max_depth", 2) or 2),
            minimum=1,
            maximum=int(getattr(settings, "tavily_map_max_depth", 2) or 2),
        ),
        "max_breadth": int(getattr(settings, "tavily_map_max_breadth", 20) or 20),
        "limit": _clamp_int(
            limit,
            default=int(getattr(settings, "tavily_map_limit", 50) or 50),
            minimum=1,
            maximum=int(getattr(settings, "tavily_map_limit", 50) or 50),
        ),
        "allow_external": allow_external,
        "timeout": float(getattr(settings, "tavily_map_timeout_seconds", 30.0) or 30.0),
        "include_usage": True,
    }
    for key, value in {"select_paths": select_paths, "exclude_paths": exclude_paths}.items():
        cleaned = _clean_string_list(value)
        if cleaned:
            params[key] = cleaned
    if instructions:
        params["instructions"] = instructions
    try:
        response = client.map(**params)
    except TimeoutError as exc:
        return _error(str(exc), operation=operation, retryable=True)
    except Exception as exc:
        return _error(f"Map failed: {exc}", operation=operation)
    return _json(_normalize_site_response(operation=operation, response=response))


@mcp.tool()
def tavily_crawl(
    url: str,
    instructions: str | None = None,
    max_depth: int | None = None,
    limit: int | None = None,
    select_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    include_images: bool = False,
    allow_external: bool = False,
) -> str:
    """Crawl a bounded website section and return extracted page content.

    Use this for multi-page site or docs research when the user asks for a
    bounded set of pages. Prefer `tavily_map` for URL discovery and
    `tavily_extract` for one or a few known URLs.
    """
    operation = "crawl"
    if not str(url or "").strip():
        return _error("A root URL is required for crawling.", operation=operation)
    try:
        client = _make_client()
    except Exception as exc:
        return _error(str(exc), operation=operation)

    params: dict[str, Any] = {
        "url": str(url).strip(),
        "max_depth": _clamp_int(
            max_depth,
            default=int(getattr(settings, "tavily_crawl_max_depth", 1) or 1),
            minimum=1,
            maximum=int(getattr(settings, "tavily_crawl_max_depth", 1) or 1),
        ),
        "max_breadth": int(getattr(settings, "tavily_crawl_max_breadth", 10) or 10),
        "limit": _clamp_int(
            limit,
            default=int(getattr(settings, "tavily_crawl_limit", 20) or 20),
            minimum=1,
            maximum=int(getattr(settings, "tavily_crawl_limit", 20) or 20),
        ),
        "allow_external": allow_external,
        "include_images": include_images,
        "extract_depth": str(getattr(settings, "tavily_extract_default_depth", "basic") or "basic"),
        "format": str(getattr(settings, "tavily_extract_default_format", "markdown") or "markdown"),
        "timeout": float(getattr(settings, "tavily_crawl_timeout_seconds", 45.0) or 45.0),
        "include_usage": True,
    }
    for key, value in {"select_paths": select_paths, "exclude_paths": exclude_paths}.items():
        cleaned = _clean_string_list(value)
        if cleaned:
            params[key] = cleaned
    if instructions:
        params["instructions"] = instructions
        params["chunks_per_source"] = 3
    try:
        response = client.crawl(**params)
    except TimeoutError as exc:
        return _error(str(exc), operation=operation, retryable=True)
    except Exception as exc:
        return _error(f"Crawl failed: {exc}", operation=operation)
    return _json(_normalize_site_response(operation=operation, response=response))


def _normalize_site_response(*, operation: str, response: Any) -> dict[str, Any]:
    response = response if isinstance(response, dict) else {}
    results = response.get("results") or []
    payload = {
        "provider": "tavily",
        "operation": operation,
        "base_url": response.get("base_url", ""),
        "results": results,
        "total_results": len(results),
    }
    for key in ("usage", "request_id", "response_time"):
        if key in response:
            payload[key] = response[key]
    return payload


if __name__ == "__main__":
    mcp.run(transport="stdio")
