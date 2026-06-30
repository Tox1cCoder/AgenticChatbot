# Tavily Retrieval Tools Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the built-in Tavily MCP server from search-only into production-ready Search, Extract, Map, and Crawl tools while keeping prompt/tool-schema size small through deferred discovery and conservative pins.

**Architecture:** Keep the existing in-process `tavily` MCP server and add focused tools with compact schemas: `tavily_search`, `tavily_extract`, `tavily_map`, and `tavily_crawl`. Only `tavily_search` remains pinned for the search agent; Extract/Map/Crawl are discoverable through `tool_search`, with tool-search scoring upgraded so the model can choose the right retrieval primitive dynamically.

**Tech Stack:** Python, FastMCP, Tavily Python SDK (`tavily-python`), LangChain MCP adapters, existing deferred tool binding, existing HITL policy, pytest, Ruff.

---

## Spec

### User Stories

- As a user asking for current information, I want the assistant to search the web with bounded result size and cite returned URLs.
- As a user providing a URL or asking for details from a specific page, I want the assistant to use webpage extraction instead of repeating broad search.
- As a user asking what pages exist on a site, I want the assistant to map URLs without fetching full page bodies.
- As a user asking to research a bounded docs/site section, I want the assistant to crawl with explicit depth/limit guardrails.
- As an operator, I want Tavily defaults, caps, timeout behavior, and quota-sensitive options to be configurable without hardcoding expensive behavior.

### Non-Goals

- Do not implement Tavily Research API or research tasks.
- Do not replace the repo's MCP manager with Tavily's hosted MCP server.
- Do not pin all Tavily tools into every prompt.
- Do not add a separate persistent crawl ingestion pipeline or write crawled pages to RAG storage.
- Do not remove Brave image search; Tavily search can still return image candidates, but Brave remains the dedicated image-reference tool.

### Current Code Audit

- `app/ai/mcp_servers/tavily_server.py` exposes only `tavily_search`.
- `tavily_search` hardcodes `search_depth="advanced"`, `include_images=True`, and `include_image_descriptions=True`, which increases latency/cost for every Tavily search regardless of need.
- `tavily_search` imports `TavilyClient` at module import time, which makes tests fail in environments where the active Python does not have `tavily` installed even when the project environment does.
- The current wrapper has no per-operation defaults/caps in `settings`; only the generic `TOOL_EXECUTION_TIMEOUT` protects execution.
- `app/ai/deferred_tool_binding.py` correctly pins `time::get_current_time` and `tavily::tavily_search` only for `search`.
- `tool_search` already prevents schema bloat through inventory mode, compact descriptions, confidence-gated autoload, TTL, and loaded-tool caps.
- `app/ai/tool_search_profiles.py` is tuned for shell/file/config capability intent, not web retrieval primitives. New Tavily tools will work lexically, but reliable dynamic selection needs lightweight `web_search`, `web_extract`, `web_map`, and `web_crawl` signals.
- `SEARCH_SYSTEM_PROMPT` currently names `tavily_search` in several places. That should become generic web retrieval guidance so Extract/Map/Crawl can be chosen by tool descriptions and discovery rather than prompt hardcoding.

### Tavily API Sources

- Tavily Search: `https://docs.tavily.com/documentation/api-reference/endpoint/search.md`
- Tavily Extract: `https://docs.tavily.com/documentation/api-reference/endpoint/extract.md`
- Tavily Crawl: `https://docs.tavily.com/documentation/api-reference/endpoint/crawl.md`
- Tavily Map: `https://docs.tavily.com/documentation/api-reference/endpoint/map.md`

Key current API facts from the docs:

- Search supports `search_depth` values `basic`, `fast`, `ultra-fast`, and `advanced`; `advanced` costs 2 credits while basic/fast/ultra-fast cost 1 credit.
- Search supports `auto_parameters`, but Tavily may select advanced depth, so the wrapper should default it off unless explicitly configured.
- Search `max_results` has an API maximum of 20.
- Extract supports `urls`, `query`, `chunks_per_source`, `extract_depth`, `include_images`, `format`, `timeout`, and `include_usage`; API maximum is 20 URLs per request.
- Map and Crawl both support `url`, `instructions`, `max_depth`, `max_breadth`, `limit`, path/domain include/exclude regex lists, `allow_external`, `timeout`, and `include_usage`; docs show timeout range 10-150 seconds.
- Crawl additionally returns extracted content and supports extraction options; Map returns discovered URLs only.

## Target Behavior

### Tool Surface

Bound by default for `search_agent`:

```text
time::get_current_time
tavily::tavily_search
brave_image_search::brave_image_search
widgets::widget_create
widgets::widget_update
widgets::widget_get_state
```

Discoverable through `tool_search`:

```text
tavily::tavily_extract
tavily::tavily_map
tavily::tavily_crawl
```

### Tool Selection Guidance

- Use `tavily_search` for broad web discovery, current facts, current news, and source finding.
- Use `tavily_extract` when the user provides one or more URLs, when search snippets are insufficient, or when a source needs deeper page content.
- Use `tavily_map` when the user asks what pages exist on a site, where documentation/pricing/legal/support pages are, or which URLs should be inspected next.
- Use `tavily_crawl` when the user asks to inspect a bounded website or docs section and needs page content from multiple related URLs.

### Production Defaults

Recommended defaults:

```env
TAVILY_SEARCH_DEFAULT_MAX_RESULTS=5
TAVILY_SEARCH_MAX_RESULTS=10
TAVILY_SEARCH_DEFAULT_DEPTH=basic
TAVILY_SEARCH_INCLUDE_IMAGES=true
TAVILY_SEARCH_INCLUDE_IMAGE_DESCRIPTIONS=true
TAVILY_SEARCH_AUTO_PARAMETERS=false
TAVILY_EXTRACT_MAX_URLS=5
TAVILY_EXTRACT_DEFAULT_DEPTH=basic
TAVILY_EXTRACT_DEFAULT_FORMAT=markdown
TAVILY_EXTRACT_TIMEOUT_SECONDS=20
TAVILY_MAP_MAX_DEPTH=2
TAVILY_MAP_MAX_BREADTH=20
TAVILY_MAP_LIMIT=50
TAVILY_MAP_TIMEOUT_SECONDS=30
TAVILY_CRAWL_MAX_DEPTH=1
TAVILY_CRAWL_MAX_BREADTH=10
TAVILY_CRAWL_LIMIT=20
TAVILY_CRAWL_TIMEOUT_SECONDS=45
```

Rationale:

- Search no longer forces `advanced` on every call.
- Extract is cheap and common enough to be discoverable without approval.
- Map and Crawl remain bounded; operators can use existing per-user HITL policy to require approval for `tavily::tavily_crawl` or the whole `tavily` server.
- Query logging remains disabled by default through existing `MCP_TOOL_SEARCH_LOG_QUERIES=false`.

## File Map

Modify:

- `app/core/config.py` - Tavily operational defaults, caps, and validators.
- `.env.example` - documented Tavily settings.
- `app/ai/mcp_servers/tavily_server.py` - shared helpers plus Search, Extract, Map, and Crawl tools.
- `app/ai/deferred_tool_binding.py` - assert only `tavily_search` stays pinned.
- `app/ai/tool_search_profiles.py` - add web retrieval intent and tool capability profiles.
- `app/ai/tool_search_scoring.py` - add web capability-specific ranking adjustments.
- `app/ai/prompts.py` - generic web retrieval wording; remove Tavily-search-only phrasing where it blocks Extract/Map/Crawl.
- `README.md` - update bundled MCP server purpose, Tavily settings, and recommended HITL guidance.

Create:

- `tests/test_tavily_server.py` - focused wrapper/config/normalization tests.

Modify tests:

- `tests/test_search_agent_time_context.py`
- `tests/test_tool_search_scoring.py`
- `tests/test_unified_tool_search.py`
- `tests/test_mcp_global_allowlist.py`

---

## Task 1: Add Tavily Configuration Defaults And Validation

**Files:**
- Modify: `app/core/config.py`
- Modify: `.env.example`
- Test: `tests/test_tavily_server.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/test_tavily_server.py` with the imports and first tests:

```python
from __future__ import annotations

import json

from app.ai.mcp_servers import tavily_server


def test_tavily_clamp_count_uses_default_and_cap(monkeypatch):
    monkeypatch.setattr(tavily_server.settings, "tavily_search_default_max_results", 5, raising=False)
    monkeypatch.setattr(tavily_server.settings, "tavily_search_max_results", 10, raising=False)

    assert tavily_server._clamp_int(None, default=5, minimum=1, maximum=10) == 5
    assert tavily_server._clamp_int(25, default=5, minimum=1, maximum=10) == 10
    assert tavily_server._clamp_int("bad", default=5, minimum=1, maximum=10) == 5


def test_tavily_error_payload_is_compact_json():
    payload = json.loads(tavily_server._error("missing key", operation="search", retryable=False))

    assert payload == {
        "error": "missing key",
        "provider": "tavily",
        "operation": "search",
        "retryable": False,
    }
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_tavily_server.py -q
```

Expected: fails because `_clamp_int` and `_error` are not implemented.

- [ ] **Step 3: Add config fields**

In `app/core/config.py`, add these fields near the existing `tavily_api_key` and Brave image settings:

```python
    tavily_search_default_max_results: int = Field(
        default=5,
        description="Default Tavily search result count.",
    )
    tavily_search_max_results: int = Field(
        default=10,
        description="Hard cap on Tavily search results returned per call.",
    )
    tavily_search_default_depth: str = Field(
        default="basic",
        description="Default Tavily search depth: basic, fast, ultra-fast, or advanced.",
    )
    tavily_search_include_images: bool = Field(
        default=True,
        description="Include Tavily search image candidates by default.",
    )
    tavily_search_include_image_descriptions: bool = Field(
        default=True,
        description="Include Tavily image descriptions when search images are enabled.",
    )
    tavily_search_auto_parameters: bool = Field(
        default=False,
        description="Allow Tavily to auto-select search parameters. May increase credit use.",
    )
    tavily_extract_max_urls: int = Field(
        default=5,
        description="Hard cap on URLs accepted by Tavily extract per call.",
    )
    tavily_extract_default_depth: str = Field(
        default="basic",
        description="Default Tavily extract depth: basic or advanced.",
    )
    tavily_extract_default_format: str = Field(
        default="markdown",
        description="Default Tavily extract format: markdown or text.",
    )
    tavily_extract_timeout_seconds: float = Field(
        default=20.0,
        description="Timeout sent to Tavily Extract, in seconds.",
    )
    tavily_map_max_depth: int = Field(default=2, description="Maximum Tavily map depth.")
    tavily_map_max_breadth: int = Field(default=20, description="Maximum Tavily map breadth.")
    tavily_map_limit: int = Field(default=50, description="Maximum Tavily map URL count.")
    tavily_map_timeout_seconds: float = Field(default=30.0, description="Tavily Map timeout.")
    tavily_crawl_max_depth: int = Field(default=1, description="Maximum Tavily crawl depth.")
    tavily_crawl_max_breadth: int = Field(default=10, description="Maximum Tavily crawl breadth.")
    tavily_crawl_limit: int = Field(default=20, description="Maximum Tavily crawl page count.")
    tavily_crawl_timeout_seconds: float = Field(default=45.0, description="Tavily Crawl timeout.")
```

Add validators beside the existing numeric validators:

```python
    @field_validator(
        "tavily_search_default_max_results",
        "tavily_search_max_results",
        "tavily_extract_max_urls",
        "tavily_map_max_depth",
        "tavily_map_max_breadth",
        "tavily_map_limit",
        "tavily_crawl_max_depth",
        "tavily_crawl_max_breadth",
        "tavily_crawl_limit",
        mode="before",
    )
    @classmethod
    def validate_positive_tavily_int(cls, v):
        if v in (None, ""):
            return v
        parsed = int(v)
        if parsed < 1:
            raise ValueError("Tavily numeric settings must be positive")
        return parsed

    @field_validator("tavily_search_default_depth", mode="before")
    @classmethod
    def validate_tavily_search_depth(cls, v):
        value = str(v or "basic").strip().lower()
        if value not in {"basic", "fast", "ultra-fast", "advanced"}:
            raise ValueError("TAVILY_SEARCH_DEFAULT_DEPTH must be basic, fast, ultra-fast, or advanced")
        return value

    @field_validator("tavily_extract_default_depth", mode="before")
    @classmethod
    def validate_tavily_extract_depth(cls, v):
        value = str(v or "basic").strip().lower()
        if value not in {"basic", "advanced"}:
            raise ValueError("TAVILY_EXTRACT_DEFAULT_DEPTH must be basic or advanced")
        return value

    @field_validator("tavily_extract_default_format", mode="before")
    @classmethod
    def validate_tavily_extract_format(cls, v):
        value = str(v or "markdown").strip().lower()
        if value not in {"markdown", "text"}:
            raise ValueError("TAVILY_EXTRACT_DEFAULT_FORMAT must be markdown or text")
        return value
```

- [ ] **Step 4: Add env documentation**

In `.env.example`, below `TAVILY_API_KEY=`, add:

```env
TAVILY_SEARCH_DEFAULT_MAX_RESULTS=5
TAVILY_SEARCH_MAX_RESULTS=10
TAVILY_SEARCH_DEFAULT_DEPTH=basic
TAVILY_SEARCH_INCLUDE_IMAGES=true
TAVILY_SEARCH_INCLUDE_IMAGE_DESCRIPTIONS=true
TAVILY_SEARCH_AUTO_PARAMETERS=false
TAVILY_EXTRACT_MAX_URLS=5
TAVILY_EXTRACT_DEFAULT_DEPTH=basic
TAVILY_EXTRACT_DEFAULT_FORMAT=markdown
TAVILY_EXTRACT_TIMEOUT_SECONDS=20
TAVILY_MAP_MAX_DEPTH=2
TAVILY_MAP_MAX_BREADTH=20
TAVILY_MAP_LIMIT=50
TAVILY_MAP_TIMEOUT_SECONDS=30
TAVILY_CRAWL_MAX_DEPTH=1
TAVILY_CRAWL_MAX_BREADTH=10
TAVILY_CRAWL_LIMIT=20
TAVILY_CRAWL_TIMEOUT_SECONDS=45
```

- [ ] **Step 5: Run config tests**

Run:

```powershell
python -m pytest tests/test_tavily_server.py -q
python -m ruff check app/core/config.py
```

Expected: tests pass; Ruff passes.

- [ ] **Step 6: Commit**

```powershell
git add app/core/config.py .env.example tests/test_tavily_server.py
git commit -m "feat: add tavily retrieval configuration caps"
```

---

## Task 2: Refactor Tavily Server Helpers And Search Defaults

**Files:**
- Modify: `app/ai/mcp_servers/tavily_server.py`
- Test: `tests/test_tavily_server.py`
- Test: `tests/test_rich_response_sources.py`

- [ ] **Step 1: Add failing search normalization tests**

Append to `tests/test_tavily_server.py`:

```python
class _FakeTavilyClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_search_uses_configured_basic_depth_and_preserves_images(monkeypatch):
    client = _FakeTavilyClient(
        {
            "query": "openai news",
            "answer": "",
            "images": [{"url": "https://example.com/a.jpg", "description": "A"}],
            "results": [
                {
                    "title": "Source",
                    "url": "https://example.com",
                    "content": "Snippet",
                    "score": 0.9,
                    "raw_content": "Full text",
                }
            ],
            "usage": {"credits": 1},
            "request_id": "req-1",
        }
    )
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)
    monkeypatch.setattr(tavily_server.settings, "tavily_search_default_depth", "basic", raising=False)
    monkeypatch.setattr(tavily_server.settings, "tavily_search_auto_parameters", False, raising=False)

    payload = json.loads(tavily_server.tavily_search("openai news", max_results=25))

    assert client.calls[0]["search_depth"] == "basic"
    assert client.calls[0]["max_results"] == 10
    assert payload["provider"] == "tavily"
    assert payload["operation"] == "search"
    assert payload["images"][0]["url"] == "https://example.com/a.jpg"
    assert payload["results"][0]["raw_content"] == "Full text"
    assert payload["usage"] == {"credits": 1}
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_tavily_server.py::test_search_uses_configured_basic_depth_and_preserves_images -q
```

Expected: fails because `_make_client` does not exist and search still hardcodes advanced depth.

- [ ] **Step 3: Implement shared helpers and lazy client import**

In `app/ai/mcp_servers/tavily_server.py`, remove the top-level `from tavily import TavilyClient` import. Add:

```python
from typing import Any

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
```

- [ ] **Step 4: Rewrite `tavily_search`**

Use this compact signature:

```python
@mcp.tool()
def tavily_search(
    query: str,
    max_results: int | None = None,
    search_depth: str | None = None,
    include_raw_content: bool = False,
) -> str:
```

Build params with settings:

```python
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
```

Add:

```python
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
```

Keep the docstring concise and explicit:

```python
"""Search the web for current facts, news, recent information, or source discovery.

Use this for broad web discovery. If the user provides a specific URL or the
search snippets are not enough, use `tavily_extract` after search discovers the
URL. For site structure use `tavily_map`; for bounded multi-page content use
`tavily_crawl`.
"""
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_tavily_server.py tests/test_rich_response_sources.py -q
python -m ruff check app/ai/mcp_servers/tavily_server.py tests/test_tavily_server.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add app/ai/mcp_servers/tavily_server.py tests/test_tavily_server.py
git commit -m "feat: harden tavily search defaults"
```

---

## Task 3: Add `tavily_extract`

**Files:**
- Modify: `app/ai/mcp_servers/tavily_server.py`
- Test: `tests/test_tavily_server.py`

- [ ] **Step 1: Add failing extract tests**

Append:

```python
class _FakeExtractClient:
    def __init__(self):
        self.calls = []

    def extract(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "results": [
                {
                    "url": "https://example.com/a",
                    "raw_content": "# Title\nBody",
                    "images": ["https://example.com/a.png"],
                    "favicon": "https://example.com/favicon.ico",
                }
            ],
            "failed_results": [],
            "usage": {"credits": 1},
            "request_id": "req-extract",
        }


def test_extract_accepts_string_url_and_query_rerank(monkeypatch):
    client = _FakeExtractClient()
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)
    monkeypatch.setattr(tavily_server.settings, "tavily_extract_max_urls", 5, raising=False)

    payload = json.loads(
        tavily_server.tavily_extract("https://example.com/a", query="pricing", include_images=True)
    )

    assert client.calls[0]["urls"] == ["https://example.com/a"]
    assert client.calls[0]["query"] == "pricing"
    assert client.calls[0]["extract_depth"] == "basic"
    assert client.calls[0]["format"] == "markdown"
    assert payload["operation"] == "extract"
    assert payload["results"][0]["raw_content"] == "# Title\nBody"
    assert payload["results"][0]["images"] == ["https://example.com/a.png"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_tavily_server.py::test_extract_accepts_string_url_and_query_rerank -q
```

Expected: fails because `tavily_extract` does not exist.

- [ ] **Step 3: Add URL coercion helper**

In `tavily_server.py`:

```python
def _coerce_urls(urls: str | list[str], *, maximum: int) -> list[str]:
    if isinstance(urls, str):
        raw_urls = [urls]
    elif isinstance(urls, list):
        raw_urls = urls
    else:
        raw_urls = []
    cleaned = [str(url).strip() for url in raw_urls if str(url or "").strip()]
    return cleaned[:maximum]
```

- [ ] **Step 4: Implement `tavily_extract`**

Add:

```python
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
            default=str(getattr(settings, "tavily_extract_default_format", "markdown") or "markdown"),
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
```

Add:

```python
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
```

- [ ] **Step 5: Run extract tests**

Run:

```powershell
python -m pytest tests/test_tavily_server.py -q
python -m ruff check app/ai/mcp_servers/tavily_server.py tests/test_tavily_server.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add app/ai/mcp_servers/tavily_server.py tests/test_tavily_server.py
git commit -m "feat: add tavily extract tool"
```

---

## Task 4: Add `tavily_map` And `tavily_crawl` With Caps

**Files:**
- Modify: `app/ai/mcp_servers/tavily_server.py`
- Test: `tests/test_tavily_server.py`

- [ ] **Step 1: Add failing Map/Crawl tests**

Append:

```python
class _FakeSiteClient:
    def __init__(self):
        self.map_calls = []
        self.crawl_calls = []

    def map(self, **kwargs):
        self.map_calls.append(kwargs)
        return {"base_url": kwargs["url"], "results": ["https://docs.example.com/a"], "usage": {"credits": 1}}

    def crawl(self, **kwargs):
        self.crawl_calls.append(kwargs)
        return {
            "base_url": kwargs["url"],
            "results": [{"url": "https://docs.example.com/a", "raw_content": "A"}],
            "usage": {"credits": 1},
        }


def test_map_clamps_site_traversal(monkeypatch):
    client = _FakeSiteClient()
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)

    payload = json.loads(tavily_server.tavily_map("https://docs.example.com", max_depth=5, limit=999))

    assert client.map_calls[0]["max_depth"] == 2
    assert client.map_calls[0]["limit"] == 50
    assert payload["operation"] == "map"
    assert payload["results"] == ["https://docs.example.com/a"]


def test_crawl_clamps_site_traversal_and_disables_external_by_default(monkeypatch):
    client = _FakeSiteClient()
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)

    payload = json.loads(
        tavily_server.tavily_crawl("https://docs.example.com", instructions="Find API pages", max_depth=5, limit=999)
    )

    assert client.crawl_calls[0]["max_depth"] == 1
    assert client.crawl_calls[0]["limit"] == 20
    assert client.crawl_calls[0]["allow_external"] is False
    assert payload["operation"] == "crawl"
    assert payload["results"][0]["raw_content"] == "A"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_tavily_server.py::test_map_clamps_site_traversal tests/test_tavily_server.py::test_crawl_clamps_site_traversal_and_disables_external_by_default -q
```

Expected: fails because Map/Crawl tools do not exist.

- [ ] **Step 3: Add helper for optional string lists**

```python
def _clean_string_list(values: list[str] | None) -> list[str] | None:
    if not isinstance(values, list):
        return None
    cleaned = [str(value).strip() for value in values if str(value or "").strip()]
    return cleaned or None
```

- [ ] **Step 4: Implement `tavily_map`**

```python
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
```

- [ ] **Step 5: Implement `tavily_crawl`**

```python
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
```

Add:

```python
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
```

- [ ] **Step 6: Run Map/Crawl tests**

Run:

```powershell
python -m pytest tests/test_tavily_server.py -q
python -m ruff check app/ai/mcp_servers/tavily_server.py tests/test_tavily_server.py
```

Expected: all pass.

- [ ] **Step 7: Commit**

```powershell
git add app/ai/mcp_servers/tavily_server.py tests/test_tavily_server.py
git commit -m "feat: add tavily map and crawl tools"
```

---

## Task 5: Upgrade Tool Search Scoring For Web Retrieval Intent

**Files:**
- Modify: `app/ai/tool_search_profiles.py`
- Modify: `app/ai/tool_search_scoring.py`
- Test: `tests/test_tool_search_scoring.py`
- Test: `tests/test_unified_tool_search.py`

- [ ] **Step 1: Add failing web intent scoring tests**

Append to `tests/test_tool_search_scoring.py`:

```python
def test_web_url_query_prefers_extract_over_search():
    from app.ai.mcp_tool_catalog import ToolDescriptor
    from app.ai.tool_search_scoring import rank_tool_candidates

    tools = [
        ToolDescriptor("tavily_search", "tavily", "Search the web broadly.", ["query"], ["query"], "fp1"),
        ToolDescriptor("tavily_extract", "tavily", "Extract page content from known URLs.", ["urls"], ["urls"], "fp2"),
        ToolDescriptor("tavily_crawl", "tavily", "Crawl a bounded site section.", ["url"], ["url"], "fp3"),
    ]

    ranked = rank_tool_candidates(query="extract this URL https://example.com/pricing", candidates=tools)

    assert ranked[0].tool.tool_name == "tavily_extract"
    assert ranked[0].confidence == "high"
    assert ranked[0].autoload_eligible is True


def test_site_structure_query_prefers_map_over_crawl():
    from app.ai.mcp_tool_catalog import ToolDescriptor
    from app.ai.tool_search_scoring import rank_tool_candidates

    tools = [
        ToolDescriptor("tavily_map", "tavily", "Discover URLs on a website.", ["url"], ["url"], "fp-map"),
        ToolDescriptor("tavily_crawl", "tavily", "Crawl pages and return content.", ["url"], ["url"], "fp-crawl"),
    ]

    ranked = rank_tool_candidates(query="map the docs site and list API pages", candidates=tools)

    assert ranked[0].tool.tool_name == "tavily_map"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_tool_search_scoring.py::test_web_url_query_prefers_extract_over_search tests/test_tool_search_scoring.py::test_site_structure_query_prefers_map_over_crawl -q
```

Expected: at least one assertion fails because web retrieval capabilities are not modeled.

- [ ] **Step 3: Add web intent terms**

In `app/ai/tool_search_profiles.py`, add:

```python
_WEB_SEARCH_TERMS = {"web", "search", "current", "recent", "news", "source", "sources"}
_WEB_EXTRACT_TERMS = {"extract", "url", "urls", "page", "article", "content", "source"}
_WEB_MAP_TERMS = {"map", "sitemap", "site", "structure", "pages", "urls", "discover"}
_WEB_CRAWL_TERMS = {"crawl", "site", "website", "docs", "documentation", "section", "pages"}
```

In `infer_query_intent`, add:

```python
    if tokens & _WEB_SEARCH_TERMS and tokens & {"web", "search", "current", "recent", "news"}:
        capabilities.add("web_search")
    if tokens & _WEB_EXTRACT_TERMS and tokens & {"url", "urls", "extract", "page", "article"}:
        capabilities.add("web_extract")
    if tokens & _WEB_MAP_TERMS and tokens & {"map", "sitemap", "structure", "discover", "urls"}:
        capabilities.add("web_map")
    if tokens & _WEB_CRAWL_TERMS and tokens & {"crawl", "site", "website", "docs", "documentation"}:
        capabilities.add("web_crawl")
```

In `infer_tool_profile`, add:

```python
    if "tavily" in server_name.lower():
        if "search" in name_tokens:
            capabilities.add("web_search")
        if "extract" in name_tokens:
            capabilities.add("web_extract")
        if "map" in name_tokens:
            capabilities.add("web_map")
        if "crawl" in name_tokens:
            capabilities.add("web_crawl")
```

In `_compact_purpose`, add before the shell/file cases:

```python
    if "web_search" in capabilities:
        return "Search the web for current facts, news, and source discovery."
    if "web_extract" in capabilities:
        return "Extract content from one or more known web page URLs."
    if "web_map" in capabilities:
        return "Discover URLs and structure for a website."
    if "web_crawl" in capabilities:
        return "Crawl a bounded site section and return page content."
```

- [ ] **Step 4: Add capability adjustments**

In `app/ai/tool_search_scoring.py::_capability_specific_adjustment`, add:

```python
    if "web_search" in intent.capabilities:
        if "web_search" in profile.capabilities:
            score += 30.0
        if "web_extract" in profile.capabilities:
            score -= 8.0
        if "web_crawl" in profile.capabilities:
            score -= 15.0

    if "web_extract" in intent.capabilities:
        if "web_extract" in profile.capabilities:
            score += 35.0
        if "web_search" in profile.capabilities and {"url", "urls"} & intent.tokens:
            score -= 12.0

    if "web_map" in intent.capabilities:
        if "web_map" in profile.capabilities:
            score += 35.0
        if "web_crawl" in profile.capabilities and "crawl" not in intent.tokens:
            score -= 12.0

    if "web_crawl" in intent.capabilities:
        if "web_crawl" in profile.capabilities:
            score += 35.0
        if "web_map" in profile.capabilities and "content" in intent.tokens:
            score -= 8.0
```

- [ ] **Step 5: Run scoring suites**

Run:

```powershell
python -m pytest tests/test_tool_search_scoring.py tests/test_unified_tool_search.py -q
python scripts/evaluate_tool_search_accuracy.py
python -m ruff check app/ai/tool_search_profiles.py app/ai/tool_search_scoring.py tests/test_tool_search_scoring.py
```

Expected: tests pass; accuracy script does not regress existing shell/file cases.

- [ ] **Step 6: Commit**

```powershell
git add app/ai/tool_search_profiles.py app/ai/tool_search_scoring.py tests/test_tool_search_scoring.py
git commit -m "feat: teach tool search web retrieval intent"
```

---

## Task 6: Preserve Compact Binding And Update Search Prompt

**Files:**
- Modify: `app/ai/deferred_tool_binding.py`
- Modify: `app/ai/prompts.py`
- Test: `tests/test_search_agent_time_context.py`

- [ ] **Step 1: Add failing pin and prompt tests**

In `tests/test_search_agent_time_context.py`, add:

```python
def test_search_agent_does_not_pin_heavy_tavily_tools(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

    pinned_specs = _get_pinned_specs("search")

    assert "tavily::tavily_search" in pinned_specs
    assert "tavily::tavily_extract" not in pinned_specs
    assert "tavily::tavily_map" not in pinned_specs
    assert "tavily::tavily_crawl" not in pinned_specs


def test_search_prompt_allows_extract_map_and_crawl_via_tool_search():
    prompt = SEARCH_SYSTEM_PROMPT

    assert "specific URL" in prompt
    assert "site structure" in prompt
    assert "bounded site" in prompt
    assert "Never make `tavily_search` your first actual web-search call" not in prompt
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_search_agent_time_context.py::test_search_agent_does_not_pin_heavy_tavily_tools tests/test_search_agent_time_context.py::test_search_prompt_allows_extract_map_and_crawl_via_tool_search -q
```

Expected: prompt test fails until wording is updated.

- [ ] **Step 3: Keep pins unchanged**

Confirm this block remains unchanged in `app/ai/deferred_tool_binding.py`:

```python
_SEARCH_AGENT_PINNED_SPECS = (
    "time::get_current_time",
    "tavily::tavily_search",
)
```

Do not add Extract, Map, or Crawl to `_SEARCH_AGENT_PINNED_SPECS`.

- [ ] **Step 4: Update search prompt wording**

In `app/ai/prompts.py`, update the Tavily-specific search rule in `SEARCH_SYSTEM_PROMPT` to generic web retrieval wording:

```text
- Once an actual web-search tool is available, call `get_current_time`, then call that search tool.
- Do not make a web/news search your first actual web retrieval call in a turn; anchor time first.
- For a specific URL or source page, discover and use an extraction tool rather than doing another broad search.
- For site structure or URL discovery, discover and use a site mapping tool.
- For bounded site or documentation research across multiple pages, discover and use a crawl tool with narrow depth and limit.
```

In `SEARCH_WITH_RESULTS_SYSTEM_PROMPT`, replace:

```text
7. Never make `tavily_search` your first actual web-search call in a turn
```

with:

```text
7. Never make a web/news search your first actual web retrieval call in a turn; anchor time first.
```

- [ ] **Step 5: Run prompt and binding tests**

Run:

```powershell
python -m pytest tests/test_search_agent_time_context.py -q
python -m ruff check app/ai/deferred_tool_binding.py app/ai/prompts.py tests/test_search_agent_time_context.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add app/ai/prompts.py app/ai/deferred_tool_binding.py tests/test_search_agent_time_context.py
git commit -m "docs: guide search agent toward tavily retrieval tools"
```

---

## Task 7: Update MCP Inventory And Documentation

**Files:**
- Modify: `README.md`
- Test: `tests/test_mcp_global_allowlist.py`
- Test: `tests/test_unified_tool_search.py`

- [ ] **Step 1: Add inventory regression tests**

In `tests/test_mcp_global_allowlist.py`, add:

```python
def test_tavily_remains_one_global_server_with_multiple_tools():
    servers = _load_config()["mcp_servers"]

    assert servers["tavily"]["enabled"] is True
    assert "tavily_server.py" in " ".join(servers["tavily"]["args"])
```

In `tests/test_unified_tool_search.py`, add a fake inventory assertion near existing inventory tests:

```python
def test_tavily_inventory_can_report_multiple_retrieval_tools():
    catalog = McpToolCatalog(mcp_manager=object())
    catalog._tools_by_server = {
        "tavily": [
            ToolDescriptor("tavily_search", "tavily", "Search web.", ["query"], ["query"], "fp1"),
            ToolDescriptor("tavily_extract", "tavily", "Extract URL content.", ["urls"], ["urls"], "fp2"),
            ToolDescriptor("tavily_map", "tavily", "Map site URLs.", ["url"], ["url"], "fp3"),
            ToolDescriptor("tavily_crawl", "tavily", "Crawl site content.", ["url"], ["url"], "fp4"),
        ]
    }
    catalog._server_descriptions = {"tavily": "Tavily web retrieval tools"}

    assert catalog.get_server_inventory() == [
        {
            "server_name": "tavily",
            "description": "Tavily web retrieval tools",
            "tool_count": 4,
        }
    ]
```

- [ ] **Step 2: Run tests**

Run:

```powershell
python -m pytest tests/test_mcp_global_allowlist.py tests/test_unified_tool_search.py -q
```

Expected: pass. The test module should already import `McpToolCatalog` and `ToolDescriptor` from `app.ai.mcp_tool_catalog`; add those imports explicitly if this test file was simplified before the plan is executed.

- [ ] **Step 3: Update README MCP server table**

Change the Tavily row from:

```markdown
| `tavily_server.py` | Web search adapter |
```

to:

```markdown
| `tavily_server.py` | Tavily Search, Extract, Map, and Crawl web retrieval tools |
```

Update the global default paragraph to say:

```markdown
`tavily` is one global server with multiple retrieval tools. Only `tavily_search`
is pinned for the search agent; `tavily_extract`, `tavily_map`, and
`tavily_crawl` are discovered through `tool_search` when needed.
```

- [ ] **Step 4: Add Tavily settings docs**

Near the environment table, update `TAVILY_API_KEY` from `Web search` to:

```markdown
| `TAVILY_API_KEY` | - | Tavily Search, Extract, Map, and Crawl |
```

Add a short subsection:

```markdown
Tavily defaults keep broad search cheap and site-level operations bounded.
Use `TAVILY_SEARCH_DEFAULT_DEPTH=basic` unless you need advanced search by
default. Use existing HITL settings or per-user approval policy to require
approval for `tavily::tavily_crawl` in production deployments where crawl cost
or external traffic needs review.
```

- [ ] **Step 5: Run docs-related tests**

Run:

```powershell
python -m pytest tests/test_mcp_global_allowlist.py tests/test_unified_tool_search.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add README.md tests/test_mcp_global_allowlist.py tests/test_unified_tool_search.py
git commit -m "docs: document expanded tavily retrieval tools"
```

---

## Task 8: End-To-End Verification

**Files:**
- Existing tests only.

- [ ] **Step 1: Run Tavily focused suite**

```powershell
python -m pytest tests/test_tavily_server.py tests/test_search_agent_time_context.py -q
```

Expected: all pass.

- [ ] **Step 2: Run tool discovery suites**

```powershell
python -m pytest tests/test_tool_search_scoring.py tests/test_unified_tool_search.py tests/test_tool_search_prompt_guidance.py -q
python scripts/evaluate_tool_search_accuracy.py
```

Expected: all pass; accuracy output does not show regressions for existing shell/file scenarios.

- [ ] **Step 3: Run image/rich response regressions**

```powershell
python -m pytest tests/test_rich_response_sources.py tests/test_article_image_flow.py -q
```

Expected: all pass; Tavily search image extraction still feeds existing rich response candidates.

- [ ] **Step 4: Run lint on touched files**

```powershell
python -m ruff check app/ai/mcp_servers/tavily_server.py app/ai/tool_search_profiles.py app/ai/tool_search_scoring.py app/ai/deferred_tool_binding.py app/ai/prompts.py app/core/config.py tests/test_tavily_server.py tests/test_search_agent_time_context.py tests/test_tool_search_scoring.py
```

Expected: no new Ruff errors.

- [ ] **Step 5: Manual smoke with real Tavily key**

Run the app with `TAVILY_API_KEY` configured. In a conversation:

1. Ask: `What changed in the Tavily API docs this month?`
   - Expected: search agent calls `get_current_time`, then `tavily_search`.
2. Ask: `Extract this page: https://docs.tavily.com/documentation/api-reference/endpoint/extract`
   - Expected: agent discovers or uses `tavily_extract`; no broad search is required.
3. Ask: `Map the Tavily docs site and find API reference pages.`
   - Expected: agent discovers `tavily_map`.
4. Ask: `Crawl only the Tavily docs API reference section and summarize available endpoints.`
   - Expected: agent discovers `tavily_crawl` and uses bounded depth/limit values.

- [ ] **Step 6: Commit verification notes if a progress log is used**

If implementation uses this file as a progress log, append a short dated verification note under this task and commit it:

```powershell
git add tavily_upgrade.md
git commit -m "docs: record tavily upgrade verification"
```

## Rollout Notes

- Default behavior should become cheaper than today because search no longer hardcodes `advanced`.
- Existing `tavily_search` call sites continue to work with `query` and `max_results`.
- Search image output shape remains compatible with `extract_images_from_tool_result()`.
- The enabled global server set does not change; only the tool count under the existing `tavily` server changes.
- Production deployments that want review before site-level traversal should set per-user HITL policy for `tavily::tavily_crawl` or a server-level Tavily approval rule.

## Acceptance Criteria

- `tavily_search`, `tavily_extract`, `tavily_map`, and `tavily_crawl` are available from the `tavily` MCP server.
- Only `tavily_search` is pinned for the search agent.
- `tool_search(query="extract URL content")` ranks `tavily_extract` ahead of broad search.
- `tool_search(query="map site structure")` ranks `tavily_map` ahead of crawl.
- `tool_search(query="crawl docs site content")` ranks `tavily_crawl` ahead of map/search.
- Tavily search defaults are configurable and no longer force `advanced`.
- Tavily site-level operations are capped by settings.
- Existing rich image extraction from Tavily search still passes.
- README and `.env.example` document the new operational settings and approval guidance.
