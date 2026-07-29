# Rich Image Selection Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make inline web images appear reliably and relevantly by treating a deliberate image search as the intent to display, anchoring placement on the model's own image query instead of scraped provider text.

**Architecture:** A `brave_image_search` result collapses into one `image_group` rich item (or a single `image` item when only one candidate survives eligibility). At persistence time, a model-authored marker wins; absent a marker, the item is anchored on its image query, with a first-prose-block fallback for deliberate image searches only. Tavily stops requesting images by default, so incidental candidates — the source of today's misplacements — disappear at the source.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2 (discriminated unions), LangGraph, FastMCP tool servers, prometheus_client, Streamlit (`demo.py`), pytest with `asyncio_mode = "auto"`.

**Spec:** `docs/superpowers/specs/2026-07-29-rich-image-selection-hardening-design.md`

## Global Constraints

- Test runner is the app venv: `.venv\Scripts\python.exe -m pytest`. Do not use a bare `pytest`; three Python environments on this machine have drifted on langchain/langgraph pins and only `.venv` is v3-capable.
- No image bytes may be fetched during answer generation or persistence. Registration is metadata-only.
- No new model call, no vision reranker, no embedding service, no second placement pass.
- No image failure may fail or delay text persistence or terminal streaming. Every image path is wrapped so a raise degrades to text-only.
- Prometheus labels may never contain a query, caption, title, URL, tenant id, conversation id, or candidate id.
- Rich item ids must match `^[A-Za-z0-9_\-.:]+$` and stay within 128 characters (`RICH_ITEM_ID_MAX_LENGTH`).
- Public payload models set `extra="forbid"`. Provider-specific metadata goes in `provenance`, never in `payload`.
- Commit after every task. Feature branch only; never commit to `master`.
- Another session may be committing this checkout with a broad `git add`. Always stage explicit paths, never `git add -A`.

## Interface Summary

Names used across tasks. Each is defined by the task that creates it.

| Symbol | Defined in | Signature |
|---|---|---|
| `resolve_tavily_search_params` | Task 1 | `(*, search_depth: str \| None, auto_parameters: bool \| None, default_depth: str, default_auto: bool) -> dict[str, Any]` |
| `validate_tavily_query` | Task 1 | `(query: str) -> str` (raises `ValueError`) |
| `image_aspect_ratio_ok` | Task 4 | `(width: Any, height: Any, *, minimum: float, maximum: float) -> bool` |
| `is_junk_image_url` | Task 4 | `(url: str) -> bool` |
| `order_tavily_images` | Task 5 | `(images: list[dict]) -> list[dict]` |
| `RichItemType.image_group` | Task 7 | enum member `"image_group"` |
| `ImageGroupItem`, `ImageGroupPayload`, `ImageGroupRichItem` | Task 7 | Pydantic models |
| `ImageAnchorEntry` | Task 9 | frozen dataclass `(item_id, query, origin, anchorable)` |
| `anchor_image_items_by_query` | Task 9 | `(content, *, entries, min_score, max_images) -> tuple[str, dict[str, str]]` |
| `build_inline_image_group_html` | Task 12 | `(cells, *, alt_text, max_width_px=INLINE_IMAGE_MAX_WIDTH_PX) -> str` |
| `RichImageMetrics.record_candidate` / `record_presentation` / `record_anchor` / `record_final_selection` | Task 6 (defined), wired in Tasks 6, 10, 10, 11 | see Task 6 |
| `build_rich_item_inventory_block` | existing, gains `image_max_items: int` in Task 10 | `(items, *, max_items, max_chars, summary_chars, image_max_items) -> str` |
| `_image_anchor_entries` | Task 10 | `(metadata: dict) -> list[ImageAnchorEntry]` |

New settings, each added by the task that first reads it:

| Setting | Default | Task |
|---|---|---|
| `tavily_search_include_images` | `False` (changed from `True`) | 2 |
| `rich_image_min_aspect_ratio` | `0.2` | 4 |
| `rich_image_max_aspect_ratio` | `5.0` | 4 |
| `rich_image_group_max_items` | `3` | 8 |
| `rich_image_anchor_min_score` | `0.34` | 9 |
| `rich_query_anchored_images_enabled` | `False`, flipped to `True` in Task 16 | 9 |
| `rich_auto_place_max_images` | `3` → `2` | 10 |

---

## Phase 1 — Tavily parameter resolver

### Task 1: Deterministic search-depth resolution and query validation

Today the adapter always sends both `search_depth` and `auto_parameters`
(`app/ai/mcp_servers/tavily_server.py:129-139`). Tavily cannot pick a depth when
an explicit depth is present, so automatic parameters are inert.

**Files:**
- Modify: `app/ai/mcp_servers/tavily_server.py:91-152`
- Test: `tests/test_tavily_server.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `resolve_tavily_search_params(*, search_depth, auto_parameters, default_depth, default_auto) -> dict[str, Any]` returning `{"search_depth": str | None, "auto_parameters": bool}`. A `None` depth means the caller must omit the key entirely. `validate_tavily_query(query: str) -> str` returns the stripped query or raises `ValueError`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_tavily_server.py`:

```python
import pytest

from app.ai.mcp_servers.tavily_server import (
    resolve_tavily_search_params,
    validate_tavily_query,
)


def test_explicit_depth_wins_and_disables_auto_parameters():
    resolved = resolve_tavily_search_params(
        search_depth="advanced",
        auto_parameters=True,
        default_depth="basic",
        default_auto=True,
    )
    assert resolved == {"search_depth": "advanced", "auto_parameters": False}


def test_auto_mode_omits_search_depth():
    resolved = resolve_tavily_search_params(
        search_depth=None,
        auto_parameters=True,
        default_depth="basic",
        default_auto=False,
    )
    assert resolved == {"search_depth": None, "auto_parameters": True}


def test_no_depth_and_auto_disabled_sends_default_depth():
    resolved = resolve_tavily_search_params(
        search_depth=None,
        auto_parameters=None,
        default_depth="basic",
        default_auto=False,
    )
    assert resolved == {"search_depth": "basic", "auto_parameters": False}


def test_omitted_auto_parameters_falls_back_to_configured_default():
    resolved = resolve_tavily_search_params(
        search_depth=None,
        auto_parameters=None,
        default_depth="basic",
        default_auto=True,
    )
    assert resolved == {"search_depth": None, "auto_parameters": True}


def test_unsupported_depth_falls_back_to_default_depth():
    resolved = resolve_tavily_search_params(
        search_depth="turbo",
        auto_parameters=None,
        default_depth="basic",
        default_auto=False,
    )
    assert resolved == {"search_depth": "basic", "auto_parameters": False}


def test_validate_query_strips_and_returns():
    assert validate_tavily_query("  red panda habitat  ") == "red panda habitat"


@pytest.mark.parametrize("bad", ["", "   ", "x" * 401])
def test_validate_query_rejects_empty_and_overlong(bad):
    with pytest.raises(ValueError):
        validate_tavily_query(bad)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tavily_server.py -k "resolve or validate_query" -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_tavily_search_params'`

- [ ] **Step 3: Implement the resolver and validator**

In `app/ai/mcp_servers/tavily_server.py`, add above `tavily_search`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tavily_server.py -k "resolve or validate_query" -v`
Expected: PASS

- [ ] **Step 5: Write the failing integration test for the tool signature**

```python
def test_tavily_search_omits_depth_in_auto_mode(monkeypatch):
    captured = {}

    class _FakeClient:
        def search(self, **params):
            captured.update(params)
            return {"results": [], "images": []}

    monkeypatch.setattr(tavily_server, "_make_client", lambda: _FakeClient())
    tavily_server.tavily_search.fn(query="red panda", auto_parameters=True)
    assert "search_depth" not in captured
    assert captured["auto_parameters"] is True


def test_tavily_search_rejects_empty_query(monkeypatch):
    monkeypatch.setattr(tavily_server, "_make_client", lambda: object())
    result = json.loads(tavily_server.tavily_search.fn(query="   "))
    assert result["error"]
    assert result.get("retryable") is not True
```

Match the existing accessor style in `tests/test_tavily_server.py` — if that file
calls the tool object directly rather than through `.fn`, use its convention.

- [ ] **Step 6: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tavily_server.py -k "omits_depth or rejects_empty" -v`
Expected: FAIL — `tavily_search` has no `auto_parameters` parameter

- [ ] **Step 7: Wire the resolver into `tavily_search`**

Replace the parameter-building block at `app/ai/mcp_servers/tavily_server.py:107-139`:

```python
    operation = "search"
    try:
        cleaned_query = validate_tavily_query(query)
    except ValueError as exc:
        return _error(str(exc), operation=operation)
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
    resolved = resolve_tavily_search_params(
        search_depth=search_depth,
        auto_parameters=auto_parameters,
        default_depth=str(getattr(settings, "tavily_search_default_depth", "basic") or "basic"),
        default_auto=bool(getattr(settings, "tavily_search_auto_parameters", False)),
    )
    images_enabled = (
        bool(getattr(settings, "tavily_search_include_images", False))
        if include_images is None
        else bool(include_images)
    )
    params: dict[str, Any] = {
        "query": cleaned_query,
        "max_results": result_count,
        "include_images": images_enabled,
        "include_image_descriptions": images_enabled
        and bool(getattr(settings, "tavily_search_include_image_descriptions", True)),
        "include_raw_content": include_raw_content,
        "auto_parameters": resolved["auto_parameters"],
        "include_usage": True,
    }
    if resolved["search_depth"] is not None:
        params["search_depth"] = resolved["search_depth"]
```

Add `auto_parameters: bool | None = None` to the signature after
`include_images`, and use `cleaned_query` in the `_normalize_search_response`
call.

- [ ] **Step 8: Run the full Tavily suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tavily_server.py -v`
Expected: PASS. Existing tests that assert `search_depth` is always present must
be updated to the new contract — an explicit depth still sends the key.

- [ ] **Step 9: Commit**

```bash
git add app/ai/mcp_servers/tavily_server.py tests/test_tavily_server.py
git commit -m "fix: make Tavily automatic search parameters actually effective"
```

---

## Phase 2 — Retrieval intent

### Task 2: Stop requesting images by default and teach query formulation

This is the cause-fix. It ships before any placement change so the effect of
removing incidental candidates is observable on its own.

**Files:**
- Modify: `app/core/config.py:326-329`
- Modify: `app/ai/mcp_servers/tavily_server.py:99-106` (docstring)
- Modify: `app/ai/prompts.py:18-23` (`MEDIA_CAPABILITY_SNIPPET`)
- Test: `tests/test_tavily_server.py`, `tests/test_rich_response_prompt_inventory.py`

**Interfaces:**
- Consumes: `resolve_tavily_search_params` (Task 1).
- Produces: no new symbols. `settings.tavily_search_include_images` now defaults to `False`.

- [ ] **Step 1: Write the failing tests**

```python
def test_default_search_does_not_request_images(monkeypatch):
    captured = {}

    class _FakeClient:
        def search(self, **params):
            captured.update(params)
            return {"results": [], "images": []}

    monkeypatch.setattr(tavily_server, "_make_client", lambda: _FakeClient())
    tavily_server.tavily_search.fn(query="chip export rules 2026")
    assert captured["include_images"] is False
    assert captured["include_image_descriptions"] is False


def test_explicit_include_images_still_requests_them(monkeypatch):
    captured = {}

    class _FakeClient:
        def search(self, **params):
            captured.update(params)
            return {"results": [], "images": []}

    monkeypatch.setattr(tavily_server, "_make_client", lambda: _FakeClient())
    tavily_server.tavily_search.fn(query="apple park", include_images=True)
    assert captured["include_images"] is True
```

And in `tests/test_rich_response_prompt_inventory.py`:

```python
def test_media_guidance_requires_disambiguated_image_query():
    from app.ai.prompts import MEDIA_CAPABILITY_SNIPPET

    text = MEDIA_CAPABILITY_SNIPPET.lower()
    assert "brave_image_search" in text
    assert "disambiguat" in text
    assert "same tool block" in text or "parallel" in text
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tavily_server.py -k "does_not_request_images or explicit_include_images" tests/test_rich_response_prompt_inventory.py -k "disambiguated or does_not_request" -v`
Expected: FAIL — default is currently `True`, and the snippet has no query guidance

- [ ] **Step 3: Flip the setting default**

`app/core/config.py:326`:

```python
    tavily_search_include_images: bool = Field(
        default=False,
        description=(
            "Include Tavily search image candidates by default. Off by default: "
            "ordinary text research must not manufacture image candidates. "
            "Callers pass include_images=True for source-bound visuals."
        ),
    )
```

- [ ] **Step 4: Update the tool docstring**

Replace the `tavily_search` docstring body:

```python
    """Search the web for current facts, news, recent information, or source discovery.

    Images are OFF by default. Pass ``include_images=True`` only when an image
    tied to a cited source result would materially help the answer. For focused
    visual discovery ("what does X look like", galleries, examples), call
    ``brave_image_search`` instead — it is a ranked image index and gives far
    better visuals than page-scraped images.

    Pass ``auto_parameters=True`` to let Tavily pick the search depth when the
    query intent is genuinely ambiguous; an explicit ``search_depth`` always
    wins. If the user provides a specific URL or snippets are insufficient, use
    ``tavily_extract`` after discovery.
    """
```

- [ ] **Step 5: Rewrite the media guidance snippet**

`app/ai/prompts.py:18-23`:

```python
MEDIA_CAPABILITY_SNIPPET = """

Media and visuals:
- You can display provided rich items inline with `<!--rich:<id>-->`; use only available IDs and never invent image URLs.
- For focused visual discovery ("what does X look like", examples, galleries), call `brave_image_search`. For ordinary research call `tavily_search` without images; pass `include_images=True` only when an image tied to a cited source helps.
- Write the image query yourself: a concrete subject plus any disambiguator the conversation implies (company vs fruit, language vs island, city vs person), plus a form qualifier when it matters (`photo`, `diagram`, `map`, `chart`, `screenshot`). Never reuse the user's question verbatim and never include question words. One subject per call.
- When an answer needs both research and a visual, issue the search and the image search in the same tool block so they run in parallel. Never search, answer partially, then search again for images.
- Place at most two image items per answer, near the text they support, and keep the prose useful without them.
- Do not add media for decoration. Use images/widgets only when they clarify, compare, document, or illustrate the answer."""
```

- [ ] **Step 6: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tavily_server.py tests/test_rich_response_prompt_inventory.py -v`
Expected: PASS

- [ ] **Step 7: Update `.env.example` and run the broader suite**

Set `TAVILY_SEARCH_INCLUDE_IMAGES=false` in `.env.example` with a one-line
comment that Brave is the visual path.

Run: `.venv\Scripts\python.exe -m pytest tests/test_tavily_server.py tests/test_rich_response_sources.py tests/test_article_image_flow.py -v`
Expected: PASS. Any test asserting Tavily images by default must be updated to
pass `include_images=True` explicitly.

- [ ] **Step 8: Commit**

```bash
git add app/core/config.py app/ai/mcp_servers/tavily_server.py app/ai/prompts.py .env.example tests/test_tavily_server.py tests/test_rich_response_prompt_inventory.py
git commit -m "feat: stop requesting Tavily images by default and require disambiguated image queries"
```

---

## Phase 3 — Candidate metadata, gates, ordering, metrics

### Task 3: Preserve source title, image query, and a truly absent score

`source_title` is emitted by the adapter (`tavily_server.py:194`) but dropped by
the candidate builder's copy list (`tool_execution.py:213-220`). The adapter also
defaults a missing score to `0`, which would make the "missing sorts last" rule
in Task 5 vacuous.

**Files:**
- Modify: `app/ai/mcp_servers/tavily_server.py:187-201`
- Modify: `app/ai/tool_execution.py:205-225`
- Test: `tests/test_rich_response_sources.py`

**Interfaces:**
- Consumes: nothing.
- Produces: candidate `provenance` now carries `source_title` and `query`; Tavily images carry `result_score: float | None`.

- [ ] **Step 1: Write the failing test**

```python
def test_candidate_provenance_keeps_source_title_and_query():
    payload = json.dumps(
        {
            "provider": "tavily",
            "query": "apple park cupertino aerial",
            "images": [
                {
                    "url": "https://cdn.example.com/park.jpg",
                    "description": "Aerial view of Apple Park",
                    "source_url": "https://example.com/apple-park",
                    "source_title": "Inside Apple Park",
                    "source_domain": "example.com",
                    "result_rank": 0,
                    "result_score": 0.91,
                    "width": 1200,
                    "height": 800,
                }
            ],
        }
    )
    candidates = build_image_candidates_from_tool_result(
        payload, tool_call_id="call_1", tool_name="tavily_search"
    )
    provenance = candidates[0]["provenance"]
    assert provenance["source_title"] == "Inside Apple Park"
    assert provenance["query"] == "apple park cupertino aerial"


def test_missing_tavily_score_is_none_not_zero():
    normalized = tavily_server._normalize_search_response(
        query="q",
        response={"results": [{"url": "https://e.com/a", "title": "A", "images": ["https://e.com/i.jpg"]}]},
        include_images=True,
    )
    assert normalized["images"][0]["result_score"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_response_sources.py -k "source_title or missing_tavily_score" -v`
Expected: FAIL — `KeyError: 'source_title'` and score is `0`

- [ ] **Step 3: Emit an absent score as `None`**

`app/ai/mcp_servers/tavily_server.py`, inside the result-bound image loop:

```python
                raw_score = result.get("score")
                image.update(
                    {
                        "source_url": source_url,
                        "source_title": source_title,
                        "source_domain": source_domain,
                        "result_rank": result_rank,
                        "result_score": (
                            float(raw_score) if isinstance(raw_score, (int, float)) else None
                        ),
                    }
                )
```

- [ ] **Step 4: Carry both fields through the candidate builder**

`app/ai/tool_execution.py`, extend the copy list and add the query:

```python
        for meta_key in (
            "thumbnail_url",
            "source_domain",
            "source_title",
            "provider",
            "result_rank",
            "result_score",
            "query_level",
        ):
            meta_value = image.get(meta_key)
            if meta_value is not None:
                provenance[meta_key] = meta_value
        result_query = str(parsed.get("query") or "").strip()
        if result_query:
            provenance["query"] = result_query
```

- [ ] **Step 5: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_response_sources.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/ai/mcp_servers/tavily_server.py app/ai/tool_execution.py tests/test_rich_response_sources.py
git commit -m "feat: preserve source title and image query through the candidate handoff"
```

### Task 4: Aspect-ratio and junk-URL eligibility gates

Both are metadata-only. Neither fetches or decodes an image, so neither adds
request-path latency. Note the coverage asymmetry: Tavily supplies no
dimensions (`tavily_server.py:226-237`), so the aspect gate only ever applies to
Brave candidates. The junk-URL gate applies to both.

**Files:**
- Modify: `app/core/config.py` (after `rich_image_min_height_px`)
- Modify: `app/ai/tool_execution.py:130-182`
- Test: `tests/test_rich_response_sources.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `image_aspect_ratio_ok(width, height, *, minimum, maximum) -> bool` and `is_junk_image_url(url: str) -> bool`, both in `app/ai/tool_execution.py`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from app.ai.tool_execution import image_aspect_ratio_ok, is_junk_image_url


@pytest.mark.parametrize(
    "width,height,expected",
    [
        (2000, 200, False),   # wide hero strip, ratio 10.0
        (300, 1500, False),   # ratio 0.2 is the boundary and is rejected below it
        (1200, 800, True),    # ordinary photo
        (800, 2600, True),    # tall infographic, ratio ~0.31
        (2200, 500, True),    # panorama, ratio 4.4
        (None, 800, True),    # unknown dimensions never reject
        (1200, None, True),
        (0, 0, True),         # nonsense dimensions are not a rejection signal
    ],
)
def test_aspect_ratio_gate(width, height, expected):
    assert image_aspect_ratio_ok(width, height, minimum=0.2, maximum=5.0) is expected


@pytest.mark.parametrize(
    "url",
    [
        "https://e.com/favicon.ico",
        "https://e.com/assets/sprite-v2.png",
        "https://e.com/img/spacer.gif",
        "https://e.com/t/1x1.png",
        "https://e.com/pixel.gif",
        "https://e.com/users/avatar/12.jpg",
    ],
)
def test_junk_urls_are_rejected(url):
    assert is_junk_image_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://e.com/new-logo-reveal.jpg",
        "https://e.com/logos/brand.png",
        "https://e.com/photos/apple-park.jpg",
    ],
)
def test_legitimate_urls_including_logos_are_accepted(url):
    assert is_junk_image_url(url) is False


def test_wide_strip_candidate_is_rejected_end_to_end():
    payload = json.dumps(
        {
            "provider": "brave_image_search",
            "query": "apple park",
            "images": [
                {"url": "https://e.com/strip.jpg", "width": 2000, "height": 200},
                {"url": "https://e.com/ok.jpg", "width": 1200, "height": 800},
            ],
        }
    )
    candidates = build_image_candidates_from_tool_result(
        payload, tool_call_id="c1", tool_name="brave_image_search"
    )
    urls = json.dumps(candidates)
    assert "strip.jpg" not in urls
    assert "ok.jpg" in urls
```

`test_wide_strip_candidate_is_rejected_end_to_end` asserts on rejection only, so
it stays valid after Task 8 changes the returned shape to a group.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_response_sources.py -k "aspect or junk or legitimate or wide_strip" -v`
Expected: FAIL with `ImportError: cannot import name 'image_aspect_ratio_ok'`

- [ ] **Step 3: Add the settings**

`app/core/config.py`, after `rich_image_min_height_px`:

```python
    rich_image_min_aspect_ratio: float = Field(
        default=0.2,
        gt=0,
        description=(
            "Reject provider images narrower than this width/height ratio when both "
            "dimensions are known. Deliberately loose: a tight photo-shaped band "
            "rejects tall infographics, screenshots, and flowcharts."
        ),
    )
    rich_image_max_aspect_ratio: float = Field(
        default=5.0,
        gt=0,
        description=(
            "Reject provider images wider than this width/height ratio when both "
            "dimensions are known. Catches hero strips the minimum-dimension "
            "gates miss; still admits panoramas and wide charts."
        ),
    )
```

- [ ] **Step 4: Implement both gates**

`app/ai/tool_execution.py`, near `_guess_mime_from_url`:

```python
#: Unambiguous non-content markers in an image path. Kept as a module constant
#: rather than a setting so no unvalidated pattern arrives through config.
#: "logo" is deliberately absent — "what does the new X logo look like" is a
#: legitimate visual query.
_JUNK_IMAGE_URL_MARKERS: tuple[str, ...] = (
    "favicon",
    "sprite",
    "spacer",
    "1x1",
    "pixel.gif",
    "avatar",
)


def image_aspect_ratio_ok(width: Any, height: Any, *, minimum: float, maximum: float) -> bool:
    """Return False only when both dimensions are known and the ratio is extreme.

    Unknown or nonsense dimensions are not a rejection signal. Tavily supplies no
    dimensions at all, so this gate applies in practice only to Brave results.
    """
    if not isinstance(width, int) or not isinstance(height, int):
        return True
    if width <= 0 or height <= 0:
        return True
    ratio = width / height
    return minimum <= ratio <= maximum


def is_junk_image_url(url: str) -> bool:
    """Return True when the URL path names a known non-content asset."""
    path = urlsplit(str(url or "")).path.lower()
    if not path:
        return False
    return any(marker in path for marker in _JUNK_IMAGE_URL_MARKERS)
```

- [ ] **Step 5: Apply them in the candidate loop**

In `build_image_candidates_from_tool_result`, inside `if display_url:` after the
duplicate check:

```python
            if is_junk_image_url(display_url):
                with suppress(Exception):
                    rich_image_metrics.record_selection(
                        provider=metric_provider,
                        outcome="rejected",
                    )
                continue
            if not image_aspect_ratio_ok(
                width,
                height,
                minimum=float(getattr(settings, "rich_image_min_aspect_ratio", 0.2)),
                maximum=float(getattr(settings, "rich_image_max_aspect_ratio", 5.0)),
            ):
                with suppress(Exception):
                    rich_image_metrics.record_selection(
                        provider=metric_provider,
                        outcome="rejected",
                    )
                continue
```

Task 6 replaces these `record_selection` calls with reason-bearing ones.

- [ ] **Step 6: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_response_sources.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add app/core/config.py app/ai/tool_execution.py tests/test_rich_response_sources.py
git commit -m "feat: add aspect-ratio and junk-URL image eligibility gates"
```

### Task 5: Deterministic Tavily candidate ordering

**Files:**
- Modify: `app/ai/tool_execution.py` (new helper, called before the candidate loop)
- Test: `tests/test_rich_response_sources.py`

**Interfaces:**
- Consumes: `result_score: float | None` from Task 3.
- Produces: `order_tavily_images(images: list[dict]) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
from app.ai.tool_execution import order_tavily_images


def test_source_bound_images_precede_query_level():
    ordered = order_tavily_images(
        [
            {"url": "q", "query_level": True},
            {"url": "s", "result_rank": 3, "result_score": 0.1},
        ]
    )
    assert [i["url"] for i in ordered] == ["s", "q"]


def test_higher_score_then_lower_rank_wins():
    ordered = order_tavily_images(
        [
            {"url": "a", "result_rank": 2, "result_score": 0.5},
            {"url": "b", "result_rank": 0, "result_score": 0.9},
            {"url": "c", "result_rank": 1, "result_score": 0.9},
        ]
    )
    assert [i["url"] for i in ordered] == ["b", "c", "a"]


def test_absent_score_sorts_after_any_numeric_score():
    ordered = order_tavily_images(
        [
            {"url": "none", "result_rank": 0, "result_score": None},
            {"url": "low", "result_rank": 9, "result_score": 0.01},
        ]
    )
    assert [i["url"] for i in ordered] == ["low", "none"]


def test_ordering_is_stable_for_equivalent_candidates():
    ordered = order_tavily_images(
        [
            {"url": "first", "result_rank": 1, "result_score": 0.5},
            {"url": "second", "result_rank": 1, "result_score": 0.5},
        ]
    )
    assert [i["url"] for i in ordered] == ["first", "second"]


def test_ordering_never_drops_a_candidate():
    images = [{"url": str(n)} for n in range(7)]
    assert len(order_tavily_images(images)) == 7
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_response_sources.py -k "order_tavily or source_bound_images_precede or absent_score" -v`
Expected: FAIL with `ImportError: cannot import name 'order_tavily_images'`

- [ ] **Step 3: Implement the ordering**

```python
def order_tavily_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order Tavily image dicts by provenance strength.

    Source-bound before query-level, then higher parent score, then lower parent
    rank, with original provider order as the stable tie-breaker. Ordering never
    rejects a candidate.
    """

    def sort_key(indexed: tuple[int, dict[str, Any]]) -> tuple[int, int, float, int, int]:
        index, image = indexed
        query_level = 1 if image.get("query_level") else 0
        raw_score = image.get("result_score")
        has_score = 0 if isinstance(raw_score, (int, float)) else 1
        score = -float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
        raw_rank = image.get("result_rank")
        rank = int(raw_rank) if isinstance(raw_rank, int) and raw_rank >= 0 else 10**6
        return (query_level, has_score, score, rank, index)

    return [image for _, image in sorted(enumerate(images), key=sort_key)]
```

- [ ] **Step 4: Call it for Tavily results only**

In `build_image_candidates_from_tool_result`, after `metric_provider` is
computed:

```python
    if metric_provider == "tavily":
        images = order_tavily_images([i for i in images if isinstance(i, dict)])
```

Brave results keep provider rank order untouched.

- [ ] **Step 5: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_response_sources.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/ai/tool_execution.py tests/test_rich_response_sources.py
git commit -m "feat: order Tavily image candidates by provenance strength"
```

### Task 6: Stage-truthful metrics

`record_selection(outcome="selected")` currently fires for every *eligible*
candidate (`tool_execution.py:241`), which hides the difference between
transport eligibility and a displayed image. New counters emit in parallel with
the old one for one release.

**Files:**
- Modify: `app/observability/rich_images.py`
- Modify: `app/ai/tool_execution.py` (replace `record_selection` call sites)
- Test: `tests/test_rich_image_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces on `RichImageMetrics`: `record_candidate(*, provider: str, outcome: str)`, `record_presentation(*, provider: str, count: int)`, `record_anchor(*, provider: str, outcome: str)`, `record_final_selection(*, provider: str, count: int)`.

- [ ] **Step 1: Write the failing test**

```python
def test_stage_counters_are_bounded_and_content_free():
    metrics = RichImageMetrics()
    metrics.record_candidate(provider="tavily", outcome="rejected_aspect_ratio")
    metrics.record_candidate(provider="brave", outcome="not-a-real-outcome")
    metrics.record_presentation(provider="brave", count=3)
    metrics.record_anchor(provider="brave", outcome="fallback_anchored")
    metrics.record_anchor(provider="brave", outcome="unplaced")
    metrics.record_final_selection(provider="brave", count=1)
    body = metrics.render().decode()

    assert 'outcome="rejected_aspect_ratio"' in body
    assert 'outcome="other"' in body
    assert 'outcome="fallback_anchored"' in body
    assert 'outcome="unplaced"' in body
    assert "rich_image_presented_total" in body
    assert "rich_image_final_selection_total" in body
    for forbidden in ("query", "caption", "http", "conversation"):
        assert f'{forbidden}="' not in body


def test_old_selection_counter_still_emits_during_compatibility_window():
    metrics = RichImageMetrics()
    metrics.record_selection(provider="tavily", outcome="selected")
    assert "rich_image_selections_total" in metrics.render().decode()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_image_metrics.py -v`
Expected: FAIL — `RichImageMetrics` has no `record_candidate`

- [ ] **Step 3: Add the counters**

`app/observability/rich_images.py`:

```python
_CANDIDATE_OUTCOMES = {
    "eligible",
    "rejected_malformed",
    "rejected_scheme",
    "rejected_duplicate",
    "rejected_dimensions",
    "rejected_aspect_ratio",
    "rejected_junk_url",
}
_ANCHOR_OUTCOMES = {"marker", "query_anchored", "fallback_anchored", "unplaced"}
```

In `__init__`:

```python
        self.candidates = Counter(
            "rich_image_candidates_total",
            "Deterministic rich-image eligibility outcomes by reason.",
            ("provider", "outcome"),
            registry=self.registry,
        )
        self.presented = Counter(
            "rich_image_presented_total",
            "Image items included in the model-facing inventory.",
            ("provider",),
            registry=self.registry,
        )
        self.anchors = Counter(
            "rich_image_anchor_outcomes_total",
            "How each image item reached (or failed to reach) the answer body.",
            ("provider", "outcome"),
            registry=self.registry,
        )
        self.final_selections = Counter(
            "rich_image_final_selection_total",
            "Image items surviving finalization and persisted with the message.",
            ("provider",),
            registry=self.registry,
        )
```

And the methods:

```python
    def record_candidate(self, *, provider: str, outcome: str) -> None:
        self.candidates.labels(
            provider=_provider(provider),
            outcome=_bounded(outcome, _CANDIDATE_OUTCOMES),
        ).inc()

    def record_presentation(self, *, provider: str, count: int) -> None:
        if count > 0:
            self.presented.labels(provider=_provider(provider)).inc(int(count))

    def record_anchor(self, *, provider: str, outcome: str) -> None:
        self.anchors.labels(
            provider=_provider(provider),
            outcome=_bounded(outcome, _ANCHOR_OUTCOMES),
        ).inc()

    def record_final_selection(self, *, provider: str, count: int) -> None:
        if count > 0:
            self.final_selections.labels(provider=_provider(provider)).inc(int(count))
```

- [ ] **Step 4: Replace the call sites with reason-bearing ones**

In `app/ai/tool_execution.py`, define a local helper and use it at each `continue`
so the reason is explicit, while keeping the deprecated counter emitting:

```python
    def _reject(reason: str) -> None:
        with suppress(Exception):
            rich_image_metrics.record_candidate(provider=metric_provider, outcome=reason)
            rich_image_metrics.record_selection(provider=metric_provider, outcome="rejected")
```

Call `_reject("rejected_malformed")`, `_reject("rejected_scheme")`,
`_reject("rejected_duplicate")`, `_reject("rejected_dimensions")`,
`_reject("rejected_junk_url")`, `_reject("rejected_aspect_ratio")` at the
matching branches, and on acceptance:

```python
        with suppress(Exception):
            rich_image_metrics.record_candidate(provider=metric_provider, outcome="eligible")
            rich_image_metrics.record_selection(provider=metric_provider, outcome="selected")
```

- [ ] **Step 5: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_image_metrics.py tests/test_rich_response_sources.py -v`
Expected: PASS

- [ ] **Step 6: Document the deprecation**

Add to `docs/frontend/rich-image-rendering.md` (or the ops doc referenced there)
a short "Deprecated metrics" note: `rich_image_selections_total` is superseded by
`rich_image_candidates_total` and `rich_image_final_selection_total`, and is
removed in Task 17. Dashboards and alerts must migrate before then.

- [ ] **Step 7: Commit**

```bash
git add app/observability/rich_images.py app/ai/tool_execution.py tests/test_rich_image_metrics.py docs/frontend/rich-image-rendering.md
git commit -m "feat: emit stage-truthful rich-image metrics alongside the deprecated counter"
```

---

## Phase 4 — Grouping and query-anchored placement

### Task 7: `image_group` rich-item schema

**Files:**
- Modify: `app/core/rich_response.py:76-83` (enum), `121-142` (payloads), `213-253` (items and union), `722+` (`__all__`)
- Test: `tests/test_rich_response_contract.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `RichItemType.image_group`, `ImageGroupItem`, `ImageGroupPayload`, `ImageGroupRichItem`, all exported.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from app.core.rich_response import validate_public_rich_item


def _group(cells):
    return {
        "id": "imagegroup:tool:call_1",
        "type": "image_group",
        "source": "image_search",
        "display_policy": "inline_only",
        "alt_text": "Photos of a red panda",
        "payload": {"items": cells},
        "provenance": {"provider": "brave_image_search", "query": "red panda photo"},
    }


def _cell(url="https://e.com/a.jpg"):
    return {"url": url, "mime_type": "image/jpeg"}


def test_image_group_validates_with_two_cells():
    item = validate_public_rich_item(_group([_cell(), _cell("https://e.com/b.jpg")]))
    assert item.type.value == "image_group"
    assert len(item.payload.items) == 2


def test_image_group_rejects_fewer_than_two_cells():
    with pytest.raises(Exception):
        validate_public_rich_item(_group([_cell()]))


def test_image_group_rejects_unsupported_mime():
    with pytest.raises(Exception):
        validate_public_rich_item(
            _group([{"url": "https://e.com/a.svg", "mime_type": "image/svg+xml"}, _cell()])
        )


def test_image_group_accepts_protected_relative_cell_url():
    item = validate_public_rich_item(
        _group([{"url": "/web-images/abc", "mime_type": "image/jpeg"}, _cell()])
    )
    assert item.payload.items[0].url == "/web-images/abc"


def test_image_group_rejects_unknown_payload_field():
    bad = _group([_cell(), _cell("https://e.com/b.jpg")])
    bad["payload"]["carousel"] = True
    with pytest.raises(Exception):
        validate_public_rich_item(bad)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_response_contract.py -k image_group -v`
Expected: FAIL — `image_group` is not a valid discriminator value

- [ ] **Step 3: Add the enum member**

```python
class RichItemType(str, Enum):
    image = "image"
    image_group = "image_group"
    live_widget = "live_widget"
    tool_render = "tool_render"
    canvas_artifact = "canvas_artifact"
    citation = "citation"
    resource_link = "resource_link"
```

- [ ] **Step 4: Add the payload models**

After `ImagePayload`:

```python
class ImageGroupItem(PublicPayload):
    """One cell of an image group. Cells are always remote or protected URLs;
    inline base64 cells are not supported because a group is only ever built
    from provider search results."""

    url: str
    mime_type: str
    source_url: str | None = None
    description: str | None = None
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_cell(self) -> ImageGroupItem:
        if not self.url:
            raise ValueError("image group cell requires a url")
        if self.mime_type not in ALLOWED_IMAGE_MIME_TYPES:
            raise ValueError(
                f"unsupported image mime_type {self.mime_type!r}; allowed: "
                f"{sorted(ALLOWED_IMAGE_MIME_TYPES)}"
            )
        _validate_image_url(self.url)
        _validate_url_scheme(self.source_url)
        return self


class ImageGroupPayload(PublicPayload):
    items: list[ImageGroupItem] = Field(min_length=2)
```

- [ ] **Step 5: Add the item model and extend the union**

```python
class ImageGroupRichItem(RichItemBase):
    type: Literal[RichItemType.image_group]
    display_policy: Literal[RichDisplayPolicy.inline_only] = RichDisplayPolicy.inline_only
    alt_text: str
    payload: ImageGroupPayload


RichItem = Annotated[
    ImageRichItem
    | ImageGroupRichItem
    | LiveWidgetRichItem
    | ToolRenderRichItem
    | CanvasRichItem
    | CitationRichItem
    | ResourceLinkRichItem,
    Field(discriminator="type"),
]
```

Add `"ImageGroupItem"`, `"ImageGroupPayload"`, `"ImageGroupRichItem"` to
`__all__`.

- [ ] **Step 6: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_response_contract.py tests/test_rich_response_metadata.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add app/core/rich_response.py tests/test_rich_response_contract.py
git commit -m "feat: add the image_group rich-item type to the v1 union"
```

### Task 8: Collapse an image search into one rich item

**Files:**
- Modify: `app/ai/tool_execution.py:83-247`
- Modify: `app/core/config.py` (add `rich_image_group_max_items`)
- Test: `tests/test_rich_response_sources.py`

**Interfaces:**
- Consumes: `RichItemType.image_group` (Task 7), `order_tavily_images` (Task 5), gates (Task 4).
- Produces: `build_image_candidates_from_tool_result` may return a single `image_group` dict for an image-search result. Group id format `imagegroup:tool:<tool_call_id>`; group `source` is `"image_search"`.

- [ ] **Step 1: Write the failing tests**

```python
def _brave_payload(count):
    return json.dumps(
        {
            "provider": "brave_image_search",
            "query": "red panda photo",
            "images": [
                {
                    "url": f"https://e.com/{n}.jpg",
                    "thumbnail_url": f"https://cdn.brave.com/{n}.jpg",
                    "mime_type": "image/jpeg",
                    "width": 1200,
                    "height": 800,
                    "source_url": f"https://e.com/page-{n}",
                    "description": f"red panda {n}",
                }
                for n in range(count)
            ],
        }
    )


def test_single_eligible_brave_candidate_stays_an_image_item():
    candidates = build_image_candidates_from_tool_result(
        _brave_payload(1), tool_call_id="c1", tool_name="brave_image_search"
    )
    assert len(candidates) == 1
    assert candidates[0]["type"] == "image"


def test_multiple_brave_candidates_collapse_into_one_group():
    candidates = build_image_candidates_from_tool_result(
        _brave_payload(4), tool_call_id="c1", tool_name="brave_image_search"
    )
    assert len(candidates) == 1
    group = candidates[0]
    assert group["type"] == "image_group"
    assert group["id"] == "imagegroup:tool:c1"
    assert group["source"] == "image_search"
    assert len(group["payload"]["items"]) == 3  # capped
    assert group["provenance"]["query"] == "red panda photo"


def test_group_cells_prefer_brave_thumbnails():
    candidates = build_image_candidates_from_tool_result(
        _brave_payload(2), tool_call_id="c1", tool_name="brave_image_search"
    )
    urls = [cell["url"] for cell in candidates[0]["payload"]["items"]]
    assert all(url.startswith("https://cdn.brave.com/") for url in urls)


def test_group_alt_text_comes_from_the_image_query():
    candidates = build_image_candidates_from_tool_result(
        _brave_payload(2), tool_call_id="c1", tool_name="brave_image_search"
    )
    assert "red panda photo" in candidates[0]["alt_text"]


def test_tavily_images_are_never_grouped():
    payload = json.dumps(
        {
            "provider": "tavily",
            "query": "chip rules",
            "images": [
                {"url": "https://e.com/a.jpg", "description": "a", "source_url": "https://e.com/1"},
                {"url": "https://e.com/b.jpg", "description": "b", "source_url": "https://e.com/2"},
            ],
        }
    )
    candidates = build_image_candidates_from_tool_result(
        payload, tool_call_id="c1", tool_name="tavily_search"
    )
    assert {c["type"] for c in candidates} == {"image"}


def test_group_is_not_emitted_when_all_candidates_are_ineligible():
    payload = json.dumps(
        {
            "provider": "brave_image_search",
            "query": "x",
            "images": [
                {"url": "http://e.com/a.jpg"},
                {"url": "https://e.com/favicon.ico"},
            ],
        }
    )
    assert (
        build_image_candidates_from_tool_result(
            payload, tool_call_id="c1", tool_name="brave_image_search"
        )
        == []
    )
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_response_sources.py -k "brave or group" -v`
Expected: FAIL — the builder returns one `image` item per candidate

- [ ] **Step 3: Add the group cap setting**

```python
    rich_image_group_max_items: int = Field(
        default=3,
        ge=2,
        le=6,
        description=(
            "Maximum cells in one image_group. A deliberate image search collapses "
            "into a single rich item so the model copies one marker, not N."
        ),
    )
```

- [ ] **Step 4: Group the eligible candidates**

At the end of `build_image_candidates_from_tool_result`, replace `return candidates`:

```python
    if metric_provider == "brave" and len(candidates) >= 2:
        return [_group_image_candidates(candidates, tool_call_id=tool_call_id, query=result_query)]
    return candidates
```

`result_query` must be hoisted out of the loop (it is derived from `parsed`, not
from an individual image). Add the builder:

```python
def _group_image_candidates(
    candidates: list[dict[str, Any]],
    *,
    tool_call_id: str | None,
    query: str,
) -> dict[str, Any]:
    """Collapse eligible image-search candidates into one image_group item."""
    cap = max(2, int(getattr(settings, "rich_image_group_max_items", 3)))
    selected = candidates[:cap]
    cells: list[dict[str, Any]] = []
    for candidate in selected:
        payload = candidate.get("payload") or {}
        cell: dict[str, Any] = {
            "url": payload.get("url"),
            "mime_type": payload.get("mime_type") or "image/jpeg",
        }
        for key in ("source_url", "description", "width", "height"):
            value = payload.get(key)
            if value is not None:
                cell[key] = value
        cells.append(cell)
    first_provenance = selected[0].get("provenance") or {}
    return {
        "id": f"imagegroup:tool:{tool_call_id or 'tool'}",
        "type": RichItemType.image_group.value,
        "source": "image_search",
        "display_policy": RichDisplayPolicy.inline_only.value,
        "alt_text": f"Images of {query}" if query else GENERIC_IMAGE_ALT_TEXT,
        "payload": {"items": cells},
        "provenance": {
            "tool_call_id": tool_call_id,
            "tool": first_provenance.get("tool"),
            "provider": first_provenance.get("provider"),
            "query": query,
        },
    }
```

Import `RichDisplayPolicy` if not already imported in this module.

- [ ] **Step 5: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_response_sources.py -v`
Expected: PASS

- [ ] **Step 6: Run the surrounding suites for regressions**

Run: `.venv\Scripts\python.exe -m pytest tests/test_article_image_flow.py tests/test_rich_response_metadata.py tests/test_rich_response_prompt_inventory.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add app/core/config.py app/ai/tool_execution.py tests/test_rich_response_sources.py
git commit -m "feat: collapse an image search result into one image_group item"
```

### Task 9: Query-anchored placement

**Files:**
- Modify: `app/core/rich_placement.py` (add anchoring beside the existing placement)
- Modify: `app/core/config.py` (add `rich_image_anchor_min_score`, `rich_query_anchored_images_enabled`)
- Test: `tests/test_rich_placement.py`

**Interfaces:**
- Consumes: `RichItemType.image_group` (Task 7), candidate `provenance["query"]` (Task 3), group `source == "image_search"` (Task 8).
- Produces: `ImageAnchorEntry(item_id, query, origin, anchorable)` and `anchor_image_items_by_query(content, *, entries, min_score, max_images) -> tuple[str, dict[str, str]]`. Outcome values are `marker`, `query_anchored`, `fallback_anchored`, `unplaced`.

- [ ] **Step 1: Write the failing tests**

```python
from app.core.rich_placement import ImageAnchorEntry, anchor_image_items_by_query

BODY = """Apple Inc reported record services revenue this quarter.

Apple Park in Cupertino remains the company headquarters and cost about five billion dollars to build.

The fruit industry is unrelated to this discussion entirely.
"""


def test_anchors_after_the_block_matching_the_image_query():
    content, outcomes = anchor_image_items_by_query(
        BODY,
        entries=[
            ImageAnchorEntry(
                item_id="imagegroup:tool:c1",
                query="Apple Park Cupertino headquarters",
                origin="image_search",
            )
        ],
        min_score=0.34,
        max_images=2,
    )
    lines = content.split("\n")
    marker_index = lines.index("<!--rich:imagegroup:tool:c1-->")
    assert "Cupertino" in lines[marker_index - 2]
    assert outcomes["imagegroup:tool:c1"] == "query_anchored"


def test_image_search_falls_back_to_first_prose_block_when_nothing_matches():
    content, outcomes = anchor_image_items_by_query(
        BODY,
        entries=[
            ImageAnchorEntry(
                item_id="imagegroup:tool:c1",
                query="quantum chromodynamics lattice diagram",
                origin="image_search",
            )
        ],
        min_score=0.34,
        max_images=2,
    )
    assert "<!--rich:imagegroup:tool:c1-->" in content
    assert outcomes["imagegroup:tool:c1"] == "fallback_anchored"


def test_source_bound_tavily_image_has_no_fallback():
    content, outcomes = anchor_image_items_by_query(
        BODY,
        entries=[
            ImageAnchorEntry(
                item_id="image:tool:c1:0",
                query="quantum chromodynamics lattice diagram",
                origin="web_search_source_bound",
            )
        ],
        min_score=0.34,
        max_images=2,
    )
    assert content == BODY
    assert outcomes["image:tool:c1:0"] == "unplaced"


def test_query_level_image_is_never_anchored():
    content, outcomes = anchor_image_items_by_query(
        BODY,
        entries=[
            ImageAnchorEntry(
                item_id="image:tool:c1:0",
                query="Apple Park Cupertino headquarters",
                origin="web_search_query_level",
                anchorable=False,
            )
        ],
        min_score=0.34,
        max_images=2,
    )
    assert content == BODY
    assert outcomes["image:tool:c1:0"] == "unplaced"


def test_existing_marker_wins_and_is_never_duplicated():
    body = BODY + "\n<!--rich:imagegroup:tool:c1-->\n"
    content, outcomes = anchor_image_items_by_query(
        body,
        entries=[
            ImageAnchorEntry(
                item_id="imagegroup:tool:c1",
                query="Apple Park Cupertino headquarters",
                origin="image_search",
            )
        ],
        min_score=0.34,
        max_images=2,
    )
    assert content == body
    assert content.count("<!--rich:imagegroup:tool:c1-->") == 1
    assert outcomes["imagegroup:tool:c1"] == "marker"


def test_max_images_cap_is_respected():
    entries = [
        ImageAnchorEntry(item_id=f"imagegroup:tool:c{n}", query="Apple Park Cupertino", origin="image_search")
        for n in range(3)
    ]
    content, outcomes = anchor_image_items_by_query(
        BODY, entries=entries, min_score=0.34, max_images=2
    )
    assert sum(1 for v in outcomes.values() if v != "unplaced") == 2


def test_never_anchors_inside_a_fenced_code_block():
    body = "```\nApple Park Cupertino headquarters\n```\n"
    content, outcomes = anchor_image_items_by_query(
        body,
        entries=[
            ImageAnchorEntry(
                item_id="imagegroup:tool:c1",
                query="Apple Park Cupertino headquarters",
                origin="image_search",
            )
        ],
        min_score=0.34,
        max_images=2,
    )
    assert content == body
    assert outcomes["imagegroup:tool:c1"] == "unplaced"


def test_invalid_item_id_is_never_inserted():
    content, outcomes = anchor_image_items_by_query(
        BODY,
        entries=[
            ImageAnchorEntry(item_id="bad id!", query="Apple Park Cupertino", origin="image_search")
        ],
        min_score=0.34,
        max_images=2,
    )
    assert content == BODY
    assert outcomes["bad id!"] == "unplaced"
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_placement.py -k anchor -v`
Expected: FAIL with `ImportError: cannot import name 'ImageAnchorEntry'`

- [ ] **Step 3: Add the settings**

```python
    rich_query_anchored_images_enabled: bool = Field(
        default=False,
        description=(
            "Anchor unreferenced image items on the model's own image-search query "
            "instead of the provider description. While False, the legacy "
            "description-anchored path runs, so this is a straight rollback control."
        ),
    )
    rich_image_anchor_min_score: float = Field(
        default=0.34,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fraction of image-query tokens a paragraph must contain to "
            "receive that image's marker."
        ),
    )
```

- [ ] **Step 4: Implement the anchoring**

In `app/core/rich_placement.py`, after `auto_place_rich_items`:

```python
#: Origins allowed to anchor without a scoring match. Running an image search is
#: itself the intent to display, so a missed keyword match must not silently
#: discard the result.
_FALLBACK_ANCHOR_ORIGINS = frozenset({"image_search"})

#: Minimum post-stopword token count for a block to accept a fallback anchor, so
#: the image never lands under a bare heading or a two-word line.
_FALLBACK_MIN_BLOCK_TOKENS = 12


@dataclass(frozen=True)
class ImageAnchorEntry:
    """One unreferenced image item eligible for query anchoring."""

    item_id: str
    query: str
    origin: str
    anchorable: bool = True


def _first_fallback_block_line(blocks: list[_Block], insertions: dict[int, str]) -> int:
    for block in blocks:
        if block.end_line in insertions:
            continue
        if len(block.tokens) >= _FALLBACK_MIN_BLOCK_TOKENS:
            return block.end_line
    return -1


def anchor_image_items_by_query(
    content: str,
    *,
    entries: list[ImageAnchorEntry],
    min_score: float,
    max_images: int,
) -> tuple[str, dict[str, str]]:
    """Insert markers for unreferenced image items using their image query.

    Returns ``(new_content, outcomes)`` where ``outcomes`` maps each entry id to
    ``"marker"``, ``"query_anchored"``, ``"fallback_anchored"``, or
    ``"unplaced"``. A model-authored marker always wins: its position is
    authoritative and nothing is inserted for that item.
    """
    if not entries:
        return content, {}
    outcomes: dict[str, str] = {}
    if not content:
        return content, {entry.item_id: "unplaced" for entry in entries}

    referenced = set(parse_inline_rich_references(content))
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    blocks = [b for b in _segment_blocks(lines) if not b.is_code]

    insertions: dict[int, str] = {}
    placed = 0
    for entry in entries:
        if entry.item_id in referenced:
            outcomes[entry.item_id] = "marker"
            continue
        if (
            not entry.anchorable
            or not blocks
            or placed >= max_images
            or len(entry.item_id) > RICH_ITEM_ID_MAX_LENGTH
            or not _ITEM_ID_PATTERN.match(entry.item_id)
        ):
            outcomes[entry.item_id] = "unplaced"
            continue

        query_tokens = _tokens(entry.query or "")
        best_line, best = -1, 0.0
        for block in blocks:
            if block.end_line in insertions:
                continue
            score = _score(query_tokens, block.tokens)
            if score > best:
                best_line, best = block.end_line, score

        if best_line >= 0 and best >= min_score:
            outcome = "query_anchored"
        elif entry.origin in _FALLBACK_ANCHOR_ORIGINS:
            best_line = _first_fallback_block_line(blocks, insertions)
            outcome = "fallback_anchored" if best_line >= 0 else "unplaced"
        else:
            outcome = "unplaced"

        if outcome == "unplaced":
            outcomes[entry.item_id] = "unplaced"
            continue
        insertions[best_line] = f"<!--rich:{entry.item_id}-->"
        outcomes[entry.item_id] = outcome
        placed += 1

    if not insertions:
        return content, outcomes

    out: list[str] = []
    for idx, line in enumerate(lines):
        out.append(line)
        marker = insertions.get(idx)
        if marker is not None:
            out.append("")
            out.append(marker)
    return "\n".join(out), outcomes
```

- [ ] **Step 5: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_placement.py -v`
Expected: PASS, including all pre-existing tests

- [ ] **Step 6: Commit**

```bash
git add app/core/config.py app/core/rich_placement.py tests/test_rich_placement.py
git commit -m "feat: anchor unreferenced image items on the model's image query"
```

### Task 10: Wire anchoring into finalization and cap the inventory

**Files:**
- Modify: `app/core/rich_placement.py:271-311` (`finalize_article_content`)
- Modify: `app/core/rich_response.py:667-719` (`build_rich_item_inventory_block`)
- Modify: `app/core/config.py:1581` (`rich_auto_place_max_images` default 3 → 2)
- Test: `tests/test_rich_placement.py`, `tests/test_rich_response_prompt_inventory.py`

**Interfaces:**
- Consumes: `anchor_image_items_by_query`, `ImageAnchorEntry` (Task 9).
- Produces: `_image_anchor_entries(metadata) -> list[ImageAnchorEntry]` in `rich_placement.py`.

- [ ] **Step 1: Write the failing tests**

```python
def test_finalize_uses_query_anchoring_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    monkeypatch.setattr(settings, "rich_query_anchored_images_enabled", True)
    candidate = {
        "id": "imagegroup:tool:c1",
        "type": "image_group",
        "source": "image_search",
        "payload": {"items": []},
        "provenance": {"query": "Apple Park Cupertino headquarters", "provider": "brave_image_search"},
    }
    response = _make_response(BODY, candidates=[candidate])
    content = finalize_article_content(response, BODY)
    assert "<!--rich:imagegroup:tool:c1-->" in content
    assert response.message.content == content


def test_finalize_never_anchors_when_the_flag_is_off(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    monkeypatch.setattr(settings, "rich_query_anchored_images_enabled", False)
    candidate = {
        "id": "imagegroup:tool:c1",
        "type": "image_group",
        "source": "image_search",
        "payload": {"items": []},
        "provenance": {"query": "Apple Park Cupertino headquarters"},
    }
    response = _make_response(BODY, candidates=[candidate])
    assert "imagegroup" not in finalize_article_content(response, BODY)


def test_widgets_still_auto_place_when_query_anchoring_is_on(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    monkeypatch.setattr(settings, "rich_query_anchored_images_enabled", True)
    response = _make_response(
        BODY,
        artifacts=[
            {
                "type": "live_widget",
                "widget_id": "w1",
                "title": "Apple Park cost breakdown",
            }
        ],
    )
    assert "<!--rich:widget:w1-->" in finalize_article_content(response, BODY)
```

Reuse the existing `_make_response` helper in this test file; if its artifact
shape differs, match the shape that `_widget_placement_entries` already
consumes via `extract_live_widgets_from_artifacts`.

And in `tests/test_rich_response_prompt_inventory.py`:

```python
def test_inventory_caps_image_entries_and_counts_a_group_as_one():
    items = [
        {"id": "widget:w1", "type": "live_widget", "title": "W"},
        {"id": "imagegroup:tool:c1", "type": "image_group", "title": "G1"},
        {"id": "imagegroup:tool:c2", "type": "image_group", "title": "G2"},
        {"id": "image:tool:c3:0", "type": "image", "title": "I3"},
    ]
    block = build_rich_item_inventory_block(
        items, max_items=12, max_chars=4000, summary_chars=180, image_max_items=2
    )
    assert "widget:w1" in block
    assert block.count("image_group") + block.count("| image |") == 2
    assert "image:tool:c3:0" not in block
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_placement.py -k "query_anchoring or widgets_still" tests/test_rich_response_prompt_inventory.py -k caps_image_entries -v`
Expected: FAIL — finalization has no anchoring branch, inventory has no image cap

- [ ] **Step 3: Lower the per-answer image cap**

`app/core/config.py:1581`:

```python
    rich_auto_place_max_images: int = Field(
        default=2,
        description=(
            "Maximum image items per answer. Governs both the model-facing "
            "inventory (a group counts as one) and the anchoring path."
        ),
    )
```

- [ ] **Step 4: Build anchor entries and branch in finalization**

In `app/core/rich_placement.py`:

```python
def _image_anchor_entries(metadata: dict[str, Any]) -> list[ImageAnchorEntry]:
    """Map turn-scoped image candidates to anchoring entries.

    Origin decides fallback eligibility: a deliberate image search may anchor
    without a keyword match, a source-bound web-search image may not, and a
    query-level image is never anchored because it carries no page provenance.
    """
    image_types = {RichItemType.image.value, RichItemType.image_group.value}
    entries: list[ImageAnchorEntry] = []
    for candidate in metadata.get("_rich_item_candidates") or []:
        if not isinstance(candidate, dict) or candidate.get("type") not in image_types:
            continue
        item_id = candidate.get("id")
        if not isinstance(item_id, str) or not item_id:
            continue
        provenance = candidate.get("provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        query = str(provenance.get("query") or "").strip()
        if candidate.get("source") == "image_search":
            origin, anchorable = "image_search", True
        elif provenance.get("query_level"):
            origin, anchorable = "web_search_query_level", False
        else:
            origin, anchorable = "web_search_source_bound", bool(query)
        entries.append(
            ImageAnchorEntry(
                item_id=item_id, query=query, origin=origin, anchorable=anchorable
            )
        )
    return entries
```

Then in `finalize_article_content`, replace the item assembly and placement call:

```python
    items = _widget_placement_entries(metadata, getattr(response, "tool_artifacts", None))
    query_anchored = bool(getattr(settings, "rich_query_anchored_images_enabled", False))
    anchor_entries = _image_anchor_entries(metadata) if query_anchored else []
    if not query_anchored:
        items.extend(_image_placement_entries(metadata))
    if not items and not anchor_entries:
        return content

    known_ids = {entry[0] for entry in items} | {e.item_id for e in anchor_entries}
    repaired = _repair_unprefixed_markers(content, known_ids)
    new_content, _placed = auto_place_rich_items(
        repaired,
        items=items,
        max_images=settings.rich_auto_place_max_images,
        min_score=settings.rich_auto_place_min_score,
    )
    if anchor_entries:
        new_content, outcomes = anchor_image_items_by_query(
            new_content,
            entries=anchor_entries,
            min_score=float(getattr(settings, "rich_image_anchor_min_score", 0.34)),
            max_images=int(getattr(settings, "rich_auto_place_max_images", 2)),
        )
        _record_anchor_outcomes(metadata, outcomes)
    if new_content == content:
        return content
```

Add the metrics hook, kept non-fatal:

```python
def _record_anchor_outcomes(metadata: dict[str, Any], outcomes: dict[str, str]) -> None:
    """Record anchor outcomes. Never raises: telemetry must not fail an answer."""
    if not outcomes:
        return
    try:
        from app.observability.rich_images import rich_image_metrics

        providers = {}
        for candidate in metadata.get("_rich_item_candidates") or []:
            if isinstance(candidate, dict) and isinstance(candidate.get("id"), str):
                provenance = candidate.get("provenance")
                providers[candidate["id"]] = (
                    str(provenance.get("provider") or "other")
                    if isinstance(provenance, dict)
                    else "other"
                )
        for item_id, outcome in outcomes.items():
            rich_image_metrics.record_anchor(
                provider=providers.get(item_id, "other"), outcome=outcome
            )
    except Exception:  # noqa: BLE001  # telemetry is best-effort by contract
        return
```

- [ ] **Step 5: Cap image entries in the inventory**

In `build_rich_item_inventory_block`, replace the ordering block:

`app/core/rich_response.py` deliberately does not import `settings` — it is a
pure schema/presentation module and `rich_placement.py` is what reads config. Keep
it that way: add an explicit keyword argument instead.

Change the signature to add `image_max_items: int`, and replace the ordering
block:

```python
    non_image = [item for item in materialized if not _is_image_item(item)]
    image_items = [item for item in materialized if _is_image_item(item)]
    ordered = [*non_image, *image_items[: max(0, int(image_max_items))]]
```

Extend `_is_image_item` so a group counts as an image entry:

```python
def _is_image_item(item: Any) -> bool:
    return _get_type(item) in {RichItemType.image.value, RichItemType.image_group.value}
```

Then pass it from `build_rich_response_guidance` in `app/ai/prompts.py`, which
already reads settings:

```python
    inventory = build_rich_item_inventory_block(
        candidates,
        max_items=max_items if max_items is not None else settings.rich_item_inventory_max_items,
        max_chars=max_chars if max_chars is not None else settings.rich_item_inventory_max_chars,
        summary_chars=(
            summary_chars if summary_chars is not None else settings.rich_item_summary_max_chars
        ),
        image_max_items=settings.rich_auto_place_max_images,
    )
```

The test in Step 1 calls `build_rich_item_inventory_block` directly, so add
`image_max_items=2` to that call.

- [ ] **Step 6: Wire the presentation counter**

`record_presentation` was defined in Task 6 and has no caller yet. A counter that
is never incremented is a dead metric, so wire it where items actually enter the
inventory. Write the failing test first:

```python
def test_presentation_counter_records_each_presented_image(monkeypatch):
    from app.ai import prompts

    recorded = []

    class _Metrics:
        def record_presentation(self, *, provider, count):
            recorded.append((provider, count))

    monkeypatch.setattr(prompts, "rich_image_metrics", _Metrics(), raising=False)
    prompts.build_rich_response_guidance(
        candidates=[
            {
                "id": "imagegroup:tool:c1",
                "type": "image_group",
                "title": "G",
                "provenance": {"provider": "brave_image_search"},
            }
        ],
        enabled=True,
        capability=True,
    )
    assert recorded == [("brave_image_search", 1)]
```

Then in `app/ai/prompts.py`, after the inventory block is built and found
non-empty:

```python
    with suppress(Exception):
        from app.observability.rich_images import rich_image_metrics

        presented: dict[str, int] = {}
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("type") not in {"image", "image_group"}:
                continue
            provenance = candidate.get("provenance")
            provider = (
                str(provenance.get("provider") or "other")
                if isinstance(provenance, dict)
                else "other"
            )
            presented[provider] = presented.get(provider, 0) + 1
        for provider, count in presented.items():
            rich_image_metrics.record_presentation(provider=provider, count=count)
```

Import `suppress` from `contextlib` in `prompts.py`. Telemetry must never break
prompt construction.

Note this counts candidates offered to the inventory builder, which the item and
character caps may then trim. That is the intended meaning for rollout
comparison; if trimming turns out to matter, move the call inside
`build_rich_item_inventory_block` and pass the provider map in.

- [ ] **Step 7: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_placement.py tests/test_rich_response_prompt_inventory.py tests/test_rich_response_metadata.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add app/core/config.py app/core/rich_placement.py app/core/rich_response.py app/ai/prompts.py tests/test_rich_placement.py tests/test_rich_response_prompt_inventory.py
git commit -m "feat: finalize images through query anchoring behind a rollback flag"
```

### Task 11: Externalize every group cell to a protected reference

**Files:**
- Modify: `app/services/message_service.py:2324-2400`
- Test: `tests/test_message_service_web_image_externalization.py`

**Interfaces:**
- Consumes: `image_group` payload shape (Task 7).
- Produces: no new public symbols; `_externalize_remote_rich_images` now handles `image_group`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_group_cells_each_become_protected_references(service, conversation_id, user_id):
    metadata = {
        "rich_items": [
            {
                "id": "imagegroup:tool:c1",
                "type": "image_group",
                "payload": {
                    "items": [
                        {"url": "https://e.com/a.jpg", "mime_type": "image/jpeg"},
                        {"url": "https://e.com/b.jpg", "mime_type": "image/jpeg"},
                    ]
                },
                "provenance": {"provider": "brave_image_search"},
            }
        ]
    }
    content = "Body\n\n<!--rich:imagegroup:tool:c1-->\n"
    new_content, new_metadata = await service._externalize_remote_rich_images(
        content, metadata, conversation_id, user_id
    )
    urls = [c["url"] for c in new_metadata["rich_items"][0]["payload"]["items"]]
    assert all(u.startswith("/web-images/") for u in urls)
    assert len(set(urls)) == 2
    assert "<!--rich:imagegroup:tool:c1-->" in new_content


@pytest.mark.asyncio
async def test_one_failing_cell_keeps_the_group_and_its_marker():
    reference_id = uuid4()

    async def _register(**kwargs):
        if kwargs["upstream_url"].endswith("b.jpg"):
            raise RuntimeError("upstream rejected")
        return SimpleNamespace(id=reference_id)

    web_images = AsyncMock()
    web_images.register.side_effect = _register
    service = _service(web_images)
    content, externalized = await service._externalize_remote_rich_images(
        "Body\n\n<!--rich:imagegroup:tool:c1-->\n",
        _group_metadata(),
        uuid4(),
        uuid4(),
    )

    cells = externalized["rich_items"][0]["payload"]["items"]
    assert [c["url"] for c in cells] == [f"/web-images/{reference_id}"]
    assert "<!--rich:imagegroup:tool:c1-->" in content


@pytest.mark.asyncio
async def test_group_losing_every_cell_drops_the_item_and_marker():
    web_images = AsyncMock()
    web_images.register.side_effect = RuntimeError("upstream rejected")
    service = _service(web_images)
    content, externalized = await service._externalize_remote_rich_images(
        "Body\n\n<!--rich:imagegroup:tool:c1-->\n\nAfter",
        _group_metadata(),
        uuid4(),
        uuid4(),
    )

    assert externalized["rich_items"] == []
    assert "<!--rich:imagegroup:tool:c1-->" not in content
    assert "After" in content
```

Add the shared fixture builder next to the existing `_metadata()` helper in this
file:

```python
def _group_metadata() -> dict:
    return {
        "rich_items": [
            {
                "id": "imagegroup:tool:c1",
                "type": "image_group",
                "source": "image_search",
                "display_policy": "inline_only",
                "alt_text": "Images of a red panda",
                "payload": {
                    "items": [
                        {"url": "https://img.example/a.jpg", "mime_type": "image/jpeg"},
                        {"url": "https://img.example/b.jpg", "mime_type": "image/jpeg"},
                    ]
                },
                "provenance": {"provider": "brave_image_search"},
            }
        ]
    }
```

The first test uses one `reference_id` for both successful calls, so
`test_group_cells_each_become_protected_references` needs distinct ids — give it
`web_images.register.side_effect = [SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())]`
instead of a single `return_value`.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_message_service_web_image_externalization.py -k group -v`
Expected: FAIL — a group item is passed through untouched because the loop
filters `type != "image"`

- [ ] **Step 3: Handle groups in the externalization loop**

Extract the single-URL registration into a helper and add a group branch. Inside
`_externalize_remote_rich_images`, before the existing `type != "image"` check:

```python
            if item.get("type") == RichItemType.image_group.value:
                payload = item.get("payload")
                cells = payload.get("items") if isinstance(payload, dict) else None
                if not isinstance(cells, list):
                    kept_items.append(item)
                    continue
                kept_cells: list[dict[str, Any]] = []
                for cell in cells:
                    if not isinstance(cell, dict):
                        continue
                    reference_url = await self._register_web_image_url(
                        cell.get("url"),
                        expected_mime=cell.get("mime_type"),
                        provider=self._provider_of(item),
                        conversation_id=conversation_id,
                        user_id=user_id,
                    )
                    if reference_url is None:
                        continue
                    cell["url"] = reference_url
                    kept_cells.append(cell)
                if not kept_cells:
                    updated_content = remove_inline_rich_reference(
                        updated_content, str(item.get("id") or "")
                    )
                    continue
                payload["items"] = kept_cells
                kept_items.append(item)
                continue
```

Add the two helpers on the same class:

```python
    @staticmethod
    def _provider_of(item: dict[str, Any]) -> str:
        provenance = item.get("provenance")
        if isinstance(provenance, dict):
            return str(provenance.get("provider") or "other")
        return "other"

    async def _register_web_image_url(
        self,
        raw_url: Any,
        *,
        expected_mime: Any,
        provider: str,
        conversation_id: UUID,
        user_id: UUID,
    ) -> str | None:
        """Return a protected reference URL, or None when it cannot be made.

        Registration is metadata-only and performs no upstream request.
        """
        if not isinstance(raw_url, str) or not raw_url.strip():
            return None
        image_url = raw_url.strip()
        if image_url.startswith(PROTECTED_IMAGE_URL_PREFIXES):
            return image_url
        if urlsplit(image_url).scheme.lower() != "https":
            logging.warning("Web image reference skipped code=web_image_reference_failed")
            return None
        try:
            reference = await self.web_image_service.register(
                conversation_id=conversation_id,
                user_id=user_id,
                upstream_url=image_url,
                expected_mime=expected_mime,
                provider=provider,
            )
            reference_id = (
                reference.get("id")
                if isinstance(reference, dict)
                else getattr(reference, "id", None)
            )
            if reference_id is None:
                raise ValueError("missing reference id")
        except Exception:
            logging.warning("Web image reference skipped code=web_image_reference_failed")
            return None
        return f"/web-images/{reference_id}"
```

`_provider_of` is a `@staticmethod`, so call it as `self._provider_of(item)`.
Refactor the existing single-image branch to call `_register_web_image_url` too,
so both paths share one registration contract.

Note: a group can drop below two cells here, after schema validation has already
run. That is intentional — the persisted record is a projection, and Task 12's
renderer treats a one-cell group as a single figure. Do not re-validate the
payload against `ImageGroupPayload` after externalization.

- [ ] **Step 4: Wire the final-selection counter**

`record_final_selection` was defined in Task 6 and has no caller yet. The true
"survived finalization" point is here, once `kept_items` is known. Write the
failing test first:

```python
@pytest.mark.asyncio
async def test_final_selection_counter_records_surviving_images(monkeypatch):
    from app.services import message_service as module

    recorded = []

    class _Metrics:
        def record_final_selection(self, *, provider, count):
            recorded.append((provider, count))

    monkeypatch.setattr(module, "rich_image_metrics", _Metrics(), raising=False)
    web_images = AsyncMock()
    web_images.register.side_effect = [
        SimpleNamespace(id=uuid4()),
        SimpleNamespace(id=uuid4()),
    ]
    service = _service(web_images)
    await service._externalize_remote_rich_images(
        "Body\n\n<!--rich:imagegroup:tool:c1-->\n",
        _group_metadata(),
        uuid4(),
        uuid4(),
    )
    assert recorded == [("brave_image_search", 1)]
```

A group counts as one final selection, not one per cell — the item is the unit
the model selected.

Then in `_externalize_remote_rich_images`, just before the `return`:

```python
        with suppress(Exception):
            surviving: dict[str, int] = {}
            for kept in kept_items:
                if not isinstance(kept, dict):
                    continue
                if kept.get("type") not in {
                    RichItemType.image.value,
                    RichItemType.image_group.value,
                }:
                    continue
                provider = self._provider_of(kept)
                surviving[provider] = surviving.get(provider, 0) + 1
            for provider, count in surviving.items():
                rich_image_metrics.record_final_selection(provider=provider, count=count)
```

Import `suppress` from `contextlib` and `rich_image_metrics` from
`app.observability.rich_images` at module level in `message_service.py`, so the
test's `monkeypatch.setattr(module, "rich_image_metrics", ...)` binds. Telemetry
must never fail persistence.

- [ ] **Step 5: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_message_service_web_image_externalization.py tests/test_rich_image_metrics.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/services/message_service.py tests/test_message_service_web_image_externalization.py
git commit -m "feat: externalize every image_group cell to a protected reference"
```

---

## Phase 5 — Rendering

### Task 12: Group figure markup

**Files:**
- Modify: `app/ui/rich_response.py:31-98`
- Test: `tests/test_demo_rich_response.py`

**Interfaces:**
- Consumes: `image_group` payload shape (Task 7).
- Produces: `build_inline_image_group_html(cells: list[dict[str, Any]], *, alt_text: str | None = None, max_width_px: int = INLINE_IMAGE_MAX_WIDTH_PX) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
from app.ui.rich_response import build_inline_image_group_html


def _cells(n=3):
    return [
        {
            "url": f"/web-images/{i}",
            "mime_type": "image/jpeg",
            "source_url": f"https://e{i}.com/page",
            "description": f"cell {i}",
        }
        for i in range(n)
    ]


def test_group_renders_one_figure_per_cell_in_a_row():
    html = build_inline_image_group_html(_cells(3), alt_text="Images of red panda")
    assert html.count("<img") == 3
    assert "display:flex" in html or "grid-template-columns" in html


def test_each_cell_links_its_own_source():
    html = build_inline_image_group_html(_cells(2), alt_text="x")
    assert "https://e0.com/page" in html
    assert "https://e1.com/page" in html


def test_single_cell_group_renders_as_one_image():
    html = build_inline_image_group_html(_cells(1), alt_text="x")
    assert html.count("<img") == 1


def test_cell_failure_replaces_only_that_cell():
    html = build_inline_image_group_html(_cells(2), alt_text="x")
    assert "data-role=\"cell-fallback\"" in html
    assert "onerror" in html


def test_group_escapes_hostile_metadata():
    hostile = [
        {"url": '/web-images/1" onload="alert(1)', "description": "<script>x</script>"},
        {"url": "/web-images/2"},
    ]
    html = build_inline_image_group_html(hostile, alt_text='"><script>')
    assert "<script>" not in html
    assert "onload=\"alert(1)\"" not in html


def test_empty_cells_render_nothing():
    assert build_inline_image_group_html([], alt_text="x") == ""
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_demo_rich_response.py -k group -v`
Expected: FAIL with `ImportError: cannot import name 'build_inline_image_group_html'`

- [ ] **Step 3: Implement the group markup**

In `app/ui/rich_response.py`:

```python
#: Maximum cells rendered in one inline group row. Beyond this the row stops
#: being readable at chat width.
INLINE_IMAGE_GROUP_MAX_CELLS: int = 3


def build_inline_image_group_html(
    cells: list[dict[str, Any]],
    *,
    alt_text: str | None = None,
    max_width_px: int = INLINE_IMAGE_MAX_WIDTH_PX,
) -> str:
    """Return one responsive figure holding a row of source-linked image cells.

    A single-cell group renders as one ordinary image. A cell whose image fails
    to load is swapped for a neutral in-place block, so sibling cells and the
    surrounding prose are unaffected.
    """
    usable = [cell for cell in (cells or []) if isinstance(cell, dict) and cell.get("url")]
    if not usable:
        return ""
    usable = usable[:INLINE_IMAGE_GROUP_MAX_CELLS]
    if len(usable) == 1:
        cell = usable[0]
        return build_inline_image_html(
            str(cell.get("url") or ""),
            alt_text=str(cell.get("description") or alt_text or ""),
            source_url=cell.get("source_url"),
            width=cell.get("width"),
            height=cell.get("height"),
            max_width_px=max_width_px,
        )

    group_alt = _html.escape(alt_text or "", quote=True)
    onerror = (
        "const c=this.closest('[data-role=cell]');"
        "c.querySelector('[data-role=cell-fallback]').style.display='block';"
        "this.remove()"
    )
    rendered: list[str] = []
    for cell in usable:
        src = _html.escape(str(cell.get("url") or ""), quote=True)
        cell_alt = _html.escape(str(cell.get("description") or alt_text or ""), quote=True)
        link = _source_link_html(cell.get("source_url"))
        fallback = (
            '<div data-role="cell-fallback" style="display:none;padding:12px;'
            "border-radius:8px;background:#f1f5f9;color:#64748b;font-size:12px;"
            'text-align:center;">Visual unavailable</div>'
        )
        caption = (
            f'<div style="color:#64748b;font-size:12px;margin-top:4px;">{link}</div>'
            if link
            else ""
        )
        rendered.append(
            f'<div data-role="cell" style="flex:1 1 0;min-width:0;">'
            f'<img src="{src}" alt="{cell_alt}" class="img-thumb" loading="lazy" '
            f'style="width:100%;height:auto;border-radius:8px;cursor:zoom-in;" '
            f'onerror="{onerror}" />{fallback}{caption}</div>'
        )
    row = "".join(rendered)
    return (
        f'<figure data-state="loaded" aria-label="{group_alt}" '
        f'style="margin:8px 0;display:flex;gap:8px;align-items:flex-start;'
        f'width:min({int(max_width_px) * 2}px, 100%);">{row}</figure>'
    )
```

Add `"INLINE_IMAGE_GROUP_MAX_CELLS"` and `"build_inline_image_group_html"` to
`__all__`.

- [ ] **Step 4: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_demo_rich_response.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/ui/rich_response.py tests/test_demo_rich_response.py
git commit -m "feat: render image groups as a source-linked row with per-cell failure"
```

### Task 13: Render group segments in `demo.py`

**Files:**
- Modify: `demo.py` (the rich-segment renderer that currently handles `type == "image"`)
- Test: `tests/test_demo_stream_rendering.py`

**Interfaces:**
- Consumes: `build_inline_image_group_html` (Task 12).
- Produces: no new symbols.

**Call sites.** There are exactly two, both needing the group branch:

| Line | Context |
|---|---|
| `demo.py:4675` import, `demo.py:4714` call | one path |
| `demo.py:7771` import, `demo.py:7794` call | the other path |

Confirm they have not moved before editing:

```bash
git grep -n "build_inline_image_html" demo.py
```

- [ ] **Step 1: Confirm both call sites**

Run the `git grep` above. Expected: four hits — two imports, two calls. If the
count differs, re-read the surrounding function before editing so the group
branch lands in every path that renders a `rich` segment.

- [ ] **Step 2: Write the failing test**

```python
def test_group_segment_renders_group_html(monkeypatch):
    from app.ui.rich_response import build_rich_response_view

    metadata = {
        "rich_items_version": 1,
        "rich_items": [
            {
                "id": "imagegroup:tool:c1",
                "type": "image_group",
                "alt_text": "Images of red panda",
                "payload": {
                    "items": [
                        {"url": "/web-images/1", "mime_type": "image/jpeg"},
                        {"url": "/web-images/2", "mime_type": "image/jpeg"},
                    ]
                },
            }
        ],
    }
    view = build_rich_response_view("Body\n\n<!--rich:imagegroup:tool:c1-->\n", metadata)
    rich_segments = [s for s in view.segments if s.kind == "rich"]
    assert len(rich_segments) == 1
    assert rich_segments[0].item["type"] == "image_group"
```

This asserts the view model already routes groups as `rich` segments, which it
does by id lookup — no view-model change is required, only the Streamlit branch.

- [ ] **Step 3: Run to verify the view model already passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_demo_stream_rendering.py -k group_segment -v`
Expected: PASS immediately. If it fails, the view model filters by type and must
be fixed before continuing.

- [ ] **Step 4: Add the group branch at both call sites**

At `demo.py:4714` and `demo.py:7794`, before the existing `image` handling:

```python
            if item_type == "image_group":
                cells = (item.get("payload") or {}).get("items") or []
                group_html = build_inline_image_group_html(
                    cells,
                    alt_text=item.get("alt_text"),
                )
                if group_html:
                    st.markdown(group_html, unsafe_allow_html=True)
                continue
```

Import `build_inline_image_group_html` alongside the existing
`build_inline_image_html` import at both `demo.py:4675` and `demo.py:7771`.

The two sites differ: the `4714` call passes only `src` and `caption`, while
`7794` passes the fuller keyword set. Read each surrounding function to get the
local variable names for the item dict before pasting the branch — do not assume
both call the loop variable `item`.

- [ ] **Step 5: Run the demo suites**

Run: `.venv\Scripts\python.exe -m pytest tests/test_demo_stream_rendering.py tests/test_demo_rich_response.py tests/test_demo_plan_widget.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add demo.py tests/test_demo_stream_rendering.py
git commit -m "feat: render image_group items in the Streamlit chat"
```

### Task 14: Flatten groups into AI SDK file parts

Without this, `image_group` yields no file parts and images silently disappear
for every non-rich-capable client (`ai_sdk_projection.py:221`).

**Files:**
- Modify: `app/services/event_streaming/ai_sdk_projection.py:210-238`
- Test: `tests/test_ai_sdk_context_window.py`

**Interfaces:**
- Consumes: `image_group` payload shape (Task 7).
- Produces: no new symbols; `selected_image_file_parts_from_rich_items` now flattens groups.

- [ ] **Step 1: Write the failing tests**

```python
from app.services.event_streaming.ai_sdk_projection import (
    selected_image_file_parts_from_rich_items,
)


def test_group_cells_become_file_parts_in_order():
    metadata = {
        "rich_items_version": 1,
        "rich_items": [
            {
                "id": "imagegroup:tool:c1",
                "type": "image_group",
                "payload": {
                    "items": [
                        {"url": "/web-images/1", "mime_type": "image/jpeg"},
                        {"url": "/web-images/2", "mime_type": "image/png"},
                    ]
                },
            }
        ],
    }
    parts = selected_image_file_parts_from_rich_items(metadata)
    assert [p["url"] for p in parts] == ["/web-images/1", "/web-images/2"]
    assert [p["mediaType"] for p in parts] == ["image/jpeg", "image/png"]


def test_group_and_image_duplicates_are_deduplicated():
    metadata = {
        "rich_items_version": 1,
        "rich_items": [
            {
                "id": "image:tool:c1:0",
                "type": "image",
                "payload": {"url": "/web-images/1", "mime_type": "image/jpeg"},
            },
            {
                "id": "imagegroup:tool:c2",
                "type": "image_group",
                "payload": {"items": [{"url": "/web-images/1", "mime_type": "image/jpeg"}]},
            },
        ],
    }
    assert len(selected_image_file_parts_from_rich_items(metadata)) == 1


def test_unknown_rich_type_is_skipped_not_rendered():
    metadata = {
        "rich_items_version": 1,
        "rich_items": [{"id": "future:1", "type": "future_thing", "payload": {"x": 1}}],
    }
    assert selected_image_file_parts_from_rich_items(metadata) == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ai_sdk_context_window.py -k "group_cells or duplicates or unknown_rich_type" -v`
Expected: FAIL — group produces no parts

- [ ] **Step 3: Flatten groups**

Replace the loop body in `selected_image_file_parts_from_rich_items`:

```python
    for item in rich_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        payload = item.get("payload") or {}
        if item_type == "image":
            sources = [payload]
        elif item_type == "image_group":
            raw_cells = payload.get("items")
            sources = [c for c in raw_cells if isinstance(c, dict)] if isinstance(raw_cells, list) else []
        else:
            # Unknown/never-rendered types are skipped so a future item type
            # degrades to text rather than leaking raw payload to a client.
            continue
        for source in sources:
            url = source.get("url")
            data = source.get("data")
            mime_type = source.get("mime_type") or "image/png"
            if url:
                file_part = {"url": str(url), "mediaType": str(mime_type)}
            elif data:
                file_part = {"url": f"data:{mime_type};base64,{data}", "mediaType": str(mime_type)}
            else:
                continue
            key = (file_part["url"], file_part["mediaType"])
            if key in seen:
                continue
            seen.add(key)
            file_parts.append(file_part)
```

- [ ] **Step 4: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ai_sdk_context_window.py tests/test_rich_response_streaming.py tests/test_message_history_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Add the pre-v1 retention regression test**

This guards the Task 17 cleanup from sweeping up the legacy reader.

```python
def test_pre_v1_message_still_yields_image_file_parts():
    from app.services.event_streaming.ai_sdk_projection import visible_image_file_parts

    message = {
        "content": "old answer",
        "metadata": {"images": [{"url": "/chat-images/9", "mime_type": "image/png"}]},
    }
    assert visible_image_file_parts(message) != []
```

- [ ] **Step 6: Run it and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ai_sdk_context_window.py -v`
Expected: PASS

```bash
git add app/services/event_streaming/ai_sdk_projection.py tests/test_ai_sdk_context_window.py
git commit -m "fix: flatten image_group cells into AI SDK file parts"
```

---

## Phase 6 — Enable and observe

### Task 15: Turn query anchoring on and document the rollout watch

**Files:**
- Modify: `app/core/config.py` (`rich_query_anchored_images_enabled` default `False` → `True`)
- Modify: `.env.example`
- Modify: `docs/frontend/rich-image-rendering.md`
- Test: `tests/test_rich_placement.py`

**Interfaces:**
- Consumes: everything from Phases 1-5.
- Produces: no new symbols.

- [ ] **Step 1: Write the failing test**

```python
def test_query_anchoring_is_enabled_by_default():
    from app.core.config import Settings

    assert Settings().rich_query_anchored_images_enabled is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_placement.py -k enabled_by_default -v`
Expected: FAIL — default is `False`

- [ ] **Step 3: Flip the default and document it**

Change the field default to `True`. Add `RICH_QUERY_ANCHORED_IMAGES_ENABLED=true`
to `.env.example`.

In `docs/frontend/rich-image-rendering.md`, add an "Image placement" section
covering: `image_group` shape and per-cell failure states, the three anchor
origins and their fallback rules, and the rollout watch list —
`rich_image_anchor_outcomes_total{outcome="unplaced"}` rising is the signal to set
`RICH_QUERY_ANCHORED_IMAGES_ENABLED=false`, and
`rich_image_selections_total` is deprecated pending Task 17.

- [ ] **Step 4: Run the whole affected surface**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_placement.py tests/test_rich_response_sources.py tests/test_rich_response_contract.py tests/test_rich_response_metadata.py tests/test_rich_response_prompt_inventory.py tests/test_rich_response_streaming.py tests/test_article_image_flow.py tests/test_message_service_web_image_externalization.py tests/test_ai_sdk_context_window.py tests/test_demo_rich_response.py tests/test_demo_stream_rendering.py tests/test_tavily_server.py tests/test_rich_image_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Lint and type-check**

Run: `.venv\Scripts\python.exe -m ruff check app demo.py tests`
Expected: no NEW findings. The repository carries roughly 161 pre-existing ruff
findings; compare against a baseline captured on `master` rather than expecting
zero.

- [ ] **Step 6: Commit**

```bash
git add app/core/config.py .env.example docs/frontend/rich-image-rendering.md tests/test_rich_placement.py
git commit -m "feat: enable query-anchored image placement by default"
```

---

## Phase 7 — Mandatory cleanup

### Task 16: Remove the legacy placement path, the rollback flag, and the deprecated counter

Entry condition, from the spec: query anchoring has run enabled in production for
one full observation window with `unplaced` at or below its pre-change level, and
dashboards and alerts have migrated off `rich_image_selections_total`. Do not
start this task before both hold. Everything below lands in one change with no
partial retention.

**Files:**
- Modify: `app/core/rich_placement.py` (remove `_image_placement_entries`, the `GENERIC_IMAGE_ALT_TEXT` sentinel, the `max_images` parameter and `is_image` branch of `auto_place_rich_items`, and the flag branch in `finalize_article_content`)
- Modify: `app/core/config.py` (remove `rich_query_anchored_images_enabled`)
- Modify: `app/observability/rich_images.py` (remove `selections`, `record_selection`, `_SELECTION_OUTCOMES`)
- Modify: `app/ai/tool_execution.py` (remove `record_selection` calls)
- Modify: `.env.example`, `README.md` (remove stale `RICH_AUTO_PLACE_*` image guidance and the retired flag)
- Test: `tests/test_rich_placement.py`, `tests/test_rich_image_metrics.py`

**Interfaces:**
- Consumes: everything from Phases 1-6.
- Produces: `auto_place_rich_items(content, *, items, min_score) -> tuple[str, list[str]]` — the `max_images` parameter is gone.

**Deliberately retained.** Do not remove these; each has a live reader and a
regression test:

| Item | Why |
|---|---|
| `_extract_image_file_parts_from_metadata` and the `metadata["images"]` reader | renders images in conversations persisted before rich items v1 |
| `has_images`, `images_count`, `agentic_images_count` | written on every turn today by `chat_agent.py:89`, `rag_agent.py:903-906`, `graph.py:607-611` |
| `GENERIC_IMAGE_ALT_TEXT` and its use at `tool_execution.py:335` | MCP image content blocks carry no image query and still need a fallback |
| `_repair_unprefixed_markers` | models still author markers for widgets and explicitly selected images |
| `rich_auto_place_min_score`, `rich_auto_place_enabled` | still govern widget auto-placement |
| `rich_auto_place_max_images` | read by the anchoring path and the inventory cap |

- [ ] **Step 1: Write the failing cleanup assertions**

```python
def test_legacy_image_placement_helper_is_gone():
    import app.core.rich_placement as rp

    assert not hasattr(rp, "_image_placement_entries")


def test_auto_place_no_longer_takes_max_images():
    import inspect

    from app.core.rich_placement import auto_place_rich_items

    assert "max_images" not in inspect.signature(auto_place_rich_items).parameters


def test_rollback_flag_is_removed():
    from app.core.config import Settings

    assert not hasattr(Settings(), "rich_query_anchored_images_enabled")


def test_retained_settings_still_have_live_readers():
    from app.core.config import Settings

    settings_obj = Settings()
    assert settings_obj.rich_auto_place_max_images >= 1
    assert settings_obj.rich_auto_place_min_score >= 0
```

In `tests/test_rich_image_metrics.py`:

```python
def test_deprecated_selection_counter_is_removed():
    metrics = RichImageMetrics()
    assert not hasattr(metrics, "record_selection")
    assert "rich_image_selections_total" not in metrics.render().decode()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_placement.py -k "legacy or no_longer_takes or rollback_flag or retained" tests/test_rich_image_metrics.py -k deprecated -v`
Expected: FAIL — all of these still exist

- [ ] **Step 3: Narrow `auto_place_rich_items` to widgets**

Remove the `max_images` parameter, the `images_placed` counter, and the
`is_image` branch. Update the docstring to say it places widget-class items only.
Delete `_image_placement_entries` and its `GENERIC_IMAGE_ALT_TEXT` import if that
import becomes unused.

- [ ] **Step 4: Collapse the flag branch in `finalize_article_content`**

```python
    items = _widget_placement_entries(metadata, getattr(response, "tool_artifacts", None))
    anchor_entries = _image_anchor_entries(metadata)
    if not items and not anchor_entries:
        return content

    known_ids = {entry[0] for entry in items} | {e.item_id for e in anchor_entries}
    repaired = _repair_unprefixed_markers(content, known_ids)
    new_content, _placed = auto_place_rich_items(
        repaired,
        items=items,
        min_score=settings.rich_auto_place_min_score,
    )
    if anchor_entries:
        new_content, outcomes = anchor_image_items_by_query(
            new_content,
            entries=anchor_entries,
            min_score=float(getattr(settings, "rich_image_anchor_min_score", 0.34)),
            max_images=int(getattr(settings, "rich_auto_place_max_images", 2)),
        )
        _record_anchor_outcomes(metadata, outcomes)
    if new_content == content:
        return content
```

- [ ] **Step 5: Remove the deprecated counter**

Delete the `selections` Counter, `record_selection`, and `_SELECTION_OUTCOMES`
from `app/observability/rich_images.py`. Remove every `record_selection` call
from `app/ai/tool_execution.py`, leaving only `record_candidate`. Simplify the
local `_reject` helper accordingly.

- [ ] **Step 6: Remove the retired settings and stale docs**

Delete `rich_query_anchored_images_enabled` from `app/core/config.py` and every
reader. Remove `RICH_QUERY_ANCHORED_IMAGES_ENABLED` from `.env.example`. Remove
stale `RICH_AUTO_PLACE_*` image wording from `.env.example` and `README.md`,
leaving the widget-only description. Update
`docs/frontend/rich-image-rendering.md` to drop the deprecated-metric note.

- [ ] **Step 7: Verify nothing dangles and nothing was over-swept**

Run each and confirm no hits outside `docs/`:

```bash
git grep -n "_image_placement_entries"
git grep -n "record_selection"
git grep -n "rich_image_selections_total"
git grep -n "rich_query_anchored_images_enabled"
git grep -n "RICH_QUERY_ANCHORED_IMAGES_ENABLED"
```

Then confirm the retained items survived:

```bash
git grep -n "_extract_image_file_parts_from_metadata"
git grep -n "GENERIC_IMAGE_ALT_TEXT"
git grep -n "_repair_unprefixed_markers"
git grep -n "rich_auto_place_max_images"
```

Each must still return hits in `app/`.

- [ ] **Step 8: Run the full affected surface plus the retention guards**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rich_placement.py tests/test_rich_image_metrics.py tests/test_rich_response_sources.py tests/test_rich_response_contract.py tests/test_rich_response_metadata.py tests/test_rich_response_prompt_inventory.py tests/test_rich_response_streaming.py tests/test_article_image_flow.py tests/test_message_service_web_image_externalization.py tests/test_ai_sdk_context_window.py tests/test_demo_rich_response.py tests/test_demo_stream_rendering.py tests/test_demo_plan_widget.py -v`
Expected: PASS, including `test_pre_v1_message_still_yields_image_file_parts`
from Task 14 and every pre-existing widget auto-placement test unchanged.

- [ ] **Step 9: Lint**

Run: `.venv\Scripts\python.exe -m ruff check app demo.py tests`
Expected: no new findings versus the `master` baseline; unused-import findings
introduced by deletions must be fixed.

- [ ] **Step 10: Commit**

```bash
git add app/core/rich_placement.py app/core/config.py app/observability/rich_images.py app/ai/tool_execution.py .env.example README.md docs/frontend/rich-image-rendering.md tests/test_rich_placement.py tests/test_rich_image_metrics.py
git commit -m "refactor: remove the legacy image placement path and its rollback controls"
```

---

## Self-Review Notes

Spec coverage check, section by section:

| Spec section | Task |
|---|---|
| Decision 1, retrieval intent | 2 |
| Decision 2, image query formulation | 2 |
| Decision 3, Tavily depth contract | 1 |
| Decision 4, eligibility, gates, ordering, `source_title` | 3, 4, 5 |
| Decision 5, one item per image search | 7, 8 |
| Decision 6, query-anchored placement | 9, 10 |
| Decision 7, alt text and caption ownership | 8 (group alt from query), 10 |
| Decision 8, per-cell delivery and failure | 11, 12, 14 |
| Decision 9, stage-truthful metrics | 6 defines all four counters; 6 wires `record_candidate`, 10 wires `record_presentation` and `record_anchor`, 11 wires `record_final_selection` |
| Rollout steps 1-6 | Tasks 1-15 |
| Cleanup phase | 16 |

Defects found and fixed during this review:

1. `record_presentation` and `record_final_selection` were defined in Task 6 with
   no caller — dead metrics, the exact failure the cleanup phase forbids. Wired in
   Task 10 Step 6 and Task 11 Step 4, each with its own failing test first.
2. Task 11 called `_provider_of(item)` inside a method; corrected to
   `self._provider_of(item)`.
3. Task 10 offered an either/or for the inventory cap (import settings *or* add a
   keyword). Resolved to the keyword argument, because `rich_response.py`
   deliberately imports no config.
4. Task 11's failure tests had elided bodies; written out, along with the
   `_group_metadata()` fixture and the distinct-id fix for the success case.

Known gap accepted deliberately: the spec's per-cell alt-text fallback of "query
plus source domain" is implemented in Task 12 as description-or-group-alt.
Adding the domain suffix is a one-line change in `build_inline_image_group_html`
if screen-reader review asks for it; it is not worth a separate task.

Ambiguous-entity coverage: Task 9 tests the mechanism with the Apple case. The
spec also names Java, Gemini, and Paris; those are the same code path with
different tokens, so they belong in the same parametrized test rather than as
separate tasks. Add them to
`test_anchors_after_the_block_matching_the_image_query` as parameters if the
reviewer wants the spec's list covered literally.
