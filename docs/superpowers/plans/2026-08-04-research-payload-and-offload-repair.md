# Research Payload and Offload Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Tavily a text-only research path and make a truncated tool result recoverable, so the model stops re-searching to compensate for a preview that contained only scraped image metadata.

**Architecture:** Three independent changes. The Tavily MCP tool stops requesting and emitting page images and starts requesting Tavily's synthesized answer. The offload preview becomes structure-aware, budgeting the preview across `answer` and `results` instead of taking a blind character prefix. A new in-process `read_tool_result` tool lets the model read the rest of an offloaded blob, scoped to its own conversation and user.

**Tech Stack:** Python 3.11+, FastMCP (stdio MCP servers), LangChain `StructuredTool`, Pydantic v2 settings, SQLAlchemy 2.x, dependency-injector, pytest.

This is Phase 1 of `docs/superpowers/specs/2026-08-04-vision-verified-image-injection-design.md`. It ships unflagged and is independent of the vision verifier in Phase 2.

## Global Constraints

- Run every command from the repository root with the app runtime: `.venv/Scripts/python.exe -m pytest ...`. Do not use a bare `python`; three interpreters exist in this checkout and only `.venv` is the app runtime.
- Functions: 100 lines max, cyclomatic complexity 8 max, 5 positional parameters max, 100-character lines.
- Zero new warnings from `ruff`. The repository has roughly 161 pre-existing `ruff` findings; do not fix unrelated ones, and do not add any.
- Comments explain WHY, not WHAT. Delete commented-out code rather than keeping it.
- `results` is serialized first in the Tavily search payload, ahead of every optional diagnostic key. Field order is part of the contract because truncation is order-sensitive.
- The `answer` share of the preview budget is `0.25` initially. The per-result content floor is `200` characters initially. Both are configuration, not literals at the call site.
- The `read_tool_result` slice cap is `8000` characters initially, and a request above the cap is clamped, never rejected.
- A blob belonging to another conversation or another user must be reported not-found, with the two cases indistinguishable to the caller.
- Another session may commit in this checkout with a broad `git add`. Stage explicit paths only, never `git add -A` or `git add .`, and verify with `git status --short` rather than trusting an exit code.
- Do not stage `example_run.txt` or `superpowers-main.zip`; both are intentionally untracked.

---

### Task 1: Tavily search returns text and sources only

Removes the scraped-image array that dominated the offloaded payload, and requests the synthesized answer that the normalizer already reads but never receives.

**Files:**
- Modify: `app/ai/mcp_servers/tavily_server.py:134-296`
- Modify: `app/core/config.py:326-343` (delete two settings)
- Modify: `app/ai/prompts.py:34`
- Test: `tests/test_tavily_server.py:106-254`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `tavily_search(query: str, max_results: int | None = None, search_depth: str | None = None, include_raw_content: bool = False, auto_parameters: bool | None = None) -> str`. The returned JSON object has key order `results`, `total_results`, `answer`, `provider`, `operation`, `query`, then any of `auto_parameters`, `usage`, `request_id`, `response_time` that the provider returned. There is no `images` key. `_normalize_search_response(*, query: str, response: Any) -> dict[str, Any]` loses its `include_images` parameter.

- [ ] **Step 1: Write the failing tests**

Replace the five image-related tests in `tests/test_tavily_server.py` (`test_search_uses_configured_basic_depth_and_preserves_images`, `test_search_explicit_false_suppresses_provider_images`, `test_search_normalizes_result_bound_images_with_parent_provenance`, `test_default_search_requests_source_bound_images`, `test_explicit_include_images_still_requests_them`, and `test_explicit_include_images_false_suppresses_them`) with these. Keep every other test in the file untouched — `tavily_extract` and `tavily_crawl` keep their own `include_images` parameter and their tests must still pass.

```python
def test_search_requests_answer_and_never_requests_images(monkeypatch):
    client = _FakeTavilyClient(
        {
            "query": "openai news",
            "answer": "OpenAI shipped a model.",
            "images": [{"url": "https://example.com/a.jpg", "description": "A"}],
            "results": [
                {
                    "title": "Source",
                    "url": "https://example.com",
                    "content": "Snippet",
                    "score": 0.9,
                    "raw_content": "Full text",
                    "images": [{"url": "https://example.com/bound.jpg"}],
                }
            ],
            "usage": {"credits": 1},
            "request_id": "req-1",
        }
    )
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)
    monkeypatch.setattr(
        tavily_server.settings, "tavily_search_default_depth", "basic", raising=False
    )
    monkeypatch.setattr(
        tavily_server.settings, "tavily_search_auto_parameters", False, raising=False
    )

    payload = json.loads(tavily_server.tavily_search("openai news", max_results=25))

    assert client.calls[0]["search_depth"] == "basic"
    assert client.calls[0]["max_results"] == 10
    assert client.calls[0]["include_answer"] is True
    assert "include_images" not in client.calls[0]
    assert "include_image_descriptions" not in client.calls[0]
    assert "images" not in payload
    assert payload["answer"] == "OpenAI shipped a model."
    assert payload["results"][0]["raw_content"] == "Full text"
    assert payload["usage"] == {"credits": 1}


def test_search_payload_orders_results_before_diagnostics(monkeypatch):
    client = _FakeTavilyClient(
        {
            "answer": "a",
            "results": [{"title": "T", "url": "https://e.example", "content": "c", "score": 1}],
            "usage": {"credits": 1},
            "request_id": "req-2",
            "response_time": 0.4,
        }
    )
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)

    keys = list(json.loads(tavily_server.tavily_search("ordering")).keys())

    assert keys[0] == "results"
    assert keys.index("results") < keys.index("usage")
    assert keys.index("results") < keys.index("request_id")
    assert keys.index("results") < keys.index("response_time")


def test_search_rejects_an_include_images_argument(monkeypatch):
    monkeypatch.setattr(tavily_server, "_make_client", lambda: _FakeTavilyClient({"results": []}))

    with pytest.raises(TypeError):
        tavily_server.tavily_search("apple park", include_images=True)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tavily_server.py -v`
Expected: the three new tests FAIL — `include_answer` missing from the captured params, `images` present in the payload, `keys[0]` is `"provider"`, and no `TypeError` is raised.

- [ ] **Step 3: Remove the image path from the search tool**

In `app/ai/mcp_servers/tavily_server.py`, delete the `include_images` parameter from `tavily_search`, delete the `images_enabled` block, and add `include_answer`:

```python
@mcp.tool()
def tavily_search(
    query: str,
    max_results: int | None = None,
    search_depth: str | None = None,
    include_raw_content: bool = False,
    auto_parameters: bool | None = None,
) -> str:
    """Search the web for current facts, news, recent information, or source discovery.

    Returns a synthesized ``answer`` plus ranked text results with their source
    URLs. This tool never returns images; ``brave_image_search`` is the only
    source of web images.

    Pass ``auto_parameters=True`` to let Tavily pick the search depth when the
    query intent is genuinely ambiguous; an explicit ``search_depth`` always
    wins. If the user provides a specific URL or snippets are insufficient, use
    ``tavily_extract`` after discovery.
    """
```

Replace the `params` dict with:

```python
    params: dict[str, Any] = {
        "query": cleaned_query,
        "max_results": result_count,
        "include_answer": True,
        "include_raw_content": include_raw_content,
        "auto_parameters": resolved["auto_parameters"],
        "include_usage": True,
    }
```

Change the return to `_normalize_search_response(query=cleaned_query, response=response)`.

- [ ] **Step 4: Reorder and shrink the normalized payload**

Replace `_normalize_search_response` and delete `_normalize_search_image` entirely (nothing else calls it — `tavily_extract` passes its own `images` through untouched):

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
```

- [ ] **Step 5: Delete the two dead settings**

In `app/core/config.py`, delete the `tavily_search_include_images` field (lines 326-338) and the `tavily_search_include_image_descriptions` field (lines 339-343). Leave `tavily_search_default_depth`, `tavily_search_auto_parameters`, `tavily_search_default_max_results`, and `tavily_search_max_results` alone.

Then confirm nothing still reads them:

```bash
grep -rn "tavily_search_include_image" app/ tests/ --include=*.py
```

Expected: no output. If `.env.example` or `README.md` documents `TAVILY_SEARCH_INCLUDE_IMAGES` or `TAVILY_SEARCH_INCLUDE_IMAGE_DESCRIPTIONS`, delete those lines too.

- [ ] **Step 6: Correct the model-facing guidance**

In `app/ai/prompts.py`, replace line 34 of `MEDIA_CAPABILITY_SNIPPET`:

```
- `tavily_search` returns text and sources only. `brave_image_search` is the only source of web images, so call it whenever the subject is visual.
```

Leave the rest of the snippet unchanged; Phase 2 rewrites the block around `web_research`.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tavily_server.py -v`
Expected: PASS, including the untouched `tavily_extract`, `tavily_map`, and `tavily_crawl` tests.

- [ ] **Step 8: Check the wider blast radius**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rich_image_selection.py tests/test_rich_image_selection_runtime.py tests/test_article_image_flow.py tests/test_rich_response_sources.py tests/test_rich_response_prompt_inventory.py -v`

Expected: PASS. These exercise `build_image_candidates_from_tool_result` with synthetic Tavily payloads rather than the live tool, so they are unaffected. If one fails because it asserts on the live tool's `images` key, update that test to the new contract; do not restore the key.

Then: `.venv/Scripts/python.exe -m ruff check app/ai/mcp_servers/tavily_server.py app/core/config.py app/ai/prompts.py`
Expected: no new findings.

- [ ] **Step 9: Commit**

```bash
git add app/ai/mcp_servers/tavily_server.py app/core/config.py app/ai/prompts.py tests/test_tavily_server.py
git status --short
git commit -m "feat: make tavily search text-only with answer synthesis"
```

Confirm from `git status --short` that only those four files were staged before committing.

---

### Task 2: Structure-aware offload preview with a reachable blob id

A 4000-character prefix of a 16000-character JSON payload is the reason the model saw no facts. This budgets the preview across the fields that carry facts and tells the model exactly what is missing and how to get it.

**Files:**
- Create: `app/services/tool_result_preview.py`
- Modify: `app/services/tool_result_blob_service.py:19-69`
- Modify: `app/core/config.py` (add two settings near `tool_result_offload_preview_chars`)
- Modify: `app/core/container.py:315-321`
- Create: `tests/test_tool_result_preview.py`
- Modify: `tests/test_tool_result_blob_service.py:35-61`

**Interfaces:**
- Consumes: the Task 1 payload shape — `results` first, each entry carrying `title`, `url`, `content`.
- Produces: `build_tool_result_preview(output_text: str, *, budget_chars: int, answer_share: float = 0.25, min_result_content_chars: int = 200) -> ToolResultPreview` where `ToolResultPreview` is a frozen dataclass with `text: str`, `omitted_arrays: tuple[tuple[str, int], ...]`, `omitted_results: int`, and `structured: bool`. `ToolResultBlobService.__init__` gains keyword-only `answer_share: float = 0.25` and `min_result_content_chars: int = 200`. `offload_if_large` still returns `{"output": str, "blob_id": str | None, "size_bytes": int}`.

- [ ] **Step 1: Write the failing preview tests**

Create `tests/test_tool_result_preview.py`:

```python
from __future__ import annotations

import json

from app.services.tool_result_preview import build_tool_result_preview


def _payload(result_count: int, content_chars: int) -> str:
    return json.dumps(
        {
            "results": [
                {
                    "index": index,
                    "title": f"Title {index}",
                    "url": f"https://example.com/{index}",
                    "content": "c" * content_chars,
                    "score": 0.9,
                }
                for index in range(1, result_count + 1)
            ],
            "total_results": result_count,
            "answer": "a" * 400,
            "provider": "tavily",
            "operation": "search",
            "query": "t1 league of legends",
            "images": [{"url": f"https://cdn.example/{i}.jpg"} for i in range(24)],
            "usage": {"credits": 1},
        }
    )


def test_structured_preview_keeps_every_result_and_drops_the_array():
    preview = build_tool_result_preview(_payload(5, 3000), budget_chars=4000)

    assert preview.structured is True
    parsed = json.loads(preview.text)
    assert [entry["title"] for entry in parsed["results"]] == [
        "Title 1",
        "Title 2",
        "Title 3",
        "Title 4",
        "Title 5",
    ]
    assert all(entry["url"].startswith("https://example.com/") for entry in parsed["results"])
    assert "images" not in parsed
    assert preview.omitted_arrays == (("images", 24),)
    assert preview.omitted_results == 0
    assert len(preview.text) <= 4000


def test_answer_is_capped_at_its_configured_share():
    preview = build_tool_result_preview(_payload(3, 1000), budget_chars=4000, answer_share=0.25)

    assert len(json.loads(preview.text)["answer"]) <= 1000


def test_titles_and_urls_are_never_truncated_when_content_is():
    preview = build_tool_result_preview(_payload(4, 5000), budget_chars=1200)

    parsed = json.loads(preview.text)
    assert parsed["results"][0]["title"] == "Title 1"
    assert parsed["results"][0]["url"] == "https://example.com/1"
    assert len(parsed["results"][0]["content"]) < 5000


def test_later_results_are_dropped_whole_below_the_content_floor():
    preview = build_tool_result_preview(
        _payload(10, 4000), budget_chars=1000, min_result_content_chars=200
    )

    parsed = json.loads(preview.text)
    assert 1 <= len(parsed["results"]) < 10
    assert preview.omitted_results == 10 - len(parsed["results"])
    assert all(len(entry["content"]) >= 200 for entry in parsed["results"])


def test_non_json_payload_keeps_character_prefix():
    preview = build_tool_result_preview("x" * 100, budget_chars=10)

    assert preview.structured is False
    assert preview.text == "x" * 10
    assert preview.omitted_arrays == ()


def test_json_without_a_results_list_keeps_character_prefix():
    preview = build_tool_result_preview(json.dumps({"provider": "x", "note": "y" * 100}), budget_chars=20)

    assert preview.structured is False
    assert len(preview.text) <= 20
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_result_preview.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.tool_result_preview'`.

- [ ] **Step 3: Write the preview builder**

Create `app/services/tool_result_preview.py`:

```python
"""Budgeted, structure-aware previews for offloaded tool results.

A blind character prefix of a JSON tool result keeps whichever key happens to
be serialized first and silently discards the rest. Research payloads paid for
that: the model received image metadata and no facts, and re-ran the search.
This module spends the preview budget on the fields that answer the question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

_IDENTITY_KEYS = ("provider", "operation", "query", "tool", "total_results")
_RESULT_VERBATIM_KEYS = ("index", "title", "url", "score")


@dataclass(frozen=True, slots=True)
class ToolResultPreview:
    """The inline text for an offloaded result and what it left out."""

    text: str
    omitted_arrays: tuple[tuple[str, int], ...] = ()
    omitted_results: int = 0
    structured: bool = False


def build_tool_result_preview(
    output_text: str,
    *,
    budget_chars: int,
    answer_share: float = 0.25,
    min_result_content_chars: int = 200,
) -> ToolResultPreview:
    """Return a preview of at most ``budget_chars`` characters."""

    budget = max(1, int(budget_chars))
    parsed = _parse_object_with_results(output_text)
    if parsed is None:
        return ToolResultPreview(text=output_text[:budget].rstrip())

    results = parsed["results"]
    preview: dict[str, Any] = {"results": []}
    for key in _IDENTITY_KEYS:
        value = parsed.get(key)
        if value is not None and not isinstance(value, (list, dict)):
            preview[key] = value

    answer = parsed.get("answer")
    if isinstance(answer, str) and answer:
        preview["answer"] = answer[: max(1, int(budget * float(answer_share)))]

    omitted_arrays = tuple(
        (key, len(value))
        for key, value in parsed.items()
        if key != "results" and isinstance(value, list)
    )

    remaining = budget - len(_dump(preview))
    kept, share = _result_allocation(
        result_count=len(results),
        remaining=remaining,
        floor=max(1, int(min_result_content_chars)),
    )
    preview["results"] = [_shrink_result(entry, share) for entry in results[:kept]]

    text = _dump(preview)
    return ToolResultPreview(
        text=text if len(text) <= budget else text[:budget],
        omitted_arrays=omitted_arrays,
        omitted_results=max(0, len(results) - kept),
        structured=True,
    )


def _parse_object_with_results(output_text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(output_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
        return None
    return parsed


def _result_allocation(*, result_count: int, remaining: int, floor: int) -> tuple[int, int]:
    """Return how many results to keep and how many content chars each gets."""

    if result_count <= 0 or remaining <= 0:
        return 0, floor
    share = remaining // result_count
    if share >= floor:
        return result_count, share
    return max(1, remaining // floor), floor


def _shrink_result(entry: Any, content_chars: int) -> dict[str, Any]:
    source = entry if isinstance(entry, dict) else {}
    shrunk: dict[str, Any] = {
        key: source[key] for key in _RESULT_VERBATIM_KEYS if key in source
    }
    content = source.get("content")
    if isinstance(content, str):
        shrunk["content"] = content[:content_chars]
    return shrunk


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_result_preview.py -v`
Expected: PASS. If `test_structured_preview_keeps_every_result_and_drops_the_array` fails on the length assertion, the identity keys plus a 1000-character answer are consuming the budget — that is the intended trade, so verify the failure is a genuine overflow past `budget_chars` and not a test expectation that is too strict.

- [ ] **Step 5: Write the failing service tests**

Replace `test_offload_if_large_stores_content_in_db_and_creates_no_files` in `tests/test_tool_result_blob_service.py` and add two tests:

```python
def test_offload_stores_content_and_notice_carries_the_blob_id(tmp_path):
    repo = FakeRepository()
    service = ToolResultBlobService(repo, storage_root=tmp_path, threshold_chars=10)

    result = service.offload_if_large(
        conversation_id=uuid4(),
        user_id=uuid4(),
        tool_call_id="call-1",
        tool_name="search_documents",
        output_text="abcdefghijklmnopqrstuvwxyz",
    )

    assert result["output"].startswith("abcdefghij")
    assert f"blob_id={result['blob_id']}" in result["output"]
    assert "26 chars" in result["output"]
    assert "read_tool_result" in result["output"]
    assert result["size_bytes"] == 26
    record = repo.created[0]
    assert record["content"] == "abcdefghijklmnopqrstuvwxyz"
    assert record["storage_path"] is None
    assert list(tmp_path.iterdir()) == [], "offload must not create files on disk"


def test_offload_notice_names_the_omitted_array_and_dropped_results(tmp_path):
    repo = FakeRepository()
    service = ToolResultBlobService(
        repo, storage_root=tmp_path, threshold_chars=100, preview_chars=600
    )
    payload = json.dumps(
        {
            "results": [
                {"title": f"T{i}", "url": f"https://e.example/{i}", "content": "c" * 900}
                for i in range(6)
            ],
            "total_results": 6,
            "answer": "",
            "provider": "tavily",
            "images": [{"url": "https://cdn.example/a.jpg"} for _ in range(24)],
        }
    )

    result = service.offload_if_large(
        conversation_id=uuid4(),
        user_id=uuid4(),
        tool_call_id="call-2",
        tool_name="tavily_search",
        output_text=payload,
    )

    assert "images (24 entries)" in result["output"]
    assert "further results" in result["output"]
    assert "https://cdn.example/a.jpg" not in result["output"]
```

Add `import json` to the file's imports.

- [ ] **Step 6: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_result_blob_service.py -v`
Expected: FAIL — the notice is still the fixed string `"[Output offloaded: use blob_id to read the full result.]"` with no id, no length, and no omission list.

- [ ] **Step 7: Wire the builder into the service**

In `app/services/tool_result_blob_service.py`, add the import and two constructor parameters, and replace the preview and notice construction:

```python
from app.services.tool_result_preview import ToolResultPreview, build_tool_result_preview
```

```python
    def __init__(
        self,
        repository,
        *,
        storage_root: str | Path,
        threshold_chars: int,
        preview_chars: int | None = None,
        answer_share: float = 0.25,
        min_result_content_chars: int = 200,
    ):
        self.repository = repository
        self.storage_root = Path(storage_root)
        self.threshold_chars = max(1, int(threshold_chars))
        self.preview_chars = max(1, int(preview_chars or threshold_chars))
        self.answer_share = min(0.9, max(0.0, float(answer_share)))
        self.min_result_content_chars = max(1, int(min_result_content_chars))
```

Replace the two lines that build `preview` and the returned `output` with:

```python
        preview = build_tool_result_preview(
            output_text,
            budget_chars=self.preview_chars,
            answer_share=self.answer_share,
            min_result_content_chars=self.min_result_content_chars,
        )
        record_id = record["id"] if isinstance(record, dict) else record.id
        return {
            "output": f"{preview.text.rstrip()}\n\n{_notice(record_id, len(output_text), preview)}",
            "blob_id": str(record_id),
            "size_bytes": len(encoded),
        }
```

Add the module-level notice builder below the class:

```python
def _notice(blob_id: Any, total_chars: int, preview: ToolResultPreview) -> str:
    """Describe what the preview omitted and how to read the rest."""

    omissions = [f"{key} ({count} entries)" for key, count in preview.omitted_arrays]
    if preview.omitted_results:
        omissions.append(f"{preview.omitted_results} further results")
    detail = f" Omitted: {'; '.join(omissions)}." if omissions else ""
    return (
        f"[Output offloaded: {total_chars} chars stored as blob_id={blob_id}.{detail}"
        f' Call read_tool_result(blob_id="{blob_id}") to read the rest —'
        " do not repeat the search.]"
    )
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_result_blob_service.py tests/test_tool_result_preview.py -v`
Expected: PASS.

- [ ] **Step 9: Add the settings and container wiring**

In `app/core/config.py`, directly after `tool_result_offload_preview_chars`:

```python
    tool_result_offload_answer_share: float = Field(
        default=0.25,
        ge=0.0,
        le=0.9,
        description=(
            "Share of the preview budget reserved for a synthesized answer before "
            "results are allocated."
        ),
    )
    tool_result_offload_min_result_chars: int = Field(
        default=200,
        ge=1,
        description=(
            "Per-result content floor in a preview. Below this, later results are "
            "dropped whole instead of shrinking every result into uselessness."
        ),
    )
```

In `app/core/container.py`, extend the existing `tool_result_blob_service` provider:

```python
    tool_result_blob_service = providers.Singleton(
        ToolResultBlobService,
        repository=tool_result_blob_repository,
        storage_root=providers.Object(settings.tool_result_blob_storage_dir),
        threshold_chars=providers.Object(settings.tool_result_offload_threshold_chars),
        preview_chars=providers.Object(settings.tool_result_offload_preview_chars),
        answer_share=providers.Object(settings.tool_result_offload_answer_share),
        min_result_content_chars=providers.Object(settings.tool_result_offload_min_result_chars),
    )
```

- [ ] **Step 10: Run the offload integration tests and lint**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_result_blob_service.py tests/test_tool_result_preview.py -v`
Then: `.venv/Scripts/python.exe -m ruff check app/services/tool_result_preview.py app/services/tool_result_blob_service.py app/core/config.py app/core/container.py`
Expected: PASS, no new findings.

- [ ] **Step 11: Commit**

```bash
git add app/services/tool_result_preview.py app/services/tool_result_blob_service.py app/core/config.py app/core/container.py tests/test_tool_result_preview.py tests/test_tool_result_blob_service.py
git status --short
git commit -m "feat: budget offloaded tool previews across facts"
```

---

### Task 3: read_tool_result internal tool

The offload notice has told the model to "use blob_id" since it was written, while no tool existed to do so and the id was never printed. This is that tool.

**Files:**
- Modify: `app/repositories/tool_result_blob.py:25-32`
- Create: `app/ai/tool_result_read_tool.py`
- Modify: `app/core/config.py` (one setting after `tool_result_offload_min_result_chars`)
- Create: `tests/test_tool_result_read_tool.py`

**Interfaces:**
- Consumes: `ToolResultBlobService.read_text(record) -> str` (unchanged), and the `blob_id` printed by the Task 2 notice.
- Produces: `create_read_tool_result_tool(*, repository=None, service=None) -> StructuredTool` named `read_tool_result`, with metadata `{"tool_origin": "internal", "qualified_tool_id": "internal::read_tool_result"}`. Its JSON result is `{"blob_id", "offset", "returned_chars", "total_chars", "next_offset", "content"}` on success, or `{"status": "error", "error_type": "not_found", "retryable": false, "hint": ...}` on any failure. `ToolResultBlobRepository.get_for_user_and_conversation(blob_id: UUID, user_id: UUID, conversation_id: UUID) -> ToolResultBlob | None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tool_result_read_tool.py`:

```python
from __future__ import annotations

import json
from uuid import uuid4

import pytest

from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.tool_result_read_tool import create_read_tool_result_tool

CONVERSATION_ID = str(uuid4())
USER_ID = str(uuid4())
BLOB_ID = str(uuid4())


class FakeRepository:
    def __init__(self, record=None):
        self.record = record
        self.calls = []

    def get_for_user_and_conversation(self, blob_id, user_id, conversation_id):
        self.calls.append((str(blob_id), str(user_id), str(conversation_id)))
        return self.record


class FakeService:
    def __init__(self, text: str):
        self.text = text

    def read_text(self, record):
        return self.text


@pytest.fixture(autouse=True)
def _clean_context():
    clear_tool_context()
    yield
    clear_tool_context()


async def _invoke(tool, **kwargs) -> dict:
    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id=USER_ID, agent_key="search"
    ):
        return json.loads(await tool.ainvoke(kwargs))


@pytest.mark.asyncio
async def test_returns_a_bounded_slice_and_next_offset():
    repository = FakeRepository(record={"id": BLOB_ID})
    tool = create_read_tool_result_tool(
        repository=repository, service=FakeService("0123456789")
    )

    payload = await _invoke(tool, blob_id=BLOB_ID, offset=0, limit=4)

    assert payload["content"] == "0123"
    assert payload["returned_chars"] == 4
    assert payload["total_chars"] == 10
    assert payload["next_offset"] == 4
    assert repository.calls == [(BLOB_ID, USER_ID, CONVERSATION_ID)]


@pytest.mark.asyncio
async def test_final_slice_reports_no_next_offset():
    tool = create_read_tool_result_tool(
        repository=FakeRepository(record={"id": BLOB_ID}), service=FakeService("0123456789")
    )

    payload = await _invoke(tool, blob_id=BLOB_ID, offset=8, limit=50)

    assert payload["content"] == "89"
    assert payload["next_offset"] is None


@pytest.mark.asyncio
async def test_limit_above_the_cap_is_clamped(monkeypatch):
    monkeypatch.setattr(
        "app.ai.tool_result_read_tool.settings.tool_result_read_max_chars", 5, raising=False
    )
    tool = create_read_tool_result_tool(
        repository=FakeRepository(record={"id": BLOB_ID}), service=FakeService("a" * 100)
    )

    payload = await _invoke(tool, blob_id=BLOB_ID, limit=10_000)

    assert payload["returned_chars"] == 5


@pytest.mark.asyncio
async def test_blob_outside_the_conversation_is_not_found():
    tool = create_read_tool_result_tool(
        repository=FakeRepository(record=None), service=FakeService("secret")
    )

    payload = await _invoke(tool, blob_id=BLOB_ID)

    assert payload["status"] == "error"
    assert payload["error_type"] == "not_found"
    assert "secret" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_malformed_blob_id_is_not_found_without_touching_the_repository():
    repository = FakeRepository(record={"id": BLOB_ID})
    tool = create_read_tool_result_tool(repository=repository, service=FakeService("x"))

    payload = await _invoke(tool, blob_id="not-a-uuid")

    assert payload["error_type"] == "not_found"
    assert repository.calls == []


@pytest.mark.asyncio
async def test_missing_tool_context_is_not_found():
    repository = FakeRepository(record={"id": BLOB_ID})
    tool = create_read_tool_result_tool(repository=repository, service=FakeService("x"))

    payload = json.loads(await tool.ainvoke({"blob_id": BLOB_ID}))

    assert payload["error_type"] == "not_found"
    assert repository.calls == []


def test_tool_identity_is_internal():
    tool = create_read_tool_result_tool(
        repository=FakeRepository(), service=FakeService("x")
    )

    assert tool.name == "read_tool_result"
    assert tool.metadata["tool_origin"] == "internal"
    assert tool.metadata["qualified_tool_id"] == "internal::read_tool_result"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_result_read_tool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ai.tool_result_read_tool'`.

- [ ] **Step 3: Add the conversation-scoped repository lookup**

In `app/repositories/tool_result_blob.py`, add below `get_for_user`:

```python
    def get_for_user_and_conversation(
        self,
        blob_id: UUID,
        user_id: UUID,
        conversation_id: UUID,
    ) -> ToolResultBlob | None:
        with self.session_factory() as db:  # type: Session
            statement = select(ToolResultBlob).where(
                ToolResultBlob.id == blob_id,
                ToolResultBlob.user_id == user_id,
                ToolResultBlob.conversation_id == conversation_id,
                ToolResultBlob.deleted_at.is_(None),
            )
            return db.execute(statement).scalars().first()
```

- [ ] **Step 4: Add the slice cap setting**

In `app/core/config.py`, after `tool_result_offload_min_result_chars`:

```python
    tool_result_read_max_chars: int = Field(
        default=8000,
        ge=1,
        description="Maximum characters returned by one read_tool_result call.",
    )
```

- [ ] **Step 5: Write the tool**

Create `app/ai/tool_result_read_tool.py`:

```python
"""Internal tool that reads an offloaded tool result the model was shown a preview of.

Scoping is deliberately blunt: a blob is readable only from the conversation and
user that produced it, and every failure — unknown id, wrong conversation,
missing context, malformed id — returns the same not-found payload so the tool
cannot be used to probe for other conversations' results.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from uuid import UUID

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..core.config import settings
from .tool_context import get_tool_context

logger = logging.getLogger(__name__)

_DESCRIPTION = (
    "Read the full text of a large tool result that was offloaded and shown to you "
    "only as a preview. Pass the blob_id printed in the offload notice. Returns a "
    "bounded slice plus next_offset; call again with that offset to continue. Use "
    "this instead of repeating a search whose result was truncated."
)

_NOT_FOUND = {
    "status": "error",
    "error_type": "not_found",
    "retryable": False,
    "hint": (
        "No offloaded result with that blob_id exists in this conversation. Use the "
        "blob_id exactly as printed in the offload notice."
    ),
}


class ReadToolResultInput(BaseModel):
    blob_id: str = Field(description="The blob_id printed in the offload notice.")
    offset: int = Field(default=0, ge=0, description="Character offset to read from.")
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Characters to return. Clamped to the configured maximum.",
    )


def create_read_tool_result_tool(
    *,
    repository: Any | None = None,
    service: Any | None = None,
) -> StructuredTool:
    """Build the ``read_tool_result`` tool, resolving DI lazily when not injected."""

    async def _read(blob_id: str, offset: int = 0, limit: int | None = None) -> str:
        context = get_tool_context()
        identity = _scoped_identity(blob_id, context.user_id, context.conversation_id)
        if identity is None:
            return json.dumps(_NOT_FOUND)
        resolved_repository, resolved_service = _resolve(repository, service)
        if resolved_repository is None or resolved_service is None:
            return json.dumps(_NOT_FOUND)

        parsed_blob_id, user_uuid, conversation_uuid = identity
        record = await asyncio.to_thread(
            resolved_repository.get_for_user_and_conversation,
            parsed_blob_id,
            user_uuid,
            conversation_uuid,
        )
        if record is None:
            return json.dumps(_NOT_FOUND)
        text = await asyncio.to_thread(resolved_service.read_text, record)

        cap = max(1, int(getattr(settings, "tool_result_read_max_chars", 8000)))
        window = cap if limit is None else max(1, min(int(limit), cap))
        start = max(0, int(offset))
        chunk = text[start : start + window]
        end = start + len(chunk)
        return json.dumps(
            {
                "blob_id": str(parsed_blob_id),
                "offset": start,
                "returned_chars": len(chunk),
                "total_chars": len(text),
                "next_offset": end if end < len(text) else None,
                "content": chunk,
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        coroutine=_read,
        name="read_tool_result",
        description=_DESCRIPTION,
        args_schema=ReadToolResultInput,
        metadata={
            "tool_origin": "internal",
            "qualified_tool_id": "internal::read_tool_result",
        },
    )


def _scoped_identity(
    blob_id: str,
    user_id: str | None,
    conversation_id: str | None,
) -> tuple[UUID, UUID, UUID] | None:
    if not user_id or not conversation_id:
        return None
    try:
        return (
            UUID(str(blob_id).strip()),
            UUID(str(user_id)),
            UUID(str(conversation_id)),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _resolve(repository: Any | None, service: Any | None) -> tuple[Any | None, Any | None]:
    if repository is not None and service is not None:
        return repository, service
    try:
        from ..core.container import Container

        container = Container()
        return (
            repository or container.tool_result_blob_repository(),
            service or container.tool_result_blob_service(),
        )
    except Exception as exc:  # pragma: no cover - DI unavailable in some contexts
        logger.debug("Tool result blob access unavailable: %s", exc)
        return None, None
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_result_read_tool.py -v`
Expected: PASS. If the async tests error with "async def functions are not natively supported", check that `pyproject.toml` sets `asyncio_mode` or add `@pytest.mark.asyncio` consistently — the repository already runs async tests, so copy the convention from `tests/test_message_service_web_image_externalization.py`.

- [ ] **Step 7: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/ai/tool_result_read_tool.py app/repositories/tool_result_blob.py app/core/config.py
git add app/ai/tool_result_read_tool.py app/repositories/tool_result_blob.py app/core/config.py tests/test_tool_result_read_tool.py
git status --short
git commit -m "feat: add read_tool_result for offloaded results"
```

---

### Task 4: Bind read_tool_result and teach recovery

A tool nobody binds is a tool nobody calls. This makes it available to every agent that can produce an offloaded result, and tells the model to use it instead of re-searching.

**Files:**
- Modify: `app/ai/agents/base_agent.py:449-482`
- Modify: `app/ai/prompts.py:343-356`
- Create: `tests/test_read_tool_result_binding.py`

**Interfaces:**
- Consumes: `create_read_tool_result_tool()` from Task 3.
- Produces: no new public interface. `_get_tools_for_binding` includes a tool named `read_tool_result` whenever `settings.tool_result_offload_enabled` is true.

- [ ] **Step 1: Write the failing binding test**

Create `tests/test_read_tool_result_binding.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

from app.ai.agents.base_agent import BaseAgent
from app.ai.schemas import AgentType


class _BindingTestAgent(BaseAgent):
    def _init_gemini(self) -> None:
        self.gemini_client = None
        self.langchain_model = None

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CHAT

    @property
    def agent_id(self) -> str:
        return "binding-test"

    def _get_base_system_prompt(self) -> str:
        return "binding-test"


def _bound_names(monkeypatch, *, offload_enabled: bool) -> list[str]:
    agent = _BindingTestAgent(agent_config_key="search")
    agent.tools = []
    agent.mcp_manager = None
    monkeypatch.setattr(
        "app.ai.agents.base_agent.settings.tool_result_offload_enabled",
        offload_enabled,
        raising=False,
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading",
        lambda _agent_key: True,
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.build_deferred_tool_list",
        lambda **kwargs: list(kwargs.get("internal_tools") or []),
    )
    monkeypatch.setattr(agent, "_get_client_runtime_tools", lambda **kwargs: [])
    monkeypatch.setattr(
        agent, "_get_skills_internal_tools", lambda **kwargs: [SimpleNamespace(name="noop")]
    )
    return [tool.name for tool in agent._get_tools_for_binding(conversation_id="c1")]


def test_read_tool_result_is_bound_when_offload_is_enabled(monkeypatch):
    assert "read_tool_result" in _bound_names(monkeypatch, offload_enabled=True)


def test_read_tool_result_is_absent_when_offload_is_disabled(monkeypatch):
    assert "read_tool_result" not in _bound_names(monkeypatch, offload_enabled=False)


def test_tool_context_prompt_points_at_the_reader():
    from app.ai.prompts import TOOL_CONTEXT_SUFFIX

    assert "read_tool_result" in TOOL_CONTEXT_SUFFIX
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_read_tool_result_binding.py -v`
Expected: FAIL — `read_tool_result` is not in the bound list and not in `TOOL_CONTEXT_SUFFIX`.

- [ ] **Step 3: Bind the tool**

In `app/ai/agents/base_agent.py`, add the import near the other tool factory imports:

```python
from ..tool_result_read_tool import create_read_tool_result_tool
```

Then in `_get_tools_for_binding`, immediately after the `for tool in skills_tools: _add_internal(tool)` loop:

```python
        # A preview-only tool result is unusable without a reader, and the model's
        # only other recovery is to repeat the search that produced it.
        if getattr(settings, "tool_result_offload_enabled", False):
            _add_internal(create_read_tool_result_tool())
```

- [ ] **Step 4: Teach the recovery path**

In `app/ai/prompts.py`, add one line to `TOOL_CONTEXT_SUFFIX` after the `"status":"error"` bullet:

```
- If a tool result ends with an offload notice, the full result is stored. Call `read_tool_result` with the printed blob_id to read the rest. Do NOT repeat the search — the missing content is retrievable, and a near-duplicate query returns the same thing.
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_read_tool_result_binding.py -v`
Expected: PASS.

- [ ] **Step 6: Run the surrounding binding suites**

Run: `.venv/Scripts/python.exe -m pytest tests/test_client_tool_scope.py tests/test_client_tool_isolation.py tests/test_base_agent_dynamic_handoff.py tests/test_custom_agents_tools.py tests/test_graph_tool_budget.py -v`

Expected: PASS. A test that asserts an exact bound-tool list will now see one extra tool; update that expectation to include `read_tool_result` rather than removing the binding.

- [ ] **Step 7: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/ai/agents/base_agent.py app/ai/prompts.py
git add app/ai/agents/base_agent.py app/ai/prompts.py tests/test_read_tool_result_binding.py
git status --short
git commit -m "feat: bind read_tool_result and document recovery"
```

---

### Task 5: Phase-1 regression gate

Proves the two reported symptoms are gone at the boundary a user would feel, and records the behavior change that Phase 1 introduces on its own.

**Files:**
- Create: `tests/test_research_payload_regression.py`

**Interfaces:**
- Consumes: `tavily_search` from Task 1, `build_tool_result_preview` from Task 2.
- Produces: nothing.

- [ ] **Step 1: Write the regression test**

Create `tests/test_research_payload_regression.py`:

```python
"""Trace-shaped regression for the T1 research turn in example_run.txt.

The original failure: a Tavily payload led with a large scraped-image array, the
offload preview kept only that array, the model received no facts, and it
re-ran the search twice.
"""

from __future__ import annotations

import json

from app.ai.mcp_servers import tavily_server
from app.services.tool_result_preview import build_tool_result_preview

SOURCE_URL = "https://www.sheepesports.com/en/all/articles/lol-t1-completed-2026-lck-roster/en"


def _t1_provider_response() -> dict:
    return {
        "query": "T1 League of Legends Esports team news roster 2026",
        "answer": "T1 is a South Korean esports organization.",
        "images": [{"url": f"https://cdn.example/{i}.jpg", "description": "Moi"} for i in range(24)],
        "results": [
            {
                "title": "LoL: T1 completed 2026 LCK roster",
                "url": SOURCE_URL,
                "content": "T1 finalized its 2026 LCK roster. " * 120,
                "score": 0.887,
                "images": [{"url": "https://cdn.example/bound.jpg", "description": "Moi"}],
            },
            {
                "title": "T1 - Leaguepedia",
                "url": "https://lol.fandom.com/wiki/T1",
                "content": "T1 is a South Korean esports organization. " * 120,
                "score": 0.873,
            },
        ],
    }


def test_search_result_carries_facts_and_no_image_metadata(monkeypatch):
    class _Client:
        def search(self, **params):
            return _t1_provider_response()

    monkeypatch.setattr(tavily_server, "_make_client", lambda: _Client())

    raw = tavily_server.tavily_search("T1 League of Legends Esports team news roster 2026")
    payload = json.loads(raw)

    assert "images" not in payload
    assert "cdn.example" not in raw
    assert payload["answer"].startswith("T1 is a South Korean")
    assert len(payload["results"]) == 2


def test_offload_preview_of_that_result_still_contains_both_sources(monkeypatch):
    class _Client:
        def search(self, **params):
            return _t1_provider_response()

    monkeypatch.setattr(tavily_server, "_make_client", lambda: _Client())
    raw = tavily_server.tavily_search("T1 League of Legends Esports team news roster 2026")

    preview = build_tool_result_preview(raw, budget_chars=4000)

    parsed = json.loads(preview.text)
    assert [entry["url"] for entry in parsed["results"]] == [
        SOURCE_URL,
        "https://lol.fandom.com/wiki/T1",
    ]
    assert "South Korean" in json.dumps(parsed)
    assert preview.omitted_arrays == ()
    assert preview.omitted_results == 0
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python.exe -m pytest tests/test_research_payload_regression.py -v`
Expected: PASS. `preview.omitted_arrays` is empty because Task 1 removed the array before offloading ever sees it — that is the point of the test.

- [ ] **Step 3: Run the full affected surface**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_tavily_server.py tests/test_tool_result_preview.py tests/test_tool_result_blob_service.py tests/test_tool_result_read_tool.py tests/test_read_tool_result_binding.py tests/test_research_payload_regression.py tests/test_rich_image_selection.py tests/test_rich_image_selection_runtime.py tests/test_article_image_flow.py tests/test_rich_response_sources.py -v
```

Expected: PASS. Note in the commit body that `tests/test_live_server_document_upload.py` fails for environmental reasons unrelated to this work and is not a regression.

- [ ] **Step 4: Commit**

```bash
git add tests/test_research_payload_regression.py
git status --short
git commit -m "test: add T1 research payload regression"
```

---

## Expected behavior change from Phase 1 alone

Recording this so it is not mistaken for a bug during review:

**Tavily-sourced images disappear.** Today `tavily_search` is the only image supply that does not depend on the model choosing to call an image search. After Task 1 there is no such supply: an answer contains an image only when the model calls `brave_image_search` and a candidate survives the existing deterministic gates. Some answers that previously carried a (frequently irrelevant) image will be text-only until Phase 2 lands.

That is the intended trade — the trace's admitted image was an author portrait, and the one relevant photograph was rejected by a URL-parsing artifact — but it is visible, and it is the reason Phase 2 follows immediately.

## Self-review notes

- Spec coverage for this phase: the Tavily payload contract (Task 1), structure-aware preview and informative notice (Task 2), conversation-scoped reachable blob (Task 3), binding and guidance (Task 4), and the offload-recovery plus research-payload acceptance tests (Tasks 1, 2, 3, 5). The remaining spec sections — `web_research`, the turn-local research budget, the visual verifier, candidate acquisition, the selector removals, and the byte cache — are Phase 2 and are deliberately absent here.
- `read_tool_result`'s not-found payload intentionally reuses the `status`/`error_type`/`retryable`/`hint` shape that `TOOL_CONTEXT_SUFFIX` already teaches the model to read.
- `ToolResultPreview.omitted_arrays` stays a tuple of pairs rather than a dict so the notice's ordering is stable across runs.
