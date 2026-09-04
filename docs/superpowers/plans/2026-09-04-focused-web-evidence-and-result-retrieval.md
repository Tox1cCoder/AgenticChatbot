# Focused Web Evidence and Tool-Result Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace vague, oversized web calls and raw blob paging with typed search intent, focused extraction, bounded evidence, and one objective-driven result retrieval call.

**Architecture:** Ordinary chat/search agents receive three product tools: `web_search` for discovery, `web_open` for question-focused page extraction, and `image_search` for image discovery. Raw Tavily/Brave MCP tools are hidden from ordinary `tool_search`. Large outputs remain durably stored, but `read_tool_result` becomes a bounded relevance retriever over the blob instead of an offset pager. Runtime date context is normalized in Python and carried in tool arguments; prompt text is supporting guidance, not the only correctness layer.

**Tech Stack:** Python 3.10+, Pydantic v2, LangChain `StructuredTool`, Tavily MCP, Brave MCP, PostgreSQL-backed tool-result blobs, pytest/pytest-asyncio.

## Global Constraints

- Preserve raw provider tools for explicitly authorized diagnostic/admin scopes only.
- Never put an unrestricted full page or full blob in ordinary model context.
- Require a search objective and a focused page question; reject empty or ambiguous inputs at the tool boundary.
- Use the server's timezone-aware current date as authoritative. Do not require a `get_current_time` tool round before search.
- Preserve explicit historical/as-of dates. Repair stale years only for `recent` intent.
- Keep search and extraction results source-addressable, deduplicated, and bounded by configuration.
- Keep the existing `/tool-results/{blob_id}` owner-scoped API for human diagnostics; only the model-facing reader changes.
- Do not change generation lifecycle or Continue/Stop behavior in this plan.

---

## File Structure

- Create `app/ai/web_query_contract.py`: typed intent, temporal validation, and provider argument builders.
- Modify `app/ai/mcp_servers/tavily_server.py`: accept validated start/end dates at the provider boundary.
- Create `app/ai/focused_tool_result.py`: JSON/paragraph candidate extraction, scoring, deduplication, and bounded excerpt selection.
- Create `app/ai/web_tools.py`: `web_search`, `web_open`, and `image_search` product tools.
- Modify `app/ai/agents/base_agent.py`: bind the product tools and objective-based blob reader.
- Modify `app/ai/deferred_tool_binding.py`: exclude raw web providers from ordinary discovery.
- Modify `app/ai/prompts.py`: remove mandatory time-tool choreography and document the product-tool contract.
- Modify `app/ai/tool_result_read_tool.py`: replace offsets with focused retrieval inputs and output.
- Modify `app/services/tool_result_blob_service.py`: advertise focused retrieval, not paging.
- Modify `app/core/config.py`: declare web/result evidence limits.
- Remove `app/ai/web_research_tool.py` after callers and tests migrate.
- Create `tests/test_web_query_contract.py` and `tests/test_focused_tool_result.py`.
- Replace `tests/test_web_research_tool.py` with `tests/test_web_tools.py`.
- Modify deferred binding, prompt, result reader, blob service, and research regression tests.

### Task 1: Declare Typed Web Intent and Temporal Normalization

**Files:**
- Create: `app/ai/web_query_contract.py`
- Modify: `app/ai/mcp_servers/tavily_server.py:165-235`
- Modify: `app/core/config.py:1310-1410`
- Create: `tests/test_web_query_contract.py`
- Modify: `tests/test_tavily_server.py`

**Interfaces:**

```python
Freshness = Literal["timeless", "recent", "as_of"]

class WebSearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    objective: str = Field(min_length=3, max_length=500)
    freshness: Freshness = "timeless"
    start_date: date | None = None
    end_date: date | None = None
    locale: str | None = Field(default=None, max_length=32)
    include_domains: list[str] = Field(default_factory=list, max_length=10)
    max_results: int = Field(default=5, ge=1)

def normalize_web_search(
    request: WebSearchRequest,
    *,
    now: datetime,
    configured_max_results: int,
) -> NormalizedWebSearch: ...
```

- [ ] **Step 1: Write failing contract tests**

Cover these cases explicitly:

```python
def test_recent_query_repairs_stale_year_from_authoritative_now():
    request = WebSearchRequest(
        query="best local LLMs in 2024",
        objective="Find the currently strongest local models",
        freshness="recent",
    )
    normalized = normalize_web_search(
        request,
        now=datetime(2026, 9, 4, 12, tzinfo=ZoneInfo("Asia/Bangkok")),
        configured_max_results=8,
    )
    assert normalized.query == "best local LLMs in 2026"
    assert normalized.end_date == date(2026, 9, 4)


def test_as_of_query_preserves_historical_year():
    request = WebSearchRequest(
        query="Python packaging guidance in 2024",
        objective="Report what the guidance said at the end of 2024",
        freshness="as_of",
        end_date=date(2024, 12, 31),
    )
    normalized = normalize_web_search(
        request,
        now=datetime(2026, 9, 4, tzinfo=timezone.utc),
        configured_max_results=8,
    )
    assert "2024" in normalized.query
    assert normalized.end_date == date(2024, 12, 31)
```

Also test whitespace normalization, domain normalization, invalid future ranges, missing `end_date` for `as_of`, max-result clamping, and that `timeless` leaves years untouched.

- [ ] **Step 2: Run the new tests and verify failure**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_web_query_contract.py
```

Expected: import failure because the contract module does not exist.

- [ ] **Step 3: Add declared settings**

Add these validated settings in `app/core/config.py`:

```python
web_search_max_results: int = Field(default=8, ge=1, le=20)
web_search_result_max_chars: int = Field(default=24_000, ge=2_000, le=100_000)
web_open_max_urls: int = Field(default=4, ge=1, le=10)
web_open_max_excerpts: int = Field(default=8, ge=1, le=20)
web_open_max_chars: int = Field(default=18_000, ge=2_000, le=80_000)
tool_result_focus_max_excerpts: int = Field(default=8, ge=1, le=20)
tool_result_focus_max_chars: int = Field(default=16_000, ge=2_000, le=80_000)
```

- [ ] **Step 4: Implement normalization as a pure function**

Use an immutable `NormalizedWebSearch` model. Strip duplicate whitespace, lowercase/IDNA-normalize domains, validate ranges against `now.date()`, clamp `max_results`, and use a word-boundary four-digit-year regex. For `recent`, replace past years in the query with `now.year`; never replace a future year silently—raise a validation error so the model receives a corrective tool error.

Provider construction must be centralized:

```python
def tavily_search_args(value: NormalizedWebSearch) -> dict[str, Any]:
    args = {
        "query": value.query,
        "max_results": value.max_results,
        "search_depth": "advanced",
        "include_raw_content": False,
        "topic": "news" if value.freshness == "recent" else "general",
    }
    if value.start_date:
        args["start_date"] = value.start_date.isoformat()
    if value.end_date:
        args["end_date"] = value.end_date.isoformat()
    if value.include_domains:
        args["include_domains"] = value.include_domains
    return args
```

- [ ] **Step 5: Extend the provider wrapper with exact dates**

Add optional ISO `start_date` and `end_date` fields to `tavily_search`, validate
them with `date.fromisoformat`, require start <= end, and forward them to
`client.search`. Keep `time_range` mutually exclusive with an explicit range.
Add provider-wrapper tests that assert both dates reach the fake client and
malformed/reversed ranges return a bounded `invalid_request` result.

- [ ] **Step 6: Run and commit Task 1**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_web_query_contract.py tests/test_tavily_server.py tests/test_config_validation.py
.\.venv\Scripts\python.exe -m ruff check app/ai/web_query_contract.py app/ai/mcp_servers/tavily_server.py tests/test_web_query_contract.py tests/test_tavily_server.py
git add app/ai/web_query_contract.py app/ai/mcp_servers/tavily_server.py app/core/config.py tests/test_web_query_contract.py tests/test_tavily_server.py
git commit -m "feat: validate web search intent and time context"
```

### Task 2: Build a Bounded Evidence Selector

**Files:**
- Create: `app/ai/focused_tool_result.py`
- Create: `tests/test_focused_tool_result.py`

**Interfaces:**

```python
class FocusedExcerpt(BaseModel):
    source_path: str
    text: str
    score: float

class FocusedResult(BaseModel):
    objective: str
    excerpts: list[FocusedExcerpt]
    total_candidates: int
    omitted_candidates: int
    truncated: bool

def select_focused_excerpts(
    payload: str,
    *,
    objective: str,
    max_excerpts: int,
    max_chars: int,
) -> FocusedResult: ...
```

- [ ] **Step 1: Write failing selector tests**

Tests must prove:

- JSON leaf strings keep stable paths such as `$.results[2].content`.
- Plain text is split into paragraphs with `paragraph[3]` paths.
- exact duplicates and normalized near-duplicates are returned once;
- objective terms rank the matching passage above an unrelated long passage;
- output never exceeds `max_excerpts` or `max_chars` after JSON serialization;
- malformed JSON falls back to paragraph handling;
- empty/no-match payload returns a bounded explanatory object, not the raw payload;
- URLs and titles adjacent to content are preserved as source metadata.

- [ ] **Step 2: Run tests and verify import failure**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_focused_tool_result.py
```

- [ ] **Step 3: Implement deterministic extraction and scoring**

Use only stdlib/Pydantic. Tokenize with Unicode word characters, remove a small static stop-word set, score term coverage plus phrase occurrence, and use original candidate order as the final tie-breaker. Cap every candidate before scoring so a single page cannot dominate memory. Serialize once during selection and trim the final excerpt if necessary to honor the exact character budget.

Do not call an embedding model or LLM inside this utility: retrieval must be cheap, deterministic, trace-light, and available during degraded operation.

- [ ] **Step 4: Run and commit Task 2**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_focused_tool_result.py
.\.venv\Scripts\python.exe -m ruff check app/ai/focused_tool_result.py tests/test_focused_tool_result.py
git add app/ai/focused_tool_result.py tests/test_focused_tool_result.py
git commit -m "feat: select bounded evidence from large tool results"
```

### Task 3: Replace Offset Paging with Objective-Driven Blob Retrieval

**Files:**
- Modify: `app/ai/tool_result_read_tool.py`
- Modify: `app/services/tool_result_blob_service.py:96-121`
- Modify: `tests/test_tool_result_read_tool.py`
- Modify: `tests/test_tool_result_blob_service.py`
- Modify: `tests/test_read_tool_result_binding.py`

**Interfaces:**

```python
class ReadToolResultInput(BaseModel):
    blob_id: str
    objective: str = Field(min_length=3, max_length=500)
    max_excerpts: int | None = Field(default=None, ge=1, le=20)
    max_chars: int | None = Field(default=None, ge=1000, le=80_000)
```

- [ ] **Step 1: Rewrite tests around focused retrieval**

Delete assertions for `offset`, `limit`, and `next_offset`. Add assertions that one invocation returns the relevant tail evidence, reports omitted candidate counts, stays bounded, and retains owner/conversation scoping. Assert the input schema has no `offset` field.

Update the end-to-end offload test to call once:

```python
payload = await _invoke(
    tool,
    blob_id=notice_ids[0],
    objective="Find the TAIL-8 value",
)
assert "TAIL-8" in " ".join(item["text"] for item in payload["excerpts"])
assert "next_offset" not in payload
```

- [ ] **Step 2: Run reader/blob tests and verify failure**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_tool_result_read_tool.py tests/test_tool_result_blob_service.py tests/test_read_tool_result_binding.py
```

- [ ] **Step 3: Use the selector in the model-facing tool**

After the existing scoped repository lookup and decompression, call `select_focused_excerpts`. Clamp caller values to the configured maxima. Return `FocusedResult.model_dump_json()`.

Use this description:

```python
READ_TOOL_RESULT_DESCRIPTION = (
    "Retrieve the passages most relevant to a specific objective from a large "
    "offloaded tool result. Pass the blob_id and a precise question or fact to "
    "find. The response is bounded and source-addressed; do not call repeatedly "
    "with the same objective."
)
```

- [ ] **Step 4: Change the offload notice**

The notice must say:

```text
Call read_tool_result(blob_id="...", objective="the exact fact still needed")
once if the preview does not contain the needed evidence.
```

Remove every instruction to read the full text or continue with an offset. Preserve all existing detail about dropped keys, shortened fields, and omitted arrays.

- [ ] **Step 5: Run and commit Task 3**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_tool_result_read_tool.py tests/test_tool_result_blob_service.py tests/test_read_tool_result_binding.py tests/test_research_payload_regression.py
.\.venv\Scripts\python.exe -m ruff check app/ai/tool_result_read_tool.py app/services/tool_result_blob_service.py tests/test_tool_result_read_tool.py
git add app/ai/tool_result_read_tool.py app/services/tool_result_blob_service.py tests/test_tool_result_read_tool.py tests/test_tool_result_blob_service.py tests/test_read_tool_result_binding.py
git commit -m "refactor: retrieve focused excerpts from offloaded results"
```

### Task 4: Introduce Product Web Tools

**Files:**
- Create: `app/ai/web_tools.py`
- Modify: `app/ai/image_discovery_flow.py`
- Create: `tests/test_web_tools.py`

**Interfaces:**

```python
def create_web_search_tool() -> BaseTool: ...
def create_web_open_tool() -> BaseTool: ...
def create_image_search_tool() -> BaseTool: ...
```

`web_search` uses `WebSearchRequest`. `web_open` accepts `urls: list[AnyHttpUrl]` plus required `question`; `image_search` accepts a concrete `query`, optional `intent`, and bounded count.

- [ ] **Step 1: Write failing product-tool tests**

Cover provider argument mapping, date normalization, focused extract query, maximum URL/result clamps, provider error normalization, content caps, deduplication, and cancellation of a sibling image task. Assert:

```python
assert extract_tool.calls == [{
    "urls": ["https://example.com/a"],
    "query": "Which release date and version are stated?",
    "chunks_per_source": 3,
    "include_images": False,
}]
```

Also assert a direct test invocation can supply `now` through an injected clock, so tests never depend on wall-clock time.

- [ ] **Step 2: Run and verify failure**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_web_tools.py
```

- [ ] **Step 3: Implement `web_search`**

Resolve Tavily lazily, normalize `WebSearchRequest`, call `tavily_search` with `include_raw_content=False`, project each result to `title`, `url`, `published_date`, `snippet`, and score, deduplicate canonical URLs, and enforce `web_search_result_max_chars` before returning JSON.

The tool should not automatically extract every search result. The model selects URLs and calls `web_open` only if snippets are insufficient.

- [ ] **Step 4: Implement `web_open`**

Require `question`. Call `tavily_extract` with `query=question` and `chunks_per_source=3`, never with a missing query. Feed the provider response through `select_focused_excerpts` using the same question, cap to the configured extract limits, and include per-URL failure records without returning raw provider metadata.

- [ ] **Step 5: Implement `image_search`**

Wrap `discover_images` as its own tool. Keep the existing rich image/artifact stream contract and provider result cap. Do not couple it to textual search latency.

- [ ] **Step 6: Run and commit Task 4**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_web_tools.py tests/test_brave_image_search_server.py tests/test_image_preview_stream.py tests/test_rich_response_streaming.py
.\.venv\Scripts\python.exe -m ruff check app/ai/web_tools.py app/ai/image_discovery_flow.py tests/test_web_tools.py
git add app/ai/web_tools.py app/ai/image_discovery_flow.py tests/test_web_tools.py
git commit -m "feat: add focused web search open and image tools"
```

### Task 5: Bind Product Tools and Hide Raw Providers

**Files:**
- Modify: `app/ai/agents/base_agent.py:450-490`
- Modify: `app/ai/deferred_tool_binding.py:65-90`
- Modify: `app/ai/prompts.py:250-310`
- Modify: `tests/test_web_research_binding.py`
- Modify: `tests/test_search_agent_time_context.py`
- Modify: `tests/test_deferred_tool_binding.py`
- Modify: `tests/test_tool_search_runtime_exclusions.py`

- [ ] **Step 1: Write failing binding and exclusion tests**

For chat/search specialists assert `web_search`, `web_open`, `image_search`, and `read_tool_result` are bound. Assert `web_research` is absent. Search `tool_search` with matching descriptions and assert these qualified raw tools cannot be returned in ordinary scope:

```python
RAW_WEB_TOOL_NAMES = frozenset({
    "tavily_search",
    "tavily_extract",
    "brave_image_search",
})
```

`excluded_tool_names` is name-based, so the policy set must contain unqualified
names. Tests should additionally assert the corresponding qualified catalog IDs
(`tavily::...`, `brave::...`) are absent from search results.

Keep an explicit diagnostic-scope test proving an authorized caller can opt into raw tools.

- [ ] **Step 2: Run and verify failures**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_web_research_binding.py tests/test_search_agent_time_context.py tests/test_deferred_tool_binding.py tests/test_tool_search_runtime_exclusions.py
```

- [ ] **Step 3: Change binding and discovery policy**

Bind the three product tools directly in `BaseAgent`. Merge `RAW_WEB_TOOL_NAMES` into the ordinary `excluded_tool_names` passed to deferred discovery. Put the denylist in one exported policy constant, not duplicated prompt strings.

- [ ] **Step 4: Replace mandatory time-tool instructions**

Update search guidance to say:

```text
- Express the user's goal as a concrete objective.
- Use freshness="recent" for current/latest requests and freshness="as_of"
  with an explicit date for historical cutoffs.
- Search first. Open only the few URLs whose snippets cannot answer the question.
- Every web_open call must include the exact question to extract.
- Stop when independent sources support the answer; do not repeat an unchanged query.
```

Remove the instruction that forces `get_current_time` before every web/news search and remove it from `_SEARCH_AGENT_PINNED_SPECS`. The server-injected current-date prompt and Python normalizer remain authoritative.

- [ ] **Step 5: Run and commit Task 5**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_web_research_binding.py tests/test_search_agent_time_context.py tests/test_deferred_tool_binding.py tests/test_tool_search_runtime_exclusions.py tests/test_read_tool_result_binding.py
.\.venv\Scripts\python.exe -m ruff check app/ai/agents/base_agent.py app/ai/deferred_tool_binding.py app/ai/prompts.py
git add app/ai/agents/base_agent.py app/ai/deferred_tool_binding.py app/ai/prompts.py tests/test_web_research_binding.py tests/test_search_agent_time_context.py tests/test_deferred_tool_binding.py tests/test_tool_search_runtime_exclusions.py
git commit -m "refactor: expose bounded web tools to ordinary agents"
```

### Task 6: Remove the Combined Compatibility Tool and Verify Behavior

**Files:**
- Delete: `app/ai/web_research_tool.py`
- Delete: `tests/test_web_research_tool.py`
- Modify: any remaining imports returned by `rg`
- Modify: `docs/operations/routing-v2-rollout.md`

- [ ] **Step 1: Prove there are no production callers**

```powershell
rg -n "web_research|create_web_research_tool" app tests
```

Expected before deletion: only the old module/tests and explicit migration assertions remain. Resolve every production import before deleting.

- [ ] **Step 2: Delete the compatibility module and migrate residual tests**

Use `apply_patch` to delete the two files. Keep regression coverage in `tests/test_web_tools.py`; do not simply discard cancellation, payload, or image coverage.

- [ ] **Step 3: Add rollout observations**

Document metrics/log fields for normalized freshness, search/open count, deduplicated result count, provider payload chars, model-visible chars, focused-reader calls, and repeated-objective rejection. Values derived from user text must be attributes with normal redaction, never metric labels.

- [ ] **Step 4: Run full focused-web verification**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_web_query_contract.py tests/test_focused_tool_result.py tests/test_web_tools.py tests/test_tool_result_read_tool.py tests/test_tool_result_blob_service.py tests/test_read_tool_result_binding.py tests/test_research_payload_regression.py tests/test_deferred_tool_binding.py tests/test_tool_search_runtime_exclusions.py tests/test_search_agent_time_context.py tests/test_specialist_tool_pipeline.py tests/test_ai_sdk_v6_stream_contract.py tests/test_internal_sse_stream_contract.py
.\.venv\Scripts\python.exe -m ruff check app/ai/web_query_contract.py app/ai/focused_tool_result.py app/ai/web_tools.py app/ai/tool_result_read_tool.py app/ai/agents/base_agent.py app/ai/deferred_tool_binding.py app/ai/prompts.py app/services/tool_result_blob_service.py
```

Expected: all pass; `rg` finds no production `web_research` caller and no prompt tells a model to page blobs.

- [ ] **Step 5: Commit Task 6**

```powershell
git add -A app/ai/web_research_tool.py tests/test_web_research_tool.py docs/operations/routing-v2-rollout.md
git commit -m "chore: retire combined web research tool"
```

## Acceptance Checklist

- [ ] Ordinary agents cannot discover raw Tavily/Brave tools.
- [ ] Search requests carry a concrete objective and typed freshness.
- [ ] Current requests use the server's actual year without a preceding time-tool call.
- [ ] Historical/as-of years are preserved.
- [ ] Page extraction always has a focused question and bounded output.
- [ ] `read_tool_result` has no offset/next-offset loop and returns bounded relevant excerpts in one call.
- [ ] Human/admin raw blob access remains owner-scoped and available.
- [ ] Search, extraction, image, offload, and stream regression suites pass.

## Execution Handoff

Execute this plan first. The trace-parenting plan should then instrument
`app/ai/web_tools.py`, which is the durable product boundary created here.
