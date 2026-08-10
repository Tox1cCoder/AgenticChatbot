# Provider-Native Retrieval Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace image LLM verification with Brave-native confidence selection and align Tavily retrieval with its current API contract while preserving protected image delivery.

**Architecture:** `web_research` continues launching Tavily and Brave concurrently. Tavily returns ranked source chunks without a provider-generated answer; Brave returns normalized candidates that a deterministic discovery flow selects by provider confidence and rank. Selected images enter a renamed turn-scoped sink, are persisted as protected references, and are securely fetched only when rendered.

**Tech Stack:** Python 3.11+, asyncio, LangChain `StructuredTool`, FastMCP, httpx, Tavily Python SDK 0.7.x, Pydantic, Prometheus, pytest, Ruff.

## Global Constraints

- Brave remains the only remote answer-image source.
- Tavily remains text-only in `web_research`; full page bodies remain the responsibility of `tavily_extract`.
- Preserve existing rich-response capability, feature, and turn-budget gates.
- Preserve protected `/web-images/{id}` delivery and its SSRF, redirect, MIME, byte-size, decode, dimension, and aspect-ratio protections.
- Do not expose upstream original image URLs in public message metadata.
- Do not add an image LLM, metadata reranker, local Tavily timeout wrapper, Tavily timeout setting, or in-turn provider retry.
- Pass `timeout=10` directly to `TavilyClient.search`.
- Preserve unrelated worktree changes and the user's untracked `example_run.txt` and `example_run2.json` files.

---

### Task 1: Preserve Brave's native retrieval contract

**Files:**
- Modify: `app/ai/mcp_servers/brave_image_search_server.py`
- Modify: `tests/test_brave_image_search_server.py`

**Interfaces:**
- Consumes: Brave `GET /res/v1/images/search`; existing count, SafeSearch, country, language, and timeout settings.
- Produces: normalized JSON with `query_metadata`, `safety`, and image fields `result_rank`, `confidence`, `thumbnail_width`, `thumbnail_height`, and optional original-image metadata; structured errors with `error_type`, `retryable`, and optional `status_code`.

- [ ] **Step 1: Write failing request-contract tests**

Extend the fake response/client so tests can exercise HTTP status codes, then add:

```python
def test_request_enables_spellcheck_and_keeps_strict_safesearch(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    calls = _install_fake_httpx(monkeypatch, payload=_representative_brave_payload())

    srv.brave_image_search("T1 team photo")

    assert calls[0]["params"]["spellcheck"] is True
    assert calls[0]["params"]["safesearch"] == "strict"
```

- [ ] **Step 2: Write failing normalization tests**

Make the representative response contain `query.altered`,
`extra.might_be_offensive`, confidence, crawl time, thumbnail dimensions, and a
result with a thumbnail but no `properties.url`:

```python
def test_normalization_preserves_native_relevance_and_proxy_metadata(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    _install_fake_httpx(monkeypatch, payload=_representative_brave_payload())

    payload = json.loads(srv.brave_image_search("T1 teem photo"))

    assert payload["query_metadata"]["altered"] == "T1 team photo"
    assert payload["safety"] == {"might_be_offensive": False}
    assert payload["images"][0]["result_rank"] == 1
    assert payload["images"][0]["confidence"] == "high"
    assert payload["images"][0]["thumbnail_width"] == 500
    assert payload["images"][0]["thumbnail_height"] == 281
    assert payload["images"][1]["url"] == "https://img.test/thumb-only.jpg"
    assert "original_image_url" not in payload["images"][1]
```

- [ ] **Step 3: Write failing HTTP classification tests**

```python
@pytest.mark.parametrize(
    ("status", "error_type", "retryable"),
    [
        (400, "invalid_request", False),
        (401, "authentication", False),
        (403, "subscription", False),
        (422, "invalid_request", False),
        (429, "rate_limit", True),
        (500, "upstream", True),
    ],
)
def test_http_errors_remain_classifiable(monkeypatch, status, error_type, retryable):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    _install_fake_httpx(monkeypatch, status_code=status, payload={"error": {}})

    payload = json.loads(srv.brave_image_search("T1 team photo"))

    assert payload["status_code"] == status
    assert payload["error_type"] == error_type
    assert payload["retryable"] is retryable
    assert payload["images"] == []
```

- [ ] **Step 4: Run the Brave tests and confirm RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_brave_image_search_server.py
```

Expected: failures because `spellcheck`, native relevance metadata, thumbnail-only results, and bounded HTTP classifications are absent.

- [ ] **Step 5: Implement bounded provider errors**

Change `_error` to accept bounded machine-readable fields:

```python
def _error(
    message: str,
    *,
    error_type: str,
    retryable: bool = False,
    status_code: int | None = None,
) -> str:
    payload: dict[str, Any] = {
        "error": message[:300],
        "error_type": error_type,
        "provider": "brave_image_search",
        "retryable": retryable,
        "images": [],
        "total_results": 0,
    }
    if status_code is not None:
        payload["status_code"] = status_code
    return json.dumps(payload)
```

Catch `httpx.HTTPStatusError` separately and map the response status using one
small helper:

```python
def _classify_http_status(status_code: int) -> tuple[str, bool]:
    if status_code == 429:
        return "rate_limit", True
    if status_code >= 500:
        return "upstream", True
    if status_code == 401:
        return "authentication", False
    if status_code == 403:
        return "subscription", False
    if status_code in {400, 404, 422}:
        return "invalid_request", False
    return "provider_error", False
```

Keep timeout errors retryable with `error_type="timeout"`; keep
transport failures retryable only when they are `httpx.TransportError`.

- [ ] **Step 6: Implement provider-native normalization**

In `_normalize_results`, use `thumbnail.src` as the normalized `url` when it is
present, fall back to `properties.url`, and skip only when neither exists.
Preserve internal original URL only under the private adapter field
`original_image_url`; the public candidate builder will digest and discard it.

```python
display_url = thumbnail_url or direct_url
image = {
    "url": str(display_url),
    "provider": "brave_image_search",
    "result_rank": rank,
    "confidence": str(result.get("confidence") or "").lower(),
}
```

Return:

```python
return {
    "query": query,
    "query_metadata": {
        "original": str(query_data.get("original") or query),
        "altered": query_data.get("altered"),
        "spellcheck_off": bool(query_data.get("spellcheck_off", False)),
        "show_strict_warning": bool(query_data.get("show_strict_warning", False)),
    },
    "safety": {
        "might_be_offensive": bool(extra.get("might_be_offensive", False)),
    },
    "provider": "brave_image_search",
    "images": images,
    "total_results": len(images),
}
```

Add `"spellcheck": True` to request parameters.

- [ ] **Step 7: Run tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_brave_image_search_server.py tests/test_brave_image_search_config.py
.\.venv\Scripts\python.exe -m ruff check app/ai/mcp_servers/brave_image_search_server.py tests/test_brave_image_search_server.py
```

Expected: PASS.

Commit:

```powershell
git add app/ai/mcp_servers/brave_image_search_server.py tests/test_brave_image_search_server.py
git commit -m "fix: preserve Brave image relevance metadata"
```

---

### Task 2: Build deterministic Brave selection

**Files:**
- Create: `app/ai/image_discovery_flow.py`
- Create: `tests/test_image_discovery_flow.py`
- Modify: `app/ai/tool_execution.py`
- Modify: `app/observability/rich_images.py`
- Modify: `tests/test_rich_response_sources.py`
- Modify: `tests/test_rich_image_metrics.py`

**Interfaces:**
- Consumes: normalized Brave JSON and
  `build_image_candidates_from_tool_result(result_text, tool_call_id=None, tool_name="brave_image_search", group_images=False)`.
- Produces: `async discover_images(*, brave_tool, image_query, image_intent=None) -> list[dict[str, Any]]`; pure `select_brave_candidates(raw, *, image_query, image_intent=None) -> list[dict[str, Any]]`; terminal outcomes `selected`, `no_match`, `unavailable`, `search_failure`.

- [ ] **Step 1: Write failing confidence-tier tests**

Create helpers that produce normalized Brave payloads and add:

```python
def test_high_confidence_wins_in_provider_order():
    selected = select_brave_candidates(
        _payload([("medium", 1), ("high", 2), ("high", 3)]),
        image_query="T1 team photo",
    )

    assert [item["provenance"]["result_rank"] for item in selected] == [2, 3]


def test_medium_is_used_only_when_no_high_candidate_survives():
    selected = select_brave_candidates(
        _payload([("medium", 1), ("low", 2), ("medium", 3)]),
        image_query="T1 team photo",
    )

    assert [item["provenance"]["result_rank"] for item in selected] == [1, 3]


@pytest.mark.parametrize("confidence", ["low", "", "unknown"])
def test_low_missing_and_unknown_confidence_are_rejected(confidence):
    assert select_brave_candidates(
        _payload([(confidence, 1)]), image_query="T1 team photo"
    ) == []
```

- [ ] **Step 2: Write failing safety, dimensions, and gallery tests**

```python
def test_offensive_response_selects_nothing():
    assert select_brave_candidates(
        _payload([("high", 1)], offensive=True), image_query="art"
    ) == []


def test_thumbnail_dimensions_do_not_trigger_source_minimum_rejection():
    selected = select_brave_candidates(
        _payload(
            [("high", 1)],
            original_dimensions=None,
            thumbnail_dimensions=(200, 112),
        ),
        image_query="T1 team photo",
    )
    assert len(selected) == 1


def test_gallery_groups_selected_candidates_after_confidence_selection():
    selected = select_brave_candidates(
        _payload([("high", 1), ("high", 2), ("medium", 3)]),
        image_query="T1 roster",
        image_intent="gallery",
    )
    assert len(selected) == 1
    assert selected[0]["type"] == "image_group"
    assert len(selected[0]["payload"]["items"]) == 2
```

- [ ] **Step 3: Write failing provider-error tests**

```python
@pytest.mark.asyncio
async def test_structured_brave_error_is_search_failure(metrics):
    selected = await discover_images(
        brave_tool=_Tool(_error_payload("rate_limit", retryable=True)),
        image_query="T1 team photo",
    )

    assert selected == []
    metrics.record_discovery_outcome.assert_called_once_with(
        outcome="search_failure", duration_seconds=ANY
    )
```

- [ ] **Step 4: Run the new tests and confirm RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_image_discovery_flow.py
```

Expected: collection failure because `app.ai.image_discovery_flow` does not exist.

- [ ] **Step 5: Preserve the Brave fields needed by selection**

In `build_image_candidates_from_tool_result`, add `confidence`,
`thumbnail_width`, `thumbnail_height`, and `original_image_url` to the internal
metadata it reads. Convert `original_image_url` immediately into the existing
`original_image_digests` structure and never put the URL itself in provenance.
Keep original source dimensions in the payload; keep thumbnail dimensions only
in provenance.

Add a regression asserting public provenance contains confidence/rank but not
`original_image_url`.

- [ ] **Step 6: Add bounded discovery outcome metrics**

Add `_DISCOVERY_OUTCOMES`, `rich_image_discovery_outcome_total`,
`rich_image_discovery_duration_seconds`, and:

```python
def record_discovery_outcome(self, *, outcome: str, duration_seconds: float) -> None:
    label = _bounded(outcome, _DISCOVERY_OUTCOMES)
    self.discovery_outcomes.labels(outcome=label).inc()
    self.discovery_duration.labels(outcome=label).observe(
        max(0.0, float(duration_seconds))
    )
```

Add a metrics regression proving unknown outcomes are bucketed as `other` and
tenant content never reaches labels.

- [ ] **Step 7: Implement pure selection**

Create `image_discovery_flow.py` with these focused helpers:

```python
_OPERATIONAL_FAILURES = frozenset({"unavailable", "search_failure"})


def _object_payload(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _might_be_offensive(payload: Mapping[str, Any]) -> bool:
    safety = payload.get("safety")
    return isinstance(safety, Mapping) and bool(safety.get("might_be_offensive"))


def _confidence(candidate: Mapping[str, Any]) -> str:
    provenance = candidate.get("provenance")
    if not isinstance(provenance, Mapping):
        return ""
    return str(provenance.get("confidence") or "").strip().lower()


def select_brave_candidates(
    raw: str,
    *,
    image_query: str,
    image_intent: str | None = None,
) -> list[dict[str, Any]]:
    payload = _object_payload(raw)
    if payload.get("error") or _might_be_offensive(payload):
        return []
    candidates = build_image_candidates_from_tool_result(
        raw,
        tool_call_id=None,
        tool_name="brave_image_search",
        group_images=False,
    )
    high = [item for item in candidates if _confidence(item) == "high"]
    medium = [item for item in candidates if _confidence(item) == "medium"]
    tier = high or medium
    if str(image_intent or "figure").lower() == "gallery" and len(tier) >= 2:
        return [
            _group_image_candidates(
                tier,
                tool_call_id=None,
                query=image_query,
                metric_provider="brave",
                max_items=max(2, int(settings.rich_image_gallery_max_items)),
            )
        ]
    return tier[: max(0, int(settings.rich_auto_place_max_images))]
```

The URL/dimension/aspect/dedup eligibility remains in the existing candidate
builder. Ensure its minimum-size check uses original dimensions only; thumbnail
dimensions stay metadata and therefore do not trigger source-size rejection.

- [ ] **Step 8: Implement asynchronous discovery and outcome reporting**

Add the module-level terminal helper used by every branch:

```python
def record_discovery_outcome(
    outcome: str, *, started: float | None = None
) -> list[dict[str, Any]]:
    elapsed = 0.0 if started is None else time.perf_counter() - started
    with suppress(Exception):
        rich_image_metrics.record_discovery_outcome(
            outcome=outcome, duration_seconds=elapsed
        )
    if outcome in _OPERATIONAL_FAILURES:
        logger.warning(
            "Image discovery produced no image (%s) after %.2fs", outcome, elapsed
        )
    return []
```

Then implement the provider call:

```python
async def discover_images(
    *,
    brave_tool: Any | None,
    image_query: str,
    image_intent: str | None = None,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    if brave_tool is None:
        return record_discovery_outcome("unavailable", started=started)
    try:
        raw = provider_result_text(
            await brave_tool.ainvoke({"query": image_query}),
            tool_name="brave_image_search",
        )
    except Exception:
        return record_discovery_outcome("search_failure", started=started)
    payload = _object_payload(raw)
    if payload.get("error"):
        return record_discovery_outcome("search_failure", started=started)
    selected = select_brave_candidates(
        raw, image_query=image_query, image_intent=image_intent
    )
    record_discovery_outcome("selected" if selected else "no_match", started=started)
    return selected
```

Metrics are best effort and warnings occur only for operational failures.

- [ ] **Step 9: Run focused tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_image_discovery_flow.py tests/test_rich_response_sources.py tests/test_rich_image_metrics.py -k "brave or confidence or discovery or gallery"
.\.venv\Scripts\python.exe -m ruff check app/ai/image_discovery_flow.py app/ai/tool_execution.py app/observability/rich_images.py tests/test_image_discovery_flow.py tests/test_rich_response_sources.py tests/test_rich_image_metrics.py
```

Expected: PASS.

Commit:

```powershell
git add app/ai/image_discovery_flow.py app/ai/tool_execution.py app/observability/rich_images.py tests/test_image_discovery_flow.py tests/test_rich_response_sources.py tests/test_rich_image_metrics.py
git commit -m "feat: select Brave images by native confidence"
```

---

### Task 3: Replace the verified-image channel and cut over web research

**Files:**
- Create: `app/ai/selected_image_sink.py`
- Create: `tests/test_selected_image_sink.py`
- Delete: `app/ai/verified_image_sink.py`
- Delete: `tests/test_verified_image_sink.py`
- Modify: `app/ai/web_research_tool.py`
- Modify: `app/ai/tool_execution.py`
- Modify: `app/ai/agents/base_agent.py`
- Modify: `app/core/config.py`
- Modify: `tests/test_web_research_tool.py`
- Modify: `tests/test_web_research_dependency_resolution.py`
- Modify: `tests/test_web_research_binding.py`

**Interfaces:**
- Consumes: `discover_images`, research budget, tool context, selected-image sink.
- Produces: `selected_image_sink() -> Iterator[list[dict[str, Any]]]`,
  `offer_selected_images(candidates: Sequence[Mapping[str, Any]]) -> None`;
  `create_web_research_tool` no longer accepts `web_image_service`,
  `verifier_model`, or `recorder`; image gate reads
  `settings.remote_image_enrichment_enabled`.

- [ ] **Step 1: Write the selected-sink tests before renaming production code**

Copy the concurrency/isolation behavior tests into the new terminology:

```python
def test_selected_images_are_collected_in_order():
    with selected_image_sink() as sink:
        offer_selected_images([{"id": "image:selected:a"}])
        offer_selected_images([{"id": "image:selected:b"}])
    assert [item["id"] for item in sink] == [
        "image:selected:a",
        "image:selected:b",
    ]
```

Retain the nested-context, await, child-task, and concurrent-turn isolation
tests from `tests/test_verified_image_sink.py` with renamed imports/functions.

- [ ] **Step 2: Write failing web-research cutover tests**

Replace verifier fixtures with confidence-bearing Brave payloads and assert:

```python
@pytest.mark.asyncio
async def test_web_research_offers_provider_selected_images_without_a_verifier():
    brave = _FakeTool("brave_image_search", _brave_payload(confidence="high"))
    tool = create_web_research_tool(
        tavily_tool=_FakeTool("tavily_search", TAVILY_PAYLOAD),
        brave_tool=brave,
    )

    with selected_image_sink() as sink:
        await tool.ainvoke({"query": "T1 roster", "image_query": "T1 team photo"})

    assert sink
    assert brave.calls == [{"query": "T1 team photo"}]
```

Add a signature regression:

```python
def test_web_research_has_no_verifier_dependencies():
    parameters = inspect.signature(create_web_research_tool).parameters
    assert "verifier_model" not in parameters
    assert "web_image_service" not in parameters
    assert "recorder" not in parameters
```

- [ ] **Step 3: Run cutover tests and confirm RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_selected_image_sink.py tests/test_web_research_tool.py -k "selected or verifier_dependencies"
```

Expected: collection/import failures because the selected sink and simplified tool signature do not exist.

- [ ] **Step 4: Implement and wire the selected sink**

Create `selected_image_sink.py` as the same narrow `ContextVar` channel with
renamed identifiers:

```python
_sink: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "selected_image_sink", default=None
)


@contextmanager
def selected_image_sink() -> Iterator[list[dict[str, Any]]]:
    collected: list[dict[str, Any]] = []
    token = _sink.set(collected)
    try:
        yield collected
    finally:
        _sink.reset(token)


def offer_selected_images(candidates: Sequence[Mapping[str, Any]]) -> None:
    sink = _sink.get()
    if sink is not None:
        sink.extend(dict(candidate) for candidate in candidates)
```

Update `tool_execution.py` to open `selected_image_sink()` around internal tool
execution. Rename `_attach_rich_candidates_to_artifact(...,
verified_images=None)` to `_attach_rich_candidates_to_artifact(...,
selected_images=None)` and port its artifact-attachment regression to the new
sink test. Delete the old sink module/test only after every import is migrated.

- [ ] **Step 5: Simplify `web_research` image orchestration**

Remove verifier/image-service/recorder parameters and imports. Resolve only the
Brave tool, then call:

```python
return await discover_images(
    brave_tool=brave_tool,
    image_query=image_query,
    image_intent=image_intent,
)
```

Offer successful results with `offer_selected_images(selected)`. Rename local
variables from `approved` to `selected`. Keep concurrent task creation,
explicit opt-out, budget reuse, and text-only error isolation unchanged.

Update `base_agent.py` to call `create_web_research_tool()` with no usage
recorder and replace verifier-history comments with the current provider split.

- [ ] **Step 6: Rename the image feature gate**

Add the replacement setting alongside the old setting for this cutover commit:

```python
remote_image_enrichment_enabled: bool = Field(
    default=True,
    description="Enable Brave-backed remote image enrichment for rich responses.",
)
```

Read it in `_image_path_open` and update focused tests to monkeypatch the new
name. Task 6 removes the now-unused old setting and the remaining verifier-only
configuration.

- [ ] **Step 7: Run focused tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_selected_image_sink.py tests/test_web_research_tool.py tests/test_web_research_dependency_resolution.py tests/test_web_research_binding.py
.\.venv\Scripts\python.exe -m ruff check app/ai/selected_image_sink.py app/ai/web_research_tool.py app/ai/tool_execution.py app/ai/agents/base_agent.py app/core/config.py
```

Expected: PASS.

Commit:

```powershell
git add app/ai/selected_image_sink.py app/ai/web_research_tool.py app/ai/tool_execution.py app/ai/agents/base_agent.py app/core/config.py tests/test_selected_image_sink.py tests/test_web_research_tool.py tests/test_web_research_dependency_resolution.py tests/test_web_research_binding.py
git rm app/ai/verified_image_sink.py tests/test_verified_image_sink.py
git commit -m "refactor: use provider-selected image discovery"
```

---

### Task 4: Align Tavily search with its retrieval role

**Files:**
- Modify: `app/ai/mcp_servers/tavily_server.py`
- Modify: `tests/test_tavily_server.py`

**Interfaces:**
- Consumes: Tavily Python SDK `TavilyClient.search`.
- Produces:
  `tavily_search(query, max_results=None, search_depth=None, include_raw_content=False, auto_parameters=None, topic=None, time_range=None) -> str`
  with default `include_answer=False`, direct SDK/API `timeout=10`, compact
  published-date results, canonical URL deduplication, and bounded error
  retryability.

- [ ] **Step 1: Write failing default-request tests**

Replace the current generated-answer expectation:

```python
def test_search_requests_ranked_sources_without_provider_answer(monkeypatch):
    client = _FakeTavilyClient({"results": []})
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)

    tavily_server.tavily_search("openai news")

    params = client.calls[0]
    assert params["search_depth"] == "basic"
    assert params["max_results"] == 5
    assert params["include_answer"] is False
    assert params["include_raw_content"] is False
    assert params["include_usage"] is True
    assert params["timeout"] == 10
    assert "include_images" not in params
```

Add a source scan contract proving there is no `asyncio.timeout`,
`asyncio.wait_for`, or Tavily search timeout setting.

- [ ] **Step 2: Write failing topic, time, and publication-date tests**

```python
def test_topic_and_time_range_are_forwarded(monkeypatch):
    client = _FakeTavilyClient({"results": []})
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)

    tavily_server.tavily_search(
        "latest T1 results", topic="news", time_range="week"
    )

    assert client.calls[0]["topic"] == "news"
    assert client.calls[0]["time_range"] == "week"


def test_news_publication_date_survives_normalization(monkeypatch):
    client = _FakeTavilyClient(
        {
            "results": [
                {
                    "title": "T1 wins",
                    "url": "https://news.example/t1",
                    "content": "Result",
                    "score": 0.9,
                    "published_date": "2026-08-09",
                }
            ]
        }
    )
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)

    payload = json.loads(tavily_server.tavily_search("T1", topic="news"))

    assert payload["results"][0]["published_date"] == "2026-08-09"
```

- [ ] **Step 3: Write failing canonical deduplication test**

```python
def test_duplicate_urls_merge_unique_chunks_and_keep_first_rank():
    payload = _normalize_search_response(
        query="T1 roster",
        response={
            "results": [
                {
                    "title": "First",
                    "url": "https://EXAMPLE.com/team/?utm_source=x#roster",
                    "content": "A [...] B",
                    "score": 0.9,
                },
                {
                    "title": "Duplicate",
                    "url": "https://example.com/team",
                    "content": "B [...] C",
                    "score": 0.8,
                },
            ]
        },
    )

    assert len(payload["results"]) == 1
    assert payload["results"][0]["title"] == "First"
    assert payload["results"][0]["url"].startswith("https://EXAMPLE.com")
    assert payload["results"][0]["content"] == "A [...] B [...] C"
```

- [ ] **Step 4: Write failing SDK error tests**

Patch the fake client to raise Tavily SDK errors:

```python
@pytest.mark.parametrize(
    ("exc", "retryable"),
    [
        (tavily_errors.TimeoutError(10), True),
        (tavily_errors.UsageLimitExceededError("rate limited"), True),
        (tavily_errors.BadRequestError("bad query"), False),
        (tavily_errors.InvalidAPIKeyError("bad key"), False),
        (tavily_errors.ForbiddenError("plan"), False),
    ],
)
def test_search_errors_have_bounded_retryability(monkeypatch, exc, retryable):
    monkeypatch.setattr(tavily_server, "_make_client", lambda: _RaisingClient(exc))
    payload = json.loads(tavily_server.tavily_search("T1"))
    assert payload["retryable"] is retryable
```

An unclassified `requests.HTTPError` with a 5xx response is retryable; other
unclassified exceptions remain non-retryable.

- [ ] **Step 5: Run Tavily tests and confirm RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_tavily_server.py
```

Expected: failures because the adapter requests an answer, omits timeout/topic/time, drops publication dates, and does not deduplicate.

- [ ] **Step 6: Implement the request contract**

Add bounded choices:

```python
SUPPORTED_TOPICS = {"general", "news", "finance"}
SUPPORTED_TIME_RANGES = {"day", "week", "month", "year"}
```

Add `_optional_choice(value, allowed, name)` that returns `None` for omitted
values and raises `ValueError(f"Unsupported {name} {value!r}")` for an explicit
unsupported value. Convert that validation error through `_error` without
calling Tavily.

Extend `tavily_search` and construct parameters with:

```python
params = {
    "query": cleaned_query,
    "max_results": result_count,
    "include_answer": False,
    "include_raw_content": include_raw_content,
    "auto_parameters": resolved["auto_parameters"],
    "include_usage": True,
    "timeout": 10,
    "topic": _optional_choice(
        topic, allowed=SUPPORTED_TOPICS, name="topic"
    ) or "general",
}
if time_range is not None:
    params["time_range"] = _optional_choice(
        time_range, allowed=SUPPORTED_TIME_RANGES, name="time_range"
    )
```

Do not add a settings field or local timeout wrapper.

- [ ] **Step 7: Implement normalization and error mapping**

Add the canonical key:

```python
def _canonical_result_key(url: str) -> str:
    parsed = urlsplit(str(url or "").strip())
    host = str(parsed.hostname or "").lower()
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), host, path, "", ""))
```

Normalize in provider order with a `key -> result` map. For a duplicate, split
both content strings on `"[...]"`, strip chunks, append only unseen chunks, and
join them with `" [...] "`. Keep the first title, URL, score, index, and
publication date. Copy `published_date` when present.

Catch the installed SDK's `tavily.errors` types explicitly. Keep returned error
messages bounded and free of secrets. Treat `UsageLimitExceededError` as
retryable because SDK 0.7.x maps HTTP 429 to it; treat bad request, key, plan,
and forbidden errors as non-retryable.

Use this classifier:

```python
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
        return ("upstream", True) if status >= 500 else ("provider_error", False)
    return "provider_error", False
```

- [ ] **Step 8: Run tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_tavily_server.py tests/test_research_payload_regression.py
.\.venv\Scripts\python.exe -m ruff check app/ai/mcp_servers/tavily_server.py tests/test_tavily_server.py
```

Expected: PASS.

Commit:

```powershell
git add app/ai/mcp_servers/tavily_server.py tests/test_tavily_server.py tests/test_research_payload_regression.py
git commit -m "fix: configure Tavily for ranked source retrieval"
```

---

### Task 5: Expose Tavily topic and recency through `web_research`

**Files:**
- Modify: `app/ai/web_research_tool.py`
- Modify: `app/ai/prompts.py`
- Modify: `tests/test_web_research_tool.py`
- Modify: `tests/test_prompts_media_capability.py`
- Modify: `tests/test_rich_response_prompt_inventory.py`

**Interfaces:**
- Consumes: Tavily `topic` and `time_range` parameters.
- Produces: `WebResearchInput.topic: Literal["general", "news", "finance"] | None`; `WebResearchInput.time_range: Literal["day", "week", "month", "year"] | None`; structured Tavily errors remain tool errors rather than successful research payloads.

- [ ] **Step 1: Write failing schema and forwarding tests**

```python
@pytest.mark.asyncio
async def test_news_topic_and_time_range_reach_tavily():
    tavily = _FakeTool("tavily_search", TAVILY_PAYLOAD)
    await _run(
        create_web_research_tool(tavily_tool=tavily),
        query="latest T1 match",
        topic="news",
        time_range="week",
        skip_images=True,
    )
    assert tavily.calls == [
        {
            "query": "latest T1 match",
            "topic": "news",
            "time_range": "week",
        }
    ]
```

Add input-validation tests that reject unsupported literals before calling the provider.

- [ ] **Step 2: Write a failing structured-error regression**

```python
@pytest.mark.asyncio
async def test_tavily_error_is_not_reported_as_successful_research():
    tavily = _FakeTool(
        "tavily_search",
        json.dumps(
            {
                "provider": "tavily",
                "operation": "search",
                "error": "rate limited",
                "retryable": True,
            }
        ),
    )
    payload, _ = await _run(
        create_web_research_tool(tavily_tool=tavily),
        query="latest T1 match",
        skip_images=True,
    )
    assert payload["status"] == "error"
    assert payload["retryable"] is True
    assert "research" not in payload
```

- [ ] **Step 3: Run focused tests and confirm RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_web_research_tool.py -k "topic or time_range or tavily_error"
```

Expected: schema/forwarding failures and a successful-looking error payload.

- [ ] **Step 4: Extend the combined tool schema and forwarding**

Add literal fields with concise descriptions. Extend `_research` and
`_run_search` signatures. Only include keys in Tavily arguments when the caller
provided them:

```python
if topic is not None:
    args["topic"] = topic
if time_range is not None:
    args["time_range"] = time_range
```

After `provider_result_text`, parse the object. If it contains `error`, raise a
small internal provider exception carrying the bounded message and retryability;
convert it through `_error_payload(message, retryable=retryable)`.

```python
class ResearchProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        self.retryable = bool(retryable)
        super().__init__(str(message or "Research provider failed")[:300])


def _raise_for_provider_error(result_text: str) -> str:
    try:
        payload = json.loads(result_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return result_text
    if isinstance(payload, dict) and payload.get("error"):
        raise ResearchProviderError(
            str(payload["error"]), retryable=bool(payload.get("retryable"))
        )
    return result_text
```

Catch `ResearchProviderError` before the generic search exception and pass its
`retryable` value to `_error_payload`.

- [ ] **Step 5: Update model guidance without adding a routing taxonomy**

Add only:

```text
For current events, use topic="news" and add time_range only when the requested
recency is clear. Use topic="finance" for market and company financial news.
Leave both unset for general factual research.
```

Keep image-query, marker, no-invention, and text-independence rules unchanged.

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_web_research_tool.py tests/test_prompts_media_capability.py tests/test_rich_response_prompt_inventory.py
.\.venv\Scripts\python.exe -m ruff check app/ai/web_research_tool.py app/ai/prompts.py
```

Expected: PASS.

Commit:

```powershell
git add app/ai/web_research_tool.py app/ai/prompts.py tests/test_web_research_tool.py tests/test_prompts_media_capability.py tests/test_rich_response_prompt_inventory.py
git commit -m "feat: expose Tavily topic and recency controls"
```

---

### Task 6: Remove verifier-only infrastructure and configuration

**Files:**
- Delete: `app/ai/image_verification_flow.py`
- Delete: `app/ai/visual_verifier.py`
- Delete: `app/services/thumbnail_batch.py`
- Delete: `app/services/verified_image_bytes.py`
- Delete: `tests/test_image_verification_flow_metrics.py`
- Delete: `tests/test_visual_verification_metrics.py`
- Delete: `tests/test_visual_verifier.py`
- Delete: `tests/test_thumbnail_batch.py`
- Delete: `tests/test_verified_image_bytes.py`
- Modify: `app/core/config.py`
- Modify: `.env.example`
- Modify: `app/services/message_service.py`
- Modify: `tests/test_web_image_byte_cache.py`
- Modify: `app/observability/rich_images.py`
- Modify: `tests/test_rich_image_metrics.py`
- Create: `tests/test_remote_image_enrichment_config.py`

**Interfaces:**
- Consumes: the completed provider-native discovery cutover.
- Produces: `Settings.remote_image_enrichment_enabled: bool = True`; no verifier settings/modules/calls; discovery terminal metric `record_discovery_outcome(outcome, duration_seconds)` with bounded outcomes.

- [ ] **Step 1: Write a failing removal contract**

```python
def test_remote_image_enrichment_replaces_visual_verification_settings():
    fields = Settings.model_fields
    assert fields["remote_image_enrichment_enabled"].default is True
    for removed in (
        "vision_image_verification_enabled",
        "image_verification_model",
        "image_verification_media_resolution",
        "image_verification_thinking_level",
        "image_verification_confidence_threshold",
        "image_verification_max_candidates",
        "image_verification_timeout_seconds",
        "image_verification_thumbnail_timeout_seconds",
        "verified_image_cache_max_bytes",
    ):
        assert removed not in fields
```

Add a filesystem/import inventory test asserting the four deleted production
modules do not exist and no application Python source imports their symbols.

- [ ] **Step 2: Write a failing obsolete-metric removal test**

```python
def test_discovery_outcomes_are_bounded_and_content_free():
    metrics = RichImageMetrics(registry=CollectorRegistry())
    metrics.record_discovery_outcome(outcome="selected", duration_seconds=0.1)
    metrics.record_discovery_outcome(
        outcome="https://tenant.example/private", duration_seconds=0.2
    )
    body = metrics.render().decode()
    assert "rich_image_discovery_outcome_total" in body
    assert 'outcome="selected"' in body
    assert 'outcome="other"' in body
    assert "tenant.example" not in body
    assert "rich_image_verification" not in body
```

The discovery collectors already exist from Task 2; this test is RED because
the obsolete verification collectors still render alongside them.

- [ ] **Step 3: Run removal tests and confirm RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_remote_image_enrichment_config.py tests/test_rich_image_metrics.py -k "enrichment or discovery_outcomes"
```

Expected: failures because old settings and verifier metrics still exist.

- [ ] **Step 4: Replace configuration and environment contract**

Retain `remote_image_enrichment_enabled` from Task 3 and remove the entire old
verifier settings block.

Remove the verifier/cache environment variables from `.env.example` and add
`REMOTE_IMAGE_ENRICHMENT_ENABLED=true` beside Brave configuration.

- [ ] **Step 5: Remove verifier-only byte hand-off**

In `message_service.py`, remove `take_verified_bytes` and always call
`web_image_service.register` without passing its optional `cached` argument.
Retain `WebImageService.register(cached: FetchedWebImage | None = None)` and its
generic cache tests because
the protected-image service supports already-owned bytes independently; remove
only the final end-to-end verifier hand-off test and rewrite stale verifier
comments in `tests/test_web_image_byte_cache.py`.

- [ ] **Step 6: Remove verifier telemetry**

Retain the discovery outcome metrics introduced in Task 2. Delete only the
obsolete verification stage/outcome collectors, constants, and methods, then
assert no `rich_image_verification` series remains.

- [ ] **Step 7: Delete obsolete modules and tests**

Use `git rm` for the exact files listed in this task only after the removal
contract passes against all new imports. Do not delete the generic
`WebImageService` cached-byte support or database columns.

- [ ] **Step 8: Run focused tests and stale-reference scan**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_remote_image_enrichment_config.py tests/test_image_discovery_flow.py tests/test_rich_image_metrics.py tests/test_web_image_byte_cache.py tests/test_message_service_web_image_externalization.py
rg -n "visual_verifier|image_verification_flow|thumbnail_batch|verified_image_bytes|vision_image_verification_enabled|image_verification_" app tests .env.example -g "*.py" -g ".env.example"
```

Expected: pytest PASS; `rg` returns no matches.

- [ ] **Step 9: Commit cleanup**

```powershell
git add app/core/config.py .env.example app/services/message_service.py app/observability/rich_images.py app/ai/image_discovery_flow.py tests/test_remote_image_enrichment_config.py tests/test_rich_image_metrics.py tests/test_web_image_byte_cache.py
git rm app/ai/image_verification_flow.py app/ai/visual_verifier.py app/services/thumbnail_batch.py app/services/verified_image_bytes.py tests/test_image_verification_flow_metrics.py tests/test_visual_verification_metrics.py tests/test_visual_verifier.py tests/test_thumbnail_batch.py tests/test_verified_image_bytes.py
git commit -m "refactor: remove visual verification infrastructure"
```

---

### Task 7: Prove the complete provider-native pipeline and update docs

**Files:**
- Create: `tests/test_selected_image_reaches_the_model.py`
- Create: `tests/test_provider_selected_image_injection.py`
- Delete: `tests/test_verified_image_reaches_the_model.py`
- Delete: `tests/test_vision_verified_injection_regression.py`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-04-vision-verified-image-injection-design.md`
- Modify: `docs/superpowers/specs/2026-08-07-automatic-visual-enrichment-design.md`

**Interfaces:**
- Consumes: simplified Tavily adapter, Brave adapter, deterministic discovery, selected-image sink, rich-item inventory, protected registration.
- Produces: end-to-end proof that provider-selected images reach the model without an image LLM and factual research survives independent Brave failure.

- [ ] **Step 1: Replace the end-to-end image regression**

Port the useful inventory/marker assertions from
`test_verified_image_reaches_the_model.py`, but use a confidence-bearing Brave
payload and no image service/verifier fixture:

```python
@pytest.mark.asyncio
async def test_a_provider_selected_image_becomes_a_marker_the_model_can_copy():
    tool = create_web_research_tool(
        tavily_tool=_Tavily(),
        brave_tool=_Brave(confidence="high"),
    )
    with selected_image_sink() as sink:
        await tool.ainvoke({"query": "T1 roster 2026", "image_query": "T1 team photo"})

    artifact: dict = {}
    _attach_rich_candidates_to_artifact(
        artifact,
        raw_result=None,
        result_text="{}",
        render=None,
        tool_call_id="call-1",
        tool_name="web_research",
        selected_images=sink,
    )
    context: dict = {}
    ToolLoopMixin._lift_rich_candidates(context, [artifact])
    apply_rich_image_selection(context)
    assert context["rich_item_candidates"][0]["source"] == "image_search"
```

Retain privacy assertions that public metadata contains no original image URL.

- [ ] **Step 2: Replace the injection regressions**

Port and rename the behavior tests from
`test_vision_verified_injection_regression.py`:

- high confidence selected;
- high tier prevents medium mixing;
- medium fallback when high is absent;
- explicit opt-out skips Brave;
- disabled `remote_image_enrichment_enabled` skips Brave;
- Brave structured failure leaves Tavily text successful.

Delete verifier decisions, portrait-kind heuristics, and confidence-threshold fixtures.

- [ ] **Step 3: Run replacements before deleting old files**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_selected_image_reaches_the_model.py tests/test_provider_selected_image_injection.py tests/test_message_service_web_image_externalization.py tests/test_web_images_api.py
```

Expected: PASS.

- [ ] **Step 4: Update current documentation**

In `README.md`:

- describe Brave native confidence and proxied thumbnail selection;
- describe Tavily as ranked source retrieval with no provider answer by default;
- list `topic` and `time_range` on `web_research`;
- replace the verifier flag/settings with `REMOTE_IMAGE_ENRICHMENT_ENABLED`;
- state that `timeout=10` is passed to the Tavily SDK/API and has no environment setting.

Add a prominent `Superseded on 2026-08-10` note to each older active design,
linking to
`2026-08-10-provider-native-retrieval-simplification-design.md`. Do not rewrite
historical implementation plans.

- [ ] **Step 5: Delete superseded tests and scan active documentation**

```powershell
git rm tests/test_verified_image_reaches_the_model.py tests/test_vision_verified_injection_regression.py
rg -n "vision verifier|visual verification|IMAGE_VERIFICATION_|VISION_IMAGE_VERIFICATION" README.md .env.example app tests
```

Expected: no current-contract matches; historical plan files are intentionally excluded.

- [ ] **Step 6: Run focused integration and privacy suites**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_brave_image_search_server.py tests/test_tavily_server.py tests/test_image_discovery_flow.py tests/test_web_research_tool.py tests/test_web_research_dependency_resolution.py tests/test_selected_image_sink.py tests/test_selected_image_reaches_the_model.py tests/test_provider_selected_image_injection.py tests/test_rich_response_sources.py tests/test_message_service_web_image_externalization.py tests/test_web_image_service.py tests/test_web_images_api.py
```

Expected: PASS.

- [ ] **Step 7: Run full verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
git diff --check
```

Expected: complete test suite passes with only established skips, Ruff reports no violations, and diff check is clean.

- [ ] **Step 8: Review the final diff against the spec**

Verify each acceptance criterion in
`docs/superpowers/specs/2026-08-10-provider-native-retrieval-simplification-design.md`:

- no image-specific LLM call or supporting module;
- no default Tavily-generated answer;
- high-confidence Brave selection with medium-only fallback;
- thumbnail-only Brave result support;
- Tavily topic/time forwarding and publication dates;
- distinguishable provider errors;
- direct Tavily SDK/API `timeout=10`, with no local wrapper/setting;
- concurrent optional providers;
- protected render-time validation.

- [ ] **Step 9: Commit integration and documentation**

```powershell
git add README.md docs/superpowers/specs/2026-08-04-vision-verified-image-injection-design.md docs/superpowers/specs/2026-08-07-automatic-visual-enrichment-design.md tests/test_selected_image_reaches_the_model.py tests/test_provider_selected_image_injection.py
git rm tests/test_verified_image_reaches_the_model.py tests/test_vision_verified_injection_regression.py
git commit -m "docs: document provider-native retrieval pipeline"
```
