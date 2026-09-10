# Production Web Research and Image Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the competing web and image paths with one provider-neutral research pipeline that produces verified clickable sources and lets the existing answer-model call inspect and select validated images.

**Architecture:** A turn-scoped `WebResearchSession` owns structural budgets, source IDs, validated image candidates, and provider-neutral evidence. Thin product tools call that session; specialist middleware rebuilds evidence after every tool step and attaches actual low-detail candidates to vision-capable answer models. One server-owned grounding resolver validates source and image tokens for streaming and persistence, while both Streamlit and AI SDK projections consume the same canonical source records.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, LangChain/LangGraph agent middleware, SQLAlchemy/PostgreSQL, httpx, Pillow, Streamlit, Vercel AI SDK v6 SSE, pytest/pytest-asyncio, Ruff.

## Global Constraints

- The approved specification is `docs/superpowers/specs/2026-09-10-production-web-research-and-image-grounding-design.md`.
- Research has no fixed end-to-end wall-clock deadline and no speculative p95 release threshold.
- `quick` permits one search, five admitted results, and two page opens; `agentic` permits three materially distinct searches and eight unique public sources.
- Ordinary visual research offers at most four validated candidates; explicit gallery research offers at most six.
- Bound work through calls, sources, page opens, candidates, bytes, redirects, content types, dimensions, and decompression—not elapsed total research time.
- Retain configurable connection and idle-read liveness controls; valid progress must not be cancelled merely because total elapsed time crossed a threshold.
- Add no classifier, reviewer, or separate vision-model call to the normal runtime path.
- Use `[[source:S1]]` and `[[image:I1]]` as model-authored tokens; models never author public URLs or final rich-item markers.
- Never offer metadata-only image selection. If the active answer model lacks vision support, emit `answer_model_not_vision_capable` and answer without an image.
- HTTPS is required for production web-image retrieval, including every redirect; existing SSRF, byte, MIME, dimension, and decompression protections remain mandatory.
- Unselected candidates, private upstream image URLs, provider payloads, credentials, and image bytes never enter public message metadata.
- Legacy message formats may remain in an isolated read-only compatibility projection, but no legacy format remains an active write path.
- Tests and production code contain no topic-specific routing or image-selection rules derived from reported examples.
- Use `C:\Users\ADMIN\miniconda3\envs\agents\python.exe` for repository verification in this workspace.

---

## File and ownership map

Create the focused package `app/ai/web_research/`:

- `contracts.py`: provider-neutral public and private Pydantic contracts.
- `source_registry.py`: canonical URL normalization, turn-scoped source IDs, and deduplication.
- `policy.py`: research requirement and structural limit decisions.
- `providers.py`: provider protocols, MCP resolver, Tavily/Brave adapters, and normalized failures.
- `service.py`: retry/fallback/circuit/cache orchestration and turn-scoped `WebResearchSession`.
- `model_context.py`: compact evidence serialization and multimodal candidate injection.
- `grounding.py`: whole-answer and incremental source/image token resolution.

Keep these existing responsibilities:

- `app/services/web_image_service.py`: secure image fetching, decoding, and protected reference registration.
- `app/core/rich_response.py`: public rich-item schemas and marker grammar.
- `app/services/event_streaming/*`: canonical-to-client transport projection.
- `app/core/response_constants.py`: terminal public message metadata construction.

Do not add a database table for sources in this project. Public sources are versioned assistant-message metadata; protected web images continue to use `web_image_references`.

---

### Task 1: Provider-neutral contracts and turn-scoped source registry

**Files:**
- Create: `app/ai/web_research/__init__.py`
- Create: `app/ai/web_research/contracts.py`
- Create: `app/ai/web_research/source_registry.py`
- Create: `tests/test_web_research_contracts.py`
- Create: `tests/test_web_source_registry.py`

**Interfaces:**
- Produces: `ResearchMode`, `VisualIntent`, `ResearchRequest`, `ProviderSource`, `SourceRecord`, `ProviderImageCandidate`, `ImageCandidateRecord`, `ResearchFailure`, `WebEvidenceBundle`, and `ResearchScope`.
- Produces: `canonicalize_public_url(url: str) -> str | None` and `SourceRegistry.admit(candidates: Sequence[ProviderSource]) -> tuple[SourceRecord, ...]`.
- Depends only on Pydantic and the standard library.

- [ ] **Step 1: Write failing contract and registry tests**

```python
def test_registry_deduplicates_tracking_variants_and_keeps_first_rank():
    registry = SourceRegistry()
    admitted = registry.admit([
        ProviderSource(provider="a", url="https://EXAMPLE.com/x/?utm_source=n#top", title="First", snippet="one", rank=1),
        ProviderSource(provider="b", url="https://example.com/x", title="Second", snippet="two", rank=2),
    ])
    assert [item.source_id for item in admitted] == ["S1"]
    assert str(admitted[0].url) == "https://example.com/x"
    assert admitted[0].title == "First"


def test_bundle_rejects_an_image_without_a_source_record():
    with pytest.raises(ValidationError):
        WebEvidenceBundle(
            mode="quick",
            sources=(),
            images=(ImageCandidateRecord(candidate_id="I1", source_id="S9", delivery_url="/web-images/x", mime_type="image/jpeg", width=640, height=360),),
        )
```

- [ ] **Step 2: Run the new tests and verify missing-module failures**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_contracts.py tests/test_web_source_registry.py -q`

Expected: FAIL because `app.ai.web_research` does not exist.

- [ ] **Step 3: Add the typed contracts**

```python
ResearchMode = Literal["none", "quick", "agentic"]
VisualIntent = Literal["none", "figure", "comparison", "gallery"]


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query: str = Field(min_length=3, max_length=400)
    objective: str = Field(min_length=3, max_length=500)
    mode: ResearchMode = "quick"
    freshness: Freshness = "timeless"
    start_date: date | None = None
    end_date: date | None = None
    locale: str | None = Field(default=None, max_length=32)
    include_domains: tuple[str, ...] = ()
    visual_intent: VisualIntent = "none"
    image_query: str | None = Field(default=None, max_length=300)


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    url: AnyHttpUrl
    title: str | None = None
    snippet: str | None = None
    status: Literal["search_result", "opened", "snippet_only"]
    published_at: datetime | None = None
    provider: str
    query_index: int = Field(ge=1)
```

Define the remaining contracts with these exact fields and the same `extra="forbid"`/frozen policy:

- `ResearchScope`: `conversation_id`, `user_id`, `logical_turn_id`, and optional `device_id`.
- `ProviderSource`: `provider`, `url`, optional `title`, optional `snippet`, positive `rank`, positive `query_index`, and optional `published_at`.
- `ProviderImageCandidate`: `provider`, `image_url`, `source_url`, optional `title`/`description`, optional positive `width`/`height`, positive `rank`, and optional `published_at`.
- `ImageCandidateRecord`: `candidate_id`, `source_id`, protected `delivery_url`, `mime_type`, positive `width`/`height`, SHA-256 `digest`, optional `title`/`description`, and `provider`.
- `ResearchFailure`: `operation`, `provider`, bounded `code`, `retryable`, and optional `source_id`; it contains no exception text.
- `WebEvidenceBundle`: `status`, `mode`, positive `query_index`, tuples of `sources`, `images`, `failures`, and `providers_used`, plus `reused`, `omitted_source_count`, and `omitted_image_count`.

Add a bundle validator that requires every `ImageCandidateRecord.source_id` to exist in `sources`, unique source/candidate IDs, and cardinality at or below the mode/visual-intent limits.

- [ ] **Step 4: Implement canonicalization and stable admission**

```python
class SourceRegistry:
    def __init__(self, *, max_sources: int = 8) -> None:
        self._max_sources = max_sources
        self._by_url: OrderedDict[str, SourceRecord] = OrderedDict()

    def admit(self, candidates: Sequence[ProviderSource]) -> tuple[SourceRecord, ...]:
        for candidate in candidates:
            url = canonicalize_public_url(candidate.url)
            if url is None or url in self._by_url or len(self._by_url) >= self._max_sources:
                continue
            source_id = f"S{len(self._by_url) + 1}"
            self._by_url[url] = SourceRecord(
                source_id=source_id,
                url=url,
                title=candidate.title,
                snippet=candidate.snippet,
                status="search_result",
                published_at=candidate.published_at,
                provider=candidate.provider,
                query_index=candidate.query_index,
            )
        return tuple(self._by_url.values())
```

Canonicalization must lowercase/IDNA-normalize the host, remove fragments and default ports, normalize the empty path to `/`, sort retained query parameters, and discard known tracking parameters without widening a domain restriction.

- [ ] **Step 5: Run focused tests and commit**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_contracts.py tests/test_web_source_registry.py -q`

Expected: PASS.

```powershell
git add app/ai/web_research tests/test_web_research_contracts.py tests/test_web_source_registry.py
git commit -m "feat: add canonical web research contracts"
```

---

### Task 2: Research requirement policy and structural accounting

**Files:**
- Create: `app/ai/web_research/policy.py`
- Modify: `app/ai/research_budget.py:62-520`
- Modify: `app/ai/workflow/contracts.py:104-116`
- Modify: `app/ai/workflow/routing.py:535-565,710-890`
- Test: `tests/test_research_budget.py`
- Create: `tests/test_web_research_policy.py`
- Modify: `tests/test_production_workflow_graph.py`

**Interfaces:**
- Consumes: `ResearchMode`, `ResearchRequest`, and `RoutingDecision`.
- Produces: `ResearchLimits.for_mode(mode: ResearchMode) -> ResearchLimits`.
- Produces: `ResearchRequirementPolicy.apply(decision: RoutingDecision, user_text: str) -> RoutingDecision`.
- Extends: `ResearchBudget.reserve_page_open(urls)`, `admit_sources(count)`, and schema-v2 continuation state.

- [ ] **Step 1: Add failing policy and accounting tests**

```python
def test_explicit_verification_forces_quick_without_a_new_model_call():
    decision = RoutingDecision(agent_id="chat_agent", confidence=0.8, reason="general", requires_web=False)
    revised = ResearchRequirementPolicy().apply(decision, "Please verify the current package version")
    assert revised.requires_web is True
    assert revised.research_mode == "quick"


def test_quick_budget_is_structural_not_elapsed_time():
    budget = ResearchBudget.for_mode("quick")
    assert budget.reserve_search("one", scope=("general",)) is True
    assert budget.reserve_search("two", scope=("general",)) is False
    assert budget.reserve_page_open(["https://a.test", "https://b.test"]) == 2
    assert budget.reserve_page_open(["https://c.test"]) == 0
    assert not hasattr(budget, "deadline")
```

- [ ] **Step 2: Run the focused tests and verify failures**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_policy.py tests/test_research_budget.py -q`

Expected: FAIL because the policy, routing fields, and page/source counters are absent.

- [ ] **Step 3: Extend the routing decision and router instruction**

```python
class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    agent_id: str = Field(min_length=1, max_length=160)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=500)
    requires_web: bool = False
    research_mode: Literal["none", "quick", "agentic"] = "none"
```

Update `ROUTER_SYSTEM_PROMPT` to classify meaning across languages, require `quick` for current/external verification, reserve `agentic` for comparisons or evidence synthesis, and never use confidence to control execution. Apply `ResearchRequirementPolicy` after parsing and before `RoutingDecisionValidator.validate` so explicit browse/verify requests and unambiguous temporal categories cannot be downgraded by router output.

- [ ] **Step 4: Evolve research accounting without losing Continue compatibility**

Bump `RESEARCH_ACCOUNTING_SCHEMA_VERSION` to `2`. Serialize `mode`, `page_urls`, and admitted-source count. `research_budget_from_state` must accept version 1 by assigning its existing search/image history and defaulting new counters to zero; unknown versions still raise `ResearchAccountingUnreadable`.

Use `ResearchLimits` constants:

```python
_LIMITS = {
    "none": ResearchLimits(searches=0, sources=0, page_opens=0, model_images=0),
    "quick": ResearchLimits(searches=1, sources=5, page_opens=2, model_images=4),
    "agentic": ResearchLimits(searches=3, sources=8, page_opens=4, model_images=4),
}
```

Gallery raises only `model_images` to six; it does not raise text-search or page-open limits.

- [ ] **Step 5: Run routing/accounting tests and commit**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_policy.py tests/test_research_budget.py tests/test_production_workflow_graph.py -q`

Expected: PASS.

```powershell
git add app/ai/web_research/policy.py app/ai/research_budget.py app/ai/workflow/contracts.py app/ai/workflow/routing.py tests/test_web_research_policy.py tests/test_research_budget.py tests/test_production_workflow_graph.py
git commit -m "feat: enforce structural web research policy"
```

---

### Task 3: Provider adapters and normalized failure semantics

**Files:**
- Create: `app/ai/web_research/providers.py`
- Create: `tests/fixtures/web_research/tavily_search_success.json`
- Create: `tests/fixtures/web_research/tavily_extract_partial.json`
- Create: `tests/fixtures/web_research/brave_images_success.json`
- Create: `tests/test_web_research_providers.py`
- Modify: `app/ai/web_query_contract.py:103-185`

**Interfaces:**
- Consumes: normalized query/date/domain helpers from `app.ai.web_query_contract`.
- Produces: `TextSearchProvider.search(request)`, `PageOpenProvider.open(request)`, `ImageSearchProvider.search(request)`, and `ProviderResolver.resolve(kind, exclude=())`.
- Produces: `ProviderFailure(code, retryable, provider)` with bounded public codes.

- [ ] **Step 1: Record sanitized provider fixtures and failing adapter tests**

```python
@pytest.mark.asyncio
async def test_tavily_adapter_drops_raw_content_and_assigns_query_index(fixture_json):
    tool = FakeMcpTool(fixture_json("tavily_search_success.json"))
    records = await TavilyTextSearchProvider(tool).search(NORMALIZED_REQUEST, query_index=1)
    assert records[0].provider == "tavily"
    assert records[0].query_index == 1
    assert "raw_content" not in records[0].model_dump_json()


@pytest.mark.asyncio
async def test_provider_cancellation_is_not_normalized_as_retryable_failure():
    task = asyncio.create_task(TavilyTextSearchProvider(BlockingTool()).search(NORMALIZED_REQUEST, query_index=1))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
```

- [ ] **Step 2: Run provider tests and verify failures**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_providers.py -q`

Expected: FAIL because the adapter module does not exist.

- [ ] **Step 3: Implement protocols and MCP-backed adapters**

```python
class TextSearchProvider(Protocol):
    name: str
    async def search(self, request: NormalizedWebSearch, *, query_index: int) -> tuple[ProviderSource, ...]:
        raise NotImplementedError


class ProviderFailure(RuntimeError):
    def __init__(self, code: str, *, provider: str, retryable: bool) -> None:
        self.code = code
        self.provider = provider
        self.retryable = retryable
        super().__init__(code)
```

Move provider payload parsing and MCP resolution out of `web_tools.py`. Preserve `normalize_web_search`, stale-year repair, focused excerpt selection, and bounded failure projection. Convert authentication/schema/policy failures to non-retryable codes and transport/`429`/`5xx` failures to retryable codes. Re-raise `CancelledError`, `KeyboardInterrupt`, and LangGraph control-flow exceptions unchanged.

- [ ] **Step 4: Run adapter and existing query tests**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_providers.py tests/test_web_query_contract.py tests/test_focused_tool_result.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/ai/web_research/providers.py app/ai/web_query_contract.py tests/fixtures/web_research tests/test_web_research_providers.py
git commit -m "refactor: isolate web provider adapters"
```

---

### Task 4: Text research orchestration, retry, fallback, circuit, and cache

**Files:**
- Create: `app/ai/web_research/service.py`
- Create: `tests/test_web_research_service.py`

**Interfaces:**
- Consumes: provider protocols, `SourceRegistry`, `ResearchBudget`, `ResearchLimits`, and `ResearchRequest`.
- Produces: `WebResearchService.new_session(scope, budget) -> WebResearchSession`.
- Produces: `WebResearchSession.search(request) -> WebEvidenceBundle` and `WebResearchSession.open(urls, question) -> WebEvidenceBundle`.
- Produces: `ProviderHealthRegistry` and `ResearchResultCache` as injectable collaborators.

- [ ] **Step 1: Write failing orchestration tests**

```python
@pytest.mark.asyncio
async def test_retry_then_fallback_preserves_one_source_registry():
    primary = SequenceProvider([ProviderFailure("rate_limited", provider="primary", retryable=True), ProviderFailure("rate_limited", provider="primary", retryable=True)])
    fallback = SequenceProvider([[ProviderSource(provider="fallback", url="https://docs.test/a", title="A", snippet="evidence", rank=1)]])
    session = service(primary, fallback).new_session(SCOPE, ResearchBudget.for_mode("quick"))
    bundle = await session.search(REQUEST)
    assert [source.source_id for source in bundle.sources] == ["S1"]
    assert bundle.failures[0].code == "rate_limited"
    assert bundle.providers_used == ("primary", "fallback")


@pytest.mark.asyncio
async def test_in_progress_search_has_no_total_elapsed_deadline():
    provider = ProgressingProvider(chunks=50)
    bundle = await service(provider).new_session(SCOPE, ResearchBudget.for_mode("quick")).search(REQUEST)
    assert bundle.sources
    assert provider.cancelled is False
```

- [ ] **Step 2: Run service tests and verify failures**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_service.py -q`

Expected: FAIL because the service and session do not exist.

- [ ] **Step 3: Implement one-retry ordered fallback**

```python
async def _call_chain(self, operation: str, invoke: Callable[[Any], Awaitable[T]]) -> tuple[T, tuple[ResearchFailure, ...]]:
    failures: list[ResearchFailure] = []
    for provider in await self._resolver.resolve(operation):
        if self._health.is_open(provider.name):
            failures.append(ResearchFailure(provider=provider.name, code="circuit_open", retryable=True))
            continue
        for attempt in range(2):
            try:
                value = await invoke(provider)
                self._health.record_success(provider.name)
                return value, tuple(failures)
            except ProviderFailure as exc:
                failures.append(ResearchFailure(provider=exc.provider, code=exc.code, retryable=exc.retryable))
                self._health.record_failure(exc.provider, exc.code)
                if not exc.retryable or attempt == 1:
                    break
                await self._retry_backoff(attempt)
    raise AllProvidersFailed(tuple(failures))
```

Do not wrap the chain in `asyncio.wait_for` or `asyncio.timeout`. Provider clients retain their own connect/idle-read controls. Always re-raise cancellation.

- [ ] **Step 4: Add bounded cache and circuit collaborators**

Cache keys include the fully normalized request, provider operation, and freshness bucket. Cache values are immutable normalized records, never provider payloads. `recent` entries receive the shortest configured TTL; `as_of` and `timeless` may use longer TTLs. The cache and circuit registry are injected so tests use deterministic clocks and stores.

Circuit state must count only retryable provider failures, open after the configured consecutive-failure count, and close after a successful half-open probe. It must never treat empty-but-valid results, invalid user input, cancellation, or policy refusal as provider health failures.

- [ ] **Step 5: Run service and cancellation tests and commit**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_service.py tests/test_message_stream_cancellation.py tests/test_research_budget_failure_release.py -q`

Expected: PASS.

```powershell
git add app/ai/web_research/service.py tests/test_web_research_service.py
git commit -m "feat: orchestrate resilient web research"
```

---

### Task 5: Secure visual candidate preparation and protected references

**Files:**
- Modify: `app/services/web_image_service.py:49-390`
- Modify: `app/repositories/web_image_reference.py:16-70`
- Modify: `app/ai/web_research/service.py`
- Create: `app/ai/web_research/model_context.py`
- Create: `tests/test_web_research_images.py`
- Modify: `tests/test_web_image_service.py`
- Modify: `tests/test_web_image_reference_repository.py`

**Interfaces:**
- Consumes: `ProviderImageCandidate`, `ResearchScope`, `VisualIntent`, `WebImageService.fetch_url`, and the session's structural budget.
- Produces: private `PreparedImage(candidate, fetched, rich_item)` records held only by `WebResearchSession`.
- Produces: `WebResearchSession.model_evidence_blocks(*, supports_vision: bool) -> list[dict[str, Any]]`.
- Produces: `WebResearchSession.finish(selected_candidate_ids) -> dict[str, Any]`, `WebImageService.release_references(ids, user_id)`, and `WebImageReferenceRepository.asoft_delete_many(ids, user_id)`.

- [ ] **Step 1: Write failing visual preparation tests**

```python
@pytest.mark.asyncio
async def test_only_validated_bytes_are_offered_to_the_model():
    image_service = FakeImageService({"https://img.test/1.jpg": FETCHED_JPEG, "https://img.test/2.jpg": WebImageRejected("mime")})
    session = make_session(image_service=image_service)
    bundle = await session.search(REQUEST.model_copy(update={"visual_intent": "comparison", "image_query": "two interfaces"}))
    assert [image.candidate_id for image in bundle.images] == ["I1"]
    blocks = session.model_evidence_blocks(supports_vision=True)
    assert any(part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/jpeg;base64,") for part in blocks)


@pytest.mark.asyncio
async def test_non_vision_model_gets_no_candidate_metadata_selection():
    session = prepared_session(2)
    blocks = session.model_evidence_blocks(supports_vision=False)
    assert all(part.get("type") != "image_url" for part in blocks)
    assert session.reason_codes == {"answer_model_not_vision_capable"}
```

- [ ] **Step 2: Run image tests and verify failures**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_images.py tests/test_web_image_service.py -q`

Expected: FAIL because prepared candidates and cleanup methods are absent.

- [ ] **Step 3: Reuse the safe fetcher and prepare bounded candidates concurrently**

Use `WebImageService.fetch_url` unchanged for HTTPS, DNS/IP pinning, redirect revalidation, bounded streaming, MIME sniffing, Pillow verification, and pixel limits. Add a digest property to `FetchedWebImage` or calculate SHA-256 in the session. Validate only the provider-bounded candidate pool, preserve provider order among successful candidates, and stop admitting after four successes or six for `gallery`.

Do not impose an aggregate elapsed-time deadline. A stalled socket is still handled by `WebImageService` connect/read liveness controls, and cancellation must propagate through the candidate tasks.

- [ ] **Step 4: Register protected references before model selection and clean unused rows**

Register each admitted candidate with its already validated bytes so any later selected marker is retrievable without a second third-party fetch. Build a deterministic rich-item ID from the protected reference, retain the candidate ID separately, and expose only `/web-images/<uuid>` to public records.

```python
async def finish(self, selected_candidate_ids: Collection[str]) -> dict[str, Any]:
    selected = {value for value in selected_candidate_ids if value in self._prepared_images}
    unused_ids = [item.reference_id for key, item in self._prepared_images.items() if key not in selected]
    await self._image_service.release_references(unused_ids, user_id=self.scope.user_id)
    return {
        "web_sources_version": 1,
        "web_sources": [record.model_dump(mode="json") for record in self.source_registry.records],
        "_rich_item_candidates": [self._prepared_images[key].rich_item for key in selected],
        "web_research": self.public_trace(selected),
    }
```

If registration fails, remove that candidate before the answer call. Public metadata never points at a reference that was not committed.

- [ ] **Step 5: Run image/storage tests and commit**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_images.py tests/test_web_image_service.py tests/test_web_image_reference_repository.py tests/test_web_images_api.py -q`

Expected: PASS.

```powershell
git add app/ai/web_research/model_context.py app/ai/web_research/service.py app/services/web_image_service.py app/repositories/web_image_reference.py tests/test_web_research_images.py tests/test_web_image_service.py tests/test_web_image_reference_repository.py
git commit -m "feat: prepare verified web images for model selection"
```

---

### Task 6: Turn session wiring and thin product tools

**Files:**
- Modify: `app/ai/tool_context.py:15-145`
- Modify: `app/ai/web_tools.py:1-900`
- Modify: `app/ai/agents/base_agent.py:75-85,450-505`
- Modify: `app/ai/workflow/middleware.py:180-245,400-490`
- Modify: `app/ai/workflow/specialists.py:567-762`
- Modify: `app/ai/graph.py:214-310,1680-1710,2600-2640`
- Modify: `app/core/container.py:440-470,620-700`
- Create: `tests/test_web_research_tool_session.py`
- Modify: `tests/test_web_tool_binding.py`

**Interfaces:**
- Consumes: `WebResearchService.new_session`, `ResearchBudget`, routing policy, and `ToolContext`.
- Produces: `ToolContext.web_research_session` and a shared session instance for tools plus model middleware.
- Produces: thin `create_web_search_tool` and `create_web_open_tool`; `image_search` is no longer bound.

- [ ] **Step 1: Add failing session-isolation and tool-shape tests**

```python
@pytest.mark.asyncio
async def test_web_tool_writes_into_the_same_session_the_specialist_owns():
    session = FakeResearchSession()
    tool = create_web_search_tool()
    with tool_execution_context(conversation_id="c", user_id="u", agent_key="search", web_research_session=session):
        payload = json.loads(await tool.ainvoke({"query": "release notes", "objective": "find the current release", "mode": "quick", "visual_intent": "figure", "image_query": "release UI"}))
    assert session.requests[0].visual_intent == "figure"
    assert payload["sources"][0]["source_id"] == "S1"


def test_chat_and_search_bind_only_canonical_product_web_tools(agent):
    names = {tool.name for tool in agent_tools(agent)}
    assert {"web_search", "web_open"} <= names
    assert "image_search" not in names
    assert "tavily_search" not in names
    assert "brave_image_search" not in names
```

- [ ] **Step 2: Run tool session tests and verify failures**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_tool_session.py tests/test_web_tool_binding.py -q`

Expected: FAIL because context has no session and `image_search` remains bound.

- [ ] **Step 3: Extend `ToolContext` and specialist construction**

Add `web_research_session: Any | None` to `ToolContext` and `tool_execution_context`. `SpecialistFactory._build` creates exactly one session from authenticated request scope and the turn's `ResearchBudget`, then passes the same object to `SpecialistToolScope` and the grounding middleware. Worker and top-level invocations each receive their own session; Continue hydrates its new session from pair-complete carried web ToolMessages.

Inject `WebResearchService` through `Container -> create_workflow -> MultiAgentWorkflow -> SpecialistFactory`; do not resolve the container from inside a tool call.

- [ ] **Step 4: Reduce `web_tools.py` to validated adapters over the session**

`WebSearchRequest` gains `mode`, `visual_intent`, and optional `image_query`. `create_web_search_tool` validates arguments, checks server/client scope, gets `get_tool_context().web_research_session`, and returns `bundle.model_dump_json()`. `WebEvidenceBundle` is already the public, byte-free tool contract. The session runs text and image provider operations with `asyncio.gather` only when both were requested.

`create_web_open_tool` accepts source IDs as well as user-supplied URLs; source IDs resolve through the session registry. Remove provider resolution, projection, and selected-image copy from tool factories.

- [ ] **Step 5: Run tool, isolation, and continuation tests and commit**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_tool_session.py tests/test_web_tool_binding.py tests/test_client_invocation_isolation.py tests/test_conversation_compaction_message_paths.py tests/test_research_budget.py -q`

Expected: PASS.

```powershell
git add app/ai/tool_context.py app/ai/web_tools.py app/ai/agents/base_agent.py app/ai/workflow/middleware.py app/ai/workflow/specialists.py app/ai/graph.py app/core/container.py tests/test_web_research_tool_session.py tests/test_web_tool_binding.py
git commit -m "refactor: route product web tools through turn sessions"
```

---

### Task 7: Dynamic post-tool evidence and actual image input

**Files:**
- Modify: `app/ai/web_research/model_context.py`
- Modify: `app/ai/workflow/middleware.py:245-430,691-770`
- Modify: `app/ai/workflow/specialists.py:683-762`
- Modify: `app/ai/prompts.py:35-90,140-330,533-610`
- Create: `tests/test_web_research_model_context.py`
- Modify: `tests/test_selected_image_reaches_the_model.py`
- Modify: `tests/test_specialist_rich_image_lift.py`

**Interfaces:**
- Consumes: the shared `WebResearchSession` and `RuntimeModelMiddleware.runtime_config`.
- Produces: `WebEvidenceMiddleware.awrap_model_call(request, handler)`.
- Produces: `build_model_evidence_content(session, supports_vision) -> list[dict[str, Any]]`.

- [ ] **Step 1: Replace the static-inventory regression with failing dynamic tests**

```python
@pytest.mark.asyncio
async def test_second_model_call_sees_images_discovered_by_the_first_tool_round():
    model = RecordingTwoRoundModel(tool_name="web_search", tool_args=VISUAL_SEARCH_ARGS)
    await specialist_factory(model=model, research_service=prepared_service()).invoke(REQUEST)
    second = model.requests[1]
    assert any(part.get("type") == "image_url" for message in second.messages for part in list_content(message))
    assert "candidate_id=I1" in flatten_text(second.messages)


@pytest.mark.asyncio
async def test_fallback_model_capability_is_checked_on_each_attempt():
    first, fallback = failing_vision_model(), recording_text_only_model()
    await invoke_with_fallback(first, fallback)
    assert "candidate_id=I1" not in flatten_text(fallback.requests[0].messages)
    assert "answer_model_not_vision_capable" in session.reason_codes
```

- [ ] **Step 2: Run dynamic-context tests and verify the current failure**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_model_context.py tests/test_selected_image_reaches_the_model.py -q`

Expected: FAIL because `SpecialistFactory` resolves its prompt once before tools and no middleware attaches candidate pixels.

- [ ] **Step 3: Add `WebEvidenceMiddleware` inside runtime model resolution**

```python
class WebEvidenceMiddleware(AgentMiddleware):
    def __init__(self, session: WebResearchSession, runtime_config_provider: Callable[[], ResolvedRuntimeModelConfig | None]) -> None:
        self._session = session
        self._runtime_config_provider = runtime_config_provider

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        config = self._runtime_config_provider()
        supports_vision = bool(config and config.capabilities.get("supports_vision"))
        messages = inject_latest_web_evidence(request.messages, self._session, supports_vision=supports_vision)
        response = await handler(request.override(messages=messages))
        await self._session.observe_model_response(response)
        return response
```

Insert it after `RuntimeModelMiddleware` and request-budget preflight, but before `ToolExecutionMiddleware`, so provider fallback recalculates vision capability and tool execution writes into the same session before the next model call.

- [ ] **Step 4: Replace rich inventory prompting with token instructions**

Prompt text must say: cite only supplied `[[source:S#]]` IDs; place only visually inspected `[[image:I#]]` candidates; use zero when irrelevant; use one for a figure, two-to-four for comparisons/multiple entities, and up to six only for explicit galleries. Remove the claim that selected images are already in an available rich-item inventory.

- [ ] **Step 5: Run specialist/model-context tests and commit**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_model_context.py tests/test_selected_image_reaches_the_model.py tests/test_specialist_rich_image_lift.py tests/test_base_agent_image_history.py -q`

Expected: PASS.

```powershell
git add app/ai/web_research/model_context.py app/ai/workflow/middleware.py app/ai/workflow/specialists.py app/ai/prompts.py tests/test_web_research_model_context.py tests/test_selected_image_reaches_the_model.py tests/test_specialist_rich_image_lift.py
git commit -m "feat: show verified web images to the answer model"
```

---

### Task 8: Server-owned grounding tokens and rich-image materialization

**Files:**
- Create: `app/ai/web_research/grounding.py`
- Modify: `app/ai/workflow/finalization.py:53-330,408-590`
- Modify: `app/ai/workflow/contracts.py:130-170`
- Modify: `app/core/response_constants.py:186-570`
- Modify: `app/core/rich_placement.py:156-560`
- Create: `tests/test_web_grounding.py`
- Modify: `tests/test_output_validation.py`
- Modify: `tests/test_rich_placement.py`

**Interfaces:**
- Consumes: `SourceRecord`, prepared candidate-to-rich-item mappings, and model content.
- Produces: `resolve_grounding_tokens(text, sources, images) -> GroundingResolution`.
- Produces: `GroundingStreamFilter.learn(bundle)`, `.feed(delta) -> GroundingDelta`, and `.flush() -> GroundingDelta`.
- Adds: web sources and selected rich items to `OutcomeProvenance`/response metadata from server-owned session state.

- [ ] **Step 1: Write failing whole-text and split-stream token tests**

```python
def test_resolver_links_known_source_and_drops_unknown_image():
    result = resolve_grounding_tokens(
        "Released in March [[source:S1]].\n[[image:I9]]",
        sources={"S1": SOURCE},
        images={},
    )
    assert result.text == "Released in March [1](https://docs.test/release)."
    assert result.selected_image_ids == ()
    assert result.warnings == ({"code": "unknown_image_candidate", "id": "I9"},)


def test_stream_filter_handles_tokens_split_at_every_character():
    expected = "See [1](https://docs.test/release)."
    for cut in range(len("See [[source:S1]].") + 1):
        stream = GroundingStreamFilter()
        stream.learn(BUNDLE)
        actual = stream.feed(TEXT[:cut]).text + stream.feed(TEXT[cut:]).text + stream.flush().text
        assert actual == expected
```

Add cases for fenced code, inline code, punctuation spacing, repeated citations, stale-turn IDs, selected image order, duplicate image tokens, and image cardinality.

Add a Planning worker case proving that worker web sources and selected rich items travel through `WorkerResult.artifacts` into the parent synthesis provenance without becoming a public worker answer or receiving new IDs.

- [ ] **Step 2: Run grounding tests and verify failures**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_grounding.py tests/test_output_validation.py -q`

Expected: FAIL because no web grounding resolver exists.

- [ ] **Step 3: Implement one parser for stream and terminal paths**

The parser must hold incomplete `[[source:`/`[[image:` prefixes across chunks, ignore token-like text inside fenced/indented/inline code, sanitize link labels and URLs from server records only, and return selected candidate IDs as a side channel. It must never parse arbitrary model-authored Markdown URLs as provenance.

```python
@dataclass(frozen=True)
class GroundingResolution:
    text: str
    selected_image_ids: tuple[str, ...]
    selected_rich_items: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, str], ...]
```

- [ ] **Step 4: Apply grounding before output validation and persistence**

After `agent.ainvoke` returns, `SpecialistFactory.invoke` obtains the selected candidate IDs recorded by `WebEvidenceMiddleware`, awaits `session.finish(selected_ids)`, and passes the resulting immutable grounding metadata into `_to_outcome`. `invoke_worker` performs the same closeout and adds the immutable web provenance to `WorkerResult.artifacts` for parent synthesis. Both entry points call `session.abort_if_open()` in `finally`; a finished session is a no-op, while an exception, cancellation, interrupt, or hard-limit return releases all prepared references without swallowing the original control flow.

Extend the `_build` return value to a named `SpecialistBuild` dataclass containing the agent, tool middleware, accountant, evidence middleware, and research session so this lifecycle cannot be lost through tuple ordering.

`PublicResponseFinalizer` then resolves tokens from that server-owned outcome provenance, replaces response content, adds only selected rich items to `_rich_item_candidates`, and persists `web_sources_version=1`, `web_sources`, and the bounded research trace. It never reaches back into a live session.

Remove image selection and image anchoring from `finalize_article_content`; retain generic auto-placement only for `inline_or_append` non-image items such as widgets. A valid image token deterministically becomes `<!--rich:<server-id>-->` and is not subjected to another text-overlap score.

- [ ] **Step 5: Run finalization/rich tests and commit**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_grounding.py tests/test_output_validation.py tests/test_rich_placement.py tests/test_rich_response_contract.py tests/test_rich_response_metadata.py -q`

Expected: PASS.

```powershell
git add app/ai/web_research/grounding.py app/ai/workflow/finalization.py app/ai/workflow/contracts.py app/core/response_constants.py app/core/rich_placement.py tests/test_web_grounding.py tests/test_output_validation.py tests/test_rich_placement.py
git commit -m "feat: validate web citations and image selections"
```

---

### Task 9: Canonical source events, clickable client links, and history parity

**Files:**
- Modify: `app/services/event_streaming/events.py:20-105,260-310`
- Modify: `app/services/event_streaming/graph_public_projection.py:55-125,240-340`
- Modify: `app/services/event_streaming/internal_sse.py:1-170`
- Modify: `app/services/event_streaming/ai_sdk_v6.py:109-590`
- Modify: `app/services/event_streaming/ai_sdk_projection.py:259-470`
- Modify: `app/services/message_service.py:340-410,1360-1420,1730-1905,2470-2590`
- Modify: `app/schemas/message.py:80-135`
- Modify: `demo.py:7940-8410,10250-10320,10950-11020`
- Create: `tests/test_web_source_streaming.py`
- Modify: `tests/test_ai_sdk_v6_stream_contract.py`
- Modify: `tests/test_internal_sse_stream_contract.py`
- Modify: `tests/test_message_service_event_streaming.py`

**Interfaces:**
- Consumes: `SourceRecord` and `GroundingStreamFilter`.
- Produces: canonical `sources_upsert` events carrying `{"sources": [...]}`.
- Produces: legacy/internal SSE `{"type": "sources", "sources": [...]}`.
- Produces: AI SDK v6 `{type: "source-url", sourceId, url, title}` parts.

- [ ] **Step 1: Write failing transport parity tests**

```python
@pytest.mark.asyncio
async def test_same_source_identity_reaches_internal_and_ai_sdk_streams():
    canonical = make_event("sources_upsert", sequence=3, data={"sources": [SOURCE.model_dump(mode="json")]})
    internal = legacy_event_from_v3(canonical)
    ai_sdk = decode_sse([chunk async for chunk in AISDKV6StreamAdapter(source(canonical)).stream()])
    assert internal["sources"][0]["source_id"] == "S1"
    assert next(part for part in ai_sdk if part["type"] == "source-url")["sourceId"] == "S1"


def test_history_projects_persisted_sources_as_ai_sdk_parts():
    projected = project_ai_sdk_message_for_capability(MESSAGE_WITH_WEB_SOURCES, inline_rich_response_v1=True)
    assert {part["type"] for part in projected["parts"]} >= {"text", "source-url"}
```

- [ ] **Step 2: Run stream tests and verify failures**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_source_streaming.py tests/test_ai_sdk_v6_stream_contract.py tests/test_internal_sse_stream_contract.py -q`

Expected: FAIL because `sources_upsert` is not a canonical event and history has no source parts.

- [ ] **Step 3: Teach the graph projector to learn and publish tool evidence**

When a successful `tool_execution_end` contains a canonical web bundle, parse it with `WebEvidenceBundle.model_validate_json`, teach the turn's `GroundingStreamFilter`, emit one deduplicated `sources_upsert` before answer text, and keep candidate descriptors private until the filter reports a selected image token. Emit the selected rich-item upsert before the converted rich marker delta.

Apply the same source-ID learning to subagent tool events, but keep worker sources inside the subagent activity channel until the parent answer cites them. When parent synthesis provenance admits those records, publish their `sources_upsert` once under their original IDs.

Do not reconstruct sources from arbitrary tool prose or model-authored links.

- [ ] **Step 4: Add both client projections and persistence**

Map sources to Streamlit state keyed by source ID and render numbered links using the final canonical URL. In AI SDK v6 use flat native parts:

```python
{
    "type": "source-url",
    "sourceId": source["source_id"],
    "url": source["url"],
    "title": source.get("title"),
}
```

Persist sources through `build_bot_metadata` for complete, partial, interrupted, stopped, and resumed assistant messages. History projection derives the same `source-url` parts from metadata. Capability scrubbing may hide rich images from old clients but must not strip ordinary public web sources.

- [ ] **Step 5: Run stream/history tests and commit**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_source_streaming.py tests/test_ai_sdk_v6_stream_contract.py tests/test_internal_sse_stream_contract.py tests/test_message_service_event_streaming.py tests/test_routing_v2_continuation_streaming.py -q`

Expected: PASS.

```powershell
git add app/services/event_streaming app/services/message_service.py app/schemas/message.py demo.py tests/test_web_source_streaming.py tests/test_ai_sdk_v6_stream_contract.py tests/test_internal_sse_stream_contract.py tests/test_message_service_event_streaming.py
git commit -m "feat: stream and persist canonical web sources"
```

---

### Task 10: Required-web finalization and honest degraded answers

**Files:**
- Modify: `app/ai/workflow/finalization.py:66-220,267-330`
- Modify: `app/ai/workflow/specialists.py:820-880`
- Modify: `app/ai/prompts.py:268-330,533-610`
- Create: `tests/test_web_research_output_policy.py`
- Modify: `tests/test_output_validation.py`

**Interfaces:**
- Consumes: `RoutingDecision.requires_web`, public source provenance, and research failure codes.
- Produces: `WebEvidencePolicy.validate(outcome)` and a server-owned unverifiable-current-information response.

- [ ] **Step 1: Write failing required-web policy tests**

```python
def test_required_web_turn_cannot_publish_uncited_model_knowledge():
    outcome = outcome_for("The current version is 9.2", requires_web=True, sources=())
    finalized = finalizer.finalize(outcome)
    assert finalized.response.message.content == "I couldn't verify the current information because web research returned no usable public source."
    assert finalized.response.metadata["web_research"]["status"] == "unverified"


def test_image_failure_does_not_discard_valid_cited_text():
    outcome = cited_outcome(image_failures=[{"code": "image_transport_stalled"}])
    assert finalizer.finalize(outcome).response.message.content.startswith("Verified")
```

- [ ] **Step 2: Run policy tests and verify failures**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_output_policy.py tests/test_output_validation.py -q`

Expected: FAIL because finalization does not distinguish required web evidence.

- [ ] **Step 3: Implement evidence policy and prompt behavior**

Select `WebEvidencePolicy` whenever the routing decision requires web or the answer contains source tokens. Permit a normal answer only when at least one admitted source supports the response. If all providers fail or all sources are rejected, replace unsupported current claims with the exact server-owned limitation above and persist bounded reason codes.

Image failures remain optional degradation. A page-open failure downgrades that source to `snippet_only`; it does not delete a valid search-result source.

- [ ] **Step 4: Run finalization tests and commit**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_output_policy.py tests/test_output_validation.py tests/test_production_readiness_contract.py -q`

Expected: PASS.

```powershell
git add app/ai/workflow/finalization.py app/ai/workflow/specialists.py app/ai/prompts.py tests/test_web_research_output_policy.py tests/test_output_validation.py
git commit -m "feat: require evidence for current web answers"
```

---

### Task 11: Configuration, metrics, and operational health

**Files:**
- Modify: `app/core/config.py:1360-1450,1940-2050`
- Modify: `.env.example`
- Create: `app/observability/web_research.py`
- Modify: `app/core/container.py:360-470`
- Create: `tests/test_web_research_config.py`
- Create: `tests/test_web_research_metrics.py`

**Interfaces:**
- Consumes: service/provider/session outcomes.
- Produces: `WebResearchMetrics.record_operation(*, operation: str, mode: ResearchMode, provider: str, outcome: str, reason_code: str | None, duration_seconds: float) -> None`.
- Produces: `WebResearchMetrics.record_source(*, mode: ResearchMode, status: str, cited: bool) -> None`.
- Produces: `WebResearchMetrics.record_image(*, visual_intent: VisualIntent, outcome: str, reason_code: str | None) -> None`.
- Produces: `get_web_research_metrics_recorder()` and validated settings for structural limits, retry, cache, circuit, and liveness controls.

- [ ] **Step 1: Write failing config and bounded-label tests**

```python
def test_default_limits_match_the_approved_structural_policy():
    value = Settings(_env_file=None)
    assert value.web_research_quick_max_searches == 1
    assert value.web_research_quick_max_sources == 5
    assert value.web_research_agentic_max_searches == 3
    assert value.web_research_agentic_max_sources == 8
    assert value.web_research_max_model_images == 4
    assert value.web_research_gallery_max_model_images == 6
    for forbidden in ("web_research_deadline_seconds", "web_research_quick_timeout_seconds", "web_research_agentic_timeout_seconds"):
        assert forbidden not in value.model_fields


def test_metrics_reject_query_text_as_a_label():
    with pytest.raises(TypeError):
        metrics.record_operation(operation="search", outcome="success", query="private text")
```

- [ ] **Step 2: Run config/metrics tests and verify failures**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_config.py tests/test_web_research_metrics.py -q`

Expected: FAIL because the canonical settings and recorder do not exist.

- [ ] **Step 3: Add settings without aggregate search time limits**

Add feature flags and structural limits for quick/agentic calls, sources, opens, model images, candidate pool, retry count, cache TTL by freshness, and circuit failure/cooldown. Keep existing web-image connect/read settings and describe them as transport liveness controls. Do not add `web_research_deadline_seconds`, `quick_timeout_seconds`, or `agentic_timeout_seconds`.

- [ ] **Step 4: Add bounded metrics and health snapshot**

Metrics may label only operation, mode, provider, outcome, reason code, visual intent, and cache/circuit state. Queries, objectives, URLs, titles, snippets, and user text are forbidden as labels and ordinary log fields. Histograms record observed duration without enforcing it.

Expose an internal health snapshot containing configured providers, circuit states, cache counters, and aggregate outcomes; never include keys or user-derived content.

- [ ] **Step 5: Run tests and commit**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_config.py tests/test_web_research_metrics.py tests/test_model_usage_docs.py -q`

Expected: PASS.

```powershell
git add app/core/config.py app/core/container.py app/observability/web_research.py .env.example tests/test_web_research_config.py tests/test_web_research_metrics.py
git commit -m "feat: add web research operations controls"
```

---

### Task 12: General automated evaluation and optional live canary

**Files:**
- Create: `app/evaluation/web_research/contracts.py`
- Create: `app/evaluation/web_research/metrics.py`
- Create: `app/evaluation/web_research/harness.py`
- Create: `eval/web_research/cases.json`
- Create: `scripts/evaluate_web_research.py`
- Create: `tests/test_web_research_evaluation.py`
- Create: `tests/live/test_web_research_live_provider.py`

**Interfaces:**
- Consumes: `WebResearchService`, canonical recorded providers, and finished assistant metadata.
- Produces: `WebResearchEvalCase`, `WebResearchEvalResult`, `evaluate_case`, and a machine-readable JSON report.

- [ ] **Step 1: Add failing general evaluation tests**

```python
def test_eval_matrix_has_no_topic_specific_rule_and_covers_dimensions():
    cases = load_cases("eval/web_research/cases.json")
    assert {case.freshness for case in cases} >= {"timeless", "recent", "as_of"}
    assert {case.visual_intent for case in cases} >= {"none", "figure", "comparison", "gallery"}
    assert any(case.query_language != case.answer_language for case in cases)
    assert all(not case.expected_exact_result_url for case in cases)


def test_contract_metrics_fail_an_unsupported_image_claim():
    result = score_case(case=FIGURE_CASE, response=RESPONSE_WITH_IMAGE_CLAIM, metadata={"rich_items": []})
    assert result.unsupported_image_claims == 1
    assert result.release_invariants_pass is False
```

- [ ] **Step 2: Run evaluation tests and verify failures**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_evaluation.py -q`

Expected: FAIL because the evaluation package and cases are absent.

- [ ] **Step 3: Implement the capability matrix and deterministic scorers**

Cases vary stable/current/historical, factual/comparison/exploratory, same-language/cross-language, clear/ambiguous entities, visual intent, candidate quality, and provider outcome. Use synthetic URLs and recorded provider results. Assertions cover source membership, selected-candidate membership, retrievability, cardinality, degraded status, cancellation, and stream/history parity—not exact live ranking.

Add metamorphic variants by applying deterministic paraphrase and locale substitutions stored in the case file. Do not add named incident fixtures or production keyword exceptions.

- [ ] **Step 4: Add optional offline visual judge and live-provider marker**

The deterministic suite is the release gate. The visual judge is an optional harness collaborator that receives only the test image, requested visual subject, and time requirement; its score is reported but cannot independently pass a failed invariant. Mark live provider tests with the existing `live_provider` marker and skip unless credentials plus `RUN_LIVE_WEB_RESEARCH_TESTS=1` are present.

- [ ] **Step 5: Run deterministic evaluation and commit**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_evaluation.py -q`

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe scripts/evaluate_web_research.py --cases eval/web_research/cases.json --output output/audits/web-research-eval.json`

Expected: tests PASS; command exits `0` and writes a report whose `release_invariants_pass` is `true`.

```powershell
git add app/evaluation/web_research eval/web_research/cases.json scripts/evaluate_web_research.py tests/test_web_research_evaluation.py tests/live/test_web_research_live_provider.py
git commit -m "test: add general web research evaluation"
```

---

### Task 13: Active-path migration and mandatory feature-owned cleanup

**Files:**
- Delete: `app/ai/selected_image_sink.py`
- Delete: `app/ai/image_discovery_flow.py`
- Delete: `tests/test_selected_image_sink.py`
- Delete or rewrite: `tests/test_image_discovery_flow.py`
- Delete or rewrite: `tests/test_provider_selected_image_injection.py`
- Delete or rewrite: `tests/test_multi_image_research.py`
- Modify: `app/ai/tool_execution.py:60-430,500-650,1880-2240`
- Modify: `app/ai/deferred_tool_binding.py:70-105`
- Modify: `app/ai/mcp_tool_catalog.py`
- Modify: `app/core/rich_placement.py`
- Modify: `app/core/response_constants.py`
- Modify: `app/services/event_streaming/ai_sdk_projection.py`
- Modify: `app/core/config.py`
- Modify: `README.md:780-830,1390-1470`
- Create: `tests/test_web_research_active_path_inventory.py`

**Interfaces:**
- Consumes: all canonical service, grounding, and transport tests from Tasks 1-12.
- Produces: one active web/image pipeline and an explicit read-only legacy history projection.

- [ ] **Step 1: Add a failing import/active-path inventory test**

```python
def test_removed_web_image_paths_are_not_imported_by_production_code():
    forbidden = {
        "selected_image_sink",
        "offer_selected_images",
        "select_brave_candidates",
        "anchor_image_items_by_query",
        "create_image_search_tool",
    }
    hits = production_symbol_hits(forbidden, roots=(Path("app"),))
    assert hits == {}


def test_raw_provider_tools_are_not_ordinary_agent_capabilities():
    names = ordinary_agent_tool_names()
    assert names.isdisjoint({"tavily_search", "tavily_extract", "brave_image_search", "image_search"})
```

- [ ] **Step 2: Run the inventory test and capture all current failures**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_active_path_inventory.py -q`

Expected: FAIL and list the remaining production imports/callers.

- [ ] **Step 3: Remove superseded execution paths**

Delete the selected-image context sink and metadata-only Brave selector. Remove image-search candidate harvesting from `execute_tool_calls`; generic MCP image content blocks remain supported as ordinary tool-produced images. Remove image query anchoring and model inventory selection from rich placement. Stop writing active legacy `metadata["images"]` for canonical web images.

Raw Tavily and Brave tools remain provider-adapter implementation details and cannot be discovered or bound by ordinary answer agents. Preserve only narrowly named compatibility readers that project historical `metadata["images"]` and old rich-item messages.

- [ ] **Step 4: Remove obsolete settings, prompts, tests, and docs**

Remove settings whose only callers were the deleted pipeline, including the selected-image sink/image-anchor controls and independent image-search call budget. Update old tests to exercise canonical contracts or delete them when their asserted behavior is intentionally gone. Update README architecture and configuration tables to name `WebResearchService`, structural budgets, native AI SDK source parts, and progress-aware transport liveness.

- [ ] **Step 5: Prove no dead active path remains and commit**

Run: `rg -n "selected_image_sink|offer_selected_images|select_brave_candidates|anchor_image_items_by_query|create_image_search_tool" app tests README.md`

Expected: no production hits; test/docs hits only when explicitly describing historical compatibility, otherwise none.

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_active_path_inventory.py tests/test_web_tools.py tests/test_rich_placement.py tests/test_ai_sdk_context_window.py -q`

Expected: PASS.

```powershell
git add app/ai/tool_execution.py app/ai/deferred_tool_binding.py app/ai/mcp_tool_catalog.py app/core/rich_placement.py app/core/response_constants.py app/services/event_streaming/ai_sdk_projection.py app/core/config.py README.md app/ai/selected_image_sink.py app/ai/image_discovery_flow.py tests/test_selected_image_sink.py tests/test_image_discovery_flow.py tests/test_provider_selected_image_injection.py tests/test_multi_image_research.py tests/test_web_research_active_path_inventory.py
git commit -m "refactor: remove superseded web image paths"
```

---

### Task 14: Full verification, rollout documentation, and release evidence

**Files:**
- Create: `docs/operations/web-research-rollout.md`
- Modify: `docs/operations/routing-v2-rollout.md`
- Modify: `docs/frontend/rich-image-rendering.md`
- Modify: `.env.example`
- Create: `tests/test_web_research_docs.py`

**Interfaces:**
- Consumes: the completed canonical path and metrics.
- Produces: operator rollout/rollback procedure, configuration reference, canary queries by capability class, and final verification evidence.

- [ ] **Step 1: Write a failing documentation contract test**

```python
def test_rollout_doc_names_required_controls_and_forbids_total_deadline():
    text = Path("docs/operations/web-research-rollout.md").read_text(encoding="utf-8")
    for phrase in ("structural budgets", "transport liveness", "provider fallback", "source-url", "rollback"):
        assert phrase in text
    assert "total research deadline" not in text.lower()
```

- [ ] **Step 2: Write operations and frontend documentation**

Document flags, provider order, retry/circuit/cache behavior, source and image reason codes, metrics, safe live-canary invocation, canary promotion, rollback, compatibility reads, and unselected-reference cleanup. Explain that latency is observed during rollout and no total elapsed cap interrupts active research. Document Streamlit `sources` events and AI SDK `source-url` parts with exact example payloads.

- [ ] **Step 3: Run focused documentation and contract suites**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_docs.py tests/test_web_research_active_path_inventory.py tests/test_web_source_streaming.py tests/test_web_research_evaluation.py -q`

Expected: PASS.

- [ ] **Step 4: Run the complete relevant test matrix**

Run:

```powershell
C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_web_research_contracts.py tests/test_web_source_registry.py tests/test_web_research_policy.py tests/test_web_research_providers.py tests/test_web_research_service.py tests/test_web_research_images.py tests/test_web_research_tool_session.py tests/test_web_research_model_context.py tests/test_web_grounding.py tests/test_web_source_streaming.py tests/test_web_research_output_policy.py tests/test_web_research_config.py tests/test_web_research_metrics.py tests/test_web_research_evaluation.py tests/test_web_tools.py tests/test_research_budget.py tests/test_output_validation.py tests/test_rich_placement.py tests/test_rich_response_contract.py tests/test_rich_response_metadata.py tests/test_ai_sdk_v6_stream_contract.py tests/test_internal_sse_stream_contract.py tests/test_message_service_event_streaming.py tests/test_message_stream_cancellation.py tests/test_routing_v2_continuation_streaming.py -q
```

Expected: PASS with no live-provider calls.

- [ ] **Step 5: Run repository quality gates**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m ruff check app client_backend tests scripts`

Expected: PASS.

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest -q -m "not live_provider"`

Expected: PASS.

- [ ] **Step 6: Generate deterministic release evidence**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe scripts/evaluate_web_research.py --cases eval/web_research/cases.json --output output/audits/web-research-eval.json`

Expected: exit `0`, all release invariants true, and observed latency distributions recorded without pass/fail thresholds.

- [ ] **Step 7: Commit documentation and verification contracts**

```powershell
git add docs/operations/web-research-rollout.md docs/operations/routing-v2-rollout.md docs/frontend/rich-image-rendering.md .env.example tests/test_web_research_docs.py
git commit -m "docs: add web research rollout runbook"
```

---

## Completion gate

Do not declare this project complete until all of the following are true:

- Chat and search specialists bind only the canonical product web tools.
- Required-web turns cannot publish unsupported current claims.
- The post-tool answer request contains actual validated image bytes and candidate IDs when the resolved model supports vision.
- Source and image tokens are validated by the same parser during streaming and terminal finalization.
- Streamlit and AI SDK expose identical source identities; AI SDK uses native `source-url` parts.
- Selected images resolve to committed protected references, and unselected candidate references are soft-deleted.
- No model-visible message promises an unavailable image.
- Structural budgets and cancellation tests pass without a total research deadline.
- The general evaluation matrix passes without topic-specific production logic or live-provider dependence.
- Obsolete active search/image paths, prompts, settings, and tests are removed; historical compatibility is isolated and read-only.
- Ruff, the non-live full suite, and deterministic release evidence all pass.
