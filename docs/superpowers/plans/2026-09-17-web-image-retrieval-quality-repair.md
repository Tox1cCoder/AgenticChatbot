# Web Image Retrieval Quality Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make image search return the requested visual itself, keep the best candidates across repeated searches, and pass compact non-duplicated evidence to the answer model.

**Architecture:** Preserve the existing provider/session/grounding pipeline. Improve Brave request fidelity at the provider boundary; turn the session's append-only prepared-image list into a bounded candidate catalog plus replaceable active vision window; and distinguish the complete stable source registry from each tool operation's public delta. Relevance ordering stays as one small function in `service.py`, with no new ranking service, visual taxonomy, or answer-time fallback.

**Tech Stack:** Python 3.10+, asyncio, Pydantic v2, LangChain message blocks, pytest, pytest-asyncio.

## Global Constraints

- Do not add `visual_kind`, `alt_text` routing, a query-rewrite model call, or a generic ranking framework.
- Keep provider-neutral contracts; Brave-specific locale/domain behavior belongs only in `providers.py`.
- Treat `include_domains` as a restriction: never return a disallowed-domain image as fallback.
- Preserve original-resolution publication, protected `/web-images/{id}` delivery, and model-only previews.
- Keep stable `S#` IDs and never bind an existing `I#` to different bytes.
- Keep the complete source registry for validation and grounding, but expose only the current operation's source delta in tool output.
- Admit image sources before text sources and enforce separate bounded quotas so either class cannot starve the other.
- Penalize small images; do not hard-reject them when they are the only relevant evidence.
- Use `.venv\Scripts\python.exe` for verification in this checkout.

---

### Task 1: Preserve image-provider locale, domain constraints, and quality metadata

**Files:**
- Modify: `app/ai/web_research/contracts.py`
- Modify: `app/ai/web_research/providers.py`
- Modify: `tests/test_web_research_providers.py`
- Add: `tests/fixtures/web_research/brave_images_mixed_domains.json`
- Add: `tests/fixtures/web_research/brave_images_t1_official.json`

**Interfaces:**
- Extend `ProviderImageCandidate` with optional `confidence` and `source_domain` fields.
- Keep `ImageSearchProvider.search(request)` unchanged.
- `BraveImageSearchProvider` still returns one provider-neutral tuple; its internal calls may include bounded domain probes.

- [ ] **Step 1: Add failing adapter tests for locale forwarding and domain enforcement**

Replace the one-payload `_Tool` test double with a call-recording sequence double while retaining compatibility with existing tests:

```python
class _Tool:
    def __init__(self, *payloads: object) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict) -> str:
        self.calls.append(args)
        payload = self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)]
        if isinstance(payload, Exception):
            raise payload
        return json.dumps(payload)
```

Add a test using `locale="vi-VN"` and `include_domains=("t1.gg",)` where the broad fixture includes Reddit and the second fixture includes an official T1 result. Assert:

```python
assert tool.calls[0] == {
    "query": "T1 League of Legends current roster full team official team photo",
    "count": 10,
    "safesearch": "strict",
    "country": "VN",
    "search_lang": "vi",
}
assert tool.calls[1]["query"].endswith(" site:t1.gg")
assert {image.source_domain for image in images} == {"t1.gg", "www.t1.gg"}
assert all("reddit.com" not in image.source_url for image in images)
```

Also assert that provider `confidence`, declared dimensions, and provider rank survive normalization.

- [ ] **Step 2: Add failing tests for bounded probing and deduplication**

Cover these cases:

```python
request = ResearchRequest(
    query="team",
    objective="find the official team photo",
    visual_intent="gallery",
    image_query="full T1 team photo",
    include_domains=("t1.gg", "lolesports.com", "x.com", "fourth.test"),
)
```

- The adapter makes the ordinary query plus at most three `site:` probes.
- A source host may equal the allowed domain or be its subdomain.
- A repeated `(canonical source URL, image URL)` appears once, preserving first-provider order.
- Failure of an optional domain probe does not erase already valid allowed-domain results.
- With no `include_domains`, the adapter makes exactly one call and preserves current behavior.

- [ ] **Step 3: Run provider tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_research_providers.py
```

Expected: the new tests fail because locale and domain constraints are not forwarded, no supplemental probe occurs, and quality metadata is discarded.

- [ ] **Step 4: Extend the provider-neutral candidate contract**

Add only these two fields to the current contract; leave all current fields in
place:

```python
confidence: str | None = Field(default=None, max_length=32)
source_domain: str | None = Field(default=None, max_length=253)
```

Do not introduce a confidence enum: Brave values may evolve, while the session ranking deliberately maps only known values and treats everything else as unknown.

- [ ] **Step 5: Implement locale parsing and strict domain filtering in the Brave adapter**

Add small private functions in `providers.py`:

```python
def _brave_locale(locale: str | None) -> dict[str, str]:
    parts = str(locale or "").replace("_", "-").split("-")
    args: dict[str, str] = {}
    if parts and len(parts[0]) == 2:
        args["search_lang"] = parts[0].lower()
    if len(parts) > 1 and len(parts[1]) == 2:
        args["country"] = parts[1].upper()
    return args


def _host_matches(host: str | None, allowed: str) -> bool:
    host = str(host or "").lower().rstrip(".")
    allowed = allowed.lower().removeprefix("www.").rstrip(".")
    return host == allowed or host.endswith(f".{allowed}")
```

In `BraveImageSearchProvider.search`:

1. Call Brave once with the untouched image query, safesearch, count, and locale args.
2. Normalize the allowed domains once.
3. If restrictions exist, keep only matching results.
4. If fewer than `count` matching results remain, call at most the first three domains with `f"{query} site:{domain}"`.
5. Normalize and deduplicate the combined result stream by canonical source URL plus image URL.
6. Return only allowed hosts when restrictions were supplied.

Use `urlsplit` for host extraction and `canonicalize_public_url` for the source half of the dedupe key. Log optional probe failures at warning level with provider/domain/code; allow the first ordinary-call failure to keep its existing `ProviderFailure` behavior.

Add these arguments to the current `ProviderImageCandidate(...)` construction,
without interpreting them:

```python
confidence=str(raw["confidence"]).lower() if raw.get("confidence") else None,
source_domain=str(
    raw.get("source_domain") or urlsplit(str(source_url)).hostname or ""
) or None,
```

- [ ] **Step 6: Verify provider behavior and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_research_providers.py tests/test_web_image_capacity_and_resolution.py -k "brave or provider"
.\.venv\Scripts\python.exe -m ruff check app/ai/web_research/contracts.py app/ai/web_research/providers.py tests/test_web_research_providers.py
git diff --check
```

Expected: all selected tests and checks pass.

- [ ] **Step 7: Commit provider fidelity**

```powershell
git add -- app/ai/web_research/contracts.py app/ai/web_research/providers.py tests/test_web_research_providers.py tests/fixtures/web_research/brave_images_mixed_domains.json tests/fixtures/web_research/brave_images_t1_official.json
git commit -m "fix: preserve image search constraints"
```

---

### Task 2: Replace the append-only image list with a bounded quality catalog

**Files:**
- Modify: `app/ai/web_research/contracts.py`
- Modify: `app/ai/web_research/service.py`
- Modify: `app/ai/web_research/source_registry.py`
- Modify: `tests/test_web_source_registry.py`
- Modify: `tests/test_web_research_contracts.py`
- Modify: `tests/test_web_image_capacity_and_resolution.py`
- Modify: `tests/test_web_research_images.py`

**Interfaces:**
- `SourceRegistry.admit()` returns only records newly admitted by that call; `records` remains the complete stable registry.
- `PreparedImage` gains an internal immutable `catalog_key`.
- `ImageCandidateRecord` gains optional `source_domain` for model-visible labeling.
- `WebResearchSession.prepared_images` remains the active `I# -> PreparedImage` map consumed by grounding and closeout.

- [ ] **Step 1: Write the failing source-registry delta test**

Add to `tests/test_web_source_registry.py`:

```python
def test_admit_returns_only_the_current_delta() -> None:
    registry = SourceRegistry(max_sources=3)
    first = registry.admit((_provider_source("https://example.com/a"),))
    second = registry.admit(
        (
            _provider_source("https://example.com/a"),
            _provider_source("https://example.com/b"),
        )
    )

    assert [record.source_id for record in first] == ["S1"]
    assert [record.source_id for record in second] == ["S2"]
    assert [record.source_id for record in registry.records] == ["S1", "S2"]
```

Update existing callers that need the complete set to use `registry.records`; `import_records()` already resolves each imported URL and needs no semantic change.

- [ ] **Step 2: Write failing active-window regression tests**

Build deterministic fake image providers/candidates in `tests/test_web_image_capacity_and_resolution.py`. Cover all of these in two searches on one session configured with `ResearchBudget(max_image_searches=3)`:

- Search 1 returns small 140×140 roster cards plus a 1200×675 full-team photo.
- The full-team photo ranks ahead of the small cards even when its provider rank is later.
- Search 2 returns worse, low-confidence 100×100 results; the good prior photo stays active.
- A separate search 2 returns a 1920×1080 preferred-domain official photo; it enters the active window and the weakest prepared candidate is released.
- Retained prepared images keep their original `I#`; the replacement receives a fresh, never-reused ID.
- The evicted image's pending reference is released and cannot be selected through grounding.
- A saturated text result set does not consume the image catalog's reserved source capacity.

Use fake bytes and the existing recording image service; do not make network calls.

- [ ] **Step 3: Run catalog tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_source_registry.py tests/test_web_image_capacity_and_resolution.py tests/test_web_research_images.py
```

Expected: repeated searches cannot improve a full first window, provider rank wins over quality, IDs can be reused through `len(prepared_images) + 1`, and `admit()` returns the whole registry.

- [ ] **Step 4: Make source admission delta-aware and reserve separate quotas**

Change `SourceRegistry.admit()` to collect and return only the `SourceRecord` objects it creates:

```python
def admit(self, candidates: Sequence[ProviderSource]) -> tuple[SourceRecord, ...]:
    admitted: list[SourceRecord] = []
    for candidate in candidates:
        url = canonicalize_public_url(candidate.url)
        if url is None or url in self._by_url:
            continue
        if len(self._by_url) >= self._max_sources:
            continue
        record = SourceRecord(
            source_id=f"S{len(self._by_url) + 1}",
            url=url,
            title=candidate.title,
            snippet=candidate.snippet,
            status="search_result",
            published_at=candidate.published_at,
            provider=candidate.provider,
            query_index=candidate.query_index,
        )
        self._by_url[url] = record
        admitted.append(record)
    return tuple(admitted)
```

At session construction, set the complete registry ceiling to the two explicit quotas:

```python
self._text_source_capacity = self.limits.max_sources
self._image_catalog_capacity = (
    self.service.max_candidate_pool * max(1, int(self.budget.max_image_searches))
)
self.source_registry = SourceRegistry(
    max_sources=self._text_source_capacity + self._image_catalog_capacity
)
self._admitted_text_sources = 0
self._admitted_image_sources = 0
```

During `_search`, admit normalized image source pages first, capped by remaining image quota, then text pages capped by remaining text quota. Count newly admitted records, not raw provider records, so duplicates consume no quota. Remove the visual-intent `grow_capacity()` path; the session already owns its bounded catalog capacity.

- [ ] **Step 5: Add one explicit candidate-priority function**

Keep this in `service.py`; do not add a class or module:

```python
_CONFIDENCE_PRIORITY = {"high": 3, "medium": 2, "low": 1}


def _image_candidate_priority(
    candidate: ProviderImageCandidate,
    *,
    preferred: bool,
) -> tuple[int, int, int, int, int]:
    confidence = _CONFIDENCE_PRIORITY.get(str(candidate.confidence or "").lower(), 0)
    width = candidate.width or 0
    height = candidate.height or 0
    adequate = int(max(width, height) >= 640)
    area = width * height
    return (-int(preferred), -confidence, -adequate, -area, candidate.rank)
```

`_domain_matches` must use exact/subdomain semantics, matching Task 1. Small images lose a bucket but remain eligible. Provider rank is only the final tie-breaker.

- [ ] **Step 6: Implement catalog merge and active-window replacement**

Add session state using built-in dictionaries/sets, not a new abstraction:

```python
self._candidate_catalog: dict[tuple[str, str], ProviderImageCandidate] = {}
self._preferred_candidate_keys: set[tuple[str, str]] = set()
self._prepared_by_key: dict[tuple[str, str], PreparedImage] = {}
self._next_candidate_index = 1
self._latest_image_query: str | None = None
self._latest_image_objective: str | None = None
```

The catalog key is `(canonical source URL, image URL)`. For each image search:

1. Remember the literal `image_query` and `objective`.
2. Merge up to `max_candidate_pool` results from that cohort while total catalog capacity remains.
3. For candidates whose source host matches this cohort's `include_domains`, add
   their keys to `_preferred_candidate_keys`. This keeps the preference attached
   to the request that produced the candidate rather than globally boosting an
   unrelated old candidate when a later search names another domain.
4. Sort the whole catalog by `_image_candidate_priority`.
5. Walk the sorted list until `max_model_images` valid prepared images exist. Retain already-prepared keys without refetching; fetch newly active keys; after a fetch/validation failure, continue to the next ranked key.
6. Only after replacements are prepared, release references for keys outside the final active set, subtract their preview bytes from `_model_image_bytes`, and remove their digests. Keep `_downloaded_image_bytes` cumulative for the whole session so repeated replacement cannot bypass the download budget.
7. Allocate a new `I#` from `_next_candidate_index` for every newly prepared candidate and increment it permanently.

Store `catalog_key` on `PreparedImage`, and copy `source_domain` into `ImageCandidateRecord`. Keep `prepared_images` insertion order equal to final priority order so model evidence blocks and bundle images present the best candidates first.

Delete `pending_provider_images` and the early return based on `remaining == 0`; those two pieces are the root cause of later searches being unable to improve the window.

- [ ] **Step 7: Update bundle validation for a multi-cohort registry**

Keep these `WebEvidenceBundle` invariants:

- unique `S#` IDs;
- unique active `I#` IDs;
- every active image refers to a source in the complete registry;
- active image count respects `image_capacity()`.

Remove the `len(sources) <= source_capacity(...)` assertion from the Pydantic model. Runtime source capacity is now session-configured as text quota plus multi-cohort image catalog quota; duplicating a smaller one-cohort limit in the contract would reject valid bounded sessions. Add a contract test proving that a bundle may contain a larger stable registry while still rejecting too many active images and unknown source IDs.

- [ ] **Step 8: Verify catalog behavior and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_source_registry.py tests/test_web_research_contracts.py tests/test_web_image_capacity_and_resolution.py tests/test_web_research_images.py tests/test_web_grounding.py
.\.venv\Scripts\python.exe -m ruff check app/ai/web_research/contracts.py app/ai/web_research/source_registry.py app/ai/web_research/service.py tests/test_web_source_registry.py tests/test_web_image_capacity_and_resolution.py tests/test_web_research_images.py
git diff --check
```

Expected: all tests pass; full-resolution publication and protected delivery regressions remain green.

- [ ] **Step 9: Commit the catalog repair**

```powershell
git add -- app/ai/web_research/contracts.py app/ai/web_research/source_registry.py app/ai/web_research/service.py tests/test_web_source_registry.py tests/test_web_research_contracts.py tests/test_web_image_capacity_and_resolution.py tests/test_web_research_images.py
git commit -m "fix: rank web images across search cohorts"
```

---

### Task 3: Make tool evidence incremental and image selection literal

**Files:**
- Modify: `app/ai/web_research/contracts.py`
- Modify: `app/ai/web_research/service.py`
- Modify: `app/ai/web_research/model_context.py`
- Modify: `app/ai/web_tools.py`
- Modify: `tests/test_web_research_contracts.py`
- Modify: `tests/test_web_research_model_context.py`
- Modify: `tests/test_web_research_tool_session.py`
- Modify: `tests/test_web_tool_output_privacy.py`

**Interfaces:**
- Add `operation_source_ids: tuple[str, ...] = ()` to `WebEvidenceBundle`.
- `_bundle(..., operation_source_ids=...)` stores current-operation IDs while `sources` remains the complete registry.
- Expose read-only `latest_image_query` and `latest_image_objective` properties on `WebResearchSession`.
- Keep `_project()`'s JSON envelope stable; only its `sources` list becomes the operation delta and snippets become bounded.

- [ ] **Step 1: Write failing operation-delta tests**

In `tests/test_web_research_tool_session.py`, perform two searches through one session/tool context and assert:

```python
first = json.loads(await tool.ainvoke(first_args))
second = json.loads(await tool.ainvoke(second_args))

assert {item["source_id"] for item in first["sources"]} == {"S1", "S2"}
assert {item["source_id"] for item in second["sources"]} == {"S3"}
```

Then open `S1` and assert the open response includes the refreshed `S1` even though its original `query_index` is 1. This proves deltas are explicit and are not inferred from `query_index`.

In contract tests, assert `operation_source_ids` are unique and every ID exists in `sources`.

- [ ] **Step 2: Write failing compact-context and literal-selection tests**

Update the model-context session double to expose:

```python
latest_image_query = "full T1 League of Legends team photo"
latest_image_objective = "identify the complete current roster"
```

Assert the injected text:

- contains `S#: title | URL` mappings;
- does not contain an 800-character source snippet;
- contains the exact latest image query and says to match the requested visual form literally;
- explicitly distinguishes a full team photo from a roster graphic or list of names;
- still says selecting none is valid.

Update the prepared-image block test to assert a label shaped like:

```text
Image candidate I7; source S4; 1920x1080; domain www.t1.gg.
```

Do not generate descriptive alt text or infer a visual type.

- [ ] **Step 3: Run evidence tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_research_contracts.py tests/test_web_research_model_context.py tests/test_web_research_tool_session.py tests/test_web_tool_output_privacy.py
```

Expected: the second tool response repeats the whole registry, open cannot be represented as an explicit delta, source snippets repeat in synthetic context, and image labels omit dimensions/domain.

- [ ] **Step 4: Thread explicit operation source IDs through search and open**

Add the field and validator:

```python
operation_source_ids: tuple[str, ...] = ()

if len(self.operation_source_ids) != len(set(self.operation_source_ids)):
    raise ValueError("duplicate operation source ID")
if not set(self.operation_source_ids).issubset(known_sources):
    raise ValueError("operation source ID is not in sources")
```

Extend `_bundle`:

```python
def _bundle(
    self,
    operation_index: int,
    *,
    operation_source_ids: tuple[str, ...] = (),
    visual_intent: VisualIntent | None = None,
    failures: tuple[ResearchFailure, ...] = (),
    providers: tuple[str, ...] = (),
) -> WebEvidenceBundle:
```

For search, pass the IDs returned by the image-first and text-second `admit()` calls. For open, pass IDs of the records successfully opened in this operation, including records already present before the operation.

- [ ] **Step 5: Project only current-operation sources**

In `web_tools.py`, select the explicit delta while preserving complete bundles internally:

```python
operation_ids = set(bundle.operation_source_ids)
public_sources = [source for source in bundle.sources if source.source_id in operation_ids]
```

Project `public_sources` only. Bound each current snippet to 1,200 characters before JSON encoding so one verbose provider result cannot dominate the tool transcript:

```python
"snippet": (source.snippet or "")[:1200] or None,
```

The response still includes failures, reuse state, and omission counters.

- [ ] **Step 6: Compact synthetic context and expose literal image target**

Add simple properties on the session:

```python
@property
def latest_image_query(self) -> str | None:
    return self._latest_image_query

@property
def latest_image_objective(self) -> str | None:
    return self._latest_image_objective
```

In `model_context.py`, emit each source once as mapping-only text:

```python
lines.extend(
    f"{source.source_id}: {source.title or 'Untitled'} | {source.url}"
    for source in sources
)
```

When image blocks exist, append the exact target and selection rule:

```python
target = session.latest_image_query or session.latest_image_objective
lines.append(
    f"IMAGE TARGET: {target}. Match the requested visual form literally: a photo "
    "must be a photo, a close-up must show the named detail, and a settings "
    "screenshot must show the requested control. A roster graphic or list of "
    "names is not a full team photo. Select none if no candidate visibly matches."
)
```

In `model_evidence_blocks`, include only trusted normalized metadata in the label:

```python
domain = record.source_domain or "unknown"
label = (
    f"Image candidate {candidate_id}; source {record.source_id}; "
    f"{record.width}x{record.height}; domain {domain}."
)
```

- [ ] **Step 7: Verify evidence compactness, privacy, and grounding**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_research_contracts.py tests/test_web_research_model_context.py tests/test_web_research_tool_session.py tests/test_web_tool_output_privacy.py tests/test_web_grounding.py tests/test_web_research_images.py
.\.venv\Scripts\python.exe -m ruff check app/ai/web_research/contracts.py app/ai/web_research/service.py app/ai/web_research/model_context.py app/ai/web_tools.py tests/test_web_research_model_context.py tests/test_web_research_tool_session.py tests/test_web_tool_output_privacy.py
git diff --check
```

Expected: all tests pass; candidate bytes and original image URLs remain absent from public tool output.

- [ ] **Step 8: Commit compact, literal evidence delivery**

```powershell
git add -- app/ai/web_research/contracts.py app/ai/web_research/service.py app/ai/web_research/model_context.py app/ai/web_tools.py tests/test_web_research_contracts.py tests/test_web_research_model_context.py tests/test_web_research_tool_session.py tests/test_web_tool_output_privacy.py
git commit -m "fix: deliver focused web image evidence"
```

---

### Task 4: Lock the three reported visual cases into acceptance coverage

**Files:**
- Add: `tests/test_web_image_relevance_regressions.py`
- Modify: `docs/operations/web-research-rollout.md`

**Acceptance cases:**
- T1: small roster cards plus a later official full-team photo.
- Character hair: a high-confidence 1024×576 close-up remains selectable.
- Windows: a settings-menu screenshot beats unrelated product imagery.

- [ ] **Step 1: Add deterministic end-to-end regression fixtures in code**

Create candidates with no network dependency and route them through `WebResearchService`, the fake image service, `inject_latest_web_evidence`, and `GroundingParser`.

For T1, assert that after two searches:

```python
assert active[0].source_domain == "www.t1.gg"
assert (active[0].width, active[0].height) == (1920, 1080)
assert "full T1 League of Legends team photo" in injected_text
```

For the hair and Windows cases, assert the high-confidence, adequate-resolution candidate is active and that the literal target appears unchanged in model context. Also assert selecting its `I#` resolves to a rich item with the full-resolution record, while selecting an evicted `I#` produces no image.

- [ ] **Step 2: Run the new acceptance tests and verify they pass only through production code**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_image_relevance_regressions.py
```

Expected: all three cases pass without monkeypatching the ranking function or inspecting private implementation helpers.

- [ ] **Step 3: Document the runtime behavior and bounded costs**

Update `docs/operations/web-research-rollout.md` with:

- Brave locale forwarding and strict domain semantics;
- the per-cohort candidate cap and session catalog bound;
- active vision-window limits (4 for figure/comparison, 6 for gallery);
- operation-delta tool output versus complete stable grounding registry;
- logs/metrics to inspect for optional domain-probe failures, omitted candidates, and image fetch rejection.

Do not document a fallback that publishes provider links as images; publication still requires a validated, model-selected candidate.

- [ ] **Step 4: Run the complete focused web-research suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_research_providers.py tests/test_web_source_registry.py tests/test_web_research_contracts.py tests/test_web_image_capacity_and_resolution.py tests/test_web_research_images.py tests/test_web_research_model_context.py tests/test_web_research_service.py tests/test_web_research_tool_session.py tests/test_web_tool_output_privacy.py tests/test_web_grounding.py tests/test_web_image_relevance_regressions.py
.\.venv\Scripts\python.exe -m ruff check app/ai/web_research app/ai/web_tools.py tests/test_web_research_providers.py tests/test_web_source_registry.py tests/test_web_research_contracts.py tests/test_web_image_capacity_and_resolution.py tests/test_web_research_images.py tests/test_web_research_model_context.py tests/test_web_research_tool_session.py tests/test_web_tool_output_privacy.py tests/test_web_image_relevance_regressions.py
git diff --check
```

Expected: all focused tests and static checks pass.

- [ ] **Step 5: Commit acceptance coverage and operations notes**

```powershell
git add -- tests/test_web_image_relevance_regressions.py docs/operations/web-research-rollout.md
git commit -m "test: cover web image relevance regressions"
```

---

### Task 5: Final verification and review

**Files:**
- Review all files changed by Tasks 1–4.

- [ ] **Step 1: Run the broader adjacent suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_*.py tests/test_message_service_web_image_externalization.py tests/test_ai_sdk_v6_stream_contract.py tests/test_rich_response_metadata.py
```

Expected: all tests pass. Investigate any failure before proceeding; do not label an unrelated failure without reproducing it on the base commit.

- [ ] **Step 2: Re-run static and repository checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check app/ai/web_research app/ai/web_tools.py tests/test_web_*.py
git diff --check
git status --short
git log --oneline -5
```

Expected: Ruff and diff checks pass; only intentional changes remain; the four implementation commits are visible.

- [ ] **Step 3: Request code review**

Use `superpowers:requesting-code-review`. Require the review to verify:

- `include_domains` never leaks disallowed results;
- candidate IDs cannot be rebound or reused;
- evicted pending references are released exactly once;
- text/image source quotas remain bounded independently;
- tool deltas do not remove sources needed by grounding;
- image bytes and source-image URLs do not leak in tool JSON;
- no taxonomy/ranking framework or answer-time fallback was introduced.

- [ ] **Step 4: Apply valid review findings and repeat verification**

Use `superpowers:receiving-code-review` for any findings. Add a regression test before each behavior fix, rerun the relevant focused command, then repeat Steps 1–2.

- [ ] **Step 5: Finish the development branch**

Use `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch`. Report exact test counts and the commit list; do not claim completion from stale output.
