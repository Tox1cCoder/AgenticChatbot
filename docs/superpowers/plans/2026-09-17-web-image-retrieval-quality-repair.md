# Web Image Retrieval Quality Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make image search return the requested visual itself, keep the best candidates across repeated searches, and pass compact non-duplicated evidence to the answer model.

**Architecture:** Preserve the existing provider/session/grounding pipeline. Improve Brave request fidelity at the provider boundary; turn the session's append-only prepared-image list into a bounded candidate catalog plus replaceable active vision window; and distinguish three audiences for the source registry — the complete registry (grounding and citation resolution), the current operation's delta (tool output), and the answer-relevant subset (what the user is shown). Relevance ordering stays as one small function in `service.py`, with no new ranking service, visual taxonomy, or answer-time fallback.

**What this plan does and does not do about relevance.** Candidate ordering is a *quality* ordering: resolution adequacy, provider confidence, requested-domain provenance, pixel area, provider rank. Nothing here can tell a full team photo from a 1920x1080 roster infographic, and a large graphic will outrank a smaller photo. Matching the requested *visual form* is the answer model's judgment, made from the pixels it is shown plus the explicit `IMAGE TARGET` line added in Task 3. What this plan buys is that the good candidate is present in the vision window at all — which today it frequently is not. Task 5 encodes that boundary as a negative control so no future reader mistakes the ranking function for a relevance oracle.

**Tech Stack:** Python 3.10+, asyncio, Pydantic v2, LangChain message blocks, pytest, pytest-asyncio.

## Global Constraints

- Do not add `visual_kind`, `alt_text` routing, a query-rewrite model call, or a generic ranking framework.
- Host normalization and host matching live once, in `app/ai/web_query_contract.py`, and are shared by the provider and the session. Only Brave's *request shaping* (locale argument names, `site:` probe syntax) belongs in `providers.py`.
- Never trust `ResearchRequest.include_domains` as written: it carries whatever the model typed. Normalize every entry through `bare_host` before comparing or probing.
- Treat `include_domains` as a restriction at the provider: never return a disallowed-domain image. Inside the session it is only a provenance *tiebreaker*, and it never outranks resolution adequacy.
- Preserve original-resolution publication, protected `/web-images/{id}` delivery, and model-only previews.
- Keep stable `S#` IDs and never bind an existing `I#` to different bytes.
- Keep the complete source registry for validation and grounding; expose only the current operation's delta in tool output; publish only answer-relevant sources to the user.
- Admit image sources before text sources and enforce separate bounded quotas so either class cannot starve the other.
- Penalize small images; do not hard-reject them when they are the only relevant evidence.
- Use `.venv\Scripts\python.exe` for verification in this checkout.

## Baseline

The nine focused web-research test files pass at `47bef3cc` (91 tests). Any failure after a task is that task's, not a pre-existing one.

---

### Task 1: Preserve image-provider locale, domain constraints, and quality metadata

**Files:**
- Modify: `app/ai/web_query_contract.py`
- Modify: `app/ai/web_research/contracts.py`
- Modify: `app/ai/web_research/providers.py`
- Modify: `tests/test_web_research_providers.py`
- Modify: `tests/test_web_query_contract.py`
- Add: `tests/fixtures/web_research/brave_images_mixed_domains.json`
- Add: `tests/fixtures/web_research/brave_images_t1_official.json`

**Interfaces:**
- Promote `web_query_contract._bare_host` to public `bare_host`, and add `host_matches(host, allowed)` beside it. Both are exported.
- Extend `ProviderImageCandidate` with optional `confidence` and `source_domain` fields.
- Keep `ImageSearchProvider.search(request)` unchanged.
- `BraveImageSearchProvider` still returns one provider-neutral tuple; internally it may make **one** bounded domain probe.

**Why one probe, not three.** Brave's image endpoint has a 2.5s timeout (`brave_image_search_timeout_seconds`) and the free tier allows roughly one request per second, while the whole tool call has a 30s soft timeout shared with the parallel Tavily text search. Four sequential calls spend up to 10s and risk HTTP 429; issuing them concurrently trades the latency for the rate limit. One supplemental probe against the first allowed domain covers the real case — the model names one authoritative site — at a worst case of two calls and ~5s.

- [x] **Step 1: Add failing host-matching tests in the query contract**

In `tests/test_web_query_contract.py`, assert the shared helpers:

```python
assert bare_host("https://T1.gg/roster?x=1") == "t1.gg"
assert bare_host("WWW.T1.GG") == "www.t1.gg"
assert host_matches("www.t1.gg", "t1.gg") is True
assert host_matches("t1.gg", "https://www.t1.gg/") is True
assert host_matches("not-t1.gg", "t1.gg") is False
assert host_matches("", "t1.gg") is False
```

`host_matches` normalizes **both** sides through `bare_host`, then strips a leading `www.` from the allowed side, so an entry the model wrote as a URL still matches.

- [x] **Step 2: Add failing adapter tests for locale forwarding and domain enforcement**

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

Add a test using `locale="vi-VN"` and `include_domains=("https://T1.gg/",)` — deliberately written as a URL, because that is what a model actually sends — where the broad fixture includes Reddit and the second fixture includes an official T1 result. Assert:

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

- [x] **Step 3: Add failing tests for bounded probing and deduplication**

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

- The adapter makes the ordinary query plus **at most one** `site:` probe, against the first allowed domain, and only when the ordinary call returned fewer than `count` allowed results.
- A source host may equal the allowed domain or be its subdomain.
- A repeated `(canonical source URL, image URL)` appears once, preserving first-provider order.
- Failure of the optional probe does not erase already valid allowed-domain results.
- An `include_domains` entry the model wrote as a full URL still restricts and still probes correctly.
- With no `include_domains`, the adapter makes exactly one call and preserves current behavior.

- [x] **Step 4: Run provider and contract tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_research_providers.py tests/test_web_query_contract.py
```

Expected: `bare_host`/`host_matches` do not exist, locale and domain constraints are not forwarded, no supplemental probe occurs, and quality metadata is discarded.

- [x] **Step 5: Publish the shared host helpers**

In `web_query_contract.py`, rename `_bare_host` to `bare_host`, update its internal callers, and add:

```python
def host_matches(host: str, allowed: str) -> bool:
    """Whether ``host`` is ``allowed`` or one of its subdomains."""

    normalized = bare_host(host).rstrip(".")
    target = bare_host(allowed).removeprefix("www.").rstrip(".")
    if not normalized or not target:
        return False
    return normalized == target or normalized.endswith(f".{target}")
```

Add both names to `__all__`.

- [x] **Step 6: Extend the provider-neutral candidate contract**

Add only these two fields to the current contract; leave all current fields in
place:

```python
confidence: str | None = Field(default=None, max_length=32)
source_domain: str | None = Field(default=None, max_length=253)
```

Do not introduce a confidence enum: Brave values may evolve, while the session ranking deliberately maps only known values and treats everything else as unknown.

- [x] **Step 7: Implement locale parsing and strict domain filtering in the Brave adapter**

Add a module logger to `providers.py` (it currently has none):

```python
import logging

logger = logging.getLogger(__name__)
```

Add one small private function, and import the shared host helpers:

```python
from app.ai.web_query_contract import (
    NormalizedWebSearch,
    bare_host,
    host_matches,
    tavily_search_args,
)


def _brave_locale(locale: str | None) -> dict[str, str]:
    parts = str(locale or "").replace("_", "-").split("-")
    args: dict[str, str] = {}
    if parts and len(parts[0]) == 2:
        args["search_lang"] = parts[0].lower()
    if len(parts) > 1 and len(parts[1]) == 2:
        args["country"] = parts[1].upper()
    return args
```

In `BraveImageSearchProvider.search`:

1. Call Brave once with the untouched image query, safesearch, count, and locale args.
2. Normalize the allowed domains once with `bare_host`, dropping empties.
3. If restrictions exist, keep only results whose source host satisfies `host_matches`.
4. If fewer than `count` matching results remain, make one more call with `f"{query} site:{allowed[0]}"` — using the *normalized* host, never the raw entry.
5. Normalize and deduplicate the combined result stream by canonical source URL plus image URL.
6. Return only allowed hosts when restrictions were supplied.

Use `urlsplit` for host extraction and `canonicalize_public_url` for the source half of the dedupe key. Log the optional probe's failure at warning level with provider/domain/code; allow the first ordinary-call failure to keep its existing `ProviderFailure` behavior.

Add these arguments to the current `ProviderImageCandidate(...)` construction,
without interpreting them:

```python
confidence=str(raw["confidence"]).lower() if raw.get("confidence") else None,
source_domain=str(
    raw.get("source_domain") or urlsplit(str(source_url)).hostname or ""
) or None,
```

- [x] **Step 8: Verify provider behavior and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_research_providers.py tests/test_web_query_contract.py tests/test_web_image_capacity_and_resolution.py -k "brave or provider or host"
.\.venv\Scripts\python.exe -m ruff check app/ai/web_query_contract.py app/ai/web_research/contracts.py app/ai/web_research/providers.py tests/test_web_research_providers.py tests/test_web_query_contract.py
git diff --check
```

Expected: all selected tests and checks pass.

- [x] **Step 9: Commit provider fidelity**

```powershell
git add -- app/ai/web_query_contract.py app/ai/web_research/contracts.py app/ai/web_research/providers.py tests/test_web_research_providers.py tests/test_web_query_contract.py tests/fixtures/web_research/brave_images_mixed_domains.json tests/fixtures/web_research/brave_images_t1_official.json
git commit -m "fix: preserve image search constraints"
```

---

### Task 2: Replace the append-only image list with a bounded quality catalog

**Files:**
- Modify: `app/core/config.py`
- Modify: `app/core/container.py`
- Modify: `app/ai/web_research/contracts.py`
- Modify: `app/ai/web_research/service.py`
- Modify: `app/ai/web_research/policy.py`
- Modify: `app/ai/web_research/source_registry.py`
- Modify: `tests/test_web_research_config.py`
- Modify: `tests/test_web_source_registry.py`
- Modify: `tests/test_web_research_contracts.py`
- Modify: `tests/test_web_image_capacity_and_resolution.py`
- Modify: `tests/test_web_research_images.py`
- Modify: `tests/test_web_research_service.py`

**Interfaces:**
- `SourceRegistry.admit()` returns only records newly admitted by that call; `records` remains the complete stable registry.
- `PreparedImage` gains an internal immutable `catalog_key`.
- `ImageCandidateRecord` gains optional `source_domain` for model-visible labeling.
- `WebResearchSession.prepared_images` remains the active `I# -> PreparedImage` map consumed by grounding and closeout.
- New setting `web_research_max_candidate_catalog`, threaded through the container like `web_research_max_candidate_pool`.

**Why an explicit catalog setting.** The obvious source for this bound, `ResearchBudget.max_image_searches`, does not bound anything: `reserve_image_search`, `may_image_search`, and `record_image_search` have **no production callers** — only `tests/test_research_budget.py` — so nothing caps how many image searches a turn makes. Deriving catalog capacity from it would dress an unenforced number as a budget. Configure the catalog directly instead. Wiring the image-search budget is out of scope for this plan.

- [x] **Step 1: Write the failing source-registry delta test**

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

No production caller and no existing test reads `admit()`'s return value for the complete set — all three call sites in `service.py` discard it (the fourth caller is `SourceRegistry.import_records`, which re-resolves each URL), and both existing registry tests already index `[0]`. Nothing else needs updating.

- [x] **Step 2: Write failing active-window regression tests**

Build deterministic fake image providers/candidates in `tests/test_web_image_capacity_and_resolution.py`. Cover all of these in two searches on one session:

- Search 1 returns small 140x140 roster cards plus a 1200x675 full-team photo.
- The full-team photo ranks ahead of the small cards even when its provider rank is later.
- Search 2 returns worse, low-confidence 100x100 results; the good prior photo stays active.
- A separate search 2 returns a 1920x1080 preferred-domain official photo; it enters the active window and the weakest prepared candidate is released.
- Retained prepared images keep their original `I#`; the replacement receives a fresh, never-reused ID.
- The evicted image's pending reference is released and cannot be selected through grounding.
- A saturated text result set does not consume the image catalog's reserved source capacity.

**Give the second search a distinct scope or a genuinely different query.** `ResearchBudget.reserve_search` refuses a near-duplicate — token overlap at or above `near_duplicate_threshold` (0.75) — unless the scope tuple `(freshness, start_date, end_date, include_domains)` differs. A second search that only rewords the first is refused with `duplicate_query` and the test will fail for a reason that has nothing to do with the catalog. Adding `include_domains` to search 2 changes the scope and is the natural shape for these cases anyway.

Use fake bytes and the existing recording image service; do not make network calls.

- [x] **Step 3: Update the existing capacity assertions this task invalidates**

Two assertions in `tests/test_web_image_capacity_and_resolution.py` encode the old one-cohort arithmetic and must move with the design, not be worked around:

- `test_a_full_text_search_does_not_starve_image_candidates` asserts `len(bundle.sources) == 5 + 4`. With separate quotas the registry now holds every admitted image page: 5 text + 6 image = **11**. Keep the test's point by asserting that all five text sources *and* all six image pages are present, rather than a bare total.
- `test_over_supply_is_counted_but_is_not_a_failure` asserts `omitted_image_count == 4` at `image_count=8`. Under the catalog those four are *retained for a later window*, not omitted, so the count is now 0. Preserve the test's intent by raising the fixture to `image_count=12`, which exceeds `max_candidate_pool` (8) and omits 4 at the catalog boundary.

**Define the omission counter once, here.** `omitted_image_count` means *candidates this session will never offer the model*: candidates beyond `max_candidate_pool` for their cohort, candidates past the catalog capacity, candidates whose source page could not be admitted, candidates whose fetch or validation failed, and duplicate digests. A candidate held in the catalog but outside the current active window is **not** omitted. Record this sentence as a comment above the counter.

- [x] **Step 4: Run catalog tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_source_registry.py tests/test_web_image_capacity_and_resolution.py tests/test_web_research_images.py
```

Expected: repeated searches cannot improve a full first window, provider rank wins over quality, and `admit()` returns the whole registry. Note that "IDs can be reused through `len(prepared_images) + 1`" is **not** a current failure — nothing removes from `prepared_images` today, so IDs are contiguous. It is a defect this task's eviction would introduce, which Step 8 must prevent; write it as a regression test, not as a RED expectation.

- [x] **Step 5: Add the catalog capacity setting**

In `config.py`, beside `web_research_max_candidate_pool`:

```python
web_research_max_candidate_catalog: int = Field(default=24, ge=4, le=64)
```

Thread it through `container.py` as `max_candidate_catalog=providers.Object(settings.web_research_max_candidate_catalog)`, accept it in `WebResearchService.__init__` as `max_candidate_catalog: int = 24` stored with `max(1, int(...))`, and add `assert fields["web_research_max_candidate_catalog"].default == 24` to `tests/test_web_research_config.py`.

- [x] **Step 6: Make source admission delta-aware and reserve separate quotas**

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
self._image_catalog_capacity = self.service.max_candidate_catalog
self.source_registry = SourceRegistry(
    max_sources=self._text_source_capacity + self._image_catalog_capacity
)
self._admitted_text_sources = 0
self._admitted_image_sources = 0
self._image_only_source_ids: set[str] = set()
self._omitted_source_count = 0
```

During `_search`, admit normalized image source pages first, then text pages, each capped by its own remaining quota:

```python
image_room = max(0, self._image_catalog_capacity - self._admitted_image_sources)
image_admitted = self.source_registry.admit(image_sources[:image_room])
self._admitted_image_sources += len(image_admitted)
self._image_only_source_ids.update(record.source_id for record in image_admitted)

text_room = max(0, self._text_source_capacity - self._admitted_text_sources)
text_admitted = self.source_registry.admit(tuple(text_records)[:text_room])
self._admitted_text_sources += len(text_admitted)
for candidate in text_records:
    record = self.source_registry.resolve(candidate.url)
    if record is None:
        self._omitted_source_count += 1
    else:
        # A page both a text search and an image cohort returned is a text
        # source: the image cohort merely got there first.
        self._image_only_source_ids.discard(record.source_id)
```

Pass `self._omitted_source_count` into `_bundle` as `omitted_source_count`; it is currently hardcoded to 0 and a text quota that can now reject results makes that a lie.

Remove the visual-intent `grow_capacity()` path; the session already owns its bounded catalog capacity.

- [x] **Step 7: Add one explicit candidate-priority function**

Keep this in `service.py`; do not add a class or module:

```python
_CONFIDENCE_PRIORITY = {"high": 3, "medium": 2, "low": 1}


def _image_candidate_priority(
    candidate: ProviderImageCandidate,
    *,
    preferred: bool,
) -> tuple[int, int, int, int, int]:
    """Order candidates by usable quality. Resolution adequacy dominates.

    ``preferred`` sits *below* adequacy on purpose. Task 1 filters a
    domain-restricted cohort strictly, so every candidate it produced is
    preferred; if the flag led, any candidate from a restricted search would
    outrank every candidate from an unrestricted one, and a 200x150 logo from
    the named site would displace a 4000x3000 photo. Provenance breaks ties
    between comparable images; it does not buy a small one a window slot.
    """

    confidence = _CONFIDENCE_PRIORITY.get(str(candidate.confidence or "").lower(), 0)
    width = candidate.width or 0
    height = candidate.height or 0
    adequate = int(max(width, height) >= 640)
    area = width * height
    return (-adequate, -confidence, -int(preferred), -area, candidate.rank)
```

Small images lose the leading bucket but remain eligible. Provider rank is only the final tie-breaker; across cohorts, equal tuples fall back to catalog insertion order, which is why the catalog must be an ordinary insertion-ordered `dict`.

- [x] **Step 8: Implement catalog merge and active-window replacement**

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
2. Merge up to `max_candidate_pool` results from that cohort while total catalog capacity remains. Count anything dropped here into `omitted_image_count`.
3. For candidates whose source host satisfies `host_matches` against this cohort's normalized `include_domains`, add their keys to `_preferred_candidate_keys`. This keeps the preference attached to the request that produced the candidate rather than globally boosting an unrelated old candidate when a later search names another domain.
4. Sort the whole catalog with `sorted(self._candidate_catalog.items(), key=lambda item: _image_candidate_priority(item[1], preferred=item[0] in self._preferred_candidate_keys))`.
5. Walk the sorted list until `max_model_images` valid prepared images exist. Retain already-prepared keys without refetching; fetch newly active keys; after a fetch/validation failure, continue to the next ranked key.
6. Only after replacements are prepared, release references for keys outside the final active set, subtract their preview bytes from `_model_image_bytes`, and remove their digests. Coerce the scope exactly as `finish`/`abort` do — `user_id=UUID(self.scope.user_id)`, `conversation_id=UUID(self.scope.conversation_id)` — because `ResearchScope` holds strings. Keep `_downloaded_image_bytes` cumulative for the whole session so repeated replacement cannot bypass the download budget.
7. Allocate a new `I#` from `_next_candidate_index` for every newly prepared candidate and increment it permanently. Never derive an ID from `len(prepared_images)` again: with eviction that rebinds a retired ID to different bytes.

**Replace the `image_source_capacity` signal.** That failure is currently computed from `pending_provider_images`, which this step deletes. Emit it from the new merge instead: when a cohort returned candidates but **none** entered the catalog — every source page unadmittable, or the catalog already full — append `ResearchFailure(operation="image_search", provider=<cohort provider>, code="image_source_capacity", retryable=False)`. Over-supply stays a counter, never a failure; losing the whole cohort stays a failure. `test_losing_every_candidate_is_reported` asserts exactly this.

Store `catalog_key` on `PreparedImage`, and copy `source_domain` into `ImageCandidateRecord`. Keep `prepared_images` insertion order equal to final priority order so model evidence blocks and bundle images present the best candidates first.

Delete `pending_provider_images` and the early return based on `remaining == 0`; those two pieces are the root cause of later searches being unable to improve the window.

- [x] **Step 9: Update bundle validation for a multi-cohort registry**

Keep these `WebEvidenceBundle` invariants:

- unique `S#` IDs;
- unique active `I#` IDs;
- every active image refers to a source in the complete registry;
- active image count respects `image_capacity()`.

Remove the `len(sources) <= source_capacity(...)` assertion from the Pydantic model. Runtime source capacity is now session-configured as text quota plus catalog quota; duplicating a smaller one-cohort limit in the contract would reject valid bounded sessions. Add a contract test proving that a bundle may contain a larger stable registry while still rejecting too many active images and unknown source IDs.

- [x] **Step 10: Delete what the new capacity model orphans**

These are unreferenced after Step 6 and Ruff will not flag them, because they are public:

- `SourceRegistry.grow_capacity`
- `SourceRegistry.free_slots`
- `SourceRegistry.capacity`
- `ResearchLimits.max_registry_sources`

`free_slots` and `capacity` already have no callers today. Delete all four and confirm with `grep -rn "grow_capacity\|free_slots\|max_registry_sources\|\.capacity" app/ tests/`. `contracts.source_capacity` stays — `ResearchLimits.for_mode` still uses it.

- [x] **Step 11: Verify catalog behavior and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_source_registry.py tests/test_web_research_contracts.py tests/test_web_research_config.py tests/test_web_image_capacity_and_resolution.py tests/test_web_research_images.py tests/test_web_grounding.py
.\.venv\Scripts\python.exe -m ruff check app/core/config.py app/core/container.py app/ai/web_research tests/test_web_source_registry.py tests/test_web_image_capacity_and_resolution.py tests/test_web_research_images.py
git diff --check
```

Expected: all tests pass; full-resolution publication and protected delivery regressions remain green.

- [x] **Step 12: Commit the catalog repair**

```powershell
git add -- app/core/config.py app/core/container.py app/ai/web_research/contracts.py app/ai/web_research/policy.py app/ai/web_research/source_registry.py app/ai/web_research/service.py tests/test_web_research_config.py tests/test_web_source_registry.py tests/test_web_research_contracts.py tests/test_web_image_capacity_and_resolution.py tests/test_web_research_images.py
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
- Keep `_project()`'s JSON envelope stable; its `sources` list becomes the operation delta, snippets become bounded by status, and two counters are added so an empty delta is readable.

**Snippet bounds are per status, not one number.** `web_open` exists to read pages "whose search snippets were insufficient"; capping its result at the same bound as a triage snippet would gut the deep-read path. Use 1,200 characters for `search_result` and 3,000 for `opened`. Worst case for one operation is an agentic open of 4 pages at 3,000 = 12,000 characters, which stays under `tool_result_offload_threshold_chars` (16,000) so the deep read is not pushed out of the transcript into a blob.

- [x] **Step 1: Write failing operation-delta tests**

In `tests/test_web_research_tool_session.py`, perform two searches through one session/tool context and assert:

```python
first = json.loads(await tool.ainvoke(first_args))
second = json.loads(await tool.ainvoke(second_args))

assert {item["source_id"] for item in first["sources"]} == {"S1", "S2"}
assert {item["source_id"] for item in second["sources"]} == {"S3"}
```

Then open `S1` and assert the open response includes the refreshed `S1` even though its original `query_index` is 1. This proves deltas are explicit and are not inferred from `query_index`.

**All three existing `SimpleNamespace` bundle doubles must gain `operation_source_ids`** — two in this file (`test_web_search_uses_the_exact_session_from_tool_context` *and* `test_web_open_uses_the_same_turn_session`, whose bundle also passes through `_project`) and one in `tests/test_web_tool_output_privacy.py`. Without it `_project` raises `AttributeError`, and the search test additionally asserts `public["sources"][0]["source_id"] == "S1"`, which an empty delta fails. Set `operation_source_ids=("S1",)` on the search double and `()` on the other two, whose bundles have no sources anyway.

In contract tests, assert `operation_source_ids` are unique and every ID exists in `sources`.

- [x] **Step 2: Write failing compact-context and literal-selection tests**

Update the model-context session double to expose:

```python
latest_image_query = "full T1 League of Legends team photo"
latest_image_objective = "identify the complete current roster"
```

Assert the injected text:

- contains `S#: title | URL` mappings;
- does not repeat an 800-character snippet for a `search_result` source;
- **does** carry a bounded excerpt for an `opened` source, because a deliberately read page must survive tool-output offload;
- contains the exact latest image query and says to match the requested visual form literally;
- explicitly distinguishes a full team photo from a roster graphic or list of names;
- still says selecting none is valid.

Update the prepared-image block test to assert a label shaped like:

```text
Image candidate I7; source S4; 1920x1080; domain www.t1.gg.
```

Do not generate descriptive alt text or infer a visual type.

- [x] **Step 3: Run evidence tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_research_contracts.py tests/test_web_research_model_context.py tests/test_web_research_tool_session.py tests/test_web_tool_output_privacy.py
```

Expected: the second tool response repeats the whole registry, open cannot be represented as an explicit delta, search snippets repeat in synthetic context, and image labels omit dimensions/domain.

- [x] **Step 4: Thread explicit operation source IDs through search and open**

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

This *adds to* Task 2 Step 6's change; it does not replace it. The body still
passes `omitted_source_count=self._omitted_source_count`.

For search, pass the IDs of `image_admitted` followed by `text_admitted` from Task 2 Step 6. For open, collect the IDs from the `mark_opened` loop — records successfully opened in this operation, including records already present before it — not from `admit()`'s delta, which by definition excludes them.

Leave `status` computed from the complete registry. It answers "does this session have evidence", which is the question the caller asks; Step 5 makes an empty delta readable without overloading it.

- [x] **Step 5: Project only current-operation sources**

In `web_tools.py`, select the explicit delta while preserving complete bundles internally:

```python
operation_ids = set(bundle.operation_source_ids)
public_sources = [source for source in bundle.sources if source.source_id in operation_ids]
```

Project `public_sources` only, bounding each snippet by the source's status so one verbose provider result cannot dominate the tool transcript while a deliberate page read survives it:

```python
_SNIPPET_BOUNDS = {"opened": 3000}

"snippet": (source.snippet or "")[: _SNIPPET_BOUNDS.get(source.status, 1200)] or None,
```

Add two counters so `{"status":"success","sources":[]}` is not a riddle — that shape is legitimate when a search returns only pages the session already knows:

```python
"new_source_count": len(public_sources),
"total_source_count": len(bundle.sources),
```

The response still includes failures, reuse state, and omission counters.

- [x] **Step 6: Compact synthetic context and expose literal image target**

Add simple properties on the session:

```python
@property
def latest_image_query(self) -> str | None:
    return self._latest_image_query

@property
def latest_image_objective(self) -> str | None:
    return self._latest_image_objective
```

In `model_context.py`, keep reading `session.source_registry.records` — the model must be able to cite any admitted `S#`, and image candidates name image-page sources — but emit each one as a mapping line, appending an excerpt only for a page the model deliberately opened:

```python
def _source_line(source: Any) -> str:
    head = f"{source.source_id}: {source.title or 'Untitled'} | {source.url}"
    if source.status != "opened" or not source.snippet:
        return head
    return f"{head}\n{source.snippet[:1500]}"


lines.extend(_source_line(source) for source in sources)
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

This line is the *only* thing in the system that judges visual form. Ordering is quality-only by design; do not add a taxonomy here to compensate.

In `model_evidence_blocks`, include only trusted normalized metadata in the label:

```python
domain = record.source_domain or "unknown"
label = (
    f"Image candidate {candidate_id}; source {record.source_id}; "
    f"{record.width}x{record.height}; domain {domain}."
)
```

- [x] **Step 7: Verify evidence compactness, privacy, and grounding**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_research_contracts.py tests/test_web_research_model_context.py tests/test_web_research_tool_session.py tests/test_web_tool_output_privacy.py tests/test_web_grounding.py tests/test_web_research_images.py
.\.venv\Scripts\python.exe -m ruff check app/ai/web_research/contracts.py app/ai/web_research/service.py app/ai/web_research/model_context.py app/ai/web_tools.py tests/test_web_research_model_context.py tests/test_web_research_tool_session.py tests/test_web_tool_output_privacy.py
git diff --check
```

Expected: all tests pass; candidate bytes and original image URLs remain absent from public tool output.

- [x] **Step 8: Commit compact, literal evidence delivery**

```powershell
git add -- app/ai/web_research/contracts.py app/ai/web_research/service.py app/ai/web_research/model_context.py app/ai/web_tools.py tests/test_web_research_contracts.py tests/test_web_research_model_context.py tests/test_web_research_tool_session.py tests/test_web_tool_output_privacy.py
git commit -m "fix: deliver focused web image evidence"
```

---

### Task 4: Keep the published source list answer-relevant

**Files:**
- Modify: `app/ai/web_research/service.py`
- Modify: `app/ai/workflow/specialists.py`
- Add: `tests/test_web_published_source_scope.py`

**The problem this task exists to prevent.** `specialists.py` publishes **every** `source_registry.records` entry to the user in three places: the citation-required check, the worker's `evidence`, and `metadata["web_sources"]` (which also feeds the `sources` stream event and message history). Task 2 raises registry capacity from `text + max_model_images` (at most 11 quick / 14 agentic, both at `gallery`) to `text + max_candidate_catalog` (5 + 24 = **29** quick, 32 agentic). Without this task, a gallery turn shows the reader roughly two dozen image-host pages it never cited. Task 3's delta fixes the *tool* transcript only; this fixes the *published* list.

**Three audiences, three views.** The complete registry stays complete for grounding — `GroundingParser` resolves any `[[source:S#]]` the model was offered, and `model_context` still lists every admitted `S#`. Task 3's `operation_source_ids` is what the tool returns. `answer_sources` is what a human sees.

- [x] **Step 1: Write the failing scope test**

In `tests/test_web_published_source_scope.py`, run one session with a saturating text provider and an image cohort whose pages share no host with the text results, then assert:

- `len(session.source_registry.records)` exceeds `len(session.answer_sources)`;
- every text page appears in `answer_sources`;
- exactly the image pages backing active `prepared_images` appear, and no others;
- after an eviction replaces an active image, the evicted image's page leaves `answer_sources` while its `S#` still resolves through `source_registry.resolve` and still grounds a `[[source:S#]]` token;
- a page returned by **both** a text search and an image cohort stays in `answer_sources` even though the image cohort admitted it first;
- an image page the model explicitly opened with `web_open` stays in `answer_sources` whether or not its image is still active.

Assert the specialist wiring by reading `specialists.py` and confirming `source_registry.records` no longer appears in the three publication sites, in the spirit of `tests/test_web_research_active_path_inventory.py`.

- [x] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_published_source_scope.py
```

Expected: `answer_sources` does not exist.

- [x] **Step 3: Add the answer-relevant view**

On `WebResearchSession`, using the `_image_only_source_ids` set Task 2 Step 6 already maintains:

```python
@property
def answer_sources(self) -> tuple[SourceRecord, ...]:
    """Sources worth showing a reader: text pages, opened pages, cited image pages.

    The registry keeps every image-cohort page so grounding can resolve any
    ``S#`` the model was offered. Most of those pages are gallery hosts whose
    candidate never entered the vision window, and publishing them buries the
    handful of sources the answer actually rests on.
    """

    active = {prepared.record.source_id for prepared in self.prepared_images.values()}
    return tuple(
        record
        for record in self.source_registry.records
        if record.source_id not in self._image_only_source_ids
        or record.source_id in active
        or record.status == "opened"
    )
```

- [x] **Step 4: Publish the narrowed view**

In `specialists.py`, replace `web_research_session.source_registry.records` with `web_research_session.answer_sources` at all three sites: the citation-required correction check, the worker `evidence` tuple, and the `metadata["web_sources"]` construction.

The correction check narrows correctly: a turn whose only registry entries are image pages with no active image and no open has no evidence a citation could name, so skipping the correction there is the right behavior, not a loosening.

- [x] **Step 5: Verify scope and adjacent streaming**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_published_source_scope.py tests/test_web_source_streaming.py tests/test_web_research_worker_remap.py tests/test_web_research_active_path_inventory.py tests/test_web_grounding.py
.\.venv\Scripts\python.exe -m ruff check app/ai/web_research/service.py app/ai/workflow/specialists.py tests/test_web_published_source_scope.py
git diff --check
```

- [x] **Step 6: Commit the publication scope**

```powershell
git add -- app/ai/web_research/service.py app/ai/workflow/specialists.py tests/test_web_published_source_scope.py
git commit -m "fix: publish only answer-relevant web sources"
```

---

### Task 5: Lock the three reported visual cases into acceptance coverage

**Files:**
- Add: `tests/test_web_image_relevance_regressions.py`
- Modify: `docs/operations/web-research-rollout.md`

**Acceptance cases:**
- T1: small roster cards plus a later official full-team photo.
- Character hair: a high-confidence 1024x576 close-up remains selectable.
- Windows: a settings-menu screenshot beats unrelated product imagery.

**What these tests may and may not claim.** They prove the good candidate reaches the vision window and that the `IMAGE TARGET` line reaches the model. They must not be written so as to imply the ranking picks the right *kind* of picture — it cannot. Step 2's negative control makes that boundary explicit and failing.

- [x] **Step 1: Add deterministic end-to-end regression fixtures in code**

Create candidates with no network dependency and route them through `WebResearchService`, the fake image service, `inject_latest_web_evidence`, and `GroundingParser`. Give each case's second search a distinct scope, per Task 2 Step 2.

For T1, assert that after two searches:

```python
assert active[0].source_domain == "www.t1.gg"
assert (active[0].width, active[0].height) == (1920, 1080)
assert "full T1 League of Legends team photo" in injected_text
```

For the hair and Windows cases, assert the high-confidence, adequate-resolution candidate is active and that the literal target appears unchanged in model context. Also assert selecting its `I#` resolves to a rich item with the full-resolution record, while selecting an evicted `I#` produces no image.

- [x] **Step 2: Add the negative control**

Add one case where the T1 cohort also contains a 1920x1080 roster *graphic* alongside the 1200x675 team *photo*, and assert:

```python
# Ordering is quality-only: the larger graphic leads, and that is correct
# behavior for a ranking function that cannot see subject matter.
assert active[0].title == "T1 2026 roster graphic"
# Both reach the model, and the model is told what to look for.
assert {record.title for record in active} >= {
    "T1 2026 roster graphic",
    "T1 team photo",
}
assert "A roster graphic or list of names is not a full team photo" in injected_text
```

If a future change makes the photo lead by subject matter, this test should fail and be read before it is edited — it means a taxonomy was introduced.

- [x] **Step 3: Run the new acceptance tests and verify they pass only through production code**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_image_relevance_regressions.py
```

Expected: all cases pass without monkeypatching the ranking function or inspecting private implementation helpers.

- [x] **Step 4: Document the runtime behavior and bounded costs**

Update `docs/operations/web-research-rollout.md` with:

- Brave locale forwarding, strict domain semantics, and the single supplemental `site:` probe with its rate-limit rationale;
- the per-cohort candidate cap (`web_research_max_candidate_pool`) and session catalog bound (`web_research_max_candidate_catalog`);
- active vision-window limits (4 for figure/comparison, 6 for gallery);
- the three registry views — complete registry for grounding, operation delta in tool output, `answer_sources` for the published list — and why the published list is smaller than the registry;
- that ordering is quality-only and visual-form matching is the answer model's job via `IMAGE TARGET`;
- logs/metrics to inspect for the optional domain-probe failure, omitted candidates, and image fetch rejection.

Correct the paragraph that still claims image pages get "up to 4 image pages (6 for gallery)" of registry capacity — that is the *vision window*, and after Task 2 the registry holds the whole catalog.

Do not document a fallback that publishes provider links as images; publication still requires a validated, model-selected candidate.

- [x] **Step 5: Run the complete focused web-research suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_research_providers.py tests/test_web_query_contract.py tests/test_web_source_registry.py tests/test_web_research_contracts.py tests/test_web_research_config.py tests/test_web_image_capacity_and_resolution.py tests/test_web_research_images.py tests/test_web_research_model_context.py tests/test_web_research_service.py tests/test_web_research_tool_session.py tests/test_web_tool_output_privacy.py tests/test_web_published_source_scope.py tests/test_web_grounding.py tests/test_web_research_docs.py tests/test_web_image_relevance_regressions.py
.\.venv\Scripts\python.exe -m ruff check app/ai/web_research app/ai/web_tools.py app/ai/web_query_contract.py app/ai/workflow/specialists.py tests/test_web_*.py
git diff --check
```

Expected: all focused tests and static checks pass.

- [x] **Step 6: Commit acceptance coverage and operations notes**

```powershell
git add -- tests/test_web_image_relevance_regressions.py docs/operations/web-research-rollout.md
git commit -m "test: cover web image relevance regressions"
```

---

### Task 6: Final verification and review

**Files:**
- Review all files changed by Tasks 1-5.

- [x] **Step 1: Run the broader adjacent suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_*.py tests/test_research_budget.py tests/test_message_service_web_image_externalization.py tests/test_ai_sdk_v6_stream_contract.py tests/test_rich_response_metadata.py tests/test_container_web_image_wiring.py
```

Expected: all tests pass. The baseline for the nine focused files was 91 passing at `47bef3cc`; investigate any failure before proceeding, and do not label an unrelated failure without reproducing it on the base commit.

- [x] **Step 2: Re-run static and repository checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check app/ai/web_research app/ai/web_tools.py app/ai/web_query_contract.py app/ai/workflow/specialists.py app/core/config.py app/core/container.py tests/test_web_*.py
git diff --check
git status --short
git log --oneline -6
```

Expected: Ruff and diff checks pass; only intentional changes remain; the five implementation commits are visible.

- [x] **Step 3: Request code review**

Use `superpowers:requesting-code-review`. Require the review to verify:

- `include_domains` is normalized before comparison and never leaks disallowed results;
- a URL-shaped `include_domains` entry restricts rather than silently emptying the result set;
- candidate IDs cannot be rebound or reused after eviction;
- evicted pending references are released exactly once, with UUID-coerced scope;
- text/image source quotas remain bounded independently;
- tool deltas and `answer_sources` do not remove sources needed by grounding;
- image bytes and source-image URLs do not leak in tool JSON;
- an opened page's content still reaches the answer model;
- no taxonomy/ranking framework or answer-time fallback was introduced.

- [x] **Step 4: Apply valid review findings and repeat verification**

Use `superpowers:receiving-code-review` for any findings. Add a regression test before each behavior fix, rerun the relevant focused command, then repeat Steps 1-2.

- [x] **Step 5: Finish the development branch**

Use `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch`. Report exact test counts and the commit list; do not claim completion from stale output.

---

## Deviations taken during execution

- `tests/test_web_research_service.py::test_valid_image_sources_are_admitted_after_text_sources`
  encoded the old admission order and had to move with the design. The plan did
  not list that file; it is now in Task 2's set and the test asserts the
  inverted order.
- The active-window fixtures give every candidate a confidence, because Brave
  grades every result. A cohort mixing graded and ungraded candidates is not a
  shape the provider produces, and ordering one against the other proves
  nothing.
- The acceptance cases place the good candidate at a *later* provider rank than
  the noise. A case whose good candidate is already rank 1 passes under
  rank-only ordering too; all five were mutation-checked against a
  rank-only `_image_candidate_priority` and a renamed `IMAGE TARGET` line.

## Review findings applied after Task 5

A review of `685c1d74..f4633bea` found nine issues; eight were fixed with a
regression test each, one was declined.

- An unguarded `release_references` in `_retain_active` let a database write
  failure throw away an otherwise successful search. Now best effort.
- `unusable` was per-call, so a candidate that could not be fetched was
  refetched on every later search and counted in `omitted_image_count` twice.
  Promoted to session state, which is what the counter's own definition says.
- `answer_sources` could omit a page the answer visibly *cited*: the model is
  offered every admitted `S#`, so it can cite an image page whose candidate
  never entered the window. `published_sources(cited_source_ids)` adds those
  back; `answer_sources` remains the no-citations view.
- The registry had no headroom for `web_open`, so a saturated catalog silently
  swallowed a deliberate read. Capacity is now text + catalog + `max_page_opens`.
  The plan's "Known gaps" entry reasoned only about the opposite direction.
- The `site:` probe kept a `www.` prefix that `host_matches` strips, asking
  Brave for strictly less than the restriction allowed.
- Image source pages were admitted for candidates beyond `max_candidate_pool`,
  which could never enter the catalog.
- `_omitted_source_count` counted events rather than distinct lost pages.
- Declined: adding a replacement absolute source cap to `WebEvidenceBundle`.
  The runtime ceiling is session-configured and the registry is its only
  producer; a second invented number would be the thing that goes stale.

## Known gaps this plan deliberately leaves open

- `ResearchBudget.reserve_image_search`, `may_image_search`, and `record_image_search` have no production callers, so `research_max_image_searches_per_turn` bounds nothing. Wiring it is separate work; Task 2 avoids depending on it.
- Nothing bounds how many image searches one turn performs. The catalog capacity bounds the *consequences*, not the provider calls.
- The supplemental `site:` probe fires whenever a domain-restricted search
  returns fewer than `count` (10) allowed results, which after strict filtering
  is nearly always. A restricted image search therefore costs two Brave calls
  in practice, within the ~5s worst case the rationale accounts for.
- `web_open` admissions are not counted against the text quota. They are bounded instead by `max_page_opens` (2 quick / 4 agentic), which is small enough that the leak cannot exhaust the registry.
