# Vision-Verified Image Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A remote web image reaches an answer only after a vision model confirms its visible content depicts the user's subject, and one model-facing `web_research` operation replaces the model's ad-hoc coordination of a text search and an image search.

**Architecture:** One in-process `web_research` tool owns orchestration: it runs Tavily text search and Brave image search concurrently under a turn-local call budget, downloads Brave thumbnails through the existing hardened image fetcher, submits them to one batched vision call, and offers only approved candidates to the rich-item inventory through a context-scoped sink the model never sees. The old metadata-based remote selection path is deleted rather than tuned.

**Tech Stack:** Python 3.11+, LangChain `StructuredTool` and `with_structured_output`, `ChatGoogleGenerativeAI` (Gemini), httpx + Pillow (existing `WebImageService`), Pydantic v2, SQLAlchemy 2.x + Alembic, prometheus_client, pytest + pytest-asyncio.

This is Phase 2 of `docs/superpowers/specs/2026-08-04-vision-verified-image-injection-design.md`. **Phase 1 (`docs/superpowers/plans/2026-08-04-research-payload-and-offload-repair.md`) must be merged first**; Task 1 here assumes `tavily_search` no longer accepts or returns images.

## Carried over from Phase 1

Phase 1 shipped clean (final review: ready to merge) but left four tracked follow-ups. None block Phase 2; the first is the only one with production impact.

1. **`offload_if_large` runs a blocking DB commit on the event loop.** `ToolResultBlobService.offload_if_large` calls `ToolResultBlobRepository.create`, a synchronous SQLAlchemy `commit()` of the full multi-MB payload, from inside `async def execute_tool_calls` (`app/ai/tool_execution.py`). Phase 1 removed a 26-second CPU stall there but did not thread the call. The read path next to it already threads its DB work (`app/ai/tool_result_read_tool.py`), so the write path is inconsistent with its own neighbour. Roughly a three-line fix. Phase 2 adds no new offloaded output, so it neither worsens nor depends on this.
2. **`app/api/tool_result_blobs.py` calls `service.read_text(record)` unguarded**, so a corrupt record returns HTTP 500 on the download endpoint while the tool path now returns a structured not-found. Pre-existing; Phase 1 fixed only the tool path.
3. **`TOOL_CONTEXT_SUFFIX` says "Do NOT repeat the search"** in a bullet that applies to every tool, which reads oddly for extract, map, or SQL. The operative instruction in the same bullet is tool-neutral, so this is wording only. Phase 2's Task 6 rewrites nearby prompt text and can absorb it.
4. **The preview's `content` share is now diluted on search payloads that carry `raw_content`** — measured 543 → 232 chars of curated content at budget 4000 with 5 results. Inherent to dropping the key whitelist (which was necessary: the whitelist reduced a 41 KB `tavily_extract` payload to 189 characters of URLs). `include_raw_content` defaults to `False`, so the common path is unaffected.

Also inherited: every `tests/integration/*_postgres.py` skips silently because nothing in this repo sets `TEST_DATABASE_URL` — no CI config, no `addopts`, no `conftest` default. The file tree therefore overstates coverage. Phase 1 worked around it with a DB-free statement-capture test; if Phase 2 adds Postgres-only properties, do the same or make the skip loud.

## Global Constraints

- Run every command from the repository root with the app runtime: `.venv/Scripts/python.exe -m pytest ...`. Only `.venv` is the app runtime; the other two interpreters in this checkout have drifted pins.
- Functions: 100 lines max, cyclomatic complexity 8 max, 5 positional parameters max, 100-character lines.
- Zero `ruff` findings. Repo-wide `ruff check . --no-cache` is clean as of 2026-08-04 and must stay that way: "zero new findings" means `All checks passed!`, not "no worse than a large baseline". (An earlier draft of this constraint claimed ~161 pre-existing findings, quoting a superseded 2026-06 measurement; a Phase 1 implementer relied on it and left the repo's only lint error in place.)
- Confidence threshold is `0.85` initially. Maximum candidates submitted to the verifier is `6`. Image-path deadline is `4.0` seconds initially. All are configuration, never literals at a call site.
- The image count follows the declared layout intent, and the model never states a number: `figure` intent admits at most `rich_auto_place_max_images` (2) individual items; `gallery` intent admits exactly one grid item holding up to `rich_image_gallery_max_items` (6). An absent intent with a non-empty `image_query` means `figure`.
- Candidates are discovered, fetched, and verified **individually**, never as a pre-grouped grid. Grouping happens only after admission, over the survivors. Grouping earlier both hides individual images from the verifier and caps discovery below the candidate budget.
- Verifier decisions, confidence values, content kinds, and rejection reasons must never appear in public rich items, response metadata, persisted messages, logs, or metric labels. Metrics carry aggregate counts, durations, and bounded reason enums only.
- The answering model must never see a rejected candidate's id, URL, title, or description.
- Every failure in the image path — Brave error, thumbnail failure, verifier timeout, malformed structured output, zero approvals — is a successful text-only answer, not an error.
- Remote web image injection stays off unless `settings.vision_image_verification_enabled` is true AND `inline_rich_response_enabled` is true AND the request advertised the inline rich-response capability.
- Another session may commit in this checkout with a broad `git add`. Stage explicit paths only, never `git add -A` or `git add .`, and verify with `git status --short` rather than trusting an exit code.

---

### Task 1: Turn-local research budget

Normalizes and deduplicates factual queries within one user turn, and caps Tavily at two network requests and Brave at one.

**Files:**
- Create: `app/ai/research_budget.py`
- Modify: `app/ai/graph.py:488-552` (reset in `_build_initial_state_from_request`)
- Modify: `app/core/config.py` (three settings)
- Create: `tests/test_research_budget.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `normalize_query_tokens(query: str) -> frozenset[str]` — NFKC case-folded alphanumeric tokens with short identifiers retained.
  - `near_duplicate(a: frozenset[str], b: frozenset[str], *, threshold: float) -> bool`
  - `ResearchBudget` with `find_reuse(query: str) -> str | None`, `reserve_search(query: str) -> bool`, `record_search(query: str, result_text: str) -> None`, `may_image_search() -> bool`, `record_image_search(candidates: list[dict]) -> None`, `image_result() -> list[dict]`, `accumulated() -> list[str]`, and read-only `search_calls: int`.

**Reservation must be atomic, and `may_search` is not.** A caller checks the
budget, then awaits a network call, then records the result — so with two
concurrent `web_research` calls in one turn (the model can emit parallel tool
calls) both would pass a `may_search` check before either recorded, and the
"at most two network requests" invariant would not hold. `reserve_search` claims
a slot and returns whether the caller may proceed, in one indivisible step:
it refuses when a reuse exists or when claimed-plus-completed searches already
fill the budget, and otherwise increments an in-flight count that
`record_search` clears as it stores the result. A search that fails does **not**
release its slot: a provider error should not buy the model another attempt at
the same broken call. Guard the compound operations with a `threading.Lock` on
the instance so a threaded caller cannot interleave either.
  - `get_research_budget(conversation_id: str | None) -> ResearchBudget` and `reset_research_budget(conversation_id: str | None) -> None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_research_budget.py`:

```python
from __future__ import annotations

from app.ai.research_budget import (
    ResearchBudget,
    get_research_budget,
    near_duplicate,
    normalize_query_tokens,
    reset_research_budget,
)

TRACE_Q1 = "T1 League of Legends Esports team news roster 2026"
TRACE_Q2 = "T1 League of Legends team overview roster news 2026"
TRACE_Q3 = "T1 League of Legends current roster 2026 achievements lck summer"


def test_short_identifiers_survive_normalization():
    tokens = normalize_query_tokens("T1 F1 3M vs G2")

    assert {"t1", "f1", "3m", "g2"} <= tokens


def test_non_ascii_queries_normalize_without_losing_tokens():
    tokens = normalize_query_tokens("thông tin về T1")

    assert "t1" in tokens
    assert len(tokens) == 4


def test_trace_second_query_is_a_near_duplicate_of_the_first():
    assert near_duplicate(
        normalize_query_tokens(TRACE_Q1), normalize_query_tokens(TRACE_Q2), threshold=0.75
    )


def test_trace_third_query_is_genuinely_distinct():
    assert not near_duplicate(
        normalize_query_tokens(TRACE_Q1), normalize_query_tokens(TRACE_Q3), threshold=0.75
    )


def test_trace_produces_two_network_searches_and_one_reuse():
    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)

    assert budget.find_reuse(TRACE_Q1) is None
    budget.record_search(TRACE_Q1, "first result")

    assert budget.find_reuse(TRACE_Q2) == "first result"

    assert budget.find_reuse(TRACE_Q3) is None
    assert budget.may_search(TRACE_Q3) is True
    budget.record_search(TRACE_Q3, "third result")

    assert budget.search_calls == 2


def test_a_fourth_distinct_query_is_refused_and_returns_accumulated_results():
    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)
    budget.record_search("alpha topic one", "A")
    budget.record_search("beta topic two", "B")

    assert budget.may_search("gamma topic three") is False
    assert budget.accumulated() == ["A", "B"]


def test_only_one_image_search_per_turn():
    budget = ResearchBudget(max_search_calls=2, near_duplicate_threshold=0.75)

    assert budget.may_image_search() is True
    budget.record_image_search([{"id": "image:verified:1"}])

    assert budget.may_image_search() is False
    assert budget.image_result() == [{"id": "image:verified:1"}]


def test_budget_is_per_conversation_and_resettable():
    first = get_research_budget("conv-a")
    first.record_search("alpha", "A")

    assert get_research_budget("conv-a") is first
    assert get_research_budget("conv-b") is not first

    reset_research_budget("conv-a")
    assert get_research_budget("conv-a").search_calls == 0


def test_missing_conversation_id_gets_an_isolated_budget():
    reset_research_budget(None)
    budget = get_research_budget(None)
    budget.record_search("alpha", "A")

    reset_research_budget(None)
    assert get_research_budget(None).search_calls == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_research_budget.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ai.research_budget'`.

- [ ] **Step 3: Write the budget**

Create `app/ai/research_budget.py`:

```python
"""Turn-local accounting for factual and image research calls.

A model that receives a thin result reaches for another search. Bounding that
within a turn keeps one broad question from spending three provider calls on
near-identical queries. Nothing here outlives the turn: the store is reset when
a new user turn builds its initial graph state.

Tokens keep short identifiers such as ``T1``, ``F1`` and ``3M`` because those
are frequently the only word that identifies the subject. No stopword list is
used: the corpus is multilingual and a per-language table would buy nothing the
threshold does not already provide.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ..core.config import settings

_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_MAX_QUERY_CHARS = 2048
_MAX_TOKENS = 128
_MAX_TRACKED_CONVERSATIONS = 256


def normalize_query_tokens(query: str) -> frozenset[str]:
    """Return the comparable token set for a research query."""

    bounded = str(query or "")[:_MAX_QUERY_CHARS]
    normalized = unicodedata.normalize("NFKC", bounded).casefold()
    return frozenset(_TOKEN_PATTERN.findall(normalized)[:_MAX_TOKENS])


def near_duplicate(a: frozenset[str], b: frozenset[str], *, threshold: float) -> bool:
    """Return whether two token sets overlap by at least ``threshold``."""

    smaller = min(len(a), len(b))
    if smaller == 0:
        return False
    return len(a & b) / smaller >= float(threshold)


@dataclass
class ResearchBudget:
    """One turn's research accounting for a single conversation."""

    max_search_calls: int = 2
    near_duplicate_threshold: float = 0.75
    _searches: list[tuple[frozenset[str], str]] = field(default_factory=list)
    _image_searched: bool = False
    _image_candidates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def search_calls(self) -> int:
        return len(self._searches)

    def find_reuse(self, query: str) -> str | None:
        """Return an existing result for an exact or near-duplicate query."""

        tokens = normalize_query_tokens(query)
        for recorded_tokens, result_text in self._searches:
            if recorded_tokens == tokens or near_duplicate(
                recorded_tokens, tokens, threshold=self.near_duplicate_threshold
            ):
                return result_text
        return None

    def may_search(self, query: str) -> bool:
        if self.find_reuse(query) is not None:
            return False
        return self.search_calls < max(1, int(self.max_search_calls))

    def record_search(self, query: str, result_text: str) -> None:
        self._searches.append((normalize_query_tokens(query), result_text))

    def accumulated(self) -> list[str]:
        return [result_text for _, result_text in self._searches]

    def may_image_search(self) -> bool:
        return not self._image_searched

    def record_image_search(self, candidates: list[dict[str, Any]]) -> None:
        self._image_searched = True
        self._image_candidates = list(candidates)

    def image_result(self) -> list[dict[str, Any]]:
        return list(self._image_candidates)


_lock = threading.Lock()
_budgets: OrderedDict[str, ResearchBudget] = OrderedDict()


def _key(conversation_id: str | None) -> str:
    return str(conversation_id or "__no_conversation__")


def get_research_budget(conversation_id: str | None) -> ResearchBudget:
    """Return the live budget for a conversation, creating it on first use."""

    key = _key(conversation_id)
    with _lock:
        budget = _budgets.get(key)
        if budget is None:
            budget = ResearchBudget(
                max_search_calls=max(1, int(settings.research_max_search_calls_per_turn)),
                near_duplicate_threshold=float(settings.research_near_duplicate_threshold),
            )
            _budgets[key] = budget
        _budgets.move_to_end(key)
        while len(_budgets) > _MAX_TRACKED_CONVERSATIONS:
            _budgets.popitem(last=False)
        return budget


def reset_research_budget(conversation_id: str | None) -> None:
    """Drop a conversation's budget so the next turn starts clean."""

    with _lock:
        _budgets.pop(_key(conversation_id), None)
```

- [ ] **Step 4: Add the settings**

In `app/core/config.py`, after `tool_result_read_max_chars` (added in Phase 1):

```python
    research_max_search_calls_per_turn: int = Field(
        default=2,
        ge=1,
        description=(
            "Distinct Tavily network requests allowed per user turn. Further calls "
            "return the accumulated research result instead of searching again."
        ),
    )
    research_near_duplicate_threshold: float = Field(
        default=0.75,
        gt=0.0,
        le=1.0,
        description=(
            "Token-set overlap, as a share of the smaller query, above which a "
            "research query reuses the existing result."
        ),
    )
    research_budget_enabled: bool = Field(
        default=True,
        description="Kill switch for turn-local research dedup and call caps.",
    )
```

- [ ] **Step 5: Reset the budget at the start of each user turn**

In `app/ai/graph.py`, add the import beside the other `app.ai` imports:

```python
from .research_budget import reset_research_budget
```

In `_build_initial_state_from_request`, immediately after the line `initial_state["planning_call_count"] = 0`:

```python
        # A new user turn gets a clean research budget; the previous turn's
        # deduplication must not suppress a legitimate follow-up question.
        reset_research_budget(str(request.conversation_id) if request.conversation_id else None)
```

Confirm the attribute name first:

```bash
grep -n "conversation_id" app/ai/schemas.py | head -20
```

If `WorkflowExecutionRequest` exposes the conversation id under a different attribute or nested object, use that path instead — do not invent one.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_research_budget.py -v`
Expected: PASS.

Then: `.venv/Scripts/python.exe -m pytest tests/test_graph_tool_budget.py -v`
Expected: PASS (unchanged behavior; the reset is additive).

- [ ] **Step 7: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/ai/research_budget.py app/ai/graph.py app/core/config.py
git add app/ai/research_budget.py app/ai/graph.py app/core/config.py tests/test_research_budget.py
git status --short
git commit -m "feat: add turn-local research budget"
```

---

### Task 2: Thumbnail fetch seam on the existing hardened fetcher

Verification needs validated image bytes. Every guard it requires — HTTPS parsing, public-IP assertion with a pinned transport, per-hop redirect revalidation, byte caps, decompression-bomb bounds, MIME decoded from bytes — already exists in `WebImageService` but is reachable only through a persisted record. This exposes it for a bare URL and adds a concurrent batch.

**Files:**
- Modify: `app/services/web_image_service.py:122-150`
- Create: `app/services/thumbnail_batch.py`
- Create: `tests/test_thumbnail_batch.py`
- Modify: `tests/test_web_image_service.py` (add two tests; create the file if absent)

**Interfaces:**
- Consumes: `WebImageService.fetch(record)`, `FetchedWebImage(content, media_type, width, height)`, `WebImageRejected`, `WebImageUpstreamFailure` — all existing.
- Produces:
  - `WebImageService.fetch_url(url: str, *, provider: str = "other") -> FetchedWebImage`
  - `fetch_thumbnails(service, urls: Sequence[str], *, provider: str, per_item_timeout: float, batch_deadline: float) -> list[FetchedThumbnail | None]` where `FetchedThumbnail` is a frozen dataclass with `url: str`, `image: FetchedWebImage`. The returned list is positionally aligned with `urls`; a failed item is `None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_thumbnail_batch.py`:

```python
from __future__ import annotations

import asyncio

import pytest

from app.services.thumbnail_batch import fetch_thumbnails
from app.services.web_image_service import (
    FetchedWebImage,
    WebImageRejected,
    WebImageUpstreamFailure,
)


class _FakeService:
    def __init__(self, behavior: dict[str, object], delay: float = 0.0):
        self.behavior = behavior
        self.delay = delay
        self.calls: list[str] = []

    async def fetch_url(self, url: str, *, provider: str = "other") -> FetchedWebImage:
        self.calls.append(url)
        if self.delay:
            await asyncio.sleep(self.delay)
        outcome = self.behavior[url]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _image(width: int = 800) -> FetchedWebImage:
    return FetchedWebImage(content=b"bytes", media_type="image/jpeg", width=width, height=600)


@pytest.mark.asyncio
async def test_one_failure_does_not_discard_the_batch():
    urls = ["https://a.example/1.jpg", "https://b.example/2.jpg", "https://c.example/3.jpg"]
    service = _FakeService(
        {
            urls[0]: _image(800),
            urls[1]: WebImageRejected("private_address"),
            urls[2]: _image(900),
        }
    )

    fetched = await fetch_thumbnails(
        service, urls, provider="brave", per_item_timeout=1.0, batch_deadline=2.0
    )

    assert [item is None for item in fetched] == [False, True, False]
    assert fetched[0].url == urls[0]
    assert fetched[2].image.width == 900


@pytest.mark.asyncio
async def test_upstream_failure_is_isolated_too():
    urls = ["https://a.example/1.jpg"]
    service = _FakeService({urls[0]: WebImageUpstreamFailure("timeout")})

    assert await fetch_thumbnails(
        service, urls, provider="brave", per_item_timeout=1.0, batch_deadline=1.0
    ) == [None]


@pytest.mark.asyncio
async def test_downloads_run_concurrently_within_the_batch_deadline():
    urls = [f"https://a.example/{index}.jpg" for index in range(6)]
    service = _FakeService({url: _image() for url in urls}, delay=0.2)

    started = asyncio.get_running_loop().time()
    fetched = await fetch_thumbnails(
        service, urls, provider="brave", per_item_timeout=1.0, batch_deadline=2.0
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert all(item is not None for item in fetched)
    assert elapsed < 0.6, "six 0.2s fetches must overlap, not serialize"


@pytest.mark.asyncio
async def test_batch_deadline_fires_when_it_is_the_tighter_bound():
    """The outer deadline must bound the batch even when per-item timeouts cannot.

    Corrected 2026-08-05: this test originally used per_item_timeout=0.05 against
    batch_deadline=0.1, so every fetch resolved to None through its own inner
    timeout and the outer `except TimeoutError` branch was never entered. Deleting
    the entire outer wrapper and its cancellation loop left the whole file green.
    per_item_timeout must exceed batch_deadline for this test to mean anything.
    """
    urls = ["https://a.example/1.jpg", "https://b.example/2.jpg"]
    service = _FakeService({url: _image() for url in urls}, delay=2.0)

    started = asyncio.get_running_loop().time()
    fetched = await fetch_thumbnails(
        service, urls, provider="brave", per_item_timeout=10.0, batch_deadline=0.2
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert fetched == [None, None]
    assert elapsed < 1.0, "the batch deadline, not the per-item timeout, must bound this"


@pytest.mark.asyncio
async def test_empty_url_list_makes_no_calls():
    service = _FakeService({})

    assert await fetch_thumbnails(
        service, [], provider="brave", per_item_timeout=1.0, batch_deadline=1.0
    ) == []
    assert service.calls == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_thumbnail_batch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.thumbnail_batch'`.

- [ ] **Step 3: Add the URL fetch seam**

In `app/services/web_image_service.py`, replace `_fetch_redirects` with a record wrapper plus a URL implementation, and add the public `fetch_url`:

```python
    async def fetch_url(self, url: str, *, provider: str = "other") -> FetchedWebImage:
        """Fetch and validate an upstream image URL with no persisted record.

        Same guards as ``fetch``: HTTPS-only, public-address assertion, pinned
        transport, revalidated redirects, byte cap, decoded MIME and dimensions.
        """
        started = time.perf_counter()
        outcome = "success"
        try:
            return await self._fetch_url_with_redirects(url)
        except (WebImageRejected, WebImageUpstreamFailure) as exc:
            outcome = exc.reason
            raise
        finally:
            if self.metrics is not None:
                with suppress(Exception):
                    self.metrics.record_fetch(
                        provider=provider,
                        outcome=outcome,
                        duration_seconds=time.perf_counter() - started,
                    )

    async def _fetch_redirects(self, record: Any) -> FetchedWebImage:
        return await self._fetch_url_with_redirects(self._record_value(record, "upstream_url"))

    async def _fetch_url_with_redirects(self, url: str) -> FetchedWebImage:
        current_url = url
        for redirect_count in range(self.max_redirects + 1):
            outcome = await self._fetch_once(current_url)
            if isinstance(outcome, FetchedWebImage):
                return outcome
            if redirect_count >= self.max_redirects:
                raise WebImageRejected("redirect_limit")
            current_url = urljoin(current_url, outcome)
        raise WebImageRejected("redirect_limit")
```

- [ ] **Step 4: Write the batch**

Create `app/services/thumbnail_batch.py`:

```python
"""Concurrent, individually-isolated thumbnail downloads for visual verification.

One blocked or malformed candidate must cost only itself. The batch deadline is
the hard bound: whatever has not arrived by then is treated as absent, because
the answer is already waiting on it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.services.web_image_service import FetchedWebImage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FetchedThumbnail:
    """One successfully fetched and decoded candidate thumbnail."""

    url: str
    image: FetchedWebImage


async def fetch_thumbnails(
    service: Any,
    urls: Sequence[str],
    *,
    provider: str,
    per_item_timeout: float,
    batch_deadline: float,
) -> list[FetchedThumbnail | None]:
    """Fetch every URL concurrently. Results align positionally with ``urls``."""

    if not urls:
        return []

    async def _one(url: str) -> FetchedThumbnail | None:
        try:
            async with asyncio.timeout(max(0.001, float(per_item_timeout))):
                image = await service.fetch_url(url, provider=provider)
        except Exception as exc:
            logger.debug("Thumbnail candidate dropped: %s", type(exc).__name__)
            return None
        return FetchedThumbnail(url=url, image=image)

    tasks = [asyncio.create_task(_one(url)) for url in urls]
    try:
        async with asyncio.timeout(max(0.001, float(batch_deadline))):
            return list(await asyncio.gather(*tasks))
    except TimeoutError:
        results: list[FetchedThumbnail | None] = []
        for task in tasks:
            if task.done() and not task.cancelled() and task.exception() is None:
                results.append(task.result())
            else:
                task.cancel()
                results.append(None)
        return results
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_thumbnail_batch.py -v`
Expected: PASS.

- [ ] **Step 6: Prove the seam still enforces every guard**

Add to `tests/test_web_image_service.py` (create the file with the existing fixtures' style if it does not exist — model the fakes on the `resolver` and `transport_factory` constructor hooks):

```python
@pytest.mark.asyncio
async def test_fetch_url_rejects_a_private_address():
    service = WebImageService(
        repository=object(),
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_redirects=1,
        max_bytes=1024,
        max_pixels=10_000_000,
        resolver=lambda hostname: ["127.0.0.1"],
    )

    with pytest.raises(WebImageRejected) as excinfo:
        await service.fetch_url("https://internal.example/a.png")

    assert excinfo.value.reason == "private_address"


@pytest.mark.asyncio
async def test_fetch_url_rejects_a_non_https_scheme():
    service = WebImageService(
        repository=object(),
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_redirects=1,
        max_bytes=1024,
        max_pixels=10_000_000,
        resolver=lambda hostname: ["93.184.216.34"],
    )

    with pytest.raises(WebImageRejected) as excinfo:
        await service.fetch_url("http://example.com/a.png")

    assert excinfo.value.reason == "scheme"
```

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_image_service.py tests/test_message_service_web_image_externalization.py -v`
Expected: PASS. The record-based `fetch` path must still pass unchanged — that is the regression guard for the refactor.

- [ ] **Step 7: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/services/web_image_service.py app/services/thumbnail_batch.py
git add app/services/web_image_service.py app/services/thumbnail_batch.py tests/test_thumbnail_batch.py tests/test_web_image_service.py
git status --short
git commit -m "feat: expose a guarded url fetch and thumbnail batch"
```

---

### Task 3: Visual verifier

One batched vision call decides from pixels. Metadata may disambiguate what the model can see; it can never establish relevance on its own.

**Files:**
- Create: `app/ai/visual_verifier.py`
- Modify: `app/core/config.py` (six settings)
- Create: `tests/test_visual_verifier.py`

**Interfaces:**
- Consumes: `FetchedThumbnail` from Task 2.
- Produces:
  - `VisualCandidateDecision` (Pydantic) with `candidate_id: str`, `depicts_requested_subject: bool`, `materially_supports_answer: bool`, `confidence: float`, `content_kind: Literal[...]`.
  - `VisualVerificationResult` (Pydantic) with `decisions: list[VisualCandidateDecision]`.
  - `SubmittedCandidate` frozen dataclass: `candidate_id: str`, `thumbnail: FetchedThumbnail`, `title: str`, `description: str`.
  - `admit_candidates(result: VisualVerificationResult | None, submitted: Sequence[SubmittedCandidate], *, threshold: float, max_items: int, requested_kinds: frozenset[str]) -> list[SubmittedCandidate]`
  - `async verify_candidates(submitted, *, user_request: str, image_query: str, factual_query: str, result_titles: Sequence[str], model=None, timeout: float) -> VisualVerificationResult | None`
  - `SPECIALIZED_KINDS: frozenset[str]`

- [ ] **Step 1: Write the failing admission tests**

Create `tests/test_visual_verifier.py`:

```python
from __future__ import annotations

import asyncio

import pytest

from app.ai.visual_verifier import (
    SubmittedCandidate,
    VisualCandidateDecision,
    VisualVerificationResult,
    admit_candidates,
    verify_candidates,
)
from app.services.thumbnail_batch import FetchedThumbnail
from app.services.web_image_service import FetchedWebImage


def _submitted(candidate_id: str, title: str = "t") -> SubmittedCandidate:
    return SubmittedCandidate(
        candidate_id=candidate_id,
        thumbnail=FetchedThumbnail(
            url=f"https://cdn.example/{candidate_id}.jpg",
            image=FetchedWebImage(
                content=b"bytes", media_type="image/jpeg", width=995, height=565
            ),
        ),
        title=title,
        description="d",
    )


def _decision(candidate_id: str, **overrides) -> VisualCandidateDecision:
    payload = {
        "candidate_id": candidate_id,
        "depicts_requested_subject": True,
        "materially_supports_answer": True,
        "confidence": 0.95,
        "content_kind": "photo",
    }
    payload.update(overrides)
    return VisualCandidateDecision(**payload)


def _admit(decisions, submitted, **overrides):
    kwargs = {"threshold": 0.85, "max_items": 2, "requested_kinds": frozenset()}
    kwargs.update(overrides)
    return admit_candidates(VisualVerificationResult(decisions=decisions), submitted, **kwargs)


def test_admits_a_confident_relevant_photo():
    submitted = [_submitted("c1")]

    assert _admit([_decision("c1")], submitted) == submitted


def test_rejects_low_confidence_even_at_provider_rank_one():
    submitted = [_submitted("c1"), _submitted("c2")]
    decisions = [_decision("c1", confidence=0.5), _decision("c2", confidence=0.9)]

    assert [item.candidate_id for item in _admit(decisions, submitted)] == ["c2"]


def test_rejects_a_relevant_image_that_does_not_support_the_answer():
    submitted = [_submitted("c1")]

    assert _admit([_decision("c1", materially_supports_answer=False)], submitted) == []


def test_rejects_a_portrait_unless_the_user_asked_for_one():
    submitted = [_submitted("c1")]
    decisions = [_decision("c1", content_kind="portrait")]

    assert _admit(decisions, submitted) == []
    assert len(_admit(decisions, submitted, requested_kinds=frozenset({"portrait"}))) == 1


def test_an_all_uncertain_batch_admits_nothing():
    submitted = [_submitted("c1"), _submitted("c2")]
    decisions = [_decision("c1", confidence=0.4), _decision("c2", confidence=0.6)]

    assert _admit(decisions, submitted) == []


def test_hallucinated_and_duplicate_ids_fail_closed():
    submitted = [_submitted("c1")]

    assert _admit([_decision("ghost")], submitted) == []
    assert _admit([_decision("c1"), _decision("c1")], submitted) == []


def test_out_of_range_confidence_rejects_only_that_candidate():
    submitted = [_submitted("c1"), _submitted("c2")]
    decisions = [_decision("c1", confidence=1.9), _decision("c2")]

    assert [item.candidate_id for item in _admit(decisions, submitted)] == ["c2"]


def test_a_response_level_failure_rejects_the_whole_batch():
    submitted = [_submitted("c1"), _submitted("c2")]

    assert admit_candidates(
        None, submitted, threshold=0.85, max_items=2, requested_kinds=frozenset()
    ) == []


def test_provider_order_is_preserved_and_capped():
    submitted = [_submitted("c1"), _submitted("c2"), _submitted("c3")]
    decisions = [_decision("c3"), _decision("c1"), _decision("c2")]

    assert [item.candidate_id for item in _admit(decisions, submitted)] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_verifier_timeout_returns_none():
    class _SlowModel:
        async def ainvoke(self, _messages):
            await asyncio.sleep(1.0)
            return VisualVerificationResult(decisions=[])

    result = await verify_candidates(
        [_submitted("c1")],
        user_request="cho t thong tin ve t1",
        image_query="T1 League of Legends team photo",
        factual_query="T1 roster 2026",
        result_titles=["LoL: T1 completed 2026 LCK roster"],
        model=_SlowModel(),
        timeout=0.05,
    )

    assert result is None


@pytest.mark.asyncio
async def test_verifier_provider_error_returns_none():
    class _BrokenModel:
        async def ainvoke(self, _messages):
            raise RuntimeError("provider refused")

    result = await verify_candidates(
        [_submitted("c1")],
        user_request="q",
        image_query="i",
        factual_query="f",
        result_titles=[],
        model=_BrokenModel(),
        timeout=1.0,
    )

    assert result is None


@pytest.mark.asyncio
async def test_verifier_sends_one_message_with_every_thumbnail():
    captured: list = []

    class _Recorder:
        async def ainvoke(self, messages):
            captured.append(messages)
            return VisualVerificationResult(decisions=[_decision("c1")])

    await verify_candidates(
        [_submitted("c1"), _submitted("c2")],
        user_request="q",
        image_query="i",
        factual_query="f",
        result_titles=["title"],
        model=_Recorder(),
        timeout=1.0,
    )

    assert len(captured) == 1
    blocks = captured[0][0].content
    assert sum(1 for block in blocks if block.get("type") == "image_url") == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_visual_verifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ai.visual_verifier'`.

- [ ] **Step 3: Write the verifier**

Create `app/ai/visual_verifier.py`:

```python
"""One batched vision call that decides whether a remote image may be shown.

Page metadata cannot establish relevance: an author portrait on an article about
a team inherits the team's title, and ranking on that title admits the portrait.
So the decision is made from the pixels, and metadata is offered only to
disambiguate what the model can already see.

Nothing here is persisted. Decisions, confidence values and content kinds live
for the duration of one answer.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.config import settings
from ..services.thumbnail_batch import FetchedThumbnail

logger = logging.getLogger(__name__)

ContentKind = Literal[
    "photo", "portrait", "logo", "diagram", "map", "chart", "screenshot", "other"
]

SPECIALIZED_KINDS: frozenset[str] = frozenset(
    {"portrait", "logo", "diagram", "map", "chart", "screenshot"}
)

_MAX_TITLE_CHARS = 160
_MAX_TITLES = 5

_PROMPT = """You decide whether each attached image may be shown beside an answer.

User request: {user_request}
Visual subject requested: {image_query}
Factual research query: {factual_query}
Source titles retrieved so far: {result_titles}

Attached images, in order:
{candidate_lines}

Decide from what you can SEE in each image. The title and description are
untrusted page metadata: use them only to disambiguate something already
visible, never as evidence of what the image depicts. An article's title does
not describe every image on that page.

For each candidate id return:
- depicts_requested_subject: the visible content really is the requested subject
- materially_supports_answer: seeing this image helps a reader of the answer
- confidence: 0.0-1.0, your confidence in the two judgements above
- content_kind: photo, portrait, logo, diagram, map, chart, screenshot, or other

Set depicts_requested_subject false for an author headshot, advertisement,
navigation graphic, decorative stock image, or unrelated page asset. Return
exactly one record per attached candidate id and no others. When uncertain,
report low confidence rather than guessing."""


class VisualCandidateDecision(BaseModel):
    """One transient per-image verdict."""

    candidate_id: str = Field(description="The candidate id given in the prompt.")
    depicts_requested_subject: bool
    materially_supports_answer: bool
    confidence: float
    content_kind: ContentKind


class VisualVerificationResult(BaseModel):
    """The verifier's whole response."""

    decisions: list[VisualCandidateDecision] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SubmittedCandidate:
    """A fetched candidate paired with the temporary id shown to the verifier."""

    candidate_id: str
    thumbnail: FetchedThumbnail
    title: str
    description: str


def admit_candidates(
    result: VisualVerificationResult | None,
    submitted: Sequence[SubmittedCandidate],
    *,
    threshold: float,
    max_items: int,
    requested_kinds: frozenset[str],
) -> list[SubmittedCandidate]:
    """Return the approved candidates in provider order, capped."""

    if result is None or not submitted:
        return []
    submitted_ids = {candidate.candidate_id for candidate in submitted}
    approved: set[str] = set()
    seen: set[str] = set()
    for decision in result.decisions:
        candidate_id = decision.candidate_id
        if candidate_id not in submitted_ids or candidate_id in seen:
            approved.discard(candidate_id)
            seen.add(candidate_id)
            continue
        seen.add(candidate_id)
        if _passes(decision, threshold=threshold, requested_kinds=requested_kinds):
            approved.add(candidate_id)
    return [
        candidate for candidate in submitted if candidate.candidate_id in approved
    ][: max(0, int(max_items))]


def _passes(
    decision: VisualCandidateDecision,
    *,
    threshold: float,
    requested_kinds: frozenset[str],
) -> bool:
    if not decision.depicts_requested_subject or not decision.materially_supports_answer:
        return False
    if not 0.0 <= decision.confidence <= 1.0:
        return False
    if decision.confidence < float(threshold):
        return False
    if decision.content_kind in SPECIALIZED_KINDS:
        return decision.content_kind in requested_kinds
    return True


async def verify_candidates(
    submitted: Sequence[SubmittedCandidate],
    *,
    user_request: str,
    image_query: str,
    factual_query: str,
    result_titles: Sequence[str],
    model: Any | None = None,
    timeout: float,
) -> VisualVerificationResult | None:
    """Run one structured vision call. Returns ``None`` on any failure."""

    if not submitted:
        return None
    resolved_model = model if model is not None else build_verifier_model()
    if resolved_model is None:
        return None
    message = _build_message(
        submitted,
        user_request=user_request,
        image_query=image_query,
        factual_query=factual_query,
        result_titles=result_titles,
    )
    try:
        async with asyncio.timeout(max(0.001, float(timeout))):
            response = await resolved_model.ainvoke([message])
    except Exception as exc:
        logger.debug("Visual verification unavailable: %s", type(exc).__name__)
        return None
    if isinstance(response, VisualVerificationResult):
        return response
    try:
        return VisualVerificationResult.model_validate(response)
    except Exception:
        logger.debug("Visual verification returned an unusable response shape")
        return None


def _build_message(
    submitted: Sequence[SubmittedCandidate],
    *,
    user_request: str,
    image_query: str,
    factual_query: str,
    result_titles: Sequence[str],
) -> Any:
    from langchain_core.messages import HumanMessage

    candidate_lines = "\n".join(
        f"- {candidate.candidate_id}: title={_bounded(candidate.title)!r} "
        f"description={_bounded(candidate.description)!r}"
        for candidate in submitted
    )
    text = _PROMPT.format(
        user_request=_bounded(user_request, 500),
        image_query=_bounded(image_query),
        factual_query=_bounded(factual_query),
        result_titles="; ".join(_bounded(title) for title in result_titles[:_MAX_TITLES]) or "none",
        candidate_lines=candidate_lines,
    )
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for candidate in submitted:
        encoded = base64.b64encode(candidate.thumbnail.image.content).decode("ascii")
        blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{candidate.thumbnail.image.media_type};base64,{encoded}"
                },
            }
        )
    return HumanMessage(content=blocks)


def _bounded(value: Any, limit: int = _MAX_TITLE_CHARS) -> str:
    return " ".join(str(value or "").split())[:limit]


def build_verifier_model() -> Any | None:
    """Build the configured vision model with structured output, or None."""

    try:
        from .model_factory import ModelFactory

        model = ModelFactory.create_model(
            provider="gemini",
            model=str(settings.image_verification_model),
            api_key=str(settings.gemini_api_key or ""),
            temperature=0.0,
            media_resolution=str(settings.image_verification_media_resolution),
        )
        return model.with_structured_output(VisualVerificationResult)
    except Exception as exc:
        logger.warning("Visual verifier model unavailable: %s", exc)
        return None
```

- [ ] **Step 4: Add the settings**

In `app/core/config.py`, after `research_budget_enabled`:

```python
    vision_image_verification_enabled: bool = Field(
        default=False,
        description=(
            "Rollout flag for vision-verified remote web images. When False, no "
            "remote web image reaches an answer."
        ),
    )
    image_verification_model: str = Field(
        default="gemini-3-flash-preview",
        description="Vision model used to verify remote image relevance.",
    )
    image_verification_media_resolution: str = Field(
        default="low",
        description="Media resolution for verifier thumbnails: low, medium, or high.",
    )
```

**Corrected 2026-08-05.** `"low"` is a friendly name, not the wire value.
`google.genai.types.MediaResolution` accepts only `MEDIA_RESOLUTION_UNSPECIFIED`,
`MEDIA_RESOLUTION_LOW`, `MEDIA_RESOLUTION_MEDIUM`, `MEDIA_RESOLUTION_HIGH`.
Passing `"low"` produces `UserWarning: low is not a valid MediaResolution` and a
synthetic non-canonical enum member that the live API will reject or ignore —
which means the vision call never actually works, silently, because
`verify_candidates` catches the failure and returns `None`. The setting stays
human-friendly; `build_verifier_model` maps it:

```python
_MEDIA_RESOLUTIONS = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
}
```

An unrecognized value falls back to `MEDIA_RESOLUTION_LOW` rather than being
forwarded raw. A test must construct the real model and assert no
`UserWarning` is emitted — every async test injects a fake model, so nothing
otherwise exercises `build_verifier_model` at all.
    image_verification_confidence_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum verifier confidence for admitting a remote image.",
    )
    image_verification_max_candidates: int = Field(
        default=6,
        ge=1,
        le=10,
        description="Maximum candidates submitted to one verifier call.",
    )
    image_verification_deadline_seconds: float = Field(
        default=4.0,
        gt=0,
        description=(
            "Hard end-to-end deadline for the image path, from image-search "
            "dispatch to verifier verdict. Exceeding it yields a text-only answer."
        ),
    )
    image_verification_thumbnail_timeout_seconds: float = Field(
        default=1.5,
        gt=0,
        description="Per-thumbnail download timeout during verification.",
    )
    rich_image_gallery_max_items: int = Field(
        default=6,
        ge=2,
        le=8,
        description=(
            "Images in one verified gallery grid. Only reachable through "
            "image_intent='gallery'; figure mode stays bound by "
            "rich_auto_place_max_images."
        ),
    )
```

Leave the existing `rich_image_group_max_items` (default 3, `le=3`) alone. It still
governs the legacy grouping path inside `build_image_candidates_from_tool_result`,
which remains reachable when a model loads `brave_image_search` directly through
`tool_search`.

In the same edit, lower the existing Brave timeout so image search plus one
thumbnail leaves the verifier room inside the 4-second deadline:

```python
    brave_image_search_timeout_seconds: float = Field(
        default=2.0,
        gt=0,
        description=(
            "Brave image search request timeout. Kept under the image-path "
            "deadline so a slow provider cannot consume the verifier's budget."
        ),
    )
```

Find the existing field first and edit it in place rather than adding a second
definition:

```bash
grep -n "brave_image_search_timeout_seconds" app/core/config.py
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_visual_verifier.py -v`
Expected: PASS. If `test_verifier_sends_one_message_with_every_thumbnail` fails on `block.get`, the content blocks are not plain dicts — inspect `captured[0][0].content` and adjust the assertion to the real block type; do not change the message shape, since `langchain-google-genai` expects `image_url` blocks.

- [ ] **Step 6: Confirm the timeouts cannot exceed the deadline**

Add to `tests/test_visual_verifier.py`:

```python
def test_configured_timeouts_fit_inside_the_image_deadline():
    from app.core.config import settings

    brave_timeout = float(settings.brave_image_search_timeout_seconds)
    thumbnail_timeout = float(settings.image_verification_thumbnail_timeout_seconds)
    deadline = float(settings.image_verification_deadline_seconds)

    assert brave_timeout + thumbnail_timeout < deadline, (
        "image search plus one thumbnail must leave room for the verifier call"
    )
```

Run: `.venv/Scripts/python.exe -m pytest tests/test_visual_verifier.py -v`
Expected: PASS — 2.0 + 1.5 = 3.5 is under the 4.0 deadline, leaving 0.5s of headroom for the verifier call. If this fails, the Brave timeout edit in Step 4 did not land. Lower a provider timeout; never raise the deadline.

- [ ] **Step 7: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/ai/visual_verifier.py app/core/config.py
git add app/ai/visual_verifier.py app/core/config.py tests/test_visual_verifier.py
git status --short
git commit -m "feat: add transient visual relevance verifier"
```

---

### Task 4: Verified-image sink

Approved candidates must reach the inventory without ever passing through the model-visible tool result. A context-scoped sink keeps rejected candidates unreachable by construction rather than by careful string handling.

**Files:**
- Create: `app/ai/verified_image_sink.py`
- Modify: `app/ai/tool_execution.py:580-618` and `:1999-2050`
- Create: `tests/test_verified_image_sink.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `verified_image_sink()` context manager yielding a `list[dict[str, Any]]`.
  - `offer_verified_images(candidates: Sequence[Mapping[str, Any]]) -> None`
  - `_attach_rich_candidates_to_artifact(...)` gains a keyword-only `verified_images: list[dict[str, Any]] | None = None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_verified_image_sink.py`:

```python
from __future__ import annotations

import asyncio

import pytest

from app.ai.tool_execution import _attach_rich_candidates_to_artifact
from app.ai.verified_image_sink import offer_verified_images, verified_image_sink


def test_offers_are_collected_inside_the_sink():
    with verified_image_sink() as sink:
        offer_verified_images([{"id": "image:verified:a"}])
        offer_verified_images([{"id": "image:verified:b"}])

    assert [item["id"] for item in sink] == ["image:verified:a", "image:verified:b"]


def test_offers_outside_a_sink_are_dropped_silently():
    offer_verified_images([{"id": "image:verified:orphan"}])  # must not raise


def test_nested_sinks_do_not_leak_into_each_other():
    with verified_image_sink() as outer:
        with verified_image_sink() as inner:
            offer_verified_images([{"id": "inner"}])
        offer_verified_images([{"id": "outer"}])

    assert [item["id"] for item in inner] == ["inner"]
    assert [item["id"] for item in outer] == ["outer"]


@pytest.mark.asyncio
async def test_offers_from_an_awaited_coroutine_reach_the_sink():
    async def _tool():
        offer_verified_images([{"id": "from-await"}])

    with verified_image_sink() as sink:
        await _tool()

    assert [item["id"] for item in sink] == ["from-await"]


@pytest.mark.asyncio
async def test_offers_from_a_child_task_reach_the_sink():
    async def _tool():
        offer_verified_images([{"id": "from-task"}])

    with verified_image_sink() as sink:
        await asyncio.create_task(_tool())

    assert [item["id"] for item in sink] == ["from-task"]


def test_verified_images_are_attached_to_the_artifact():
    artifact: dict = {}

    _attach_rich_candidates_to_artifact(
        artifact,
        raw_result=None,
        result_text="{}",
        render=None,
        tool_call_id="call-1",
        tool_name="web_research",
        verified_images=[{"id": "image:verified:a", "type": "image"}],
    )

    assert artifact["_rich_item_candidates"] == [{"id": "image:verified:a", "type": "image"}]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_verified_image_sink.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ai.verified_image_sink'`.

- [ ] **Step 3: Write the sink**

Create `app/ai/verified_image_sink.py`:

```python
"""A turn-scoped channel for candidates that passed visual verification.

Approved candidates must reach the rich-item inventory without appearing in the
model-visible tool result, and rejected candidates must be unreachable rather
than merely unmentioned. The sink is a list owned by the tool-execution layer
and mutated by the tool, so no candidate is ever serialized into the text the
model reads.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_sink: ContextVar[list[dict[str, Any]] | None] = ContextVar("verified_image_sink", default=None)


@contextmanager
def verified_image_sink() -> Iterator[list[dict[str, Any]]]:
    """Collect verified candidates offered while the block is active."""

    collected: list[dict[str, Any]] = []
    token = _sink.set(collected)
    try:
        yield collected
    finally:
        _sink.reset(token)


def offer_verified_images(candidates: Sequence[Mapping[str, Any]]) -> None:
    """Offer approved candidates to the active sink, if any."""

    sink = _sink.get()
    if sink is None:
        return
    sink.extend(dict(candidate) for candidate in candidates)
```

- [ ] **Step 4: Accept verified images in the attachment seam**

In `app/ai/tool_execution.py`, add the keyword-only parameter to `_attach_rich_candidates_to_artifact` and append the offered candidates:

```python
def _attach_rich_candidates_to_artifact(
    artifact: dict[str, Any],
    *,
    raw_result: Any,
    result_text: str,
    render: dict[str, Any] | None,
    tool_call_id: str | None,
    tool_name: str,
    verified_images: list[dict[str, Any]] | None = None,
) -> None:
```

Immediately before the closing `if candidates:` block:

```python
    # Verified remote images arrive out-of-band: they must never be serialized
    # into the model-visible result, because that is also how a rejected
    # candidate would become placeable.
    for candidate in verified_images or []:
        if isinstance(candidate, dict):
            candidates.append(candidate)
```

- [ ] **Step 5: Open the sink around each tool invocation**

In `app/ai/tool_execution.py`, add the import:

```python
from .verified_image_sink import verified_image_sink
```

Then wrap the invocation at line 1999-2005 and thread the collected list into the attachment call at line 2042:

```python
        try:
            with verified_image_sink() as offered_images:
                (
                    result,
                    error_detail,
                    error_content,
                    execution_detail,
                ) = await invoke_tool_with_policy(
                    tool,
                    tool_args,
                    tool_name=tool_name,
                    tool_map=tool_map,
                )
```

and:

```python
            _attach_rich_candidates_to_artifact(
                artifact,
                raw_result=result,
                result_text=result_text,
                render=normalized_result.render,
                tool_call_id=tool_id,
                tool_name=tool_name,
                verified_images=offered_images,
            )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_verified_image_sink.py tests/test_tool_execution_rendering.py -v`
Expected: PASS.

- [ ] **Step 7: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/ai/verified_image_sink.py app/ai/tool_execution.py
git add app/ai/verified_image_sink.py app/ai/tool_execution.py tests/test_verified_image_sink.py
git status --short
git commit -m "feat: add out-of-band channel for verified images"
```

---

### Task 5: web_research orchestrator

The single model-facing research operation. It runs the two providers concurrently, spends the turn budget, verifies, and offers only approved candidates.

**Files:**
- Create: `app/ai/web_research_tool.py`
- Create: `app/ai/image_verification_flow.py`
- Modify: `app/ai/tool_execution.py` (two additive keyword parameters — see Step 4b)
- Create: `tests/test_web_research_tool.py`
- Modify: `tests/test_tool_execution_rendering.py`

**Interfaces:**
- Consumes: `get_research_budget` (Task 1), `fetch_thumbnails` (Task 2), `verify_candidates`/`admit_candidates`/`SubmittedCandidate` (Task 3), `offer_verified_images` (Task 4), `build_image_candidates_from_tool_result` and `_group_image_candidates` (existing, for Brave results only).
- Produces: `create_web_research_tool(*, tavily_tool=None, brave_tool=None, web_image_service=None, verifier_model=None) -> StructuredTool` named `web_research`, with metadata `{"tool_origin": "internal", "qualified_tool_id": "internal::web_research"}`. Its JSON result is the Tavily payload plus `"research": {"reused": bool, "searches_used": int}`, and never any image field.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_research_tool.py`:

```python
from __future__ import annotations

import asyncio
import json
import re

import pytest

from app.ai.research_budget import reset_research_budget
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.verified_image_sink import verified_image_sink
from app.ai.visual_verifier import VisualCandidateDecision, VisualVerificationResult
from app.ai.web_research_tool import create_web_research_tool
from app.services.web_image_service import FetchedWebImage

CONVERSATION_ID = "11111111-1111-1111-1111-111111111111"

TAVILY_PAYLOAD = json.dumps(
    {
        "results": [
            {
                "index": 1,
                "title": "LoL: T1 completed 2026 LCK roster",
                "url": "https://sheepesports.example/t1",
                "content": "T1 finalized its roster.",
                "score": 0.88,
            }
        ],
        "total_results": 1,
        "answer": "T1 is a South Korean esports organization.",
        "provider": "tavily",
        "operation": "search",
        "query": "T1 roster 2026",
    }
)

BRAVE_PAYLOAD = json.dumps(
    {
        "query": "T1 League of Legends team photo",
        "provider": "brave_image_search",
        "images": [
            {
                "url": "https://cdn.example/portrait.jpg",
                "provider": "brave_image_search",
                "mime_type": "image/jpeg",
                "title": "Moi",
                "description": "Moi",
                "width": 1080,
                "height": 1600,
                "source_url": "https://sheepesports.example/t1",
            },
            {
                "url": "https://cdn.example/team.jpg",
                "provider": "brave_image_search",
                "mime_type": "image/jpeg",
                "title": "T1 roster",
                "description": "T1 roster",
                "width": 995,
                "height": 565,
                "source_url": "https://sheepesports.example/t1",
            },
        ],
        "total_results": 2,
    }
)


class _FakeTool:
    def __init__(self, name: str, payload: str, delay: float = 0.0):
        self.name = name
        self.payload = payload
        self.delay = delay
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict) -> str:
        self.calls.append(dict(args))
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.payload


class _FakeImageService:
    def __init__(self):
        self.fetched: list[str] = []

    async def fetch_url(self, url: str, *, provider: str = "other") -> FetchedWebImage:
        self.fetched.append(url)
        return FetchedWebImage(
            content=b"bytes", media_type="image/jpeg", width=995, height=565
        )


class _ApproveOnlyTeamPhoto:
    """Approves whichever candidate line mentions the team, rejects the portrait."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        text = messages[0].content[0]["text"]
        decisions = []
        # Match only candidate lines; the prompt's instruction bullets also
        # start with "- " and must not be read as candidate ids.
        for candidate_id, line in re.findall(r"^- (c\d+): (.*)$", text, flags=re.MULTILINE):
            is_portrait = "Moi" in line
            decisions.append(
                VisualCandidateDecision(
                    candidate_id=candidate_id,
                    depicts_requested_subject=not is_portrait,
                    materially_supports_answer=not is_portrait,
                    confidence=0.95,
                    content_kind="portrait" if is_portrait else "photo",
                )
            )
        return VisualVerificationResult(decisions=decisions)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)
    monkeypatch.setattr(
        "app.ai.web_research_tool.settings.vision_image_verification_enabled",
        True,
        raising=False,
    )
    yield
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)


def _tool(tavily, brave, service, verifier):
    return create_web_research_tool(
        tavily_tool=tavily,
        brave_tool=brave,
        web_image_service=service,
        verifier_model=verifier,
    )


async def _run(tool, **kwargs):
    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ), verified_image_sink() as sink:
        raw = await tool.ainvoke(kwargs)
    return json.loads(raw), sink


@pytest.mark.asyncio
async def test_only_the_verified_team_photo_is_offered():
    tavily = _FakeTool("tavily_search", TAVILY_PAYLOAD)
    brave = _FakeTool("brave_image_search", BRAVE_PAYLOAD)

    payload, sink = await _run(
        _tool(tavily, brave, _FakeImageService(), _ApproveOnlyTeamPhoto()),
        query="T1 roster 2026",
        image_query="T1 League of Legends team photo",
    )

    assert len(sink) == 1
    assert sink[0]["payload"]["url"] == "https://cdn.example/team.jpg"
    assert "portrait.jpg" not in json.dumps(sink)
    assert "images" not in payload
    assert "portrait.jpg" not in json.dumps(payload)
    assert payload["answer"].startswith("T1 is a South Korean")


def _brave_payload(count: int) -> str:
    """A Brave result with ``count`` distinct, verifiable team photos."""
    return json.dumps(
        {
            "query": "T1 League of Legends team photo",
            "provider": "brave_image_search",
            "images": [
                {
                    "url": f"https://cdn.example/team-{index}.jpg",
                    "provider": "brave_image_search",
                    "mime_type": "image/jpeg",
                    "title": f"T1 roster {index}",
                    "description": f"T1 roster {index}",
                    "width": 995,
                    "height": 565,
                    "source_url": "https://sheepesports.example/t1",
                }
                for index in range(count)
            ],
            "total_results": count,
        }
    )


@pytest.mark.asyncio
async def test_gallery_intent_returns_one_grid_item_holding_every_survivor():
    verifier = _ApproveOnlyTeamPhoto()

    _, sink = await _run(
        _tool(
            _FakeTool("tavily_search", TAVILY_PAYLOAD),
            _FakeTool("brave_image_search", _brave_payload(4)),
            _FakeImageService(),
            verifier,
        ),
        query="T1 roster 2026",
        image_query="T1 League of Legends team photo",
        image_intent="gallery",
    )

    assert len(sink) == 1
    assert sink[0]["type"] == "image_group"
    assert len(sink[0]["payload"]["items"]) == 4
    assert verifier.calls == 1


@pytest.mark.asyncio
async def test_gallery_candidates_reach_the_verifier_individually():
    """Grouping before verification would hide images and cap discovery."""

    seen: list[str] = []

    class _Recorder:
        calls = 0

        async def ainvoke(self, messages):
            text = messages[0].content[0]["text"]
            seen.extend(re.findall(r"^- (c\d+): ", text, flags=re.MULTILINE))
            return VisualVerificationResult(decisions=[])

    await _run(
        _tool(
            _FakeTool("tavily_search", TAVILY_PAYLOAD),
            _FakeTool("brave_image_search", _brave_payload(5)),
            _FakeImageService(),
            _Recorder(),
        ),
        query="T1 roster 2026",
        image_query="T1 League of Legends team photo",
        image_intent="gallery",
    )

    assert len(seen) == 5, "every candidate must be judged on its own pixels"


@pytest.mark.asyncio
async def test_figure_intent_caps_at_two_individual_items():
    _, sink = await _run(
        _tool(
            _FakeTool("tavily_search", TAVILY_PAYLOAD),
            _FakeTool("brave_image_search", _brave_payload(4)),
            _FakeImageService(),
            _ApproveOnlyTeamPhoto(),
        ),
        query="T1 roster 2026",
        image_query="T1 League of Legends team photo",
    )

    assert len(sink) == 2
    assert all(item["type"] == "image" for item in sink)


@pytest.mark.asyncio
async def test_gallery_with_a_single_survivor_is_not_a_one_cell_grid():
    _, sink = await _run(
        _tool(
            _FakeTool("tavily_search", TAVILY_PAYLOAD),
            _FakeTool("brave_image_search", _brave_payload(1)),
            _FakeImageService(),
            _ApproveOnlyTeamPhoto(),
        ),
        query="T1 roster 2026",
        image_query="T1 League of Legends team photo",
        image_intent="gallery",
    )

    assert len(sink) == 1
    assert sink[0]["type"] == "image"


@pytest.mark.asyncio
async def test_no_image_query_skips_brave_and_the_verifier():
    tavily = _FakeTool("tavily_search", TAVILY_PAYLOAD)
    brave = _FakeTool("brave_image_search", BRAVE_PAYLOAD)
    verifier = _ApproveOnlyTeamPhoto()

    payload, sink = await _run(
        _tool(tavily, brave, _FakeImageService(), verifier), query="explain big-O notation"
    )

    assert brave.calls == []
    assert verifier.calls == 0
    assert sink == []
    assert payload["results"]


@pytest.mark.asyncio
async def test_providers_run_concurrently():
    tavily = _FakeTool("tavily_search", TAVILY_PAYLOAD, delay=0.3)
    brave = _FakeTool("brave_image_search", BRAVE_PAYLOAD, delay=0.3)

    started = asyncio.get_running_loop().time()
    await _run(
        _tool(tavily, brave, _FakeImageService(), _ApproveOnlyTeamPhoto()),
        query="T1 roster 2026",
        image_query="T1 team photo",
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.55, "tavily and brave must overlap"


@pytest.mark.asyncio
async def test_near_duplicate_query_reuses_the_first_result():
    tavily = _FakeTool("tavily_search", TAVILY_PAYLOAD)
    tool = _tool(tavily, _FakeTool("brave_image_search", BRAVE_PAYLOAD), _FakeImageService(), None)

    await _run(tool, query="T1 League of Legends Esports team news roster 2026")
    payload, _ = await _run(tool, query="T1 League of Legends team overview roster news 2026")

    assert len(tavily.calls) == 1
    assert payload["research"]["reused"] is True


@pytest.mark.asyncio
async def test_second_image_query_does_not_launch_another_brave_call():
    brave = _FakeTool("brave_image_search", BRAVE_PAYLOAD)
    tool = _tool(_FakeTool("tavily_search", TAVILY_PAYLOAD), brave, _FakeImageService(), _ApproveOnlyTeamPhoto())

    _, first = await _run(tool, query="T1 roster 2026", image_query="T1 team photo")
    _, second = await _run(tool, query="T1 sponsors 2026", image_query="T1 jersey photo")

    assert len(brave.calls) == 1
    assert [item["id"] for item in second] == [item["id"] for item in first]


@pytest.mark.asyncio
async def test_brave_failure_yields_a_normal_text_answer():
    class _Broken:
        name = "brave_image_search"

        async def ainvoke(self, args):
            raise RuntimeError("brave down")

    payload, sink = await _run(
        _tool(_FakeTool("tavily_search", TAVILY_PAYLOAD), _Broken(), _FakeImageService(), _ApproveOnlyTeamPhoto()),
        query="T1 roster 2026",
        image_query="T1 team photo",
    )

    assert sink == []
    assert payload["results"]


@pytest.mark.asyncio
async def test_verifier_returning_nothing_yields_a_text_answer():
    class _RejectAll:
        async def ainvoke(self, messages):
            return VisualVerificationResult(decisions=[])

    _, sink = await _run(
        _tool(_FakeTool("tavily_search", TAVILY_PAYLOAD), _FakeTool("brave_image_search", BRAVE_PAYLOAD), _FakeImageService(), _RejectAll()),
        query="T1 roster 2026",
        image_query="T1 team photo",
    )

    assert sink == []


@pytest.mark.asyncio
async def test_tavily_failure_is_reported_as_a_research_error():
    class _Broken:
        name = "tavily_search"

        async def ainvoke(self, args):
            raise RuntimeError("tavily down")

    payload, _ = await _run(
        _tool(_Broken(), _FakeTool("brave_image_search", BRAVE_PAYLOAD), _FakeImageService(), None),
        query="T1 roster 2026",
    )

    assert payload["status"] == "error"
    assert payload["retryable"] is True


@pytest.mark.asyncio
async def test_disabled_flag_skips_the_image_path_entirely(monkeypatch):
    monkeypatch.setattr(
        "app.ai.web_research_tool.settings.vision_image_verification_enabled",
        False,
        raising=False,
    )
    brave = _FakeTool("brave_image_search", BRAVE_PAYLOAD)

    _, sink = await _run(
        _tool(_FakeTool("tavily_search", TAVILY_PAYLOAD), brave, _FakeImageService(), _ApproveOnlyTeamPhoto()),
        query="T1 roster 2026",
        image_query="T1 team photo",
    )

    assert brave.calls == []
    assert sink == []


def test_tool_identity_is_internal():
    tool = _tool(None, None, None, None)

    assert tool.name == "web_research"
    assert tool.metadata["qualified_tool_id"] == "internal::web_research"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_research_tool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ai.web_research_tool'`.

- [ ] **Step 3: Write the orchestrator**

Create `app/ai/web_research_tool.py`:

```python
"""The single model-facing research operation.

The model asks one question and optionally names a visual subject. The server
decides everything else: whether to hit the network at all, whether to look for
an image, and whether any image it found may be shown. Prompted coordination of
two providers proved unreliable — a trace shows three sequential text searches
and no image search at all — so the sequencing lives here instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..core.config import settings
from .research_budget import get_research_budget
from .tool_context import get_tool_context
from .verified_image_sink import offer_verified_images

logger = logging.getLogger(__name__)

_DESCRIPTION = (
    "Research the web. Returns a synthesized answer plus ranked sources with URLs.\n\n"
    "Set image_query to a short, concrete visual subject when a picture would help "
    "the reader see what the answer is about — a product, device, place, building, "
    "artwork, organism, vehicle, or screen. Write the subject yourself: no question "
    "words, one subject, plus a disambiguator or a form word (photo, diagram, map, "
    "chart) when it matters.\n\n"
    "Leave image_query unset for abstract subjects (code, math, policy, definitions, "
    "planning) and whenever you are unsure whether an image would help. An uncertain "
    "image decision uses no image_query at all.\n\n"
    "Set image_intent='gallery' when the user asks to SEE several instances or to "
    "compare things — a roster, a set of logos, colour options, a lineup. Otherwise "
    "leave it unset: the default places up to two images beside the prose they "
    "support. Never state how many images you want; the layout decides, and only "
    "images verified against the subject survive.\n\n"
    "Approved images appear in your available rich items. Not every image_query "
    "produces one, and a complete answer never depends on an image. A gallery "
    "arrives as ONE grid item with one marker."
)


class WebResearchInput(BaseModel):
    query: str = Field(description="The factual research query.")
    image_query: str | None = Field(
        default=None,
        description="Short concrete visual subject, or omit when an image would not help.",
    )
    image_intent: Literal["figure", "gallery"] | None = Field(
        default=None,
        description=(
            "Layout: 'figure' (default) for up to two images beside the prose, "
            "'gallery' for a grid when the user asks to see several instances or "
            "to compare things. Never state a count."
        ),
    )
    max_results: int | None = Field(default=None, description="Optional result count.")
    search_depth: str | None = Field(default=None, description="Optional Tavily depth.")


def create_web_research_tool(
    *,
    tavily_tool: Any | None = None,
    brave_tool: Any | None = None,
    web_image_service: Any | None = None,
    verifier_model: Any | None = None,
) -> StructuredTool:
    """Build the ``web_research`` tool. Dependencies are injected in tests."""

    async def _research(
        query: str,
        image_query: str | None = None,
        image_intent: str | None = None,
        max_results: int | None = None,
        search_depth: str | None = None,
    ) -> str:
        conversation_id = get_tool_context().conversation_id
        budget = get_research_budget(conversation_id)
        wants_image = bool(str(image_query or "").strip())

        reused = budget.find_reuse(query) if settings.research_budget_enabled else None
        search_task: asyncio.Task[str] | None = None
        if reused is None:
            # reserve_search claims the slot in one step; a bare check here would
            # race a concurrent web_research call across the await below.
            if settings.research_budget_enabled and not budget.reserve_search(query):
                return _budget_reused_payload(budget)
            search_task = asyncio.create_task(
                _run_search(tavily_tool, query, max_results, search_depth)
            )

        image_task: asyncio.Task[list[dict[str, Any]]] | None = None
        if wants_image and _image_path_open(budget):
            image_task = asyncio.create_task(
                _discover_and_verify(
                    brave_tool=brave_tool,
                    web_image_service=web_image_service,
                    verifier_model=verifier_model,
                    user_request=query,
                    image_query=str(image_query).strip(),
                    factual_query=query,
                    image_intent=image_intent,
                )
            )

        if search_task is not None:
            try:
                search_text = await search_task
            except Exception as exc:
                if image_task is not None:
                    image_task.cancel()
                logger.warning("Research search failed: %s", exc)
                return _error_payload(str(exc))
            budget.record_search(query, search_text)
            search_reused = False
        else:
            search_text = reused or ""
            search_reused = True

        approved = await _collect_images(image_task, budget, wants_image)
        if approved:
            offer_verified_images(approved)
        return _with_research_meta(search_text, reused=search_reused, budget=budget)

    return StructuredTool.from_function(
        coroutine=_research,
        name="web_research",
        description=_DESCRIPTION,
        args_schema=WebResearchInput,
        metadata={
            "tool_origin": "internal",
            "qualified_tool_id": "internal::web_research",
        },
    )


def _image_path_open(budget: Any) -> bool:
    if not settings.vision_image_verification_enabled:
        return False
    if not settings.inline_rich_response_enabled:
        return False
    return budget.may_image_search()


async def _collect_images(
    image_task: asyncio.Task[list[dict[str, Any]]] | None,
    budget: Any,
    wants_image: bool,
) -> list[dict[str, Any]]:
    if image_task is None:
        return budget.image_result() if wants_image else []
    try:
        approved = await image_task
    except Exception as exc:
        logger.debug("Image path abandoned: %s", type(exc).__name__)
        approved = []
    budget.record_image_search(approved)
    return approved


async def _run_search(
    tavily_tool: Any | None,
    query: str,
    max_results: int | None,
    search_depth: str | None,
) -> str:
    tool = tavily_tool if tavily_tool is not None else await _resolve_tool("tavily", "tavily_search")
    if tool is None:
        raise RuntimeError("tavily_search is unavailable")
    args: dict[str, Any] = {"query": query}
    if max_results is not None:
        args["max_results"] = max_results
    if search_depth is not None:
        args["search_depth"] = search_depth
    return str(await tool.ainvoke(args))


async def _discover_and_verify(
    *,
    brave_tool: Any | None,
    web_image_service: Any | None,
    verifier_model: Any | None,
    user_request: str,
    image_query: str,
    factual_query: str,
    image_intent: str | None = None,
) -> list[dict[str, Any]]:
    """Return public candidate dicts for approved images, or an empty list."""

    from .image_verification_flow import discover_and_verify_images

    try:
        async with asyncio.timeout(float(settings.image_verification_deadline_seconds)):
            return await discover_and_verify_images(
                brave_tool=brave_tool,
                web_image_service=web_image_service,
                verifier_model=verifier_model,
                user_request=user_request,
                image_query=image_query,
                factual_query=factual_query,
                image_intent=image_intent,
            )
    except Exception as exc:
        logger.debug("Image verification abandoned: %s", type(exc).__name__)
        return []


async def _resolve_tool(server_name: str, tool_name: str) -> Any | None:
    try:
        from .mcp_registry import get_global_mcp_manager

        manager = get_global_mcp_manager()
        for tool in await manager.get_server_tools(server_name):
            if getattr(tool, "name", None) == tool_name:
                return tool
    except Exception as exc:
        logger.warning("MCP tool %s unavailable: %s", tool_name, exc)
    return None


def _with_research_meta(search_text: str, *, reused: bool, budget: Any) -> str:
    try:
        payload = json.loads(search_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return search_text
    if not isinstance(payload, dict):
        return search_text
    payload.pop("images", None)
    payload["research"] = {"reused": reused, "searches_used": budget.search_calls}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _budget_reused_payload(budget: Any) -> str:
    return json.dumps(
        {
            "research": {
                "reused": True,
                "searches_used": budget.search_calls,
                "note": (
                    "The per-turn research budget is spent. Answer from the results "
                    "already gathered in this turn; another search would return the same "
                    "sources."
                ),
            },
            "accumulated_results": budget.accumulated(),
        },
        ensure_ascii=False,
        indent=2,
    )


def _error_payload(message: str) -> str:
    return json.dumps(
        {
            "status": "error",
            "error_type": "provider_error",
            "retryable": True,
            "hint": "Research is temporarily unavailable. Say so rather than guessing.",
            "message": message[:500],
        },
        ensure_ascii=False,
    )
```

- [ ] **Step 4: Write the discovery-and-verification flow**

Create `app/ai/image_verification_flow.py`:

```python
"""Brave discovery, guarded thumbnail download, verification, admission.

Kept separate from the research tool so the orchestration order stays readable
and each stage is testable on its own.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.config import settings
from ..services.thumbnail_batch import fetch_thumbnails
from .tool_execution import (
    _group_image_candidates,
    build_image_candidates_from_tool_result,
)
from .visual_verifier import (
    SubmittedCandidate,
    admit_candidates,
    verify_candidates,
)

logger = logging.getLogger(__name__)


async def discover_and_verify_images(
    *,
    brave_tool: Any | None,
    web_image_service: Any | None,
    verifier_model: Any | None,
    user_request: str,
    image_query: str,
    factual_query: str,
    image_intent: str | None = None,
) -> list[dict[str, Any]]:
    """Return public candidate dicts for verifier-approved images only.

    ``image_intent`` selects the layout and through it the cap: ``gallery``
    returns a single grid item, anything else returns individual images.
    """

    if brave_tool is None or web_image_service is None:
        return []
    raw = str(await brave_tool.ainvoke({"query": image_query}))
    # group_images=False is load-bearing: the legacy path collapses two or more
    # Brave results into one capped grid, which would both hide individual images
    # from the verifier and cap discovery below the candidate budget.
    candidates = build_image_candidates_from_tool_result(
        raw,
        tool_call_id=None,
        tool_name="brave_image_search",
        group_images=False,
    )
    candidates = candidates[: max(1, int(settings.image_verification_max_candidates))]
    if not candidates:
        return []

    thumbnails = await fetch_thumbnails(
        web_image_service,
        [candidate["payload"]["url"] for candidate in candidates],
        provider="brave",
        per_item_timeout=float(settings.image_verification_thumbnail_timeout_seconds),
        batch_deadline=float(settings.image_verification_deadline_seconds),
    )
    submitted = [
        SubmittedCandidate(
            candidate_id=f"c{index}",
            thumbnail=thumbnail,
            title=str(candidate.get("title") or ""),
            description=str(candidate["payload"].get("description") or ""),
        )
        for index, (candidate, thumbnail) in enumerate(zip(candidates, thumbnails))
        if thumbnail is not None
    ]
    if not submitted:
        return []

    result = await verify_candidates(
        submitted,
        user_request=user_request,
        image_query=image_query,
        factual_query=factual_query,
        result_titles=[],
        model=verifier_model,
        timeout=float(settings.image_verification_deadline_seconds),
    )
    gallery = str(image_intent or "figure").strip().lower() == "gallery"
    approved = admit_candidates(
        result,
        submitted,
        threshold=float(settings.image_verification_confidence_threshold),
        max_items=(
            max(2, int(settings.rich_image_gallery_max_items))
            if gallery
            else max(0, int(settings.rich_auto_place_max_images))
        ),
        requested_kinds=_requested_kinds(f"{user_request} {image_query}"),
    )
    by_id = {f"c{index}": candidate for index, candidate in enumerate(candidates)}
    public = [
        _with_decoded_dimensions(by_id[item.candidate_id], item)
        for item in approved
        if item.candidate_id in by_id
    ]
    if gallery and len(public) >= 2:
        return [
            _group_image_candidates(
                public,
                tool_call_id=None,
                query=image_query,
                metric_provider="brave",
                max_items=max(2, int(settings.rich_image_gallery_max_items)),
            )
        ]
    return public


def _with_decoded_dimensions(
    candidate: dict[str, Any], submitted: SubmittedCandidate
) -> dict[str, Any]:
    """Stamp the dimensions actually decoded from the bytes we fetched."""

    updated = {**candidate, "payload": dict(candidate["payload"])}
    updated["payload"]["width"] = submitted.thumbnail.image.width
    updated["payload"]["height"] = submitted.thumbnail.image.height
    updated["payload"]["mime_type"] = submitted.thumbnail.image.media_type
    return updated


_KIND_WORDS = {
    "portrait": ("portrait", "headshot", "chân dung"),
    "logo": ("logo", "emblem", "badge"),
    "diagram": ("diagram", "schematic", "sơ đồ"),
    "map": ("map", "bản đồ"),
    "chart": ("chart", "graph", "biểu đồ"),
    "screenshot": ("screenshot", "screen shot", "ảnh chụp màn hình"),
}


def _requested_kinds(text: str) -> frozenset[str]:
    """Kinds the user explicitly asked for, which overrides the generic bias."""

    lowered = str(text or "").casefold()
    return frozenset(
        kind for kind, words in _KIND_WORDS.items() if any(word in lowered for word in words)
    )
```

- [ ] **Step 4b: Make grouping opt-out and its cap explicit**

The flow above needs two small changes in `app/ai/tool_execution.py`, both additive
and both defaulted so every existing caller behaves exactly as before.

In `build_image_candidates_from_tool_result`, add a keyword-only parameter and
guard the collapse at the end of the function:

```python
def build_image_candidates_from_tool_result(
    result_text: str,
    *,
    tool_call_id: str | None,
    tool_name: str,
    group_images: bool = True,
) -> list[dict[str, Any]]:
```

```python
    if group_images and metric_provider == "brave" and len(candidates) >= 2:
```

In `_group_image_candidates`, let the caller supply the cap instead of always
reading the legacy setting:

```python
def _group_image_candidates(
    candidates: list[dict[str, Any]],
    *,
    tool_call_id: str | None,
    query: str,
    metric_provider: str,
    max_items: int | None = None,
) -> dict[str, Any]:
```

```python
    cap = max(2, int(max_items if max_items is not None else
                     getattr(settings, "rich_image_group_max_items", 3)))
```

Add two tests to `tests/test_tool_execution_rendering.py` (or the existing file
covering this function): that `group_images=False` returns individual `image`
candidates for a multi-result Brave payload rather than one `image_group`, and
that an explicit `max_items` overrides the legacy setting. Both must fail before
the change.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_research_tool.py -v`
Expected: PASS. If `test_only_the_verified_team_photo_is_offered` fails because the Brave payload collapsed into an `image_group`, confirm `_flatten` expanded it — the verifier must see individual images, never a grid.

- [ ] **Step 6: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/ai/web_research_tool.py app/ai/image_verification_flow.py
git add app/ai/web_research_tool.py app/ai/image_verification_flow.py tests/test_web_research_tool.py
git status --short
git commit -m "feat: add web_research orchestrator with visual verification"
```

---

### Task 6: Route agents through web_research

Binding, unpinning, and prompt guidance. Until this lands, the orchestrator exists but nothing calls it.

**Files:**
- Modify: `app/ai/deferred_tool_binding.py:75-97`
- Modify: `app/ai/agents/base_agent.py` (bind for chat and search)
- Modify: `app/ai/prompts.py:28-37`
- Create: `tests/test_web_research_binding.py`

**Interfaces:**
- Consumes: `create_web_research_tool()` from Task 5.
- Produces: no new interface. `_SEARCH_AGENT_PINNED_SPECS` loses `tavily::tavily_search`; `_IMAGE_SEARCH_PINNED_SPEC` and its agent keys are deleted.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_research_binding.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

from app.ai.agents.base_agent import BaseAgent
from app.ai.deferred_tool_binding import _get_required_pinned_specs
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


def test_tavily_and_brave_are_no_longer_pinned():
    for agent_key in ("chat", "search"):
        specs = _get_required_pinned_specs(agent_key)
        assert "tavily::tavily_search" not in specs
        assert "brave_image_search::brave_image_search" not in specs


def test_time_stays_pinned_for_the_search_agent():
    assert "time::get_current_time" in _get_required_pinned_specs("search")


def _bound_names(monkeypatch, agent_key: str) -> list[str]:
    agent = _BindingTestAgent(agent_config_key=agent_key)
    agent.tools = []
    agent.mcp_manager = None
    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading", lambda _key: True
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.build_deferred_tool_list",
        lambda **kwargs: list(kwargs.get("internal_tools") or []),
    )
    monkeypatch.setattr(agent, "_get_client_runtime_tools", lambda **kwargs: [])
    monkeypatch.setattr(agent, "_get_skills_internal_tools", lambda **kwargs: [])
    return [tool.name for tool in agent._get_tools_for_binding(conversation_id="c1")]


def test_web_research_is_bound_for_chat_and_search(monkeypatch):
    assert "web_research" in _bound_names(monkeypatch, "chat")
    assert "web_research" in _bound_names(monkeypatch, "search")


def test_web_research_is_not_bound_for_other_agents(monkeypatch):
    assert "web_research" not in _bound_names(monkeypatch, "rag")


def test_media_guidance_describes_web_research_only():
    from app.ai.prompts import MEDIA_CAPABILITY_SNIPPET

    assert "web_research" in MEDIA_CAPABILITY_SNIPPET
    assert "image_query" in MEDIA_CAPABILITY_SNIPPET
    assert "image_intent" in MEDIA_CAPABILITY_SNIPPET
    assert "brave_image_search" not in MEDIA_CAPABILITY_SNIPPET
    assert "include_images" not in MEDIA_CAPABILITY_SNIPPET
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_research_binding.py -v`
Expected: FAIL — the pins are still present, `web_research` is unbound, and the media snippet still names `brave_image_search`.

- [ ] **Step 3: Drop the provider pins**

In `app/ai/deferred_tool_binding.py`, replace lines 75-80 with:

```python
# Research reaches Tavily and Brave through the in-process ``web_research``
# tool, which owns the turn budget and the visual verifier. Pinning the raw
# provider tools would let the model bypass both.
_SEARCH_AGENT_PINNED_SPECS = ("time::get_current_time",)
```

Delete `_IMAGE_SEARCH_PINNED_AGENT_KEYS`, `_IMAGE_SEARCH_PINNED_SPEC`, and the two lines in `_get_required_pinned_specs` that append the image-search spec.

- [ ] **Step 4: Bind web_research for chat and search**

In `app/ai/agents/base_agent.py`, add the import:

```python
from ..web_research_tool import create_web_research_tool
```

In `_get_tools_for_binding`, after the `read_tool_result` block added in Phase 1:

```python
        # Research is server-orchestrated: one operation runs the text and image
        # providers, spends the turn budget, and verifies images before the model
        # can place them.
        if self.agent_config_key in {"chat", "search"}:
            _add_internal(create_web_research_tool())
```

- [ ] **Step 5: Rewrite the media guidance**

In `app/ai/prompts.py`, replace `MEDIA_CAPABILITY_SNIPPET` (lines 28-37) with:

```python
MEDIA_CAPABILITY_SNIPPET = """

Media and visuals:
- Display provided rich items inline with `<!--rich:<id>-->`; use only available IDs and never invent image URLs.
- Research the web with `web_research`. Set `image_query` when the answer is about something the reader would expect to SEE — a product, device, place, building, artwork, organism, vehicle, or screen. Reviews, comparisons, recommendations and "tell me about X" on a concrete thing all qualify; do not wait to be asked for pictures.
- Omit `image_query` for abstract subjects (code, math, policy, definitions, planning, conversation) and whenever you are unsure an image would help. Never add media as decoration.
- Add `image_intent="gallery"` when the user asks to see several instances or to compare things — a team roster, a set of logos, colour or trim options, a lineup. Leave it unset otherwise. Never ask for a number of images: the layout decides the count, and only images verified against the subject survive. A gallery arrives as one grid item with a single marker.
- Write the image subject yourself: a concrete subject plus any disambiguator the context implies (company vs fruit, city vs person), plus a form word when it matters (`photo`, `diagram`, `map`, `chart`). No question words, no verbatim reuse of the user's question, one subject per call.
- Images are verified against the subject before they reach you. An approved image appears in your available rich items; many turns will have none, which is normal. Never claim an image exists that is not listed.
- At most two image items per answer, near the text they support; keep the prose useful without them."""
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_research_binding.py -v`
Expected: PASS.

- [ ] **Step 7: Run the binding and prompt suites**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_client_tool_scope.py tests/test_client_tool_isolation.py tests/test_base_agent_dynamic_handoff.py tests/test_custom_agents_tools.py tests/test_search_agent_time_context.py tests/test_read_tool_result_binding.py tests/test_unified_tool_search.py tests/test_tool_search_scoring.py -v
```

Expected: PASS. A test asserting `tavily_search` is pinned for the search agent must be updated to assert `web_research` is bound instead — the pin removal is the intended change.

- [ ] **Step 8: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/ai/deferred_tool_binding.py app/ai/agents/base_agent.py app/ai/prompts.py
git add app/ai/deferred_tool_binding.py app/ai/agents/base_agent.py app/ai/prompts.py tests/test_web_research_binding.py
git status --short
git commit -m "feat: route research through web_research"
```

---

### Task 7: Delete the metadata relevance path

The old path is not disabled behind a flag; it is removed. Leaving it in place invites a fallback that would restore exactly the failure this work exists to fix.

**Files:**
- Modify: `app/core/rich_image_selection.py:33-64, 121-195, 239-249`
- Modify: `app/ai/tool_execution.py:94-311`
- Modify: `tests/test_rich_image_selection.py`
- Modify: `tests/test_article_image_flow.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `_REMOTE_DISCOVERY_SOURCES` becomes `frozenset({"image_search"})`. `order_tavily_images` and `_description_overlap` are deleted. `_url_dimension_hint`, `_URL_WIDTH_PATTERNS`, `_URL_HEIGHT_PATTERNS`, and `_URL_SIZE_PATTERN` are deleted; `_payload_dimensions` reads only the payload. `build_image_candidates_from_tool_result` returns `[]` for `tool_name == "tavily_search"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_rich_image_selection.py`:

```python
def test_a_web_search_image_candidate_is_never_selected():
    candidate = {
        "id": "image:tool:call-1:0",
        "type": "image",
        "source": "web_search",
        "payload": {
            "url": "https://cdn.example/photo.jpg",
            "mime_type": "image/jpeg",
            "source_url": "https://publisher.example/article",
            "width": 995,
            "height": 565,
        },
        "provenance": {"query": "t1 roster", "source_title": "T1 roster 2026"},
    }

    assert select_rich_item_candidates([candidate], policy=_policy()) == []


def test_a_resize_parameter_no_longer_fabricates_an_aspect_ratio():
    candidate = {
        "id": "image:tool:call-1:0",
        "type": "image",
        "source": "image_search",
        "payload": {
            "url": (
                "https://www.sheepesports.com/_next/image?url=https%3A%2F%2Fcdn.sanity.io"
                "%2Fimages%2Fproduction%2F674b8ca2-995x565.webp&w=3840&q=75"
            ),
            "mime_type": "image/webp",
            "width": 995,
            "height": 565,
        },
        "provenance": {},
    }

    assert len(select_rich_item_candidates([candidate], policy=_policy())) == 1


def test_a_resize_url_with_unknown_dimensions_is_still_selected():
    """The actual production failure: no payload dimensions at all.

    Corrected 2026-08-06. The test above supplies explicit width/height, and the
    old code read `_positive_dimension(payload.get("width")) or hinted_width` —
    a truthy payload width short-circuited the URL fallback, so that test passed
    identically with and without the bug. The incident had NO payload dimensions;
    they were inferred solely from `&w=3840` beside the real height parsed from
    the same URL, producing a fake 6.8 ratio that failed the bounds check.
    """
    candidate = {
        "id": "image:tool:call-1:0",
        "type": "image",
        "source": "image_search",
        "payload": {
            "url": (
                "https://www.sheepesports.com/_next/image?url=https%3A%2F%2Fcdn.sanity.io"
                "%2Fimages%2Fproduction%2F674b8ca2-995x565.webp&w=3840&q=75"
            ),
            "mime_type": "image/webp",
        },
        "provenance": {},
    }

    assert len(select_rich_item_candidates([candidate], policy=_policy())) == 1


def test_tavily_results_produce_no_image_candidates():
    from app.ai.tool_execution import build_image_candidates_from_tool_result

    payload = json.dumps(
        {
            "images": [
                {"url": "https://cdn.example/a.jpg", "description": "Moi", "provider": "tavily"}
            ],
            "results": [],
        }
    )

    assert (
        build_image_candidates_from_tool_result(
            payload, tool_call_id="call-1", tool_name="tavily_search"
        )
        == []
    )
```

Add a `_policy()` helper to the file if one does not already exist:

```python
def _policy() -> ImageSelectionPolicy:
    return ImageSelectionPolicy(
        max_items=2,
        min_width_px=320,
        min_height_px=180,
        min_aspect_ratio=0.2,
        max_aspect_ratio=5.0,
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rich_image_selection.py -v`
Expected: FAIL — the `web_search` candidate is still selected, the resize parameter still produces a 6.80 ratio and rejects a valid image, and Tavily still yields candidates.

- [ ] **Step 3: Remove the remote metadata path from the selector**

In `app/core/rich_image_selection.py`:

- Change `_REMOTE_DISCOVERY_SOURCES` to `frozenset({"image_search"})`.
- Delete `_URL_WIDTH_PATTERNS`, `_URL_HEIGHT_PATTERNS`, `_URL_SIZE_PATTERN`, `_url_dimension_hint`, `_RELEVANCE_FIELD_MAX_CHARS`, `_RELEVANCE_FIELD_MAX_TOKENS`, `_normalized_tokens`, and `_description_overlap`.
- Replace `_payload_dimensions` with:

```python
def _payload_dimensions(payload: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """Dimensions come from decoded bytes only.

    A URL's resize parameter is not an intrinsic dimension: reading ``&w=3840``
    beside a real height of 565 fabricated a 6.8 aspect ratio and rejected the
    one relevant image in the T1 trace.
    """
    return (
        _positive_dimension(payload.get("width")),
        _positive_dimension(payload.get("height")),
    )
```

- In `_intent_rank`, delete every `web_search` branch so the function reads:

```python
def _intent_rank(candidate: Mapping[str, Any]) -> int | None:
    source = str(candidate.get("source") or "")
    if source in _DIRECT_SOURCES:
        return 0
    if source == "image_search":
        return 1
    if source == "web_search":
        return None
    return 1
```

- In `_rank_key`, delete the `-_description_overlap(candidate)` element from the returned tuple.

- [ ] **Step 4: Remove Tavily candidate construction**

In `app/ai/tool_execution.py`, delete `order_tavily_images` and, in `build_image_candidates_from_tool_result`, replace the Tavily pre-filter block (the `if metric_provider == "tavily":` branch) with an early return placed immediately after `metric_provider` is computed:

```python
    if metric_provider == "tavily":
        # Page-scraped images cannot establish relevance from page metadata, and
        # the search tool no longer returns them. Nothing to build.
        return []
```

Leave `_TYPED_WEB_IMAGE_TOOLS` unchanged: keeping `tavily_search` in it prevents the legacy extraction path from resurrecting page images.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rich_image_selection.py -v`
Expected: PASS.

- [ ] **Step 6: Update the tests that asserted the deleted behavior**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rich_image_selection.py tests/test_rich_image_selection_runtime.py tests/test_article_image_flow.py tests/test_rich_response_sources.py tests/test_rich_response_prompt_inventory.py tests/test_tool_execution_rendering.py -v`

Every failure will be a test asserting one of: Tavily-sourced candidates existing, `web_search` ordering, source-title token overlap, or URL-derived dimensions. Delete those tests — they encode the behavior this task removes. Keep every test covering direct sources (`rag_document`, `tool_image`, `generated_image`), `image_search`, junk-URL rejection, deduplication, and caps. Re-run until green.

- [ ] **Step 7: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/core/rich_image_selection.py app/ai/tool_execution.py
git add app/core/rich_image_selection.py app/ai/tool_execution.py tests/
git status --short
git commit -m "refactor: remove metadata-based remote image selection"
```

Check `git status --short` before committing: `git add tests/` stages the whole directory, so confirm no unrelated test file from another session is included.

---

### Task 8: Verification metrics

Content-free counters for the new path, so a silent collapse to text-only is visible.

**Files:**
- Modify: `app/observability/rich_images.py:9-35, 96-145`
- Modify: `app/ai/image_verification_flow.py`
- Create: `tests/test_visual_verification_metrics.py`

**Interfaces:**
- Consumes: `rich_image_metrics` (existing singleton).
- Produces: `record_verification(*, stage: str, count: int) -> None` where `stage` is bounded to `{"discovered", "fetched", "submitted", "approved"}`, and `record_verification_outcome(*, outcome: str, duration_seconds: float) -> None` where `outcome` is bounded to `{"approved", "no_match", "timeout", "malformed", "transport", "unavailable"}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_visual_verification_metrics.py`:

```python
from __future__ import annotations

from app.observability.rich_images import RichImageMetrics


def test_verification_stages_are_counted():
    metrics = RichImageMetrics()

    metrics.record_verification(stage="discovered", count=6)
    metrics.record_verification(stage="approved", count=1)
    rendered = metrics.render().decode("utf-8")

    assert 'stage="discovered"' in rendered
    assert 'stage="approved"' in rendered


def test_an_unknown_stage_is_bucketed_not_recorded_verbatim():
    metrics = RichImageMetrics()

    metrics.record_verification(stage="league of legends t1 roster", count=1)
    rendered = metrics.render().decode("utf-8")

    assert "league" not in rendered
    assert 'stage="other"' in rendered


def test_an_unknown_outcome_is_bucketed():
    metrics = RichImageMetrics()

    metrics.record_verification_outcome(outcome="portrait of Moi", duration_seconds=0.2)
    rendered = metrics.render().decode("utf-8")

    assert "Moi" not in rendered
    assert 'outcome="other"' in rendered
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_visual_verification_metrics.py -v`
Expected: FAIL with `AttributeError: 'RichImageMetrics' object has no attribute 'record_verification'`.

- [ ] **Step 3: Add the collectors**

In `app/observability/rich_images.py`, add the bounded sets beside the existing ones:

```python
_VERIFICATION_STAGES = {"discovered", "fetched", "submitted", "approved"}
_VERIFICATION_OUTCOMES = {
    "approved",
    "no_match",
    "timeout",
    "malformed",
    "transport",
    "unavailable",
}
```

In `__init__`, after `self.registrations`:

```python
        self.verification_stages = Counter(
            "rich_image_verification_stage_total",
            "Candidates reaching each stage of visual verification.",
            ("stage",),
            registry=self.registry,
        )
        self.verification_outcomes = Counter(
            "rich_image_verification_outcome_total",
            "Terminal outcome of the visual verification path.",
            ("outcome",),
            registry=self.registry,
        )
        self.verification_duration = Histogram(
            "rich_image_verification_duration_seconds",
            "Duration of the visual verification path.",
            ("outcome",),
            registry=self.registry,
        )
```

And the recorders beside `record_fetch`:

```python
    def record_verification(self, *, stage: str, count: int) -> None:
        if count > 0:
            self.verification_stages.labels(
                stage=_bounded(stage, _VERIFICATION_STAGES)
            ).inc(int(count))

    def record_verification_outcome(self, *, outcome: str, duration_seconds: float) -> None:
        label = _bounded(outcome, _VERIFICATION_OUTCOMES)
        self.verification_outcomes.labels(outcome=label).inc()
        self.verification_duration.labels(outcome=label).observe(
            max(0.0, float(duration_seconds))
        )
```

- [ ] **Step 4: Record from the flow**

In `app/ai/image_verification_flow.py`, add `import time` and `from contextlib import suppress`, plus `from ..observability.rich_images import rich_image_metrics`. Record each stage as it completes and one terminal outcome, wrapping every metrics call in `with suppress(Exception):` so telemetry can never drop a candidate decision:

```python
    started = time.perf_counter()

    def _outcome(label: str) -> None:
        with suppress(Exception):
            rich_image_metrics.record_verification_outcome(
                outcome=label, duration_seconds=time.perf_counter() - started
            )
```

Call `rich_image_metrics.record_verification(stage="discovered", count=len(candidates))` after discovery, `stage="fetched"` with the non-`None` thumbnail count, `stage="submitted"` with `len(submitted)`, and `stage="approved"` with `len(approved)`. Call `_outcome("unavailable")` on the missing-dependency early return, `_outcome("transport")` when `submitted` is empty, `_outcome("malformed")` when `result is None`, `_outcome("no_match")` when `approved` is empty, and `_outcome("approved")` otherwise.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_visual_verification_metrics.py tests/test_web_research_tool.py -v`
Expected: PASS.

- [ ] **Step 6: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/observability/rich_images.py app/ai/image_verification_flow.py
git add app/observability/rich_images.py app/ai/image_verification_flow.py tests/test_visual_verification_metrics.py
git status --short
git commit -m "feat: add content-free verification metrics"
```

---

### Task 9: Trace regression and privacy gate

The acceptance test for the whole phase: the trace's portrait is rejected, the team photo is approved, and nothing about a rejected candidate is reachable by the model or by persistence.

**Files:**
- Create: `tests/test_vision_verified_injection_regression.py`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing.

- [ ] **Step 1: Write the regression test**

Create `tests/test_vision_verified_injection_regression.py`:

```python
"""Trace-shaped regression for the T1 turn in example_run.txt.

Original failure: the model was offered an author portrait labelled "Moi" and a
wiki asset labelled "research", while the one image that depicted the team was
rejected by a URL resize parameter. Relevance was inferred from the page title.
"""

from __future__ import annotations

import json
import re

import pytest

from app.ai.research_budget import reset_research_budget
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.verified_image_sink import verified_image_sink
from app.ai.visual_verifier import VisualCandidateDecision, VisualVerificationResult
from app.ai.web_research_tool import create_web_research_tool
from app.services.web_image_service import FetchedWebImage

CONVERSATION_ID = "22222222-2222-2222-2222-222222222222"
PORTRAIT_URL = "https://cdn.example/moi-1200x1600.jpg"
TEAM_URL = "https://cdn.example/t1-team-995x565.webp?w=3840&q=75"

TAVILY_PAYLOAD = json.dumps(
    {
        "results": [
            {
                "index": 1,
                "title": "LoL: T1 completed 2026 LCK roster",
                "url": "https://sheepesports.example/t1",
                "content": "T1 finalized its 2026 LCK roster.",
                "score": 0.887,
            }
        ],
        "total_results": 1,
        "answer": "T1 is a South Korean esports organization.",
        "provider": "tavily",
        "operation": "search",
        "query": "T1 League of Legends Esports team news roster 2026",
    }
)

BRAVE_PAYLOAD = json.dumps(
    {
        "query": "T1 League of Legends team photo",
        "provider": "brave_image_search",
        "images": [
            {
                "url": PORTRAIT_URL,
                "provider": "brave_image_search",
                "mime_type": "image/jpeg",
                "title": "Moi",
                "description": "Moi",
                "width": 1200,
                "height": 1600,
                "source_url": "https://sheepesports.example/t1",
            },
            {
                "url": TEAM_URL,
                "provider": "brave_image_search",
                "mime_type": "image/webp",
                "title": "T1 2026 roster",
                "description": "T1 2026 roster",
                "width": 995,
                "height": 565,
                "source_url": "https://sheepesports.example/t1",
            },
        ],
        "total_results": 2,
    }
)


class _Tool:
    def __init__(self, payload: str):
        self.payload = payload

    async def ainvoke(self, args: dict) -> str:
        return self.payload


class _Service:
    async def fetch_url(self, url: str, *, provider: str = "other") -> FetchedWebImage:
        if url == PORTRAIT_URL:
            return FetchedWebImage(
                content=b"portrait", media_type="image/jpeg", width=1200, height=1600
            )
        return FetchedWebImage(
            content=b"team", media_type="image/webp", width=995, height=565
        )


class _Verifier:
    """Rejects the portrait on visible content, approves the team photo."""

    async def ainvoke(self, messages):
        text = messages[0].content[0]["text"]
        decisions = []
        for candidate_id, line in re.findall(r"^- (c\d+): (.*)$", text, flags=re.MULTILINE):
            portrait = "Moi" in line
            decisions.append(
                VisualCandidateDecision(
                    candidate_id=candidate_id,
                    depicts_requested_subject=not portrait,
                    materially_supports_answer=not portrait,
                    confidence=0.93 if not portrait else 0.91,
                    content_kind="portrait" if portrait else "photo",
                )
            )
        return VisualVerificationResult(decisions=decisions)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)
    monkeypatch.setattr(
        "app.ai.web_research_tool.settings.vision_image_verification_enabled",
        True,
        raising=False,
    )
    yield
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)


@pytest.mark.asyncio
async def test_portrait_rejected_team_photo_approved():
    tool = create_web_research_tool(
        tavily_tool=_Tool(TAVILY_PAYLOAD),
        brave_tool=_Tool(BRAVE_PAYLOAD),
        web_image_service=_Service(),
        verifier_model=_Verifier(),
    )

    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ), verified_image_sink() as sink:
        raw = await tool.ainvoke(
            {
                "query": "T1 League of Legends Esports team news roster 2026",
                "image_query": "T1 League of Legends team photo",
            }
        )

    serialized_sink = json.dumps(sink)
    assert len(sink) == 1
    assert sink[0]["payload"]["url"] == TEAM_URL
    assert "Moi" not in serialized_sink
    assert PORTRAIT_URL not in serialized_sink
    assert PORTRAIT_URL not in raw
    assert "Moi" not in raw


@pytest.mark.asyncio
async def test_no_verifier_field_reaches_the_public_candidate():
    tool = create_web_research_tool(
        tavily_tool=_Tool(TAVILY_PAYLOAD),
        brave_tool=_Tool(BRAVE_PAYLOAD),
        web_image_service=_Service(),
        verifier_model=_Verifier(),
    )

    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ), verified_image_sink() as sink:
        await tool.ainvoke(
            {"query": "T1 roster 2026", "image_query": "T1 League of Legends team photo"}
        )

    serialized = json.dumps(sink)
    for forbidden in (
        "confidence",
        "content_kind",
        "depicts_requested_subject",
        "materially_supports_answer",
        "candidate_id",
    ):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_approved_candidate_carries_decoded_dimensions():
    tool = create_web_research_tool(
        tavily_tool=_Tool(TAVILY_PAYLOAD),
        brave_tool=_Tool(BRAVE_PAYLOAD),
        web_image_service=_Service(),
        verifier_model=_Verifier(),
    )

    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ), verified_image_sink() as sink:
        await tool.ainvoke(
            {"query": "T1 roster 2026", "image_query": "T1 League of Legends team photo"}
        )

    assert sink[0]["payload"]["width"] == 995
    assert sink[0]["payload"]["height"] == 565
    assert sink[0]["payload"]["mime_type"] == "image/webp"
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vision_verified_injection_regression.py -v`
Expected: PASS.

- [ ] **Step 3: Run the whole affected surface**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_research_budget.py tests/test_thumbnail_batch.py tests/test_visual_verifier.py tests/test_verified_image_sink.py tests/test_web_research_tool.py tests/test_web_research_binding.py tests/test_visual_verification_metrics.py tests/test_vision_verified_injection_regression.py tests/test_rich_image_selection.py tests/test_rich_image_selection_runtime.py tests/test_rich_response_sources.py tests/test_rich_response_prompt_inventory.py tests/test_rich_placement.py tests/test_article_image_flow.py tests/test_message_service_web_image_externalization.py tests/test_web_image_service.py tests/test_tool_execution_rendering.py -v
```

Expected: PASS. `tests/test_live_server_document_upload.py` fails for environmental reasons unrelated to this work; note that in the commit body if you run the full suite.

- [ ] **Step 4: Commit**

```bash
git add tests/test_vision_verified_injection_regression.py
git status --short
git commit -m "test: add vision-verified injection regression"
```

---

### Task 10: Cache verified bytes (deferrable)

Today a placed image is fetched again at render time, so an image can pass every check, be placed, and then die at render. Verification already holds validated bytes; storing them makes each approved image one fetch and makes a rendered figure one whose bytes already decoded.

Ship Tasks 1-9 first. This task is independently valuable and independently revertible.

**Files:**
- Modify: `app/models/web_image_reference.py`
- Create: `app/alembic/versions/<revision>_add_web_image_cached_bytes.py`
- Modify: `app/services/web_image_service.py` (serve cached bytes in `fetch`)
- Modify: `app/services/message_service.py:2465-2512` (pass bytes at registration)
- Create: `tests/test_web_image_byte_cache.py`

**Interfaces:**
- Consumes: `FetchedWebImage` from Task 2.
- Produces: `WebImageService.register(..., cached: FetchedWebImage | None = None)`. `WebImageService.fetch(record)` returns the cached bytes when `record.content` is present, without any network call.

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_image_byte_cache.py`:

```python
from __future__ import annotations

import pytest

from app.services.web_image_service import FetchedWebImage, WebImageService


class _ExplodingResolver:
    def __call__(self, hostname):  # pragma: no cover - must never run
        raise AssertionError("a cached image must not touch the network")


@pytest.mark.asyncio
async def test_cached_bytes_are_served_without_a_fetch():
    service = WebImageService(
        repository=object(),
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_redirects=1,
        max_bytes=4096,
        max_pixels=10_000_000,
        resolver=_ExplodingResolver(),
    )
    record = {
        "upstream_url": "https://cdn.example/a.jpg",
        "provider": "brave",
        "content": b"cached-bytes",
        "expected_mime": "image/jpeg",
        "cached_width": 995,
        "cached_height": 565,
    }

    fetched = await service.fetch(record)

    assert fetched == FetchedWebImage(
        content=b"cached-bytes", media_type="image/jpeg", width=995, height=565
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_image_byte_cache.py -v`
Expected: FAIL — `fetch` goes to `_fetch_redirects` and the resolver raises.

- [ ] **Step 3: Add the columns**

In `app/models/web_image_reference.py`, add after `provider`:

```python
    content = Column(LargeBinary, nullable=True)
    cached_width = Column(Integer, nullable=True)
    cached_height = Column(Integer, nullable=True)
```

Import `Integer` and `LargeBinary` from `sqlalchemy`.

- [ ] **Step 4: Write the migration**

Create the revision with `.venv/Scripts/python.exe -m alembic revision -m "add web image cached bytes"`, then fill it in:

```python
def upgrade() -> None:
    op.add_column("web_image_references", sa.Column("content", sa.LargeBinary(), nullable=True))
    op.add_column("web_image_references", sa.Column("cached_width", sa.Integer(), nullable=True))
    op.add_column("web_image_references", sa.Column("cached_height", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("web_image_references", "cached_height")
    op.drop_column("web_image_references", "cached_width")
    op.drop_column("web_image_references", "content")
```

Set `down_revision` to the current head from `.venv/Scripts/python.exe -m alembic heads`. Note in the commit body whether the live database is up to date — migration `d7e8f9a0b1c2` (chat images) was pending as of 2026-08-04, so confirm the ordering before applying anything.

- [ ] **Step 5: Serve the cache**

In `app/services/web_image_service.py`, at the top of `fetch`, before the timing block:

```python
        cached = self._cached_image(record)
        if cached is not None:
            return cached
```

and add:

```python
    def _cached_image(self, record: Any) -> FetchedWebImage | None:
        """Return the verified bytes stored at selection time, if any."""

        content = record.get("content") if isinstance(record, dict) else getattr(record, "content", None)
        if not content:
            return None
        media_type = self._record_value(record, "expected_mime")
        width = record.get("cached_width") if isinstance(record, dict) else getattr(record, "cached_width", None)
        height = record.get("cached_height") if isinstance(record, dict) else getattr(record, "cached_height", None)
        if media_type not in ALLOWED_WEB_IMAGE_MIME_TYPES or not width or not height:
            return None
        return FetchedWebImage(
            content=bytes(content), media_type=media_type, width=int(width), height=int(height)
        )
```

Extend `register` with a `cached: FetchedWebImage | None = None` keyword and persist `content`, `cached_width`, `cached_height` when it is provided.

- [ ] **Step 6: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_web_image_byte_cache.py tests/test_web_image_service.py tests/test_message_service_web_image_externalization.py tests/test_image_stream_http_contract.py -v`
Expected: PASS. A registration that supplies no bytes must still work — that is the fallback path when the cache write fails.

- [ ] **Step 7: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/models/web_image_reference.py app/services/web_image_service.py app/services/message_service.py
git add app/models/web_image_reference.py app/services/web_image_service.py app/services/message_service.py app/alembic/versions tests/test_web_image_byte_cache.py
git status --short
git commit -m "feat: serve verified image bytes from cache"
```

---

## Rollout

1. Merge Phase 1 first. Tasks 1-9 here leave `vision_image_verification_enabled=False`, so remote web images stay off and answers are text-only. That is the intended fail-closed state.
2. Enable `vision_image_verification_enabled` in one environment and watch `rich_image_verification_outcome_total`. A healthy visual turn shows `discovered` ≥ `submitted` ≥ `approved`, and `approved` at 0 or 1 most of the time. Sustained `transport` means the thumbnail deadline is too tight; sustained `malformed` means the verifier model is not honoring the structured output.
3. Rollback is the flag alone. It must not restore Tavily page images, the `images` key, or URL-derived dimension inference — Task 7 deletes all three, and reverting that deletion is not a rollback path.

## Self-review notes

- Spec coverage: `web_research` (Task 5), escape hatches (Task 6), turn-local budget (Task 1), candidate acquisition and safety (Task 2), verifier input/output/admission (Task 3), inventory isolation (Task 4), removal list (Task 7), latency and failure policy (Tasks 2, 3, 5), observability (Task 8), acceptance tests (Tasks 5, 9), byte cache (Task 10).
- The spec's "bounded Tavily result titles when already available" is implemented as `result_titles=[]` in Task 5: the image path starts concurrently with the search and must never wait for it, so titles are usually unavailable at dispatch. The verifier receives the factual query and the user request, which the spec names as the required subject context. Passing real titles would require the image path to await Tavily, which the spec forbids.
- `SubmittedCandidate.candidate_id` uses positional `c0`/`c1` ids scoped to one verifier call, never leaving the process. The spec's "stable, unguessable temporary ID" requirement is satisfied by scope rather than entropy: the ids are never persisted, logged, or shown to the answering model, and a hallucinated id fails closed in `admit_candidates`.
- `image_verification_deadline_seconds` bounds both the thumbnail batch and the verifier call, and Task 3 Step 6 asserts the provider timeouts fit inside it.
- **Amended 2026-08-05, after review of the fixed two-image cap.** The original draft capped every answer at two individual images and, in Task 5, flattened Brave's `image_group` into singles before verification. Two defects followed. First, gallery answers would have regressed against today's behaviour: the existing pipeline already supports two items where each may be a three-cell group (six images), and flatten-then-cap-at-two would have reduced that to two. Second, `build_image_candidates_from_tool_result` collapses a multi-result Brave payload into one group capped at `rich_image_group_max_items` (3), so flattening it could never yield more than three candidates — `image_verification_max_candidates` (6) was dead. Both are fixed by discovering and verifying individuals (`group_images=False`) and grouping only the survivors. The count now follows a declared `image_intent` rather than a fixed constant, and the model never states a number, because an unverifiable model judgement about images is the failure class this design exists to remove. Raising the gallery ceiling is nearly free: the verifier already receives the whole batch in one call.
