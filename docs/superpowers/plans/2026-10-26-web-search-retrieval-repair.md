# Web Search Retrieval Repair Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Allow targeted follow-up retrieval after a full first search and correct the public result-limit and search-topic contracts.

**Architecture:** Retain the current session, adapters, source registry and grounding. Separate per-search ceilings from the stable citation store's total capacity; propagate result count and topic through existing typed boundaries.

**Tech Stack:** Python, Pydantic, LangChain structured tools, Tavily, pytest.

## Global Constraints

- Per-search ceilings remain five results for quick mode and eight for agentic mode.
- Turn-wide text-source capacity is 64; source IDs must never be evicted or reassigned.
- Keep image-source capacity and page-open headroom independent.
- Preserve exact/near-duplicate protection, user/device isolation, continuation and citation contracts.
- No new provider, frontend component, model call, or dependency.
- Existing untracked rich-answer review belongs to the user and is outside this change.

### Task 1: Repair search capacity and result counts

**Files:** `app/ai/web_research/{contracts,policy,service}.py`, `app/ai/web_tools.py`, `tests/test_web_research_service.py`, `tests/test_web_research_tool_session.py`.

**Interfaces:** `ResearchRequest.max_results: int = 5`; `ResearchLimits.max_total_text_sources: int`; `SourceRegistry.admit(candidates, *, max_new_sources: int | None = None)` bounds new admissions without dropping duplicates' later valid neighbors.

- [x] Add a public-tool regression reproducing five preview sources followed by completed-score evidence; require the second result to contain a new stable source ID.
- [x] Add parameterized public-tool tests with `max_results=1` and `500`; assert returned source counts of one and five.
- [x] Add actual-capacity tests and duplicate/invalid URL admission tests; run them red.
- [x] Forward the request count, clamp to the mode ceiling and remaining capacity, and admit valid new text records against a distinct 64-slot store. At full capacity skip the text provider while allowing images.
- [x] Run capacity, registry and public-tool tests green. Update earlier tests that asserted exhaustion after five results to exercise the actual 64-source boundary.

### Task 2: Separate topic from freshness

**Files:** `app/ai/web_query_contract.py`, `app/ai/web_research/{contracts,service}.py`, `app/ai/web_tools.py`, `tests/test_web_query_contract.py`, `tests/test_web_research_tool_session.py`.

**Interfaces:** Public, internal and normalized requests gain `topic: Literal['general', 'news'] = 'general'`; `tavily_search_args` forwards the topic independently of date bounds; topic is included in the search deduplication scope.

- [x] Add red regressions for current general searches, explicit news searches, invalid topic validation, and a topic switch before/after restored research accounting.
- [x] Forward topic through schemas, normalization, provider arguments and scope. Leave freshness validation intact.
- [x] Run query, provider, public-tool and budget tests green.

### Task 3: Align guidance and verify the complete repair

**Files:** `app/ai/prompts.py`, `app/ai/web_tools.py`, `app/ai/web_query_contract.py`, this plan and the approved design.

- [x] Replace rigid one-search guidance with an evidence-sufficiency policy: open promising incomplete pages, reformulate off-target searches, distinguish results from previews, and stop on adequate support or a real limit.
- [x] Explain general/news topic selection, runtime event years, true capacity exhaustion and the actual purpose of objective. Preserve citation instructions.
- [x] Run the affected web research, provider, registry, query, budget, continuation, grounding, source publication and streaming suites with `.venv/Scripts/python.exe -m pytest ... -q`.
- [x] Run `.venv/Scripts/python.exe -m ruff check` on changed Python files and `git diff --check`.
- [x] Obtain independent code review, resolve substantive findings, and report verified results and remaining live-answer-quality limitations.

## Baseline evidence

The original 60 focused tests and 174 additional contract tests passed. Local reproduction confirmed that a second targeted search never reached the provider; a public request for one result reached the provider as five. A live identical-query comparison found mostly previews/rankings in news mode and tournament results/schedule pages in general mode. These source titles do not alone establish score correctness.


## Final verification

Implemented on 2026-10-26 in the existing checkout. No commit, push or deployment was performed.

- All 556 selected tests passed across 49 files (14.23 seconds). This covers the public-tool repair regressions, quick/agentic count clamping, real citation resolution, JSON and legacy-scope restoration, carried capacity, disabled search, web research/provider/grounding/streaming behavior, and specialist/continuation integration.
- Ruff checks passed for all changed Python files and the web research package. `git diff --check` passed.
- Independent review identified continuation compatibility and disabled-mode edge cases; these were reproduced with failing tests, repaired and re-reviewed. No substantive findings remain.
- Live replay through the repaired public search tool and real Tavily adapter: the original two queries reached the provider twice; the second admitted five new records (S6-S10), total ten, with no failures. The original trace's second query never reached the provider.
- A separate public `web_open` canary read two discovered pages with a concrete question and returned evidence with no failures. Search and extract diagnostics remain in ignored `.codex_logs/` files.

Review refinements: canonical search scopes now use JSON primitives and migrate legacy category semantics when restoring persisted research accounting. The specialist imports carried sources through the session so Continue does not reset total source accounting. Because carried records lack admission-category metadata, they are conservatively charged as text; this can reduce available capacity after Continue but preserves prior citations and bounds growth. None mode retains zero text-search capacity.

The canaries verify live retrieval mechanics, not the correctness or completeness of an entire model-generated answer. Some sources still contain noisy or irrelevant text. Broader real-question answer-quality evaluation remains necessary before declaring production readiness or ChatGPT parity. Restart the backend to load the repair and examine fresh LangSmith traces for the representative Vietnamese question.
