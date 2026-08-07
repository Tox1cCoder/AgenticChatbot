# Automatic Visual Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `web_research` attempt verified images by default, remove the obsolete shared image deadline, classify failures accurately, and delete redundant image-path code and commentary.

**Architecture:** `web_research` starts Tavily and the image path concurrently unless `skip_images=true` or an existing server gate closes the path. The image path uses independent Brave, thumbnail-batch, and verifier timeouts; it owns terminal outcome classification and logging, while approved candidates continue through the existing sink, inventory, protected registration, and placement pipeline.

**Tech Stack:** Python 3.11+, asyncio, LangChain structured tools, Pydantic settings/models, pytest, Prometheus metrics.

## Global Constraints

- Preserve Tavily as a text-only research source and Brave as the only remote answer-image source.
- Preserve rich-response capability, rollout-flag, and turn-budget gates.
- Preserve `WebImageService` SSRF, redirect, MIME, size, decode, dimension, aspect-ratio, and deduplication checks.
- Preserve protected `/web-images/{id}` delivery and never expose the upstream asset URL.
- Image failures remain optional and must not turn successful factual research into a tool error.
- Perform no unrelated repository-wide refactor.
- Keep comments only for non-obvious invariants, security boundaries, public contracts, or concurrency requirements.

---

### Task 1: Make visual enrichment default-on

**Files:**
- Modify: `app/ai/web_research_tool.py:31-179`
- Modify: `app/ai/prompts.py:28-37`
- Modify: `tests/test_web_research_tool.py:295-325`
- Modify: `tests/test_prompts_media_capability.py`

**Interfaces:**
- Consumes: `create_web_research_tool(...) -> StructuredTool`, `ResearchBudget.may_image_search() -> bool`, and `ToolContext.rich_response_capable`.
- Produces: `WebResearchInput.skip_images: bool = False`; `_research(..., skip_images: bool = False)`; an image search query equal to the trimmed `image_query` when present and otherwise the trimmed factual `query`.

- [ ] **Step 1: Replace the opt-in regression with a failing default-on regression**

```python
@pytest.mark.asyncio
async def test_missing_image_query_uses_the_factual_query_for_images():
    tavily = _FakeTool("tavily_search", TAVILY_PAYLOAD)
    brave = _FakeTool("brave_image_search", BRAVE_PAYLOAD)

    _, sink = await _run(
        _tool(tavily, brave, _FakeImageService(), _ApproveOnlyTeamPhoto()),
        query="cho t thông tin về T1",
    )

    assert brave.calls == [{"query": "cho t thông tin về T1"}]
    assert sink
```

- [ ] **Step 2: Run the default-on regression and confirm RED**

Run: `\.venv\Scripts\python.exe -m pytest -q tests/test_web_research_tool.py::test_missing_image_query_uses_the_factual_query_for_images`

Expected: FAIL because `brave.calls` is empty under the current `wants_image = bool(image_query)` behavior.

- [ ] **Step 3: Add a failing explicit-opt-out regression**

```python
@pytest.mark.asyncio
async def test_skip_images_prevents_brave_and_verifier_calls():
    brave = _FakeTool("brave_image_search", BRAVE_PAYLOAD)
    verifier = _ApproveOnlyTeamPhoto()

    payload, sink = await _run(
        _tool(_FakeTool("tavily_search", TAVILY_PAYLOAD), brave, _FakeImageService(), verifier),
        query="explain big-O notation",
        skip_images=True,
    )

    assert brave.calls == []
    assert verifier.calls == 0
    assert sink == []
    assert payload["results"]
```

- [ ] **Step 4: Run the opt-out regression and confirm RED**

Run: `\.venv\Scripts\python.exe -m pytest -q tests/test_web_research_tool.py::test_skip_images_prevents_brave_and_verifier_calls`

Expected: FAIL during tool input validation because `skip_images` is not in `WebResearchInput`.

- [ ] **Step 5: Make the complete-pipeline regression fail for the same omission**

In `tests/test_verified_image_reaches_the_model.py`, change `_approved_candidates()` to invoke:

```python
await tool.ainvoke({"query": "cho t thông tin về T1"})
```

Run: `\.venv\Scripts\python.exe -m pytest -q tests/test_verified_image_reaches_the_model.py::test_an_approved_image_becomes_a_marker_the_model_can_copy`

Expected: FAIL because the omitted `image_query` prevents Brave discovery, leaving the verified-image sink empty.

- [ ] **Step 6: Implement the minimal default-on tool contract**

In `WebResearchInput`, add:

```python
skip_images: bool = Field(
    default=False,
    description="Set true only when an image cannot help the answer.",
)
```

In `_research`, accept `skip_images: bool = False` and replace `wants_image` with:

```python
visual_query = str(image_query or query).strip()
wants_image = bool(visual_query) and not skip_images
```

Pass `visual_query` to `_discover_and_verify`. Keep `_image_path_open` as the server-side capability/flag/budget gate and keep budget reuse behavior unchanged.

- [ ] **Step 7: Replace prompt taxonomy with a failing contract test**

```python
def test_media_guidance_describes_automatic_visual_enrichment_without_taxonomy():
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()

    assert "automatically considers" in snippet
    assert "skip_images" in snippet
    assert "product, device" not in snippet
    assert "code, math" not in snippet
```

Run: `\.venv\Scripts\python.exe -m pytest -q tests/test_prompts_media_capability.py::test_media_guidance_describes_automatic_visual_enrichment_without_taxonomy`

Expected: FAIL because the current prompt routes by named topic categories and does not mention `skip_images`.

- [ ] **Step 8: Simplify the model-facing descriptions**

Replace `_DESCRIPTION` and the media prompt routing prose with concise rules:

```text
Web research automatically considers a verified image. Set image_query only to
make the visual subject more precise. Set skip_images=true only when a visual
cannot support the answer. Use gallery only when the user asks to see or compare
several instances.
```

Keep the existing inventory-ID, marker, no-invention, and text-independence rules.

- [ ] **Step 9: Run focused tests and commit**

Run: `\.venv\Scripts\python.exe -m pytest -q tests/test_web_research_tool.py tests/test_prompts_media_capability.py tests/test_web_research_binding.py tests/test_verified_image_reaches_the_model.py`

Expected: PASS.

Commit:

```powershell
git add app/ai/web_research_tool.py app/ai/prompts.py tests/test_web_research_tool.py tests/test_prompts_media_capability.py tests/test_verified_image_reaches_the_model.py
git commit -m "feat: make verified images default in web research"
```

---

### Task 2: Replace the shared deadline with stage timeouts

**Files:**
- Modify: `app/core/config.py:304-309,1193-1237,2038-2058`
- Modify: `app/ai/web_research_tool.py:201-260`
- Modify: `app/ai/image_verification_flow.py:1-202`
- Modify: `app/ai/visual_verifier.py:248-285,351-390`
- Modify: `.env.example:320-325`
- Modify: `tests/test_visual_verifier.py:131-214,349-388,471-489`
- Modify: `tests/test_web_research_tool.py:458-486`

**Interfaces:**
- Consumes: `fetch_thumbnails(..., per_item_timeout: float, batch_deadline: float)` and the existing 30-second internal-tool execution policy.
- Produces: `Settings.image_verification_timeout_seconds: float = 10.0`; `verify_candidates(...)` propagates `TimeoutError` and provider exceptions, returns `None` only for malformed structured output, and raises `VisualVerifierUnavailable` when no verifier model can be built.

- [ ] **Step 1: Write failing configuration-contract tests**

```python
def test_verifier_has_a_stage_timeout_without_an_image_path_deadline():
    fields = Settings.model_fields

    assert "image_verification_deadline_seconds" not in fields
    assert fields["image_verification_timeout_seconds"].default == 10.0
    assert fields["image_verification_thumbnail_timeout_seconds"].default == 2.0
```

Delete the three tests that compare Brave/thumbnail values with the old shared deadline or assert the removed startup warning.

- [ ] **Step 2: Run the configuration test and confirm RED**

Run: `\.venv\Scripts\python.exe -m pytest -q tests/test_visual_verifier.py::test_verifier_has_a_stage_timeout_without_an_image_path_deadline`

Expected: FAIL because the deadline field exists and the verifier-stage field does not.

- [ ] **Step 3: Implement the settings replacement**

Remove `image_verification_deadline_seconds` and `_warn_if_image_verification_budget_is_tight`. Add:

```python
image_verification_timeout_seconds: float = Field(
    default=10.0,
    gt=0,
    description="Timeout for the single visual-verification model call.",
)
```

Shorten the Brave, candidate-count, thinking-level, and thumbnail descriptions to current behavior. Add `IMAGE_VERIFICATION_TIMEOUT_SECONDS=10` beside the Brave image settings in `.env.example`.

- [ ] **Step 4: Write failing verifier-propagation tests**

Change the existing timeout test to:

```python
with pytest.raises(TimeoutError):
    await verify_candidates(
        submitted,
        user_request="u",
        image_query="i",
        factual_query="f",
        result_titles=[],
        model=_SlowModel(),
        timeout=0.01,
    )
```

Add a model-unavailable test that patches `build_verifier_model` to return `None` and expects `VisualVerifierUnavailable`.

- [ ] **Step 5: Run verifier tests and confirm RED**

Run: `\.venv\Scripts\python.exe -m pytest -q tests/test_visual_verifier.py::test_verifier_timeout_propagates tests/test_visual_verifier.py::test_missing_verifier_model_raises_unavailable`

Expected: FAIL because `verify_candidates` currently swallows both failures into `None`.

- [ ] **Step 6: Make verifier failures classifiable**

Add one focused exception:

```python
class VisualVerifierUnavailable(RuntimeError):
    pass
```

In `verify_candidates`, raise it when model resolution returns `None`; remove the broad `except Exception` around `_invoke_verifier`; retain `None` only from `_unwrap_verifier_response` for malformed structured output. Remove the redundant model-construction warning so the orchestration layer owns the terminal warning.

- [ ] **Step 7: Remove the nested overall timeout**

In `web_research_tool._discover_and_verify`, remove `time`, `suppress`, `rich_image_metrics`, the outer `asyncio.timeout`, its timeout warning/metric branch, and its broad exception branch. Resolve dependencies and directly await `discover_and_verify_images`.

In `image_verification_flow`, delete `_remaining_seconds` and `deadline_at`. Call:

```python
thumbnails = await fetch_thumbnails(
    web_image_service,
    urls,
    provider="brave",
    per_item_timeout=float(settings.image_verification_thumbnail_timeout_seconds),
    batch_deadline=float(settings.image_verification_thumbnail_timeout_seconds),
)
```

and pass `float(settings.image_verification_timeout_seconds)` to `verify_candidates`.

- [ ] **Step 8: Run focused timeout tests and commit**

Run: `\.venv\Scripts\python.exe -m pytest -q tests/test_visual_verifier.py tests/test_web_research_tool.py tests/test_thumbnail_batch.py`

Expected: PASS with no deadline-headroom or outer-deadline tests remaining.

Commit:

```powershell
git add .env.example app/core/config.py app/ai/web_research_tool.py app/ai/image_verification_flow.py app/ai/visual_verifier.py tests/test_visual_verifier.py tests/test_web_research_tool.py
git commit -m "refactor: use stage timeouts for image verification"
```

---

### Task 3: Classify image outcomes once and remove redundant logging

**Files:**
- Modify: `app/ai/image_verification_flow.py`
- Modify: `app/ai/web_research_tool.py:166-179,263-290`
- Modify: `app/observability/rich_images.py:36-44`
- Modify: `tests/test_image_verification_flow_metrics.py`
- Modify: `tests/test_visual_verification_metrics.py`

**Interfaces:**
- Consumes: `VisualVerifierUnavailable`, built-in `TimeoutError`, `rich_image_metrics.record_verification_outcome(...)`.
- Produces: the bounded outcomes `approved`, `skipped`, `unavailable`, `search_failure`, `fetch_failure`, `verifier_timeout`, `verifier_failure`, `malformed`, and `no_match`; one warning from `app.ai.image_verification_flow` for operational failures only.

- [ ] **Step 1: Write failing outcome tests**

Extend the flow fixture with a raising Brave tool, a slow verifier, a raising verifier, and a malformed verifier. Assert:

```python
assert _outcome_labels(metrics) == ["search_failure"]
assert _outcome_labels(metrics) == ["fetch_failure"]
assert _outcome_labels(metrics) == ["verifier_timeout"]
assert _outcome_labels(metrics) == ["verifier_failure"]
assert _outcome_labels(metrics) == ["malformed"]
```

Each call must assert `record_verification_outcome.assert_called_once()`.

- [ ] **Step 2: Run outcome tests and confirm RED**

Run: `\.venv\Scripts\python.exe -m pytest -q tests/test_image_verification_flow_metrics.py`

Expected: FAIL because transport and verifier failures currently collapse into `transport` or `malformed`, and Brave failure is classified outside the flow.

- [ ] **Step 3: Implement stage-local classification**

In `discover_and_verify_images`:

- catch Brave invocation errors as `search_failure`;
- classify zero downloaded thumbnails as `fetch_failure`;
- catch `TimeoutError` from `verify_candidates` as `verifier_timeout`;
- catch `VisualVerifierUnavailable` as `unavailable`;
- catch other verifier invocation errors as `verifier_failure`;
- classify a `None` parsed result as `malformed`;
- retain `no_match` for empty discovery and zero admissions;
- retain `approved` for admitted candidates.

Keep one `_finish(outcome)` helper that records duration/metrics and warns only for operational outcomes. It must be the only terminal image-path logger.

- [ ] **Step 4: Bound the new metric labels**

Replace `_VERIFICATION_OUTCOMES` with:

```python
_VERIFICATION_OUTCOMES = {
    "approved",
    "skipped",
    "unavailable",
    "search_failure",
    "fetch_failure",
    "verifier_timeout",
    "verifier_failure",
    "malformed",
    "no_match",
}
```

Update the unknown-label test to keep proving tenant content is bucketed as `other`.

- [ ] **Step 5: Remove duplicate and swallowed logger paths**

Remove the debug catch from `_collect_images`; expected failures are already converted to `[]` by the flow. Retain one defensive catch there only for unexpected programming errors, using `logger.exception("Image enrichment failed unexpectedly")` and returning `[]` so factual research survives.

When `_collect_images` has no task and no cached approved result, record the bounded `skipped` outcome with zero duration and no warning. This covers explicit opt-out and closed server gates without inventing another logger.

Make `_resolve_tool` and `_from_container` return `None` without logging; Tavily unavailability is already reported by `_research`, and image dependency absence is reported by the flow.

- [ ] **Step 6: Verify one-warning behavior**

Update `tests/test_visual_verification_metrics.py` so operational failure asserts one warning from `app.ai.image_verification_flow`, while `no_match` and `skipped` assert no warning.

Run: `\.venv\Scripts\python.exe -m pytest -q tests/test_image_verification_flow_metrics.py tests/test_visual_verification_metrics.py tests/test_visual_verifier.py tests/test_web_research_tool.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add app/ai/image_verification_flow.py app/ai/web_research_tool.py app/observability/rich_images.py tests/test_image_verification_flow_metrics.py tests/test_visual_verification_metrics.py
git commit -m "fix: report image enrichment outcomes precisely"
```

---

### Task 4: Remove stale prose and prove the complete pipeline

**Files:**
- Modify: `app/ai/web_research_tool.py`
- Modify: `app/ai/image_verification_flow.py`
- Modify: `app/ai/visual_verifier.py`
- Modify: `app/core/config.py`
- Modify: `tests/test_image_verification_flow_metrics.py`
- Modify: `tests/test_visual_verifier.py`
- Modify: `tests/test_verified_image_reaches_the_model.py:120-165`
- Modify: `docs/superpowers/specs/2026-08-04-vision-verified-image-injection-design.md:412-443,537-543`

**Interfaces:**
- Consumes: the default-on `web_research` contract and existing verified-image sink/inventory pipeline.
- Produces: concise current-behavior comments; an end-to-end regression in which no `image_query` is supplied; active documentation with no shared image-deadline claim.

- [ ] **Step 1: Trim implementation-history prose**

Shorten touched module/class/function docstrings to state present contracts only. Remove comments describing prior outages, measured incident timings, review history, or why earlier patches failed. Preserve only concurrency, capability, privacy, cache hand-off, structured-output, and security invariants that are not evident from code.

Apply the same rule to the touched tests: test names and assertions should explain behavior; remove multi-paragraph incident narratives.

- [ ] **Step 2: Update the prior active design**

Rewrite its latency policy to state that Brave, thumbnail, and verifier stages have independent timeouts and the internal-tool policy is the outer bound. Remove the four-second guarantee, timeout-sum acceptance criterion, and references to `image_verification_deadline_seconds`.

- [ ] **Step 3: Run stale-contract and diff checks**

Run:

```powershell
rg -n "image_verification_deadline_seconds|four-second cap|initially four seconds|remaining_seconds" app tests .env.example docs/superpowers/specs/2026-08-04-vision-verified-image-injection-design.md
git diff --check
```

Expected: `rg` returns no matches; `git diff --check` exits zero.

- [ ] **Step 4: Run focused integration and privacy tests**

Run:

```powershell
\.venv\Scripts\python.exe -m pytest -q tests/test_verified_image_reaches_the_model.py tests/test_vision_verified_injection_regression.py tests/test_web_image_byte_cache.py tests/test_message_service_web_image_externalization.py tests/test_rich_placement.py
```

Expected: PASS.

- [ ] **Step 5: Run full verification**

Run:

```powershell
\.venv\Scripts\python.exe -m pytest -q
\.venv\Scripts\python.exe -m ruff check .
```

Expected: the full suite passes with only the repository's established skips, and Ruff reports no violations.

- [ ] **Step 6: Commit the cleanup**

```powershell
git add app/ai/web_research_tool.py app/ai/image_verification_flow.py app/ai/visual_verifier.py app/core/config.py tests/test_image_verification_flow_metrics.py tests/test_visual_verifier.py tests/test_verified_image_reaches_the_model.py docs/superpowers/specs/2026-08-04-vision-verified-image-injection-design.md
git commit -m "refactor: remove obsolete image path machinery"
```
