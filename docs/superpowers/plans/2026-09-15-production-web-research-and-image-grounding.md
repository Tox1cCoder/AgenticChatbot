# Production Web Research and Image Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active metadata-only web-image path with one provider-neutral research session that supplies verified sources and actual validated image bytes to the answer model, then publishes only model-selected images.

**Architecture:** One turn-scoped `WebResearchSession` is shared by the canonical web tools, one model-call middleware, and one grounding parser. Existing provider resolution, safe image transport, rich-item schemas, and stream adapters are reused directly; old sinks, image inventory prompting, automatic image anchoring, and raw agent-facing provider tools are deleted after migration. Search count remains model-driven and is bounded by the existing generation-wide tool budget, while source, page, candidate, concurrency, and byte admission are structurally bounded.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, LangChain/LangGraph middleware, SQLAlchemy/PostgreSQL, httpx, Pillow, pytest/pytest-asyncio, Ruff.

## Global Constraints

- The approved specification is `docs/superpowers/specs/2026-09-10-production-web-research-and-image-grounding-design.md`, revised 2026-09-15.
- Do not add a search-specific call ceiling or change `ResearchBudget.reserve_search()` from its current `None | refusal_code` contract.
- `quick` admits at most five public sources and two page opens; `agentic` admits at most eight public sources and four page opens.
- Ordinary visual research offers at most four validated candidates; explicit gallery research offers at most six.
- Enforce per-image and per-turn downloaded/model byte ceilings plus bounded image-fetch concurrency.
- Evidence injection runs after runtime-model resolution and before request-budget preflight.
- Use `[[source:S1]]` and `[[image:I1]]`; the server authors public URLs and rich markers.
- No image token means no web image. There is no automatic image anchoring or provider-rank fallback.
- A text-only answer model receives neither image bytes nor candidate-selection metadata.
- Required-web answer text is not publicly streamed until terminal citation validation succeeds.
- Interrupts release uncheckpointed prepared images. A resumed model must perform a new canonical search before it can select an image; source records may still be carried as text evidence.
- Planning remaps worker source IDs but never forwards worker web-image markers or rich items. Parent image publication stays disabled until Planning has a parent-scoped multimodal evidence pass.
- Public tool events exclude unselected image descriptors, private origins, bytes, and provider payloads.
- Keep one session, one evidence middleware, and one grounding parser. Remove forwarding helpers and old active paths once callers migrate.
- Preserve only a narrow read-only historical-message projection; legacy formats are never written by new turns.
- Use the repository interpreter `.venv\Scripts\python.exe` in this checkout.

---

### Task 1: Canonical contracts, registry, and structural limits

**Files:**
- Create: `app/ai/web_research/__init__.py`
- Create: `app/ai/web_research/contracts.py`
- Create: `app/ai/web_research/source_registry.py`
- Create: `app/ai/web_research/policy.py`
- Create: `tests/test_web_research_contracts.py`
- Create: `tests/test_web_source_registry.py`
- Create: `tests/test_web_research_policy.py`

**Interfaces:**
- Produces: `ResearchRequest`, `ProviderSource`, `SourceRecord`, `ProviderImageCandidate`, `ImageCandidateRecord`, `ResearchFailure`, `WebEvidenceBundle`, `ResearchScope`, and `ResearchLimits.for_mode()`.
- Produces: `SourceRegistry.admit()`, `.resolve()`, `.mark_opened()`, and `.import_records()`.
- Preserves: current `ResearchBudget.reserve_search()` semantics and duplicate-query behavior.

- [ ] **Step 1: Write failing contract and policy tests**

```python
def test_gallery_limit_is_validated_from_bundle_visual_intent():
    bundle = evidence_bundle(visual_intent="gallery", images=image_records(6))
    assert len(bundle.images) == 6
    with pytest.raises(ValidationError):
        evidence_bundle(visual_intent="figure", images=image_records(5))


def test_distinct_search_count_remains_model_driven():
    budget = ResearchBudget()
    for query in ("alpha release", "beta release", "gamma release", "delta release"):
        assert budget.reserve_search(query) is None
        budget.record_search(query, query)
```

- [ ] **Step 2: Run tests and verify missing-module failures**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_contracts.py tests/test_web_source_registry.py tests/test_web_research_policy.py -q -p no:cacheprovider`

- [ ] **Step 3: Implement frozen contracts and deterministic registry**

`WebEvidenceBundle` includes `mode`, `visual_intent`, positive `operation_index`, sources, images, failures, providers, reuse and omission counts. Validate unique IDs, image-to-source membership, `/web-images/<uuid>` delivery URLs, and cardinality from `visual_intent`. Registry mutation occurs only after concurrent provider operations settle; text sources are admitted before image source pages.

- [ ] **Step 4: Implement limits without a search count field**

```python
_LIMITS = {
    "none": ResearchLimits(sources=0, page_opens=0, model_images=0),
    "quick": ResearchLimits(sources=5, page_opens=2, model_images=4),
    "agentic": ResearchLimits(sources=8, page_opens=4, model_images=4),
}
```

Gallery changes only `model_images` to six. Add an inventory test asserting `research_max_search_calls_per_turn` remains absent.

- [ ] **Step 5: Run focused tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_contracts.py tests/test_web_source_registry.py tests/test_web_research_policy.py tests/test_search_count_is_model_driven.py tests/test_research_budget.py -q -p no:cacheprovider`

Commit: `feat: add canonical web research contracts`

---

### Task 2: Provider adapters and resilient text research

**Files:**
- Create: `app/ai/web_research/providers.py`
- Create: `app/ai/web_research/service.py`
- Create: `tests/fixtures/web_research/tavily_search_success.json`
- Create: `tests/fixtures/web_research/tavily_extract_partial.json`
- Create: `tests/fixtures/web_research/brave_images_success.json`
- Create: `tests/test_web_research_providers.py`
- Create: `tests/test_web_research_service.py`
- Modify: `app/ai/web_query_contract.py`

**Interfaces:**
- Produces: provider protocols/adapters, bounded `ProviderFailure`, `ProviderResolver`, `ProviderHealthRegistry`, `ResearchResultCache`, `WebResearchService.new_session()`, and session `.search()`/`.open()`.

- [ ] **Step 1: Write failing normalization, cancellation, retry, and partial-result tests**

```python
@pytest.mark.asyncio
async def test_image_failure_does_not_discard_text_success():
    session = make_session(text=[SOURCE], images=ProviderFailure("timeout", provider="brave", retryable=True))
    bundle = await session.search(VISUAL_REQUEST)
    assert bundle.sources and not bundle.images
    assert bundle.failures[-1].operation == "image_search"


@pytest.mark.asyncio
async def test_health_is_partitioned_by_provider_configuration():
    health.record_failure("tavily:key-a", "rate_limited")
    assert health.is_open("tavily:key-a")
    assert not health.is_open("tavily:key-b")
```

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_providers.py tests/test_web_research_service.py -q -p no:cacheprovider`

- [ ] **Step 3: Move provider parsing behind adapters**

Re-raise cancellation and LangGraph control flow. Retry only transport, `429`, and `5xx` failures once; ordered fallback follows. Circuit keys use a non-secret provider configuration fingerprint, never provider name alone. Cache only immutable normalized records.

- [ ] **Step 4: Implement deterministic concurrent orchestration**

Use branch-local results and `asyncio.gather(..., return_exceptions=True)`. Admit text records, then image source pages, independent of completion order. `open()` accepts source IDs or public URLs and updates immutable source records through the registry.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_providers.py tests/test_web_research_service.py tests/test_web_query_contract.py tests/test_focused_tool_result.py tests/test_message_stream_cancellation.py -q -p no:cacheprovider`

Commit: `feat: add resilient web research service`

---

### Task 3: Validated image preparation and reference lifecycle

**Files:**
- Modify: `app/models/web_image_reference.py`
- Modify: `app/repositories/web_image_reference.py`
- Modify: `app/services/web_image_service.py`
- Create: `app/alembic/versions/a3b4c5d6e7f8_add_web_image_lifecycle.py`
- Modify: `app/ai/web_research/service.py`
- Create: `app/ai/web_research/model_context.py`
- Create: `tests/test_web_research_images.py`
- Modify: `tests/test_web_image_service.py`
- Modify: `tests/test_web_image_reference_repository.py`
- Modify: `tests/test_web_image_reference_model.py`

**Interfaces:**
- Produces: session-private `PreparedImage`, ordered `.finish(selected_ids)`, `.abort()`, and bounded expiry cleanup for pending rows.
- Produces repository transitions `amark_selected`, `arelease_many`, `aget_pending_for_user`, and `arelease_expired`.

- [ ] **Step 1: Write failing byte, ordering, deduplication, and lifecycle tests**

```python
@pytest.mark.asyncio
async def test_finish_preserves_authored_order_and_releases_unselected():
    session = prepared_session(["I1", "I2", "I3"])
    result = await session.finish(("I2", "I1", "I2"))
    assert [item["candidate_id"] for item in result.selected_images] == ["I2", "I1"]
    assert released_ids() == [reference_for("I3")]


@pytest.mark.asyncio
async def test_interrupt_releases_uncheckpointed_images():
    await session.abort()
    assert await repository.aget_pending_for_user(reference_id, USER_ID) is None
```

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_images.py tests/test_web_image_reference_repository.py tests/test_web_image_reference_model.py -q -p no:cacheprovider`

- [ ] **Step 3: Add the minimal lifecycle columns and repository operations**

Add indexed `lifecycle_state` (`pending`, `selected`, `released`) and nullable `expires_at`. Every transition filters by `user_id`, conversation, current state, and ID. Released rows set `deleted_at`; expiry cleanup affects pending rows only.

- [ ] **Step 4: Prepare images with aggregate limits**

Reuse `WebImageService.fetch_url`. Bound concurrency with one semaphore, stop scheduling after the configured candidate-pool bound, reject when aggregate downloaded or model bytes would exceed the turn limit, and deduplicate by SHA-256 digest before registration. Store validated bytes once.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_images.py tests/test_web_image_service.py tests/test_web_image_reference_repository.py tests/test_web_image_reference_model.py tests/test_web_images_api.py tests/test_alembic_full_chain_postgres.py -q -p no:cacheprovider`

Commit: `feat: validate and lifecycle-manage web images`

---

### Task 4: One session wired to canonical product tools

**Files:**
- Modify: `app/ai/tool_context.py`
- Replace focused portions: `app/ai/web_tools.py`
- Modify: `app/ai/agents/base_agent.py`
- Modify: `app/ai/workflow/middleware.py`
- Modify: `app/ai/workflow/specialists.py`
- Modify: `app/ai/graph.py`
- Modify: `app/core/container.py`
- Create: `tests/test_web_research_tool_session.py`
- Modify: `tests/test_web_tool_binding.py`

**Interfaces:**
- Adds `ToolContext.web_research_session`.
- Produces only `web_search` and `web_open` as ordinary product web tools; no separate `image_search`.

- [ ] **Step 1: Write failing shared-session and binding tests**

Assert the tool and specialist middleware receive the same object, separate turns/workers do not share sessions, and chat/search expose neither raw provider tools nor `image_search`.

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_tool_session.py tests/test_web_tool_binding.py -q -p no:cacheprovider`

- [ ] **Step 3: Inject the service once through the existing container chain**

Create a named `SpecialistBuild` containing the agent, tool middleware, accountant, evidence middleware, and session. Do not add a service-locator helper or wrapper class around the session.

- [ ] **Step 4: Reduce web tools to validation plus session calls**

`web_search` accepts objective, mode, freshness, visual intent, and optional image query. It returns a model projection of the bundle; the tool artifact retains the private canonical bundle for the stream projector. `web_open` resolves source IDs through the same session.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_tool_session.py tests/test_web_tool_binding.py tests/test_client_invocation_isolation.py tests/test_tool_trace_parenting.py -q -p no:cacheprovider`

Commit: `refactor: route web tools through turn sessions`

---

### Task 5: Dynamic multimodal evidence before budget preflight

**Files:**
- Modify: `app/ai/web_research/model_context.py`
- Modify: `app/ai/workflow/middleware.py`
- Modify: `app/ai/workflow/specialists.py`
- Modify: `app/ai/prompts.py`
- Replace: `tests/test_selected_image_reaches_the_model.py`
- Modify: `tests/test_specialist_rich_image_lift.py`
- Create: `tests/test_web_research_model_context.py`

**Interfaces:**
- Produces one `WebEvidenceMiddleware` and `inject_latest_web_evidence()`.

- [ ] **Step 1: Write the real two-round regression tests**

```python
@pytest.mark.asyncio
async def test_second_request_contains_labeled_candidate_pixels_before_preflight():
    result = await two_round_specialist(images=[RED_IMAGE, BLUE_IMAGE], answer="Blue [[image:I2]]")
    request = result.model_requests[1]
    assert adjacent_image_label(request.messages, "I2") == data_url(BLUE_IMAGE)
    assert result.preflight_requests[-1].messages == request.messages


@pytest.mark.asyncio
async def test_text_only_fallback_receives_no_candidate_ids_or_pixels():
    result = await vision_failure_then_text_fallback()
    assert not image_parts(result.fallback_request)
    assert "candidate_id=I" not in flatten_text(result.fallback_request.messages)
```

- [ ] **Step 2: Verify RED against the current static-prompt path**

Run: `.venv\Scripts\python.exe -m pytest tests/test_selected_image_reaches_the_model.py tests/test_web_research_model_context.py -q -p no:cacheprovider`

- [ ] **Step 3: Implement one idempotent evidence middleware**

Place it in the list immediately after `RuntimeModelMiddleware` and before `RequestBudgetMiddleware`. Replace the previous turn-owned evidence message rather than appending duplicates on context-overflow retry. Recalculate vision support for every primary/fallback attempt. Evidence text marks web content and pixels as untrusted.

- [ ] **Step 4: Replace inventory instructions with grounding-token instructions**

Tell the model to cite supplied `S#`, select only visually inspected `I#`, and use zero when irrelevant. Remove every promise that a selected image is already in an available rich-item inventory.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_selected_image_reaches_the_model.py tests/test_web_research_model_context.py tests/test_specialist_rich_image_lift.py tests/test_base_agent_image_history.py tests/test_token_counter.py -q -p no:cacheprovider`

Commit: `feat: show verified web images to answer models`

---

### Task 6: One grounding parser and deterministic closeout

**Files:**
- Create: `app/ai/web_research/grounding.py`
- Modify: `app/ai/workflow/contracts.py`
- Modify: `app/ai/workflow/specialists.py`
- Modify: `app/ai/workflow/finalization.py`
- Modify: `app/core/response_constants.py`
- Modify: `app/core/rich_placement.py`
- Create: `tests/test_web_grounding.py`
- Modify: `tests/test_output_validation.py`
- Modify: `tests/test_rich_placement.py`

**Interfaces:**
- Produces `GroundingParser.feed()/flush()/resolve()` and `GroundingResolution` with ordered source/image IDs, text, rich items, and bounded warnings.

- [ ] **Step 1: Write failing whole-text, every-split, order, and zero-token tests**

Include fenced, indented, and inline code; malformed/stale IDs; duplicates; punctuation; cardinality; `I2,I1,I2` ordering; and an answer with no image token.

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_grounding.py tests/test_output_validation.py -q -p no:cacheprovider`

- [ ] **Step 3: Implement the parser once**

Incremental methods and terminal resolution share one state machine. URLs and rich-item IDs come only from server records. Unknown tokens are removed and reported with bounded codes.

- [ ] **Step 4: Resolve before session finish**

After `agent.ainvoke`, resolve final content, pass `resolution.selected_image_ids` to `session.finish()`, then create immutable `OutcomeProvenance`. On `GraphBubbleUp`, suspend; on cancellation/exception/hard failure, abort. Delete image handling from `finalize_article_content`; it continues to place non-image widgets only.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_grounding.py tests/test_output_validation.py tests/test_rich_placement.py tests/test_rich_response_contract.py tests/test_rich_response_metadata.py -q -p no:cacheprovider`

Commit: `feat: ground web citations and selected images`

---

### Task 7: Public source events, private candidate projection, and history parity

**Files:**
- Modify: `app/services/event_streaming/events.py`
- Modify: `app/services/event_streaming/graph_public_projection.py`
- Modify: `app/services/event_streaming/internal_sse.py`
- Modify: `app/services/event_streaming/ai_sdk_v6.py`
- Modify: `app/services/event_streaming/ai_sdk_projection.py`
- Modify: `app/services/message_service.py`
- Modify: `app/schemas/message.py`
- Modify: `demo.py`
- Create: `tests/test_web_source_streaming.py`
- Create: `tests/test_web_tool_output_privacy.py`
- Modify: `tests/test_ai_sdk_v6_stream_contract.py`
- Modify: `tests/test_internal_sse_stream_contract.py`

**Interfaces:**
- Produces canonical `sources_upsert`, internal `sources`, and AI SDK `source-url` parts.

- [ ] **Step 1: Write failing source parity and candidate-leak tests**

Assert the public `tool_execution_end` omits images/private bundle fields, selected rich-item upsert precedes its marker, and stream/history source IDs match.

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_source_streaming.py tests/test_web_tool_output_privacy.py -q -p no:cacheprovider`

- [ ] **Step 3: Sanitize at the graph-to-public boundary**

Parse the private canonical artifact into projection context, emit its sources once, retain candidate mapping only in that context, and replace the public tool output with bounded status/source data. Never reconstruct records from arbitrary prose.

- [ ] **Step 4: Persist and reload canonical sources**

Write `web_sources_version=1` and selected rich items for complete/partial/stopped/resumed messages. AI SDK history emits flat `source-url` parts; Streamlit keys source state by ID.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_source_streaming.py tests/test_web_tool_output_privacy.py tests/test_ai_sdk_v6_stream_contract.py tests/test_internal_sse_stream_contract.py tests/test_message_service_event_streaming.py -q -p no:cacheprovider`

Commit: `feat: stream canonical web sources privately`

---

### Task 8: Required-web routing and safe answer streaming

**Files:**
- Modify: `app/ai/workflow/contracts.py`
- Modify: `app/ai/workflow/routing.py`
- Modify: `app/ai/workflow/finalization.py`
- Modify: `app/services/event_streaming/graph_public_projection.py`
- Create: `tests/test_web_research_output_policy.py`
- Create: `tests/test_required_web_streaming.py`
- Modify: `tests/test_production_workflow_graph.py`

**Interfaces:**
- Adds `requires_web` and `research_mode` to routing decisions.
- Produces `WebEvidencePolicy` and required-web answer buffering.

- [ ] **Step 1: Write failing routing, terminal, and no-leak stream tests**

Assert explicit browse/current requests cannot be downgraded, any turn that used canonical web evidence requires at least one valid citation, and uncited raw deltas are absent from both public adapters.

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_output_policy.py tests/test_required_web_streaming.py tests/test_production_workflow_graph.py -q -p no:cacheprovider`

- [ ] **Step 3: Implement policy without another model call**

Apply deterministic routing requirements after schema parsing. Select the output policy when routing requires web or canonical web evidence exists. With no valid cited source, replace content with the exact server-owned unverified response and bounded reason codes.

- [ ] **Step 4: Buffer only required-web answer text**

Continue streaming tools, reasoning, and sources. Hold answer deltas in projection context until the finalized response is available, then emit only finalized content. Non-required turns retain normal token streaming.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_output_policy.py tests/test_required_web_streaming.py tests/test_output_validation.py tests/test_production_workflow_graph.py -q -p no:cacheprovider`

Commit: `feat: require grounded web answers`

---

### Task 9: Fail-closed Continue and Planning worker isolation

**Files:**
- Modify: `app/ai/workflow/continuation.py`
- Modify: `app/ai/workflow/planning_execution.py`
- Modify: `app/ai/workflow/contracts.py`
- Modify: `app/ai/workflow/specialists.py`
- Create: `tests/test_web_research_continuation.py`
- Create: `tests/test_web_research_worker_remap.py`

**Interfaces:**
- Carries canonical source records across Continue.
- Remaps duplicate worker source IDs and strips worker web-image markers/items before Planning synthesis.

- [ ] **Step 1: Write failing Continue and two-worker collision tests**

Worker A and B both return local `S1/I1`; assert parent sources receive unique IDs and worker image markers/items never reach synthesis or public metadata. A human-approval interrupt must release prepared references because their byte/ID state is intentionally not checkpointed. A validated budget pause persists its selected items and sources before advertising Continue.

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_continuation.py tests/test_web_research_worker_remap.py -q -p no:cacheprovider`

- [ ] **Step 3: Carry only source evidence that is safe to checkpoint**

Persist the paused epoch's validated source records and selected rich items with
its partial assistant message, then carry only source records into the next
epoch. Never serialize image bytes, upstream image URLs, or candidate mappings
into graph state. A resumed answer can publish a new image only after a fresh
canonical search offers pixels to that model call.

- [ ] **Step 4: Remap sources and isolate worker images before synthesis**

Allocate parent source IDs by stable URL order. Strip rich markers from worker synthesis text and release worker web-image references; do not aggregate worker web rich items into the parent outcome.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_continuation.py tests/test_web_research_worker_remap.py tests/test_routing_v2_continuation_streaming.py tests/test_planning_subagents.py -q -p no:cacheprovider`

Commit: `feat: preserve grounded web evidence across execution scopes`

---

### Task 10: Delete superseded active and forwarding code

**Files:**
- Delete: `app/ai/selected_image_sink.py`
- Delete: `app/ai/image_discovery_flow.py`
- Delete: `tests/test_selected_image_sink.py`
- Delete: `tests/test_image_discovery_flow.py`
- Delete: `tests/test_provider_selected_image_injection.py`
- Delete: `tests/test_multi_image_research.py`
- Modify: `tests/test_image_recency.py`
- Modify: `tests/test_tool_trace_parenting.py`
- Modify: `tests/test_web_tool_dependency_resolution.py`
- Modify: `tests/test_web_tools.py`
- Modify: `app/ai/tool_execution.py`
- Modify: `app/ai/deferred_tool_binding.py`
- Modify: `app/ai/mcp_tool_catalog.py`
- Modify: `app/core/rich_placement.py`
- Modify: `app/core/response_constants.py`
- Modify: `app/core/config.py`
- Modify: `README.md`
- Create: `tests/test_web_research_active_path_inventory.py`

**Interfaces:**
- Produces one active web/image execution path and one explicitly named historical projection.

- [ ] **Step 1: Write the failing active-path inventory test**

Forbid production imports/symbols `selected_image_sink`, `offer_selected_images`, `select_brave_candidates`, `anchor_image_items_by_query`, `create_image_search_tool`, and raw Tavily/Brave tools in ordinary agent capabilities.

- [ ] **Step 2: Verify RED and capture all callers**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_active_path_inventory.py -q -p no:cacheprovider`

- [ ] **Step 3: Delete old execution paths and collapse trivial wrappers**

Remove old sink/discovery modules, image-search harvesting, inventory prompting, query anchoring, active legacy `metadata["images"]` writes, obsolete settings, and one-caller forwarding helpers created during migration. Preserve generic MCP image content blocks and the narrow historical reader.

- [ ] **Step 4: Verify GREEN and commit**

Run: `rg -n "selected_image_sink|offer_selected_images|select_brave_candidates|anchor_image_items_by_query|create_image_search_tool" app tests README.md`

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_active_path_inventory.py tests/test_web_tools.py tests/test_rich_placement.py tests/test_ai_sdk_context_window.py -q -p no:cacheprovider`

Commit: `refactor: remove legacy web image execution paths`

---

### Task 11: Configuration, metrics, evaluation, and operations

**Files:**
- Modify: `app/core/config.py`
- Modify: `.env.example`
- Create: `app/observability/web_research.py`
- Create: `app/evaluation/web_research/contracts.py`
- Create: `app/evaluation/web_research/metrics.py`
- Create: `app/evaluation/web_research/harness.py`
- Create: `eval/web_research/cases.json`
- Create: `scripts/evaluate_web_research.py`
- Create: `docs/operations/web-research-rollout.md`
- Modify: `docs/frontend/rich-image-rendering.md`
- Create: `tests/test_web_research_config.py`
- Create: `tests/test_web_research_metrics.py`
- Create: `tests/test_web_research_evaluation.py`
- Create: `tests/test_web_research_docs.py`

**Interfaces:**
- Produces bounded operational metrics, one short-lived rollout switch, structural settings, deterministic evaluation, and expiry cleanup instructions.

- [ ] **Step 1: Write failing bounded-setting, label, and evaluation tests**

Assert there is no search-count setting or total research deadline; assert aggregate image byte/concurrency/lifecycle expiry defaults; reject user-derived metric labels; score source/image membership, zero-token behavior, and stream/history parity.

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_config.py tests/test_web_research_metrics.py tests/test_web_research_evaluation.py -q -p no:cacheprovider`

- [ ] **Step 3: Implement bounded configuration and metrics**

Labels are limited to operation, mode, provider fingerprint class, outcome, reason code, visual intent, and lifecycle transition. No query, URL, title, user, tenant, or raw exception appears in labels/log fields.

- [ ] **Step 4: Implement recorded-provider evaluation and rollout docs**

The deterministic scorer covers source/image membership, no-token/no-image, and stream/history parity. Real focused tests gate byte injection, text fallback, required-web suppression, fail-closed Continue, Planning isolation, privacy, concurrency, URL safety, and cleanup wiring. Live provider canaries remain opt-in and non-blocking.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_config.py tests/test_web_research_metrics.py tests/test_web_research_evaluation.py tests/test_web_research_docs.py -q -p no:cacheprovider`

Commit: `test: add web research release controls`

---

### Task 12: Full verification and release evidence

**Files:**
- Modify only files required by failures proven in this task.

- [ ] **Step 1: Run the focused canonical matrix**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_research_contracts.py tests/test_web_source_registry.py tests/test_web_research_policy.py tests/test_web_research_providers.py tests/test_web_research_service.py tests/test_web_research_images.py tests/test_web_research_tool_session.py tests/test_web_research_model_context.py tests/test_web_grounding.py tests/test_web_source_streaming.py tests/test_web_tool_output_privacy.py tests/test_web_research_output_policy.py tests/test_required_web_streaming.py tests/test_web_research_continuation.py tests/test_web_research_worker_remap.py tests/test_web_research_active_path_inventory.py tests/test_web_research_config.py tests/test_web_research_metrics.py tests/test_web_research_evaluation.py -q -p no:cacheprovider`

- [ ] **Step 2: Run repository quality gates**

Run: `.venv\Scripts\python.exe -m ruff check app client_backend tests scripts`

Run: `.venv\Scripts\python.exe -m pytest -q -m "not live_provider" -p no:cacheprovider`

- [ ] **Step 3: Generate deterministic evidence**

Run: `.venv\Scripts\python.exe scripts/evaluate_web_research.py --cases eval/web_research/cases.json --output output/audits/web-research-eval.json`

- [ ] **Step 4: Inspect the active-path inventory and diff**

Run: `rg -n "selected_image_sink|offer_selected_images|select_brave_candidates|anchor_image_items_by_query|create_image_search_tool" app tests README.md`

Expected: no production hits and no test/docs hits outside the historical-message projection contract.

- [ ] **Step 5: Commit any verification-only corrections**

Commit: `chore: verify production web research rollout`

## Completion Gate

- The real second answer-model request contains validated bytes and adjacent IDs, and budget preflight measured that same request.
- Choosing `I2` publishes the protected bytes/digest for `I2`; choosing no ID publishes no image.
- No current active path can auto-anchor or append an unselected web image.
- Required-web raw answer text is never published without a valid source citation.
- Continue preserves sources while releasing uncheckpointed images; Planning remaps sources and cannot relay worker web images.
- Public tool events and metadata contain no unselected candidate, private origin, or bytes.
- Current model-driven search-count behavior and general tool ceilings remain intact.
- Old feature-owned execution code and trivial migration wrappers are removed.
- Focused tests, Ruff, the non-live suite, and deterministic evaluation pass.
