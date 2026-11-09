# Inline Answer Visuals and Routing Repair Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task. Steps use checkbox syntax for tracking. This is a proposal for review; it does not authorize implementation or deployment.

**Goal:** Complete `answerVisualsV1` and `sourceFiguresV1` across generation, routing, streaming, persistence, history, Resume, Continue, and Streamlit, without building parallel infrastructure.

**Architecture:** Extend the existing typed rich-item contract and two existing image types. Keep the answering specialist responsible for inline visuals, add two narrow native tools, and reuse the current tool artifacts, grounding, capability projector, checkpoints, protected media, and rendering paths.

**Tech stack:** Existing Python/Pydantic/LangChain/LangGraph, FastAPI, AI SDK v6 SSE, PostgreSQL/checkpoints, and Streamlit. No new dependency.

## Scope and evidence

The user selected **both visual capabilities end to end**, with particular emphasis on avoiding extra helpers and code bloat. This plan replaces the implementation task breakdown in [the earlier visual review](../../reviews/2026-10-26-rich-answer-visuals-review.md); that review remains design history.

Read-only inspection of saved conversations on 2026-11-09 established:

| Case | Saved result | Required result |
|---|---|---|
| Response times, conversation `392f4a7d-2455-4771-aad7-68a956945777` | Router directly chose Canvas, confidence 0.95; `canvas_artifact` | Answering Chat specialist, inline static `bars` |
| Volcano comparison, conversation `f833096b-5eca-410c-be54-59fbf3bfcc42` | Search specialist; two `image` items and literal fenced ASCII | Source figure plus a typed `steps` visual |
| Latest volcano retry, conversation `45b53afb-928c-4c08-a418-8ff3d44d7484` | Router chose Canvas; interactive artifact and two images | Answering Chat/Search specialist, inline figure and steps |

The actual Streamlit view builder preserved the ASCII as Markdown. It received no steps descriptor. The fault starts upstream of rendering.

Additional code findings that the implementation must address:

- The active router prompt is in `app/ai/workflow/routing.py`. `app/ai/prompts.py` contains an inactive router prompt, and several visual-routing tests assert against that inactive copy.
- `RichItemType` has no `answer_visual`; Chat/Resume/Continue schemas have neither new capability flag.
- `build_tool_render_candidate()` and the existing artifact path already supply stable tool-backed rich candidates. `AIService._build_tool_end_rich_items()` already calls existing candidate functions; a new generic candidate framework is unnecessary.
- `SpecialistFactory._to_outcome()` currently obtains `rich_items` from web grounding. New tool-created records must also reach `OutcomeProvenance.rich_items` and final metadata.
- `response_constants._finalize_rich_items()` checks references for images, but does not enforce `inline_only` on all non-image candidates despite its comment. New static visuals would otherwise persist when unused.
- Image groups already have storage, externalization, authenticated loading, and HTML rendering. They lack composition, visible cell labels, and group captions. Group validation currently requires two cells, while degradation/rendering paths can retain one.
- `WebResearchSession._retain_active()` releases evicted references; `SpecialistFactory.invoke()` aborts sessions on interrupts. Both would discard a composed-but-not-yet-published figure unless its dependencies are explicitly retained.
- `ContinuationPausePayload` already carries selected `rich_items`, and `OutcomeProvenance` already supports them. Existing research accounting carries quotas/deduplication, not a resumable image/figure catalog.
- Streamlit Continue omits the base rich flag and ignores live rich items and approval interrupts. These are part of this repair.

Baseline: **257 tests passed** across existing prompt, routing, tool-render, rich metadata, capability, Streamlit, continuation, and fork suites. Passing these tests does not establish semantic routing or the unimplemented visual features.

## Approach and code-size constraints

Three approaches were considered:

1. **Prompt-only routing correction:** smallest change, but cannot generate the missing typed visual/figure payloads. Insufficient for the selected scope.
2. **Encode everything as generic `tool_render`:** reuses charts, but steps, assessments, evidence, fallback, and independent capability negotiation become undocumented exceptions inside a free-form payload.
3. **Small typed extension of the existing contract — recommended:** introduce `answer_visual`; enrich `image`/`image_group`; reuse the pipeline and presentation primitives already present.

Implementation constraints:

- At most **one new production module**, `app/ai/answer_visual_tools.py`, for the new static authoring tool. It is domain behavior, not a helper package. Schemas stay in their existing owner, `app/core/rich_response.py`; figure composition stays in the existing web-research session/tool modules.
- No new agent, service layer, registry service, candidate manager, capability wrapper class, event family, database table/migration, queue, cache, charting package, or widget session.
- Extend existing functions before adding functions. New schema validators and the explicitly listed tool/session methods below are justified domain operations. Do not add forwarding wrappers, single-use utility functions, or alternate formatters/parsers.
- One fallback implementation per typed payload, reused by projection and copy/export. One marker grammar. One candidate construction path for each native tool. No duplicated stream/history formatting.
- Keep `rich_items_version: 1`, `<!--rich:<id>-->`, `data-rich-items`, and the existing protected media routes.
- Preserve generic chart/table tool results, widgets, canvas artifacts, ordinary retrieved images, custom-agent restrictions, and the existing routing model/schema/retry behavior.
- No phrase/regex routing overrides, confidence thresholds, silent Chat substitution, model-name allowlists, or restoration of the removed blanket vision-capability gate.
- No production conversations, stored model settings, or existing artifacts are rewritten. Live acceptance uses disposable conversations.
- Keep the existing private Planning-worker image boundary: source-figure composition is supported by public answering specialists, not private workers. Custom agents gain no implicit tool permissions.
- The source-figure authoring input uses the current web session's inspected `I#` candidates. Existing RAG image delivery remains intact; static visuals can cite admitted RAG evidence. This feature does not add another document-image retrieval system.

## Ownership and file map

| Responsibility | Existing files to extend |
|---|---|
| Routing boundary and descriptions | `app/ai/workflow/routing.py`, `app/ai/agent_metadata.py`, `app/ai/prompts.py` |
| Public static/image schemas, marker handling, fallback | `app/core/rich_response.py` |
| Static tool | **Create:** `app/ai/answer_visual_tools.py` |
| Figure tool and owned dependencies | `app/ai/web_tools.py`, `app/ai/web_research/service.py`, `app/ai/web_research/grounding.py`, `app/ai/web_research/model_context.py` |
| Per-request binding, native tool identity, artifacts | `app/ai/tool_context.py`, `app/ai/tool_execution.py`, `app/ai/agents/base_agent.py`, `app/ai/workflow/middleware.py`, `app/ai/workflow/specialists.py` |
| RAG binding/evidence | `app/ai/agents/rag_agent.py`, `app/ai/workflow/rag_execution.py`, `app/ai/rag_tool_actions.py` |
| Admission and validation | `app/core/response_constants.py`, `app/ai/graph.py`, `app/ai/workflow/finalization.py` |
| Capability propagation | `app/api/ai_sdk.py`, `app/api/messages.py`, `app/schemas/message.py`, `app/schemas/workflow.py`, `app/schemas/generation.py`, `app/ai/schemas.py`, `app/interfaces/message_service_interface.py`, `app/services/ai_service.py`, `app/services/message_service.py` |
| Wire projection | `app/services/event_streaming/ai_sdk_projection.py`, `app/services/event_streaming/ai_sdk_v6.py`, `app/services/event_streaming/internal_sse.py` |
| Resume/Continue state | `app/ai/workflow/state.py`, `app/ai/workflow/continuation.py`, existing graph/service resume methods |
| Figure degradation/fork | `app/services/message_service.py`, `app/services/conversation_fork_metadata.py` |
| Streamlit | `app/ui/rich_response.py`, `demo.py` |
| Sidecar forwarding | Existing `client_backend/api/messages.py`, `client_backend/api/proxy.py`, and `client_backend/services/server_api.py`; change only if new forwarding tests expose a loss |

This is an ownership map, not permission for broad refactoring. A listed file needs no edit if its existing path passes the new behavior tests unchanged.

## Contract decisions

### Capabilities

Add `answerVisualsV1` / `answer_visuals_v1` and `sourceFiguresV1` / `source_figures_v1`, each defaulting to `false`. Effective values require the request's base rich capability and `settings.inline_rich_response_enabled`:

```python
base = bool(settings.inline_rich_response_enabled and inline_rich_response_v1)
answer_visuals_v1 = bool(base and answer_visuals_v1)
source_figures_v1 = bool(base and source_figures_v1)
```

Use plain booleans in the existing request/context/state objects. Apply the same rule on Chat, Resume, Continue, non-streaming message creation, and history queries. Requests negotiate their own delivery; never save a client's downgraded projection as canonical content. `streamWebAnswerV1` stays an independent presentation flag.

### Static visual

Use `type: "answer_visual"`, `display_policy: "inline_only"`, with `payload.variant` discriminating `steps`, `bars`, and `comparison_table`. Tool inputs contain content/evidence, never HTML, JavaScript, arbitrary components, executable expressions, or an independently authored fallback.

Example canonical bars payload:

```json
{
  "variant": "bars",
  "title": "Illustrative response times",
  "unit": "ms",
  "basis": "illustrative",
  "assumptions": "User-supplied example values; lower is better.",
  "bars": [
    {"label": "A", "value": 120, "evidence": []},
    {"label": "B", "value": 180, "evidence": []},
    {"label": "C", "value": 300, "evidence": []}
  ],
  "scale": {"min": 0, "max": 300}
}
```

The other two variants have these concrete shapes; tool inputs use `evidence_refs` where public descriptors use resolved `evidence`:

```json
{
  "variant": "steps",
  "title": "Identify the shape",
  "steps": [
    {"title": "Trace the skyline", "note": "Look at the whole mountain.", "evidence": []},
    {"title": "Compare width and height", "note": "Compare the proportions.", "evidence": []},
    {"title": "Inspect the upper slopes", "note": "Use the outline as a clue.", "evidence": []}
  ]
}
```

```json
{
  "variant": "comparison_table",
  "title": "Illustrative plan comparison",
  "columns": ["Plan", "Export suitability"],
  "rows": [
    {"cells": [
      {"text": "Basic", "evidence": []},
      {"text": "Unavailable", "status": "negative", "reason": "This example plan has no export.", "evidence": []}
    ]},
    {"cells": [
      {"text": "Pro", "evidence": []},
      {"text": "Suitable", "status": "positive", "reason": "This example plan includes export.", "evidence": []}
    ]}
  ]
}
```

- Bounds: 12 steps; 12 bars; 12 table rows × 6 columns; 160 characters per title/label; 500 per note/reason/caption; six referenced static visuals per answer; at most six pending composed figures per logical turn, each within existing image-candidate capacity. New visual/figure descriptors are bounded to 16 KiB per item and 64 KiB combined, including fallback/evidence. Unused creations count toward pending limits until released.
- Bars: finite nonnegative values, one unit, linear zero baseline, shared maximum >= every value and > 0; default max is the greatest value, or 1 for all-zero data. Require `basis` (`measured`, `calculated`, `illustrative`); calculated/illustrative values disclose assumptions. Measured claims need admitted evidence.
- Steps: ordinal sequence, no inferred execution/completion state.
- Tables: each row has exactly as many cells as columns. A cell's optional labelled assessment status is `positive`, `neutral`, `caution`, or `negative`, with a required visible reason when status is supplied. Theme colors are renderer decisions.
- `evidence_refs` in tool input use `kind` (`web` or `rag`) and `id`. A web reference can be `{ "kind": "web", "id": "S1" }`; a RAG reference names an admitted EvidencePack record's actual `evidence_id`. Resolve them against existing session/EvidencePack records before producing public `evidence` entries. Unknown references produce a concise tool error. Membership validation does not prove factual support.
- Server-generated fallback is an ordered list, a label/value/unit Markdown table, or a semantic Markdown table, respectively. Derive it from the exact same validated payload and resolved citations. Serialize it with the descriptor; verify/recompute it at the trusted construction boundary.

### Source figure

Keep `image` and `image_group`. Reuse `ImagePayload.caption`; add optional visible `label` to cells/single images and optional group `caption`, resolved caption evidence, and server-derived fallback. Do not add a third image type or duplicate media metadata.

`compose_source_figure` receives ordered `I#` IDs, labels, alt text, and caption/evidence. It resolves owned protected media server-side. The existing `provenance` identifies the internal composition tool; no separate figure-kind registry is needed.

Creation accepts one image or a bounded group within the existing image admission limits. A group may degrade to one cell: preserve its label/source, remove the comparison-wide caption and its claims, and show an explicit comparison-unavailable state. Apply the same rule during externalization, fork, browser load failure, and history projection.

## Implementation sequence

Each task follows a failing behavioral test → minimal implementation → focused checks → reviewable commit cycle. Do not treat prompt-string assertions or mocked router decisions as proof of semantic routing.

### Task 1: Correct the active routing boundary and retire the duplicate prompt

**Modify:** `app/ai/workflow/routing.py`, `app/ai/agent_metadata.py`, `app/ai/prompts.py`.

**Tests/data:** Extend `tests/test_router.py`, `tests/test_routing_inventory.py`, `tests/test_routing_context.py`, `tests/test_prompts_visual_strategy.py`, `tests/test_prompts_media_capability.py`, and `tests/test_widget_runtime.py`; create `eval/routing/inline_visuals_v1.jsonl` using the existing dataset schema.

**Interface:** `RoutingService.route()` and `RoutingDecision` stay unchanged. The live inventory explicitly describes inline static visuals as work owned by answering specialists; Canvas requires a standalone browser artifact request or an edit to that artifact.

- [ ] Add regression coverage against the **active** prompt/inventory. Move the legacy prompt imports in the three prompt/widget suites to `app.ai.workflow.routing`; remove the unused router string from `prompts.py` and correct the stale comment in `agent_metadata.py`.
- [ ] Include these instructions in the active structured router prompt, while preserving untrusted-input handling and live inventory validation:

```text
An explanation or comparison with a compact inline chart, steps, table, or
source figure remains with its answering specialist. It does not request a
standalone browser artifact. Choose canvas_agent when creating or editing the
standalone artifact is the requested deliverable. An open canvas alone does
not make an unrelated question canvas work. Supplied example data does not
require web research merely because a visual was requested.
```

- [ ] Add paired focused evaluation cases: the exact response-time prompt → Chat; the volcano prompt → Chat/Search, excluding Canvas; uploaded-document figures → RAG; explicit standalone HTML dashboard → Canvas; edit active artifact → Canvas; discuss an artifact without editing → Chat; explicit raster generation → Image Generator. Include fresh and previous-Canvas contexts, plus Thai/Vietnamese equivalents.
- [ ] Use the existing routing CLI for live evaluation of this file. Do not change the approved golden dataset's review hash or loosen its gate to make a result pass. Test the new focused dataset and the existing golden corpus separately.
- [ ] Run the focused prompt/routing tests, then inspect actual initial/final agent IDs in disposable conversations. Commit this independently testable routing correction.

Focused dataset row shape, with no new evaluation code:

```json
{"case_id":"visual-bars-en","language":"en","category":"general_chat","context":{},"message":"Compare illustrative response times A=120 ms, B=180 ms, C=300 ms. Put a compact static bar visual inline between explanation and conclusion. No interactive widget.","primary_agent_id":"chat_agent","acceptable_agent_ids":["chat_agent"]}
```

### Task 2: Add bounded schemas and fallback behavior in the existing contract

**Modify:** `app/core/rich_response.py`, `app/core/response_constants.py`.

**Tests/data:** Extend `tests/test_rich_response_contract.py`, `tests/test_rich_response_metadata.py`; create `tests/fixtures/rich_answer_visuals/` with `steps.json`, `bars.json`, `comparison_table.json`, and `source_figure.json` as canonical shared examples.

**Interface:** `validate_public_rich_item()` accepts `AnswerVisualRichItem` in its existing discriminated union. Typed payload methods `to_fallback_markdown() -> str` derive fallback from validated values and resolved evidence. Image/group payloads retain their current fields and gain only the composition fields defined above.

- [ ] First test the bars example above wrapped in an `answer_visual` item; it currently fails because the type is unsupported. Add equivalent steps and comparison fixtures, and input rejection cases for invalid units/scales, NaN/infinity, unknown status, excessive size, and executable content.
- [ ] Add the three payload variants and `AnswerVisualRichItem` to this file. Keep input-only evidence references distinct from resolved public evidence and reject client-authored fallback in the tool schema.
- [ ] Derive fallback through the payload methods. Test identical values/citations, Markdown escaping, readable zero data, repeated markers, and Unicode labels.
- [ ] Fix `_finalize_rich_items()` to honor every candidate's declared `inline_only` policy, while preserving widget/canvas `inline_or_append` behavior. Test that an unused static item is absent and a referenced item appears once.
- [ ] Permit valid single-cell degraded groups and preserve visible labels/caption evidence in public serialization. Maintain protected URL restrictions and existing legacy image payload compatibility.
- [ ] Run rich-contract/metadata/image-selection tests and commit the typed contract, still without enabling authoring tools.

Representative behavior assertion using existing public entry points:

```python
item = validate_public_rich_item(json.loads(Path(
    "tests/fixtures/rich_answer_visuals/bars.json"
).read_text(encoding="utf-8")))
assert item.type.value == "answer_visual"
assert [bar.value for bar in item.payload.bars] == [120, 180, 300]
assert item.payload.scale.min == 0
assert "120" in item.payload.to_fallback_markdown()
```

### Task 3: Wire capability flags through existing request and resume paths

**Modify:** Request/context/API/service/interface files in the ownership map; extend `AISDKV6StreamState` in `ai_sdk_v6.py`. No new capability class.

**Tests:** Extend `tests/test_ai_sdk_resume_continue_capabilities.py`, `tests/test_workflow_request_conversion.py`, `tests/test_workflow_request_schema_parity.py`, `tests/client_backend/test_messages.py`, `tests/client_backend/test_conversation_routes.py`.

**Interface:** New plain boolean parameters `answer_visuals_v1=False`, `source_figures_v1=False` accompany `inline_rich_response_v1` on existing service/adapter methods. History accepts the camelCase query aliases. The two effective flags are request-local.

- [ ] Add schema and propagation tests for all eight base/static/figure boolean combinations, aliases, omitted defaults, and server rollout disabled. Cover Chat, non-streaming creation, Resume, Continue, and history.
- [ ] Follow the existing base-rich flag through `MessageCreate`, `InterruptResumeRequest`, `ContinueGenerationRequest`, `WorkflowExecutionRequest`, AI request conversion, `GraphContext`, service interfaces, graph execution/resume options, and adapter state. Apply the effective-capability rule at the boundary; do not mutate shared agent instances.
- [ ] On Resume/Continue, keep the current request's presentation separate from checkpointed canonical descriptors. A feature may be downgraded for one reader without deleting its stored data. New authoring calls require the current effective capability.
- [ ] Assert sidecar body and history query forwarding. Preserve its current runtime bridge and authentication behavior; add no visual logic to the proxy.
- [ ] Run capability/conversion/proxy tests and commit while the new tools remain unbound.

Example new request behavior test:

```python
request = AISDKChatRequest.model_validate({
    "inlineRichResponseV1": True,
    "answerVisualsV1": True,
    "sourceFiguresV1": False,
})
assert request.answer_visuals_v1 is True
assert request.source_figures_v1 is False
```

### Task 4: Implement the native static tool and reuse tool-artifact publication

**Create:** `app/ai/answer_visual_tools.py` and `tests/test_answer_visual_tools.py`.

**Modify:** `app/ai/tool_context.py`, `app/ai/tool_execution.py`, `app/ai/agents/base_agent.py`, `app/ai/prompts.py`, `app/ai/workflow/middleware.py`, `app/ai/workflow/specialists.py`, existing RAG binding/evidence paths, `app/ai/graph.py`, and existing finalization policies.

**Interface:** `create_answer_visual_tool(*, tool_scope: str | None = None) -> StructuredTool`, named `create_answer_visual`, with identity `internal::create_answer_visual`. It takes `visual` containing one typed variant and evidence references. It returns a compact text marker plus the validated descriptor in the existing structured result/artifact envelope.

- [ ] Add a failing real `execute_tool_calls()` test using a bound native tool. Check descriptor creation, stable identity, a marker in the model-visible result, and absence of a duplicate generic `tool_render` item.
- [ ] Add the current call ID and effective flags to the existing `ToolContext`; pass them through the existing per-call context manager. Derive IDs in the tools with `uuid5(NAMESPACE_URL, f"{logical_turn_id}:{tool_call_id}:answer_visual")`. No ID-manager helper is needed. A different call produces a different ID; a replay cannot replace an existing ID with different content.
- [ ] Reuse `normalize_tool_result_for_rendering()`, `_ToolCallResults.append_result()`, and `_attach_rich_candidates_to_artifact()`. Recognize dedicated descriptors only from the **bound internal tool identity**, not an arbitrary tool name or third-party JSON. Extend the existing candidate functions where necessary; do not add another candidate builder module.
- [ ] Make dedicated output mutually exclusive with generic tool-render candidates and duplicate answer-bubble renders. Keep the internal descriptor intact through tool-result offloading while exposing only bounded content to the model.
- [ ] Merge tool-created records with grounded records inside the active `_to_outcome()` path; carry them in `OutcomeProvenance.rich_items`. Preserve already grounded images. Deduplicate equal records by ID; reject conflicting optional records without rebinding an image or erasing answer prose.
- [ ] Extend existing artifact/Web/RAG validation policies to validate nested resolved evidence and publish sources used only in a visual. Use current `SourceRegistry` and RAG EvidencePack records; no new evidence store or validation-policy family.
- [ ] Prepare and test binding through existing per-request `internal_tools` and tool factories for public Chat/Search/RAG. Make initial and refreshed bindings consistent and respect forced synthesis, tool scope, exclusions, custom-agent allowlists, and private-worker boundaries. Never store a request's flags on a shared agent. Activate public binding only at the end of Task 7; factories can be exercised directly until then, without adding another rollout flag.
- [ ] Update answer guidance: prose/tables for ordinary answers, static tool for bounded authored visuals, source composition for inspected images, widgets for interaction, Canvas for standalone artifacts. An explicit static-visual request on a capable client calls the native tool; ASCII, a widget, or a Canvas artifact must not silently stand in for it. If creation fails, return readable fallback and state the limitation. Do not instruct the model to serialize the whole answer as JSON.
- [ ] Run tool execution, metadata, specialist, RAG-grounding, and allowlist tests; commit the static authoring path without exposing it to ordinary requests yet.

Required trusted result shape:

```python
{
    "content": [{"type": "text", "text": f"Created visual. Place <!--rich:{item_id}--> inline."}],
    "structuredContent": {"rich_item": descriptor},
}
```

### Task 5: Compose source figures inside the existing web session

**Modify:** `app/ai/web_tools.py`, `app/ai/web_research/service.py`, `app/ai/web_research/grounding.py`, `app/ai/web_research/model_context.py`, existing tool-context/binding paths.

**Create test:** `tests/test_source_figure_composition.py`. Extend `tests/test_selected_image_reaches_the_model.py`, `tests/test_web_research_images.py`, `tests/test_web_grounding.py`.

**Interface:** `create_compose_source_figure_tool(*, tool_scope: str | None = None) -> StructuredTool`, named `compose_source_figure`, identity `internal::compose_source_figure`. Its session operation has this signature; the wrapper obtains the stable ID from the real turn/call identity using the UUID5 expression with suffix `source_figure`:

```text
async WebResearchSession.compose_figure(
    self,
    *,
    item_id: str,
    cells: Sequence[dict[str, str]],  # candidate_id, label, alt_text
    caption: str | None = None,
    caption_evidence_refs: Sequence[dict[str, str]] = (),
) -> dict[str, Any]
```

The return value is one validated existing-family image/image_group descriptor.

- [ ] Test two session-owned candidates composed in a requested order, one-cell composition, unknown/foreign IDs, unpresented candidates, unsupported URLs, unknown caption evidence, and replays.
- [ ] Extend existing session state with only the necessary composed descriptors, presented-candidate IDs, and descriptor-to-reference dependencies. Keep preview bytes under the existing active-image budget. Creating a figure pins references; it does not publish them.
- [ ] Record candidate presentation on the actual evidence-message/model-attempt path. Reject composition of candidates never offered as pixels. Do not infer eligibility from a model name or restore a blanket `supports_vision` gate. A provider fallback that was not offered the pixels cannot compose them without a new presentation.
- [ ] Resolve cell media and sources from the session, construct existing image/group descriptors, and keep original cell order. Accept no model-authored asset URLs. Reuse the trusted native result/artifact path from Task 4.
- [ ] Extend `GroundingParser.resolve()` to admit only referenced, session-created figures; union their selected dependencies with ordinary `[[image:I#]]` selections. Do not search/replace serialized payload JSON.
- [ ] Keep both singleton and composed candidates available through final selection. A selected figure must survive even when its corresponding singleton is not selected; shared media references are retained once.
- [ ] Verify that no composed image/group is emitted as an early public `rich_items` candidate. Run web grounding/image/tool tests and commit.

### Task 6: Preserve visual state across suspension, Continue, and degradation

**Modify:** `WebResearchSession`, `app/ai/workflow/specialists.py`, `app/ai/workflow/middleware.py`, `app/ai/workflow/state.py`, `app/ai/workflow/continuation.py`, existing graph resume methods, `app/services/message_service.py`, `app/services/conversation_fork_metadata.py`.

**Tests:** Extend `tests/test_web_research_continuation.py`, `tests/test_workflow_continuation.py`, `tests/test_continuation_epoch_chain.py`, `tests/test_message_service_web_image_externalization.py`, `tests/test_conversation_fork_media.py`; create `tests/test_source_figure_lifecycle.py`.

**Interface:** Add `snapshot_visual_state() -> dict[str, Any]`, `restore_visual_state(snapshot: dict[str, Any]) -> None`, and `async suspend() -> None` to the existing `WebResearchSession`. Declare `SpecialistState(AgentState)` in the existing middleware module with one optional `visual_state: dict[str, Any]` channel; use it as the existing evidence middleware's `state_schema`. Add one checkpoint-safe `carried_visual_state` field to parent workflow state for epoch transfer. These carry the same snapshot at their respective boundaries; they are not additional registries.

Snapshot version 1 contains scope/turn identity, issued candidate counter, public source bindings, presented IDs, protected reference IDs, composed descriptors/dependencies, and carried static descriptors. Allowlist these fields instead of dumping `PreparedImage` or the session. Keep one current snapshot per checkpoint boundary and bound it by the existing source/candidate limits plus the six-visual limit; reject an oversize optional composition instead of growing state indefinitely. It contains no bytes, credentials, upstream asset URLs, or client-specific downgrade.

- [ ] Reproduce compose → image-window eviction → reference loss, and compose → approval interrupt → session abort. Require passing tests before enabling public figure creation.
- [ ] Reuse `_retain_active()`, `finish()`, and `abort()`: eviction releases only unpinned references; final selection releases unused dependencies; terminal abort releases unpersisted dependencies; suspension retains checkpointed dependencies within the existing pending-reference TTL.
- [ ] Persist `visual_state` in the specialist's declared state after tool completion, using the existing tool middleware's state-update return path. Extend `WebEvidenceMiddleware` with an after-model state update to capture the actual presentation records. Ensure that update is checkpointed **before** the existing approval middleware's interrupt node runs. Restore from this channel before the next model call or resumed tool call. Prove ordering with a real checkpointer, a fresh factory/session, and an approval interrupt; an in-memory assignment is insufficient.
- [ ] On epoch rollover transfer the same bounded snapshot to parent `carried_visual_state`, and restore it when the next specialist is built. Keep selected public items in `OutcomeProvenance.rich_items` and `ContinuationPausePayload.rich_items`; keep unselected private dependencies out of that public payload. Do not also embed full snapshot copies into every tool result or message.
- [ ] Restore into the next invocation under the same owner/conversation/turn authorization. Keep the candidate counter monotonic; never rebind a retired `I#`. Restore metadata/dependencies eagerly and load preview pixels only when a later model attempt needs them, through the existing owned media reader.
- [ ] Keep quota/dedup restoration in the existing research accounting path. Do not conflate that payload with visual ownership or mint fresh research quotas during restore.
- [ ] Ensure every claimed public descriptor is in `OutcomeProvenance.rich_items` before constructing a budget-pause payload. Selected partial visuals persist before Continue is advertised; unselected but checkpointed figures remain private.
- [ ] Extend existing `_externalize_group_cells()` and fork `_available_rich_items()` to remove comparison-wide captions/claims when cells are lost. Preserve surviving labels/sources and an explicit degraded state; remove the figure if none survive.
- [ ] Verify pending Stop, explicit Stop, disconnect, approval Resume, manual/automatic Continue, double redemption, process-recreated session, TTL expiry, and fork. No reference may outlive the existing owned-media cleanup guarantees because it was composed.
- [ ] Commit only after lifecycle tests pass. Keep native figure tools unbound until this task and Task 7 are complete.

### Task 7: Share capability projection across streaming and saved answers

**Modify:** `app/services/event_streaming/ai_sdk_projection.py`, `ai_sdk_v6.py`, `internal_sse.py`, `app/core/rich_response.py`, `app/services/ai_service.py`, existing route stream wrappers.

**Tests:** Extend `tests/test_ai_sdk_v6_stream_contract.py`, `tests/test_ai_sdk_assistant_ui_compat.py`, `tests/test_ai_sdk_wire_fixtures.py`, `tests/test_rich_response_streaming.py`, `tests/test_generation_transport_parity.py`.

**Interface:** Extend `project_ai_sdk_message_for_capability(message, *, inline_rich_response_v1, answer_visuals_v1=False, source_figures_v1=False)` in place; reuse it for internal SSE terminal/partial payloads as well. Extend `RichMarkerStreamFilter` to apply the same descriptor/fallback decisions to split chunks, with proper code-fence/inline-code preservation. No parallel projector or parser.

- [ ] Add the complete capability matrix using the same canonical fixtures. Rich+static preserves static markers/descriptors; rich without static replaces only static markers with derived Markdown; rich without figures gets legacy image/group payload plus readable source-linked figure fallback; non-rich gets text fallback plus the existing selected-media compatibility delivery (AI SDK file parts on that transport). Canonical objects remain unchanged.
- [ ] Replace repeated referenced new items once per ID. Preserve code examples of markers, ordinary rich items, prose, and existing malformed-marker behavior. Strip unsupported enriched fields completely.
- [ ] Use existing per-stream state inside `AIService._map_workflow_stream()` to hold new static candidates until their authorized marker is observed; emit the existing `rich_items` event at that point. Ordinary tool completion alone must not publish an unused static visual.
- [ ] Keep figure candidates private until final/partial selection. Extend existing adapter state and marker filter for capability-safe text/preview handling; do not expose marker fragments, dedicated tool payloads, raw image candidates, or duplicate generic `data-tool-render` output.
- [ ] Apply shared projection to authoritative deltas, web previews, saved partials, interrupts, completion metadata, history, and `data-answer-reconcile`. If a draft abandons a visual, the terminal registry replaces the transient registry and removes it.
- [ ] Wire new current-request flags on Resume/Continue while leaving canonical carried state intact. Confirm `streamWebAnswerV1` independence and base-rich compatibility.
- [ ] Generate Python SSE fixtures and parse them with the real pinned AI SDK through the existing Node wire tests. Activate the request-local native tool binding from Tasks 4–5 only after the lifecycle tests and this capability matrix pass. Commit when both transports converge with history.

Representative immutable-projection assertion:

```python
canonical = json.loads(Path("tests/fixtures/rich_answer_visuals/bars.json").read_text())
message = {"content": f"Before\n<!--rich:{canonical['id']}-->\nAfter",
           "metadata": {"rich_items_version": 1, "rich_items": [canonical]}}
original = copy.deepcopy(message)
projected = project_ai_sdk_message_for_capability(
    message, inline_rich_response_v1=True,
    answer_visuals_v1=False, source_figures_v1=False,
)
assert "120" in projected["content"] and "<!--rich:" not in projected["content"]
assert projected["metadata"]["rich_items"] == []
assert message == original
```

### Task 8: Render in Streamlit and repair its Continue consumer

**Modify:** `app/ui/rich_response.py`, `demo.py`.

**Tests:** Extend `tests/test_demo_rich_response.py`, `tests/test_demo_generation_controls.py`, `tests/test_demo_paused_turn_offers_continue.py`, `tests/test_demo_stream_rendering.py`, `tests/test_hitl_demo_panel.py`.

**Interface:** Keep `build_rich_response_view()` and `_StreamingRichResponseRenderer` as the owners of ordering, registry updates, and final replacement. Reuse `build_inline_image_html()` / `build_inline_image_group_html()` for figures; extend the existing rich-item dispatch for the three static variants.

- [ ] Add fixture-driven tests that fail for unsupported static types, missing figure labels/captions, single-cell degradation, and Continue dropping rich/interrupt events.
- [ ] Render steps as an ordered static sequence; bars with the validated common zero-based scale, exact labels/values/units and visible assumptions; tables with readable assessment labels/reasons. Reuse current chart/table primitives where they preserve the contract; extend them narrowly rather than creating another generic chart engine. No widget iframe or Canvas artifact is involved.
- [ ] Extend existing figure HTML to show escaped labels and captions once, retain protected loading/source links/lightbox, stack on narrow screens, and remove comparison claims on a cell's runtime load failure. Use semantic light/dark theme tokens.
- [ ] Advertise all three flags consistently on Send, Resume, and `build_continue_request()`, and on applicable history reads. Keep Streamlit on `/messages/continue`, not the AI SDK parser path.
- [ ] Extend `consume_continue_stream()` to process live rich updates, sources, persisted partials, and approval interrupts using the existing renderer/state operations. Call renderer finalization for `message_end`, `complete`, and interrupt partials; preserve pending approval data for the existing approval panel.
- [ ] Reconcile snapshot/history after a refused or interrupted command. Do not leave the spent Continue offer clickable or infer lifecycle completion from socket closure. Preserve current command keys, version checks, and previous partial messages.
- [ ] Copy/export uses typed payload fallback, with the same values and citations as delivery fallback. Do not copy raw markers or ask the model to generate separate export prose.
- [ ] Run Streamlit rich/stream/Continue/HITL regression tests and visually verify live output plus a history reload. Commit the local UI integration.

### Task 9: Frontend handoff, live acceptance, and cleanup

**Modify:** `plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md`, `plans/AI_SDK_FE_CONTRACT.md`, `plans/CONTINUE_AI_SDK_FE_CONTRACT.md`, `docs/frontend/answer-streaming.md`; update the status/dependency notes in the earlier visual review.

**Deliver:** One fixture set under `tests/fixtures/rich_answer_visuals/` and component/capability examples in the existing rich-item contract. Avoid another overlapping contract document.

- [ ] Document exact static schemas, enriched image fields, flag matrix, evidence/fallback rules, terminal replacement, failures, and history/Resume/Continue behavior. Remove the now-obsolete “planned/unimplemented” and Streamlit-gap statements only after their tests pass.
- [ ] Run disposable live cases for the user's exact bars and volcano prompts, plus steps, assessment table, explicit standalone Canvas, text-only, and unsupported-capability requests. Record route, tool identity, item type, source binding, marker placement, and stream/history equality.
- [ ] For the bars case require Chat, `answer_visual`/`bars`, values 120/180/300 ms, scale 0–300, and no Canvas or live widget. For volcanoes require Chat/Search, sourced labelled figure, typed steps, and no ASCII block substituted for the steps item on a capable client.
- [ ] Repeat focused live routing with fresh and previous-Canvas contexts and the configured router model. Existing fake-model tests are structural checks, not the semantic acceptance gate.
- [ ] Compare capability-off output with a deterministic server fallback and verify canonical metadata is identical. Repeat through Resume, Continue, history reload, and a fresh session that restores the checkpoint.
- [ ] Give the external FE developer the shared fixtures and updated existing contract. That FE repository is not in this workspace: full external FE completion requires its actual component/transport integration and recorded acceptance, not merely a successful backend wire test. Do not scaffold a substitute frontend here.
- [ ] Review the final diff for duplicate formatters, wrappers, registry/state duplication, unused legacy branches, dependency additions, and unrelated edits. Every new production function must have an owner and concrete consumer in this plan. Remove any that only rename/delegate to another function.
- [ ] Commit documentation/fixtures and report backend, Streamlit, wire-parser, and external FE acceptance separately, including any unverified portion.

## Verification commands

Run each task's focused tests first. New tests should demonstrate the missing behavior before implementation; do not add tests that merely duplicate implementation conditionals.

Offline baseline and regression batch (existing files):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_prompts_visual_strategy.py tests/test_prompts_media_capability.py tests/test_routing_inventory.py tests/test_routing_context.py tests/test_routing_service.py tests/test_tool_result_rendering.py tests/test_tool_execution_rendering.py tests/test_rich_response_contract.py tests/test_rich_response_metadata.py tests/test_ai_sdk_resume_continue_capabilities.py tests/test_demo_rich_response.py tests/test_demo_generation_controls.py tests/test_web_research_continuation.py tests/test_conversation_fork_media.py -q
```

Additional acceptance batch once the three planned test files exist:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_answer_visual_tools.py tests/test_source_figure_composition.py tests/test_source_figure_lifecycle.py tests/test_selected_image_reaches_the_model.py tests/test_workflow_continuation.py tests/test_continuation_epoch_chain.py tests/test_generation_transport_parity.py tests/test_message_service_web_image_externalization.py tests/test_ai_sdk_v6_stream_contract.py tests/test_ai_sdk_assistant_ui_compat.py tests/test_demo_paused_turn_offers_continue.py tests/client_backend/test_messages.py -q
```

Real SDK parsing, reusing the installed test package and Python fixture writer:

```powershell
$env:AI_SDK_FIXTURE_DIR = Join-Path $env:TEMP 'chatbot-visual-wire-fixtures'
.\.venv\Scripts\python.exe -m pytest tests/test_ai_sdk_wire_fixtures.py -q
npm --prefix tests/ai_sdk_wire test
Remove-Item Env:AI_SDK_FIXTURE_DIR
```

Live semantic routing, using an existing authorized account with configured router credentials:

```powershell
$routingUserId = Read-Host 'Account UUID for the disposable visual acceptance run'
1..3 | ForEach-Object {
    .\.venv\Scripts\python.exe scripts/evaluate_routing.py --dataset eval/routing/inline_visuals_v1.jsonl --user-id $routingUserId --concurrency 1 --output "eval/routing/inline-visuals-run-$_.json"
}
.\.venv\Scripts\python.exe scripts/evaluate_routing.py --dataset eval/routing/golden_v1.jsonl --user-id $routingUserId --output eval/routing/visual-repair-golden-report.json
```

Database-backed lifecycle checks must use `TEST_DATABASE_URL`, never the application database. Reuse the repository's existing isolation gates. Live answer checks use disposable app conversations and do not seed/delete database rows through pytest.

## Completion criteria

- Both capability flags are implemented and independently negotiated on every relevant entry point.
- The exact reported prompts produce their intended inline item types on capable clients. Explicit standalone artifacts still route to Canvas.
- No Canvas/widget substitution, ASCII-as-component substitution, raw descriptor JSON, duplicate rich/generic result, or unselected source image leaks into a capable answer.
- Static descriptors, composed media ownership, evidence, and stable IDs survive supported suspension/resume paths; lost comparison cells cannot leave misleading captions.
- Stream, saved partial, completion, history, downgraded client, and copy/export agree on values and citations.
- Streamlit passes live/history/Continue/approval acceptance. External FE acceptance is explicitly recorded or explicitly outstanding.
- No new production module beyond the one authoring-tool module; no dependency, migration, parallel service, or generic helper framework.

## Plan self-review

- Full selected scope is covered: routing (Task 1), contracts (2), negotiation (3), static authoring/evidence (4), figure composition (5), lifecycle (6), shared delivery (7), Streamlit (8), external handoff/acceptance (9).
- Creation is kept separate from selection/publication; tools are not fully enabled until lifecycle and projection checks pass.
- Legacy router-prompt tests are moved to the live prompt; live semantic evaluation is a separate release check.
- All new production entry points are named above; existing ownership is retained. Source-figure media remains in the current web-image system.
- This file proposes implementation. No runtime code, database content, provider configuration, or deployment was changed while preparing it.
