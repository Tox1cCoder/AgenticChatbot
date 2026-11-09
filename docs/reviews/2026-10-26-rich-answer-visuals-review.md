# Rich answer visuals: revised backend and frontend implementation plan

> For implementation: use `superpowers:executing-plans` to work through the outstanding tasks in dependency order. This document records the reviewed design and subsequent verification; future implementation tasks do not themselves authorize application changes.

Date: 2026-10-26

Scope: the server backend and contract for the separate frontend, as selected by the user. Streamlit is inspected as an existing consumer of that contract. The separate frontend source remains unavailable. The supplied LangSmith conversation and its user's stored model settings have now been inspected; see the final verification section.

Status: revised after the codebase review on 2026-10-26. All seven original review findings are addressed in the proposed design. Subsequent local candidate-preservation repairs are recorded below; the proposed new visual types remain unimplemented.

Rechecked: 2026-10-29 against the current codebase after the report that even explicit image requests produce links. Task 0 has trace, stored-catalog, and runtime-resolution evidence that `supports_vision=False` suppressed retrieved images before the answer-model call. The user subsequently requested removal of the allowlist. Both vision name checks and the web-evidence capability gate have now been removed; live delivery verification remains outstanding.

Subsequent repair: the user authorized code fixes after the trace investigation, then requested removal of the model-specific capability patch. That patch and its helper/tests have been removed. Only the model-independent candidate merge and presentation-allowlist repairs remain. The proposed new visual types and external frontend acceptance work remain future work.

**Goal:** support compact static visuals and labelled source figures without losing selected images, duplicating tool renders, or weakening readable fallback answers.

**Architecture:** retain Markdown, inline rich markers, the existing registry, and AI SDK v6 transport. Add a bounded static payload and extend the existing image payloads; share candidate construction, evidence resolution, and capability projection across live and saved answers.

**Tech stack:** Python, Pydantic, the existing specialist/tool workflow, protected web-image services, AI SDK v6 SSE, and the separate frontend's component renderer.

## Global constraints

- Keep the AI SDK v6 wire contract, inline marker grammar, and existing protected media routes.
- Persist canonical validated content; capability negotiation changes delivery, not stored facts.
- Add no widget session, Redis state, WebSocket, mandatory planning-model call, or new charting dependency for these static visuals.
- Extend existing image types and lifecycle services; do not create a parallel image store or expose upstream asset URLs.
- Keep new tool bindings behind the existing rollout gate until their capability projection and client acceptance checks pass.
- First delivery covers public answering specialists; private Planning workers retain their existing image-release policy.

## Recommendation

First restore and verify the existing single-image answering path. The new static types and figure-composition tool are not prerequisites for displaying a retrieved image. A link-only answer can originate before rendering, and adding frontend components alone cannot repair a missing retrieval request, missing model selection, or overwritten image descriptor. Treat that repair as an independently verifiable deliverable before expanding the visual contract.

Extend the existing inline rich-response system with lightweight static answer visuals and model-authored figure composition. Keep Markdown for ordinary prose and tables, HTML widgets for interactions, and canvas for standalone deliverables.

Limit the first static family to `steps`, `bars`, and `comparison_table`. Defer `callout`: none of the supplied examples requires it. Preserve existing chart/table `tool_render` output and reuse suitable rendering primitives; do not build a second general chart engine or migrate existing tool results.

The examples supplied by the user mostly need compact static presentation: an ordered sequence, a comparison table, proportional bars, a table with status badges, and a pair of source images. They do not require an interactive micro-app for every answer.

## What already exists

| Capability | Evidence | Implication |
| --- | --- | --- |
| Typed inline rich items | `app/core/rich_response.py:77` | Images, image groups, widgets, tool renders, canvas, citations, and resource links already have a shared registry. |
| Chart and table tool renders | `app/ai/tool_result_rendering.py:_infer_render_type`, `demo.py:_render_table_tool_result`, `demo.py:_render_chart_tool_result` | Charts and tables are not missing wholesale. The new family must earn its scope through stricter content semantics, evidence, fallback, and compact presentation. |
| Ordered placement in Markdown | `app/ui/rich_response.py:330` | Existing `<!--rich:<id>-->` references provide the insertion points needed for article-like answers. |
| Incremental rich events | `app/services/event_streaming/ai_sdk_v6.py:773` | Safe non-image items can already travel as `data-rich-items` upserts. |
| Final registry reconciliation | `app/ui/rich_response.py:403`, `plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md` | Final metadata replaces transient state. Stream and history must converge. |
| Images presented to the answering model | `app/ai/web_research/model_context.py:25`, `app/ai/web_research/service.py:886` | A vision-capable answer model receives candidate labels and actual preview pixels. |
| Source and image token resolution | `app/ai/web_research/grounding.py` | `[[source:S#]]` and `[[image:I#]]` resolve against the current turn rather than arbitrary model-generated URLs. |
| Protected media delivery | `plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md:232` | The frontend contract already describes authenticated loading and media lifecycle. |
| HTML micro-apps and separate canvas | `app/ai/mcp_servers/widgets_server.py:111`, `app/ai/agents/canvas_agent.py` | Interactive and standalone artifacts have existing paths. |

## Findings

### 1. There is no dedicated representation for compact static answer visuals

`RichItemType` has no dedicated steps or semantic comparison-table variants. Existing `tool_render` payloads already recognize chart-shaped and table-shaped results. A composition tool could aggregate several sources into those existing shapes; aggregation alone does not justify another rich-item type.

The remaining gap is a bounded answer-content contract with validated units/scales, evidence references, semantic assessment labels, and a derived Markdown fallback. This plan chooses a small `answer_visual` family for those guarantees while reusing the existing transport and appropriate presentation primitives.

### 2. Visual guidance strongly favors interactions

`VISUAL_STRATEGY_SNIPPET` in `app/ai/prompts.py:60` usefully tells answering agents to consider visuals. Its authored-visual guidance focuses on moving parts, controls, sliders, animation, and micro-apps. The widget tool describes the same experience, and `app/services/widget_contract.py` requires a height between 260 and 960 pixels.

That is a poor fit for the small step card and two-row bar comparison in the examples. The likely quality issue is a mismatch between the intended visual and the available authored output, rather than missing general encouragement to use visuals. This is an inference from the code, not a measured production failure rate.

### 3. Web retrieval and answer figure composition are separate missing links

`ResearchRequest.visual_intent` defaults to `none` (`app/ai/web_research/contracts.py:61`). Images therefore depend on the answering model requesting them, receiving usable candidates, and selecting them. Keep that deliberate relevance decision, but evaluate whether it misses concrete visual opportunities in real answers.

More specifically, `app/ai/web_research/service.py:769` creates an individual `ImageRichItem` for each prepared web image. `GroundingParser` resolves each selected image independently. The inspected web path has no step that turns selected candidates into a labelled comparison figure.

The web item includes a description and alt text but does not populate `payload.caption`. At the same time, the prompt tells the model not to add a Markdown caption after an image marker because the renderer owns the footer. The current image-group schema has no group caption or per-cell visible label. The Streamlit group renderer uses descriptions as alt text and displays source links as the cell footer (`app/ui/rich_response.py:105`).

Consequently, supporting the `image_group` type is insufficient by itself to reproduce the final screenshot reliably. The answer model needs a validated way to compose inspected images, labels, and an explanatory caption together.

### 4. Client negotiation can hide otherwise available features

The normative contract requires `inlineRichResponseV1: true` for chat, resume, Continue, and the corresponding history query. Compatibility clients receive marker-free text and selected image file parts; widgets and canvas rely on the rich registry. The current projector passes all rich items through for a rich-capable client and otherwise strips markers; it does not perform per-feature fallback substitution.

Verify this negotiation and `onData` handling in the separate frontend before changing the backend. Its source is outside this workspace, so this review cannot establish whether that client currently follows the contract.

### 5. Current evaluation measures contract integrity more than answer quality

`app/evaluation/web_research/metrics.py:6` measures source membership, image membership, expected zero-image behavior, and source stream/history parity. Those checks do not establish that an image depicts the intended detail, that a figure is useful, or that its labels and scales are accurate. The checked-in web-research evaluation file contains two cases.

Keep those checks and add quality cases based on the supplied examples. Passing schema and transport tests cannot establish ChatGPT-like visual answer quality.

### 6. Existing integration paths need targeted fixes before new types are enabled

| Review finding | Current code behavior | Required change |
| --- | --- | --- |
| Selected images could be overwritten | At the original reviewed revision, `app/ai/graph.py:_merge_tool_artifacts` replaced `_rich_item_candidates` with context candidates. The current working tree merges by stable ID and preserves grounded descriptors/order. | Retain the existing local repair and regressions; shared candidate construction remains outstanding. |
| Figure markers do not select their constituent images | `GroundingParser` selects through `[[image:I#]]`; `WebResearchSession.finish()` releases the other references, and `_retain_active()` releases evicted candidates. | Resolve referenced figures to their image dependencies and retain pending dependencies across searches and supported suspension. |
| New labels fail a separate selector contract | `app/core/rich_image_selection.py:_GROUP_CELL_KEYS` rejects unknown cell fields; single-cell conversion replaces the whole group payload. | Update schema and selector together, with explicit label/caption preservation and degradation. |
| Stream and persistence construct candidates separately | `app/ai/tool_execution.py:_attach_rich_candidates_to_artifact` and `app/services/ai_service.py:_build_tool_end_rich_items` have separate construction paths. | Use one validated candidate builder and suppress a generic render when a dedicated visual owns the output. |
| Unsupported clients have no content fallback | `project_ai_sdk_message_for_capability()` strips markers rather than replacing them with content. | Apply per-feature projection to text, metadata, events, and terminal reconciliation. |
| Nested visual claims bypass text grounding | `GroundingParser.resolve()` operates on answer text, not visual cells or captions. | Resolve structured evidence references before publishing, and include those citations in fallback text. |
| Proposed scope overlaps existing renderers | Chart/table tool renders exist; `callout` has no motivating example. | Preserve existing render paths, limit the new family, and defer callouts. |

## Approaches

| Approach | Benefit | Trade-off |
| --- | --- | --- |
| Prompt and tool-description changes only | Smallest change; can improve Markdown tables and tool selection | No reliable representation for badges, compact bars, figure labels, or consistent styling. |
| Composition tool returning existing chart/table `tool_render` shapes | Smallest transport change; can aggregate multiple sources and reuse existing renderers | Would still need bounded validation, evidence, fallback, steps/badge semantics, and capability filtering. Prefer this path for ordinary charts/tables already covered by the current contract. |
| **Extend rich items with static answer visuals and figure composition** | Reuses placement, streaming, persistence, and protected media; frontend controls consistent rendering | Requires a bounded schema, generation tool, capability negotiation, and frontend components. Recommended. |
| Generate HTML widgets for almost all visuals | Maximum layout freedom through an existing tool | More generated code, iframe overhead, variable styling and accessibility, and a minimum height unsuitable for compact examples. |

The repository intentionally removed structured *live widgets* in favor of HTML-only micro-apps. The recommended static answer components would be a separate rich-item family, with no widget session, Redis state, or WebSocket. This avoids reversing the interactive-widget decision.

## Proposed answer composition

Keep the existing answering agent responsible for the final response. Add a compact decision policy, rather than another mandatory planning model call:

1. Establish the answer and supporting evidence.
2. Choose the smallest presentation that helps the reader: prose, Markdown table, static visual, source figure, interactive widget, or canvas deliverable.
3. If source images matter, request a concrete visual subject and inspect the returned candidates.
4. Create a static visual or compose a source figure through a validated tool when needed.
5. Write useful prose with the returned rich markers next to the relevant explanation.

The backend validates the data, assigns IDs, publishes safe descriptors, and persists the final registry. The frontend chooses the actual typography, spacing, colors, responsive layout, and component implementation.

This fits the established tool-result-to-component pattern described in the [AI SDK generative UI documentation](https://ai-sdk.dev/docs/ai-sdk-ui/generative-user-interfaces). Its [custom streaming data documentation](https://ai-sdk.dev/docs/ai-sdk-ui/streaming-data) also distinguishes transient notifications from persisted message parts. Retain the repository's tested AI SDK v6 wire contract; these references are architectural guidance, not an SDK migration proposal.

## Match the supplied examples

| Example | Proposed representation | Content requirement |
| --- | --- | --- |
| Edge/Google sequence | `answer_visual`, variant `steps` | Ordered instructions, optional short notes, and neutral sequence icons. A check icon must not imply the app executed the step. |
| Game edition comparison | Markdown table | Consistent criteria, exact identifiers, and citations beside claims that need evidence. |
| Clock/voltage/FPS comparison | `answer_visual`, variant `bars` | Shared units and scale; numerical values; explicit measured, calculated, or illustrative status. |
| Tweak assessment table | `answer_visual`, variant `comparison_table` | Text-labelled semantic badges and a reason for each assessment. Ordinary comparisons can remain Markdown. |
| Side-by-side graphics screenshots | Composed source figure | Selected inspected image IDs, per-cell labels, source links, and one explanatory caption. |

For the FPS example, do not infer real performance directly from core clock. If a teaching example assumes proportional scaling, identify that assumption visibly. A 100 vs 101 FPS chart must preserve the actual one-percent relationship on a shared scale.

## Backend contract direction

### Static answer visuals

Add an `answer_visual` rich-item type with a discriminated payload for `steps`, `bars`, and `comparison_table`. Use ordinary Markdown instead whenever it expresses the content adequately. The model supplies content and semantic status, not arbitrary HTML, CSS, JavaScript, component names, or executable expressions.

A server-owned `create_answer_visual` tool validates the payload, assigns a stable rich ID from the logical turn and tool-call identity, registers it through the existing tool-artifact path, and returns its marker. Replaying the same successful call reuses its ID and descriptor; a revised payload receives a new ID. Do not ask the model to emit the entire assistant response as one JSON object; preserve progressive Markdown and inline placement.

Each visual has server-derived `fallback_markdown` built from the same validated content and resolved evidence as the component. The model does not supply a second independent fallback. Use `inline_only`: a created visual becomes public only when the answer references it. Enforce this policy in final registry construction, which currently filters unreferenced images but admits non-image candidates without checking `inline_only`.

Initial payload limits are at most 12 steps, 12 bars, or 12 rows by 6 columns; titles and labels are at most 160 characters, and notes/reasons are at most 500 characters. Allow at most six referenced static visuals per answer, at most 16 KiB of serialized public JSON per item and 64 KiB in total, including fallback and evidence. Reject oversize tool input with a concise retryable tool error; never truncate facts silently. These limits are distinct from the existing generic tool-result cap.

For bars, v1 accepts finite nonnegative values, one unit per visual, a linear scale starting at zero, and a shared maximum at least as large as every value and greater than zero. Derive the default maximum from the data; use 1 for an all-zero set. Reject mixed units, negative values, nonfinite values, and truncated axes. Require `basis` (`measured`, `calculated`, or `illustrative`); calculated/illustrative values also carry visible assumptions. Clock, voltage, and FPS require separate scales. Basis is a disclosure, not proof: measured claims still need supporting evidence.

Comparison assessments use text labels with a bounded status enum (`positive`, `neutral`, `caution`, `negative`) and a visible reason. Colors are frontend theme tokens. Steps use sequence numbering, never an execution-success state inferred from an instruction.

### One candidate pipeline and one final registry

Create `app/ai/answer_visual_tools.py` for the creation tool and `app/core/answer_visuals.py` for payload-specific validation, evidence binding, and deterministic fallback formatting. Keep the public rich-item union in `app/core/rich_response.py`. Extend existing tool bindings for the answering chat/search/RAG specialists; follow the current tool allowlists for custom agents instead of enabling new tools indiscriminately.

Use a single candidate builder from both `_attach_rich_candidates_to_artifact()` and `_build_tool_end_rich_items()`. Only the server-owned creation/composition tools may supply these dedicated descriptors. The adapter returns either the dedicated record or the existing generic `tool_render` candidate, never both for one result. Do not infer a new answer visual from arbitrary third-party JSON.

Merge grounded response candidates and tool-context candidates before final placement/metadata construction. Deduplicate identical records by ID; conflicting content for one ID is an invalid optional visual, not a last-write-wins update. Keep order stable, with final marker order controlling inline presentation. Do not overwrite grounded images or erase existing widget/canvas records. Apply the same merged set to marker authorization, provenance, persistence, and terminal reconciliation.

Static descriptors are small enough to retain intact in the internal artifact envelope even if the model-facing tool result is offloaded. Copy the admitted records into `OutcomeProvenance.rich_items` so continuation is not dependent on recovering them from prose.

### Ground evidence inside payloads

Tool inputs attach `evidence_refs` to factual bars, table cells/assessments, steps, and factual captions. Each reference is `{kind: "web", id: "S#"}` or `{kind: "rag", id: "<admitted evidence id>"}` and resolves against the current turn's server-owned web/RAG evidence. General instructions and clearly illustrative data may have no source; externally measured claims require evidence. Calculations disclose their assumptions and cite externally sourced inputs.

Resolve references before generating a public descriptor. Public evidence entries contain server-resolved citation targets and display information, not an unresolved turn-local token or a model-authored URL. Reuse the existing web/RAG grounding logic and source registries. Validate membership, not the truth of a claim; semantic support remains part of answer-quality evaluation.

Include sources used only inside a referenced visual in the final source/provenance set and citation-repair decision. Emit citations adjacent to the corresponding claims in both the component and its derived Markdown. Reject unknown references with an actionable tool result; do not publish a plausible-looking visual with detached citations. Do not run regex substitution over serialized JSON.

### Source figures

Add a server-owned `compose_source_figure` operation over the current turn's inspected candidate IDs. Inputs describe a single figure or ordered comparison, cell labels, alt text, and a caption. Resolve IDs server-side and reuse protected media references. Bind each cell to its own source. Use the existing `image`/`image_group` family, not a third image type or a second image store.

Extend `ImageGroupCell` with `label` and `alt_text`; extend `ImageGroupPayload` with `caption`, resolved caption evidence, and derived `fallback_markdown`. Single composed images use `ImagePayload.caption` plus the same supported label/evidence/fallback fields. Keep alt text separate from captions. Limit composition to one to four unique inspected images, with labels at most 160 characters and captions/alt text at most 500 characters. Preserve source image aspect ratios by default; a before/after comparison needs comparable views, not merely two pictures about the same topic.

Update `_GROUP_CELL_KEYS`, `_IMAGE_PAYLOAD_KEYS`, public validation, selection, and all single-cell conversion paths together. Preserve each cell's label, alt text, source, and position. Deduplicate constituent references before comparing groups against standalone image candidates so a selected group is not stripped by an unselected singleton. Do not force labelled cells into the older schema through generic dictionary copying.

If only one candidate is available at creation, compose a single figure with a caption that describes that image. If an established comparison later loses a cell during externalization, conversation fork, or rendering, remove the comparison-wide caption and render surviving cells with their own labels/sources plus an explicit comparison-unavailable state. Never retain a before/after conclusion over a single surviving image. Apply this rule in `message_service.py`, `conversation_fork_metadata.py`, and both renderers.

Maintain the rule that unselected candidates never reach the frontend. Figure composition must not expose upstream asset URLs or allow the model to invent them.

### Figure selection, retention, and interruption

Treat creation, reference selection, and persisted ownership as distinct states:

1. Record which candidate IDs were actually presented as pixels to a vision-capable answer-model attempt in this logical turn. The server can verify presentation, not whether the model understood the image. Reject composition of unknown/unpresented IDs and composition after fallback to a non-vision model; do not rely on textual `I#` labels alone.
2. Creating a figure records its ordered candidate IDs and pins their owned media references as pending dependencies. A later search may evict candidates from the active vision window, but must not release pinned dependencies. Keep this separate from the bounded active preview window; do not retain additional preview bytes in model context.
3. At grounding/finalization, resolve referenced figure markers to their dependencies and union those IDs with directly selected `[[image:I#]]` candidates. Publish only referenced figures. Reject invented figure markers. A cell's server-bound source supplies its attribution; factual caption evidence is separately validated.
4. Finish releases unused candidates and dependencies belonging only to unused figures. Dependencies of referenced figures remain pending until the existing message-persistence path calls `mark_selected()`. Shared references are retained once and released only when no surviving selected item needs them. Preserve the existing pending-reference TTL and cleanup path for persistence failures.
5. Persist a private composition snapshot at existing workflow checkpoint/suspension boundaries: logical turn, stable figure descriptors, dependency/reference IDs, presentation records, source mappings, and a non-reused candidate-ID counter. Do not checkpoint image bytes or upstream URLs. Distinguish resumable suspension from terminal abort before calling the existing `abort()` cleanup; a resumable pause retains dependencies of checkpointed compositions within the configured TTL, including compositions created before the answer has emitted its markers. Final selection still controls public visibility.
6. Resume and Continue rehydrate that snapshot under the same user/conversation authorization and verify reference availability. Carry static descriptors as well as figures across epochs. Expired dependencies degrade using the readable failure rule; never rebind a retired `I#` to different bytes. Cancellation or terminal failure releases remaining unselected dependencies. Already persisted selections keep their existing lifecycle.

First delivery targets public answering specialists. Planning workers currently release web images at their private-output boundary (`specialists.py:invoke_worker`) and the parent strips rich markers from worker summaries. Keep that policy: do not enable figure composition in private workers or claim Planning support until parent inspection/selection has its own design and tests.

### Compatibility and lifecycle

- Add `answerVisualsV1` and `sourceFiguresV1` booleans, defaulting to false, alongside `inlineRichResponseV1` on chat, resume, Continue, and history requests. Each new capability requires the base capability and the server rollout gate. Negotiate them independently: support for bars does not imply support for enriched image payloads. Propagate the effective capabilities through API/request schemas, workflow/tool context, streaming, and history projection.
- Keep the existing rich-registry envelope version and AI SDK v6 wire protocol. Item/field compatibility is enforced by projection, not by assuming every v1 client supports new payloads. Persist canonical validated content/metadata; do not save a downgraded representation for one client.
- Extend `project_ai_sdk_message_for_capability()` into the shared per-feature projection. Replace an unsupported static marker with its derived Markdown at that position and omit its descriptor. For a base-rich client without enriched figure support, expand the figure marker into its source-linked textual fallback followed by the same marker; keep its descriptor under the same ID with an exact legacy image/group payload. Strip enriched fields from that payload. For a non-rich client, emit only the textual fallback at the marker position and deliver the underlying selected media through the existing file-part path. Do not duplicate a figure through both rich items and file parts.
- Project only referenced new items, once per ID, preserving supported rich markers and prose. Preserve code-fenced marker examples, escape table delimiters and labels in fallback Markdown, and keep existing malformed-marker handling. A fallback is content, never a recursive rich-marker expansion.
- Reuse `data-rich-items` for validated static descriptors only after the corresponding answer marker is admitted; artifact construction alone must not publish an unselected visual. Preserve the rule that image candidates and composed figures are not transient public rich items. Buffer incomplete markers and substitute fallbacks from the same admitted registry when streaming to unsupported clients.
- Apply the projection to provisional answer/preview text, completion, persisted partials, resume, Continue, history, and `data-answer-reconcile`. Capability changes between requests must not change canonical stored data. Copy/export actions expand the canonical items to the same evidence-bearing Markdown even on fully capable clients.
- Keep stable IDs, marker positions, deduplication, and final registry replacement across chat, resume, Continue, and history. The terminal projected registry replaces the transient one, including removal of abandoned visuals after a rewritten draft.
- Invalid optional visuals should leave a complete readable answer. Do not leak JSON, unresolved markers, or internal validation errors into prose.

Do not combine independently implemented fallback formatters in the stream layer and history layer. Build fallback once from validated content and share the marker-aware projection. A missing or invalid descriptor removes its optional marker; the answer policy must keep essential conclusions in readable prose so failure cannot erase the answer.

## Frontend integration contract

Deliver a component map and fixtures alongside the backend change:

- Markdown renderer for prose and plain tables.
- Native components for steps, proportional bars, and badge tables, reusing existing chart/table primitives where their behavior fits the validated contract.
- A shared figure component for single images and comparisons, with captions, source links, authenticated loading, and zoom.
- Existing sandboxed widgets for interactions and canvas for standalone artifacts.

Use semantic light/dark theme tokens, compact spacing, readable captions, and text labels alongside badge colors. Stack comparison figures on narrow screens and make wide tables horizontally scrollable. Keep the same rendering logic for streaming, resumed answers, and loaded history.

The separate frontend code must implement and visually verify these components; a backend schema alone cannot produce the requested UX. Streamlit must either implement the new components or use the shared Markdown/legacy-image fallback and advertise only the capabilities it supports.

## Implementation tasks and delivery order

Each outstanding task starts with the named regression cases, runs them to demonstrate the current gap, implements the smallest change, and reruns those checks. Commit coherent tested changes during implementation. Tasks that combine completed local repairs with outstanding work remain unchecked; the completed portions are identified explicitly below.

### Task 0: Diagnose and verify existing image answers before extending the contract

**Priority:** the supplied trace lost image evidence before the answering model because the persisted catalog marked its model as non-vision. The allowlist and web-evidence gate are now removed locally. Verify a new live answer before extending the visual contract; deterministic regressions establish image delivery to the model, not successful provider/frontend rendering.

**Inspect/extend:** `app/ai/web_tools.py`, `app/ai/web_research/service.py`, `app/ai/web_research/model_context.py`, `app/ai/web_research/grounding.py`, `app/ai/workflow/middleware.py`, `app/ai/workflow/specialists.py`, `app/ai/graph.py`, and the actual frontend consuming the AI SDK response.

The following current behaviors matter to this symptom:

| Stage | Verified current behavior | Consequence for this report |
| --- | --- | --- |
| Retrieval request | `ProductWebSearchInput.visual_intent` defaults to `none`; `_search()` enables retrieval when an intent or `image_query` is supplied. Existing prompts already encourage image retrieval. | A natural-language request for images does not itself set tool arguments. Check the actual model call; a prompt-only test does not prove retrieval happened. |
| Provider wiring | `SpecialistFactory._build()` constructs the image provider only when the agent's provider tools contain `brave_image_search`. `_call_chain()` with no providers returns empty results and no failures. | A visual search with working text retrieval and no image provider can report `success`, zero images, and no failures. This is reproducible locally and currently hides a configuration failure. |
| Model inspection | `WebEvidenceMiddleware` now injects prepared candidate labels and pixels without reading runtime capability metadata. The capability argument and private suppression reason have been removed. | Old cached false flags no longer hide web images. Verify pixels reach every model attempt, including fallback; actual provider rejection uses the existing error handling. |
| Model selection | `GroundingParser` selects only valid `[[image:I#]]` tokens. A source-only answer selects nothing; an image token without an admitted source citation is removed with `image_without_source`. The specialist's corrective call checks missing source citations, not missing requested images. | A cited links-only answer can pass current finalization without satisfying the user's image request. Relevant inspected candidates need explicit selection; do not publish arbitrary candidates automatically. |
| Candidate merge | The current local `_merge_tool_artifacts()` merges grounded/context candidates by stable ID and unions grounded selections with the presentation allowlist. | Existing regressions cover selected web images alongside tool candidates. Preserve these repairs; they cannot restore images that never reached the model. |
| Planning workers | `invoke_worker()` calls `finish(())` and clears worker images at the private-output boundary. | A Planning answer assembled from worker text/evidence does not inherit worker web images. This limitation applies to ordinary retrieved images as well as the proposed source figures. Do not claim image-request support for this route without a parent inspection/selection design. |
| Client delivery | Rich-capable clients receive final image descriptors in `data-assistant-message.data.message.metadata.rich_items`, not transient image upserts or compatibility file parts. Non-rich clients receive selected image file parts. Protected image URLs require authenticated loading. | Check terminal registry replacement, file-part handling for compatibility clients, and authenticated media responses. Checking `onData` alone is insufficient. The separate frontend remains unavailable in this workspace. |

- [ ] Capture one failing explicit-image conversation end to end: selected public specialist or Planning route; effective capabilities; `web_search` arguments; provider availability; retrieval/fetch failures; candidate count; presented candidate IDs and effective vision capability; authored selection tokens; grounding warnings; candidates before/after merge; terminal/history image descriptors; and media load outcome. Use IDs/counts and reason codes, without logging credentials, binary data, or upstream asset URLs.
- [ ] Add regressions to existing `tests/test_web_research_tool_session.py`, `tests/test_web_research_images.py`, `tests/test_selected_image_reaches_the_model.py`, and `tests/test_rich_response_prompt_inventory.py` for the gaps above. In particular, make absent image-provider wiring distinguishable from a successful search with no matching images, and cover the selected-image-plus-tool merge through final metadata, rather than only the isolated grounding path.
- [ ] Add an explicit-image answer-quality case that verifies a supported public specialist requests visual evidence and selects a relevant inspected candidate when one is available. A canned model that already emits the desired token verifies plumbing only; record a live model/provider result separately when available. Cover unrelated candidates and unavailable media without forcing irrelevant image publication.
- [ ] Require a truthful, specific limitation when an explicit image request cannot be fulfilled. Links may supplement that explanation, but must not count as successful image delivery. Distinguish no retrieval attempt, absent provider, failed fetch, non-vision answering model, no relevant candidate, and unsupported route/client in diagnostics.
- [ ] Verify one selected image through generation, terminal delivery, protected loading, and history reload in the actual frontend. Check both rich capability and compatibility file-part behavior. Record the frontend revision; do not mark this passed from backend tests alone.

**Exit criterion:** an explicit image request on a supported public answering route displays a relevant image in the answer and after reload, or gives an accurate limitation when delivery is unavailable. Complete the existing candidate-preservation repair in Task 1 as part of this deliverable. Tasks 2-5 then add new presentation capabilities; they must not be used as evidence that the original image-answer problem is fixed.

### Task 1: Preserve candidate ownership and eliminate duplicate construction

**Modify:** `app/ai/graph.py`, `app/ai/tool_execution.py`, `app/services/ai_service.py`, `app/core/response_constants.py`, and `app/ai/workflow/specialists.py`.

**Interface:** one internal dedicated-candidate builder shared by tool-artifact and tool-end handling; one stable ID-based merge of grounded and tool-context candidates. Both return existing public rich-item dictionaries validated by `rich_response.py`.

**Current partial implementation:** the stable merge and grounded-selection allowlist repair already exist in the working tree, with metadata and full specialist regressions. The shared builder, non-image `inline_only` admission, and duplicate-render work below remain outstanding.

- [ ] Add regressions to `tests/test_rich_response_metadata.py`, `tests/test_tool_execution_rendering.py`, and `tests/test_rich_response_streaming.py`: grounded image plus tool candidate both survive; replay produces one item; conflicting IDs fail closed; an unreferenced `inline_only` item is absent; one dedicated output does not also append a generic render.
- [ ] Replace metadata overwrite with the shared merge and make final registry admission respect `inline_only` for non-image candidates as well as images. Exercise this with the existing non-image schema before adding the new type in Task 2.
- [ ] Route both construction paths through the shared builder and preserve admitted descriptors in outcome provenance. Existing widget/canvas append behavior must still pass its tests.

### Task 2: Add bounded static content and grounded fallback

**Create:** `app/core/answer_visuals.py`, `app/ai/answer_visual_tools.py`, `tests/test_answer_visuals.py`, and `tests/test_answer_visual_tools.py`.

**Modify:** `app/core/rich_response.py`, `app/ai/web_research/grounding.py`, `app/ai/workflow/finalization.py`, answering-agent tool bindings, and `app/ai/prompts.py`.

**Interface:** `create_answer_visual` accepts the discriminated content/evidence input described above and returns a stable marker; the internal artifact carries its validated public descriptor. The same resolved content generates `fallback_markdown` and the visual's source dependencies.

- [ ] Add tests for each variant and its exact fallback; finite-value/scale/unit limits; row/column and byte bounds; unknown evidence; visual-only citations; escaped Markdown; replayed calls; and unsupported or excluded tool bindings.
- [ ] Implement the three variants, deterministic evidence-bearing fallback, and the shared candidate adapter. Preserve existing generic chart/table outputs.
- [ ] Integrate nested evidence with final source/provenance publication and citation repair; update prompts to select the smallest useful format and keep essential conclusions readable when visuals fail.

### Task 3: Implement capability projection before enabling the tools

**Modify:** `app/api/ai_sdk.py`, `app/schemas/message.py`, `app/schemas/workflow.py`, `app/schemas/generation.py`, `app/ai/schemas.py`, `app/ai/tool_context.py`, `app/services/ai_service.py`, `app/services/message_service.py`, `app/services/event_streaming/ai_sdk_projection.py`, and `app/services/event_streaming/ai_sdk_v6.py`.

**Docs/tests:** `plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md`, `docs/frontend/answer-streaming.md`, `tests/test_ai_sdk_resume_continue_capabilities.py`, `tests/test_ai_sdk_assistant_ui_compat.py`, `tests/test_ai_sdk_v6_stream_contract.py`, `tests/test_rich_response_streaming.py`, and the existing `tests/ai_sdk_wire` fixtures.

**Interface:** the existing message projector receives the effective base/static/figure capabilities and returns a projected copy. Streaming adapters consume the same projected descriptors/fallbacks, without mutating the canonical message.

- [ ] Add fixtures for all valid base/static/figure capability combinations, mixed supported items, repeated and split markers, code fences, missing descriptors, and no raw unsupported descriptors or tool JSON reaching answer prose.
- [ ] Implement the request propagation, projection, selected-only stream admission, and terminal replacement rules above. Ensure Continue and interrupted partials use the same behavior as chat and history.
- [ ] Verify capable generation followed by legacy history loading and a legacy resume of a capable turn. Confirm canonical metadata stays unchanged and copy/export fallback retains values and citations.

### Task 4: Compose source figures with safe media lifetime

**Create:** `app/ai/source_figure_tools.py`, `tests/test_source_figure_composition.py`, and `tests/test_source_figure_lifecycle.py`.

**Modify:** `app/core/rich_response.py`, `app/core/rich_image_selection.py`, `app/ai/web_research/service.py`, `app/ai/web_research/model_context.py`, `app/ai/web_research/grounding.py`, `app/ai/workflow/specialists.py`, `app/ai/workflow/contracts.py`, `app/ai/workflow/continuation.py`, `app/ai/schemas.py`, `app/services/message_service.py`, and `app/services/conversation_fork_metadata.py`.

**Interface:** `compose_source_figure` returns an existing-family image/group descriptor and marker, plus a private ordered dependency record. The session resolves selected figure markers to media references; checkpoint state restores those dependencies and candidate identity without storing binary data.

- [ ] Add tests proving labelled cells survive selection, a singleton preserves its own metadata, a selected group survives competing unselected singletons, and an unselected group remains private.
- [ ] Add lifecycle cases: compose then search again; select by figure marker alone; two figures sharing a reference; unused figure cleanup; cancellation; failed persistence; pause/Continue; process restart from checkpoint; expired pending media; and vision-to-nonvision fallback.
- [ ] Implement pending dependency retention and selected dependency resolution before enabling the composition tool. Keep private Planning-worker figure composition disabled.
- [ ] Extend partial-failure tests in `tests/test_message_service_web_image_externalization.py` and `tests/test_conversation_fork_media.py`; remove misleading comparison captions when cells disappear. Verify protected ownership and no upstream URL exposure.

### Task 5: Integrate clients, fixtures, and answer-quality evaluation

**Modify:** `app/ui/rich_response.py`, `demo.py`, `docs/frontend/rich-image-rendering.md`, and `app/evaluation/web_research/` as needed. Add rendering regressions to `tests/test_demo_rich_response.py`. Supply shared JSON fixtures under `tests/fixtures/rich_answer_visuals/` for backend tests and the external frontend.

**External dependency:** the separate frontend repository is unavailable here. Its integration is a required delivery task, not a completed backend capability. Verify its current `onData`, protected image loading, and final reconciliation before enabling the new capability flags there; backend fixture work can proceed independently.

- [ ] Supply fixtures for all five examples, capability downgrades, mixed static/image answers, stale transient items after draft replacement, and partial figure failure. Match their expected Markdown as well as component content.
- [ ] Implement native frontend components, shared citations, copy/export expansion, and Streamlit's supported rendering or fallback. Advertise each capability only after its fixtures pass.
- [ ] Perform desktop/mobile, light/dark, keyboard, contrast, image-failure, interrupted-stream, and history/reload QA in the actual frontend. Record the frontend revision and observed results.
- [ ] Expand quality evaluation for useful visual selection, irrelevant-image omission, evidence support, numerical integrity, latency, and token cost. Enable rollout only after Tasks 1-4 and the relevant client checks pass.

### Targeted verification commands during implementation

Use the repository virtual environment. Run each relevant command after its task; the files introduced in Tasks 2 and 4 do not exist until those tasks are implemented.

```powershell
# Task 1: shared candidate construction and mixed-item persistence
& .\.venv\Scripts\python.exe -m pytest tests/test_rich_response_metadata.py tests/test_tool_execution_rendering.py tests/test_rich_response_streaming.py -q

# Task 2: bounded content, evidence, fallback, and tool registration
& .\.venv\Scripts\python.exe -m pytest tests/test_answer_visuals.py tests/test_answer_visual_tools.py tests/test_rich_response_contract.py tests/test_prompts_visual_strategy.py -q

# Task 3: capability parity across request and delivery paths
& .\.venv\Scripts\python.exe -m pytest tests/test_ai_sdk_resume_continue_capabilities.py tests/test_ai_sdk_assistant_ui_compat.py tests/test_ai_sdk_v6_stream_contract.py tests/test_ai_sdk_wire_fixtures.py tests/test_rich_response_streaming.py -q

# Task 4: composition, selection, media ownership, and continuation
& .\.venv\Scripts\python.exe -m pytest tests/test_source_figure_composition.py tests/test_source_figure_lifecycle.py tests/test_rich_image_selection.py tests/test_web_research_images.py tests/test_selected_image_reaches_the_model.py tests/test_message_service_web_image_externalization.py tests/test_conversation_fork_media.py tests/test_web_research_continuation.py -q

# Task 5: repository renderer checks; external frontend QA is also required
& .\.venv\Scripts\python.exe -m pytest tests/test_demo_rich_response.py tests/test_web_research_evaluation.py -q
```

## Acceptance checks

- The five supplied examples have explicit contract fixtures and expected rendering behavior.
- Simple questions still receive concise answers without unnecessary visuals.
- Tables, prose, visual labels, and fallback text agree on values and recommendations.
- Bar widths use shared scales; synthetic examples cannot be mistaken for benchmarks.
- An image comparison uses inspected images that show the intended subjects or settings, with cell-level sources.
- Unsupported clients receive complete Markdown answers.
- A mixed answer preserves grounded web images and static tool candidates in one registry; dedicated visuals never create duplicate generic renders.
- Labelled cells survive selection; duplicate media does not erase a selected figure; partial comparison failure cannot leave a misleading group caption.
- A figure selected only by its rich marker retains its media through later searches, persistence, resume, and Continue; unused dependencies are released.
- Claims inside visuals resolve to admitted evidence and retain citations in fallback/copy/export output.
- Stream completion, resume, Continue, and history produce the same selected visuals, order, labels, and captions for the same capabilities, without duplicates. Different capabilities change presentation only.
- Invalid payloads, image fetch failures, and interrupted generation preserve a useful readable answer.
- Browser checks cover narrow layouts, keyboard access, theme contrast, and media failures in the actual frontend.

Evaluate useful visual selection, omission of irrelevant visuals, evidence support, numerical integrity, layout, added latency, and token cost. Do not optimize the raw number of visuals per answer.

## Verification performed

During the follow-up codebase review, the following existing tests passed in this workspace: **132 passed in 12.20 seconds**. This is the baseline review result, not a claim that the proposed implementation has passed.

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_prompts_visual_strategy.py tests/test_rich_response_contract.py tests/test_rich_response_metadata.py tests/test_rich_response_streaming.py tests/test_web_research_images.py tests/test_selected_image_reaches_the_model.py tests/test_ai_sdk_resume_continue_capabilities.py tests/test_web_research_continuation.py -q
```

These verify existing prompt, schema, metadata, streaming, and image-context behavior. They do not validate a deployed frontend, live provider output quality, or the proposed new components.

Two read-only probes also reproduced the integration gaps: merging a tool candidate removed an existing selected web-image candidate, and adding `label` fields changed the image selector's output from one accepted group to zero. Tasks 1 and 4 turn those observations into permanent regressions during implementation.

This revision changes only this plan. Application tests were not rerun for the documentation edit; document consistency, referenced existing paths, and coverage of the seven review findings are checked separately.

### Follow-up verification: 2026-10-29

The eight-file baseline command above was rerun against the current workspace: **132 passed in 16.00 seconds**. Pytest reported one cache-write permission warning; the tests passed. An additional run with cache disabled passed **45 tests in 8.36 seconds**:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_rich_response_prompt_inventory.py tests/test_web_tool_binding.py tests/test_web_research_tool_session.py tests/test_ai_sdk_assistant_ui_compat.py -q -p no:cacheprovider
```

Read-only probes using the existing image-session test doubles confirmed:

- A source-only answer selects zero images with no grounding warning. An image token without a source is removed with `image_without_source`. Both image and source tokens together select the image.
- A valid selected web-image descriptor followed by a valid context `tool_render` candidate becomes only the tool candidate after `_merge_tool_artifacts()`. Finalization persists zero images and reports `unknown_rich_item` for the selected image marker.
- A visual search with a working text provider and an empty image-provider resolver returns `status="success"`, zero images, and no failures.

The proposed `answer_visual`/`compose_source_figure` implementation and new capability flags are still absent from `app/` and `tests/`. The original tasks remain future work. No application behavior was modified in this recheck. No deployed conversation, live provider/model response, or separate frontend was verified, so the precise cause of the reported production answer remains unconfirmed.

### Supplied LangSmith trace: missing image context

The subsequent authenticated investigation read all **82 runs** in [trace 01a186c1-85c3-77a2-8060-73a7e6541d82](https://smith.langchain.com/o/3bfbc373-21c2-400e-a6e1-84e32a625bbd/a/sample-chatbot/projects/p/587399a5-ae7d-485a-917c-71dd84ccd380?trace_id=01a186c1-85c3-77a2-8060-73a7e6541d82&run_id=01a186c1-85c3-77a2-8060-73a7e6541d82), starting at **2026-10-29 08:23:12 UTC / 15:23:12 Asia/Bangkok**. This supplies the live evidence that was unavailable during the earlier recheck above.

**Observation:** image retrieval returned prepared candidates, but the traced answering calls contained no candidate labels or pixels. Capability classification is a hypothesis for that boundary failure; the effective runtime capability object was not captured in the inspected trace. The earlier claim that this established the production root cause was too strong. This trace does not demonstrate the separate candidate-overwrite defect or a frontend image-loading failure.

| Boundary | Trace evidence |
| --- | --- |
| Routing and negotiation | The user asked `có ảnh k`; the router selected the public `search_agent`. Planning was disabled and `inline_rich_response_v1` was true. |
| Initial visual search | Run `01a186c1-9964-76a3-a2e2-d4ce219ae80f` requested `visual_intent="gallery"` and a concrete image query. Its result reported six image candidates and no failures. |
| Follow-up visual search | Run `01a186c1-ce77-7350-b30e-6bbbb7ff8dc9` requested official product images with `include_domains=["apple.com"]`. Its result again reported six image candidates and no failures. Brave provider calls are present. |
| Answer-model input | All four answering calls used `gpt-6.1-sol`. The three calls after retrieval contain text-only WEB EVIDENCE, zero `Image candidate` labels, zero `image_url`/`input_image` blocks, and zero image data URLs. The final call is `01a186c1-dd16-7330-843a-7f67b16f0a11`. |
| Model output | The final model answer already asks the user to click Apple links. It contains source tokens and no `[[image:I#]]` tokens. The loss therefore occurs before rich-item rendering or candidate merging. |

The trace records code revision `c45c8905`, which matches the inspected workspace HEAD. That revision contains the same capability checks:

- `ProviderService._normalize_openai_model()` in `app/services/provider_service.py:655` recognizes vision by the tokens `gpt-4o`, `gpt-4.1`, `gpt-5`, `o1`, `o3`, and `o4`. `gpt-6.1-sol` matches none and is classified as `supports_vision=False`.
- `ModelConfigService._build_capabilities()` in `app/services/model_config_service.py:504` trusts that catalog field; its no-catalog OpenAI fallback repeats the same incomplete token check.
- `WebEvidenceMiddleware` in `app/ai/workflow/middleware.py:420` passes the resolved capability to `model_evidence_blocks()`. At `app/ai/web_research/service.py:918`, false returns no candidate labels or pixels and records the private reason `answer_model_not_vision_capable`.

The resolved capability object itself is not present in the fetched trace fields. Local reproduction shows that the catalog normalizer and no-catalog fallback classify `gpt-6.1-sol` as non-vision, while the `gpt-5` control is vision-enabled. With the existing image-session fixture, false yields two retrieved candidates but zero labels/pixels; true yields both labels and both images. These observations do not establish which metadata or configuration the live request actually resolved.

**Investigation target for Task 0:** capture the effective runtime capability configuration and the evidence entering/leaving the model boundary before selecting a capability repair. Distinguish stored catalog metadata from fallback inference. Do not override a model's declared capabilities based only on its name. The model-specific override previously attempted here was withdrawn at the user's request.

The existing `tests/test_selected_image_reaches_the_model.py:161` injects `capabilities={"supports_vision": True}` and therefore does not exercise the faulty classifier. Rerunning that file with `tests/test_web_research_images.py` passed **9 tests in 14.04 seconds**, confirming the coverage gap rather than a repair. No application code, provider configuration, or stored catalog was changed during this investigation; no successful post-repair live answer is claimed.

### Retained repair scope after capability rollback: 2026-10-29

The user requested removal of the hardcoded model-capability fix and its helpers. Provider catalog and runtime capability service code have been restored to their pre-patch behavior. The model-specific helper and its tests were deleted, and the specialist regression again uses a generic vision-capable model fixture. No replacement model-name rule or capability bypass was added.

- [x] Remove the model-specific vision override, shared helper, cache-read override, and their associated regression cases.
- [x] Restore the original provider catalog and runtime capability service behavior.
- [x] Merge grounded and context candidates by stable ID in `MultiAgentWorkflow._merge_tool_artifacts()`. Preserve grounded descriptor/order; reject a conflicting later binding by retaining the original descriptor, and deduplicate replay. Verify the final public registry retains selected images alongside tool renders.
- [x] Run focused regressions, provider/model configuration checks, rich-response/streaming checks, and lint; record the results. Do not claim a successful live-provider/frontend answer from deterministic tests.

Code review identified one additional loss boundary: a tool-only prompt inventory records an empty `_presented_rich_image_ids` list before a later web search. Forwarding that list unchanged causes `finalize_article_content()` to strip a valid newly grounded image marker. The specialist now carries its authoritative grounded selection IDs; the graph unions those with inventory-presented IDs. The full specialist regression covers this placement/persistence path and also verifies that an uninspected context image remains excluded.

The earlier **255-test** result applied to the implementation before the capability rollback. It does not verify the current revision. No stored provider configuration or database catalog was modified by the removed patch. A new live-provider conversation and actual frontend rendering/reload remain unverified.

Post-removal verification: **242 passed in 9.89 seconds** across the same remaining test files, with the deleted model-specific test file excluded. Ruff passed for the four retained changed Python files, and `git diff --check` passed. The two capability services have no diff against the pre-patch revision; both deleted files are absent, and no application/test references to the removed helper remain.

### Authenticated recheck: stored capability and runtime resolution, 2026-10-29

This recheck independently fetched all **82 runs** of the supplied trace through LangSmith. Both visual searches returned `image_candidate_count=6` with no failures. Each of the three subsequent `ChatOpenAI` calls had text-only web evidence, no candidate labels, and no image blocks. The final model output contained source tokens but no image-selection tokens. The public route was `search_agent`, Planning was disabled, and `inline_rich_response_v1` was true. The failure is already visible at the model-input boundary; this trace provides no evidence of a frontend loading failure.

Read-only database queries for the traced user supplied the configuration evidence missing from the earlier investigation:

| Configuration boundary | Observed value |
| --- | --- |
| Persisted `search` agent configuration | Provider `openai`, model `gpt-6.1-sol`, `allow_custom_model=False`; updated **2026-10-15 08:45:47 Asia/Bangkok**. |
| Persisted OpenAI catalog entry | `gpt-6.1-sol` has `supports_vision=False`; provider row updated **2026-10-28 13:46:20 Asia/Bangkok**, before the failing trace at **2026-10-29 15:23:12 Asia/Bangkok**. |
| Current `ModelConfigService.resolve_runtime_config(user, "search")` | Provider `openai`, model `gpt-6.1-sol`, source `persisted`, `supports_vision=False`, no provider fallback. Resolution used database-enforced read-only sessions and made no model call. |

The concrete blocker is the resolved capability gate: `WebEvidenceMiddleware` passes that false flag to `WebResearchSession.model_evidence_blocks()`, which returns no labels or pixels even when prepared candidates exist. The local provider catalog normalizer classifies this model as non-vision using its model-name allowlist; runtime resolution trusts the stored field. The no-catalog fallback repeats that allowlist, so changing only the fallback would leave the observed stored-catalog path blocked.

A controlled probe using the existing image-session fixtures reproduced the boundary: two prepared candidates plus `supports_vision=False` produced zero labels/pixels and `answer_model_not_vision_capable`; enabling the flag supplied two labels and two image blocks. This establishes the backend suppression mechanism. It does not independently verify the external model's actual vision support or recover the historical in-memory capability object, which the trace did not record. The stored records predate the trace and current resolution matches its observed behavior.

**Task 0 decision after user feedback:** remove the model-name vision allowlist and the web-evidence capability gate rather than introduce another classifier or model-specific helper. Prepared candidate labels and pixels go to the selected model; the provider handles whether those inputs are accepted. Verify a new explicit-image conversation through model inspection/selection, terminal metadata, protected loading, and history reload before claiming image delivery is repaired.

Fresh focused verification against the current working tree passed **90 tests in 7.71 seconds**:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_selected_image_reaches_the_model.py tests/test_web_research_images.py tests/test_rich_response_metadata.py tests/test_rich_response_streaming.py tests/test_provider_reasoning_metadata.py tests/test_provider_model_context_metadata.py tests/test_model_config_reasoning.py -q -p no:cacheprovider
```

The existing specialist fixture explicitly supplies vision capability, so these passing plumbing tests do not establish that live runtime classification is correct. `answer_visual`, `compose_source_figure`, and the proposed capability extensions remain absent. This recheck changed only the review document; existing application edits, stored configuration, and provider catalog were left unchanged. No successful post-repair live image answer or external frontend check is claimed.

### Allowlist removal requested by the user: 2026-10-29

Removed the two OpenAI vision model-name checks. New catalog entries and no-catalog runtime configurations permit image input without matching model-name tokens. Removed the web-evidence capability gate, its runtime-config callback, and the capability arguments from evidence injection/building. Existing cached catalog flags cannot suppress web candidates, including on fallback. No replacement helper, model registry, or database rewrite was added. Other catalog consumers retain their existing metadata behavior.

The specialist regression now runs with absent, false, and true vision metadata and verifies candidate pixels, grounding repair, and selected-image persistence in every case. Before the change, the absent/false cases failed because only the evidence-text block reached the model; the unknown-model catalog and runtime checks failed too. After removal, all five checks passed. Focused image, middleware, catalog, metadata, and streaming verification then passed **152 tests in 9.33 seconds**; Ruff passed for all changed Python files. Live provider and frontend verification remain outstanding.

Additional runtime override, RAG, catalog staleness, research-session, continuation, output-policy, and documentation checks passed **103 tests in 21.53 seconds** with exit code zero. That run emitted a native PyArrow access-violation diagnostic during import before continuing to the passing result. Independent code review found no important issues and separately passed 42 relevant tests. `git diff --check` passed.

### Proactive image guidance: 2026-10-29

The user reports that image injection now works and requests more proactive image use. Updated the shared answering prompts, `web_search` description/`visual_intent` field guidance, and injected evidence instructions: fetch a useful concrete reference image without waiting for an explicit request, inspect it, and select its token beside the explanation with a source citation. An image may complement good prose; it need not show something prose cannot explain. Prefer one clear image, add distinct views for comparisons, honor text-only requests, and omit irrelevant or unclear candidates. No classifier, topic rules, or automatic image publication was added.

A live decision probe using the configured `gpt-6.1-sol` model and the chat system prompt requested `comparison` images for connector identification and a `figure` for a building-layout explanation, without either question asking for images. It made no tool call for a simple calculation or an explicit text-only description: **4/4 expected decisions**. These probes inspected tool-call arguments only; they did not execute searches or verify image relevance, selection, or frontend delivery.
