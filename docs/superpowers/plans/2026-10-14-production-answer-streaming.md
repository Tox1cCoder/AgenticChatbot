# Production Answer Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Use checkbox steps for tracking, `superpowers:test-driven-development` for behavior changes, and `superpowers:verification-before-completion` before completion claims. Execute inline unless the user separately requests delegation.

**Goal:** Deliver responsive answer streaming through the AI SDK endpoints and sidecar, with correct final-answer replacement, protocol-compatible events, lossless text, bounded buffering, and reliable terminal behavior.

**Architecture:** Keep the canonical V3 event layer and existing AI SDK adapter. Harden the existing `streamWebAnswerV1` capability: stream the first web draft as standard text deltas, display repair previews separately, and atomically replace the draft with authoritative persisted content. Share transport behavior across Chat, Resume, and Continue; keep durable generation lifecycle decisions in the message service.

**Tech Stack:** Python/FastAPI, asyncio, Pydantic, LangChain/LangGraph, pytest, Vercel AI SDK UI Message Stream, and a small Node test harness using the real AI SDK parser.

**Status:** Backend Tasks 1-8 and a review fix pass implemented on 2026-10-14 (from `3a8baab3`); see "Execution record" at the end. **Not release ready:** the frontend reconciliation, its browser tests, the frontend's SDK versions, a live verifier run, and the staged rollout are outstanding. The user selected fast draft streaming with frontend final-answer replacement on 2026-10-14. The frontend is maintained on another developer's device.

## Global constraints

- Retain citation/grounding validation and persisted messages as authoritative. Drafts are provisional, including when transported as `text-delta`.
- Do not enable `streamWebAnswerV1` for clients that cannot reconcile final text. Its default remains `false`; the capable frontend explicitly sends `true` on all three operations.
- Preserve message IDs, authorization, generation IDs, Stop fencing/idempotency, rich-item capability behavior, and internal Streamlit transport contracts.
- A disconnected socket or unexpected EOF is a transport outcome, not permission for the adapter to change durable generation status.
- Preserve the unrelated working-tree change in `docs/frontend/fork-conversation.md`.
- Use isolated test configuration from `tests/conftest.py`. No provider calls or application-database writes in the deterministic suite.
- No database migration, durable SSE replay service, provider migration, or general workflow refactor is included.
- Pin the SDK test dependency with a lockfile. `ai@6.0.0` is the audit baseline, not a claim about the other developer's installed version. Obtain their exact `ai`, `@ai-sdk/react`, and assistant-ui versions before release and test that dependency set too.

## Evidence and decisions

The preceding review ran 146 targeted Python tests successfully. Separate reproductions established that existing tests do not cover these defects:

| Finding | Reproduction / affected behavior | Task |
|---|---|---|
| Invalid SDK frames | Real `ai@6.0.0` parser rejects raw `heartbeat`, top-level tool `render`, and extra fields on `error` | 1 |
| Draft/final mismatch | Standard SDK keeps `Draft` after a transient reconcile event containing final text | 3, 7 |
| Text loss | Incremental `ha`, `ha`, `!` becomes `ha!` | 2 |
| Capability loss | Resume and Continue construct default stream state without the web capability | 4 |
| Unbounded buffering | Producer emits 5,000 events while consumer reads one | 6 |
| False completion | Source ends after partial text; adapter still sends a successful-looking finish | 5 |

These are review results from this conversation, not a fresh validation of future changes. Baseline commit observed during review: `905e52f6`.

Three approaches were considered:

1. **Selected: fast drafts plus explicit replacement.** Uses the existing capability and preserves responsiveness while accommodating grounding repairs. Requires coordinated frontend handling and tests.
2. **Standard text deltas only.** Interoperable with a plain SDK text renderer, but web answers must wait for validation when an earlier draft could change. Does not meet the selected UX as well.
3. **Append-only validated segments.** Requires redesigning validation so already published segments cannot be rewritten. Larger scope and inappropriate for this repair.

Official protocol references: [AI SDK stream protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol), [AI SDK custom data](https://ai-sdk.dev/docs/ai-sdk-ui/streaming-data), and [pinned v6.0.0 chunk schema](https://raw.githubusercontent.com/vercel/ai/ai@6.0.0/packages/ai/src/ui-message-stream/ui-message-chunks.ts). Use the pinned package as the compatibility oracle; the unversioned documentation may describe a newer major.

## Target contract

### Answer behavior

| Path | During generation | Authoritative end state |
|---|---|---|
| Ordinary answer | Standard `text-start`, `text-delta`, `text-end` | Displayed text agrees with persisted content |
| Web answer, capability absent | Existing transient preview events | Validated answer via standard text; no silent default change |
| Web answer, capability enabled | First draft via `text-delta`; replacement model runs via `data-answer-preview` | One `data-answer-reconcile` replaces the full answer text, never appends it |
| Web failure without saved answer | Discard provisional text/repair overlay | Empty reconciliation before error; clear failed draft from client message state |
| Validated budget pause or Stop partial | Provisional draft is not authoritative | Reconcile to the service's saved partial if supplied; otherwise fetch history/status |
| Disconnect / local abort | End local busy state | Clear provisional overlays, then recover persisted text and generation status |

Existing web payloads remain compatible:

```json
{"type":"text-delta","id":"t-1","delta":"Initial answer "}
{"type":"data-answer-preview","data":{"operation":"replace","text":"Repaired answer"},"transient":true}
{"type":"data-answer-reconcile","data":{"messageId":"m-1","content":"Final cited answer"},"transient":true}
```

The repair preview has no message ID today: the frontend must scope it to the active response's `start.messageId`, conversation, and request identity. An older request must never clear or overwrite a newer request's answer. Reconciliation must update effective message content used for rendering, copying, exporting, and later requests; a temporary visual overlay alone is insufficient.

Do not split the final answer into artificial delayed chunks to simulate streaming. Do not rename replacement previews to text deltas: that would concatenate incompatible drafts. Mask incomplete internal citation/image markers in provisional rendering; keep raw message content separate and use existing final citation/rich rendering once reconciliation arrives.

### Protocol repairs

```text
: keepalive

data: {"type":"data-tool-render","data":{"toolCallId":"c-1","render":{"type":"text","text":"Result"}},"transient":true}

data: {"type":"tool-output-available","toolCallId":"c-1","output":{"text":"Result"}}

data: {"type":"data-stream-error","data":{"statusCode":409,"errorCode":"conflict"},"transient":true}

data: {"type":"error","errorText":"The request could not be completed."}
```

Custom error metadata and any required draft cleanup precede `error`, because some SDK consumers stop processing when they receive it. Custom tool render metadata is keyed by tool call ID. No custom keys are added to standard SDK frames. Unknown exceptions use sanitized text; arbitrary upstream exception strings do not reach clients.

## File and ownership map

| Files | Responsibility |
|---|---|
| `app/services/event_streaming/ai_sdk_v6.py` | SDK frame shapes, web draft state, terminal sequence, source ownership |
| `app/services/event_streaming/langchain_v3.py`, `graph_public_projection.py`, `events.py` | Explicit text semantics and run identity; canonical projection |
| New `app/services/event_streaming/text_chunks.py` | Small deterministic incremental/cumulative text normalizer |
| `app/api/ai_sdk.py`, `app/schemas/message.py` | Shared capability fields and route propagation |
| `app/services/message_service.py`, `app/ai/graph.py` | Verify persisted final/partial content and canonical outcomes; change only when a regression proves a gap |
| `client_backend/api/messages.py`, `client_backend/services/server_api.py` | Proxy frame parity, upstream EOF classification, cancellation and backpressure |
| `app/core/config.py`, `client_backend/core/config.py` | Explicit maximum event size for each public streaming boundary |
| New `tests/ai_sdk_wire/` | Locked real-SDK parser/consumer tests and generated SSE fixtures |
| `plans/AI_SDK_FE_CONTRACT.md`, new `docs/frontend/answer-streaming.md`, `README.md` | Normative contract corrections, implementable FE handoff, short integration overview |
| New `app/observability/streaming.py` | Bounded-cardinality stream metrics |
| External frontend repository | Reconciliation, UX, Stop/recovery, browser tests; frontend developer owns implementation |

## Task 1: Make the AI SDK wire protocol valid

**Files:** Modify `app/services/event_streaming/ai_sdk_v6.py`, `client_backend/api/messages.py`, `plans/AI_SDK_FE_CONTRACT.md`; extend `tests/test_ai_sdk_v6_stream_contract.py`, `tests/client_backend/test_sse_keepalive.py`; create `tests/ai_sdk_wire/package.json`, `package-lock.json`, `wire.test.mjs`, `tests/test_ai_sdk_wire_fixtures.py`.

**Interfaces:** Keep `AISDKV6StreamAdapter.iter_sse()` returning an async iterator of framed strings. Export test-generated complete SSE streams to a temporary fixture directory provided by `AI_SDK_FIXTURE_DIR`. Add only the two custom data event shapes defined above.

- [ ] Add fixture generation through the real adapter and proxy for text, quiet tools, tool results with render metadata, structured errors, and unexpected exceptions. Preserve adversarial fixtures to demonstrate that the SDK rejects the old invalid frames.
- [ ] Pin `ai` to `6.0.0` in the test-only Node package, generate its lockfile, and use `node --test`. No JavaScript dependency is added to the Python runtime.
- [ ] Add a real parser assertion using this test-harness pattern; `processResponseStream` is accessed only in this JavaScript test harness and stays pinned to the SDK version:

```js
import assert from 'node:assert/strict';
import { DefaultChatTransport } from 'ai';

async function parseWire(wire) {
  const bytes = new TextEncoder().encode(wire);
  const body = new ReadableStream({
    start(controller) { controller.enqueue(bytes); controller.close(); }
  });
  const chunks = [];
  for await (const chunk of new DefaultChatTransport().processResponseStream(body)) {
    chunks.push(chunk);
  }
  return chunks;
}

await assert.rejects(parseWire('data: {"type":"heartbeat"}\n\n'));
assert.equal((await parseWire(': keepalive\n\ndata: [DONE]\n\n')).length, 0);
```

- [ ] Run the generated backend fixtures against the parser before changing emission. Expect failures on heartbeat/tool-render/structured-error fixtures.
- [ ] Emit heartbeat comments; keep existing proxy filtering for older servers. Move `render` and structured error fields into custom data events. Preserve strict `errorText: string`, sanitize unexpected proxy errors, and emit metadata/cleanup before the standard error frame.
- [ ] Add assertions that direct and proxied events have equivalent meaning, use the v1 response header, and contain one `[DONE]` on a normally drained response. Do not require terminal bytes after disconnect.
- [ ] Correct documentation that currently says SDK clients should ignore arbitrary raw event types: strict SDK parsing occurs before application handling.
- [ ] Run Python contract tests and `npm.cmd --prefix tests/ai_sdk_wire ci --ignore-scripts --no-audit --no-fund`, followed by `npm.cmd --prefix tests/ai_sdk_wire test`. Commit only this task's files with `fix(streaming): emit valid AI SDK frames`.

Use this PowerShell sequence to pass generated fixtures to the Node suite. The fixture test writes named `.sse` files and a `manifest.json` containing each filename and its expected accepted/rejected outcome. `wire.test.mjs` reads that manifest from `AI_SDK_FIXTURE_DIR`. The package's test script is `node --test wire.test.mjs`; fail explicitly if the manifest is missing or empty.

```powershell
$env:AI_SDK_FIXTURE_DIR = Join-Path ([System.IO.Path]::GetTempPath()) ('answer-stream-fixtures-' + [guid]::NewGuid().ToString('N'))
.venv/Scripts/python.exe -m pytest -q tests/test_ai_sdk_wire_fixtures.py tests/test_ai_sdk_v6_stream_contract.py tests/client_backend/test_sse_keepalive.py
npm.cmd --prefix tests/ai_sdk_wire ci --ignore-scripts --no-audit --no-fund
npm.cmd --prefix tests/ai_sdk_wire test
```

Check each command's exit code before continuing. Keep `node_modules` and generated fixtures out of version control; commit the package lock and fixture-generating tests. Later tasks rerun fixture generation before the Node suite so it always consumes current backend output.

**Acceptance:** All backend-generated fixtures parse with the real SDK; hostile legacy fixtures remain rejected; custom metadata remains available to the frontend.

## Task 2: Remove content-based token deduplication

**Files:** Create `app/services/event_streaming/text_chunks.py`, `tests/test_stream_text_chunks.py`; modify `graph_public_projection.py`, `langchain_v3.py`, `events.py` if needed to document metadata; extend `tests/test_graph_stream_projection.py`, `tests/test_event_streaming_langgraph_normalizer.py`, `tests/test_web_answer_preview_stream.py`.

**Interfaces:** Introduce `TextChunkAccumulator.consume(text: str, *, mode: Literal['delta', 'snapshot'], stream_key: tuple[str, ...]) -> str`. Canonical `message_delta` always means incremental text. Explicit provider snapshots are converted at ingestion; no provider is classified as cumulative merely from its name or the text's prefix.

- [ ] Add failing tests for repeated words, punctuation, code indentation, whitespace, Unicode, and chunks sharing a prefix/suffix. Include a direct projector test proving `['ha', 'ha', '!']` emits `haha!`.
- [ ] Add tests for explicitly cumulative snapshots, separate runs/blocks, and repair runs beginning with the same text. Replace the existing test that treats canonical `message_delta` as cumulative with an explicit snapshot-boundary test.

```python
def test_explicit_chunk_modes_preserve_text():
    from app.services.event_streaming.text_chunks import TextChunkAccumulator
    acc = TextChunkAccumulator()
    key = ("run-1", "message-1", "block-0")
    assert "".join(acc.consume(t, mode="delta", stream_key=key)
                   for t in ["ha", "ha", "!"]) == "haha!"
    other = ("run-2", "message-2", "block-0")
    assert acc.consume("Hello", mode="snapshot", stream_key=other) == "Hello"
    assert acc.consume("Hello!", mode="snapshot", stream_key=other) == "!"
    assert acc.consume("Hello!", mode="snapshot", stream_key=other) == ""
```

- [ ] Run those tests and confirm failure. Characterize the installed provider integration's actual chunk types with deterministic captured fixtures; avoid assuming the old Gemini comment is correct.
- [ ] Implement explicit modes: delta returns text unchanged; snapshot emits a suffix only when the snapshot extends the previous snapshot for that stream key. A rewriting snapshot raises a clear normalization error unless it belongs to an explicit replaceable web preview revision; never silently discard or splice changed text.
- [ ] Scope state by available run/message/block identity; allocate an invocation-local key at a known model boundary where upstream IDs are absent. Preserve run identity in both v3 and legacy tuple paths so web repairs become `replace`, not `append`.
- [ ] Update every `_consume_stream_text_chunk` call site, including subagent/legacy branches. Keep internal-node filtering, citation filtering, and rich-marker filtering intact. Clear accumulator state at invocation completion to avoid growth across runs.
- [ ] Run `python -m pytest -q tests/test_stream_text_chunks.py tests/test_graph_stream_projection.py tests/test_event_streaming_langgraph_normalizer.py tests/test_web_answer_preview_stream.py tests/test_internal_node_streams_stay_private.py` using `.venv/Scripts/python.exe`. Commit as `fix(streaming): preserve incremental text across model runs`.

**Acceptance:** No legitimate repeated text is lost; explicit snapshots still work; repaired drafts and separate model runs do not contaminate each other's accumulator.

## Task 3: Make web draft replacement deterministic

**Files:** Modify `app/services/event_streaming/ai_sdk_v6.py`; extend `tests/test_web_answer_preview_stream.py`; add web scenarios to `tests/test_ai_sdk_wire_fixtures.py` and `tests/ai_sdk_wire/wire.test.mjs`.

**Interfaces:** Retain `stream_web_answer_v1`, `data-answer-preview`, and `data-answer-reconcile`. Within the adapter, centralize preview clearing and reconciliation so all terminal paths use the same behavior. The authoritative content must come from the service's projected persisted message, not from an accumulated draft.

- [ ] Add failing cases for first draft, multiple repairs, identical repair prefixes, explicit clear then repair, final content equal/different/empty, source exception during repair, interrupt, and validated paused partial.
- [ ] Assert this exact answer-event sequence for an opted-in draft, repair, and completion:

```python
assert answer_event_types == [
    "text-delta",           # initial draft only
    "data-answer-preview",  # repair replaces the display
    "data-answer-preview",  # explicit clear of the active repair
    "data-answer-reconcile",# persisted final content, even if empty
]
assert reconciliation_count == 1
assert final_text_was_not_appended_as_delta
```

- [ ] Mark repair previews active when emitted, including the opted-in branch. Clear them before reconciliation, interruption, or failure. Make each cleanup idempotent; terminal processing cannot reconcile twice.
- [ ] Verify how `_apublish_continuation_pause` and Stop publish saved partials. Reconcile to a saved partial when provided, rather than blindly clearing a validated answer. If a terminal event has no authoritative saved content, discard provisional content and require status/history recovery.
- [ ] Add a real SDK consumer test proving that vanilla SDK text remains a draft without application reconciliation; make the required client behavior explicit instead of asserting the SDK replaces it automatically.
- [ ] Preserve the no-capability fallback and non-web text streaming. Compare post-stream rendered text with projected history, including citation and rich-marker normalization.
- [ ] Run `python -m pytest -q tests/test_web_answer_preview_stream.py tests/test_required_web_streaming.py tests/test_streamable_grounding.py tests/test_rich_response_streaming.py`, then the SDK harness. Commit as `fix(streaming): reconcile web drafts on every terminal path`.

**Acceptance:** One authoritative answer is visible after completion; old repair overlays never survive it; no duplicate answer is appended.

## Task 4: Preserve capabilities across Chat, Resume, and Continue

**Files:** Modify `app/schemas/message.py`, `app/api/ai_sdk.py`; extend `tests/test_ai_sdk_assistant_ui_compat.py`, `tests/client_backend/test_messages.py`; create `tests/test_ai_sdk_resume_continue_capabilities.py`.

**Interfaces:** Add `stream_web_answer_v1: bool = False` to `InterruptResumeRequest` and `ContinueGenerationRequest`, using their existing alias generator for `streamWebAnswerV1`. Pass it to `StreamState` in both routes. The flag controls the current response's presentation, not durable model state.

- [ ] Write route-level tests using fake message services for all three operations, both camelCase/snake_case, omitted/false/true, and rich-response capability combinations.

```python
@pytest.mark.parametrize("key", ["streamWebAnswerV1", "stream_web_answer_v1"])
def test_continue_schema_accepts_web_stream_capability(key):
    from app.schemas.message import ContinueGenerationRequest
    value = ContinueGenerationRequest.model_validate({
        "conversationId": "11111111-1111-4111-8111-111111111111",
        "generationId": "22222222-2222-4222-8222-222222222222",
        "continuationId": "33333333-3333-4333-8333-333333333333",
        "idempotencyKey": "continue-test-1", "expectedVersion": 1, key: True,
    })
    assert value.stream_web_answer_v1 is True
```

- [ ] Run the new tests to fail, add fields and route propagation, then test proxy forwarding without dropping capability fields or changing device/auth context.
- [ ] Verify Chat, Resume, and Continue each deliver early web `text-delta` with true and legacy previews with false. Keep route ownership, HITL identity, and continuation idempotency tests passing.
- [ ] Run `python -m pytest -q tests/test_ai_sdk_assistant_ui_compat.py tests/test_ai_sdk_resume_continue_capabilities.py tests/test_interrupt_resume_addressing.py tests/client_backend/test_messages.py`. Commit as `fix(api): preserve web streaming capabilities on continuation`.

**Acceptance:** The same capable client sees consistent answer behavior before and after approval or budget continuation.

## Task 5: Distinguish terminal outcomes from transport failure

**Files:** Modify `app/services/event_streaming/ai_sdk_v6.py`, `client_backend/services/server_api.py`, `client_backend/api/messages.py`; extend `tests/test_ai_sdk_v6_stream_contract.py`, `tests/test_generation_transport_parity.py`, `tests/test_message_stream_cancellation.py`, `tests/client_backend/test_sse_keepalive.py`.

**Interfaces:** Track adapter outcome internally as `complete`, `interrupt`, `paused`, `error`, or unresolved. Recognize service-emitted Stop/status sequences without inventing new generation transitions. Add `require_done: bool = False` to `ServerAPIClient.stream_sse`; AI SDK proxy callers enable it, internal SSE callers retain their existing behavior.

- [ ] Add failing tests for empty source, partial-only EOF, exception before/after first token, completion, HITL, validated budget pause, Stop, and duplicate terminal events.
- [ ] Encode the regression assertion: partial-only EOF emits `data-stream-error` with `errorCode: 'stream_incomplete'` before standard `error`; it never indicates a successful finish. Durable generation-event parity remains unchanged.
- [ ] Trace actual service terminal event sequences before choosing the closure point. In particular, do not stop at `continuation_available` if a following event carries the persisted partial. Recognize that sequence as a legitimate paused outcome even if it ends without ordinary completion.
- [ ] On normal completion, emit one closure sequence. On a live-connection error, clear/reconcile drafts before `error`, close open text/reasoning blocks once, and finish with `finishReason: 'error'` and one `[DONE]` where the connection still permits writes. On cancellation/GeneratorExit, close upstream resources and emit no further frames.
- [ ] Use explicit `contextlib.aclosing` around nested async iterators while retaining producer-owned source creation/closure, so ContextVar tokens reset in the same task. Remove the proxy's unconditional yield from its `finally` block.
- [ ] In AI SDK proxy mode, distinguish upstream `[DONE]` from bare EOF. Upstream EOF without the required terminator becomes a transport error, not an invented successful response. Preserve the existing behavior that trailing bytes after the first `[DONE]` are ignored.
- [ ] Test outer `iter_sse().aclose()`, not only the inner heartbeat wrapper: assert upstream cleanup completes before close returns, including during a tool wait. Keep service persistence cleanup shielded where already required.
- [ ] Run `python -m pytest -q tests/test_ai_sdk_v6_stream_contract.py tests/test_generation_transport_parity.py tests/test_message_stream_errors.py tests/test_message_stream_cancellation.py tests/test_message_stop_generation.py tests/client_backend/test_sse_keepalive.py`, then the SDK harness. Commit as `fix(streaming): classify termination and close upstream sources`.

**Acceptance:** Truncation cannot masquerade as successful completion; legitimate pauses remain usable; cancellation does not leak tasks or change adapter-owned durable state.

## Task 6: Bound buffering without deadlocking cancellation

**Files:** Modify `app/services/event_streaming/ai_sdk_v6.py`, `client_backend/api/messages.py`, `app/core/config.py`, `client_backend/core/config.py`; create `tests/test_stream_backpressure.py`; extend `tests/test_image_stream_http_contract.py` and `tests/client_backend/test_image_stream_proxy.py`.

**Interfaces:** Use one queued event per producer/consumer boundary (`asyncio.Queue(maxsize=1)`). Track producer completion separately with an `asyncio.Event` or task completion, rather than placing a sentinel into a full queue in `finally`. Add `stream_max_event_bytes` with an 8 MiB default and positive-value validation in server and client settings; enforce against UTF-8 wire bytes.

- [ ] Add deterministic slow-consumer tests: hold the consumer after one frame while the source offers 5,000 events. Assert queued events never exceed one and production cannot drain the full source; allow the currently processed and pending-put event explicitly.
- [ ] Add cancellation tests while the producer is blocked in `queue.put`, while a heartbeat is due, and while upstream raises with a full queue. Assert source closure and task completion within one second in the deterministic harness.
- [ ] First reproduce the current unrestricted producer. Then replace unbounded queues, keep source creation and iteration in one producer task, and preserve event ordering without dropping text to relieve pressure.
- [ ] The consumer drains queued data before observing producer completion. Producer errors surface once after queued data; cancellation cannot block waiting to enqueue a sentinel. Use explicit synchronization events in tests, not timing-dependent sleeps.
- [ ] Enforce the wire-size ceiling before writing a frame. Preserve existing image-reference delivery and tool offload behavior. If an event still exceeds the ceiling, emit a sanitized `stream_event_too_large` error and terminate; never silently truncate text or send invalid JSON. In the sidecar, bound the inbound frame buffer as well, before accepting an arbitrarily large line from upstream.
- [ ] Load-test 20 concurrent synthetic slow readers with fixed-size text and near-limit media events. Measure peak memory and verify it stops growing with event count; account for serialized copies and HTTP buffers separately from queue capacity.
- [ ] Run `python -m pytest -q tests/test_stream_backpressure.py tests/test_image_stream_http_contract.py tests/client_backend/test_image_stream_proxy.py tests/client_backend/test_sse_keepalive.py`. Commit as `fix(streaming): bound buffering and preserve cancellation`.

**Acceptance:** Memory use scales with active streams and bounded event size, not the number of generated events; no deadlock or silent loss under backpressure.

## Task 7: Deliver the frontend integration contract and UX

**Backend files:** Update `plans/AI_SDK_FE_CONTRACT.md`, `README.md`; create `docs/frontend/answer-streaming.md`. **External owner:** frontend developer; their repository paths must be identified there, not guessed in this backend repository.

**Interfaces:** Frontend sends `streamWebAnswerV1: true` on Chat, Resume, and Continue only after it supports the existing preview/reconcile contract plus `data-stream-error` and `data-tool-render`. Reuse existing Stop and generation status APIs; no new recovery endpoint is assumed.

- [ ] Publish full request examples for all three operations and full response fixtures for normal completion, repair, failure, pause, and Stop. Document standard transport/header requirements and a text-only client's fallback.
- [ ] Provide the frontend developer a message-scoped reducer contract. Keep its authoritative replacement separate from SDK append-only draft updates until history confirms the saved content:

```ts
type AnswerView = {
  requestId: string;
  messageId: string;
  phase: 'draft' | 'repair' | 'final';
  repair: string | null;
  authoritative: string | null; // '' is a valid clear; null means absent
};

function visibleAnswer(state: AnswerView, sdkDraft: string): string {
  return state.authoritative ?? state.repair ?? sdkDraft;
}
// Reconcile: authoritative = event.data.content; repair = null; phase = 'final'.
// Replace preview: repair = event.data.text; phase = 'repair'.
// Append preview: repair = (repair ?? '') + event.data.text.
// Late preview after phase === 'final': ignore.
```

- [ ] Require reducer tests that distinguish empty authoritative text from absent text; reject stale request IDs; clear only the active repair layer on preview clear; preserve authoritative final text across later SDK callbacks. Patch effective message text after finish and verify copying/exporting/follow-up requests use it. Preserve non-text parts and metadata.
- [ ] On errors and aborts, immediately leave busy state and discard provisional web layers; recover saved content by message/generation identity. On history load, authoritative persisted content wins. Never persist browser draft text as a final assistant message.
- [ ] Add UI states for waiting, tool activity, answering, repairing/checking sources, paused, stopped, and failed. Distinguish first answer text from keepalives or tool status. Keep Stop available during tools and answer generation.
- [ ] Pair local `stop()` with the existing explicit server Stop command using generation ID, expected version, and an idempotency key. Reconcile accepted-but-unconfirmed Stop responses through status polling. Do not assume aborting fetch alone proves generation stopped.
- [ ] Batch rendering to animation frames or a short throttle; preserve exact text. Keep Markdown/code stable across partial input, suppress incomplete internal markers, follow scrolling only while the user is near the bottom, and provide an explicit jump-to-latest action. Avoid announcing every token to screen readers.
- [ ] Frontend developer runs browser tests against direct and proxied routes with their locked SDK versions: repaired final text matches history, switching conversations does not mix drafts, navigation/abort does not leave a spinner, and copy uses final text.
- [ ] Commit the backend handoff docs as `docs(frontend): define production answer streaming integration`. Record FE implementation/test evidence in the release checklist; do not mark the external work complete merely because documentation was delivered.

**Acceptance:** The frontend demonstrates progressive text and correct replacement on its actual dependency set; no source IDs, stale drafts, duplicated answers, or perpetual busy states remain in the tested scenarios.

## Task 8: Verify the complete path and stage release

**Files:** Create `app/observability/streaming.py`, `tests/test_streaming_metrics.py`, `scripts/verify_answer_streaming_contract.py`; wire metrics into `ai_sdk_v6.py` and the existing metrics exposure pattern after inspecting `app/observability/web_research.py` and its consumers. Extend `tests/test_ai_sdk_wire_fixtures.py` and `tests/ai_sdk_wire/wire.test.mjs`.

**Interfaces:** Export `streaming_metrics` with `record_first_answer(seconds: float, *, operation: str)`, `record_reconcile(*, changed: bool)`, and `record_end(*, operation: str, outcome: str)`. Operation labels are limited to `chat`, `resume`, `continue`, `unknown`; outcomes to `complete`, `paused`, `interrupt`, `error`, `disconnect`, `unknown`. Message IDs, prompts, content, and source URLs are never metric labels.

- [ ] Add failing isolated-registry tests for first-answer timing, changed/unchanged reconciliation, terminal counts, and bounded labels. Implement counters/histograms using the repository's existing Prometheus style.
- [ ] Measure request-to-first-answer and first-model-delta-to-wire separately. Keep keepalives, status, reasoning, and source metadata out of first-answer measurements. Make clear whether latency includes provider/tool execution.
- [ ] Create a verifier that requires an explicitly supplied test server and test-account credentials via environment variables, accepts direct/sidecar origin selection, and never prints tokens. Deterministic fixtures remain the default automated gate; live provider/browser checks run only against a designated test environment.
- [ ] Cover the following acceptance matrix through backend, proxy, real SDK, frontend, and history where applicable:

| Scenario | Required assertion |
|---|---|
| Ordinary chat with repeated tokens / Unicode / code | Exact ordered text, no loss; final equals history |
| Web answer without repair | Text appears before completion; final replaces without duplication |
| Web answer with one or multiple repairs | Repair overlay clears; one authoritative final answer |
| Client without web capability | Existing validated fallback remains usable |
| Tool silent longer than 15 seconds | Parser accepts keepalives; UI remains active |
| Tool failure / structured error | Metadata available; SDK handles standard error normally |
| Image and rich response | Existing visibility and protected-media behavior preserved |
| HITL Resume / budget Continue | Same web capability and reconciliation behavior |
| Stop / disconnect during repair or full queue | Timely cleanup; no false success; recover saved state |
| Unexpected EOF and upstream malformed frame | Explicit transport failure, no endless spinner |
| Slow client and concurrent streams | Bounded buffering, ordered text, no orphan producer |
| History reload / conversation switch | Stable identity, persisted text wins, no stale overlay |

- [ ] Deterministic gates: zero parser errors, zero text loss/duplicate terminal sequences, exactly one reconciliation per streamed web terminal path that has a live connection, and no leaked producer tasks. For a source fixture delayed between chunks, prove first text is consumed before the source completes using event barriers.
- [ ] Proposed performance budgets in a controlled test environment: p95 adapter overhead under 100 ms from model delta receipt to yielded frame, frontend paint within 100 ms of received text, and deterministic cancellation cleanup under one second. These are targets to measure, not current performance claims or provider latency promises.
- [ ] Run the focused tests from Tasks 1-6 together, the Node harness with pinned dependencies, changed-file Ruff checks, and the existing non-blocking, grounding, image, rich-response, and transport parity suites. Run the configured test-database integration tests in an isolated database before release. Investigate failures rather than relaxing assertions to match defects.
- [ ] Stage rollout: deploy valid frames/lossless normalization and backend capability parity; deploy tested FE reconciliation; enable the FE request flag for test users; exercise the matrix and inspect metrics; expand to the intended user group. Never flip the backend default globally as a shortcut.
- [ ] Rollback: disable the frontend web capability to restore the validated fallback; retain protocol validity and text-loss fixes. Do not reintroduce invalid frames. Keep the prior FE deployment compatible with custom data events it ignores.
- [ ] Commit verification/metrics as `test(streaming): validate production answer delivery`. Record SDK versions, command results, latency/memory measurements, and FE sign-off in this plan before marking release ready.

**Acceptance:** Backend and frontend evidence demonstrate the selected UX on the actual integration; passing Python unit tests alone cannot authorize the production-readiness claim.

## Execution order and release gates

1. Task 1 establishes the real SDK oracle and repairs protocol violations.
2. Task 2 repairs text correctness; Task 3 makes replacement deterministic; Task 4 connects every route.
3. Tasks 5 and 6 repair termination, cleanup, and resource limits together without changing durable lifecycle ownership.
4. Task 7's contract can be handed to the frontend developer after the wire contract is settled. The external implementation is required for the selected fast-streaming UX.
5. Task 8 verifies the integrated result and controls rollout.

Before implementation, read the current working tree again and create an isolated checkout/worktree if needed. The saved plan does not authorize discarding unrelated changes, publishing a deployment, or sending messages to the frontend developer. No further design question is required for the selected fast-draft/replacement approach; the exact frontend dependency versions are a release input to collect during integration.

## Plan review checklist

- [x] All six review findings map to implementation and verification tasks.
- [x] Fast draft streaming plus final replacement reflects the user's explicit choice.
- [x] Direct and proxied SDK paths, Resume, Continue, Stop, error, and history are covered.
- [x] Provider text semantics, SDK parsing, and application replacement are tested at separate boundaries.
- [x] Bounded queues include full-queue cancellation and event-size handling.
- [x] Transport termination cannot independently mutate durable generation status.
- [x] External frontend ownership and dependency/version uncertainty are explicit.
- [x] No application code changes or production-readiness claims are implied by this planning document.

## Execution record (2026-10-14)

| Task | Commit | Task gate |
|---|---|---|
| 1 Valid wire frames | `3a8baab3` | Python contract tests 85 passed; real-SDK harness all fixtures accepted, legacy fixtures rejected |
| 2 Lossless text | `7db8bd08` | 117 passed |
| 3 Draft reconciliation | `63593b8b` | 76 passed + SDK harness |
| 4 Resume/Continue capability | `02c661a0` | 62 passed |
| 5 Terminal outcomes | `fb48342d` | 107 passed + SDK harness |
| 6 Bounded buffering | `1e73d22b` | 67 passed, 1 skipped |
| 7 Frontend handoff docs | `315bd417` | doc-contract tests 56 passed |
| 8 Metrics, verifier, matrix fixtures | `6f657f53`, `fe441ab7` | see below |
| Review fix pass | committed with this record | see below |

Evidence:

- SDK baseline: `ai@6.0.0` (lockfile `tests/ai_sdk_wire/package-lock.json`), Node v24.14.0. Final harness run: 25/25 tests, every backend fixture parsed.
- Whole-branch review (fresh Opus reviewer): no Critical findings, verdict "with fixes". One Important finding (a same-worker Stop cancelling the producer while it waits in `put`) and three Minor findings re-graded to Important (upstream connection drop mapped to `stream_incomplete`, verifier first-text detection through the sidecar, a false FE-doc sentence about aborts) were fixed with failing-first tests; the remaining Minor findings were deferred.
- Full suite after the fix pass, with `TEST_DATABASE_URL` on `chatbot_test`: 7,142 passed, 15 skipped, 5 failed, 4 errors. All 9 are `UndefinedTable` errors (`tool_approval_settings`, `task_plans`, ...) in `test_database_schema_contract.py`, `test_repository_async_twins.py`, and `test_hitl_api.py`. They reproduce when those modules run alone, and this branch changes no models, repositories, migrations, or HITL code: `chatbot_test` schema drift, not a streaming regression.
- `ruff check app tests scripts client_backend`: clean.
- Provider text semantics, captured with a fake model through langchain-core 1.6.3 / langgraph 1.2.11: v3 and tuple paths deliver incremental chunks; langchain-google-genai 4.4.0 accumulates usage only.
- Memory, 20 concurrent slow readers: text-only peak 0.30-0.36 MiB at 500, 2,000, and 8,000 events per reader; with two 3.9 MB inline previews per reader, 154.7 MiB at every event count.
- Adapter overhead (source yield to frame yielded), 20 readers x 2,000 deltas: p50 0.21 ms, p95 0.44 ms, p99 0.67 ms (target p95 < 100 ms).
- Cancellation cleanup: deterministic tests complete within their 1 s bound.

Outstanding before release (not done): frontend reconciliation and browser tests on the frontend's own SDK versions (`ai`, `@ai-sdk/react`, assistant-ui); a live run of `scripts/verify_answer_streaming_contract.py` against a test server, direct and through the sidecar; frontend sign-off; staged rollout per Task 8. Executor rulings are listed in the session's final report.
