# Answer streaming: frontend contract

How the AI SDK chat stream delivers answer text, and what the frontend must do so that
progressive web-answer drafts end as the saved, cited answer. Normative wire details live in
[`plans/AI_SDK_FE_CONTRACT.md`](../../plans/AI_SDK_FE_CONTRACT.md); this page is the
implementation guide for the frontend.

Every example stream below was produced by the backend adapter, not written by hand.

## What changed on the wire (all AI SDK clients)

These apply whether or not you opt in to web drafts. The AI SDK validates every `data:` frame
against strict schemas before your code sees it; the old frames failed that validation.

| Before | Now |
|---|---|
| `data: {"type":"heartbeat"}` every 15 s of silence | `: keepalive` SSE comment. Parsers drop it; no chunk reaches `onData`. |
| `render` key on `tool-output-available` | `data-tool-render` part `{toolCallId, render}`, sent just before the tool output. |
| `statusCode` / `errorCode` keys on `error` | `data-stream-error` part `{statusCode?, errorCode?}`, sent just before `error`. `error` carries only `errorText`. |
| Error streams ended with a plain `finish` | `finish` carries `finishReason: "error"`. |
| A stream that ended early still finished normally | `data-stream-error {errorCode: "stream_incomplete"}`, `error`, `finish {finishReason: "error"}`. |

If your current code reads `render` from tool parts or `errorCode` from the error event, move
those reads to `onData` for the two new parts. Both parts are `transient`: they are not stored on
the message, so keep them in request-scoped state.

## Opting in to web drafts

Send `streamWebAnswerV1: true` on **all three** operations, and only once the reducer below is
implemented. Without the flag the backend keeps the validated fallback (drafts as
`data-answer-preview` overlays, final text as `text-delta`), so an older build keeps working.

```http
POST /api/chat/{conversationId}
```

```json
{
  "messages": [{ "role": "user", "parts": [{ "type": "text", "text": "What is the capital of France?" }] }],
  "inlineRichResponseV1": true,
  "streamWebAnswerV1": true
}
```

```http
POST /ai/resume-interrupt
```

```json
{
  "threadId": "conversation-thread-id",
  "conversationId": "conversation-id",
  "interruptId": "interrupt-id",
  "decisions": [{ "type": "approve", "toolCallId": "tool-call-id" }],
  "inlineRichResponseV1": true,
  "streamWebAnswerV1": true
}
```

```http
POST /ai/continue
```

```json
{
  "conversationId": "conversation-id",
  "generationId": "generation-id",
  "continuationId": "continuation-id",
  "expectedVersion": 4,
  "idempotencyKey": "client-generated-unique-key",
  "inlineRichResponseV1": true,
  "streamWebAnswerV1": true
}
```

`stream_web_answer_v1` is accepted as an equivalent key. The flag affects only how this response
is presented; it is never saved.

Response headers on every stream, direct or through the local sidecar:

```http
content-type: text/event-stream; charset=utf-8
x-vercel-ai-ui-message-stream: v1
```

A fully drained stream ends with exactly one `data: [DONE]`. A closed socket carries no
terminal bytes; treat it as unknown, not as finished.

## Answer events

| Part | When | What to do |
|---|---|---|
| `text-delta` | Ordinary answers; the **first** web-answer draft when opted in | SDK appends it to the text part. Treat a web draft as provisional. |
| `data-answer-preview` `replace` / `append` | A grounding repair (opted in), or any web draft (not opted in) | Set / extend the repair layer. Never append it to the SDK text. |
| `data-answer-preview` `clear` | Before reconciliation, on interrupt, on failure | Drop the repair layer only. |
| `data-answer-reconcile` `{messageId, content}` | Once per opted-in response that streamed a draft | `content` replaces the visible answer. `""` is a real value: the draft is discarded. |
| `data-stream-error` `{statusCode?, errorCode?}` | Just before `error`, when the backend knows a code | Keep for the active request; pair with the error the SDK reports. |
| `error` `{errorText}` | Terminal failure | Leave busy state; recover (below). |
| `finish` `{finishReason?}` | Every drained stream | `finishReason: "error"` means the response did not complete. |

The SDK alone never applies `data-answer-reconcile`: a plain `useChat` text renderer keeps the
draft. The backend's Node test harness (`tests/ai_sdk_wire/`, real `ai@6.0.0` parser) asserts
exactly that, so replacement is frontend work.

## Example streams (opted in)

Web answer, no repair. The draft can contain internal markers such as `[[source:S1]]`; the
reconciled text is the saved answer.

```text
data: {"type":"start","messageId":"m-1"}
data: {"type":"start-step"}
data: {"type":"text-start","id":"t-1"}
data: {"type":"tool-input-start","toolCallId":"call-1","toolName":"web_search"}
data: {"type":"tool-input-available","toolCallId":"call-1","toolName":"web_search","input":{"query":"q"}}
data: {"type":"tool-output-available","toolCallId":"call-1","output":{"results":3}}
data: {"type":"text-delta","id":"t-1","delta":"Paris is the "}
data: {"type":"text-delta","id":"t-1","delta":"capital [[source:S1]]."}
data: {"type":"data-answer-reconcile","data":{"messageId":"m-1","content":"Paris is the capital [1]."},"transient":true}
data: {"type":"text-end","id":"t-1"}
data: {"type":"finish-step"}
data: {"type":"finish"}
data: [DONE]
```

Web answer with a repair. The repair streams as previews over the first draft; the overlay is
cleared before the authoritative text arrives.

```text
data: {"type":"text-delta","id":"t-1","delta":"Paris is the capital."}
data: {"type":"data-answer-preview","data":{"operation":"replace","text":"Paris is the capital "},"transient":true}
data: {"type":"data-answer-preview","data":{"operation":"append","text":"of France [1]."},"transient":true}
data: {"type":"data-answer-preview","data":{"operation":"clear"},"transient":true}
data: {"type":"data-answer-reconcile","data":{"messageId":"m-1","content":"Paris is the capital of France [1]."},"transient":true}
data: {"type":"text-end","id":"t-1"}
data: {"type":"finish-step"}
data: {"type":"finish"}
data: [DONE]
```

Failure during a draft. The draft is discarded before the error.

```text
data: {"type":"text-delta","id":"t-1","delta":"Paris is"}
data: {"type":"data-answer-reconcile","data":{"messageId":"m-1","content":""},"transient":true}
data: {"type":"data-stream-error","data":{"statusCode":502,"errorCode":"provider_unavailable"},"transient":true}
data: {"type":"error","errorText":"Generation failed."}
data: {"type":"text-end","id":"t-1"}
data: {"type":"finish-step"}
data: {"type":"finish","finishReason":"error"}
data: [DONE]
```

Execution-budget pause. The saved, validated partial replaces the draft, then Continue is offered
through `data-generation`.

```text
data: {"type":"text-delta","id":"t-1","delta":"Unvalidated draft"}
data: {"type":"data-answer-reconcile","data":{"messageId":"m-1","content":"Validated partial answer."},"transient":true}
data: {"type":"data-generation","id":"g-1","data":{"phase":"continuation_available","generation_id":"g-1","status":"continuable","version":4,"continuation_id":"c-1","continuation_available":true}}
data: {"type":"text-end","id":"t-1"}
data: {"type":"finish-step"}
data: {"type":"finish"}
data: [DONE]
```

Stop, or the server stream ended early. A cooperative Stop ends the server's stream without a
terminal event, so this response cannot know the saved partial: it discards the draft and
reports `stream_incomplete`. Read the outcome from generation status.

```text
data: {"type":"text-delta","id":"t-1","delta":"Paris is"}
data: {"type":"data-answer-reconcile","data":{"messageId":"m-1","content":""},"transient":true}
data: {"type":"data-stream-error","data":{"errorCode":"stream_incomplete"},"transient":true}
data: {"type":"error","errorText":"The response ended before it was complete."}
data: {"type":"text-end","id":"t-1"}
data: {"type":"finish-step"}
data: {"type":"finish","finishReason":"error"}
data: [DONE]
```

Tool approval (HITL). With a streamed draft, the paused message's content is reconciled before
`data-interrupt`.

```text
data: {"type":"text-end","id":"t-1"}
data: {"type":"data-interrupt","data":{"threadId":"t-1","next":["tools"],"interrupt":{"interrupt_id":"i-1"},"message":{"id":"m-1","content":"Approval required."}}}
data: {"type":"finish-step"}
data: {"type":"finish"}
data: [DONE]
```

## Reducer contract

Keep authoritative replacement **separate** from the SDK's append-only text until history
confirms the saved message. One `AnswerView` per response, keyed by request and message:

```ts
type AnswerView = {
  requestId: string;          // your id for this sendMessage / resume / continue call
  messageId: string;          // from the stream's `start.messageId`
  phase: 'draft' | 'repair' | 'final';
  repair: string | null;
  authoritative: string | null; // '' is a valid clear; null means absent
};

function visibleAnswer(state: AnswerView, sdkDraft: string): string {
  return state.authoritative ?? state.repair ?? sdkDraft;
}
// Reconcile:        authoritative = event.data.content; repair = null; phase = 'final'.
// Replace preview:  repair = event.data.text; phase = 'repair'.
// Append preview:   repair = (repair ?? '') + event.data.text.
// Clear preview:    repair = null (the SDK draft shows again until reconcile).
// Late preview after phase === 'final': ignore.
```

Rules:

1. Scope every update to the active response. `data-answer-preview` carries no message id: bind it
   to the `start.messageId`, conversation, and `requestId` of the stream it arrived on. A part
   from an older request must never clear or overwrite a newer request's answer.
2. `authoritative: ''` and `authoritative: null` differ. Test both.
3. After `finish`, patch the message's effective text with `visibleAnswer(...)` so rendering,
   copy, export, and the history you send in the next request all use it. A visual overlay alone
   is not enough. Preserve non-text parts and metadata.
4. Never persist a draft as an assistant message. On history load, persisted content wins.
5. Mask incomplete internal markers (`[[source:…]]`, `[[image:…]]`, `<!--rich:` lines) in
   provisional text; render the final text with the existing citation and rich-item renderer.

## Errors, aborts, Stop, and recovery

- On `error`, `finish {finishReason: "error"}`, a transport failure, or a local abort: leave
  busy state immediately and drop provisional layers (`repair`, unreconciled draft).
- Recover by identity: refetch `GET /ai/conversations/{conversationId}/messages` and, when you
  have a generation id, `GET /messages/generations/{generationId}?conversation_id={conversationId}`.
  The generation id and version come from the `data-generation` part with `phase: "start"`.
- `errorCode` values you can act on: `stream_incomplete` (read status; the turn may be stopped,
  paused, or still running), `stream_event_too_large` (413; the response was cut off server-side),
  and the HITL duplicate codes in the main contract.
- Stop pairs the local abort with the server command; aborting the fetch alone stops nothing:

```ts
async function stopGeneration(snapshot: { generationId: string; conversationId: string; version: number }) {
  stop(); // useChat: end this client's read immediately
  const response = await fetch('/messages/stop', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      generationId: snapshot.generationId,
      conversationId: snapshot.conversationId,
      expectedVersion: snapshot.version,
      idempotencyKey: crypto.randomUUID(),
    }),
  });
  // 202 = accepted, not confirmed: poll the generation until it leaves `stop_requested`.
  return response;
}
```

No new recovery endpoint exists; use the two reads above.

## UI states

| State | Enter on | Notes |
|---|---|---|
| waiting | request sent | Keepalives are not progress; show waiting until a real part arrives. |
| tool activity | `tool-input-start` | Keep Stop enabled. |
| answering | first `text-delta` | First answer text, distinct from tool status. |
| checking sources | first repair `data-answer-preview` | Draft stays visible under the repair. |
| paused | `data-generation` `continuation_available` | Offer Continue. |
| stopped | generation status `stopped` / `completed_partial` | |
| failed | `error` / `finishReason: "error"` | Show recovery, never a spinner. |

## Rendering

- Batch text updates to an animation frame or a short throttle. Keep the exact characters:
  repeated words and whitespace are real text.
- Keep Markdown and code blocks stable across partial input.
- Auto-scroll only while the user is near the bottom; offer a jump-to-latest control.
- Do not announce every token to screen readers.

## Clients that only render text

Leave `streamWebAnswerV1` unset. Web answers then arrive as validated `text-delta` (with
optional `data-answer-preview` overlays you may ignore), and a paused web turn's saved partial
arrives as `text-delta` too.

## Frontend verification (owner: frontend developer)

Run in a browser against both the direct server and the local sidecar, with your locked
dependency versions:

- [ ] Repaired final text equals the history text for the same message.
- [ ] Switching conversations mid-stream does not mix drafts between them.
- [ ] Navigation, abort, Stop, and `stream_incomplete` never leave a spinner.
- [ ] Copy and export use the final text; the next request's history carries it.
- [ ] `authoritative: ''` hides the draft; `null` keeps it.
- [ ] Resume and Continue show drafts and reconcile the same way as Chat.

Report back: exact versions of `ai`, `@ai-sdk/react`, and assistant-ui; the results of the list
above; and any frame the SDK rejected (with the raw `data:` line). The backend's parser tests
pin `ai@6.0.0`, which is a baseline, not a claim about your installed version.
