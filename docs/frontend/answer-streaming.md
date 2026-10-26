# Answer streaming: frontend contract

The AI SDK chat stream now passes the SDK's strict parser and reports failed or cut-off answers
explicitly. Sections 1-2 apply to every AI SDK client. Section 3 (progressive web answers) is
opt-in. Endpoints, auth and request bodies are otherwise unchanged. Full event catalog:
[`plans/AI_SDK_FE_CONTRACT.md`](../../plans/AI_SDK_FE_CONTRACT.md).

Deploy the updated local sidecar together with the updated server.

## 1. Metadata moved to custom parts (required)

The SDK rejects extra keys on its standard events, so backend metadata now arrives in a `data-*`
part just before the event it belongs to. Read both in `useChat({ onData })`. Both are
`transient`: keep them in request state, not on the message.

| Was | Now |
| --- | --- |
| `statusCode` / `errorCode` on `error` | `data-stream-error` `{ statusCode?, errorCode? }`, sent just before `error`. `error` carries only `errorText`. |
| `render` on `tool-output-available` | `data-tool-render` `{ toolCallId, render }`, sent just before the tool output. History still has it in `metadata.tool_artifacts`. |
| `{"type":"heartbeat"}` frames | `: keepalive` SSE comment. The parser drops it; nothing to do. |

```text
data: {"type":"data-stream-error","data":{"statusCode":409,"errorCode":"INTERRUPT_CONFLICT"},"transient":true}
data: {"type":"error","errorText":"Interrupt was claimed by a concurrent request."}
```

The approval duplicate check (`INTERRUPT_ALREADY_RESOLVED` / `INTERRUPT_CONFLICT`) must read
`errorCode` from `data-stream-error`.

## 2. Failed endings (required)

A stream that fails or ends early finishes with `finishReason: "error"`. Leave the busy state,
show the error and refetch history; never treat the text you have as complete.

```text
data: {"type":"data-stream-error","data":{"errorCode":"stream_incomplete"},"transient":true}
data: {"type":"error","errorText":"The response ended before it was complete."}
data: {"type":"text-end","id":"t-1"}
data: {"type":"finish-step"}
data: {"type":"finish","finishReason":"error"}
data: [DONE]
```

| `errorCode` | Meaning / UI action |
| --- | --- |
| `stream_incomplete` | Stream ended with no outcome (server dropped, or the turn was stopped). Read generation status and history. |
| `stream_malformed` | The sidecar received a frame it could not parse. Show the error; refetch history. |
| `stream_event_too_large` | One event exceeded 8 MiB and was not sent (`statusCode` 413). Show the error; refetch history. |
| other | Domain error (for example the HITL codes in the main contract). |

## 3. Progressive web answers (optional)

Web answers can stream as a draft while sources are checked, then get replaced by the saved, cited
answer. Turn this on only after the handling below works; without it you keep today's behaviour.

Send `"streamWebAnswerV1": true` on all three: `POST /api/chat/{conversationId}`,
`POST /ai/resume-interrupt`, `POST /ai/continue`.

| Part | Handling |
| --- | --- |
| `text-delta` | First draft. The SDK appends it as usual; treat it as provisional. |
| `data-answer-preview` | A rewrite of the draft. `replace` sets a separate repair text, `append` extends it, `clear` removes it. Never append it to the SDK text. |
| `data-answer-reconcile` | The saved answer, sent once. Show `data.content` instead of the draft. `""` means discard the draft. |

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

```ts
// What to show for one response
const visible = authoritative ?? repair ?? sdkDraftText;
// reconcile → authoritative = data.content ('' is a real value); repair = null
// preview   → replace: repair = text; append: repair += text; clear: repair = null
// any preview after reconcile → ignore
```

Rules:

1. Scope previews to the stream they arrived on (`start.messageId` plus your request id). An older
   request must never change a newer answer.
2. After `finish`, store the visible text as the message text, so copy, export and the next
   request's history use it.
3. Hide internal markers such as `[[source:S1]]` in draft text; the reconciled text is final.
4. On error, abort or Stop, drop the draft and repair text and refetch history. Never save a draft
   as the assistant message. On history load, persisted content wins.

### Answer visuals without `answerVisualsV1`

`data-answer-reconcile` also arrives on a non-web answer when the server withheld an
answer-visual marker (`<!--rich:visual:...-->`) that this client did not advertise
support for. Its `content` carries the visual's Markdown fallback in place of the marker.
Handle it exactly as above; a client that ignores it keeps a draft without the visual and
gets the fallback on the next history load. See `plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md`
section 2 for the flag.

## Stop and recovery

1. Call `stop()`, then `POST /messages/stop` with `generationId`, `conversationId`,
   `expectedVersion` and a fresh `idempotencyKey`. The id and version come from the
   `data-generation` part with `phase: "start"`.
2. `202` means accepted, not confirmed: poll
   `GET /messages/generations/{generationId}?conversation_id={conversationId}`.
3. Aborting the fetch alone ends the turn as a disconnect, not a user Stop.

## Send back

1. Exact versions of `ai`, `@ai-sdk/react` and assistant-ui (backend parser tests pin
   `ai@6.0.0`).
2. Confirmation that error codes and tool renders work from the new parts.
3. If web drafts are enabled: the final text matches history after a reload, and a Stop or failure
   never leaves a spinner.
4. Any frame the SDK rejected, with the raw `data:` line.
